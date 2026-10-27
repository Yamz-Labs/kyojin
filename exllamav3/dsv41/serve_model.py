"""
DeepSeek-V4.1-Flash SERVING model: the port graph (`DSV41Model`: verified bit for bit against the official reference)
rebuilt for serving on EXL3 weights.

What it adds over the calibration / reference graph
  * chunked prefill: any chunk length at any start position (compressor groups straddling chunk borders, the sliding
    window as [ring | chunk keys], indexer scored in query blocks so memory does not grow with context x chunk, sparse
    attention in query blocks), decode (1 token), and the speculative verify block (`forward_block`, per-position logits,
    rollback through the cache journal) all go through ONE attention implementation;
  * paged cache (`serve_cache.DSV41Cache`): pools only for the kv source layers, the shared layers read them;
  * fp32 hyper-connection streams with fp16 (or fp32) activations; the delayed-pre hyper-connection chain on the
    ds41kern kernels (`serve_ops.HipHC`) when the real dims run on a GPU, torch otherwise;
  * routed experts through a backend (`serve_moe`): per-expert loop (any storage) or the mixed-K grouped kernels;
  * linears are `Lin` callables (EXL3 `LinearEXL3`, fp16 `LinearFP16`, dense) supplied by a WeightSource
    (`pack.PackSource` for converted packs); the head is a Lin too (no dense fp32 head).

Numerics contract: with fp32 activations and dense weights this graph equals `DSV41Model` up to GEMM-shape rounding
(the fp8 / fp4 fake-quant steps can amplify a last-bit difference, same caveat as the block verify of dswire); the tests
bound it. Activations into the linears are fp16 (what EXL3 serves), the in-attention fake quants of the reference stay.
"""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from .config import DSV41Config
from . import ops, quant
from .model import DSV41Model, SharedRt, _LW
from .engram import EngramLayout, engram_inject, MemoryEngramStore, PrefetchRowStore
from .experts import SourceExpertProvider
from .serve_cache import DSV41Cache
from .vision import span_mask, splice
from . import dec2
from .serve_ops import make_hc, Router
from .linear import DenseLin
from .serve_ops import HipDense
from .serve_moe import LoopMoE, GroupedMoE


from ..modules.dsv41_fn import ds41_pf as _pf  # noqa: E402

PF_ATTN_MIN_ROWS = 64        # shorter blocks (decode, speculative verify) keep the reference attention: spec == greedy

class DSV41Serving(DSV41Model):

    def __init__(self, cfg: DSV41Config, source, provider=None, engram_store=None, token_map=None, multipliers=None,
                 primes=None, dtype=torch.float16, device="cpu", moe_backends: dict = None, hc=None,
                 prefill_chunk: int = 2048, attn_block: int = 128, index_block_bytes: int = 256 << 20,
                 engram_prefetch: bool = True, ablit_dir: str = None):
        from .model import DSV41Graph
        DSV41Graph.__init__(self, cfg.validate(), dtype, device)
        self.src = source
        # row reads overlap with compute: the rows of a pass are known at its start (PrefetchRowStore)
        if engram_store is not None and engram_prefetch and not isinstance(engram_store, (MemoryEngramStore, PrefetchRowStore)):
            engram_store = PrefetchRowStore(engram_store)
        self.engram_store = engram_store
        self.token_map, self.multipliers = token_map, multipliers
        self.compressed_vocab = cfg.engram_compressed_vocab_size
        self.engram_layout = None
        n = cfg.num_hidden_layers
        if cfg.engram_layer_ids:
            assert token_map is not None, "engram needs the compressed token map"
            if not self.compressed_vocab:
                self.compressed_vocab = int(max(token_map)) + 1
            self.engram_layout = EngramLayout.build(cfg.engram_layer_ids, cfg.engram_num_embeddings,
                                                    cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                                                    cfg.engram_n_heads, cfg.engram_head_dim, primes)
        self.prefill_chunk, self.attn_block, self.index_block_bytes = prefill_chunk, attn_block, index_block_bytes
        d, dt = self.device, dtype
        self.embed_w = source.tensor("embed", dt, d)
        self.norm_w = source.tensor("norm", torch.float32, d)
        self.head = source.linear("head", dt, d)
        self.head_w = None
        self._idx_hip = None
        # row block sizes of the prompt-chunk passes (a long chunk keeps its big per-row buffers off the GPU: 0 = whole chunk)
        ev = lambda k, d: int(os.environ.get(k, d)) or (1 << 30)
        self.pf_attn_rows = ev("DSV41_PF_ATTN_ROWS", 8192)      # q [rows, 64 * 512] + attention output
        self.pf_eng_rows = ev("DSV41_PF_ENGRAM_ROWS", 4096)     # engram gather / projection (~450 KiB per row)
        self.pf_moe_rows = ev("DSV41_PF_MOE_ROWS", 8192)        # MoE combine buffers, shared expert
        self.idx_score_bytes = ev("DSV41_PF_IDX_SCORE_MB", 1024) << 20 if os.environ.get("DSV41_PF_IDX_SCORE_MB", "1024") != "0" else 1 << 62
        self.packed_idx = os.environ.get("DSV41_IDX_PACKED", "0") == "1"
        self.lw = {}
        for l in range(n + cfg.n_mtp_layers):
            self.lw[l] = self._load_layer(l)
        self._rope = {}
        self._rope32 = {}
        self._pos32 = self._slot32 = None
        self.fast_attn = self._fast_attn_available(cfg, dtype, device)
        self.hc = hc or make_hc(cfg, dt, d)
        self.router = Router(cfg, d)
        counts = {l: cfg.moe_counts(l)[0] for l in range(n + cfg.n_mtp_layers)}
        self.provider = provider or SourceExpertProvider(source, counts, cache=True, dtype=dt, device=d)
        self.loop_moe = LoopMoE(self.provider, cfg)
        self.moe_backends = dict(moe_backends or {})
        self._warm_decode_tables()
        self.fuse = True                      # decode glue fusion (dsv41_fuse); env DSV41_FUSE=0 turns the kernels off
        self.decode_graph = os.environ.get("DSV41_DECODE_GRAPH", "0") == "1"
        self._graphs, self._graph_support, self._graph_failed = {}, {}, False
        self.mtp = None
        if cfg.n_mtp_layers:
            from .dspark import DSparkHeads
            self.mtp = DSparkHeads(self)
        self.set_ablit(ablit_dir)             # EXL3_ABLIT_RUNTIME / <ablit_dir>/uncensor_spec.json; None = off

    @staticmethod
    def _fast_attn_available(cfg, dtype, device) -> bool:
        """Fused HIP decode kernels (kv_prep, rmsnorm, attn_dec) usable: real dims on a GPU, fp16 activations."""
        from .serve_ops import hip_available
        return (dtype == torch.float16 and hip_available(device, cfg.hidden_size, cfg.hc_mult) and cfg.head_dim == 512
                and cfg.qk_rope_head_dim == 64 and cfg.sliding_window == 128 and cfg.fp8_block_size == 32
                and cfg.num_attention_heads == 64 and cfg.index_topk <= 1152 - 128 and cfg.index_n_heads == 32
                and cfg.index_head_dim == 128)

    def _dec_fast(self, B, L, start_pos) -> bool:
        return self.fast_attn and L == 1 and B == 1 and start_pos > 0 and self._pos32 is not None

    def _geom32(self, start_pos: int, ratio: int):
        """Device int32 [1] rope position and pool row of the latent completed by decode position start_pos."""
        g0 = start_pos // ratio
        t = torch.tensor([g0 * ratio, g0], dtype=torch.int32, device=self.device)
        return t[:1], t[1:]

    def _rope_f32(self, ratio: int, max_seq: int):
        key = (bool(ratio), max_seq)
        if key not in self._rope32:
            from ..modules.dsv41_fn import dsv41_hip as K
            self._rope32[key] = K.rope_table_f32(self._rope_for(ratio, max_seq))
        return self._rope32[key]

    # ------------------------------------------------------------------ HIP-graph decode (serve_graph.py)
    def _graph_ok(self, st) -> bool:
        ok = self._graph_support.get(id(st))
        if ok is None:
            from .serve_graph import graph_supported
            ok = self._graph_support[id(st)] = graph_supported(self, st)
        return ok and not self._graph_failed

    def _graph_for(self, st):
        from .serve_graph import DecodeGraph
        key = DecodeGraph.make_key(st)
        g = self._graphs.get(key)
        if g is None:
            if len(self._graphs) >= 4:
                self._graphs.pop(next(iter(self._graphs)))
            g = self._graphs[key] = DecodeGraph(self, st)
        return g

    def _warm_decode_tables(self):
        """Build the decode-step index tensors of every ring phase once, so a decode step never copies host data."""
        win, dev = self.cfg.sliding_window, self.device
        if dev.type == "cpu":
            return
        for sp in list(range(1, win)) + list(range(win, 2 * win)):
            ops.window_idxs(win, 1, 1, sp, dev)
        for ph in range(win):
            ops.table_memo(("slot", win, ph, str(dev)), lambda ph=ph: torch.tensor([ph], device=dev))

    # ------------------------------------------------------------------ weights
    def _load_layer(self, l):
        w = super()._load_layer(l)
        f32 = torch.float32
        for nm in ("attn_norm", "ffn_norm", "q_norm", "kv_norm"):
            setattr(w, nm, getattr(w, nm).to(f32))
        if w.comp is not None:
            w.comp.norm = w.comp.norm.to(f32)
        if w.idx is not None and getattr(w.idx, "owns_k", False):
            w.idx.k_norm = w.idx.k_norm.to(f32)
        if self._fast_attn_available(self.cfg, self.dtype, self.device):
            for lin in (getattr(w.comp, "wkv", None), getattr(w.comp, "wgate", None), getattr(w.idx, "weights_proj", None),
                        getattr(w.idx, "wk", None)):
                if isinstance(lin, DenseLin):
                    HipDense.attach(lin)
        fn = self.src.wo_a_fn(self.src.layer_prefix(l), self.cfg.o_groups) if hasattr(self.src, "wo_a_fn") else None
        if fn is not None:
            w.wo_a, w.wo_a_fn = None, fn
            w.wo_a_lins = getattr(self.src, "wo_a", {}).get(self.src.layer_prefix(l))
        return w

    def new_state(self, batch: int = 1, max_seq: int = 4096) -> DSV41Cache:
        return DSV41Cache(self, batch, max_seq)

    def head_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Final-normed activations [..., D] -> fp32 logits."""
        return self.head(x.to(self.dtype)).float()

    # ------------------------------------------------------------------ attention helpers
    @staticmethod
    def _latent_geom(start_pos: int, L: int, ratio: int, fc_t):
        """(rope rows, first pool row) of the compressed latents completed by positions start_pos .. start_pos+L-1.
        Group g covers positions g*ratio .. g*ratio+ratio-1, completes at its last position, rope row = its first."""
        g0, g1 = start_pos // ratio, (start_pos + L) // ratio
        return fc_t[g0 * ratio: g1 * ratio: ratio], g0

    def _compressor(self, l, x, start_pos, st):
        """Pre-RoPE compressed latent of the groups completed by this chunk (None if none): any start / length."""
        cfg, c = self.cfg, self.lw[l].comp
        B, L, _ = x.shape
        r = cfg.ratio(l)
        fast = self._dec_fast(B, L, start_pos) and st.journal is None
        if r == 1:
            if fast:
                from ..modules.dsv41_fn import dsv41_hip as K
                kv = c.wkv(x)
                K.rmsnorm_rows(kv.view(1, -1), c.norm, kv.view(1, -1), cfg.rms_norm_eps)
                return kv
            return ops.rmsnorm(c.wkv(x), c.norm, cfg.rms_norm_eps)
        dtype = x.dtype
        xf = x.float()
        kv, score = c.wkv(xf), c.wgate(xf)
        kst, sst = st.comp_kv_state[l], st.comp_score_state[l]
        if fast and r <= 8:
            from ..modules.dsv41_fn import dsv41_hip as K
            ng = (start_pos + 1) // r - start_pos // r
            out = torch.empty(1, 1, kv.size(-1), dtype=torch.float16, device=x.device) if ng else kv.new_empty(0).half()
            K.compress(kv.view(1, -1), score.view(1, -1), kst[0], sst[0], c.norm, out.view(-1, kv.size(-1)) if ng else out,
                       start_pos, r, cfg.rms_norm_eps)
            return out if ng else None
        if st.journal is not None and start_pos > 0:
            # verify block: per-position state updates, logged for the rollback (same as L decode steps)
            outs = []
            for i in range(L):
                q = start_pos + i
                slot = q % r
                st._jr_comp(l, q, slot, B)
                kst[:B, slot] = kv[:, i]
                sst[:B, slot] = score[:, i]
                if (q + 1) % r == 0:
                    outs.append((kst[:B] * sst[:B].softmax(dim=1)).sum(dim=1, keepdim=True))
            if not outs:
                return None
            return ops.rmsnorm(torch.cat(outs, dim=1).to(dtype), c.norm, cfg.rms_norm_eps)
        outs, pos, off = [], 0, start_pos % r
        if off:                                    # finish the open group of an earlier chunk
            a = min(L, r - off)
            kst[:B, off:off + a], sst[:B, off:off + a] = kv[:, :a], score[:, :a]
            pos = a
            if off + a == r:
                outs.append((kst[:B] * sst[:B].softmax(dim=1)).sum(dim=1, keepdim=True))
        m = (L - pos) // r                         # whole groups inside the chunk
        if m:
            sk = kv[:, pos:pos + m * r].unflatten(1, (m, r))
            ss = score[:, pos:pos + m * r].unflatten(1, (m, r))
            outs.append((sk * ss.softmax(dim=2)).sum(dim=2))
            kst[:B], sst[:B] = sk[:, -1], ss[:, -1]   # the state of the last closed group, as token-by-token leaves it
            pos += m * r
        if L - pos:                                # open group left for the next chunk
            kst[:B, :L - pos], sst[:B, :L - pos] = kv[:, pos:], score[:, pos:]
        if not outs:
            return None
        return ops.rmsnorm(torch.cat(outs, dim=1).to(dtype), c.norm, cfg.rms_norm_eps)

    def _indexer(self, l, x, qr, latent, start_pos, offset, st, rt, fc):
        """Indexer top-k (int32 indices into [window | compressed], offset applied). Same selection as
        DSV41Graph._indexer; the score GEMM runs in query blocks and in fp32."""
        cfg, ix = self.cfg, self.lw[l].idx
        B, L, _ = x.shape
        ratio, rd, end_pos = cfg.ratio(l), cfg.qk_rope_head_dim, start_pos + L
        nh, hd = cfg.index_n_heads, cfg.index_head_dim
        if B == 1 and self._idx_hip is None:
            from ..modules.dsv41_fn import dsv41_idx
            self._idx_hip = dsv41_idx if (dsv41_idx.available(x.device) and x.dtype == torch.half and nh == 32 and hd == 128) else False
        fast = self._dec_fast(B, L, start_pos) and nh == 32 and hd == 128
        if fast:
            from ..modules.dsv41_fn import dsv41_hip as K
        if ix.owns_k and fast and not self.packed_idx:
            if latent is not None:
                ckp, dst = self._geom32(start_pos, ratio)
                K.idx_k_store(ix.wk(latent).view(1, hd), ix.k_norm, self._rope_f32(ratio, st.max_seq), ckp, dst,
                              st.idx_k[l][0], cfg.rms_norm_eps)
            rt.index_k = st.idx_k[l]
        elif ix.owns_k:
            if latent is not None:
                ck, row0 = self._latent_geom(start_pos, L, ratio, fc)
                k = ops.rmsnorm(ix.wk(latent), ix.k_norm, cfg.rms_norm_eps)
                ops.rope_apply_(k[..., -rd:], ck)
                quant.fp4_act_quant_(k, 32)
                st.idx_k[l][:B, row0: row0 + k.size(1)] = k
                if self._idx_hip and B == 1 and self.packed_idx:
                    kq, ks = self._idx_hip.pack_keys(k[0])
                    st.idx_kq[l][0, row0: row0 + kq.size(0)] = kq
                    st.idx_ks[l][0, row0: row0 + kq.size(0)] = ks
            rt.index_k = st.idx_k[l]
            if self.packed_idx:
                rt.index_kq = (st.idx_kq[l], st.idx_ks[l])
        q = ix.wq_b(qr).unflatten(-1, (nh, hd))
        if fast:
            K.idx_q_prep(q.view(1, nh * hd), self._rope_f32(ratio, st.max_seq), self._pos32, nh)
        else:
            ops.rope_apply_(q[..., -rd:], fc[start_pos:end_pos])
            quant.fp4_act_quant_(q, 32)
        weights = ix.weights_proj(x).float() * (hd ** -0.5 * nh ** -0.5)
        if B == 1 and self._idx_hip and rt.index_k.dtype == torch.half:
            return self._indexer_hip(l, q, weights, start_pos, L, ratio, offset, rt)
        index_k = rt.index_k[:B, : end_pos // ratio].float()
        nt = index_k.size(1)
        score = torch.empty(B, L, nt, dtype=torch.float32, device=x.device)
        qb = max(1, self.index_block_bytes // max(1, nh * nt * 4 * B))
        for a in range(0, L, qb):
            s = torch.einsum("bshd,btd->bsht", q[:, a:a + qb].float(), index_k)
            score[:, a:a + qb] = (s.relu_() * weights[:, a:a + qb].unsqueeze(-1)).sum(dim=2)
        if start_pos == 0 or L > 1:
            clen = ((start_pos + torch.arange(1, L + 1, device=x.device)) // ratio).unsqueeze(-1)
            score.masked_fill_(torch.arange(nt, device=x.device) >= clen, -torch.inf)
        else:
            clen = end_pos // ratio
        if cfg.candidate_source_layer_id >= 0:
            if l == cfg.candidate_source_layer_id:
                rt.candidates = ops.select_candidate_blocks(score, clen, cfg.candidate_topk_blocks,
                                                            cfg.candidate_block_size)
            elif l > cfg.candidate_source_layer_id:
                score = score.masked_fill(~rt.candidates, -torch.inf)
        topk = min(cfg.index_topk, end_pos // ratio)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < clen, idxs + offset, -1).int()

    def _indexer_hip(self, l, q, weights, start_pos, L, ratio, offset, rt, nt=None):
        """Indexer top-k on the WMMA scoring kernels (batch 1). Same selection as the torch path: layers up to and including the
        candidate source scan the whole pool once (fp16 keys, no fp32 copy of the pool); later index sources score only the 16K
        keys of the candidate blocks, so their cost does not grow with the context."""
        cfg, K = self.cfg, self._idx_hip
        nh, hd = cfg.index_n_heads, cfg.index_head_dim
        end_pos = start_pos + L
        nt = end_pos // ratio if nt is None else nt
        kp = rt.index_k[0]
        pk = (rt.index_kq[0][0], rt.index_kq[1][0]) if self.packed_idx else None
        q2 = q.reshape(L, nh, hd).contiguous()
        w2 = weights.reshape(L, nh).contiguous()
        cs = cfg.candidate_source_layer_id
        # the dense score is [rows, nt] fp32 (cand: [rows, 16K]): rows are independent, so a long chunk is scored in row blocks
        per_row = 4 * (-(-nt // 8) * 8 if not (cs >= 0 and l > cs) else cfg.candidate_topk_blocks * 8)
        rows = max(256, self.idx_score_bytes // per_row)
        if L > rows:
            cand_l, outs, cands, whole = cs >= 0 and l > cs, [], [], rt.cand_blocks
            for a, b in self._row_blocks(L, rows):
                if cand_l:
                    rt.cand_blocks = whole[a:b]
                outs.append(self._indexer_hip(l, q2[a:b], w2[a:b], start_pos + a, b - a, ratio, offset, rt, nt=nt))
                if l == cs:
                    cands.append(rt.cand_blocks)
            rt.cand_blocks = torch.cat(cands, dim=0) if l == cs else whole
            return torch.cat(outs, dim=1)
        topk = min(cfg.index_topk, nt)
        dev = q.device
        if cs >= 0 and l > cs:
            blk = rt.cand_blocks
            score = K.score_cand(q2, w2, kp, blk, start_pos, ratio, pk)
            v, slot = score.topk(topk, dim=-1, sorted=False)
            pos = torch.gather(blk, 1, slot // 8).long() * 8 + slot % 8
            pos = torch.where(v > -torch.inf, pos, torch.iinfo(torch.int64).max)
            idxs = pos.sort(dim=-1).values
            return torch.where(idxs < torch.iinfo(torch.int64).max, idxs + offset, -1).int().unsqueeze(0)
        score = K.score_dense(q2, w2, kp, nt, start_pos, ratio, packed=pk)
        bs, nb = cfg.candidate_block_size, score.size(1) // cfg.candidate_block_size
        sc = score.view(L, nb, bs).amax(-1) if (cs >= 0 and l == cs) or (L > 1 and nb >= 8192) else None
        if cs >= 0 and l == cs:
            clen = (start_pos + torch.arange(1, L + 1, device=dev)) // ratio
            last = (clen - 1) // bs
            li = last.clamp_min(0).unsqueeze(1)
            pinned = sc.clone()
            pinned.scatter_(1, li, torch.where((last >= 0).unsqueeze(1), torch.inf, sc.gather(1, li)))
            top = pinned.topk(min(cfg.candidate_topk_blocks, nb), dim=-1)
            rt.cand_blocks = torch.where(top.values > -torch.inf, top.indices, -1).int().contiguous()
        if sc is not None and L > 1 and nb >= 8192 and topk <= nb:
            # exact two-stage selection: every top-k key sits in one of the k blocks with the best block maximum
            blocks = sc.topk(topk, dim=-1, sorted=False).indices
            cand = score.view(L, nb, bs).gather(1, blocks.unsqueeze(-1).expand(-1, -1, bs)).flatten(1)
            slot = cand.topk(topk, dim=-1, sorted=False).indices
            idxs = (torch.gather(blocks, 1, slot // bs) * bs + slot % bs).sort(dim=-1).values
        else:
            idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        clen = ((start_pos + torch.arange(1, L + 1, device=dev)) // ratio).unsqueeze(-1)
        return torch.where(idxs < clen, idxs + offset, -1).int().unsqueeze(0)

    @staticmethod
    def _window_idxs(rt, fn, win, B, L, start_pos, device):
        """Window slot table of this chunk, built once per pass and shared by every layer (read-only: torch.cat copies it)."""
        key = (fn.__name__, win, B, L, start_pos)
        cache = rt.__dict__.setdefault("widx", {})
        if key not in cache:
            cache[key] = fn(win, B, L, start_pos, device)
        return cache[key]

    # ---- prompt-chunk fast paths (HIP kernels; short blocks keep the torch ops so decode / verify numerics are unchanged)
    @staticmethod
    def _pffast(x):
        return (x.is_cuda and x.dtype == torch.half and x.size(0) == 1 and x.size(1) >= PF_ATTN_MIN_ROWS
                and os.environ.get("DSV41_KERNELS", "1") != "0")

    def _pfrms(self, fast, x, w):
        if not fast or x.size(-1) % 32 or not x.is_contiguous() or w.dtype != torch.float:
            return ops.rmsnorm(x, w, self.cfg.rms_norm_eps)
        from ..modules.dsv41_fn import dsv41_hip as K
        y = torch.empty_like(x)
        K.rmsnorm_rows(x.view(-1, x.size(-1)), w, y.view(-1, x.size(-1)), self.cfg.rms_norm_eps)
        return y

    def _pfrope(self, fast, x, fc_rows, pos_fn, heads, inverse=False):
        """Rope of the last rope_head_dim dims of x [1, rows, (heads,) 512]; fc_rows: the complex table rows of its positions;
        pos_fn() -> (fc32 table, int32 positions) for the kernel path."""
        if fast and x.is_contiguous() and x.size(-1) == 512:
            tab, pos = pos_fn()
            return _pf.rope_rows(x, tab, pos, heads, inverse)
        return ops.rope_apply_(x[..., -self.cfg.qk_rope_head_dim:], fc_rows, inverse)

    def _pftab(self, fc_t):
        t = self.__dict__.setdefault("_rope32", {})
        k = id(fc_t)
        if k not in t:
            from ..modules.dsv41_fn import dsv41_hip as K
            t[k] = (fc_t, K.rope_table_f32(fc_t))      # keep fc_t alive: the id stays unique
        return t[k][1]

    def _attention(self, l, x, start_pos, st, rt, taps=None):
        cfg, w = self.cfg, self.lw[l]
        B, L, _ = x.shape
        pf = self._pffast(x)
        H, hd, rd, win = cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim, cfg.sliding_window
        ratio = cfg.ratio(l)
        fc_t = self._rope_for(ratio, st.max_seq)
        fc = fc_t[start_pos:start_pos + L]
        fast = self._dec_fast(B, L, start_pos)
        ring = st.win[l]

        def qpos():
            return self._pftab(fc_t), torch.arange(start_pos, start_pos + L, dtype=torch.int32, device=x.device)
        if fast:
            from ..modules.dsv41_fn import dsv41_hip as K
            qk = dec2.pair(w, "QKV", (w.wq_a, w.wkv), x.view(1, -1)) if x.dtype == torch.half else None
            if qk is not None:
                qr, kvr = qk[0].view(1, 1, -1), qk[1].view(1, 1, -1)
            else:
                qr = w.wq_a(x)
                kvr = None
            K.rmsnorm_rows(qr.view(1, -1), w.q_norm, qr.view(1, -1), cfg.rms_norm_eps)
            q = w.wq_b(qr)
            if kvr is None:
                kvr = w.wkv(x)
            if st.journal is not None:
                st._jr_ring(l, [start_pos], B)
            K.kv_prep(kvr.view(1, hd), w.kv_norm, self._rope_f32(ratio, st.max_seq), self._pos32, self._slot32, None,
                      ring[0], cfg.rms_norm_eps)
            window_kv = None
        else:
            qr = self._pfrms(pf, w.wq_a(x), w.q_norm)
            if (pf and B == 1 and L > self.pf_attn_rows and st.win[l].dtype == torch.half and H == 64 and hd == 512
                    and w.sink.numel() == 64):
                q = None            # projected, roped and consumed block by block below: [L, 64 * 512] never exists
            else:
                q = w.wq_b(qr).unflatten(-1, (H, hd))
                self._pfrope(pf, q, fc, qpos, H)
            kv = self._pfrms(pf, w.wkv(x), w.kv_norm)
            self._pfrope(pf, kv, fc, qpos, 1)
            quant.act_quant_fp8_(kv, cfg.fp8_block_size)
            kv = kv.to(st.win[l].dtype)
        if not fast:
            # sliding window: keys = [ring as it was before this chunk | this chunk's keys]; query i reads the `win`
            # positions ending at its own, oldest first (-1 while fewer exist)
            if start_pos == 0:
                window_kv = kv
                idxs = self._window_idxs(rt, ops.window_idxs, win, B, L, 0, x.device)
            else:
                window_kv = torch.cat([ring[:B], kv], dim=1)
                idxs = self._window_idxs(rt, ops.window_idxs_block, win, B, L, start_pos, x.device)
            keep = min(L, win)
            positions = list(range(start_pos + L - keep, start_pos + L))
            if st.journal is not None and start_pos > 0:
                if L > win:
                    raise ValueError(f"verify block of {L} tokens exceeds the sliding window ({win})")
                st._jr_ring(l, positions, B)
            mk = lambda: torch.tensor([p % win for p in positions], device=ring.device)
            if len(positions) == 1:
                slots = ops.table_memo(("slot", win, positions[0] % win, str(ring.device)), mk)
            else:
                slots = ops.step_memo(("slots", win, tuple(positions), str(ring.device)), mk)
            ring[:B, slots] = kv[:, L - keep:]
        if ratio:
            offset = win + 1 if fast else window_kv.size(1)
            clen_total = (start_pos + L) // ratio
            latent = None
            if cfg.is_kv_source(l):
                latent = self._compressor(l, x, start_pos, st)
                rt.compress_kv = st.comp_kv[l]
                if taps is not None and latent is not None:
                    taps[f"L{l}.comp"] = latent.clone()
            if not cfg.is_index_source(l):
                cidx = rt.topk
            else:
                if clen_total == 0:
                    cidx = torch.empty(B, L, 0, dtype=torch.int32, device=x.device)
                else:
                    cidx = self._indexer(l, x, qr, latent, start_pos, offset, st, rt, fc_t)
                rt.topk = cidx
                if taps is not None:
                    taps[f"L{l}.topk"] = cidx.clone()
            if latent is not None and fast:
                ckp, dst = self._geom32(start_pos, ratio)
                K.pool_store(latent.view(1, hd), self._rope_f32(ratio, st.max_seq), ckp, dst, st.comp_kv[cfg.kv_source_of(l)][0])
            elif latent is not None:
                ck, row0 = self._latent_geom(start_pos, L, ratio, fc_t)
                g0 = start_pos // ratio
                self._pfrope(pf, latent, ck, lambda: (self._pftab(fc_t), torch.arange(
                    g0 * ratio, g0 * ratio + latent.size(1) * ratio, ratio, dtype=torch.int32, device=x.device)), 1)
                quant.fp4_act_quant_(latent, 16, e4m3_scale=True)
                st.comp_kv[cfg.kv_source_of(l)][:B, row0: row0 + latent.size(1)] = latent
            if not fast:
                ckv = rt.compress_kv[:B, :clen_total]
                kv_all, kv_pool = window_kv, ckv        # the pool is gathered in place: no O(context) concatenation
                idxs = torch.cat([idxs, cidx], dim=-1)
        else:
            kv_all, kv_pool = window_kv, None
        G = cfg.o_groups
        sc = hd ** -0.5
        if fast:
            nc = cidx.size(-1) if ratio and clen_total else 0
            o = torch.empty(B, L, H * hd, dtype=torch.float16, device=x.device)
            K.attn_dec(q.view(-1), ring[0], rt.compress_kv if nc else None, cidx.view(-1) if nc else None, nc, win + 1,
                       w.sink, self._rope_f32(ratio, st.max_seq), self._pos32, o.view(-1), sc)
            o = o.view(B, L, G, -1)
        elif q is None:
            # long prompt chunk: q, the attention output and the grouped output projection run on row blocks (rows are
            # independent: the same arithmetic as the whole chunk at once, without its two [L, 32768] fp16 buffers)
            kvc = kv_all if kv_pool is None or kv_pool.size(1) == 0 else torch.cat([kv_all, kv_pool], dim=1)
            ys = []
            for a, b in self._row_blocks(L, self.pf_attn_rows):
                pos = lambda: (self._pftab(fc_t), torch.arange(start_pos + a, start_pos + b, dtype=torch.int32, device=x.device))
                qb = w.wq_b(qr[:, a:b]).unflatten(-1, (H, hd))
                self._pfrope(pf, qb, fc[a:b], pos, H)
                ob = _pf.sparse_attention(qb, kvc, w.sink, idxs[:, a:b], sc)
                del qb
                self._pfrope(pf, ob, fc[a:b], pos, H, True)
                ob = ob.view(B, b - a, G, -1)
                quant.act_quant_fp8_(ob, cfg.fp8_block_size)
                ys.append(w.wo_b(self._wo_a(w, ob).flatten(2)))
                del ob
            return ys[0] if len(ys) == 1 else torch.cat(ys, dim=1)
        else:
            if L >= PF_ATTN_MIN_ROWS and _pf.usable(q, kv_all, idxs, w.sink):
                kvc = kv_all if kv_pool is None or kv_pool.size(1) == 0 else torch.cat([kv_all, kv_pool], dim=1)
                o = _pf.sparse_attention(q, kvc, w.sink, idxs, sc)      # WMMA kernel: prompt chunks only (not verify blocks)
            else:
                o = torch.empty_like(q)
                for a in range(0, L, self.attn_block):
                    b = a + self.attn_block
                    o[:, a:b] = ops.sparse_attention(q[:, a:b], kv_all, w.sink, idxs[:, a:b], sc, kv_pool)
            self._pfrope(pf, o, fc, qpos, H, True)
            o = o.view(B, L, G, -1)
            quant.act_quant_fp8_(o, cfg.fp8_block_size)
        o = self._wo_a(w, o)
        return w.wo_b(o.flatten(2))

    def _engram_rows(self, hh, hashes, e, keep):
        """engram_inject on row blocks, written back in place (rows are independent): its gather / projection temporaries
        are ~450 KiB per row, so a 64K chunk would need ~30 GiB at once."""
        cfg, L = self.cfg, hh.size(1)
        if L <= self.pf_eng_rows or hh.dtype != torch.float32 or not hh.is_contiguous():
            return engram_inject(hh, hashes, self.engram_store, e.pos, e.wkv, e.qk, cfg.hc_mult, cfg.hidden_size,
                                 cfg.rms_norm_eps, token_mask=keep)
        for a, b in self._row_blocks(L, self.pf_eng_rows):
            hh[:, a:b] = engram_inject(hh[:, a:b], hashes[:, a:b], self.engram_store, e.pos, e.wkv, e.qk, cfg.hc_mult,
                                       cfg.hidden_size, cfg.rms_norm_eps,
                                       token_mask=None if keep is None else keep[:, a:b])
        return hh

    def _mt(self, l, what):
        """DSV41_MEMTRACE=1: peak torch GPU memory (GiB, since the previous mark) at each stage of a long prompt chunk."""
        if os.environ.get("DSV41_MEMTRACE") != "1" or not torch.cuda.is_available():
            return
        print(f"MT L{l} {what} peak {torch.cuda.max_memory_allocated() / 2**30:.2f} cur {torch.cuda.memory_allocated() / 2**30:.2f}", flush=True)
        torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _row_blocks(L, rows):
        """Near-equal row blocks of at most `rows` (no short tail: every block keeps the kernels' row minimums)."""
        n = -(-L // max(1, rows))
        step = -(-L // n)
        return [(a, min(L, a + step)) for a in range(0, L, step)]

    def _wo_a(self, w, o):
        if o.size(0) == 1 and o.size(1) == 1:
            y = dec2.wo_a(getattr(w, "wo_a_fn", None), o)
            if y is not None:
                return y
        return super()._wo_a(w, o)

    # ------------------------------------------------------------------ MoE
    def _moe(self, l, x, taps=None):
        cfg, w = self.cfg, self.lw[l]
        shape = x.size()
        x = x.view(-1, cfg.hidden_size)
        be = self.moe_backends.get(l, self.loop_moe)
        if (self.fuse and isinstance(be, GroupedMoE) and self._img_rows is None and be.fused_ok(x)
                and self.router.fused_ok(x, w.gate_w, w.gate_b, w.topk)):
            sd = dec2.side(x.device) if (x.size(0) == 1 and taps is None) else None
            sh = None
            if sd is not None:
                # shared expert beside the router + routed experts: it only needs x; the side stream is in order after the ffn hc mix
                # (queued right after the collapse that made x)
                with torch.cuda.stream(sd.stream):
                    x.record_stream(sd.stream)
                    sh = w.shared.ffn(x, None, cfg.swiglu_limit)
                    sd.sh_done.record(sd.stream)
            wts, wh, idx, gsel = self.router.fused(x, w.gate_w, w.gate_b, w.topk, be.store.lut)
            if taps is not None:
                taps[f"L{l}.route.idx"], taps[f"L{l}.route.w"] = idx.clone(), wts.clone()
            be.prepare(idx)
            outs = be.decode_parts(x, idx, wh, gsel)
            if sd is not None:
                cur = torch.cuda.current_stream()
                cur.wait_event(sd.sh_done)
                sh.record_stream(cur)
            else:
                sh = w.shared.ffn(x, None, cfg.swiglu_limit)
            if sh.dtype == torch.half and sh.is_contiguous():
                from ..modules.dsv41_fn import dsv41_fuse as FZ
                return FZ.moe_tail(outs, sh).view(shape)
            y = outs[0].clone()
            for o in outs[1:]:
                y.add_(o)
            return (y + sh.float()).type_as(x).view(shape)
        wts, idx = self.router(x, w.gate_w, self._route_bias(w), w.topk)
        if taps is not None:
            taps[f"L{l}.route.idx"], taps[f"L{l}.route.w"] = idx.clone(), wts.clone()
        if isinstance(be, GroupedMoE):
            be.prepare(idx)
        T = x.size(0)
        if isinstance(be, GroupedMoE) and T > self.pf_moe_rows:
            # one prepare() for the whole chunk (the experts are resident), then row blocks over the SAME experts: the fp32 output,
            # the fp32 shared-expert copy and their sum exist for one block, not for the whole chunk
            wf, out = wts.float(), torch.empty_like(x)
            for a, b in self._row_blocks(T, self.pf_moe_rows):
                yb = be.run(l, x[a:b], wf[a:b], idx[a:b], w.n_exp)
                yb += w.shared.ffn(x[a:b], None, cfg.swiglu_limit).float()
                out[a:b] = yb.type_as(x)
                del yb
            return out.view(shape)
        y = be.run(l, x, wts.float(), idx, w.n_exp)
        y = y + w.shared.ffn(x, None, cfg.swiglu_limit).float()
        return y.type_as(x).view(shape)

    # ------------------------------------------------------------------ forward
    def _prefetch_engram(self, hashes, positions):
        """Queue the row reads of the engram layers for these hashes ([B, L, n_layers, cols]); they run while the layers
        in front of the engram layer compute."""
        if isinstance(self.engram_store, PrefetchRowStore):
            for pos in positions:
                self.engram_store.prefetch(pos, torch.unique(hashes[:, :, pos, :].reshape(-1)))

    @torch.inference_mode()
    def _forward(self, input_ids, start_pos, st, taps, mode: str, ahead_ids=None, spans=None, hid_rows=None):
        """One pass over a chunk. mode: 'last' (logits of the last row), 'all' (every row), 'none' (state only).
        ahead_ids: the next chunk of a chunked prefill; its engram rows are queued now (the hasher is idempotent).
        spans: vision.ImageSpan list (absolute prompt positions) spliced over the embeddings."""
        try:
            return self._forward_core(input_ids, start_pos, st, taps, mode, ahead_ids, spans, hid_rows)
        finally:
            self._img_rows = None

    def _forward_core(self, input_ids, start_pos, st, taps, mode: str, ahead_ids=None, spans=None, hid_rows=None):
        cfg = self.cfg
        input_ids = input_ids.to(self.device)
        B, L = input_ids.shape
        if input_ids.device.type == "cpu":      # device-resident ids come from our own argmax; checking = a sync
            assert input_ids.numel() == 0 or (int(input_ids.min()) >= 0 and int(input_ids.max()) < cfg.vocab_size), \
                "token id outside the vocabulary"
        if L * B >= 64 and self.moe_backends:
            # SSD tier, prefill call: start the expert reads of the first layers now, before the engram hashing / lookups and the
            # embedding (the reads of the layer in front have nothing to hide behind otherwise)
            tb0 = self.moe_backends.get(min(self.moe_backends))
            if getattr(tb0, "tier", None) is not None:
                tb0.tier.store.pf_wrap = ahead_ids is not None and ahead_ids.numel() > 0
                tb0.tier.store.prefill_issue(tb0.tier, L * B, upfront = True)
        st.ensure(start_pos + L)
        img = span_mask(spans, start_pos, L, self.device)
        keep = None if img is None else (~img).unsqueeze(0)           # [1, L] rows that take part in engram
        hashes = st.hasher(input_ids, start_pos, keep) if st.hasher is not None else None
        if hashes is not None and self.engram_store is not None:
            eg_pos = [self.lw[l].engram.pos for l in range(cfg.num_hidden_layers) if self.lw[l].engram is not None]
            self._prefetch_engram(hashes, eg_pos)
            if ahead_ids is not None and ahead_ids.numel():
                st.ensure(start_pos + L + ahead_ids.size(1))
                nm = span_mask(spans, start_pos + L, ahead_ids.size(1), self.device)
                nxt = st.hasher(ahead_ids.to(self.device), start_pos + L, None if nm is None else (~nm).unsqueeze(0))
                self._prefetch_engram(nxt, eg_pos)
        if (mode == "last" and L == 1 and B == 1 and start_pos > 0 and taps is None and st.journal is None
                and self.decode_graph and self._graph_ok(st)):
            logits, hid = self._graph_for(st).step(input_ids, start_pos, hashes)
            st.length = max(getattr(st, "length", 0), start_pos + 1)
            return logits, hid
        self._mt(-1, 'hashes')
        if img is None and L > self.pf_eng_rows:
            # long chunk: the four streams are filled block by block (no fp32 [L, dim] copy next to the [L, 4, dim] result)
            h = torch.empty(B, L, cfg.hc_mult, cfg.hidden_size, dtype=torch.float32, device=self.device)
            for a, b in self._row_blocks(L, self.pf_eng_rows):
                h[:, a:b] = F.embedding(input_ids[:, a:b], self.embed_w).float().unsqueeze(2)
        else:
            h = F.embedding(input_ids, self.embed_w).float()
            if img is not None:
                splice(h, start_pos, spans)
                self._img_rows = img.repeat(B)
            h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        S = self.hc.new(h)
        pre = h.new_zeros(B, L, cfg.hc_mult)
        pre[:, :, 0] = 1.0
        rt = SharedRt()
        main = []
        self._mt(-1, 'embed+S')
        if self.fast_attn and L == 1 and B == 1:
            self._pos32 = torch.tensor([start_pos], dtype=torch.int32, device=self.device)
            self._slot32 = self._pos32 % cfg.sliding_window
        for l in range(cfg.num_hidden_layers):
            w = self.lw[l]
            tb = self.moe_backends.get(l)
            if L * B >= 64 and getattr(tb, "tier", None) is not None:
                tb.tier.store.pf_wrap = ahead_ids is not None and ahead_ids.numel() > 0     # another chunk follows: layer 0 is next after the last layer
                tb.tier.store.prefill_issue(tb.tier, L * B)     # SSD tier: start the expert reads before this layer's attention
            if w.engram is not None:
                e = w.engram
                S.inject(lambda hh: self._engram_rows(hh, hashes[:, :, e.pos, :], e, keep))
            if l in cfg.dspark_target_layer_ids and self.mtp is not None:
                main.append(S.apply().view(B, L, cfg.hc_mult, -1)[:, -(hid_rows or L):].mean(dim=2))     # hid_rows: only the tail is read
            x = S.cn(pre, w.attn_norm)
            a_pre, a_post, a_comb = S.mix(w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
            self._mt(l, 'pre-attn')
            y = self._attention(l, x, start_pos, st, rt, taps)
            self._mt(l, 'attn')
            if self.ablit is not None:
                y = self._abl(l, 0, y)
            if taps is not None:
                taps[f"L{l}.attn"] = y.clone()
            S.push(y, a_post, a_comb)
            x = S.cn(a_pre, w.ffn_norm)
            f_pre, f_post, f_comb = S.mix(w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
            self._mt(l, 'pre-moe')
            y = self._moe(l, x, taps)
            self._mt(l, 'moe')
            if self.ablit is not None:
                y = self._abl(l, 1, y)
            if taps is not None:
                taps[f"L{l}.ffn"] = y.clone()
            S.push(y, f_post, f_comb)
            pre = f_pre
        st.length = max(getattr(st, "length", 0), start_pos + L)
        hid = torch.cat(main, dim=-1) if main else None
        if mode == "none":
            S.apply()
            return None, hid
        x = S.cn(pre, self.norm_w)
        logits = self.head_logits(x if mode == "all" else x[:, -1])
        return logits, hid

    def forward(self, input_ids: torch.Tensor, start_pos: int, st, taps: dict = None, chunk: int = None, spans=None, on_chunk=None):
        """Any chunk at any start position. Returns (logits of the last row [B, vocab] fp32, main_hidden
        [B, L, dim * n_targets] or None). A prompt longer than `chunk` (default `prefill_chunk`) runs chunked and
        main_hidden then only holds the last `sliding_window` rows (all the drafter's rings can use)."""
        return self.prefill(input_ids, st, start_pos, chunk, spans=spans, on_chunk=on_chunk)

    def _tier_active(self) -> bool:
        if not hasattr(self, "_tier_flag"):
            self._tier_flag = any(getattr(b, "tier", None) is not None for b in self.moe_backends.values())
        return self._tier_flag

    def prefill(self, input_ids: torch.Tensor, st, start_pos: int = 0, chunk: int = None, keep_hidden: int = None,
                spans=None, on_chunk=None):
        auto = not chunk
        chunk = chunk or self.prefill_chunk
        L = input_ids.size(1)
        if auto and L > chunk and self._tier_active():
            # SSD expert tier: every chunk re-reads the experts that are not resident (a chunk of >= 2048 rows picks nearly all of
            # them), whatever its size. A bigger chunk spreads the same reads over more rows, and over enough compute to hide them.
            n = -(-L // max(chunk, int(os.environ.get("EXL3_TIER_PF_CHUNK", "32768"))))
            chunk = max(chunk, -(-L // n))      # equal chunks: no short tail that would pay the full read again
        if L <= chunk:
            return self._forward(input_ids, start_pos, st, None, "last", spans=spans)
        keep = self.cfg.sliding_window if keep_hidden is None else keep_hidden
        acc, out = None, None
        for a in range(0, L, chunk):
            last = a + chunk >= L
            out, hid = self._forward(input_ids[:, a:a + chunk], start_pos + a, st, None, "last" if last else "none",
                                     ahead_ids=None if last else input_ids[:, a + chunk:a + 2 * chunk], spans=spans,
                                     hid_rows=keep or None)
            if hid is not None and keep:
                acc = hid[:, -keep:] if acc is None else torch.cat([acc, hid], dim=1)[:, -keep:]
            if on_chunk is not None and not last:
                on_chunk(start_pos + a + input_ids[:, a:a + chunk].size(1), hid)   # state is complete up to this chunk end
        return out, acc

    def forward_block(self, input_ids: torch.Tensor, start_pos: int, st, taps: dict = None):
        """Speculative verify: L tokens in one pass, logits of every position [B, L, vocab] fp32 and the target-layer
        hidden of every position. `st.begin_block(start_pos, L)` first makes `st.rollback(keep)` possible."""
        if start_pos <= 0:
            raise ValueError("forward_block needs start_pos > 0 (prefill goes through forward())")
        return self._forward(input_ids, start_pos, st, taps, "all")

    def forward_chunk(self, input_ids, start_pos, st, taps=None):
        return self.prefill(input_ids, st, start_pos)

    def generate(self, prompt_ids, max_new_tokens: int, st=None, eos_id: int = -1, sampler=None, max_seq: int = None):
        """Minimal greedy / sampled loop (`sampler(logits_row) -> int`, default argmax). Returns the new token ids."""
        ids = torch.as_tensor([list(prompt_ids)], dtype=torch.int64)
        st = st or self.new_state(1, max_seq or (ids.size(1) + max_new_tokens + 1))
        logits, _ = self.prefill(ids, st, 0)
        pos, out = ids.size(1), []
        pick = sampler or (lambda lg: int(lg.argmax()))
        for _ in range(max_new_tokens):
            tok = pick(logits[0])
            out.append(tok)
            if tok == eos_id:
                break
            logits, _ = self._forward(torch.tensor([[tok]]), pos, st, None, "last")
            pos += 1
        return out
