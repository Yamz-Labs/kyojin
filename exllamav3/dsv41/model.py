"""
DeepSeek-V4.1-Flash graph (trunk + MTP/dspark) on plain torch. CPU and HIP, any activation dtype.

Differences from the reference's code structure (not its math): per-sequence caches live in
`DSV41State` (no module buffers / process globals), shared-attention hand-offs live in a per-forward
`SharedRt`, weights come from a `WeightSource`, routed experts from an `ExpertProvider`, Engram rows
from an `EngramRowStore`.

Reference: DeepSeek-V4.1-Flash inference/model.py (line numbers in the comments).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from .config import DSV41Config
from . import ops, quant
from .engram import EngramLayout, NgramHasher, engram_inject, EngramRowStore
from .experts import ExpertProvider, ExpertHandle
from .linear import Lin, DenseLin
from .vision import span_mask, splice


# ---------------------------------------------------------------------------------------- state
class DSV41State:
    """All per-sequence mutable state of the trunk and the MTP stages."""

    def __init__(self, model: "DSV41Model", batch: int = 1, max_seq: int = 4096):
        cfg, dev, dt = model.cfg, model.device, model.dtype
        self.batch, self.max_seq = batch, max_seq
        n = cfg.num_hidden_layers
        W, hd = cfg.sliding_window, cfg.head_dim
        self.win = {l: torch.zeros(batch, W, hd, dtype=dt, device=dev) for l in range(n + cfg.n_mtp_layers)}
        self.comp_kv, self.comp_kv_state, self.comp_score_state, self.idx_k = {}, {}, {}, {}
        for l in cfg.kv_source_layer_ids:
            r = cfg.ratio(l)
            self.comp_kv[l] = torch.zeros(batch, max_seq // r, hd, dtype=dt, device=dev)
            self.idx_k[l] = torch.zeros(batch, max_seq // r, cfg.index_head_dim, dtype=dt, device=dev)
            if r > 1:
                self.comp_kv_state[l] = torch.zeros(batch, r, hd, dtype=torch.float32, device=dev)
                self.comp_score_state[l] = torch.full((batch, r, hd), -torch.inf, dtype=torch.float32, device=dev)
        self.hasher = None
        if model.engram_layout is not None:
            self.hasher = NgramHasher(model.engram_layout, model.token_map, cfg.engram_pad_token_id,
                                      model.compressed_vocab, model.multipliers, batch, max_seq, dev)
        self.journal = None

    # ---------------------------------------------------------------- block rollback (speculative verify)
    # A verify block of L tokens overwrites state that older positions still need: sliding-window ring slots
    # (position p lives in slot p % W) and the open group of the compressors (slot = position % ratio). Those
    # writes are logged here (old values only: a few rows per layer) so `rollback(keep)` can undo the rejected
    # tail without recomputing anything. Everything else a forward writes is positional and rewritten before it
    # is read again: compressed pools and indexer keys (rows >= accepted // ratio), the engram token history
    # (positions >= accepted). The hyper-connection mixes (`pre_mix`) never leave one forward.

    def begin_block(self, start_pos: int, length: int):
        """Start logging for the next forward of `length` tokens at `start_pos`."""
        if length > self.win[0].size(1):
            raise ValueError(f"verify block of {length} tokens exceeds the sliding window ({self.win[0].size(1)})")
        self.journal = _BlockJournal(start_pos, length)

    def end_block(self):
        self.journal = None

    def _jr_ring(self, l: int, positions, B: int):
        j = self.journal
        if j is None:
            return
        ring = self.win[l]
        W = ring.size(1)
        slots = torch.tensor([p % W for p in positions], device=ring.device)
        j.ring.setdefault(l, []).append((list(positions), ring[:B, slots].clone()))

    def _jr_comp(self, l: int, pos: int, slot: int, B: int):
        j = self.journal
        if j is None:
            return
        j.comp.setdefault(l, []).append((pos, slot, self.comp_kv_state[l][:B, slot].clone(),
                                         self.comp_score_state[l][:B, slot].clone()))

    def rollback(self, keep: int):
        """Make the state equal to one that only saw the first `keep` tokens of the journaled block (0 <= keep <= L).
        The journal is consumed."""
        j = self.journal
        if j is None:
            raise RuntimeError("rollback without begin_block")
        if not 0 <= keep <= j.length:
            raise ValueError(f"keep {keep} outside the block of {j.length}")
        limit = j.start + keep                         # positions >= limit are rejected
        for l, recs in j.ring.items():
            ring = self.win[l]
            W = ring.size(1)
            for positions, old in reversed(recs):
                for i, p in enumerate(positions):
                    if p >= limit:
                        ring[:old.size(0), p % W] = old[:, i]
        for l, recs in j.comp.items():
            kst, sst = self.comp_kv_state[l], self.comp_score_state[l]
            for pos, slot, ok, os_ in reversed(recs):
                if pos >= limit:
                    kst[:ok.size(0), slot] = ok
                    sst[:os_.size(0), slot] = os_
        self.journal = None


class _BlockJournal:
    def __init__(self, start: int, length: int):
        self.start, self.length = start, length
        self.ring = {}      # layer -> [(positions, old ring rows [B, n, hd])]
        self.comp = {}      # layer -> [(position, slot, old kv_state row, old score_state row)]


@dataclass
class SharedRt:
    compress_kv: torch.Tensor = None
    index_k: torch.Tensor = None
    index_kq: tuple = None                 # serving: packed fp4 copy (codes, e8m0 scales) of index_k
    topk: torch.Tensor = None
    candidates: torch.Tensor = None
    cand_blocks: torch.Tensor = None       # serving: block ids [rows, nsel] int32 (-1 none) of the candidate source


class _LW:
    pass


# ---------------------------------------------------------------------------------------- model
class DSV41Graph:
    """Layer math of the trunk / MTP blocks, independent of where the weights live. `self.lw[l]` holds the
    layer weights (callable Lin objects + raw tensors); `_moe` is supplied by the subclass. Used by DSV41Model
    (full forward) and by the calibration runner (one block at a time)."""

    def __init__(self, cfg: DSV41Config, dtype=torch.float32, device="cpu"):
        self.cfg, self.dtype, self.device = cfg, dtype, torch.device(device)
        self.lw = {}
        self._rope = {}
        self.ablit = None
        self._img_rows = None       # bool [rows] of the pass in flight: rows inside an image span (VL routing bias)

    def _route_bias(self, w):
        """Expert-selection bias: `gate_b`, or per row `gate_b_vl` for the rows of an image span."""
        m = self._img_rows
        vl = getattr(w, "gate_b_vl", None)
        if m is None or vl is None:
            return w.gate_b
        return torch.where(m.unsqueeze(-1), vl, w.gate_b)

    def set_ablit(self, model_dir = None):
        """Runtime refusal-direction projection (EXL3_ABLIT_RUNTIME=<spec>|off, or <model_dir>/uncensor_spec.json): per
        trunk layer (w_attn, w_ffn, r32, r16). Applied to the attention output and to the MoE output before hc_post."""
        from ..modules import ablit_runtime
        self.ablit = ablit_runtime.prepare_dsv41(self.cfg.num_hidden_layers, self.device, model_dir)
        return self.ablit

    def _abl(self, l, which, y):
        a = self.ablit
        if a is None or l >= len(a) or a[l] is None or a[l][which] == 0.0:
            return y
        from ..modules import ablit_runtime
        if not y.is_contiguous():
            y = y.contiguous()
        return ablit_runtime.project(y, a[l][which], *a[l][2:])

    def _rope_for(self, ratio: int, max_seq: int):
        cfg = self.cfg
        key = (bool(ratio), max_seq)
        if key not in self._rope:
            if ratio:
                a = (cfg.original_seq_len, cfg.compress_rope_theta)
            else:
                a = (0, cfg.rope_theta)       # sliding layers: plain rope, no yarn
            self._rope[key] = ops.rope_tables(cfg.qk_rope_head_dim, max_seq, a[0], a[1], cfg.rope_factor,
                                              cfg.beta_fast, cfg.beta_slow, self.device)
        return self._rope[key]


    # ---------------------------------------------------------------- attention
    def _compressor(self, l, x, start_pos, st):
        """model.py:429. Pre-RoPE compressed latent, or None while the group is incomplete."""
        cfg, c = self.cfg, self.lw[l].comp
        B, L, _ = x.shape
        ratio = cfg.ratio(l)
        if ratio == 1:
            return ops.rmsnorm(c.wkv(x), c.norm, cfg.rms_norm_eps)
        dtype = x.dtype
        xf = x.float()
        kv, score = c.wkv(xf), c.wgate(xf)
        kst, sst = st.comp_kv_state[l], st.comp_score_state[l]
        if start_pos == 0:
            do = L >= ratio
            rem = L % ratio
            cut = L - rem
            if rem:
                kv, kst[:B, :rem] = kv.split([cut, rem], dim=1)
                score, sst[:B, :rem] = score.split([cut, rem], dim=1)
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        elif L > 1:
            # verify block: the same per-position state updates as L decode steps, one batched projection
            outs = []
            for i in range(L):
                q = start_pos + i
                slot = q % ratio
                st._jr_comp(l, q, slot, B)
                kst[:B, slot] = kv[:, i]
                sst[:B, slot] = score[:, i]
                if (q + 1) % ratio == 0:
                    outs.append((kst[:B] * sst[:B].softmax(dim=1)).sum(dim=1, keepdim=True))
            if not outs:
                return None
            kv, do = torch.cat(outs, dim=1), True
        else:
            do = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            st._jr_comp(l, start_pos, slot, B)
            kst[:B, slot] = kv.squeeze(1)
            sst[:B, slot] = score.squeeze(1)
            if do:
                kv = (kst[:B] * sst[:B].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not do:
            return None
        return ops.rmsnorm(kv.to(dtype), c.norm, cfg.rms_norm_eps)

    @staticmethod
    def _latent_geom(start_pos: int, L: int, ratio: int, fc_t):
        """(rope rows, first pool row) of the compressed latents completed by positions start_pos..start_pos+L-1.
        A group is complete at its last position q ((q + 1) % ratio == 0); its rope row is its first position."""
        if start_pos == 0:
            return fc_t[: L - L % ratio: ratio], 0
        if L == 1:
            return fc_t[start_pos + 1 - ratio].unsqueeze(0), start_pos // ratio
        qs = [q for q in range(start_pos, start_pos + L) if (q + 1) % ratio == 0]
        return fc_t[torch.tensor([q + 1 - ratio for q in qs], device=fc_t.device)], qs[0] // ratio

    def _indexer(self, l, x, qr, latent, start_pos, offset, st, rt, fc):
        """model.py:488-581. Returns int32 top-k indices into [window | compressed] (offset applied)."""
        cfg, ix = self.cfg, self.lw[l].idx
        B, L, _ = x.shape
        ratio, rd, end_pos = cfg.ratio(l), cfg.qk_rope_head_dim, start_pos + L
        nh, hd = cfg.index_n_heads, cfg.index_head_dim
        if ix.owns_k:
            if latent is not None:
                ck, row0 = self._latent_geom(start_pos, L, ratio, fc)
                k = ops.rmsnorm(ix.wk(latent), ix.k_norm, cfg.rms_norm_eps)
                ops.rope_apply_(k[..., -rd:], ck)
                quant.fp4_act_quant_(k, 32)
                st.idx_k[l][:B, row0: row0 + k.size(1)] = k
            rt.index_k = st.idx_k[l]
        q = ix.wq_b(qr).unflatten(-1, (nh, hd))
        ops.rope_apply_(q[..., -rd:], fc[start_pos:end_pos])
        quant.fp4_act_quant_(q, 32)
        index_k = rt.index_k[:B, : end_pos // ratio]
        weights = ix.weights_proj(x) * (hd ** -0.5 * nh ** -0.5)
        score = torch.einsum("bshd,btd->bsht", q, index_k)
        score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0 or L > 1:
            # prefill, or a verify block: every query sees only the groups completed up to its own position
            clen = ((start_pos + torch.arange(1, L + 1, device=x.device)) // ratio).unsqueeze(-1)
            score.masked_fill_(torch.arange(end_pos // ratio, device=x.device) >= clen, -torch.inf)
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

    def _attention(self, l, x, start_pos, st, rt, taps=None):
        cfg, w = self.cfg, self.lw[l]
        B, L, _ = x.shape
        H, hd, rd, win = cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim, cfg.sliding_window
        ratio = cfg.ratio(l)
        fc_t = self._rope_for(ratio, st.max_seq)
        fc = fc_t[start_pos:start_pos + L]
        qr = ops.rmsnorm(w.wq_a(x), w.q_norm, cfg.rms_norm_eps)
        q = w.wq_b(qr).unflatten(-1, (H, hd))
        ops.rope_apply_(q[..., -rd:], fc)
        # sliding-window keys
        kv = ops.rmsnorm(w.wkv(x), w.kv_norm, cfg.rms_norm_eps)
        ops.rope_apply_(kv[..., -rd:], fc)
        quant.act_quant_fp8_(kv, cfg.fp8_block_size)
        ring = st.win[l]
        if start_pos == 0:
            if L <= win:
                ring[:B, :L] = kv
            else:
                cut = L % win
                ring[:B, cut:win], ring[:B, :cut] = kv[:, -win:].split([win - cut, cut], dim=1)
            window_kv = kv
            idxs = ops.window_idxs(win, B, L, start_pos, x.device)
        elif L == 1:
            st._jr_ring(l, [start_pos], B)
            ring[:B, start_pos % win] = kv.squeeze(1)
            window_kv = ring[:B]
            idxs = ops.window_idxs(win, B, L, start_pos, x.device)
        else:
            # verify block: keys = [ring as it was before the block | the block's own keys]; each query reads its
            # last `win` positions in time order, the order and width a decode step uses
            if L > win:
                raise ValueError(f"verify block of {L} tokens exceeds the sliding window ({win})")
            window_kv = torch.cat([ring[:B], kv], dim=1)
            idxs = ops.window_idxs_block(win, B, L, start_pos, x.device)
            positions = list(range(start_pos, start_pos + L))
            st._jr_ring(l, positions, B)
            ring[:B, torch.tensor([p % win for p in positions], device=ring.device)] = kv
        if ratio:
            offset = window_kv.size(1)
            clen_total = (start_pos + L) // ratio
            latent = None
            src = cfg.kv_source_of(l)
            if cfg.is_kv_source(l):
                latent = self._compressor(l, x, start_pos, st)
                rt.compress_kv = st.comp_kv[l]
                if taps is not None and latent is not None:
                    taps[f"L{l}.comp"] = latent.clone()
            # compressed top-k
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
            if latent is not None:
                ck, row0 = self._latent_geom(start_pos, L, ratio, fc_t)
                ops.rope_apply_(latent[..., -rd:], ck)
                quant.fp4_act_quant_(latent, 16, e4m3_scale=True)
                st.comp_kv[cfg.kv_source_of(l)][:B, row0: row0 + latent.size(1)] = latent
            ckv = rt.compress_kv[:B, :clen_total]
            kv_all = torch.cat([window_kv, ckv], dim=1)
            idxs = torch.cat([idxs, cidx], dim=-1)
        else:
            kv_all = window_kv
        o = ops.sparse_attention(q, kv_all, w.sink, idxs, hd ** -0.5)
        ops.rope_apply_(o[..., -rd:], fc, True)
        G = cfg.o_groups
        o = o.view(B, L, G, -1)
        quant.act_quant_fp8_(o, cfg.fp8_block_size)
        o = self._wo_a(w, o)
        return w.wo_b(o.flatten(2))


    def _moe(self, l, x, taps=None):  # pragma: no cover
        raise NotImplementedError

    def _wo_a(self, w, o):
        """Grouped output projection: o [B, L, G, d] -> [B, L, G, r]. `w.wo_a_fn` (per-group Linear modules, or a
        recording wrapper) overrides the dense einsum."""
        fn = getattr(w, "wo_a_fn", None)
        if fn is not None:
            return fn(o)
        G = self.cfg.o_groups
        return torch.einsum("bsgd,grd->bsgr", o, w.wo_a.view(G, self.cfg.o_lora_rank, -1))

    # ---------------------------------------------------------------- block
    def _block_pre(self, l, h, start_pos, pre_mix, st, rt, taps=None):
        """Attention half of a block. Returns (x_ffn [B,L,d] = input of the MoE, ctx for _block_post)."""
        cfg, w = self.cfg, self.lw[l]
        hc, it, he, ne = cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.rms_norm_eps
        resid = h
        a_pre, a_post, a_comb = ops.hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, hc, it, he, ne)
        x = ops.rmsnorm(ops.hc_pre(h, pre_mix), w.attn_norm, ne)
        x = self._attention(l, x, start_pos, st, rt, taps)
        if self.ablit is not None:
            x = self._abl(l, 0, x)
        if taps is not None:
            taps[f"L{l}.attn"] = x.clone()
        h = ops.hc_post(x, resid, a_post, a_comb)
        resid = h
        f_pre, f_post, f_comb = ops.hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, hc, it, he, ne)
        x = ops.rmsnorm(ops.hc_pre(h, a_pre), w.ffn_norm, ne)
        return x, (resid, f_pre, f_post, f_comb)

    def _block_post(self, l, ctx, y, taps=None):
        """hc_post of the FFN output y; returns (block output, f_pre)."""
        resid, f_pre, f_post, f_comb = ctx
        if self.ablit is not None:
            y = self._abl(l, 1, y)
        if taps is not None:
            taps[f"L{l}.ffn"] = y.clone()
        h = ops.hc_post(y, resid, f_post, f_comb)
        if taps is not None:
            taps[f"L{l}.out"] = h.clone()
        return h, f_pre

    def _block(self, l, h, start_pos, pre_mix, st, rt, taps=None):
        x, ctx = self._block_pre(l, h, start_pos, pre_mix, st, rt, taps)
        x = self._moe(l, x, taps)
        return self._block_post(l, ctx, x, taps)


class DSV41Model(DSV41Graph):
    def __init__(self, cfg: DSV41Config, source, provider: ExpertProvider, engram_store: EngramRowStore = None,
                 token_map=None, multipliers=None, primes=None, dtype=torch.float32, device="cpu", ablit_dir=None):
        super().__init__(cfg.validate(), dtype, device)
        self.provider = provider
        self.src = source
        n = cfg.num_hidden_layers
        # ---- engram layout / hash
        self.engram_layout = None
        self.engram_store = engram_store
        self.token_map, self.multipliers = token_map, multipliers
        self.compressed_vocab = cfg.engram_compressed_vocab_size
        if cfg.engram_layer_ids:
            assert token_map is not None, "engram needs the compressed token map"
            if not self.compressed_vocab:
                self.compressed_vocab = int(max(token_map)) + 1
            self.engram_layout = EngramLayout.build(cfg.engram_layer_ids, cfg.engram_num_embeddings,
                                                    cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                                                    cfg.engram_n_heads, cfg.engram_head_dim, primes)
        d, dt = self.device, dtype
        self.embed_w = source.tensor("embed", dt, d)
        self.norm_w = source.tensor("norm", dt, d)
        self.head_w = source.tensor("head", torch.float32, d)
        self.lw = {}
        for l in range(n + cfg.n_mtp_layers):
            self.lw[l] = self._load_layer(l)
        self._rope = {}
        # dspark pieces
        self.mtp = None
        if cfg.n_mtp_layers:
            from .dspark import DSparkHeads
            self.mtp = DSparkHeads(self)
        self.set_ablit(ablit_dir)

    # ---------------------------------------------------------------- weights
    def _load_layer(self, l):
        cfg, s, dt, dv = self.cfg, self.src, self.dtype, self.device
        p = s.layer_prefix(l)
        w = _LW()
        w.l = l
        f32 = torch.float32
        for t in ("attn", "ffn"):
            setattr(w, f"hc_{t}_fn", s.tensor(f"{p}.hc_{t}_fn", f32, dv))
            setattr(w, f"hc_{t}_base", s.tensor(f"{p}.hc_{t}_base", f32, dv))
            setattr(w, f"hc_{t}_scale", s.tensor(f"{p}.hc_{t}_scale", f32, dv))
        w.attn_norm = s.tensor(f"{p}.attn_norm", dt, dv)
        w.ffn_norm = s.tensor(f"{p}.ffn_norm", dt, dv)
        a = f"{p}.attn"
        w.sink = s.tensor(f"{a}.attn_sink", f32, dv)
        w.wq_a, w.wq_b = s.linear(f"{a}.wq_a", dt, dv), s.linear(f"{a}.wq_b", dt, dv)
        w.wkv, w.wo_b = s.linear(f"{a}.wkv", dt, dv), s.linear(f"{a}.wo_b", dt, dv)
        w.q_norm, w.kv_norm = s.tensor(f"{a}.q_norm", dt, dv), s.tensor(f"{a}.kv_norm", dt, dv)
        w.wo_a = s.tensor(f"{a}.wo_a", dt, dv)
        w.comp = None
        if cfg.is_kv_source(l):
            r = cfg.ratio(l)
            c = _LW()
            cdt = f32 if r > 1 else dt
            c.wkv = DenseLin(s.tensor(f"{a}.compressor.wkv", cdt, dv))
            c.wgate = DenseLin(s.tensor(f"{a}.compressor.wgate", f32, dv)) if r > 1 else None
            c.norm = s.tensor(f"{a}.compressor.norm", dt, dv)
            w.comp = c
        w.idx = None
        if cfg.is_index_source(l):
            ix = _LW()
            ix.wq_b = s.linear(f"{a}.indexer.wq_b", dt, dv)
            ix.weights_proj = DenseLin(s.tensor(f"{a}.indexer.weights_proj", dt, dv))
            ix.owns_k = cfg.is_kv_source(l)
            if ix.owns_k:
                ix.wk = DenseLin(s.tensor(f"{a}.indexer.wk", dt, dv))
                ix.k_norm = s.tensor(f"{a}.indexer.k_norm", dt, dv)
            w.idx = ix
        # moe
        n_exp, topk = cfg.moe_counts(l)
        w.gate_w = s.tensor(f"{p}.ffn.gate", f32, dv)
        w.gate_b = s.tensor(f"{p}.ffn.gate.bias", f32, dv)
        # routing bias of image-span rows (V4.1 vision); absent in text-only checkpoints and in the MTP stages
        w.gate_b_vl = s.tensor(f"{p}.ffn.gate.bias_vl", f32, dv) if s.has(f"{p}.ffn.gate.bias_vl") else None
        w.n_exp, w.topk = n_exp, topk
        w.shared = ExpertHandle(l, -1, s.linear(f"{p}.ffn.shared_experts.w1", dt, dv),
                                s.linear(f"{p}.ffn.shared_experts.w3", dt, dv),
                                s.linear(f"{p}.ffn.shared_experts.w2", dt, dv))
        w.engram = None
        if self.engram_layout is not None and l in self.engram_layout.layer_ids:
            e = _LW()
            e.pos = self.engram_layout.layer_ids.index(l)
            e.wkv = s.linear(f"{p}.engram.wkv", dt, dv)
            e.qk = s.tensor(f"{p}.engram.q_weight", f32, dv) * s.tensor(f"{p}.engram.k_weight", f32, dv)
            w.engram = e
        return w

    def new_state(self, batch: int = 1, max_seq: int = 4096) -> DSV41State:
        return DSV41State(self, batch, max_seq)

    def head_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Final-normed activations [..., D] -> fp32 logits."""
        return F.linear(x.float(), self.head_w)

    # ---------------------------------------------------------------- moe
    def _moe(self, l, x, taps=None):
        cfg, w = self.cfg, self.lw[l]
        shape = x.size()
        x = x.view(-1, cfg.hidden_size)
        wts, idx = ops.route(x, w.gate_w, self._route_bias(w), w.topk, cfg.scoring_func, cfg.norm_topk_prob,
                             cfg.routed_scaling_factor, cfg.gate_temp)
        if taps is not None:
            taps[f"L{l}.route.idx"], taps[f"L{l}.route.w"] = idx.clone(), wts.clone()
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(idx.flatten(), minlength=w.n_exp).tolist()
        used = [e for e in range(w.n_exp) if counts[e]]
        self.provider.prefetch(l, used)
        for e in used:
            h = self.provider.lookup(l, e)
            ti, top = torch.where(idx == e)
            y[ti] += h.ffn(x[ti], wts[ti, top, None], cfg.swiglu_limit)
            self.provider.release(h)
        y += w.shared.ffn(x, None, cfg.swiglu_limit)
        return y.type_as(x).view(shape)

    # ---------------------------------------------------------------- forward
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int, st: DSV41State, taps: dict = None, spans=None):
        """Prefill (start_pos == 0, any length) or decode (start_pos > 0, one token). Returns
        (logits of the last position [B, vocab] fp32, main_hidden [B, L, dim*len(targets)] or None).
        `spans`: vision.ImageSpan list (absolute positions) spliced over the embeddings of the prompt."""
        if input_ids.size(1) > 1 and start_pos > 0:
            raise ValueError("chunked prefill with start_pos > 0 goes through forward_chunk()")
        return self._forward(input_ids, start_pos, st, taps, False, spans=spans)

    @torch.inference_mode()
    def forward_block(self, input_ids: torch.Tensor, start_pos: int, st: DSV41State, taps: dict = None):
        """Speculative verify: L >= 1 tokens at start_pos > 0 in one pass (one GEMM per weight for all L rows,
        each query sees exactly the context a decode step at its position sees). Returns
        (logits of every position [B, L, vocab] fp32, main_hidden [B, L, dim*len(targets)] or None).
        Call `st.begin_block(start_pos, L)` first to be able to `st.rollback(keep)` afterwards."""
        if start_pos <= 0:
            raise ValueError("forward_block needs start_pos > 0 (prefill goes through forward())")
        return self._forward(input_ids, start_pos, st, taps, True)

    def _forward(self, input_ids, start_pos, st, taps, all_logits, spans=None):
        try:
            return self._forward_core(input_ids, start_pos, st, taps, all_logits, spans)
        finally:
            self._img_rows = None

    def _forward_core(self, input_ids, start_pos, st, taps, all_logits, spans=None):
        cfg = self.cfg
        input_ids = input_ids.to(self.device)
        img = span_mask(spans, start_pos, input_ids.size(1), self.device)
        keep = None if img is None else (~img).unsqueeze(0)           # [1, L] rows that take part in engram
        hashes = st.hasher(input_ids, start_pos, keep) if st.hasher is not None else None
        h = F.embedding(input_ids, self.embed_w)
        if img is not None:
            splice(h, start_pos, spans)
            self._img_rows = img.repeat(h.size(0))
        h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        pre_mix = h.new_zeros(h.size(0), h.size(1), cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        rt = SharedRt()
        main = []
        for l in range(cfg.num_hidden_layers):
            w = self.lw[l]
            if w.engram is not None:
                if taps is not None:
                    taps[f"L{l}.engram_in"] = h.clone()
                h = engram_inject(h, hashes[:, :, w.engram.pos, :], self.engram_store, w.engram.pos,
                                  w.engram.wkv, w.engram.qk, cfg.hc_mult, cfg.hidden_size, cfg.rms_norm_eps,
                                  token_mask=keep)
                if taps is not None:
                    taps[f"L{l}.engram"] = h.clone()
            if l in cfg.dspark_target_layer_ids:
                main.append(h.mean(dim=2))
            if taps is not None:
                taps[f"L{l}.in"], taps[f"L{l}.pre"] = h.clone(), pre_mix.clone()
            h, pre_mix = self._block(l, h, start_pos, pre_mix, st, rt, taps)
        if taps is not None:
            taps["final.in"], taps["final.pre"] = h.clone(), pre_mix.clone()
        h = ops.hc_pre(h, pre_mix)
        x = ops.rmsnorm(h, self.norm_w, cfg.rms_norm_eps)
        logits = F.linear(x.float() if all_logits else x[:, -1].float(), self.head_w)
        return logits, (torch.cat(main, dim=-1) if main else None)

    @torch.inference_mode()
    def forward_chunk(self, input_ids, start_pos, st, taps=None):
        """Exact fallback for chunked prefill: token at a time (decode semantics)."""
        out = None
        mains = []
        for i in range(input_ids.size(1)):
            out, m = self.forward(input_ids[:, i:i + 1], start_pos + i, st, taps)
            mains.append(m)
        return out, (torch.cat(mains, dim=1) if mains and mains[0] is not None else None)
