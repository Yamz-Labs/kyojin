"""
HIP-graph decode step of the DS4.1 serving model (1 token, batch 1).

Eager decode is launch-bound (about 170 tiny torch kernels per layer, host dispatch > GPU time). This module
re-expresses the decode step with static shapes and device-resident positions so the whole trunk replays as a few
captured graphs:

  * the position lives in a device tensor (`pos`): rope rows, ring slot, window indices, compressor slot / boundary and
    pool row are computed on device (gather / index_copy), never from a host int;
  * the compressed pools are read at their full paged capacity (`cap` rows) with the unused rows masked (index -1 / -inf),
    so shapes only change when a pool page is added (then the graph is re-captured);
  * the compressor-boundary `if` is a masked write: the latent is computed every step and stored only when the group
    closes (`where(do, new, old)`), so state and pools equal the eager path.

Engram layers (host row fetch + unique) stay eager: the trunk is split into one graph per run of layers between them.
Graphs are keyed on the cache tensors (their addresses are baked in); a new cache or a grown pool captures again after
one eager step. Speculative verify blocks, journals, taps, batch > 1 and loop-MoE layers use the eager path.
Enable with `DSV41_DECODE_GRAPH=1` (or `model.decode_graph = True`).
"""
from __future__ import annotations
import torch
from . import ops, quant
from .model import SharedRt


def graph_supported(model, st) -> bool:
    from .serve_moe import GroupedMoE
    cfg = model.cfg
    if model.device.type != "cuda" or getattr(st, "_kv", None) is None:
        return False
    for l in range(cfg.num_hidden_layers):
        be = model.moe_backends.get(l)
        if not (isinstance(be, GroupedMoE) and be.decode_ok):
            return False
        if cfg.ratio(l) == 1:
            return False
    return model.lw[0].engram is None


class _Seg:
    pass


class DecodeGraph:
    def __init__(self, model, st):
        self.m, self.st = model, st
        cfg, dev = model.cfg, model.device
        self.tok = torch.zeros(1, 1, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int64, device=dev)
        win = cfg.sliding_window
        self.ar_win = torch.arange(win, device=dev)
        self.wtab = torch.stack([ops._window_idxs_block(win, 1, 1, sp, dev).view(win) for sp in range(1, 2 * win)])
        self.ratios = sorted({cfg.ratio(l) for l in range(cfg.num_hidden_layers)})
        D = cfg.hidden_size
        elayers = [l for l in range(cfg.num_hidden_layers) if model.lw[l].engram is not None]
        cuts = [0] + [l for l in elayers if l > 0] + [cfg.num_hidden_layers]
        self.segs = []
        for i in range(len(cuts) - 1):
            s = _Seg()
            s.l0, s.l1 = cuts[i], cuts[i + 1]
            s.hin = None if i == 0 else torch.zeros(1, 1, cfg.hc_mult, D, dtype=torch.float32, device=dev)
            s.pin = None if i == 0 else torch.zeros(1, 1, cfg.hc_mult, dtype=torch.float32, device=dev)
            s.engram_first = i > 0 or (cuts[0] in elayers)
            s.graph, s.out = None, None
            self.segs.append(s)
        self.key = self.make_key(st)
        self.warm = False

    @staticmethod
    def make_key(st):
        ks = [t.data_ptr() for t in st.win.values()]
        ks += [st.comp_kv[l].data_ptr() for l in sorted(st.comp_kv)] + [st.idx_k[l].data_ptr() for l in sorted(st.idx_k)]
        ks += [st.comp_kv[l].size(1) for l in sorted(st.comp_kv)]
        return tuple(ks)

    # ------------------------------------------------------------------ per-step constants
    def _consts(self, st):
        """Position-derived device tensors shared by every layer (computed once per segment, not per layer)."""
        cfg, win = self.m.cfg, self.m.cfg.sliding_window
        pos = self.pos
        c = {"slot": pos % win,
             "widx": self.wtab.index_select(0, torch.where(pos < win, pos, win + pos % win) - 1).view(1, 1, win)}
        for r in self.ratios:
            fc_t = self.m._rope_for(r, st.max_seq)
            d = {"fcp": fc_t.index_select(0, pos), "fc_t": fc_t}
            if r:
                d["cslot"] = pos % r
                d["do"] = ((pos + 1) % r == 0)
                d["row"] = pos // r
                d["ck"] = fc_t.index_select(0, (pos + 1 - r).clamp_min(0))
                d["clen"] = ((pos + 1) // r).view(1, 1, 1)
            c[r] = d
        return c

    # ------------------------------------------------------------------ static attention
    def _compressor(self, l, x, st, k):
        cfg, c = self.m.cfg, self.m.lw[l].comp
        r = cfg.ratio(l)
        xf = x.float()
        kv, score = c.wkv(xf), c.wgate(xf)
        kst, sst = st.comp_kv_state[l], st.comp_score_state[l]
        slot = k["cslot"]
        kst.index_copy_(1, slot, kv.float())
        sst.index_copy_(1, slot, score.float())
        pooled = (kst * sst.softmax(dim=1)).sum(dim=1, keepdim=True)
        return ops.rmsnorm(pooled.to(x.dtype), c.norm, cfg.rms_norm_eps), k["do"]

    def _indexer(self, l, x, qr, latent, do, st, rt, k_, cap, offset):
        cfg, ix = self.m.cfg, self.m.lw[l].idx
        B = 1
        ratio, rd = cfg.ratio(l), cfg.qk_rope_head_dim
        nh, hd = cfg.index_n_heads, cfg.index_head_dim
        row, fcp = k_["row"], k_["fcp"]
        if ix.owns_k:
            if latent is not None:
                ck = k_["ck"]
                k = ops.rmsnorm(ix.wk(latent), ix.k_norm, cfg.rms_norm_eps)
                ops.rope_apply_(k[..., -rd:], ck)
                quant.fp4_act_quant_(k, 32)
                pool = st.idx_k[l]
                pool.index_copy_(1, row, torch.where(do, k.to(pool.dtype), pool.index_select(1, row)))
            rt.index_k = st.idx_k[l]
        q = ix.wq_b(qr).unflatten(-1, (nh, hd))
        ops.rope_apply_(q[..., -rd:], fcp)
        quant.fp4_act_quant_(q, 32)
        index_k = rt.index_k[:B, :cap].float()
        weights = ix.weights_proj(x).float() * (hd ** -0.5 * nh ** -0.5)
        s = torch.einsum("bshd,btd->bsht", q.float(), index_k)
        score = (s.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        clen = k_["clen"]
        score.masked_fill_(torch.arange(cap, device=x.device) >= clen, -torch.inf)
        if cfg.candidate_source_layer_id >= 0:
            if l == cfg.candidate_source_layer_id:
                rt.candidates = ops.select_candidate_blocks(score, clen, cfg.candidate_topk_blocks,
                                                            cfg.candidate_block_size)
            elif l > cfg.candidate_source_layer_id:
                score = score.masked_fill(~rt.candidates, -torch.inf)
        topk = min(cfg.index_topk, cap)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < clen, idxs + offset, -1).int()

    def _attention(self, l, x, st, rt, K):
        m = self.m
        cfg, w = m.cfg, m.lw[l]
        H, hd, rd, win = cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim, cfg.sliding_window
        ratio = cfg.ratio(l)
        k_ = K[ratio]
        fcp = k_["fcp"]
        qr = ops.rmsnorm(w.wq_a(x), w.q_norm, cfg.rms_norm_eps)
        q = w.wq_b(qr).unflatten(-1, (H, hd))
        ops.rope_apply_(q[..., -rd:], fcp)
        kv = ops.rmsnorm(w.wkv(x), w.kv_norm, cfg.rms_norm_eps)
        ops.rope_apply_(kv[..., -rd:], fcp)
        quant.act_quant_fp8_(kv, cfg.fp8_block_size)
        ring = st.win[l]
        kv = kv.to(ring.dtype)
        window_kv = torch.cat([ring[:1], kv], dim=1)
        widx = K["widx"]
        ring.index_copy_(1, K["slot"], kv)
        if ratio:
            offset = window_kv.size(1)
            latent, do = None, None
            if cfg.is_kv_source(l):
                latent, do = self._compressor(l, x, st, k_)
                rt.compress_kv = st.comp_kv[l]
            src = cfg.kv_source_of(l)
            cap = st.comp_kv[src].size(1)
            if not cfg.is_index_source(l):
                cidx = rt.topk
            else:
                cidx = self._indexer(l, x, qr, latent, do, st, rt, k_, cap, offset)
                rt.topk = cidx
            if latent is not None:
                ops.rope_apply_(latent[..., -rd:], k_["ck"])
                quant.fp4_act_quant_(latent, 16, e4m3_scale=True)
                pool, row = st.comp_kv[src], k_["row"]
                pool.index_copy_(1, row, torch.where(do, latent.to(pool.dtype), pool.index_select(1, row)))
            kv_all = torch.cat([window_kv, rt.compress_kv[:1, :cap]], dim=1)
            idxs = torch.cat([widx, cidx], dim=-1)
        else:
            kv_all, idxs = window_kv, widx
        o = ops.sparse_attention(q, kv_all, w.sink, idxs, hd ** -0.5)
        ops.rope_apply_(o[..., -rd:], fcp, True)
        o = o.view(1, 1, cfg.o_groups, -1)
        quant.act_quant_fp8_(o, cfg.fp8_block_size)
        o = m._wo_a(w, o)
        return w.wo_b(o.flatten(2))

    # ------------------------------------------------------------------ segments
    def _run_seg(self, i, rt):
        m, st, seg = self.m, self.st, self.segs[i]
        cfg = m.cfg
        dev = m.device
        if i == 0:
            h = torch.nn.functional.embedding(self.tok, m.embed_w).float()
            h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
            pre = h.new_zeros(1, 1, cfg.hc_mult)
            pre[:, :, 0] = 1.0
        else:
            h, pre = seg.hin, seg.pin
        S = m.hc.new(h)
        main = []
        K = self._consts(st)
        last = i == len(self.segs) - 1
        for l in range(seg.l0, seg.l1):
            w = m.lw[l]
            if l in cfg.dspark_target_layer_ids and m.mtp is not None:
                main.append(S.apply().view(1, 1, cfg.hc_mult, -1).mean(dim=2))
            x = S.cn(pre, w.attn_norm)
            a_pre, a_post, a_comb = S.mix(w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
            y = self._attention(l, x, st, rt, K)
            if m.ablit is not None:
                y = m._abl(l, 0, y)
            S.push(y, a_post, a_comb)
            x = S.cn(a_pre, w.ffn_norm)
            f_pre, f_post, f_comb = S.mix(w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
            y = m._moe(l, x, None)
            if m.ablit is not None:
                y = m._abl(l, 1, y)
            S.push(y, f_post, f_comb)
            pre = f_pre
        if last:
            x = S.cn(pre, m.norm_w)
            return (m.head_logits(x[:, -1]), main)
        return (S.inject(lambda hh: hh), pre, main)

    def _seg_fn(self, i, rt):
        return lambda: self._run_seg(i, rt)

    # ------------------------------------------------------------------ driver
    def step(self, input_ids, start_pos, hashes):
        """One decode token. The first call after (re)creation runs each segment eagerly on a side stream (real result,
        state writes are idempotent) and captures it; later calls replay."""
        m = self.m
        self.tok.copy_(input_ids)
        self.pos.fill_(start_pos)
        capture = not self.warm
        rt_e = SharedRt()
        if capture:
            self.rt_g = SharedRt()
        mains, out, prev = [], None, None
        for i, seg in enumerate(self.segs):
            if i > 0:
                e = m.lw[seg.l0].engram
                seg.hin.copy_(m_engram(m, prev[0], hashes, e))
                seg.pin.copy_(prev[1])
            if capture:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    res = self._run_seg(i, rt_e)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    seg.out = self._run_seg(i, self.rt_g)
                seg.graph = g
            else:
                seg.graph.replay()
                res = seg.out
            prev = res
            mains += res[-1]
            if i == len(self.segs) - 1:
                out = res[0]
        self.warm = True
        logits = out.clone()
        hid = torch.cat(mains, dim=-1) if mains else None
        return logits, hid


def m_engram(m, hend, hashes, e):
    from .engram import engram_inject
    cfg = m.cfg
    return engram_inject(hend, hashes[:, :, e.pos, :], m.engram_store, e.pos, e.wkv, e.qk, cfg.hc_mult,
                         cfg.hidden_size, cfg.rms_norm_eps)
