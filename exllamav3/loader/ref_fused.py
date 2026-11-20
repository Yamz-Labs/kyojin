"""PART E: fused replacements for two torch reference paths of the decode step (flag HY4_REF_FUSED=1, default OFF).

  router_select(logits, bias, k, w_scale)   sigmoid + (+bias) top-k + gather + normalise + scale: ONE Triton launch per call
                                            (old: sigmoid, add, topk, gather, sum, clamp, div, mul = ~8 launches)
  mla_sink_attend(...)                      same signature / result as attention_fn.mla_sinks.mla_sink_attend: scores, window /
                                            causal / allow mask, per-head sink in the denominator, softmax, p @ ckv as one
                                            online-softmax Triton kernel, grid (T, H), no [H,T,N] score tensor, no mask tensors.
Generic: shapes come from the tensors (E, k, R, P, H, N); nothing is keyed on a model name. Anything outside the kernel's range
(CPU tensor without TRITON_INTERPRET, T > MAX_T, no sinks, non-fp32) falls back to the old torch code, so a caller can always use these.
"""
import os

import torch

ENABLED = os.environ.get("HY4_REF_FUSED", "0") == "1"
HB = os.environ.get("HY4_MLA_HB", "0") == "1"      # head-batched fp16 tl.dot decode attention (opt-in)
HB_BN = int(os.environ.get("HY4_MLA_HB_BN", "16"))
HB_NW = int(os.environ.get("HY4_MLA_HB_NW", "4"))
HB_SPLIT = int(os.environ.get("HY4_MLA_HB_SPLIT", "0"))      # key splits per (row, head group): 0 = auto (fill the GPU: ~96 programs), 1 = off, n = fixed
ROWS2 = os.environ.get("HY4_MLA_ROWS2", "0") == "1"      # row-batched split-K decode attention (mla2): ONE pass over the K rows for all verify rows, q streamed (never resident in registers); default OFF
R2_BH = int(os.environ.get("HY4_MLA_ROWS2_BH", "16"))      # heads per program (M tile = BH x RG rows of the WMMA tile)
R2_RG = int(os.environ.get("HY4_MLA_ROWS2_RG", "1"))       # verify rows per program (power of two); more rows go over the grid (micro 20/11: RG 1 ~ RG 2 per row, the win is the spill-free q stream, not the row sharing)
R2_BN = int(os.environ.get("HY4_MLA_ROWS2_BN", "32"))      # keys per block
R2_BK = int(os.environ.get("HY4_MLA_ROWS2_BK", "64"))      # latent chunk of the QK reduction (q / K streamed in chunks)
R2_NW = int(os.environ.get("HY4_MLA_ROWS2_NW", "4"))
R2_PROG = int(os.environ.get("HY4_MLA_ROWS2_PROG", "64"))  # target number of split programs (auto split count)
R2_SPLIT = int(os.environ.get("HY4_MLA_ROWS2_SPLIT", "0"))  # key splits: 0 = auto, n = fixed
R2_RS = int(os.environ.get("HY4_MLA_ROWS2_RS", "8"))       # column slices of the combine
R2_MINB = int(os.environ.get("HY4_MLA_ROWS2_MINB", "2"))   # minimum key blocks per split
MAX_T = 16                  # rows per call handled by the attention kernel (decode / short verify); more = prefill -> torch GEMMs
CLAMP_MIN = 6.103515625e-5  # the old router's `sum.clamp(min=...)`

try:
    import triton
    import triton.language as tl
except Exception:           # no triton: every function below uses its torch reference
    triton = None


def _usable(*ts):
    if triton is None:
        return False
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return True
    return all(t.is_cuda for t in ts)


# ------------------------------------------------------------------------------------------------ router
def router_ref(logits, bias, k, w_scale):
    """the old code (hy4_forward._ffn), verbatim maths: logits [T, E] = x @ router.T -> sel [T, k] int64, w [T, k] fp32"""
    p = torch.sigmoid(logits)
    sel = torch.sort(p + bias, dim=-1, descending=True, stable=True).indices[:, :k]
    w = torch.gather(p, 1, sel)
    w = w / w.sum(-1, keepdim=True).clamp(min=CLAMP_MIN) * w_scale
    return sel, w


if triton is not None:
    @triton.jit
    def _router_kernel(LG, BIAS, SEL, W, E, w_scale, clamp_min, s_lg,
                       HAS_BIAS: tl.constexpr, K: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
        t = tl.program_id(0)
        offe = tl.arange(0, BE)
        me = offe < E
        x = tl.load(LG + t * s_lg + offe, mask=me, other=0.0).to(tl.float32)
        p = 1.0 / (1.0 + tl.exp(-x))
        sc = p
        if HAS_BIAS:
            sc = p + tl.load(BIAS + offe, mask=me, other=0.0).to(tl.float32)
        sc = tl.where(me, sc, float("-inf"))
        offk = tl.arange(0, BK)
        selv = tl.zeros([BK], dtype=tl.int32)
        pv = tl.zeros([BK], dtype=tl.float32)
        for j in tl.static_range(K):
            mx = tl.max(sc, axis=0)
            idx = tl.min(tl.where(sc == mx, offe, BE), axis=0)      # ties -> lowest expert id, like torch.topk
            hit = offe == idx
            pj = tl.sum(tl.where(hit, p, 0.0), axis=0)
            selv = tl.where(offk == j, idx, selv)
            pv = tl.where(offk == j, pj, pv)
            sc = tl.where(hit, float("-inf"), sc)
        s = tl.sum(pv, axis=0)
        w = pv / tl.maximum(s, clamp_min) * w_scale
        mk = offk < K
        tl.store(SEL + t * K + offk, selv.to(tl.int64), mask=mk)
        tl.store(W + t * K + offk, w, mask=mk)


def router_select(logits, bias, k, w_scale):
    """one launch: logits [T, E] fp32 (the router matmul output, unchanged) -> (sel [T, k] int64, w [T, k] fp32)"""
    if os.environ.get("HY4_REF_ROUTER", "1") == "0":   # sub-flag: router half off
        return router_ref(logits, bias, k, w_scale)
    if not _usable(logits) or logits.dtype != torch.float32 or k > logits.shape[1]:
        return router_ref(logits, bias, k, w_scale)
    logits = logits.contiguous()
    T, E = logits.shape
    sel = torch.empty(T, k, dtype=torch.int64, device=logits.device)
    w = torch.empty(T, k, dtype=torch.float32, device=logits.device)
    has_bias = bias is not None
    _router_kernel[(T,)](logits, bias.contiguous() if has_bias else logits, sel, w, E, float(w_scale), CLAMP_MIN, logits.stride(0),
                         HAS_BIAS=has_bias, K=k, BE=triton.next_power_of_2(E), BK=triton.next_power_of_2(k))
    return sel, w


# ------------------------------------------------------------------------------------------------ MLA core + sinks
def mla_sink_attend_ref(q_lat, q_pe, ckv, k_pe, qpos, kpos, scale, sinks=None, window=0, allow=None):
    """attention_fn/mla_sinks.py:mla_sink_attend, copied verbatim (the fallback and the test reference)"""
    s = torch.einsum("thr,nr->htn", q_lat, ckv) + torch.einsum("thp,np->htn", q_pe, k_pe)
    s = s * scale
    vis = kpos[None, :] <= qpos[:, None]
    if window:
        vis = vis & ((qpos[:, None] - kpos[None, :]) < window)
    if allow is not None:
        vis = vis & allow
    s = s.masked_fill(~vis[None], float("-inf"))
    if sinks is not None:
        s = torch.cat([s, sinks.float()[:, None, None].expand(-1, s.shape[1], 1)], dim=-1)
    p = torch.softmax(s, dim=-1)
    if sinks is not None:
        p = p[..., :-1]
    return torch.einsum("htn,nr->thr", p, ckv)


if triton is not None:
    @triton.jit
    def _mla_kernel(QL, QP, CK, KP, QPOS, KPOS, ALLOW, SINK, O, N, scale, window,
                    s_ql_t, s_ql_h, s_qp_t, s_qp_h, s_ck, s_kp, s_al, s_o_t, s_o_h,
                    R: tl.constexpr, P: tl.constexpr, BR: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr,
                    HAS_SINK: tl.constexpr, HAS_ALLOW: tl.constexpr, HAS_WIN: tl.constexpr, ARANGE: tl.constexpr, RING: tl.constexpr = 0):
        t = tl.program_id(0)
        h = tl.program_id(1)
        offr = tl.arange(0, BR)
        offp = tl.arange(0, BP)
        mr = offr < R
        mp = offp < P
        q = tl.load(QL + t * s_ql_t + h * s_ql_h + offr, mask=mr, other=0.0).to(tl.float32)
        qp = tl.load(QP + t * s_qp_t + h * s_qp_h + offp, mask=mp, other=0.0).to(tl.float32)
        qpos = tl.load(QPOS + t)
        if HAS_SINK:                                   # the sink is a key with logit s and a zero value: start the softmax from it
            m = tl.load(SINK + h).to(tl.float32)
            l = tl.zeros([], dtype=tl.float32) + 1.0
        else:
            m = tl.zeros([], dtype=tl.float32) + float("-inf")
            l = tl.zeros([], dtype=tl.float32)
        acc = tl.zeros([BR], dtype=tl.float32)
        lo = 0
        hi = N
        if ARANGE:                                     # kpos[j] == j: causal / window bounds skip whole blocks
            hi = tl.minimum(N, qpos + 1)
            if HAS_WIN:
                lo = tl.maximum(0, qpos - window + 1) // BN * BN
        for n0 in range(lo, hi, BN):
            offn = n0 + tl.arange(0, BN)
            inb = offn < N
            if ARANGE:
                kpos = offn
            else:
                kpos = tl.load(KPOS + offn, mask=inb, other=0)
            vis = inb & (kpos <= qpos)
            if HAS_WIN:
                vis = vis & ((qpos - kpos) < window)
            if HAS_ALLOW:
                vis = vis & (tl.load(ALLOW + t * s_al + offn, mask=inb, other=0) != 0)
            if RING > 0:                                   # window layer: the K rows live in a ring buffer, absolute position j in slot j % RING
                offs = offn % RING
            else:
                offs = offn
            ck = tl.load(CK + offs[:, None] * s_ck + offr[None, :], mask=inb[:, None] & mr[None, :], other=0.0).to(tl.float32)
            kp = tl.load(KP + offs[:, None] * s_kp + offp[None, :], mask=inb[:, None] & mp[None, :], other=0.0).to(tl.float32)
            s = (tl.sum(ck * q[None, :], axis=1) + tl.sum(kp * qp[None, :], axis=1)) * scale
            s = tl.where(vis, s, float("-inf"))
            mn = tl.maximum(m, tl.max(s, axis=0))
            mn_s = tl.where(mn == float("-inf"), 0.0, mn)
            alpha = tl.exp(m - mn_s)
            pr = tl.exp(s - mn_s)
            l = l * alpha + tl.sum(pr, axis=0)
            acc = acc * alpha + tl.sum(pr[:, None] * ck, axis=0)
            m = mn
        tl.store(O + t * s_o_t + h * s_o_h + offr, acc / l, mask=mr)

    @triton.jit
    def _mla_hb_kernel(QL, QP, CK, KP, QPOS, SINK, O, N, scale, window,
                       s_ql_t, s_ql_h, s_qp_t, s_qp_h, s_ck, s_kp, s_o_t, s_o_h,
                       R: tl.constexpr, P: tl.constexpr, BH: tl.constexpr, BN: tl.constexpr, HAS_WIN: tl.constexpr, RING: tl.constexpr):
        """head-batched decode attention: ONE program = one query row t x BH heads. The latent K rows are shared by every head (MLA), so each K tile is loaded once for
        BH heads and both products run as fp16 tl.dot (fp32 accumulation, like the prefill kernel); online softmax with the per-head sink as a key of zero value."""
        t = tl.program_id(0)
        h0 = tl.program_id(1) * BH
        offh = h0 + tl.arange(0, BH)
        offr = tl.arange(0, R)
        offp = tl.arange(0, P)
        q = tl.load(QL + t * s_ql_t + offh[:, None] * s_ql_h + offr[None, :]).to(tl.float16)
        qp = tl.load(QP + t * s_qp_t + offh[:, None] * s_qp_h + offp[None, :]).to(tl.float16)
        qpos = tl.load(QPOS + t)
        m = tl.load(SINK + offh).to(tl.float32)
        l = tl.zeros([BH], dtype=tl.float32) + 1.0
        acc = tl.zeros([BH, R], dtype=tl.float32)
        lo = 0
        hi = tl.minimum(N, qpos + 1)
        if HAS_WIN:
            lo = tl.maximum(0, qpos - window + 1) // BN * BN
        for n0 in range(lo, hi, BN):
            offn = n0 + tl.arange(0, BN)
            inb = offn < N
            vis = inb & (offn <= qpos)
            if HAS_WIN:
                vis = vis & ((qpos - offn) < window)
            if RING > 0:
                offs = offn % RING
            else:
                offs = offn
            ck = tl.load(CK + offs[:, None] * s_ck + offr[None, :], mask=inb[:, None], other=0.0).to(tl.float16)
            kp = tl.load(KP + offs[:, None] * s_kp + offp[None, :], mask=inb[:, None], other=0.0).to(tl.float16)
            s = (tl.dot(q, tl.trans(ck)) + tl.dot(qp, tl.trans(kp))) * scale
            s = tl.where(vis[None, :], s, float("-inf"))
            mn = tl.maximum(m, tl.max(s, axis=1))
            alpha = tl.exp(m - mn)
            pr = tl.exp(s - mn[:, None])
            l = l * alpha + tl.sum(pr, axis=1)
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.float16), ck)
            m = mn
        tl.store(O + t * s_o_t + offh[:, None] * s_o_h + offr[None, :], acc / l[:, None])


    @triton.jit
    def _mla_hb_split_kernel(QL, QP, CK, KP, QPOS, SINK, WA, WM, WL, N, scale, window,
                             s_ql_t, s_ql_h, s_qp_t, s_qp_h, s_ck, s_kp,
                             R: tl.constexpr, P: tl.constexpr, BH: tl.constexpr, BN: tl.constexpr, HAS_WIN: tl.constexpr, RING: tl.constexpr, S: tl.constexpr):
        """_mla_hb_kernel over one key split: grid (T, H / BH, S). Split s walks its share of the visible key blocks; split 0 carries the sink (m = sink, l = 1), the others start empty
        (m = -1e30 finite, l = 0). Writes the unnormalised partial (acc [BH, R], m, l) for _mla_hb_combine."""
        t = tl.program_id(0)
        g = tl.program_id(1)
        sp = tl.program_id(2)
        NG = tl.num_programs(1)
        h0 = g * BH
        offh = h0 + tl.arange(0, BH)
        offr = tl.arange(0, R)
        offp = tl.arange(0, P)
        q = tl.load(QL + t * s_ql_t + offh[:, None] * s_ql_h + offr[None, :]).to(tl.float16)
        qp = tl.load(QP + t * s_qp_t + offh[:, None] * s_qp_h + offp[None, :]).to(tl.float16)
        qpos = tl.load(QPOS + t)
        first = (sp == 0)
        m = tl.where(first, tl.load(SINK + offh).to(tl.float32), -1e30)
        l = tl.where(first, 1.0, 0.0) + tl.zeros([BH], dtype=tl.float32)
        acc = tl.zeros([BH, R], dtype=tl.float32)
        lo = 0
        hi = tl.minimum(N, qpos + 1)
        if HAS_WIN:
            lo = tl.maximum(0, qpos - window + 1) // BN * BN
        nblk = tl.maximum(0, (hi - lo + BN - 1) // BN)
        per = (nblk + S - 1) // S
        b0 = tl.minimum(nblk, sp * per)
        b1 = tl.minimum(nblk, b0 + per)
        for n0 in range(lo + b0 * BN, lo + b1 * BN, BN):
            offn = n0 + tl.arange(0, BN)
            inb = offn < N
            vis = inb & (offn <= qpos)
            if HAS_WIN:
                vis = vis & ((qpos - offn) < window)
            if RING > 0:
                offs = offn % RING
            else:
                offs = offn
            ck = tl.load(CK + offs[:, None] * s_ck + offr[None, :], mask=inb[:, None], other=0.0).to(tl.float16)
            kp = tl.load(KP + offs[:, None] * s_kp + offp[None, :], mask=inb[:, None], other=0.0).to(tl.float16)
            s = (tl.dot(q, tl.trans(ck)) + tl.dot(qp, tl.trans(kp))) * scale
            s = tl.where(vis[None, :], s, float("-inf"))
            mn = tl.maximum(m, tl.max(s, axis=1))
            alpha = tl.exp(m - mn)
            pr = tl.exp(s - mn[:, None])
            l = l * alpha + tl.sum(pr, axis=1)
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.float16), ck)
            m = mn
        wi = (t * NG + g) * S + sp
        tl.store(WA + wi * (BH * R) + tl.arange(0, BH)[:, None] * R + offr[None, :], acc)
        tl.store(WM + wi * BH + tl.arange(0, BH), m)
        tl.store(WL + wi * BH + tl.arange(0, BH), l)

    @triton.jit
    def _mla_hb_combine(WA, WM, WL, O, s_o_t, s_o_h, R: tl.constexpr, BH: tl.constexpr, S: tl.constexpr):
        """grid (T, H / BH): merge the S key-split partials (flash-decoding combine): O = sum_s e^(m_s - M) acc_s / sum_s e^(m_s - M) l_s"""
        t = tl.program_id(0)
        g = tl.program_id(1)
        NG = tl.num_programs(1)
        offb = tl.arange(0, BH)
        offr = tl.arange(0, R)
        base = (t * NG + g) * S
        M = tl.load(WM + base * BH + offb)
        for sp in range(1, S):
            M = tl.maximum(M, tl.load(WM + (base + sp) * BH + offb))
        L = tl.zeros([BH], dtype=tl.float32)
        A = tl.zeros([BH, R], dtype=tl.float32)
        for sp in range(0, S):
            w = tl.exp(tl.load(WM + (base + sp) * BH + offb) - M)
            L += w * tl.load(WL + (base + sp) * BH + offb)
            A += w[:, None] * tl.load(WA + (base + sp) * (BH * R) + offb[:, None] * R + offr[None, :])
        h = g * BH + offb
        tl.store(O + t * s_o_t + h[:, None] * s_o_h + offr[None, :], A / L[:, None])

    @triton.jit
    def _mla2_split_kernel(QL, QP, CK, KP, QPOS, SINK, WA, WM, WL, T, N, scale, window,
                           s_ql_t, s_ql_h, s_qp_t, s_qp_h, s_ck, s_kp, S,
                           R: tl.constexpr, P: tl.constexpr, BH: tl.constexpr, RG: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           HAS_WIN: tl.constexpr, RING: tl.constexpr):
        """row-batched split-K decode attention. ONE program = RG verify rows x BH heads (M = RG x BH tile rows: row i = (t0 + i // BH, head h0 + i % BH)) over ONE key split:
        every K block is read once for all RG rows. q is NOT held in registers (a [M, 512] fp16 dot operand needs 256+ VGPRs per wave on gfx11 and spilled ~3.7 KB / thread in the
        _mla_hb kernels): the QK reduction streams q and K in BK chunks, S = sum_c q_c @ K_c^T, so the live state is acc [M, R] fp32 + one S tile. fp16 tl.dot with fp32 accumulation,
        fp32 online softmax per row, the sink is a key of zero value on split 0. Grid (row groups, H / BH, S); partials in the layout of _mla_hb_combine ((t, g, split) major)."""
        M: tl.constexpr = RG * BH
        rg = tl.program_id(0)
        g = tl.program_id(1)
        sp = tl.program_id(2)
        NG = tl.num_programs(1)
        i = tl.arange(0, M)
        tt = rg * RG + i // BH
        hh = g * BH + i % BH
        tok = tt < T
        offk = tl.arange(0, BK)
        offp = tl.arange(0, P)
        qb = QL + tt[:, None] * s_ql_t + hh[:, None] * s_ql_h
        qpb = QP + tt[:, None] * s_qp_t + hh[:, None] * s_qp_h
        qpos = tl.load(QPOS + tt, mask=tok, other=-1)
        first = (sp == 0)
        m = tl.where(first, tl.load(SINK + hh).to(tl.float32), -1e30)
        l = tl.where(first, 1.0, 0.0) + tl.zeros([M], dtype=tl.float32)
        RQ: tl.constexpr = R // 4              # the output is kept as 4 column quarters (acc0..acc3): the PV operand tile of ONE quarter is live at a time (a [BN, 512] fp16 B operand is 128 VGPRs / wave on gfx11)
        offq = tl.arange(0, RQ)
        acc0 = tl.zeros([M, RQ], dtype=tl.float32)
        acc1 = tl.zeros([M, RQ], dtype=tl.float32)
        acc2 = tl.zeros([M, RQ], dtype=tl.float32)
        acc3 = tl.zeros([M, RQ], dtype=tl.float32)
        qlast = tl.load(QPOS + tl.minimum(rg * RG + RG - 1, T - 1))
        qfirst = tl.load(QPOS + rg * RG)
        lo = 0
        hi = tl.minimum(N, qlast + 1)
        if HAS_WIN:
            lo = tl.maximum(0, qfirst - window + 1) // BN * BN
        nblk = tl.maximum(0, (hi - lo + BN - 1) // BN)
        per = (nblk + S - 1) // S
        b0 = tl.minimum(nblk, sp * per)
        b1 = tl.minimum(nblk, b0 + per)
        for n0 in range(lo + b0 * BN, lo + b1 * BN, BN):
            offn = n0 + tl.arange(0, BN)
            inb = offn < N
            vis = inb[None, :] & (offn[None, :] <= qpos[:, None])
            if HAS_WIN:
                vis = vis & ((qpos[:, None] - offn[None, :]) < window)
            if RING > 0:
                offs = offn % RING
            else:
                offs = offn
            kb = CK + offs[None, :] * s_ck
            s = tl.zeros([M, BN], dtype=tl.float32)
            for c in range(0, R, BK):
                qc = tl.load(qb + (c + offk)[None, :], mask=tok[:, None], other=0.0).to(tl.float16)
                kc = tl.load(kb + (c + offk)[:, None], mask=inb[None, :], other=0.0).to(tl.float16)
                s = tl.dot(qc, kc, s)
            qpc = tl.load(qpb + offp[None, :], mask=tok[:, None], other=0.0).to(tl.float16)
            kpc = tl.load(KP + offs[None, :] * s_kp + offp[:, None], mask=inb[None, :], other=0.0).to(tl.float16)
            s = tl.dot(qpc, kpc, s) * scale
            s = tl.where(vis, s, float("-inf"))
            mn = tl.maximum(m, tl.max(s, axis=1))
            alpha = tl.exp(m - mn)
            pr = tl.exp(s - mn[:, None])
            l = l * alpha + tl.sum(pr, axis=1)
            p16 = pr.to(tl.float16)
            cb = CK + offs[:, None] * s_ck + offq[None, :]
            acc0 = tl.dot(p16, tl.load(cb, mask=inb[:, None], other=0.0).to(tl.float16), acc0 * alpha[:, None])
            acc1 = tl.dot(p16, tl.load(cb + RQ, mask=inb[:, None], other=0.0).to(tl.float16), acc1 * alpha[:, None])
            acc2 = tl.dot(p16, tl.load(cb + 2 * RQ, mask=inb[:, None], other=0.0).to(tl.float16), acc2 * alpha[:, None])
            acc3 = tl.dot(p16, tl.load(cb + 3 * RQ, mask=inb[:, None], other=0.0).to(tl.float16), acc3 * alpha[:, None])
            m = mn
        wi = (tt * NG + g) * S + sp
        wb = WA + wi[:, None] * (BH * R) + (i % BH)[:, None] * R + offq[None, :]
        tl.store(wb, acc0, mask=tok[:, None])
        tl.store(wb + RQ, acc1, mask=tok[:, None])
        tl.store(wb + 2 * RQ, acc2, mask=tok[:, None])
        tl.store(wb + 3 * RQ, acc3, mask=tok[:, None])
        tl.store(WM + wi * BH + i % BH, m, mask=tok)
        tl.store(WL + wi * BH + i % BH, l, mask=tok)

    @triton.jit
    def _mla2_combine(WA, WM, WL, O, s_o_t, s_o_h, R: tl.constexpr, BH: tl.constexpr, S: tl.constexpr, RC: tl.constexpr):
        """grid (T, H / BH, R / RC): flash-decoding merge of the S key-split partials for one RC-wide column slice (many small programs, the S partial loads are independent)"""
        t = tl.program_id(0)
        g = tl.program_id(1)
        rs = tl.program_id(2)
        NG = tl.num_programs(1)
        offb = tl.arange(0, BH)
        offc = rs * RC + tl.arange(0, RC)
        base = (t * NG + g) * S
        M = tl.load(WM + base * BH + offb)
        for sp in tl.static_range(1, S):
            M = tl.maximum(M, tl.load(WM + (base + sp) * BH + offb))
        L = tl.zeros([BH], dtype=tl.float32)
        A = tl.zeros([BH, RC], dtype=tl.float32)
        for sp in tl.static_range(0, S):
            w = tl.exp(tl.load(WM + (base + sp) * BH + offb) - M)
            L += w * tl.load(WL + (base + sp) * BH + offb)
            A += w[:, None] * tl.load(WA + (base + sp) * (BH * R) + offb[:, None] * R + offc[None, :])
        h = g * BH + offb
        tl.store(O + t * s_o_t + h[:, None] * s_o_h + offc[None, :], A / L[:, None])


def _mla2_launch(q_lat, q_pe, ckv, k_pe, qpos, sinks, scale, window, ring, out):
    T, H, R = q_lat.shape
    N, P = ckv.shape[0], k_pe.shape[1]
    BH = R2_BH
    RG = min(R2_RG, triton.next_power_of_2(T))
    nrg = -(-T // RG)
    ngr = H // BH
    nblk = (-(-(window + T) // R2_BN) + 1) if window else -(-N // R2_BN)      # key blocks one program can walk
    S = R2_SPLIT if R2_SPLIT > 0 else min(16, max(1, -(-R2_PROG // (nrg * ngr))), max(1, nblk // R2_MINB))
    dv = q_lat.device
    wa = torch.empty(T * ngr * S * BH * R, dtype=torch.float32, device=dv)
    wm = torch.empty(T * ngr * S * BH, dtype=torch.float32, device=dv)
    wl = torch.empty(T * ngr * S * BH, dtype=torch.float32, device=dv)
    _mla2_split_kernel[(nrg, ngr, S)](q_lat, q_pe, ckv, k_pe, qpos, sinks.float().contiguous(), wa, wm, wl, T, (1 << 30) if ring else N, float(scale), int(window),
                                      q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1), ckv.stride(0), k_pe.stride(0), S,
                                      R=R, P=P, BH=BH, RG=RG, BN=R2_BN, BK=R2_BK, HAS_WIN=bool(window), RING=int(ring), num_warps=R2_NW, num_stages=1)
    RS = max(1, min(R2_RS, R // 32))
    _mla2_combine[(T, ngr, RS)](wa, wm, wl, out, out.stride(0), out.stride(1), R=R, BH=BH, S=S, RC=R // RS, num_warps=4)
    return out


def mla_sink_attend(q_lat, q_pe, ckv, k_pe, qpos, kpos, scale, sinks=None, window=0, allow=None, kpos_is_arange=False, ring=0):
    """drop-in for mla_sink_attend: q_lat [T,H,R], q_pe [T,H,P], ckv [N,R], k_pe [N,P] (row strides free), qpos [T], kpos [N],
    sinks [H] or None -> o_lat [T,H,R] fp32. kpos_is_arange (default False = generic): the caller's kpos is torch.arange(N)
    (the Hy4 forward always passes it so), which lets causal / window bounds skip K blocks."""
    assert not ring or (kpos_is_arange and window and allow is None and window < ring), "ring K buffers need the arange / window kernel path"
    if os.environ.get("HY4_REF_MLA", "1") == "0":   # sub-flag: MLA half off
        return mla_sink_attend_ref(q_lat, q_pe, ckv, k_pe, qpos, kpos, scale, sinks, window, allow)
    T, H, R = q_lat.shape
    N, P = ckv.shape[0], k_pe.shape[1]
    if (not _usable(q_lat, ckv) or T > MAX_T or sinks is None or N == 0 or q_lat.dtype != torch.float32 or ckv.dtype != torch.float32
            or q_lat.stride(-1) != 1 or q_pe.stride(-1) != 1 or ckv.stride(-1) != 1 or k_pe.stride(-1) != 1):
        if ring:
            raise RuntimeError("ring K buffer: the fused kernel declined and the torch reference cannot read a ring")
        return mla_sink_attend_ref(q_lat, q_pe, ckv, k_pe, qpos, kpos, scale, sinks, window, allow)
    if (ROWS2 and allow is None and kpos_is_arange and R == triton.next_power_of_2(R) and P == triton.next_power_of_2(P) and H % R2_BH == 0
            and R2_BH * min(R2_RG, triton.next_power_of_2(T)) >= 16 and not (ring and not window)):
        return _mla2_launch(q_lat, q_pe, ckv, k_pe, qpos, sinks, scale, window, ring, torch.empty(T, H, R, dtype=torch.float32, device=q_lat.device))
    if HB and allow is None and kpos_is_arange and R == triton.next_power_of_2(R) and P == triton.next_power_of_2(P) and H % 16 == 0 and not (ring and not window):
        out = torch.empty(T, H, R, dtype=torch.float32, device=q_lat.device)
        ng = H // 16
        S = HB_SPLIT if HB_SPLIT > 0 else min(16, max(1, -(-96 // (T * ng))))
        if S > 1:
            wa = torch.empty(T * ng * S * 16 * R, dtype=torch.float32, device=q_lat.device)
            wm = torch.empty(T * ng * S * 16, dtype=torch.float32, device=q_lat.device)
            wl = torch.empty(T * ng * S * 16, dtype=torch.float32, device=q_lat.device)
            _mla_hb_split_kernel[(T, ng, S)](q_lat, q_pe, ckv, k_pe, qpos, sinks.float().contiguous(), wa, wm, wl, (1 << 30) if ring else N, float(scale), int(window),
                                             q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1), ckv.stride(0), k_pe.stride(0),
                                             R=R, P=P, BH=16, BN=HB_BN, HAS_WIN=bool(window), RING=int(ring), S=S, num_warps=HB_NW)
            _mla_hb_combine[(T, ng)](wa, wm, wl, out, out.stride(0), out.stride(1), R=R, BH=16, S=S, num_warps=4)
            return out
        _mla_hb_kernel[(T, H // 16)](q_lat, q_pe, ckv, k_pe, qpos, sinks.float().contiguous(), out, (1 << 30) if ring else N, float(scale), int(window),
                                     q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1), ckv.stride(0), k_pe.stride(0), out.stride(0), out.stride(1),
                                     R=R, P=P, BH=16, BN=HB_BN, HAS_WIN=bool(window), RING=int(ring), num_warps=HB_NW)
        return out
    out = torch.empty(T, H, R, dtype=torch.float32, device=q_lat.device)
    al = allow.view(torch.uint8) if allow is not None else qpos
    BN = 16
    _mla_kernel[(T, H)](q_lat, q_pe, ckv, k_pe, qpos, kpos, al, sinks.float().contiguous(), out, (1 << 30) if ring else N, float(scale), int(window),
                        q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1), ckv.stride(0), k_pe.stride(0),
                        allow.stride(0) if allow is not None else 0, out.stride(0), out.stride(1),
                        R=R, P=P, BR=triton.next_power_of_2(R), BP=triton.next_power_of_2(P), BN=BN,
                        HAS_SINK=True, HAS_ALLOW=allow is not None, HAS_WIN=bool(window), ARANGE=bool(kpos_is_arange), RING=int(ring),
                        num_warps=4)
    return out
