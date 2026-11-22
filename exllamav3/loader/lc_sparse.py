"""Long-context decode kernels for the k-pool sparse attention family (Hy4 indexer layers; generic by kpool / top_k / head dims, nothing keyed on a model name).

  select_groups(iq, wt, pool, pos, Pb, n_top, kp, tail)
      exact top-n_top selection over the pooled groups, one launch pair per layer: a tiled score kernel (sc[t, p] = sum_h wt[t, h] * relu(iq[t, h] . pool[p]),
      -inf for groups the row cannot see) and a one-program-per-row selection kernel that keeps the row in registers: bisection on the order-preserving int
      key of the fp32 score finds the n_top-th largest score, ties at that score are resolved to the LOWEST group ids (torch.topk leaves them unspecified),
      the selected group ids are compacted in ascending order. Returns (idx [T, n_top] int32, nsel [T] int32). Fewer visible groups than n_top: all of them.
  sparse_mla_attend(q_lat, q_pe, ckv, k_pe, qpos, sel, scale, sinks, kp, tail)
      the decode MLA attend (online softmax, per-head sink) over exactly the keys the dense `allow` mask would keep: the members of the selected groups plus the
      open tail group of the row. Work is O(n_top * kp), independent of the context length (the dense kernel reads every key below the position).
"""
from __future__ import annotations
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

CHECK = os.environ.get("HY4_LC_CHECK", "0") == "1"          # eager (non-captured) calls also run the dense torch selection and count differing row sets (debug / gate)
STATS = {"rows": 0, "diff_rows": 0, "diff_groups": 0, "max_gap": 0.0}
HB_ON = os.environ.get("HY4_LC_HB", "1") == "1"            # fp8 cache only (HY4_KV_FP8=1; the fp32 cache never reaches it): head-blocked sparse attend (one key decode for HB heads); HY4_LC_HB=0 = the one-head kernel
HB_SPLIT = int(os.environ.get("HY4_LC_HB_SPLIT", "-1"))      # with HY4_LC_HB: key splits of the head-blocked kernel (flash-decoding combine); n >= 1 fixed, -1 auto (16 rows <= 3, else 8), 0 = one program per (row, head block)
HB = 16
HB_BN = int(os.environ.get("HY4_LC_HB_BN", "16"))
ENABLED = os.environ.get("HY4_LC_SPARSE", "0") == "1"      # decode: indexer layers use select_groups + sparse_mla_attend instead of the dense allow mask


def _usable(*ts):
    if triton is None:
        return False
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return True
    return all(t.is_cuda for t in ts)


if triton is not None:
    @triton.jit
    def _score_kernel(IQ, WT, POOL, SC, POS, P, s_sc, KP: tl.constexpr, TAIL: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BP: tl.constexpr):
        t = tl.program_id(0)
        p = tl.program_id(1) * BP + tl.arange(0, BP)
        inb = p < P
        pos = tl.load(POS + t)
        gend = p * KP + (KP - 1)
        if TAIL:
            vis = inb & (gend < pos)
        else:
            vis = inb & (gend <= pos)
        d = tl.arange(0, D)
        pk = tl.load(POOL + p[:, None].to(tl.int64) * D + d[None, :], mask=vis[:, None], other=0.0)
        acc = tl.zeros([BP], tl.float32)
        for h in range(H):
            q = tl.load(IQ + (t * H + h) * D + d)
            s = tl.sum(pk * q[None, :], axis=1)
            acc += tl.maximum(s, 0.0) * tl.load(WT + t * H + h)
        acc = tl.where(vis, acc, float("-inf"))
        tl.store(SC + t.to(tl.int64) * s_sc + p, acc, mask=inb)

    @triton.jit
    def _select_kernel(SC, IDX, NSEL, P, s_sc, NTOP: tl.constexpr, BP: tl.constexpr):
        t = tl.program_id(0)
        p = tl.arange(0, BP)
        inb = p < P
        x = tl.load(SC + t.to(tl.int64) * s_sc + p, mask=inb, other=float("-inf"))
        vis = inb & (x > float("-inf"))
        bits = x.to(tl.int32, bitcast=True)
        key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)       # order-preserving signed key of the fp32 score
        nvis = tl.sum(vis.to(tl.int32), 0)
        lo = tl.full([], -2147483648, tl.int64)                                # invariant: count(vis & key >= lo) >= min(NTOP, nvis)
        hi = tl.full([], 2147483647, tl.int64)
        for _ in range(32):
            mid = (lo + hi + 1) // 2
            c = tl.sum((vis & (key >= mid)).to(tl.int32), 0)
            ok = c >= NTOP
            lo = tl.where(ok, mid, lo)
            hi = tl.where(ok, hi, mid - 1)
        gt = vis & (key > lo)
        eq = vis & (key == lo)
        need = NTOP - tl.sum(gt.to(tl.int32), 0)
        eqr = tl.cumsum(eq.to(tl.int32), 0)
        sel = gt | (eq & (eqr <= need))
        rank = tl.cumsum(sel.to(tl.int32), 0) - 1
        tl.store(IDX + t * NTOP + rank, p, mask=sel)
        tl.store(NSEL + t, tl.sum(sel.to(tl.int32), 0))

    @triton.jit
    def _dq8(raw):
        b = raw.to(tl.int32)
        e = (b >> 3) & 15
        m = (b & 7).to(tl.float32)
        v = tl.where(e == 0, m * 0.001953125, (8.0 + m) * tl.exp2((e - 10).to(tl.float32)))
        return tl.where(((b >> 7) & 1) == 1, -v, v)

    @triton.jit
    def _blk(q, qp, CK, KPE, SCL, offn, vis, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc, QF: tl.constexpr):
        if QF:      # HY4_KV_FP8: e4m3 latent bytes + one fp32 scale per token, decoded here
            ck = _dq8(tl.load(CK + offn[:, None].to(tl.int64) * s_ck + offr[None, :], mask=vis[:, None] & mr[None, :], other=0)) \
                * tl.load(SCL + offn.to(tl.int64), mask=vis, other=0.0)[:, None]
        else:
            ck = tl.load(CK + offn[:, None].to(tl.int64) * s_ck + offr[None, :], mask=vis[:, None] & mr[None, :], other=0.0).to(tl.float32)
        kp = tl.load(KPE + offn[:, None].to(tl.int64) * s_kp + offp[None, :], mask=vis[:, None] & mp[None, :], other=0.0).to(tl.float32)
        s = (tl.sum(ck * q[None, :], axis=1) + tl.sum(kp * qp[None, :], axis=1)) * scale
        s = tl.where(vis, s, float("-inf"))
        mn = tl.maximum(m, tl.max(s, axis=0))
        mn_s = tl.where(mn == float("-inf"), 0.0, mn)
        alpha = tl.exp(m - mn_s)
        pr = tl.exp(s - mn_s)
        l = l * alpha + tl.sum(pr, axis=0)
        acc = acc * alpha + tl.sum(pr[:, None] * ck, axis=0)
        return mn, l, acc

    @triton.jit
    def _sparse_mla_kernel(QL, QP, CK, KPE, SCL, POS, IDX, NSEL, SINK, O, scale, s_ql_t, s_ql_h, s_qp_t, s_qp_h, s_ck, s_kp, s_o_t, s_o_h,
                           R: tl.constexpr, P: tl.constexpr, BR: tl.constexpr, BPE: tl.constexpr, BN: tl.constexpr, KG: tl.constexpr, NTOP: tl.constexpr, QF: tl.constexpr):
        t = tl.program_id(0)
        h = tl.program_id(1)
        offr = tl.arange(0, BR)
        offp = tl.arange(0, BPE)
        mr = offr < R
        mp = offp < P
        q = tl.load(QL + t * s_ql_t + h * s_ql_h + offr, mask=mr, other=0.0).to(tl.float32)
        qp = tl.load(QP + t * s_qp_t + h * s_qp_h + offp, mask=mp, other=0.0).to(tl.float32)
        qpos = tl.load(POS + t)
        m = tl.load(SINK + h).to(tl.float32)
        l = tl.zeros([], dtype=tl.float32) + 1.0
        acc = tl.zeros([BR], dtype=tl.float32)
        lo = qpos - qpos % KG                                                   # open tail group of the row (select_tail)
        kk = tl.arange(0, BN)
        offn = lo + kk
        m, l, acc = _blk(q, qp, CK, KPE, SCL, offn, offn <= qpos, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc, QF)
        n = tl.load(NSEL + t)
        nk = n * KG
        for i0 in range(0, nk, BN):
            kj = i0 + kk
            inb = kj < nk
            g = tl.load(IDX + t * NTOP + kj // KG, mask=inb, other=0)
            m, l, acc = _blk(q, qp, CK, KPE, SCL, g * KG + kj % KG, inb, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc, QF)
        tl.store(O + t * s_o_t + h * s_o_h + offr, acc / l, mask=mr)


    @triton.jit
    def _sparse_mla_hb_kernel(QL, QP, CK, KPE, SCL, POS, IDX, NSEL, SINK, O, scale, s_ql_t, s_ql_h, s_qp_t, s_qp_h, s_ck, s_kp, s_o_t, s_o_h,
                              R: tl.constexpr, P: tl.constexpr, BR: tl.constexpr, BPE: tl.constexpr, BN: tl.constexpr, KG: tl.constexpr, NTOP: tl.constexpr,
                              HB: tl.constexpr):
        """fp8 latent cache, HB heads per program: every key row is read and decoded ONCE for HB heads (the one-head kernel decodes it H times); scores and
        the value sum run on tl.dot (fp16 operands: e4m3 values are exact in fp16, the per-token scale is applied to the scores / probabilities in fp32)"""
        t = tl.program_id(0)
        offh = tl.program_id(1) * HB + tl.arange(0, HB)
        offr = tl.arange(0, BR)
        offp = tl.arange(0, BPE)
        mr = offr < R
        mp = offp < P
        q = tl.load(QL + t * s_ql_t + offh[:, None] * s_ql_h + offr[None, :], mask=mr[None, :], other=0.0).to(tl.float16)
        qp = tl.load(QP + t * s_qp_t + offh[:, None] * s_qp_h + offp[None, :], mask=mp[None, :], other=0.0).to(tl.float16)
        qpos = tl.load(POS + t)
        m = tl.load(SINK + offh).to(tl.float32)
        l = tl.zeros([HB], dtype=tl.float32) + 1.0
        acc = tl.zeros([HB, BR], dtype=tl.float32)
        lo = qpos - qpos % KG
        kk = tl.arange(0, BN)
        offn = lo + kk
        m, l, acc = _blk_hb(q, qp, CK, KPE, SCL, offn, offn <= qpos, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc)
        n = tl.load(NSEL + t)
        nk = n * KG
        for i0 in range(0, nk, BN):
            kj = i0 + kk
            inb = kj < nk
            g = tl.load(IDX + t * NTOP + kj // KG, mask=inb, other=0)
            m, l, acc = _blk_hb(q, qp, CK, KPE, SCL, g * KG + kj % KG, inb, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc)
        tl.store(O + t * s_o_t + offh[:, None] * s_o_h + offr[None, :], acc / l[:, None], mask=mr[None, :])

    @triton.jit
    def _blk_hb(q, qp, CK, KPE, SCL, offn, vis, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc):
        ck = _dq8(tl.load(CK + offn[:, None].to(tl.int64) * s_ck + offr[None, :], mask=vis[:, None] & mr[None, :], other=0)).to(tl.float16)      # [BN, BR]
        sc = tl.load(SCL + offn.to(tl.int64), mask=vis, other=0.0)
        kp = tl.load(KPE + offn[:, None].to(tl.int64) * s_kp + offp[None, :], mask=vis[:, None] & mp[None, :], other=0.0).to(tl.float16)
        s = (tl.dot(q, tl.trans(ck)) * sc[None, :] + tl.dot(qp, tl.trans(kp))) * scale                                                          # [HB, BN]
        s = tl.where(vis[None, :], s, float("-inf"))
        mn = tl.maximum(m, tl.max(s, axis=1))
        mn_s = tl.where(mn == float("-inf"), 0.0, mn)
        alpha = tl.exp(m - mn_s)
        pr = tl.exp(s - mn_s[:, None])
        l = l * alpha + tl.sum(pr, axis=1)
        acc = acc * alpha[:, None] + tl.dot((pr * sc[None, :]).to(tl.float16), ck)
        return mn, l, acc


    @triton.jit
    def _sparse_mla_hbs_kernel(QL, QP, CK, KPE, SCL, POS, IDX, NSEL, SINK, WA, WM, WL, scale, s_ql_t, s_ql_h, s_qp_t, s_qp_h, s_ck, s_kp,
                               R: tl.constexpr, P: tl.constexpr, BR: tl.constexpr, BPE: tl.constexpr, BN: tl.constexpr, KG: tl.constexpr, NTOP: tl.constexpr,
                               HB: tl.constexpr, S: tl.constexpr):
        """_sparse_mla_hb_kernel over one split of the selected keys: grid (T, H / HB, S). Split s walks its share of the BN-key blocks of the row's selected groups; split 0 also
        takes the open tail group and carries the sink (m = sink, l = 1), the others start empty (m = -1e30 finite, l = 0). Writes the unnormalised partial (acc [HB, R], m, l)
        for ref_fused._mla_hb_combine (layout ((t * NG + g) * S + s) major). The one-head kernel (T x H programs, each decoding every key) filled 32 of 40 CUs at T = 1."""
        t = tl.program_id(0)
        g = tl.program_id(1)
        sp = tl.program_id(2)
        NG = tl.num_programs(1)
        offh = g * HB + tl.arange(0, HB)
        offr = tl.arange(0, BR)
        offp = tl.arange(0, BPE)
        mr = offr < R
        mp = offp < P
        q = tl.load(QL + t * s_ql_t + offh[:, None] * s_ql_h + offr[None, :], mask=mr[None, :], other=0.0).to(tl.float16)
        qp = tl.load(QP + t * s_qp_t + offh[:, None] * s_qp_h + offp[None, :], mask=mp[None, :], other=0.0).to(tl.float16)
        qpos = tl.load(POS + t)
        first = (sp == 0)
        m = tl.where(first, tl.load(SINK + offh).to(tl.float32), -1e30)
        l = tl.where(first, 1.0, 0.0) + tl.zeros([HB], dtype=tl.float32)
        acc = tl.zeros([HB, BR], dtype=tl.float32)
        kk = tl.arange(0, BN)
        if first:
            lo = qpos - qpos % KG
            offn = lo + kk
            m, l, acc = _blk_hb(q, qp, CK, KPE, SCL, offn, offn <= qpos, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc)
        nk = tl.load(NSEL + t) * KG
        nblk = (nk + BN - 1) // BN
        per = (nblk + S - 1) // S
        b0 = tl.minimum(nblk, sp * per)
        b1 = tl.minimum(nblk, b0 + per)
        for i0 in range(b0 * BN, b1 * BN, BN):
            kj = i0 + kk
            inb = kj < nk
            gg = tl.load(IDX + t * NTOP + kj // KG, mask=inb, other=0)
            m, l, acc = _blk_hb(q, qp, CK, KPE, SCL, gg * KG + kj % KG, inb, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc)
        wi = (t * NG + g) * S + sp
        ar = tl.arange(0, HB)
        tl.store(WA + wi * (HB * BR) + ar[:, None] * BR + offr[None, :], acc, mask=mr[None, :])
        tl.store(WM + wi * HB + ar, m)
        tl.store(WL + wi * HB + ar, l)

def select_groups(iq, wt, pool, pos, Pb, n_top, kp, tail):
    """iq [T, H, d] fp32 contiguous, wt [T, H] fp32, pool [>= Pb, d] fp32, pos [T] long (device) -> (idx [T, n_top] int32, nsel [T] int32)"""
    T, H, D = iq.shape
    assert iq.is_contiguous() and wt.is_contiguous() and pool.is_contiguous() and pool.shape[0] >= Pb and pool.shape[1] == D and Pb >= n_top
    sc = torch.empty(T, Pb, dtype=torch.float32, device=iq.device)
    idx = torch.empty(T, n_top, dtype=torch.int32, device=iq.device)
    nsel = torch.empty(T, dtype=torch.int32, device=iq.device)
    BP = 32
    _score_kernel[(T, triton.cdiv(Pb, BP))](iq, wt, pool, sc, pos, Pb, sc.stride(0), KP=kp, TAIL=bool(tail), H=H, D=D, BP=BP, num_warps=4)
    BPS = triton.next_power_of_2(Pb)
    _select_kernel[(T,)](sc, idx, nsel, Pb, sc.stride(0), NTOP=n_top, BP=BPS, num_warps=16 if BPS >= 4096 else 4)
    return idx, nsel


class SparseSel:
    """what the static indexer returns instead of a dense `allow` mask when HY4_LC_SPARSE is on; mla.attend routes it to sparse_mla_attend"""
    __slots__ = ("idx", "nsel", "kp", "tail", "n_top")

    def __init__(self, idx, nsel, kp, tail):
        self.idx, self.nsel, self.kp, self.tail, self.n_top = idx, nsel, kp, tail, idx.shape[1]


def sparse_mla_attend(q_lat, q_pe, ckv, k_pe, qpos, sel, scale, sinks, scl=None):
    """q_lat [T,H,R] fp32, q_pe [T,H,P], ckv [N,R], k_pe [N,P] (row strides free), qpos [T] long, sel SparseSel -> o_lat [T,H,R] fp32"""
    T, H, R = q_lat.shape
    P = k_pe.shape[1]
    qf = scl is not None           # ckv = uint8 e4m3 bytes [N, R], scl [N] fp32, k_pe bf16 [N, P]
    assert sel.tail and q_lat.dtype == torch.float32 and (ckv.dtype == torch.uint8 if qf else ckv.dtype in (torch.float32, torch.float16)) and q_lat.stride(-1) == 1 and q_pe.stride(-1) == 1 and ckv.stride(-1) == 1 and k_pe.stride(-1) == 1
    out = torch.empty(T, H, R, dtype=torch.float32, device=q_lat.device)
    if qf and HB_ON and HB_SPLIT != 0 and H % HB == 0 and R == triton.next_power_of_2(R) and R <= 1024:
        from . import ref_fused as _rf
        ng, S = H // HB, (HB_SPLIT if HB_SPLIT > 0 else (16 if T <= 3 else 8))      # -1 = auto: the micro sweep (tools/hy4/step/mla_micro.py) best split per row count
        wa = torch.empty(T * ng * S * HB * R, dtype=torch.float32, device=q_lat.device)
        wm = torch.empty(T * ng * S * HB, dtype=torch.float32, device=q_lat.device)
        wl = torch.empty(T * ng * S * HB, dtype=torch.float32, device=q_lat.device)
        _sparse_mla_hbs_kernel[(T, ng, S)](q_lat, q_pe, ckv, k_pe, scl, qpos, sel.idx, sel.nsel, sinks.float().contiguous(), wa, wm, wl, float(scale),
                                           q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1), ckv.stride(0), k_pe.stride(0),
                                           R=R, P=P, BR=R, BPE=triton.next_power_of_2(P), BN=HB_BN, KG=sel.kp, NTOP=sel.n_top, HB=HB, S=S, num_warps=4)
        _rf._mla_hb_combine[(T, ng)](wa, wm, wl, out, out.stride(0), out.stride(1), R=R, BH=HB, S=S, num_warps=4)
        return out
    if qf and HB_ON and H % HB == 0 and R <= 1024:
        _sparse_mla_hb_kernel[(T, H // HB)](q_lat, q_pe, ckv, k_pe, scl, qpos, sel.idx, sel.nsel, sinks.float().contiguous(), out, float(scale),
                                            q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1), ckv.stride(0), k_pe.stride(0), out.stride(0), out.stride(1),
                                            R=R, P=P, BR=triton.next_power_of_2(R), BPE=triton.next_power_of_2(P), BN=HB_BN, KG=sel.kp, NTOP=sel.n_top, HB=HB, num_warps=4)
        return out
    _sparse_mla_kernel[(T, H)](q_lat, q_pe, ckv, k_pe, scl if qf else ckv, qpos, sel.idx, sel.nsel, sinks.float().contiguous(), out, float(scale),
                               q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1), ckv.stride(0), k_pe.stride(0), out.stride(0), out.stride(1),
                               R=R, P=P, BR=triton.next_power_of_2(R), BPE=triton.next_power_of_2(P), BN=16, KG=sel.kp, NTOP=sel.n_top, QF=qf, num_warps=4)
    return out


def check_rows(iq, wt, pool, pos, Pb, n_top, kp, sel):
    """debug: dense reference selection (the stock torch path of step_graph.indexer_forward_static) vs the sparse one; counts rows whose group sets differ
    and the largest score gap between a swapped pair (a pure tie / rounding swap has gap ~ 1e-7)"""
    kq = torch.relu(torch.einsum("nhd,pd->nhp", iq, pool[:Pb].to(iq.dtype)))
    sc = (kq * wt[:, :, None]).sum(1)
    gend = torch.arange(Pb, device=iq.device) * kp + (kp - 1)
    vis = gend[None] < pos[:, None]
    sc = torch.where(vis, sc, torch.full_like(sc, -float("inf")))
    ref = torch.topk(sc, min(n_top, Pb), dim=1).indices
    for t in range(sc.shape[0]):
        a = set(ref[t].tolist()) - {g for g in ref[t].tolist() if not bool(vis[t, g])}
        b = set(sel.idx[t, :int(sel.nsel[t])].tolist())
        STATS["rows"] += 1
        if a != b:
            STATS["diff_rows"] += 1
            STATS["diff_groups"] += len(a ^ b)
            for g in a ^ b:
                STATS["max_gap"] = max(STATS["max_gap"], abs(float(sc[t, g]) - float(sc[t][list(b | a)].min())))
