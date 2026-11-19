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
    def _blk(q, qp, CK, KPE, offn, vis, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc):
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
    def _sparse_mla_kernel(QL, QP, CK, KPE, POS, IDX, NSEL, SINK, O, scale, s_ql_t, s_ql_h, s_qp_t, s_qp_h, s_ck, s_kp, s_o_t, s_o_h,
                           R: tl.constexpr, P: tl.constexpr, BR: tl.constexpr, BPE: tl.constexpr, BN: tl.constexpr, KG: tl.constexpr, NTOP: tl.constexpr):
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
        m, l, acc = _blk(q, qp, CK, KPE, offn, offn <= qpos, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc)
        n = tl.load(NSEL + t)
        nk = n * KG
        for i0 in range(0, nk, BN):
            kj = i0 + kk
            inb = kj < nk
            g = tl.load(IDX + t * NTOP + kj // KG, mask=inb, other=0)
            m, l, acc = _blk(q, qp, CK, KPE, g * KG + kj % KG, inb, s_ck, s_kp, offr, offp, mr, mp, scale, m, l, acc)
        tl.store(O + t * s_o_t + h * s_o_h + offr, acc / l, mask=mr)


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


def sparse_mla_attend(q_lat, q_pe, ckv, k_pe, qpos, sel, scale, sinks):
    """q_lat [T,H,R] fp32, q_pe [T,H,P], ckv [N,R], k_pe [N,P] (row strides free), qpos [T] long, sel SparseSel -> o_lat [T,H,R] fp32"""
    T, H, R = q_lat.shape
    P = k_pe.shape[1]
    assert sel.tail and q_lat.dtype == torch.float32 and ckv.dtype == torch.float32 and q_lat.stride(-1) == 1 and q_pe.stride(-1) == 1 and ckv.stride(-1) == 1 and k_pe.stride(-1) == 1
    out = torch.empty(T, H, R, dtype=torch.float32, device=q_lat.device)
    _sparse_mla_kernel[(T, H)](q_lat, q_pe, ckv, k_pe, qpos, sel.idx, sel.nsel, sinks.float().contiguous(), out, float(scale),
                               q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1), ckv.stride(0), k_pe.stride(0), out.stride(0), out.stride(1),
                               R=R, P=P, BR=triton.next_power_of_2(R), BPE=triton.next_power_of_2(P), BN=16, KG=sel.kp, NTOP=sel.n_top, num_warps=4)
    return out


def check_rows(iq, wt, pool, pos, Pb, n_top, kp, sel):
    """debug: dense reference selection (the stock torch path of step_graph.indexer_forward_static) vs the sparse one; counts rows whose group sets differ
    and the largest score gap between a swapped pair (a pure tie / rounding swap has gap ~ 1e-7)"""
    kq = torch.relu(torch.einsum("nhd,pd->nhp", iq, pool[:Pb]))
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
