"""Multi-session (slot) variants of the three decode ops that touch per-session state: MLA core over the K cache, k-pool indexer update + scoring,
K-row write. Everything else of the Hy4 step is row-wise, so rows of different sessions share one batched step.

Layout: every per-session buffer is a SLAB of a flat tensor: K `[S * cap, W]`, indexer value|gate rows `[S * cap, 2d]`, pooled keys `[S * cap / k, d]`.
A row r carries `base[r] = slot * cap` (device long) and its position `pos[r]`; the flat row of its token is `base + pos`. Session s's own view is
`flat[s * cap:(s + 1) * cap]` (a contiguous view of the same storage), so the solo code paths run on a session unchanged.
Generic by shape: nothing is keyed on a model name. No kernel here reads a row outside [base, base + N) with N <= cap (bounds are tested in
tests/test_hy4_mu_slots.py with NaN canaries around every slab)."""
from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:            # noqa: BLE001
    triton = None

MAX_T = 16


def _usable(*ts):
    if triton is None:
        return False
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return True
    return all(t.is_cuda for t in ts)


# ---------------------------------------------------------------------------------------------------- MLA core over per-row K slabs
def mla_slots_ref(q_lat, q_pe, kflat, qpos, base, N, scale, sinks, window, allow):
    """torch reference: row t attends keys base[t] + 0 .. N - 1 of the flat K buffer (causal on its own position, optional window / allow / sink)"""
    R = q_lat.shape[-1]
    T = q_lat.shape[0]
    out = torch.empty(q_lat.shape, dtype=torch.float32, device=q_lat.device)
    for t in range(T):
        n = min(N, int(qpos[t]) + 1)                          # keys past the row's own position are never read (like the kernel)
        b = int(base[t])
        ck = kflat[b:b + n]
        ar = torch.arange(n, device=kflat.device)
        s = torch.einsum("hr,nr->hn", q_lat[t], ck[:, :R]) + torch.einsum("hp,np->hn", q_pe[t], ck[:, R:])
        s = s * scale
        vis = torch.ones(n, dtype=torch.bool, device=kflat.device)
        if window:
            vis = vis & ((qpos[t] - ar) < window)
        if allow is not None:
            vis = vis & allow[t, :n]
        s = s.masked_fill(~vis[None], float("-inf"))
        if sinks is not None:
            s = torch.cat([s, sinks.float()[:, None]], dim=-1)
        p = torch.softmax(s, dim=-1)
        if sinks is not None:
            p = p[:, :-1]
        out[t] = torch.einsum("hn,nr->hr", p, ck[:, :R])
    return out


if triton is not None:
    @triton.jit
    def _mla_slots_kernel(QL, QP, CK, QPOS, KB, ALLOW, SINK, O, N, scale, window,
                          s_ql_t, s_ql_h, s_qp_t, s_qp_h, s_ck, s_al, s_o_t, s_o_h,
                          R: tl.constexpr, P: tl.constexpr, BR: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr,
                          HAS_ALLOW: tl.constexpr, HAS_WIN: tl.constexpr):
        t = tl.program_id(0)
        h = tl.program_id(1)
        offr = tl.arange(0, BR)
        offp = tl.arange(0, BP)
        mr = offr < R
        mp = offp < P
        q = tl.load(QL + t * s_ql_t + h * s_ql_h + offr, mask=mr, other=0.0).to(tl.float32)
        qp = tl.load(QP + t * s_qp_t + h * s_qp_h + offp, mask=mp, other=0.0).to(tl.float32)
        qpos = tl.load(QPOS + t)
        kb = tl.load(KB + t)
        m = tl.load(SINK + h).to(tl.float32)
        l = tl.zeros([], dtype=tl.float32) + 1.0
        acc = tl.zeros([BR], dtype=tl.float32)
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
            if HAS_ALLOW:
                vis = vis & (tl.load(ALLOW + t * s_al + offn, mask=inb, other=0) != 0)
            rows = (kb + offn) * s_ck
            ck = tl.load(CK + rows[:, None] + offr[None, :], mask=inb[:, None] & mr[None, :], other=0.0).to(tl.float32)
            kp = tl.load(CK + rows[:, None] + R + offp[None, :], mask=inb[:, None] & mp[None, :], other=0.0).to(tl.float32)
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


def mla_attend_slots(q_lat, q_pe, kflat, qpos, base, N, scale, sinks, window=0, allow=None):
    """q_lat [T, H, R], q_pe [T, H, P] fp32, kflat [S*cap, R+P] fp32 (flat K slabs), qpos [T] long, base [T] long (slot * cap), N rows scanned per row
    (bucket, <= cap; rows past qpos are never read), sinks [H], allow [T, >= N] bool or None -> o_lat [T, H, R] fp32.
    Same online-softmax maths as ref_fused._mla_kernel (sink = a key with logit s and a zero value)."""
    T, H, R = q_lat.shape
    P = q_pe.shape[-1]
    if (not _usable(q_lat, kflat) or T > MAX_T or sinks is None or q_lat.dtype != torch.float32 or kflat.dtype != torch.float32
            or q_lat.stride(-1) != 1 or q_pe.stride(-1) != 1 or kflat.stride(-1) != 1):
        return mla_slots_ref(q_lat, q_pe, kflat, qpos, base, N, scale, sinks, window, allow)
    out = torch.empty(T, H, R, dtype=torch.float32, device=q_lat.device)
    al = allow.view(torch.uint8) if allow is not None else qpos
    _mla_slots_kernel[(T, H)](q_lat, q_pe, kflat, qpos, base, al, sinks.float().contiguous(), out, int(N), float(scale), int(window),
                              q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1), kflat.stride(0),
                              allow.stride(0) if allow is not None else 0, out.stride(0), out.stride(1),
                              R=R, P=P, BR=triton.next_power_of_2(R), BP=triton.next_power_of_2(P), BN=16,
                              HAS_ALLOW=allow is not None, HAS_WIN=bool(window), num_warps=4)
    return out


# ---------------------------------------------------------------------------------------------------- indexer over slabs
def _rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class IxSlabs:
    """flat state of one indexer layer for S sessions: vgb [S*cap, 2d] (per-position value | gate rows), pool [S*cap/k, d]; `view(s)` = the solo buffers of session s.
    pool_update's Triton kernel indexes the rope table by group start: the table is tiled S times so the FLAT group start is a valid index."""

    def __init__(self, ix, S, cap):
        k, d = ix.k, ix.d
        assert cap % k == 0
        self.S, self.cap, self.k, self.d = S, cap, k, d
        self.vgb = torch.zeros(S * cap, 2 * d, device=ix.dev)
        self.pool = torch.zeros(S * cap // k, d, device=ix.dev)
        self.tiles = None

    def view(self, s):
        c, k = self.cap, self.k
        return self.pool[s * c // k:(s + 1) * c // k], self.vgb[s * c:(s + 1) * c]

    def tiled_table(self, ix):
        if self.tiles is None:
            cs, sn = ix._table(self.cap)
            self.tiles = (cs[:self.cap].repeat(self.S, 1).contiguous(), sn[:self.cap].repeat(self.S, 1).contiguous())
        return self.tiles


@torch.no_grad()
def indexer_forward_slots(ix, sl, x, qr, pos, base, Nb, ar, poolk=False):
    """indexer_forward_static for rows of different sessions. x [T, E], qr [T, R], pos [T] long = position inside the row's own session, base [T] long = slot * cap.
    Each row writes its value | gate row at base + pos, updates the pooled group of its position, then (when Nb is past the all-causal bound) scores the pooled keys of ITS
    OWN slab and selects its top groups: the maths of one solo step per row. -> allow [T, Nb] bool or None."""
    k, d, H = ix.k, ix.d, ix.H
    ik = _rms(x @ ix.w["attn_k"].T, ix.w["k_norm"], ix.eps)
    vg_new = torch.cat([ik @ ix.w["value"].T, ik @ ix.w["gate"].T], -1)
    flat = base + pos
    sl.vgb.index_copy_(0, flat, vg_new)
    if poolk:                                              # the one Triton kernel on flat positions; rope table tiled per slab
        from . import pool_fused as _pf
        cs, sn = sl.tiled_table(ix)
        _pf._pool_kernel[(pos.shape[0],)](sl.vgb, flat, ix.w["ape"], ix.w["pool_norm"], cs, sn, sl.pool, ix.eps, k=k, d=d, rd=ix.rope_dim)
    else:
        g0 = (flat // k) * k
        rows = sl.vgb[g0[:, None] + ar[:k][None, :]]
        v, g = rows[..., :d].reshape(-1, k, d), rows[..., d:].reshape(-1, k, d)
        p = (torch.softmax(g + ix.w["ape"][None], dim=1) * v).sum(1)
        pooled = ix._rope(_rms(p, ix.w["pool_norm"], ix.eps), (pos // k) * k)
        sl.pool.index_copy_(0, flat // k, pooled)
    if Nb <= (ix.n_top + 1) * k:
        return None
    T = x.shape[0]
    iq = ix._rope((qr @ ix.w["q_b"].T).reshape(T, H, d), pos)
    wt = (x.float() @ ix.w["proj"].T) / math.sqrt(d * H)
    Pb = Nb // k
    pidx = (base // k)[:, None] + ar[:Pb][None, :]
    pg = sl.pool[pidx]                                              # [T, Pb, d] pooled keys of each row's own session
    kq = torch.relu(torch.einsum("nhd,npd->nhp", iq, pg))
    sc = (kq * wt[:, :, None]).sum(1)
    gend = ar[:Pb] * k + (k - 1)
    vis = (gend[None] < pos[:, None]) if ix.tail else (gend[None] <= pos[:, None])
    sc = torch.where(vis, sc, torch.full_like(sc, -math.inf))
    idx = torch.topk(sc, min(ix.n_top, Pb), dim=1).indices
    gsel = torch.zeros(T, Pb, dtype=torch.bool, device=ix.dev)
    gsel.scatter_(1, idx, True)
    gsel &= vis
    allow = torch.zeros(T, Nb, dtype=torch.bool, device=ix.dev)
    allow[:, :Pb * k] = gsel.repeat_interleave(k, dim=1)
    lo = (pos - (pos % k)) if ix.tail else (pos - ((pos + 1) % k) + 1)
    j = ar[:Nb][None]
    allow |= (j >= lo[:, None]) & (j <= pos[:, None])
    return allow
