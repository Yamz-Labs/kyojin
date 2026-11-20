"""Unit tests of exllamav3/loader/lc_sparse.py (exact group selection + sparse MLA decode attend), with POISONED guard regions around every tensor so that any
out-of-bounds read (NaN reaches the output) or write (guard changed) fails the test BEFORE a model run.
CPU: TRITON_INTERPRET=1 DEV=cpu python tests/test_lc_sparse.py     GPU: DEV=cuda python tests/test_lc_sparse.py"""
import importlib.util
import math
import os
import sys

import torch

DEV = os.environ.get("DEV", "cpu")
here = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("lc_sparse", os.path.join(here, "..", "exllamav3", "loader", "lc_sparse.py"))
LS = importlib.util.module_from_spec(sp)
sp.loader.exec_module(LS)
GUARD = 4096


def guarded(t):
    """copy of t placed inside a NaN / -1 poisoned buffer; returns (view, check()) where check() asserts the guards are untouched"""
    flat = t.reshape(-1)
    if t.dtype.is_floating_point:
        buf = torch.full((flat.numel() + 2 * GUARD,), float("nan"), dtype=t.dtype, device=t.device)
    else:
        buf = torch.full((flat.numel() + 2 * GUARD,), -7, dtype=t.dtype, device=t.device)
    buf[GUARD:GUARD + flat.numel()] = flat
    ref = buf.clone()
    view = buf[GUARD:GUARD + flat.numel()].view(t.shape)

    def check():
        a = torch.cat([buf[:GUARD], buf[GUARD + flat.numel():]])
        b = torch.cat([ref[:GUARD], ref[GUARD + flat.numel():]])
        assert torch.equal(torch.nan_to_num(a, nan=-12345.0), torch.nan_to_num(b, nan=-12345.0)), "guard region written"
    return view, check


def ref_scores(iq, wt, pool, pos, Pb, kp, tail):
    kq = torch.relu(torch.einsum("nhd,pd->nhp", iq, pool[:Pb]))
    sc = (kq * wt[:, :, None]).sum(1)
    gend = torch.arange(Pb, device=iq.device) * kp + (kp - 1)
    vis = (gend[None] < pos[:, None]) if tail else (gend[None] <= pos[:, None])
    return torch.where(vis, sc, torch.full_like(sc, -math.inf)), vis


def case_select(T, H, D, Pb, kp, n_top, pos_list, ties=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    iq = torch.randn(T, H, D, generator=g).to(DEV)
    wt = torch.randn(T, H, generator=g).to(DEV) * 0.1
    pool = torch.randn(Pb + 5, D, generator=g).to(DEV)
    if ties:
        pool[10:40] = pool[3]
        pool[200:230] = pool[3]
    pos = torch.tensor(pos_list, dtype=torch.long, device=DEV)
    iq_g, c1 = guarded(iq); wt_g, c2 = guarded(wt); pool_g, c3 = guarded(pool); pos_g, c4 = guarded(pos)
    idx, nsel = LS.select_groups(iq_g, wt_g, pool_g, pos_g, Pb, n_top, kp, True)
    for c in (c1, c2, c3, c4):
        c()
    sc, vis = ref_scores(iq, wt, pool, pos, Pb, kp, True)
    for t in range(T):
        n = int(nsel[t])
        got = idx[t, :n].tolist()
        nv = int(vis[t].sum())
        assert n == min(n_top, nv), (t, n, nv)
        assert got == sorted(got) and len(set(got)) == n and all(0 <= x < Pb for x in got)
        assert all(bool(vis[t, x]) for x in got)
        if nv <= n_top:
            assert got == torch.nonzero(vis[t]).flatten().tolist()
        else:
            # exact: every selected score >= every unselected visible score; ties at the boundary go to the lowest ids
            s = sc[t]
            selm = torch.zeros(Pb, dtype=torch.bool, device=DEV)
            selm[torch.tensor(got, device=DEV)] = True
            smin = s[selm].min()
            smax_out = s[vis[t] & ~selm].max()
            assert smin >= smax_out, (t, float(smin), float(smax_out))
            if smin == smax_out:
                tie = torch.nonzero(vis[t] & (s == smin)).flatten().tolist()
                chosen = [x for x in tie if selm[x]]
                assert chosen == tie[:len(chosen)], "ties not resolved to the lowest ids"
            # same set as torch.topk when no ties at the boundary
            if smin > smax_out:
                tk = set(torch.topk(s, n_top).indices.tolist())
                assert tk == set(got)
    return idx, nsel, iq, pool


def case_attend(T, H, R, P, Pb, kp, n_top, pos_list, seed=1, N=None):
    idx, nsel, iq, pool = None, None, None, None
    D = 128
    g = torch.Generator().manual_seed(seed)
    iq = torch.randn(T, 16, D, generator=g).to(DEV)
    wt = torch.randn(T, 16, generator=g).to(DEV) * 0.1
    pool = torch.randn(Pb + 3, D, generator=g).to(DEV)
    pos = torch.tensor(pos_list, dtype=torch.long, device=DEV)
    idx, nsel = LS.select_groups(iq, wt, pool, pos, Pb, n_top, kp, True)
    N = N or (max(pos_list) + 1)
    ckv = torch.randn(N, R, generator=g).to(DEV)
    kpe = torch.randn(N, P, generator=g).to(DEV)
    ql = torch.randn(T, H, R, generator=g).to(DEV)
    qp = torch.randn(T, H, P, generator=g).to(DEV)
    sinks = torch.randn(H, generator=g).to(DEV)
    scale = 1.0 / 16.0
    # dense reference mask
    allow = torch.zeros(T, N, dtype=torch.bool, device=DEV)
    for t in range(T):
        for gi in idx[t, :int(nsel[t])].tolist():
            allow[t, gi * kp:gi * kp + kp] = True
        p = pos_list[t]
        allow[t, p - p % kp:p + 1] = True
    kpos = torch.arange(N, device=DEV)
    s = torch.einsum("thr,nr->htn", ql, ckv) + torch.einsum("thp,np->htn", qp, kpe)
    s = s * scale
    vis = (kpos[None, :] <= pos[:, None]) & allow
    s = s.masked_fill(~vis[None], float("-inf"))
    s = torch.cat([s, sinks[:, None, None].expand(-1, T, 1)], -1)
    pr = torch.softmax(s, -1)[..., :-1]
    want = torch.einsum("htn,nr->thr", pr, ckv)
    ql_g, c1 = guarded(ql); qp_g, c2 = guarded(qp); ckv_g, c3 = guarded(ckv); kpe_g, c4 = guarded(kpe); pos_g, c5 = guarded(pos)
    idx_g, c6 = guarded(idx); ns_g, c7 = guarded(nsel); sk_g, c8 = guarded(sinks)
    sel = LS.SparseSel(idx_g, ns_g, kp, True)
    got = LS.sparse_mla_attend(ql_g, qp_g, ckv_g, kpe_g, pos_g, sel, scale, sk_g)
    for c in (c1, c2, c3, c4, c5, c6, c7, c8):
        c()
    assert torch.isfinite(got).all()
    err = (got - want).abs().max().item()
    assert err < 2e-5, err
    return err


if __name__ == "__main__":
    # selection: small / large visible counts, ties, short rows (fewer visible groups than n_top), several rows in one call
    case_select(2, 16, 128, 700, 4, 64, [2800 - 1, 1000])
    case_select(3, 16, 128, 512, 4, 511, [2047, 2100, 2051 + 4], seed=2)
    case_select(2, 16, 128, 900, 4, 128, [3000, 3003], ties=True, seed=3)
    case_select(1, 16, 128, 1024, 4, 511, [4090], seed=4)
    print("select ok")
    e1 = case_attend(2, 4, 64, 16, 300, 4, 32, [1199, 1204])
    e2 = case_attend(3, 8, 512, 64, 640, 4, 511, [2600, 2601, 2602], N=2700)
    print("attend ok", e1, e2)
    if DEV == "cuda":          # production sizes: T = 8 verify rows, 64K / 128K context pools, 32 heads, R = 512, P = 64
        case_select(8, 16, 128, 16384, 4, 511, [65000 + 3 * i for i in range(8)], seed=5)
        case_select(8, 16, 128, 32768, 4, 511, [131000 + i for i in range(8)], seed=6)
        case_select(4, 16, 128, 16384, 4, 511, [5000, 40000, 65500, 2300], ties=True, seed=7)
        e3 = case_attend(8, 32, 512, 64, 16384, 4, 511, [65000 + 3 * i for i in range(8)], N=65536)
        e4 = case_attend(1, 32, 512, 64, 32768, 4, 511, [131070], N=131072)
        print("production sizes ok", e3, e4)
    print("ALL OK")
