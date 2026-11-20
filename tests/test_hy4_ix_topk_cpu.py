"""CPU test (TRITON_INTERPRET=1, no GPU) of the blocked fused score + linear-time top-k: python test_hy4_ix_topk_cpu.py TREE
_ix_topk_kernel must pick exactly the groups of the reference _ix_select_kernel (same ids in the same ascending order, same count) on the candidate path, the
sample-retry path and the exact fallback path (natt=0, heavy ties); ix_select_blocked must equal ix_scores_fused + ix_select(sparse=True)."""
import os, sys, types
if __name__ != "__main__":
    # script-style test: it reads the tree from sys.argv[1] and swaps stub exllamav3 packages into sys.modules at import time, so under pytest it fails at
    # collection and breaks every test file collected after it. Run it as: python tests/<this file> TREE (Triton CPU interpreter, no GPU)
    import pytest
    pytest.skip("script-style test, run as: python tests/test_hy4_ix_topk_cpu.py TREE", allow_module_level=True)
os.environ["TRITON_INTERPRET"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
tree = sys.argv[1]
for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    m = types.ModuleType(n); m.__path__ = [os.path.join(tree, "exllamav3", s)]; sys.modules[n] = m
from exllamav3.loader import prefill_kernels as PK

torch.manual_seed(1)
bad = 0


def run_topk(sc, start, n_top, KP, tail, **kw):
    T, P = sc.shape
    cap = kw.get("cap", 32)
    ck = torch.empty(T, cap, dtype=torch.int32); ci = torch.empty(T, cap, dtype=torch.int32)
    gidx = torch.full((T, n_top), -1, dtype=torch.int32); nsel = torch.empty(T, dtype=torch.int32)
    PK._ix_topk_kernel[(T,)](sc, ck, ci, gidx, nsel, P, start, n_top, KP=KP, TAIL=int(tail), BLK=kw.get("blk", 16), SM=kw.get("sm", 64), CAP=cap,
                             TARGET=kw.get("target", 16), NATT=kw.get("natt", 3), num_warps=4)
    return gidx, nsel


def check(name, sc, start, n_top, KP, tail, N, **kw):
    global bad
    ref = PK.ix_select(sc, start, N, n_top, KP, tail, sparse=True)
    gidx, nsel = run_topk(sc, start, n_top, KP, tail, **kw)
    same_n = bool((nsel == ref.nsel).all())
    same_g = all(bool((gidx[r, :int(nsel[r])] == ref.gidx[r, :int(ref.nsel[r])]).all()) for r in range(sc.shape[0])) if same_n else False
    ok = same_n and same_g
    bad += not ok
    print(("OK  " if ok else "FAIL"), name, "nsel", int(nsel.min()), int(nsel.max()), flush=True)


for tail in (1, 0):
    for (start, T, KP, n_top) in ((0, 5, 4, 6), (200, 6, 4, 6), (3, 40, 4, 9), (260, 7, 4, 12)):
        N = start + T
        P = N // KP
        pos = torch.arange(start, N)
        gend = torch.arange(P) * KP + KP - 1
        vis = (gend[None] < pos[:, None]) if tail else (gend[None] <= pos[:, None])
        for kind in ("rand", "ties", "allsame", "clustered"):
            if kind == "rand": sc = torch.randn(T, P)
            elif kind == "ties": sc = torch.randint(0, 4, (T, P)).float()
            elif kind == "allsame": sc = torch.zeros(T, P)
            else: sc = torch.randn(T, P) * 0.001 + (torch.arange(P) % 7 == 0).float() * 5
            sc = sc.masked_fill(~vis, float("-inf"))
            for kw in (dict(), dict(natt=0), dict(cap=16, target=8), dict(cap=64, sm=16, blk=8)):
                check(f"tail{tail} start{start} T{T} {kind} {kw}", sc, start, n_top, KP, tail, N, **kw)

# blocked scoring + select vs the unblocked reference
H, D, KP, n_top = 4, 16, 4, 7
for tail in (1, 0):
    start, T = 90, 23
    N = start + T
    P = N // KP
    iq = (torch.randn(T, H, D) * 0.5).half(); pk = (torch.randn(P, D) * 0.5).half(); wt = torch.rand(T, H)
    sc = PK.ix_scores_fused(iq, pk, wt, start, KP, tail)
    ref = PK.ix_select(sc, start, N, n_top, KP, tail, sparse=True)
    for rows in (7, 23, 64):
        got = PK.ix_select_blocked(iq, pk, wt, start, KP, tail, n_top, N, rows=rows, cap=32, blk=16, sm=64, target=16)
        same = bool((got.nsel == ref.nsel).all()) and bool((got.to_dense() == ref.to_dense()).all())
        bad += not same
        print(("OK  " if same else "FAIL"), "blocked tail", tail, "rows", rows, flush=True)
print("ALL OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
