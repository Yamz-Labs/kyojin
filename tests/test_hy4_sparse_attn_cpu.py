"""CPU test (TRITON_INTERPRET=1, no GPU) of the per-query sparse gather attention: python test_hy4_sparse_attn_cpu.py TREE
ix_select(sparse=True) -> PfSparse must expand to exactly the dense allow mask; mla_sparse must match an fp64 reference built from that mask (masked keys contribute 0, sink in the denominator only)."""
import os, sys, types
if __name__ != "__main__":
    # script-style test: it reads the tree from sys.argv[1] and swaps stub exllamav3 packages into sys.modules at import time, so under pytest it fails at
    # collection and breaks every test file collected after it. Run it as: python tests/<this file> TREE (Triton CPU interpreter, no GPU)
    import pytest
    pytest.skip("script-style test, run as: python tests/test_hy4_sparse_attn_cpu.py TREE", allow_module_level=True)
os.environ["TRITON_INTERPRET"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
tree = sys.argv[1]
for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    m = types.ModuleType(n); m.__path__ = [os.path.join(tree, "exllamav3", s)]; sys.modules[n] = m
from exllamav3.loader import prefill_kernels as PK, prefill_attn as PA

def ref64(qc, kc, scale, sinks, allow, R):
    H, T, D = qc.shape
    s = torch.einsum("htd,nd->htn", qc.double(), kc.double()) * scale
    s = s.masked_fill(~allow[None], float("-inf"))
    sk = sinks.double()[:, None, None].expand(H, T, 1)
    p = torch.softmax(torch.cat([s, sk], -1), -1)[..., :-1]
    return torch.einsum("htn,nr->htr", p, kc.double()[:, :R])

torch.manual_seed(0)
R, Pd, H, KP = 64, 16, 16, 4
bad = 0
for (qstart, T, n_top, tail) in ((0, 9, 6, 1), (0, 9, 6, 0), (37, 11, 6, 1), (37, 11, 6, 0), (50, 8, 3, 1), (4, 5, 20, 0), (61, 3, 5, 1), (62, 2, 5, 0), (127, 17, 8, 0)):
    N = qstart + T
    P = N // KP
    pos = torch.arange(qstart, N)
    sc = torch.randn(T, P)
    if P > 3: sc[:, :2] = sc[:, 2:3]     # ties: lowest group index must win in both paths
    gend = torch.arange(P) * KP + KP - 1
    vis = (gend[None] < pos[:, None]) if tail else (gend[None] <= pos[:, None])
    sc = sc.masked_fill(~vis, float("-inf"))
    dense = PK.ix_select(sc, qstart, N, n_top, KP, tail, want_bias=False)
    sp = PK.ix_select(sc, qstart, N, n_top, KP, tail, sparse=True)
    same = bool((sp.to_dense() == dense).all())
    qc = (torch.randn(H, T, R + Pd) * 0.5).half()
    kc = (torch.randn(N, R + Pd) * 0.5).half()
    sinks = torch.randn(H)
    scale = 0.125
    ref = ref64(qc, kc.float(), scale, sinks, dense, R)
    o = PA.mla_sparse(R, kc.float(), qstart, scale, sinks, sp, qc, cfg={"HB": 16, "BG": 4, "warps": 4, "stages": 1, "qres": int(os.environ.get("QRES", "0"))}).double()
    e = float((o - ref).norm() / ref.norm())
    ok = same and e < 3e-3 and bool(torch.isfinite(o).all())
    bad += not ok
    print(f"qstart={qstart} T={T} n_top={n_top} tail={tail}: mask_same={same} err_sparse={e:.2e} nsel_max={int(sp.nsel.max())} {'OK' if ok else 'FAIL'}", flush=True)
print("ALL OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
