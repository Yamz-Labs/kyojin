"""CPU test (TRITON_INTERPRET=1, no GPU): python test_prime_sparse_cpu.py TREE
Hy4MLA._sparse_o (draft-cache fill above the blocked-selection threshold) must equal the dense masked attention over PfSparse.to_dense(), also when the K buffer holds
rows beyond start + T (the draft head passes N rounded up to a multiple of 4: those rows are garbage / zeros and must never be read)."""
import os, sys, types
os.environ["TRITON_INTERPRET"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HY4_PF_SPARSE_HIP"] = "0"
import torch
tree = sys.argv[1]
for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    m = types.ModuleType(n); m.__path__ = [os.path.join(tree, "exllamav3", s)]; sys.modules[n] = m
from exllamav3.loader import prefill_kernels as PK, mla_hy4 as MLA

def ref64(qc, kc, scale, sinks, allow, R):
    s = torch.einsum("htd,nd->htn", qc.double(), kc.double()) * scale
    s = s.masked_fill(~allow[None], float("-inf"))
    sk = sinks.double()[:, None, None].expand(*qc.shape[:2], 1)
    p = torch.softmax(torch.cat([s, sk], -1), -1)[..., :-1]
    return torch.einsum("htn,nr->htr", p, kc.double()[:, :R])

torch.manual_seed(0)
R, Pd, H, KP = 64, 16, 16, 4
bad = 0
for (start, T, n_top, tail, extra) in ((37, 11, 6, 1, 3), (40, 8, 5, 0, 0), (61, 3, 5, 1, 2), (96, 17, 8, 0, 1)):
    end = start + T
    P = end // KP
    pos = torch.arange(start, end)
    sc = torch.randn(T, P)
    gend = torch.arange(P) * KP + KP - 1
    vis = (gend[None] < pos[:, None]) if tail else (gend[None] <= pos[:, None])
    sc = sc.masked_fill(~vis, float("-inf"))
    sp = PK.ix_select(sc, start, end, n_top, KP, tail, sparse=True)
    qlat = torch.randn(T, H, R) * 0.5
    qpe = torch.randn(T, H, Pd) * 0.5
    kc = torch.randn(end + extra, R + Pd) * 0.5
    kc[end:] = 1e4                       # rows nobody may read
    sinks = torch.randn(H)
    ns = types.SimpleNamespace(R=R, scale=0.125, sinks=sinks, tm=None)
    o = MLA.Hy4MLA._sparse_o(ns, qlat, qpe, kc, T, sp).double()
    qc = torch.cat([qlat, qpe], -1).half().float().transpose(0, 1)
    ref = ref64(qc, kc[:end].half().float(), 0.125, sinks, sp.to_dense(), R).transpose(0, 1)
    e = float((o - ref).norm() / ref.norm())
    ok = e < 3e-3 and bool(torch.isfinite(o).all()) and o.shape == (T, H, R)
    bad += not ok
    print(f"start={start} T={T} tail={tail} extra={extra}: err={e:.2e} {"OK" if ok else "FAIL"}", flush=True)
print("ALL OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
