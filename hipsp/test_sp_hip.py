"""GPU correctness: HIP sparse MLA vs fp64 dense-mask reference and vs the Triton mla_sparse. python test_sp_hip.py TREE"""
import os, sys, types, time
import torch
tree = sys.argv[1]
for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    m = types.ModuleType(n); m.__path__ = [os.path.join(tree, "exllamav3", s)]; sys.modules[n] = m
from exllamav3.loader import prefill_kernels as PK, prefill_attn as PA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp_hip
dev = "cuda"
H, R, Pd, KP, NTOP = 64, 512, 64, 4, 511
scale = 1 / 16.0
def make(N, T, tail, ntop, seed, sc_scale=1.0, sink=True):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    qstart = N - T; P = (N + KP - 1) // KP
    sc = torch.randn(T, P, device=dev, generator=g) * sc_scale
    pos = qstart + torch.arange(T, device=dev)
    gend = torch.arange(P, device=dev) * KP + KP - 1
    sc = sc.masked_fill(~(gend[None] < pos[:, None]), float("-inf"))
    sp = PK.ix_select(sc, qstart, N, ntop, KP, tail, want_bias=False, sparse=True)
    qc = (torch.randn(H, T, R + Pd, device=dev, generator=g) * 0.3).half()
    kc = torch.randn(N, R + Pd, device=dev, generator=g) * 0.3
    sinks = torch.randn(H, device=dev, generator=g) if sink else None
    return qstart, sp, qc, kc, sinks
def ref64(qstart, sp, qc, kc, sinks):
    allow = sp.to_dense()
    H_, T = qc.shape[:2]
    o = torch.empty(H_, T, R, dtype=torch.float64, device=dev)
    k64 = kc.half().double()
    for h in range(H_):
        s = (qc[h].double() @ k64.T) * scale
        s = s.masked_fill(~allow, float("-inf"))
        mx = s.max(1, keepdim=True).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        if sinks is not None: mx = torch.maximum(mx, sinks[h].double().expand_as(mx))
        p = torch.exp(s - mx)
        den = p.sum(1, keepdim=True) + (torch.exp(sinks[h].double() - mx) if sinks is not None else 0)
        o[h] = (p @ k64[:, :R]) / den.clamp_min(1e-300)
    return o
ok = True
cases = [(3000, 256, True, 511), (3000, 256, False, 511), (2999, 129, True, 511), (2998, 64, False, 40), (41, 41, True, 511), (41, 41, False, 511), (7, 7, True, 511), (1, 1, False, 511), (130, 64, True, 3),
         (33001, 48, True, 511), (33000, 48, False, 511), (4100, 32, True, 5)]
for (N, T, tail, ntop) in cases:
    for sink in (True, False):
        qstart, sp, qc, kc, sinks = make(N, T, tail, ntop, N + T, sink=sink)
        kc16 = kc.clamp(-65504.0, 65504.0).half().contiguous()
        o = sp_hip.mla_sparse_hip(R, kc16, qstart, scale, sinks, sp, qc).double()
        torch.cuda.synchronize()
        ref = ref64(qstart, sp, qc, kc, sinks)
        err = float((o - ref).norm() / ref.norm().clamp_min(1e-30))
        mx = float((o - ref).abs().max())
        try:
            tr = PA.mla_sparse(R, kc, qstart, scale, sinks, sp, qc).double(); etr = float((o - tr).norm() / tr.norm().clamp_min(1e-30)); etr0 = float((tr - ref).norm() / ref.norm().clamp_min(1e-30))
        except Exception as e:
            etr = etr0 = -1
        good = err < 1.5e-3 and bool(torch.isfinite(o).all())
        ok &= good
        print(f"N {N} T {T} tail {int(tail)} ntop {ntop} sink {int(sink)} nsel mean {float(sp.nsel.float().mean()):.0f}: err vs fp64 {err:.2e} (max {mx:.1e}) | triton vs fp64 {etr0:.2e} | hip vs triton {etr:.2e} {'OK' if good else 'FAIL'}", flush=True)
print("ALL OK" if ok else "FAILURES")
