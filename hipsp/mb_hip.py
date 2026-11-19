"""microbench: HIP sparse MLA vs Triton mla_sparse, 8192-row chunk, TF from the real key count. python mb_hip.py TREE [N,N,..]"""
import os, sys, types, time, json
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
Ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [8192, 33000, 131072]
def tm(f, reps=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.time() - t) / reps * 1000
torch.manual_seed(0)
qc8 = (torch.randn(H, 8192, R + Pd, device=dev) * 0.3).half()
sinks = torch.randn(H, device=dev)
# ceiling: fixed resident tile, 128 tiles (2048 keys) per query
kc0 = (torch.randn(8192, R + Pd, device=dev) * 0.3).half()
class S: pass
s0 = S(); s0.gidx = torch.zeros(8192, 512, dtype=torch.int32, device=dev); s0.nsel = torch.zeros(8192, dtype=torch.int32, device=dev); s0.tail = True; s0.kpool = 4
ms = tm(lambda: sp_hip.mla_sparse_hip(R, kc0, 0, scale, sinks, s0, qc8, fixed_tiles=128))
fl = 8192 * 2048 * H * (R + Pd + R) * 2
print(json.dumps({"CEIL_2048keys_ms": ms, "TF": fl / ms / 1e9}), flush=True)
for N in Ns:
    T = min(8192, N); qstart = N - T; P = N // KP
    sc = torch.randn(T, P, device=dev)
    pos = qstart + torch.arange(T, device=dev)
    gend = torch.arange(P, device=dev) * KP + KP - 1
    sc = sc.masked_fill(~(gend[None] < pos[:, None]), float("-inf"))
    sp = PK.ix_select(sc, qstart, N, NTOP, KP, True, sparse=True)
    nkeys = float((sp.nsel.double() * KP + (pos % KP + 1).double()).sum())
    qc = qc8[:, :T].contiguous()
    kc = (torch.randn(N, R + Pd, device=dev) * 0.3)
    kc16 = kc.half().contiguous()
    res = {"N": N, "T": T, "mean_keys": nkeys / T}
    res["hip_ms"] = tm(lambda: sp_hip.mla_sparse_hip(R, kc16, qstart, scale, sinks, sp, qc))
    res["hip_TF"] = nkeys * H * (R + Pd + R) * 2 / res["hip_ms"] / 1e9
    if not os.environ.get("MB_NOTRITON"):
        res["triton_ms"] = tm(lambda: PA.mla_sparse(R, kc, qstart, scale, sinks, sp, qc))
    print(json.dumps(res), flush=True)
    del sc, qc, kc, kc16, sp; torch.cuda.empty_cache()
print("MBDONE")
