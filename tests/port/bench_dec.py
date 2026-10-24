# Kill test: Kyojin's HIP decode GEMV (exl3_dec_gemv, compiled for CUDA through port.cuh) vs the
# upstream CUDA batch-1 path (BC_LinearEXL3 -> exl3_gemv) on the real linears of an EXL3 model.
# Error of both against an fp32 reference (same trellis decode, fp32 Hadamards and matmul); time per call
# from CUDA graphs (50 calls per replay, so no launch overhead).
# usage: EXL3_PORT_DEC=1 python -I bench_dec.py ENGINE_DIR MODEL_DIR OUT.json [MIN_BLOCKS,...]
import sys, os, json, time
eng, mdir, out = sys.argv[1:4]
MBS = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [32]
sys.path.insert(0, eng)
import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3, dec_workspace
assert hasattr(ext, "exl3_dec_gemv"), "build with EXL3_PORT_DEC=1 and import with EXL3_PORT_DEC=1"
torch.manual_seed(0)
cfg = Config.from_directory(mdir); model = Model.from_config(cfg); model.load()
dev = torch.device("cuda:0")

def had128():
    h = torch.ones(1, 1, dtype = torch.float, device = dev)
    while h.shape[0] < 128: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 128 ** 0.5
H = had128()
def had_rows(v):   # v [1, n]: H128 on every 128-block
    return (v.view(-1, 128) @ H).view(1, -1)

# one linear per distinct (in, out, K) that the dec kernel takes (mul1 codebook, K*2 integral, shapes)
lins = {}
for m in (sm for top in model.modules for sm in top):
    inner = getattr(m, "inner", None)
    if not isinstance(inner, LinearEXL3): continue
    if not inner.mul1 or inner.mcg or inner.suh is None or inner.svh is None: continue
    if inner.in_features % 512 or inner.out_features % 128 or float(inner.K * 2) != int(inner.K * 2): continue
    lins.setdefault((inner.in_features, inner.out_features, inner.K), (m.key, inner))
print(len(lins), "shapes", flush = True)
scratch, counters = dec_workspace(dev)

def gtime(fn, reps = 50, n = 5):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    g.replay(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(n):
        a = torch.cuda.Event(enable_timing = True); b = torch.cuda.Event(enable_timing = True)
        a.record(); g.replay(); b.record(); b.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / reps)
    return best

res = {"model": mdir, "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}, "rows": []}
for (kin, kout, K), (key, inner) in sorted(lins.items(), key = lambda t: t[0]):
    x = (torch.randn(1, kin, device = dev) * 0.5).half()
    w = inner.get_inner_weight_tensor().float()
    xr = had_rows(x.float() * inner.suh.float().view(1, -1)).half().float()
    ref = had_rows(xr @ w) * inner.svh.float().view(1, -1)
    y_up = inner.forward(x, {}).float()
    y_dec = torch.empty(1, kout, dtype = torch.half, device = dev)
    try:
        ext.exl3_dec_gemv(x, inner.trellis, inner.suh, inner.svh, y_dec, scratch, counters, float(K))
    except RuntimeError as e:   # K not in this build's EXL3_PORT_DEC_KB2_MASK
        print(json.dumps({"key": key, "K": K, "skip": str(e).splitlines()[0]}), flush = True); continue
    y_dec = y_dec.float()
    nr = ref.norm().item()
    r = {"key": key, "in": kin, "out": kout, "K": K,
         "relerr_up": (y_up - ref).norm().item() / nr, "relerr_dec": (y_dec - ref).norm().item() / nr,
         "maxdiff_dec_up": (y_dec - y_up).abs().max().item(), "ref_absmax": ref.abs().max().item()}
    yb = torch.empty(1, kout, dtype = torch.half, device = dev)
    r["us_up"] = gtime(lambda: inner.forward(x, {}))
    for mb in MBS:
        os.environ["EXL3_DEC_MIN_BLOCKS"] = str(mb)
        r[f"us_dec_mb{mb}"] = gtime(lambda: ext.exl3_dec_gemv(x, inner.trellis, inner.suh, inner.svh, yb, scratch, counters, float(K)))
    wbytes = inner.trellis.numel() * inner.trellis.element_size()
    r["GBs_up"] = wbytes / r["us_up"] / 1e3
    res["rows"].append(r); print(json.dumps(r), flush = True)
json.dump(res, open(out, "w"), indent = 1)
