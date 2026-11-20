"""dend_test.py: gm_dense_q8 vs fq_gemv_q8_0 (stock): bit equality at M 1..16 and ms per call at the real shapes (graph replay over distinct weights)."""
import os, sys, time, json, types
tree = os.environ["LC_T"]
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import torch
import engine as _eng
_eng._stub_package(tree)
from exllamav3.loader import hy4_forward as HF
from exllamav3 import gm_ext
e, g = HF._ext(), gm_ext.get()
print("ext", e.__file__ if hasattr(e, "__file__") else e, "gm", g.__file__, flush=True)
dev = "cuda"
shapes = [(1024, 3072), (576, 3072), (1536, 3072), (3072, 1536), (8192, 1024), (16384, 1024), (3072, 8192), (3072, 16384), (13824, 3072), (3072, 13824), (16384, 3072), (1001, 1056), (77, 96), (2050, 5120)]
only = os.environ.get("SHAPES")
if only:
    shapes = [shapes[int(i)] for i in only.split(",")]
cpbs = [int(c) for c in os.environ.get("CPBS", "4").split(",")]
Ts = [int(t) for t in os.environ.get("TS", "1,2,3,4,6,8,16").split(",")]
bad = 0
def mk(N, K, cp):
    w = torch.randint(-127, 128, (N, K // 32, 34), dtype=torch.int16, device=dev).to(torch.int8).view(torch.uint8)
    d = (torch.rand(N, K // 32, device=dev) * 0.02 + 0.001).half().view(torch.uint8).reshape(N, K // 32, 2)
    w[:, :, :2] = d
    return w
def run_stock(x, w, out):
    e.fq_gemv_q8_0(x, w, out)
def run_v2(x, w, out, cpb):
    R, K = x.shape
    qb = e.fq_q8_scratch(0, R, K); e.fq_quant_q8_1_rows(x, qb)
    assert g.gm_dense_q8(qb, w, out, cpb)
def run_f(x, w, out, cpb):
    assert g.gm_dense_q8_x(x, w, out, cpb)
def timeit(fns):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for f in fns: f()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for f in fns: f()
    for _ in range(3): gr.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(40):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); gr.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2] / len(fns) * 1000   # us per call
res = []
for (N, K) in shapes:
    nb = N * (K // 32) * 34
    copies = max(4, (96 << 20) // nb + 1)
    ws = [mk(N, K, i) for i in range(copies)]
    for T in Ts:
        x = torch.randn(T, K, device=dev)
        o1 = torch.empty(T, N, device=dev); o2 = torch.empty(T, N, device=dev)
        run_stock(x, ws[0], o1)
        row = dict(N=N, K=K, T=T)
        outs = {i: torch.empty(T, N, device=dev) for i in range(copies)}
        t0 = timeit([(lambda w=w, o=outs[i]: run_stock(x, w, o)) for i, w in enumerate(ws)])
        qbs = [e.fq_q8_scratch(0, T, K)] * copies
        tq = timeit([(lambda w=w: e.fq_quant_q8_1_rows(x, qbs[0])) for i, w in enumerate(ws)])
        row["quant_us"] = round(tq, 2)
        row["stock_us"] = round(t0, 2); row["stock_gbs"] = round(nb / (t0 * 1e-6) / 1e9, 1)
        for cpb in cpbs:
            for tag, fn in ((("v", run_v2),) if os.environ.get("NOF") else (("v", run_v2), ("f", run_f))):
                fn(x, ws[0], o2, cpb); torch.cuda.synchronize()
                eq = bool(torch.equal(o1, o2))
                if not eq: bad += 1; row[f"NE_{tag}{cpb}"] = 1
                t1 = timeit([(lambda w=w, o=outs[i]: fn(x, w, o, cpb)) for i, w in enumerate(ws)])
                row[f"{tag}{cpb}"] = round(t1, 1)
        print(json.dumps(row), flush=True)
        res.append(row)
    del ws
    torch.cuda.empty_cache()
print("BAD", bad, flush=True)
