"""head_dot_test.py: HY4_HEAD_DOT kernel vs the shipped fp32 row-batched kernel + fp64 torch reference; error always, time when on GPU (DEV=cuda). Loads head_bf16.py by path."""
import os, sys, statistics, importlib.util
import torch
tree = os.getcwd()
def mod(name, rel):
    sp = importlib.util.spec_from_file_location(name, os.path.join(tree, "exllamav3", "loader", rel)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
hb = mod("head_bf16", "head_bf16.py")
dev = os.environ.get("DEV", "cuda"); CPU = dev == "cpu"
V, D = (700, 256) if CPU else (129024, 4096)
torch.manual_seed(0)
W = (torch.randn(V, D, device=dev) * 0.05).to(torch.bfloat16)
hb.ENABLED = True
def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
for T in (2, 3, 6, 8):
    h = torch.randn(T, D, device=dev)
    ref = (h.double() @ W.double().T).float()
    hb.DOT = False; a = hb.gemv(h, W)
    for bv, bd, nw in (((16, 64, 4),) if CPU else ((64, 128, 4), (64, 64, 4), (128, 64, 8), (32, 128, 4), (128, 128, 8))):
        hb.DOT, hb.DOT_BV, hb.DOT_BD, hb.DOT_NW = True, bv, bd, nw
        try:
            b = hb.gemv(h, W)
        except Exception as e:
            print(f"T {T} BV {bv} BD {bd} NW {nw}: FAIL {type(e).__name__} {str(e)[:80]}", flush=True); continue
        ea = ((a - ref).norm() / ref.norm()).item(); eb = ((b - ref).norm() / ref.norm()).item()
        if CPU: print(f"T {T}: rel err old {ea:.1e} dot {eb:.1e}", flush=True)
        else:
            hb.DOT = False; ta = bench(lambda: hb.gemv(h, W)); hb.DOT = True; tb = bench(lambda: hb.gemv(h, W))
            print(f"T {T} BV {bv} BD {bd} NW {nw}: rel err old {ea:.1e} dot {eb:.1e}; old {ta:.0f} us dot {tb:.0f} us", flush=True)
print("HEAD_DOT_DONE", flush=True)
