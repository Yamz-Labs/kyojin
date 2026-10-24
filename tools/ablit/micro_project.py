#!/usr/bin/env python3
"""Cost of the runtime projection per decode round: 43 layers x (attn fp32 + mlp fp16 site), R rows,
replayed as one HIP graph. usage: micro_project.py"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3.modules import ablit_runtime as A
torch.manual_seed(0)
h = 4096; dev = "cuda"
r32 = torch.randn(h, device = dev); r32 /= r32.norm(); r16 = r32.half()

def variant_cur(y, w): A.project(y, w, r32, r16)
def variant_half_direct(y, w):
    y2 = y.view(-1, h)
    if y2.dtype == torch.float32:
        y2.addr_(torch.mv(y2, r32), r32, alpha = -w)
    else:
        y2.addr_(torch.mv(y2, r16), r16, alpha = -w)

def bench(fn, R, sites = 43):
    ya = [torch.randn(R, h, device = dev) for _ in range(sites)]
    ym = [torch.randn(R, h, device = dev).half() for _ in range(sites)]
    def body():
        for i in range(sites):
            fn(ya[i], 0.8); fn(ym[i], 0.6)
    body(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): body()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); t = time.perf_counter(); n = 50
    for _ in range(n): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

A._USE_TRITON = False
for name, fn in (("cur", variant_cur), ("half_direct", variant_half_direct)):
    for R in (1, 3):
        print(f"{name:12s} R={R}: {bench(fn, R):.3f} ms per round (86 sites)", flush = True)

A.warmup(dev); A._USE_TRITON = A.triton is not None
for R in (1, 3):
    print(f"triton       R={R}: {bench(variant_cur, R):.3f} ms per round (86 sites)", flush = True)
