#!/usr/bin/env python
"""Measure the box roofline once: read bandwidth (GB/s) and dense fp16 GEMM peak (TFLOPS). Writes hw.json.
Run on a quiet GPU:  flock ~/ds41_work/gpu.lock ~/ds41_work/cur1510/quiet.sh python tools/depth/hw.py"""
import json, os, sys, time
import torch

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw.json")
dev = torch.device("cuda:0")


def timed(fn, reps=8):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - a)
    ts.sort()
    return ts[len(ts) // 2]


bw = 0.0
for gib in (2, 4):
    x = torch.ones(gib * 2**30 // 2, dtype=torch.float16, device=dev)
    dt = timed(lambda: x.sum())
    bw = max(bw, x.numel() * 2 / dt / 1e9)
    del x
tf = 0.0
for n in (4096, 8192):
    a = torch.randn(n, n, dtype=torch.float16, device=dev); b = torch.randn(n, n, dtype=torch.float16, device=dev)
    dt = timed(lambda: a @ b)
    tf = max(tf, 2 * n**3 / dt / 1e12)
    del a, b
# skinny GEMM (chunk of 4096 rows x big weight): what the prefill linears see
a = torch.randn(4096, 4096, dtype=torch.float16, device=dev); b = torch.randn(4096, 16384, dtype=torch.float16, device=dev)
tf_chunk = 2 * 4096 * 4096 * 16384 / timed(lambda: a @ b) / 1e12
hw = dict(bw_gbs=round(bw, 1), gemm_tflops=round(tf, 1), gemm_tflops_4096rows=round(tf_chunk, 1),
          device=torch.cuda.get_device_name(0), measured=time.strftime("%Y-%m-%d"))
json.dump(hw, open(out, "w"), indent=1)
print(hw)
