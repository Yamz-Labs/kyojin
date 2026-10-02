#!/usr/bin/env python3
"""Isolated exl3_gemv microbenchmark on REAL model tensors (no generator). For each target
Linear: time ext.exl3_gemv at m = 1,2,3,4,8 (CUDA events, many iters), compute the bandwidth
roofline for the trellis bytes, and report achieved GB/s. Optionally check parity vs the
reconstruct path (EXL3_GEMV=0 -> torch matmul on the dequantized weight).

Usage: python gemv_micro.py [--iters 200] [--parity]
Env knobs pass straight through to the extension (EXL3_GEMV=3/4 force narrow/wide, etc.)."""
import os, sys, time, argparse
import torch
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.util.tensor import g_tensor_cache
import exllamav3.ext as extmod
ext = extmod.exllamav3_ext

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--ms", default="1,2,3,4,8")
ap.add_argument("--targets", default="q,k,o,gate_up,down,lm_head")
ap.add_argument("--parity", action="store_true")
args = ap.parse_args()
Ms = [int(x) for x in args.ms.split(",")]

def bw_roofline_ms(bytes_):
    return bytes_ / 236e9 * 1e3   # measured copy roofline on this box

def find_targets(model):
    """Pick one representative LinearEXL3 per shape class."""
    want = {}
    stack = list(model.modules)
    seen = {}
    while stack:
        mod = stack.pop()
        inner = getattr(mod, "inner", None)
        if isinstance(inner, LinearEXL3):
            key = (inner.in_features, inner.out_features, inner.K)
            if key not in seen:
                seen[key] = (mod.key, inner)
        for c in getattr(mod, "modules", []) or []:
            stack.append(c)
    return seen

def main():
    config = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
    model = Model.from_config(config)
    # load only enough: full load is ~21 s warm; fine
    model.load(progressbar=False)
    dev = torch.device("cuda:0")
    targets = find_targets(model)
    print(f"{len(targets)} distinct (k, n, K) shapes; timing the interesting ones\n")
    # select by shape: attention projections (2560 -> 6144/10240/12288, 6144 -> 2560), experts (2560->640/512, 640->2560), lm_head
    interesting = sorted(targets.items(), key=lambda kv: -kv[0][0] * kv[0][1])
    hdr = f"{'module':44s} {'k':>6s} {'n':>7s} {'K':>2s} | " + " ".join(f"{'m='+str(m):>9s}" for m in Ms) + " | roofline  best%"
    print(hdr); print("-" * len(hdr))
    for (k, n, K), (key, lin) in interesting:
        bytes_ = k * n * K // 8
        rl = bw_roofline_ms(bytes_)
        cells = []; best = None
        for m in Ms:
            x = torch.randn((m, k), dtype=torch.half, device=dev) * 0.5
            y = torch.empty((m, n), dtype=torch.half, device=dev)
            A_had = g_tensor_cache.get(dev, (m, k), torch.half, "exl3_gemv_a_had")
            f = lambda: ext.exl3_gemv(x, lin.trellis, y, lin.suh, A_had, lin.svh, lin.mcg, lin.mul1)
            for _ in range(10): f()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(args.iters): f()
            e1.record(); torch.cuda.synchronize()
            us = e0.elapsed_time(e1) / args.iters * 1e3
            cells.append(f"{us:7.1f}us")
            if best is None or us < best: best = us
        print(f"{key[-44:]:44s} {k:6d} {n:7d} {K:2d} | " + " ".join(f"{c:>9s}" for c in cells) + f" | {rl*1e3:6.1f}us {100*rl*1e3/best:5.0f}%")

if __name__ == "__main__":
    main()
