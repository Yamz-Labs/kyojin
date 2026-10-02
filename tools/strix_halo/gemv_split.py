#!/usr/bin/env python3
"""Split an exl3_gemv call into its three HIP launches (had_r_128 in, main kernel, had_r_128 out)
and time each in isolation with CUDA events, for the shapes that matter. Uses the same real
tensors as gemv_micro.py. Also times an empty-kernel launch to get the per-launch floor."""
import os, time, argparse
import torch
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.util.tensor import g_tensor_cache
import exllamav3.ext as extmod
ext = extmod.exllamav3_ext

ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=300); ap.add_argument("-m", type=int, default=3)
args = ap.parse_args()

def find_targets(model):
    stack = list(model.modules); seen = {}
    while stack:
        mod = stack.pop()
        inner = getattr(mod, "inner", None)
        if isinstance(inner, LinearEXL3):
            key = (inner.in_features, inner.out_features, inner.K)
            if key not in seen: seen[key] = (mod.key, inner)
        for c in getattr(mod, "modules", []) or []: stack.append(c)
    return seen

def timeit(f, iters):
    for _ in range(10): f()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): f()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3

def main():
    config = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
    model = Model.from_config(config); model.load(progressbar=False)
    dev = torch.device("cuda:0"); m = args.m
    # launch floor: a trivial torch op
    z = torch.zeros(1, device=dev)
    print(f"launch floor (tiny torch fill_): {timeit(lambda: z.fill_(1.0), args.iters):.1f} us")
    print(f"launch floor (2 tiny fills):     {timeit(lambda: (z.fill_(1.0), z.fill_(2.0)), args.iters):.1f} us\n")
    hdr = f"{'module':40s} {'k':>5s} {'n':>7s} K | {'had_in':>8s} {'had_out':>8s} {'full':>8s} {'gemv=full-hads':>14s} | bytes-floor"
    print(hdr); print("-" * len(hdr))
    for (k, n, K), (key, lin) in sorted(find_targets(model).items(), key=lambda kv: -kv[0][0]*kv[0][1]):
        x = torch.randn((m, k), dtype=torch.half, device=dev) * 0.5
        y = torch.empty((m, n), dtype=torch.half, device=dev)
        A_had = g_tensor_cache.get(dev, (m, k), torch.half, "exl3_gemv_a_had")
        t_full = timeit(lambda: ext.exl3_gemv(x, lin.trellis, y, lin.suh, A_had, lin.svh, lin.mcg, lin.mul1), args.iters)
        t_hin  = timeit(lambda: ext.had_r_128(x, A_had, lin.suh, None, 1.0), args.iters)
        t_hout = timeit(lambda: ext.had_r_128(y, y, None, lin.svh, 1.0), args.iters)
        floor = k * n * K / 8 / 236e9 * 1e6
        print(f"{key[-40:]:40s} {k:5d} {n:7d} {K} | {t_hin:7.1f}us {t_hout:7.1f}us {t_full:7.1f}us {t_full-t_hin-t_hout:13.1f}us | {floor:7.1f}us")

if __name__ == "__main__":
    main()
