#!/usr/bin/env python3
"""Minimal single-shape exl3_gemv loop for rocprofv3: loads the model, picks ONE Linear by shape,
runs N iterations of ext.exl3_gemv. Shape selected by --k --n (defaults: q_proj 2560x12288)."""
import os, argparse, torch
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.util.tensor import g_tensor_cache
import exllamav3.ext as extmod
ext = extmod.exllamav3_ext
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=2560); ap.add_argument("--n", type=int, default=12288)
ap.add_argument("-m", type=int, default=3); ap.add_argument("--iters", type=int, default=50)
args = ap.parse_args()
def main():
    config = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
    model = Model.from_config(config); model.load(progressbar=False)
    stack = list(model.modules); lin = None
    while stack and lin is None:
        mod = stack.pop(); inner = getattr(mod, "inner", None)
        if isinstance(inner, LinearEXL3) and inner.in_features == args.k and inner.out_features == args.n: lin = inner; print("using", mod.key)
        for c in getattr(mod, "modules", []) or []: stack.append(c)
    dev = torch.device("cuda:0"); m = args.m
    x = torch.randn((m, args.k), dtype=torch.half, device=dev) * 0.5
    y = torch.empty((m, args.n), dtype=torch.half, device=dev)
    A_had = g_tensor_cache.get(dev, (m, args.k), torch.half, "exl3_gemv_a_had")
    for _ in range(5): ext.exl3_gemv(x, lin.trellis, y, lin.suh, A_had, lin.svh, lin.mcg, lin.mul1)
    torch.cuda.synchronize()
    for _ in range(args.iters): ext.exl3_gemv(x, lin.trellis, y, lin.suh, A_had, lin.svh, lin.mcg, lin.mul1)
    torch.cuda.synchronize(); print("done")
if __name__ == "__main__": main()
