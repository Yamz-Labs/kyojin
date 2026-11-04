#!/usr/bin/env python3
"""Per-shape and streaming timings for the ROCm EXL3 GEMV route.

Micro mode times `ext.exl3_gemv` per distinct (in, out, K, codebook) shape at
m = 1 and reports effective weight bandwidth. Stream mode runs one pass over
every EXL3 linear in the model, which removes the per-shape L2 reuse and is the
number to compare with a decode step. `--gemm` also times `ext.exl3_gemm` when
the tree exports it (upstream routing), for an A/B against the direct GEMV.

Example:
    HIP_VISIBLE_DEVICES=1 python tools/strix_halo/gemv_perf_ab.py \
        -m <model> --mode both --gemm

This loads the full model, so use a pack that fits the visible device (or the
layer-split sibling for larger packs).
"""
import argparse
import os
import statistics
import torch
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.util.tensor import g_tensor_cache
import exllamav3.ext as extmod

ext = extmod.exllamav3_ext


def collect(model):
    mods = []
    stack = list(model.modules)
    while stack:
        mod = stack.pop()
        inner = getattr(mod, "inner", None)
        if isinstance(inner, LinearEXL3):
            mods.append(inner)
        stack.extend(getattr(mod, "modules", []) or [])
    return mods


def shape_key(l):
    return (l.in_features, l.out_features, int(l.K), int(l.mcg) + 2 * int(l.mul1))


def time_us(fn, iters):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=os.environ.get("EXL3_TEST_MODEL", ""))
    ap.add_argument("--mode", choices=("micro", "stream", "both"), default="both")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--gemm", action="store_true",
                    help="also time ext.exl3_gemm when present")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("--model or EXL3_TEST_MODEL is required")

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(progressbar=False)
    dev = torch.device("cuda:0")
    linears = collect(model)
    if not linears:
        raise SystemExit("no LinearEXL3 modules found")
    total_bytes = sum(l.in_features * l.out_features * int(l.K) // 8 for l in linears)
    print(f"model: {args.model}")
    print(f"ext:   {ext.__file__}")
    print(f"linears: {len(linears)}  weight stream: {total_bytes / 1e9:.3f} GB")

    def buffers(l):
        x = torch.randn((1, l.in_features), dtype=torch.half, device=dev) * 0.5
        y = torch.empty((1, l.out_features), dtype=torch.half, device=dev)
        ah = g_tensor_cache.get(dev, (1, l.in_features), torch.half, "exl3_gemv_a_had")
        return x, y, ah

    if args.mode in ("micro", "both"):
        seen = {}
        for l in linears:
            seen.setdefault(shape_key(l), l)
        rows = sorted(seen.items(), key=lambda kv: -(kv[0][0] * kv[0][1] * kv[0][2] // 8))
        has_gemm = args.gemm and hasattr(ext, "exl3_gemm")
        head = f"{'k':>6} {'n':>7} {'K':>2} {'cb':>2} {'MB':>8} {'us':>8} {'GB/s':>7}"
        if has_gemm:
            head += f" {'gemm_us':>8} {'gemm_GB/s':>9}"
        print("\n" + head)
        for (k, n, K, cb), l in rows[:args.top]:
            x, y, ah = buffers(l)
            b = k * n * K // 8
            f = lambda: ext.exl3_gemv(x, l.trellis, y, l.suh, ah, l.svh, l.mcg, l.mul1)
            us = time_us(f, args.iters)
            line = f"{k:6d} {n:7d} {K:2d} {cb:2d} {b/1e6:8.2f} {us:8.1f} {b/us/1e3:7.0f}"
            if has_gemm:
                g = lambda: ext.exl3_gemm(x, l.trellis, y, l.suh, ah, l.svh, -1,
                                          l.mcg, l.mul1, 0)
                usg = time_us(g, args.iters)
                line += f" {usg:8.1f} {b/usg/1e3:9.0f}"
            print(line, flush=True)

    if args.mode in ("stream", "both"):
        work = []
        for l in linears:
            x, y, ah = buffers(l)
            work.append((l, x, y, ah))

        def run_gemv():
            for l, x, y, ah in work:
                ext.exl3_gemv(x, l.trellis, y, l.suh, ah, l.svh, l.mcg, l.mul1)

        def run_gemm():
            for l, x, y, ah in work:
                ext.exl3_gemm(x, l.trellis, y, l.suh, ah, l.svh, -1, l.mcg, l.mul1, 0)

        def stream(fn, passes=3):
            fn()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(passes):
                fn()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) / passes

        print("\nstreaming pass over all linears (median of 3):")
        ms = statistics.median([stream(run_gemv) for _ in range(3)])
        print(f"  exl3_gemv: {ms:.2f} ms  {total_bytes / ms * 1e-6:.0f} GB/s")
        if args.gemm and hasattr(ext, "exl3_gemm"):
            msg = statistics.median([stream(run_gemm) for _ in range(3)])
            print(f"  exl3_gemm: {msg:.2f} ms  {total_bytes / msg * 1e-6:.0f} GB/s")


if __name__ == "__main__":
    main()
