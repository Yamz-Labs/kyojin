#!/usr/bin/env python3
"""Microprofile EXL3 tile encoding and serialization on gfx1151; wrap with rocprofv3 for traces."""

import argparse
import os
import time

import torch
from exllamav3.ext import exllamav3_ext as ext


def scratch(device, tile_count, K, L, optimized):
    edges = 65536 >> K
    if optimized:
        _, wave = getattr(ext, "quantize_tiles_scratch")(device.index, K, False, False, L)
        history_cols = edges // (8 if K == 1 else 1)
        if torch.version.hip and K <= 2:
            costs = torch.empty((wave, 2, edges), dtype=torch.float16, device=device)
        else:
            costs = torch.empty((1,), dtype=torch.float16, device=device)
        history = torch.empty((wave, L, history_cols), dtype=torch.uint8, device=device)
    else:
        _, wave = getattr(ext, "quantize_tiles_scratch")(device.index, K, False, False, L)
        n = min(tile_count, wave)
        costs = torch.empty((n, 2, edges), dtype=torch.float16, device=device)
        history = torch.empty((n, L, edges), dtype=torch.int16, device=device)
    return costs, history


def timed_events(device, fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--codebook", choices=("3inst", "mcg", "mul1"), default="3inst")
    parser.add_argument("--tiles", type=int, default=8, help="must be a multiple of 8 for reconstruction")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--weights-file")
    parser.add_argument("--tensor-key", default="model.layers.0.self_attn.q_proj.weight")
    parser.add_argument("--modes", nargs="+", choices=("0", "1"), default=("0", "1"))
    args = parser.parse_args()
    if not 1 <= args.k <= 8 or args.tiles <= 0 or args.iterations <= 0:
        parser.error("K must be 1..8; tiles and iterations must be positive")
    if args.tiles % 8:
        parser.error("tiles must be a multiple of 8 for the reconstruct check")
    if not torch.cuda.is_available():
        raise RuntimeError("HIP GPU unavailable")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if props.gcnArchName != "gfx1151":
        raise RuntimeError(f"expected gfx1151, got {props.gcnArchName}")
    print(f"device={props.name} arch={props.gcnArchName} K={args.k} tiles={args.tiles}")

    if args.weights_file:
        from safetensors import safe_open
        with safe_open(args.weights_file, framework="pt", device="cpu") as sf:
            if args.tensor_key not in sf.keys():
                raise KeyError(args.tensor_key)
            tile = sf.get_tensor(args.tensor_key).reshape(-1)[:256].float()
        rms = tile.square().mean().sqrt()
        if not torch.isfinite(tile).all() or rms == 0:
            raise ValueError("real-weight sample is non-finite or zero-RMS")
        scale = 1.0 if args.codebook == "mul1" else 1.24371088
        tile = tile * (scale / rms)
        cpu_tiles = tile.repeat(args.tiles, 1).contiguous()
        input_kind = args.tensor_key
    else:
        generator = torch.Generator().manual_seed(20260923 + args.k)
        scale = 1.0 if args.codebook == "mul1" else 1.24371088
        cpu_tiles = (torch.randn((args.tiles, 256), generator=generator) * scale).contiguous()
        input_kind = "gaussian"

    pinned_tiles = cpu_tiles.pin_memory()
    x = torch.empty_like(pinned_tiles, device=device)
    for _ in range(args.warmup):
        pinned_tiles.copy_(cpu_tiles)
        x.copy_(pinned_tiles, non_blocking=True)
        torch.cuda.synchronize(device)
    copy_start = time.perf_counter()
    for _ in range(args.iterations):
        pinned_tiles.copy_(cpu_tiles)
    print(f"python_timer stage=cpu_to_pinned_copy_ms={(time.perf_counter() - copy_start) * 1000 / args.iterations:.4f} input={input_kind}")
    h2d_ms = timed_events(device, lambda: x.copy_(pinned_tiles, non_blocking=True), args.iterations)
    print(f"python_timer stage=pinned_h2d_copy_gpu_ms={h2d_ms:.4f} input={input_kind}")

    mcg = args.codebook == "mcg"
    mul1 = args.codebook == "mul1"
    for mode in args.modes:
        os.environ["EXL3_QT_OPTIMIZED"] = mode
        optimized, wave = getattr(ext, "quantize_tiles_scratch")(device.index, args.k, mcg, mul1, 256)
        costs, history = scratch(device, args.tiles, args.k, 256, optimized)
        out = torch.empty_like(x)
        indices = torch.empty_like(x, dtype=torch.int16)

        def encode():
            getattr(ext, "quantize_tiles")(x, out, indices, costs, history, args.k, mcg, mul1)

        for _ in range(args.warmup):
            encode()
        kernel_ms = timed_events(device, encode, args.iterations)
        decoded = torch.empty_like(out)
        getattr(ext, "decode")(indices, decoded, mcg, mul1)
        torch.cuda.synchronize(device)
        if not torch.equal(out, decoded):
            raise AssertionError(f"mode={mode}: encoder and decoder outputs differ")

        packed = torch.empty((1, args.tiles, 256 * args.k // 16), dtype=torch.int16, device=device)
        encoded = indices.contiguous().reshape(1, args.tiles, 256)
        unpacked = torch.empty_like(encoded)
        reconstructed = torch.empty((16, args.tiles * 16), dtype=torch.float16, device=device)
        for _ in range(args.warmup):
            getattr(ext, "pack_trellis")(packed, encoded, args.k)
            getattr(ext, "unpack_trellis")(unpacked, packed, args.k)
            getattr(ext, "reconstruct")(reconstructed, packed, args.k, mcg, mul1)
        torch.cuda.synchronize(device)
        pack_ms = timed_events(device, lambda: getattr(ext, "pack_trellis")(packed, encoded, args.k), args.iterations)
        unpack_ms = timed_events(device, lambda: getattr(ext, "unpack_trellis")(unpacked, packed, args.k), args.iterations)
        recon_ms = timed_events(device, lambda: getattr(ext, "reconstruct")(reconstructed, packed, args.k, mcg, mul1), args.iterations)
        print(
            f"mode={mode} optimized={optimized} wave={wave} codebook={args.codebook} "
            f"python_timer stage=quantize_tiles_gpu_ms={kernel_ms:.4f} tiles_per_s={args.tiles * 1000 / kernel_ms:.0f} "
            f"pack_gpu_ms={pack_ms:.4f} unpack_gpu_ms={unpack_ms:.4f} reconstruct_gpu_ms={recon_ms:.4f}"
        )
    os.environ.pop("EXL3_QT_OPTIMIZED", None)
    print("PASS")


if __name__ == "__main__":
    main()
