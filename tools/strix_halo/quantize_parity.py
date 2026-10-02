#!/usr/bin/env python3
"""Bit-parity and throughput of the EXL3 tile-quantizer kernel families on gfx1151.

Every mode quantizes the same tiles; indices and decoded outputs must equal the base kernel's
(quantize_tiles_kernel) exactly. Inputs: Gaussian tiles, edge-case tiles, and optional real tiles
dumped by the converter (EXL3_DUMP_TILES, see exl3_lib/quantize.py).
"""

import argparse
import glob
import os
import time

import torch
from exllamav3.ext import exllamav3_ext as ext

CODEBOOKS = {"3inst": (False, False), "mcg": (True, False), "mul1": (False, True)}


def run(x, K, mcg, mul1, mode, nt=None):
    os.environ["EXL3_QT_KERNEL"] = mode
    if nt:
        os.environ["EXL3_QT_RDNA_NT"] = str(nt)
    else:
        os.environ.pop("EXL3_QT_RDNA_NT", None)
    edges = 65536 >> K
    layout, wave = ext.quantize_tiles_scratch(0, K, mcg, mul1, 256)
    n = min(wave, x.shape[0])
    if layout:
        costs = torch.empty((n, 2, edges) if K <= 2 else (1,), dtype=torch.half, device=x.device)
        history = torch.empty((n, 256, edges), dtype=torch.uint8, device=x.device)
    else:
        costs = torch.empty((n, 2, edges), dtype=torch.half, device=x.device)
        history = torch.empty((n, 256, edges), dtype=torch.int16, device=x.device)
    out = torch.empty_like(x)
    idx = torch.empty_like(x, dtype=torch.int16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ext.quantize_tiles(x, out, idx, costs, history, K, mcg, mul1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del costs, history
    return out, idx, dt, wave


def edge_tiles(scale):
    g = torch.Generator().manual_seed(7)
    t = [
        torch.zeros(256),
        torch.full((256,), 0.5 * scale),
        torch.randn(256, generator=g) * 1e-4,
        torch.randn(256, generator=g) * 300.0,          # squared errors overflow half: inf costs
        torch.linspace(-4, 4, 256) * scale,
        torch.where(torch.arange(256) % 2 == 0, 3.0, -3.0) * scale,
    ]
    return torch.stack(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--codebooks", nargs="+", default=["mul1"], choices=list(CODEBOOKS))
    ap.add_argument("--tiles", type=int, default=2048)
    ap.add_argument("--modes", nargs="+", default=["rdna", "opt"])
    ap.add_argument("--nt", type=int, nargs="*", default=[None], help="EXL3_QT_RDNA_NT values to test for rdna")
    ap.add_argument("--real", help="glob of dumped tile tensors (.pt, shape (N, 256) float32)")
    ap.add_argument("--real-max", type=int, default=8192)
    ap.add_argument("--no-base-timing", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    print(f"device={props.name} arch={props.gcnArchName} ext={ext.__file__}")
    failures = 0
    for cbname in args.codebooks:
        mcg, mul1 = CODEBOOKS[cbname]
        scale = 1.0 if mul1 else 1.24371088
        sets = []
        gen = torch.Generator().manual_seed(20260924)
        sets.append(("gaussian", torch.randn((args.tiles, 256), generator=gen) * scale))
        sets.append(("edge", edge_tiles(scale)))
        if args.real:
            for path in sorted(glob.glob(args.real)):
                t = torch.load(path).float().reshape(-1, 256)[: args.real_max]
                sets.append((os.path.basename(path), t))
        for K in args.ks:
            for name, tiles in sets:
                x = tiles.to(dev).contiguous()
                ref_out, ref_idx, ref_dt, ref_wave = run(x, K, mcg, mul1, "base")
                dec = torch.empty_like(ref_out)
                ext.decode(ref_idx, dec, mcg, mul1)
                assert torch.equal(dec, ref_out), "base: encoder/decoder mismatch"
                u = ref_idx.cpu().numpy().astype("uint16")
                assert ((u[:, 0] >> K) == (u[:, -1] & ((1 << (16 - K)) - 1))).all(), "base: tail-biting broken"
                line = f"cb={cbname} K={K} set={name} n={x.shape[0]} base={x.shape[0] / ref_dt:.0f}t/s"
                for mode in args.modes:
                    for nt in (args.nt if mode == "rdna" else [None]):
                        out, idx, dt, wave = run(x, K, mcg, mul1, mode, nt)
                        # Warm second run for timing (first call includes setup)
                        out2, idx2, dt2, _ = run(x, K, mcg, mul1, mode, nt)
                        same = torch.equal(idx, ref_idx) and torch.equal(out, ref_out) and torch.equal(idx2, ref_idx)
                        diff = int((idx != ref_idx).sum().item())
                        tag = f"{mode}{'/' + str(nt) if nt else ''}"
                        line += f" | {tag}={x.shape[0] / dt2:.0f}t/s wave={wave} {'IDENTICAL' if same else f'MISMATCH({diff})'}"
                        failures += 0 if same else 1
                print(line, flush=True)
    os.environ.pop("EXL3_QT_KERNEL", None)
    print("PARITY", "PASS" if failures == 0 else f"FAIL ({failures})")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
