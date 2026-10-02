#!/usr/bin/env python3
"""
Per-tensor harness: the fork's HIP reconstruct/GEMV paths vs the CPU reference in
ref/frac_reconstruct.py, on real tensors of the MiMo EXL3 checkpoint.

  --mode raw  : ext.reconstruct(w, packed, K, mcg, mul1)          (reconstruct_slice)
  --mode had  : ext.reconstruct_had_slice(w, packed, suh, svh, K, mcg, mul1, n_offset)
  --mode gemv : ext.exl3_gemv(x, packed, y, suh, A_had, svh, mcg, mul1) vs x @ W_ref
  --mode moe  : one K=2.5 expert's gate/up/SiLU/down routed contribution vs CPU W matrices

Raw tiles must match bit-for-bit. Had mode allows the measured fp16 butterfly-rounding delta.
GEMV reports repeat-call determinism and error against the CPU reconstructed full matrix.

Run on the GPU with the ROCm venv python.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

REF_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(REF_DIR)
EXT_DIR = os.environ.get("EXL3_EXT_DIR", REPO)
sys.path.insert(0, REF_DIR)

import frac_reconstruct as R  # noqa: E402


def get_ext():
    sys.path.insert(0, EXT_DIR)
    import exllamav3_ext as ext  # type: ignore
    return ext


def cmp_bits(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a, b: same-shape fp16 tensors. Returns bit-level and float-level agreement."""
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    same = (ai == bi)
    af, bf = a.to(torch.float32), b.to(torch.float32)
    d = (af - bf).abs()
    scale = bf.abs().mean().clamp(min=1e-12)
    return {
        "n": int(a.numel()),
        "bit_exact": int(same.sum()),
        "bit_frac": float(same.to(torch.float32).mean()),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "rel_mean_abs_diff": float(d.mean() / scale),
        "rel_l2_diff": float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(bf).clamp(min=1e-12)),
    }


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Ordered fp16 bit-pattern distance, including values with opposite signs."""
    ai = a.contiguous().view(torch.uint16).to(torch.int32)
    bi = b.contiguous().view(torch.uint16).to(torch.int32)

    def key(x):
        negative = (x & 0x8000) != 0
        return torch.where(negative, (~x) & 0xFFFF, x | 0x8000)

    return (key(ai) - key(bi)).abs()


def validate_gemv(ext, args, trellis, suh, svh, cb, K):
    tile_label = "full"
    if args.tile:
        kt, nt = tuple(args.kt), tuple(args.nt)
        trellis = trellis[kt[0]:kt[1], nt[0]:nt[1]].contiguous()
        suh = suh[16 * kt[0]:16 * kt[1]].contiguous()
        svh = svh[16 * nt[0]:16 * nt[1]].contiguous()
        tile_label = f"kt{kt}/nt{nt}"
    device = torch.device(args.device)
    mcg, mul1 = cb == R.CB_MCG, cb == R.CB_MUL1
    if K % 1 and not mul1:
        raise ValueError(f"{args.tensor}: fractional K={K} requires MUL1 codebook")

    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    gen = torch.Generator(device="cpu").manual_seed(314159)
    x_cpu = torch.randn((args.rows, k), generator=gen, dtype=torch.float16) * 0.1
    x_gpu = x_cpu.to(device).contiguous()
    trellis_gpu = trellis.to(device)
    suh_gpu, svh_gpu = suh.to(device), svh.to(device)
    a_had = torch.empty_like(x_gpu)
    y_gpu = torch.empty((args.rows, n), dtype=torch.float16, device=device)
    ext.exl3_gemv(x_gpu, trellis_gpu, y_gpu, suh_gpu, a_had, svh_gpu, mcg, mul1)
    torch.cuda.synchronize()
    y_repeat = torch.empty_like(y_gpu)
    ext.exl3_gemv(x_gpu, trellis_gpu, y_repeat, suh_gpu, a_had, svh_gpu, mcg, mul1)
    torch.cuda.synchronize()

    w_ref = R.reconstruct(trellis, K, cb, suh, svh, had=True, mask=args.mask)
    y_ref = (x_cpu.to(torch.float32) @ w_ref.to(torch.float32)).to(torch.float16)
    actual, expected = y_gpu.cpu(), y_ref
    noise = cmp_bits(actual, y_repeat.cpu())
    result = cmp_bits(actual, expected)
    print(f"[gemv] {args.tensor} K={K} cb={cb} rows={args.rows} W={tuple(w_ref.shape)} tiles={tile_label}")
    print(f"  kernel repeat: bit-exact {noise['bit_frac'] * 100:.4f}%  max|d| {noise['max_abs_diff']:.3e}")
    print(f"  y vs CPU x@W : max|d| {result['max_abs_diff']:.3e}  "
          f"mean|d| {result['mean_abs_diff']:.3e}  rel-mean {result['rel_mean_abs_diff']:.3e}  "
          f"rel-L2 {result['rel_l2_diff']:.3e}")
    print(f"  output rms GPU {float(actual.float().pow(2).mean().sqrt()):.6f}  "
          f"CPU {float(expected.float().pow(2).mean().sqrt()):.6f}")
    passed = (
        noise["bit_frac"] == 1.0 and
        result["max_abs_diff"] <= 5e-3 and
        result["rel_mean_abs_diff"] <= 1e-2 and
        result["rel_l2_diff"] <= 1e-2
    )
    print("  PASS" if passed else "  FAIL")
    return 0 if passed else 1


def validate_moe(ext, args):
    suffix = ".gate_proj"
    if not args.tensor.endswith(suffix):
        raise ValueError("MoE mode expects an expert gate projection key")
    expert = args.tensor[:-len(suffix)]
    gate_t, gate_suh, gate_svh, gate_cb = R.load_layer(args.model, expert + ".gate_proj")
    up_t, up_suh, up_svh, up_cb = R.load_layer(args.model, expert + ".up_proj")
    down_t, down_suh, down_svh, down_cb = R.load_layer(args.model, expert + ".down_proj")
    K = R.K_of(gate_t)
    if K != 2.5 or not (gate_cb == up_cb == down_cb == R.CB_MUL1):
        raise ValueError("MoE check requires uniform K=2.5 MUL1 projections")
    if R.K_of(up_t) != K or R.K_of(down_t) != K:
        raise ValueError("MoE expert projections have different bitrates")

    device = torch.device(args.device)
    k = gate_t.shape[0] * 16
    gen = torch.Generator(device="cpu").manual_seed(271828)
    x_cpu = torch.randn((args.rows, k), generator=gen, dtype=torch.float16) * 0.1
    x_gpu = x_cpu.to(device).contiguous()

    def gemv(trellis, suh, svh, cb, activations):
        y = torch.empty((args.rows, trellis.shape[1] * 16), dtype=torch.float16, device=device)
        a_had = torch.empty_like(activations)
        ext.exl3_gemv(
            activations, trellis.to(device), y, suh.to(device), a_had, svh.to(device),
            cb == R.CB_MCG, cb == R.CB_MUL1)
        return y

    gate_gpu = gemv(gate_t, gate_suh, gate_svh, gate_cb, x_gpu)
    up_gpu = gemv(up_t, up_suh, up_svh, up_cb, x_gpu)
    activation_gpu = torch.nn.functional.silu(gate_gpu) * up_gpu
    down_gpu = gemv(down_t, down_suh, down_svh, down_cb, activation_gpu.contiguous())

    gate_w = R.reconstruct(gate_t, K, gate_cb, gate_suh, gate_svh, had=True)
    up_w = R.reconstruct(up_t, K, up_cb, up_suh, up_svh, had=True)
    down_w = R.reconstruct(down_t, K, down_cb, down_suh, down_svh, had=True)
    gate_ref = (x_cpu.float() @ gate_w.float()).half()
    up_ref = (x_cpu.float() @ up_w.float()).half()
    activation_ref = torch.nn.functional.silu(gate_ref) * up_ref
    down_ref = (activation_ref.float() @ down_w.float()).half()
    route_weight = 0.375
    down_gpu.mul_(route_weight)
    down_ref.mul_(route_weight)

    comparisons = {
        "gate": cmp_bits(gate_gpu.cpu(), gate_ref),
        "up": cmp_bits(up_gpu.cpu(), up_ref),
        "down+route": cmp_bits(down_gpu.cpu(), down_ref),
    }
    print(f"[moe-fallback] {expert} K={K} MUL1 rows={args.rows} route_weight={route_weight}")
    for name, r in comparisons.items():
        print(f"  {name:10s} max|d| {r['max_abs_diff']:.3e}  rel-mean {r['rel_mean_abs_diff']:.3e}  "
              f"rel-L2 {r['rel_l2_diff']:.3e}")
    passed = all(
        torch.isfinite(down_gpu).all().item() and
        r["max_abs_diff"] <= 5e-3 and r["rel_l2_diff"] <= 1e-2
        for r in comparisons.values()
    )
    print("  PASS" if passed else "  FAIL")
    return 0 if passed else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="~/models/mimo26-exl3")
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--mode", choices=["raw", "had", "gemv", "moe"], default="raw")
    ap.add_argument("--kt", type=int, nargs=2, default=(0, 8), help="tile rows [a b)")
    ap.add_argument("--nt", type=int, nargs=2, default=(0, 8), help="tile cols [a b)")
    ap.add_argument("--mask", type=lambda s: int(s, 0), default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=1, help="GEMV input rows")
    ap.add_argument("--tile", action="store_true", help="GEMV mode: compare the --kt/--nt tile rectangle")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    ext = get_ext()
    if args.mode == "moe":
        return validate_moe(ext, args)
    tr, suh, svh, cb = R.load_layer(args.model, args.tensor)
    K = R.K_of(tr)
    if args.mode == "gemv":
        return validate_gemv(ext, args, tr, suh, svh, cb, K)
    kt, nt = tuple(args.kt), tuple(args.nt)
    packed = tr[kt[0]:kt[1], nt[0]:nt[1]].contiguous()
    mcg, mul1 = cb == R.CB_MCG, cb == R.CB_MUL1
    if K % 1 and not mul1:
        raise ValueError(f"{args.tensor}: fractional K={K} requires MUL1 codebook")

    device = torch.device(args.device)
    p_gpu = packed.to(device)
    suh_s = svh_s = None
    w_gpu = torch.empty((16 * (kt[1] - kt[0]), 16 * (nt[1] - nt[0])), dtype=torch.half, device=device)
    if args.mode == "raw":
        ext.reconstruct(w_gpu, p_gpu, K, mcg, mul1)
    else:
        suh_s = suh[16 * kt[0]:16 * kt[1]].to(device)
        svh_s = svh[16 * nt[0]:16 * nt[1]].to(device)
        ext.reconstruct_had_slice(w_gpu, p_gpu, suh_s, svh_s, K, mcg, mul1, 0)
    torch.cuda.synchronize()

    # determinism / noise floor: same call again
    w2 = torch.empty_like(w_gpu)
    if args.mode == "raw":
        ext.reconstruct(w2, p_gpu, K, mcg, mul1)
    else:
        ext.reconstruct_had_slice(w2, p_gpu, suh_s, svh_s, K, mcg, mul1, 0)
    torch.cuda.synchronize()

    ref = R.reconstruct(packed, K, cb, suh, svh, had=(args.mode == "had"), mask=args.mask,
                        kt0=kt[0], kt1=kt[1], nt0=nt[0], nt1=nt[1])

    a, b = w_gpu.cpu(), ref
    r_gg = cmp_bits(a, w2.cpu())
    r_gr = cmp_bits(a, b)
    ulp = ulp_distance(a, b)
    print(f"[{args.mode}] {args.tensor}  K={K} cb={cb} tiles kt{kt} nt{nt}  {tuple(b.shape)}  tag={args.tag}")
    print(f"  kernel vs kernel (noise floor): bit-exact {r_gg['bit_frac']*100:.4f}%  "
          f"max|d| {r_gg['max_abs_diff']:.3e}")
    print(f"  kernel vs reference          : bit-exact {r_gr['bit_frac']*100:.4f}%  "
          f"max|d| {r_gr['max_abs_diff']:.3e}  mean|d| {r_gr['mean_abs_diff']:.3e}  "
          f"rel {r_gr['rel_mean_abs_diff']:.3e}")
    if r_gr["bit_frac"] < 1.0:
        print(f"  ordered fp16 ULP kernel->ref: max {int(ulp.max())}  mean {float(ulp.float().mean()):.4f}")
    print(f"  kernel rms {float(a.to(torch.float32).pow(2).mean().sqrt()):.6f}  "
          f"ref rms {float(b.to(torch.float32).pow(2).mean().sqrt()):.6f}")
    if args.mode == "raw":
        passed = r_gr["bit_frac"] == 1.0
    else:
        # The fused HIP butterflies round at each fp16 stage; this explicit reference uses fp32
        # matmuls, so permit their measured fp16-rounding delta, but no larger error.
        passed = r_gr["max_abs_diff"] <= 5e-4 and r_gr["rel_mean_abs_diff"] <= 2e-3
    print("  PASS" if passed else "  FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
