#!/usr/bin/env python3
"""Routed-expert decode GEMV microbenchmark on gfx1151: GB/s per trellis bitrate.

Synthetic trellis bytes (no model load, no quantization step): E experts x 3 matrices at
the requested (hidden, intermediate) shape, one ext.exl3_moe_gfx12_k3 call per timed
iteration (the whole routed decode body: gate+up GEMVs, activation, down GEMV, Hadamards,
weighted reduce). Every assignment routes to a DISTINCT expert, so the bytes moved are
exactly assignments * 3 * hidden*intermediate*K/8.

  python tools/strix_halo/k2_micro.py --shape ds --k 2 3 4 --rows 1 2 3 4
  python tools/strix_halo/k2_micro.py --dump scratch/out_k2.pt   (parity fixture)

Prints us/call and GB/s per (shape, K, rows) plus the machine's read floor measured in the
same process, so the numbers are comparable across days and across lanes.
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import exllamav3.ext as extmod
ext = extmod.exllamav3_ext

# hidden, intermediate, top-k, label
SHAPES = {
    "ds":  (5120, 2304, 6,  "DeepSeek-V4.1-Flash experts (hidden 5120, inter 2304, top6)"),
    "qwen": (2560,  640, 10, "Qwen3.8-Flash-Next experts (hidden 2560, inter 640, top10)"),
    "qwen4": (2560,  768, 8,  "Qwen3.8-Flash-Next experts (hidden 2560, inter 768, top8)"),
}


def rand_bytes(n, dev, seed):
    """Arbitrary trellis bytes. A 1 MiB random tile is repeated, which is far cheaper than
    randint() on hundreds of MB and covers every bit pattern the decoder can see."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tile = torch.randint(0, 256, (1 << 20,), dtype=torch.uint8, generator=g)
    reps = (n + tile.numel() - 1) // tile.numel()
    out = tile.repeat(reps)[:n]
    return out.to(dev)


class Experts:
    """E experts x {gate, up: [inter, hidden], down: [hidden, inter]} as raw trellis bytes."""

    def __init__(self, dev, E, hidden, inter, K, seed=1234):
        self.E, self.hidden, self.inter, self.K = E, hidden, inter, K
        self.gu_bytes = hidden * inter * K // 8
        self.dn_bytes = self.gu_bytes
        self.gu = rand_bytes(E * self.gu_bytes, dev, seed)
        self.dn = rand_bytes(E * self.dn_bytes, dev, seed + 1)
        # gate and up read the same allocation in this bench (each still costs a full read,
        # so the byte count below is unchanged and both GEMVs see realistic data)
        self.ptrs_trellis = tuple(
            torch.tensor([t.data_ptr() + i * self.gu_bytes for i in range(E)],
                         dtype=torch.long, device=dev)
            for t in (self.gu, self.dn))
        # suh/svh per expert: gate/up scale hidden in / inter out, down is transposed
        # (suh: inter in, svh: hidden out). One contiguous row per expert so every
        # table entry points at a distinct, 16-byte-aligned vector.
        suh_g = torch.ones((E, hidden), dtype=torch.half, device=dev)
        suh_u = torch.ones((E, hidden), dtype=torch.half, device=dev)
        suh_d = torch.ones((E, inter), dtype=torch.half, device=dev)
        svh_g = torch.ones((E, inter), dtype=torch.half, device=dev)
        svh_u = torch.ones((E, inter), dtype=torch.half, device=dev)
        svh_d = torch.ones((E, hidden), dtype=torch.half, device=dev)
        self.ptrs_suh = tuple(
            torch.tensor([m[e].data_ptr() for e in range(E)], dtype=torch.long, device=dev)
            for m in (suh_g, suh_u, suh_d))
        self.ptrs_svh = tuple(
            torch.tensor([m[e].data_ptr() for e in range(E)], dtype=torch.long, device=dev)
            for m in (svh_g, svh_u, svh_d))
        self.keep = (suh_g, suh_u, suh_d, svh_g, svh_u, svh_d)


def read_floor(dev, mib=(32, 64, 128)):
    out = []
    for m in mib:
        n = m * (1 << 20) // 2
        x = torch.empty(n, dtype=torch.float16, device=dev).fill_(1.0)
        for _ in range(2):
            x.sum()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(5):
            x.sum()
        e1.record()
        torch.cuda.synchronize()
        gbs = x.numel() * 2 / (e0.elapsed_time(e1) / 5 * 1e-3) / 1e9
        out.append((m, gbs))
        del x
        torch.cuda.empty_cache()
    return out


def run(args):
    dev = torch.device("cuda:0")
    hidden, inter, top_k, label = SHAPES[args.shape]
    if args.topk:
        top_k = args.topk
    print(f"shape {args.shape}: {label} [effective top_k={top_k}]")
    for m, gbs in read_floor(dev):
        print(f"  read floor  {m:4d} MiB: {gbs:7.1f} GB/s")
    junk = torch.empty(args.junk_mib << 20, dtype=torch.uint8, device=dev)

    for K in args.k:
        rows_list = [r for r in args.rows if r * top_k <= args.experts]
        E = args.experts
        ex = Experts(dev, E, hidden, inter, K)
        gt = ex.ptrs_trellis[0]
        ut = ex.ptrs_trellis[0]
        dt = ex.ptrs_trellis[1]
        for rows in rows_list:
            n_asg = rows * top_k
            # Fixed seed per (K, rows): every build under test sees bit-identical inputs,
            # so --dump fixtures are comparable across .so files (base vs patched).
            torch.manual_seed(args.seed + K * 1000 + rows)
            A = (torch.randn((rows, hidden), dtype=torch.half, device=dev) * 0.05)
            sel = torch.arange(n_asg, dtype=torch.long, device=dev).view(rows, top_k)
            wts = torch.full((rows, top_k), 1.0 / top_k, dtype=torch.half, device=dev)
            out = torch.zeros((rows, hidden), dtype=torch.float32, device=dev)
            gu_had = torch.zeros((2 * n_asg, hidden), dtype=torch.half, device=dev)
            gu_out = torch.zeros((2 * n_asg, inter), dtype=torch.half, device=dev)
            dn_had = torch.zeros((n_asg, inter), dtype=torch.half, device=dev)
            dn_out = torch.zeros((n_asg, hidden), dtype=torch.float32, device=dev)

            def call():
                ext.exl3_moe_gfx12_k3(
                    A, out, sel, wts,
                    gt, ex.ptrs_suh[0], ex.ptrs_svh[0],
                    ut, ex.ptrs_suh[1], ex.ptrs_svh[1],
                    dt, ex.ptrs_suh[2], ex.ptrs_svh[2],
                    gu_had, gu_out, dn_had, dn_out, float(K), args.dedup, float(args.act_limit))

            for _ in range(5):
                call()
            torch.cuda.synchronize()
            ok = bool(torch.isfinite(out).all())
            if not ok:
                print("  NON-FINITE OUTPUT")
            if args.parity and K in (1, 2, 3, 4):
                ref = out.clone()
                gref = gu_out.clone()
                dref = dn_out.clone()
                ext.exl3_moe_valu_override(1)
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                bits_out = int((out.view(torch.int32) != ref.view(torch.int32)).sum())
                bits_gu = int((gu_out.view(torch.int16) != gref.view(torch.int16)).sum())
                bits_dn = int((dn_out.view(torch.int32) != dref.view(torch.int32)).sum())
                mx = float((out - ref).abs().max())
                print(f"  K={K} VALU-vs-WMMA parity: out {bits_out}/{out.numel()} dwords differ "
                      f"(max |d| {mx:.3e}), gu {bits_gu}/{gu_out.numel()}, dn {bits_dn}/{dn_out.numel()}"
                      f"  -> {'BITWISE' if bits_out == 0 and bits_gu == 0 and bits_dn == 0 else 'MISMATCH'}")
                ext.exl3_moe_valu_override(0)
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
            if args.dump and rows == rows_list[-1]:
                torch.save({"out": out.cpu(), "dn": dn_out.cpu(), "gu": gu_out.cpu(),
                            "sel": sel.cpu(), "K": K, "shape": args.shape, "rows": rows},
                           f"{args.dump}_k{K}_{args.shape}_{rows}.pt")
            iters = args.iters
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            tot = 0.0
            for _ in range(args.reps):
                junk.zero_()
                e0.record()
                for _ in range(iters):
                    call()
                e1.record()
                torch.cuda.synchronize()
                tot += e0.elapsed_time(e1) / iters * 1e3
            us = tot / args.reps
            byts = n_asg * 3 * hidden * inter * K / 8
            gbs = byts / (us * 1e-6) / 1e9
            print(f"  K={K} bpw  rows={rows} asg={n_asg:3d}  {us:9.1f} us   "
                  f"{byts/2**20:7.1f} MiB  {gbs:7.1f} GB/s")
            del A, out, gu_had, gu_out, dn_had, dn_out
        del ex
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="ds", choices=sorted(SHAPES))
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--topk", type=int, default=0,
                    help="override the shape's top-k (the baseline .so only accepts 8/10)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--junk-mib", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1234,
                    help="base RNG seed; A is seeded per (K, rows) so dumps compare across builds")
    ap.add_argument("--dedup", type=int, default=0)
    ap.add_argument("--act-limit", type=float, default=0.0,
                    help="swiglu limit applied in moe_silu_mul_kernel (DeepSeek-V4.1: 10.0, 0 = off)")
    ap.add_argument("--parity", action="store_true",
                    help="bitwise WMMA-vs-VALU check per K in {2,3,4} (the VALU path covers those)")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()
    print(f"EXL3_MOE_CFG={os.environ.get('EXL3_MOE_CFG', '0 (default)')} "
          f"EXL3_MOE_VALU={os.environ.get('EXL3_MOE_VALU', 'unset (K2 VALU default on)')} "
          f"dedup={args.dedup} iters={args.iters} reps={args.reps}")
    run(args)


if __name__ == "__main__":
    main()