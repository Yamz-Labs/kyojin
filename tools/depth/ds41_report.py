#!/usr/bin/env python
"""Ideal vs measured DS4.1 depth curves.

  ds41_report.py PROFILE.json [PROFILE2.json ...] [--hw tools/depth/hw.json] [--src2 1] [--src1 1] [--key-bytes 256]

Depth-dependent physics of the stack (everything else is flat by design):
  every index source that scans a whole pool reads nt keys per step (nt = D / ratio, key-bytes each: 256 for fp16 128-dim
  keys): `--src2` ratio-2 sources (3 in the real stack: layers 2, 8, 14; 1 in the depth pack) and `--src1` ratio-1 candidate
  sources (1: layer 20). The consumers (layers 24..36) read the 16K candidate keys only (flat).
  decode  : bytes / bandwidth.
  prefill : a chunk of C rows scores C x nt keys with 2 x 32 x 128 FLOP per pair -> FLOP / gemm peak (the score matrix write,
            C x nt x 4 B, is added when it exceeds nothing: max(compute, bytes) per scan).
Prints, per profile, measured ms, ideal growth over the shallowest depth, and the excess (measured growth - ideal growth).
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("profiles", nargs="+")
ap.add_argument("--hw", default="tools/depth/hw.json")
ap.add_argument("--src2", type=int, default=1)
ap.add_argument("--src1", type=int, default=1)
ap.add_argument("--key-bytes", type=float, default=256.0)
a = ap.parse_args()
hw = json.load(open(a.hw))
bw, fl = hw["bw_gbs"] * 1e9, hw["gemm_tflops"] * 1e12


def ideal_decode_ms(D):
    nt = a.src2 * D / 2 + a.src1 * D
    return nt * a.key_bytes / bw * 1e3


def ideal_prefill_ms(D, C):
    ms = 0.0
    for ratio, n in ((2, a.src2), (1, a.src1)):
        nt = D / ratio
        t = max(2 * C * 32 * 128 * nt / fl, (nt * a.key_bytes + C * nt * 4) / bw)
        ms += n * t * 1e3
    return ms


for path in a.profiles:
    P = json.load(open(path))
    ds = sorted(int(d) for d in P["points"])
    print(f"\n== {path}")
    for kind in ("decode", "prefill"):
        pts = {d: P["points"][str(d)].get(kind) for d in ds if P["points"][str(d)].get(kind)}
        if not pts:
            continue
        d0 = min(pts)
        ideal = (lambda D: ideal_decode_ms(D)) if kind == "decode" else (lambda D: ideal_prefill_ms(D, pts[d0]["chunk"]))
        print(f"{kind}:  depth   measured ms   ideal growth   measured growth   excess ms   (ideal = {ideal(d0)+0:.2f} ms at D0 scan cost)")
        for d in sorted(pts):
            m = pts[d]["ms"]
            g, mg = ideal(d) - ideal(d0), m - pts[d0]["ms"]
            print(f"  {d:>9} {m:12.2f} {g:14.2f} {mg:17.2f} {mg-g:11.2f}")
