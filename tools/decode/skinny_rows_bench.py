#!/usr/bin/env python3
"""ROCm skinny fp16 hgemm at 1..32 rows (ext.hgemm, N <= 128).
check: row j of the R-row product == the same row alone (R = 1), bitwise, for fp16 and fp32 outputs.
time: us per call for ext.hgemm vs torch.matmul (hipBLASLt) at the same shape. Needs EXL3_ROOT=<worktree>."""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="1,2,4,8,9,10,12,13,16,17,20,24,25,28,32")
ap.add_argument("--shapes", default="4096x128,4096x64,2048x128,2560x32,6144x128,4096x16")
ap.add_argument("--reps", type=int, default=100)
a = ap.parse_args()
sys.path.insert(0, os.environ["EXL3_ROOT"])
import torch
from exllamav3.ext import exllamav3_ext as ext
dev = torch.device("cuda:0")
g = torch.Generator(device=dev).manual_seed(7)
rows = [int(r) for r in a.rows.split(",")]
bad = 0
for sh in a.shapes.split(","):
    K, N = (int(v) for v in sh.split("x"))
    w = (torch.randn(K, N, device=dev, generator=g) * 0.05).half()
    x = torch.randn(32, K, device=dev, generator=g).half()
    for odt in (torch.half, torch.float):
        solo = []
        for j in range(32):
            y = torch.empty(1, N, dtype=odt, device=dev); ext.hgemm(x[j:j + 1].contiguous(), w, y); solo.append(y)
        solo = torch.cat(solo)
        for R in rows:
            y = torch.empty(R, N, dtype=odt, device=dev); ext.hgemm(x[:R].contiguous(), w, y)
            torch.cuda.synchronize()
            ok = torch.equal(y, solo[:R])
            if not ok:
                bad += 1
                print(f"MISMATCH K={K} N={N} out={odt} R={R} rows_bad={(y != solo[:R]).any(1).nonzero().flatten().tolist()}", flush=True)
    for R in rows:
        xr = x[:R].contiguous(); y = torch.empty(R, N, dtype=torch.half, device=dev)
        def t(fn):
            for _ in range(10): fn()
            torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(a.reps): fn()
            e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) * 1000 / a.reps
        us_s = t(lambda: ext.hgemm(xr, w, y)); us_m = t(lambda: torch.matmul(xr, w, out=y))
        print(f"K={K} N={N} R={R:2d} skinny {us_s:7.1f} us  matmul {us_m:7.1f} us", flush=True)
print("MISMATCHES", bad, "OK" if bad == 0 else "FAIL", flush=True)
