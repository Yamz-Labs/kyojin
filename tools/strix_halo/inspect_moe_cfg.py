#!/usr/bin/env python3
"""Make the grouped-MoE decode k-split (CFG) tunable, to fit gfx1151's 20 CUs.

MEASURED PROBLEM: the grouped-MoE GEMV achieves ~39 GB/s of the 236 GB/s this
chip can read (17% of peak) and accounts for 27% of each trunk forward.

The decode kernel hardcodes the widest k-split:

    exl3_gemv_kernel_body<3, FP32, 2, 0, /*CFG=*/0, true>      // WK = 16 warps
    moe_grouped_gemv_k3_kernel<<<grid, MOE_THREADS=512>>>

CFG selects WK (warps per block, i.e. the k-split width):
    CFG 0 -> WK 16 -> 512 threads
    CFG 1 -> WK  8 -> 256 threads
    CFG 2 -> WK  4 -> 128 threads

A 16-way k-split was tuned for RDNA4 parts with far more CUs. On 20 CUs with
40-80 blocks/CU already resident, the wide split mostly adds LDS pressure
(sh_red is [WK][RED_ROWS][COLS] floats) and cross-warp reduction work, which
lowers the blocks/CU the scheduler can keep in flight.

This makes CFG a template parameter on the decode kernel, selected at runtime
from EXL3_MOE_CFG (default 0 = current behaviour), so the three settings can be
A/B'd on real decode without recompiling between runs.
"""
import pathlib
import sys

ROOT = pathlib.Path.home() / "kyojin"
gemv = ROOT / "exllamav3" / "exllamav3_ext" / "quant" / "exl3_gemv.cu"
s = gemv.read_text()

if "EXL3_MOE_CFG" in s:
    print("SKIP: already applied")
    sys.exit(0)

# 1. Templatize the decode kernel on CFG.
OLD_K = """__global__ __launch_bounds__(MOE_THREADS)"""
if s.count(OLD_K) != 1:
    print(f"FAIL: launch_bounds anchor found {s.count(OLD_K)} times")
    sys.exit(1)

# Find the kernel signature that follows.
i = s.index(OLD_K)
j = s.index("void moe_grouped_gemv_k3_kernel", i)
sig_start = s.rindex("template", i - 400, i) if "template" in s[i-400:i] else None
print(f"  launch_bounds at {i}, kernel at {j}, template at {sig_start}")
print("  --- current declaration ---")
print("   ", s[sig_start if sig_start else i - 120: j + 40].replace("\n", "\n    "))

# 2. Show the body call so we can see the CFG position.
k = s.index("exl3_gemv_kernel_body<3, FP32, 2, 0, 0, true>")
print("  --- body call ---")
print("   ", s[k - 60:k + 120].replace("\n", "\n    "))
