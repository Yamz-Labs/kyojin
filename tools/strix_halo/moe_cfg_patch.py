#!/usr/bin/env python3
"""Make the grouped-MoE decode k-split (CFG) runtime-tunable on gfx1151.

MEASURED PROBLEM: the grouped-MoE GEMV reaches ~39 GB/s of this chip's 236 GB/s
(17% of peak) and is 27% of every trunk forward. The decode kernel hardcodes the
widest k-split, CFG=0 -> WK=16 warps -> 512 threads, which was tuned on RDNA4
parts with many more CUs than gfx1151's 20. Narrower splits cut LDS
(sh_red[WK][RED_ROWS][COLS] floats) and cross-warp reduction work.

    CFG 0 -> WK 16 -> 512 threads   (current)
    CFG 1 -> WK  8 -> 256 threads
    CFG 2 -> WK  4 -> 128 threads

Adds a CFG template parameter plus EXL3_MOE_CFG dispatch (default 0 = unchanged),
so all three can be A/B'd on real decode without a rebuild per setting.
"""
import pathlib
import sys

ROOT = pathlib.Path.home() / "kyojin"
gemv = ROOT / "exllamav3" / "exllamav3_ext" / "quant" / "exl3_gemv.cu"
s = gemv.read_text()

if "EXL3_MOE_CFG" in s:
    print("SKIP: already applied")
    sys.exit(0)

# --- 1. templatize the kernel on CFG and size the block from it ---
OLD = """template <bool FP32, bool TWO_PROJECTIONS>
__global__ __launch_bounds__(MOE_THREADS)
void moe_grouped_gemv_k3_kernel"""
NEW = """// MOE_CFG selects the k-split width WK (warps per block): 0 -> 16, 1 -> 8, 2 -> 4.
// gfx1151 has only 20 CUs, where the widest split adds LDS pressure and cross-warp
// reduction work without adding useful parallelism, so it is tunable at runtime.
template <bool FP32, bool TWO_PROJECTIONS, int MOE_CFG = 0>
__global__ __launch_bounds__(MOE_CFG == 0 ? 512 : MOE_CFG == 1 ? 256 : 128)
void moe_grouped_gemv_k3_kernel"""
if s.count(OLD) != 1:
    print(f"FAIL: kernel decl anchor count {s.count(OLD)}")
    sys.exit(1)
s = s.replace(OLD, NEW, 1)
print("  OK: kernel templated on MOE_CFG")

# --- 2. forward CFG into the shared kernel body ---
OLD_BODY = "exl3_gemv_kernel_body<3, FP32, 2, 0, 0, true>"
NEW_BODY = "exl3_gemv_kernel_body<3, FP32, 2, 0, MOE_CFG, true>"
if s.count(OLD_BODY) != 1:
    print(f"FAIL: body anchor count {s.count(OLD_BODY)}")
    sys.exit(1)
s = s.replace(OLD_BODY, NEW_BODY, 1)
print("  OK: body call forwards MOE_CFG")

# --- 3. runtime selector ---
HELPER = """
// Runtime k-split selection for the grouped-MoE decode GEMV. Default 0 keeps the
// original 512-thread/WK=16 shape; 1 and 2 narrow it for small-CU parts.
static inline int moe_decode_cfg()
{
    static const int cfg = [] {
        const char* e = getenv("EXL3_MOE_CFG");
        int v = e ? atoi(e) : 0;
        return (v < 0 || v > 2) ? 0 : v;
    }();
    return cfg;
}

static inline int moe_decode_threads(int cfg)
{
    return cfg == 0 ? 512 : cfg == 1 ? 256 : 128;
}

"""
anchor = "constexpr int MOE_THREADS = 512;"
if s.count(anchor) != 1:
    print(f"FAIL: MOE_THREADS anchor count {s.count(anchor)}")
    sys.exit(1)
s = s.replace(anchor, anchor + "\n" + HELPER, 1)
print("  OK: inserted moe_decode_cfg() / moe_decode_threads()")

# --- 4. dispatch both launch sites over CFG ---
OLD_GU = """    moe_grouped_gemv_k3_kernel<false, true><<<gu_grid, MOE_THREADS, 0, stream>>>"""
OLD_DOWN = """    moe_grouped_gemv_k3_kernel<true, false><<<down_grid, MOE_THREADS, 0, stream>>>"""

for old, fp32, twop, gridname in (
    (OLD_GU, "false", "true", "gu_grid"),
    (OLD_DOWN, "true", "false", "down_grid"),
):
    if s.count(old) != 1:
        print(f"FAIL: launch anchor count {s.count(old)} for {gridname}")
        sys.exit(1)
    # capture the argument list that follows the launch
    i = s.index(old)
    j = s.index(";", i)
    args = s[i + len(old):j]
    new = (
        f"    switch (moe_decode_cfg())\n"
        f"    {{\n"
        f"    case 1:\n"
        f"        moe_grouped_gemv_k3_kernel<{fp32}, {twop}, 1>"
        f"<<<{gridname}, moe_decode_threads(1), 0, stream>>>{args};\n"
        f"        break;\n"
        f"    case 2:\n"
        f"        moe_grouped_gemv_k3_kernel<{fp32}, {twop}, 2>"
        f"<<<{gridname}, moe_decode_threads(2), 0, stream>>>{args};\n"
        f"        break;\n"
        f"    default:\n"
        f"        moe_grouped_gemv_k3_kernel<{fp32}, {twop}, 0>"
        f"<<<{gridname}, moe_decode_threads(0), 0, stream>>>{args};\n"
        f"        break;\n"
        f"    }}"
    )
    s = s[:i] + new + s[j + 1:]
    print(f"  OK: dispatched {gridname} over MOE_CFG")

# need <cstdlib> for getenv/atoi
if "#include <cstdlib>" not in s:
    s = s.replace("#include <map>", "#include <map>\n#include <cstdlib>", 1)
    print("  OK: added <cstdlib>")

gemv.write_text(s)
print("\nwrote", gemv)
