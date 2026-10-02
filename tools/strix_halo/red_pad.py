#!/usr/bin/env python3
"""Pad sh_red's innermost extent to break the cross-warp reduce bank conflict.

Companion to EXL3_HIP_STG_PAD, same bug class, different buffer.

    __shared__ float sh_red[WK][RED_ROWS][COLS];      // float = 1 dword

With CFG=2 (the measured-best MoE config): WK=4, WNT=4, COLS = WNT*16 = 64,
RED_ROWS = 8 (MMODE 2). The cross-warp reduction at the end of the kernel is

    for (j = 0; j < WK; ++j) sum += sh_red[j][r][c];      // fixed r, c

whose per-j stride is RED_ROWS*COLS = 8*64 = 512 dwords. 512 % 32 == 0, so every
warp's slice lands on the SAME bank:

    COLS=64 : 1 bank for 4 accesses  -> 4-WAY CONFLICT
    COLS=65 : 4 banks                -> conflict-free

The write path sh_red[w][2r+h][t*16+c] stays 2-way either way (only 16 distinct
c values across 32 lanes, two half-waves), which is inherent to the fragment
layout and not what this fixes.

LDS cost at CFG=2: 4*8*64*4 = 8192 B -> 4*8*65*4 = 8320 B (+128 B).
At CFG=0 (WK=16, WNT=2, COLS=32, RED_ROWS=8): 16384 -> 16896 B (+512 B).

Gated behind EXL3_HIP_RED_PAD; default keeps the original extent. Build with:

    EXL3_HIP_DEFINES="EXL3_HIP_STG_PAD EXL3_HIP_RED_PAD" \
        pip install -e . --no-build-isolation --no-deps
"""
import pathlib
import sys

ROOT = pathlib.Path.home() / "kyojin"
k = ROOT / "exllamav3" / "exllamav3_ext" / "quant" / "exl3_gemv_kernel.cuh"
s = k.read_text()

if "EXL3_HIP_RED_PAD" in s:
    print("SKIP: already applied")
    sys.exit(0)

OLD = "    __shared__ float sh_red[WK][RED_ROWS][COLS];"
if s.count(OLD) != 1:
    print(f"FAIL: sh_red decl anchor found {s.count(OLD)} times")
    sys.exit(1)

NEW = """    // LDS banks: 32 x 1 dword, bank = dword_addr % 32. float is one dword, so the
    // innermost extent is the stride between consecutive rows -- and the per-warp
    // stride for the cross-warp reduce below is RED_ROWS*COLS. With COLS a multiple
    // of 32 that product is too, so every warp's slice shares one bank:
    //
    //   for (j = 0; j < WK; ++j) sum += sh_red[j][r][c];   // fixed r, c
    //
    //   COLS=64 (CFG2): 512-dword stride -> 1 bank for WK accesses  -> WK-way
    //   COLS=65       : 520-dword stride -> WK distinct banks       -> clean
    //
    // Padding by one dword costs +128 B at CFG=2 (+512 B at CFG=0).
    // The store path is 2-way regardless (16 distinct c over 32 lanes) -- inherent
    // to the fragment layout, not addressed here.
#if defined(EXL3_HIP_RED_PAD) && defined(USE_ROCM)
    #define EXL3_RED_COLS (COLS + 1)
#else
    #define EXL3_RED_COLS COLS
#endif
    __shared__ float sh_red[WK][RED_ROWS][EXL3_RED_COLS];"""

s = s.replace(OLD, NEW, 1)
print("  OK: sh_red declared with EXL3_RED_COLS")

# The macro is defined inside a template function body, so it must be undef'd
# before the function ends or the next instantiation redefines it.
# Find the end of this function and drop an #undef just before the closing brace
# of the translation unit's last use instead: simplest correct approach is to
# #undef immediately after the final sh_red read.
LAST_READ = "                    sum += sh_red[j][r][c];"
n = s.count(LAST_READ)
print(f"  note: {n} reduce-loop read site(s) (all use the same macro)")

# Undef at the very end of the file to keep it a single definition per TU.
if not s.rstrip().endswith("#undef EXL3_RED_COLS"):
    s = s.rstrip() + "\n\n#undef EXL3_RED_COLS\n"
    print("  OK: #undef EXL3_RED_COLS at end of header")

k.write_text(s)
print(f"\nwrote {k}")
