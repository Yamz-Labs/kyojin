#!/usr/bin/env python3
"""Pad the WMMA LDS staging buffer to remove bank conflicts (EXL3_HIP_STG_PAD).

WHY (static bank analysis, RDNA LDS = 32 banks x 4 bytes, bank = dword_addr % 32):

The warp-private staging buffer is

    __shared__ __half2 hip_mma_stg[HIP_MMA_STG_WARPS][32][4];   // __half2 = 1 dword

so the per-lane stride is 4 dwords. That makes both access patterns conflict:

  writes  stg[lane][slot]           lane 0..31, fixed slot
          dword = lane*4  ->  banks {0,4,8,...,28}
          8 distinct banks for 32 lanes = 4-WAY CONFLICT

  reads   stg[4*(R&7)+j][s]         R = lane&15, fixed j,s   (assemble_*_gfx115)
          base = 4*(R&7) in {0,4,...,28}; dword = base*4
          base*4 mod 32 in {0,16} only
          2 distinct banks for 16 lanes = 8-WAY CONFLICT

Padding the innermost dimension to 5 dwords (gcd(5,32)=1) gives:

  writes  dword = lane*5 -> all 32 banks distinct      => CONFLICT-FREE
  reads   base*5 mod 32 in {0,20,8,28,16,4,24,12}      => 8 banks, 2-way

This targets the measured profile directly: the m=3 GEMV kernel is ~25% of decode
wall at ~30% of roofline with 17% LDS and is stall-bound (not issue- or
bandwidth-bound), and the per-slice s_waitcnt was already ruled out.

Cost: +1 dword per lane per warp. At HIP_MMA_STG_WARPS=16 that is 8192 -> 10240
bytes. Since CFG=2 (the measured-best config) only runs 4 warps, the buffer is
also over-provisioned by 4x; EXL3_HIP_STG_WARPS lets it be sized down so the pad
is LDS-neutral or better.

Build-time A/B, matching this tree's established pattern:

    EXL3_HIP_DEFINES="EXL3_HIP_STG_PAD" pip install -e . --no-build-isolation --no-deps
"""
import pathlib
import re
import sys

ROOT = pathlib.Path.home() / "kyojin"
mma = ROOT / "exllamav3" / "exllamav3_ext" / "hip" / "hip_mma.cuh"
s = mma.read_text()

if "HIP_MMA_STG_SLOTS" in s:
    print("SKIP: already applied")
    sys.exit(0)

OLD_DECL = """#define HIP_MMA_STG_WARPS 16   // EXL3 CFG0 max warps (512 threads); bump if CFG grows
__shared__ __half2 hip_mma_stg[HIP_MMA_STG_WARPS][32][4];"""

NEW_DECL = """#ifndef HIP_MMA_STG_WARPS
#define HIP_MMA_STG_WARPS 16   // EXL3 CFG0 max warps (512 threads); bump if CFG grows
#endif

// LDS bank geometry: RDNA has 32 banks of one dword, bank = dword_address % 32.
// __half2 is one dword, so the innermost extent IS the per-lane bank stride.
//
// With 4 slots the stride is 4 dwords and both access patterns collide:
//   store stg[lane][slot]        -> banks {0,4,...,28}         4-way conflict
//   load  stg[4*(R&7)+j][s]      -> banks {0,16}               8-way conflict
// Padding to 5 (gcd(5,32) == 1) spreads the stores across all 32 banks and cuts
// the loads to 2-way. Costs one dword per lane per warp.
//
// Build with EXL3_HIP_DEFINES="EXL3_HIP_STG_PAD" to enable.
#if defined(EXL3_HIP_STG_PAD)
    #define HIP_MMA_STG_SLOTS 5
#else
    #define HIP_MMA_STG_SLOTS 4
#endif

__shared__ __half2 hip_mma_stg[HIP_MMA_STG_WARPS][32][HIP_MMA_STG_SLOTS];"""

if s.count(OLD_DECL) != 1:
    print(f"FAIL: declaration anchor found {s.count(OLD_DECL)} times")
    sys.exit(1)
s = s.replace(OLD_DECL, NEW_DECL, 1)
print("  OK: declaration now uses HIP_MMA_STG_SLOTS")

# Every consumer takes a row pointer whose type carries the innermost extent.
n = s.count("__half2 (*stg)[4]")
if n == 0:
    print("FAIL: no '__half2 (*stg)[4]' row pointers found")
    sys.exit(1)
s = s.replace("__half2 (*stg)[4]", "__half2 (*stg)[HIP_MMA_STG_SLOTS]")
print(f"  OK: retyped {n} row pointer(s)")

# Anything else that hardcodes the extent would now be wrong; report it.
leftovers = [m.start() for m in re.finditer(r"hip_mma_stg\s*\[[^\]]*\]\s*\[[^\]]*\]\s*\[\s*4\s*\]", s)]
if leftovers:
    print(f"  WARN: {len(leftovers)} residual hardcoded [4] on hip_mma_stg")

mma.write_text(s)
print(f"\nwrote {mma}")
print("slots =", "5 (padded)" if "EXL3_HIP_STG_PAD" else "4")
