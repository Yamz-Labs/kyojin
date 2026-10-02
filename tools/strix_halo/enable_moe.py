#!/usr/bin/env python3
"""Enable the grouped-MoE fast path on gfx11.5 (RDNA3.5 / Strix Halo).

WHY THIS IS SAFE (verified by inspection before changing anything):

exl3_moe_gfx12_k3 and exl3_moe_gfx12_k3_prefill launch only these kernels:

    moe_grouped_gemv_k3_kernel<...>          -> exl3_gemv_kernel_body<3,FP32,2,0,0,true>
    moe_prefill_grouped_gemv_k3_kernel<...>  -> exl3_gemv_kernel_body<3,FP32,2,2,CFG,true>
    moe_had_rows_kernel / moe_prefill_had_rows_kernel
    moe_silu_mul_kernel
    moe_weighted_reduce_kernel
    moe_prefill_metadata_kernel

The two grouped GEMVs are the SAME exl3_gemv_kernel_body already ported to
RDNA3.5 (WMMA family 2) -- MMODE 0 for decode, MMODE 2 for prefill, both of
which have gfx11.5 epilogues. The remaining kernels are plain elementwise /
reduction code: grep for wmma | mma_ | __builtin_amdgcn | dp4a | __shfl inside
each returns ZERO matches, so none of them carry an RDNA4 assumption.

The "gfx12" in the name is therefore historical (that was the only arch with a
WMMA GEMV when it was written), not a hardware requirement. Widen the guard to
"has a WMMA GEMV of any family" and let the shape checks do the real filtering.

Run from the repo root. Idempotent.
"""
import pathlib
import sys

ROOT = pathlib.Path.home() / "kyojin"
changed = []


def edit(path, old, new, desc, count=1):
    p = pathlib.Path(path)
    s = p.read_text()
    if new in s:
        print(f"  SKIP (already applied): {desc}")
        return
    n = s.count(old)
    if n != count:
        print(f"  FAIL: {desc} -- expected {count} anchor(s), found {n}")
        sys.exit(1)
    p.write_text(s.replace(old, new, count))
    print(f"  OK: {desc}")
    changed.append(str(p))


gemv = ROOT / "exllamav3" / "exllamav3_ext" / "quant" / "exl3_gemv.cu"

print("=== C++: allow any WMMA family in the grouped-MoE entry points ===")

edit(gemv,
"""    TORCH_CHECK(exl3_gemv_wmma_family(device) == 1,
                "exl3_moe_gfx12_k3 requires gfx1200/gfx1201");""",
"""    // Despite the name, this path only launches exl3_gemv_kernel_body (ported to
    // every WMMA family) plus elementwise helpers with no arch intrinsics, so any
    // device with a WMMA GEMV can run it.
    TORCH_CHECK(exl3_gemv_wmma_family(device) != 0,
                "exl3_moe_gfx12_k3 requires a WMMA GEMV arch (gfx1200/1201 or gfx1150/1151/1152)");""",
"exl3_moe_gfx12_k3 accepts any WMMA family")

edit(gemv,
"""    TORCH_CHECK(exl3_gemv_wmma_family(device) == 1,
                "exl3_moe_gfx12_k3_prefill requires gfx1200/gfx1201");""",
"""    TORCH_CHECK(exl3_gemv_wmma_family(device) != 0,
                "exl3_moe_gfx12_k3_prefill requires a WMMA GEMV arch (gfx1200/1201 or gfx1150/1151/1152)");""",
"exl3_moe_gfx12_k3_prefill accepts any WMMA family")

print()
print("=== Python: restore the grouped-MoE route for gfx11.5 ===")

bsm = ROOT / "exllamav3" / "modules" / "block_sparse_mlp.py"

edit(bsm,
"""            # exl3_moe_gfx12_k3 is a gfx1200/gfx1201 kernel. exl3_gemv_supported()
            # is TRUE on gfx11.5 as well (the RDNA3.5 WMMA GEMV), so it cannot be
            # used as a "is gfx12" proxy here -- require family 1 explicitly.
            if hasattr(ext, "exl3_gemv_wmma_family"):
                hip_grouped_device = (ext.exl3_gemv_wmma_family(device_index) == 1)
            else:
                hip_grouped_device = ext.exl3_gemv_supported(device_index)""",
"""            # The grouped-MoE kernels launch exl3_gemv_kernel_body (ported to both
            # WMMA families) plus elementwise helpers with no arch intrinsics, so
            # any WMMA GEMV device qualifies. The shape checks below do the real
            # filtering. NOTE: other gfx12-only kernels (routing, hyperconnection
            # fusion) still need an explicit family-1 / arch-string test.
            hip_grouped_device = ext.exl3_gemv_supported(device_index)""",
"hip_grouped_device accepts any WMMA family")

print()
print("=== SUMMARY ===")
for c in sorted(set(changed)):
    print("  modified:", c)
if not changed:
    print("  no changes (all already applied)")
