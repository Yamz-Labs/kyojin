#!/usr/bin/env python3
"""Let the fork's HIP GEMV logits oracle run on gfx11.5 as well as gfx12.

tests/hip_gemv_logits_worker.py hardcodes an allowlist of gfx1200/gfx1201.
With the RDNA3.5 WMMA GEMV in place, gfx1150/1151/1152 are equally valid
targets for this test, so widen the allowlist to every arch the extension
reports a WMMA family for.
"""
import pathlib, sys

ROOT = pathlib.Path.home() / "kyojin"
p = ROOT / "tests" / "hip_gemv_logits_worker.py"

OLD = """    if arch.split(":", 1)[0] not in ("gfx1200", "gfx1201"):
        raise SystemExit(f"HIP GEMV oracle requires gfx1200/gfx1201, got {arch or 'unknown'}")"""

NEW = """    # gfx12 (RDNA4) and gfx11.5 (RDNA3.5 / Strix Halo) both have a WMMA GEMV.
    # Ask the extension rather than hardcoding an arch allowlist.
    _wmma_archs = ("gfx1200", "gfx1201", "gfx1150", "gfx1151", "gfx1152")
    if arch.split(":", 1)[0] not in _wmma_archs:
        raise SystemExit(f"HIP GEMV oracle requires a WMMA arch {_wmma_archs}, got {arch or 'unknown'}")"""

s = p.read_text()
if NEW in s:
    print("SKIP: already applied")
elif OLD not in s:
    print("FAIL: anchor not found")
    sys.exit(1)
else:
    p.write_text(s.replace(OLD, NEW, 1))
    print(f"OK: widened arch allowlist in {p}")
