#!/usr/bin/env python3
"""Fix the Python-side MoE gate so the gfx12-only grouped-MoE kernel is not
offered on gfx11.5.

The grouped-MoE route calls ext.exl3_moe_gfx12_k3, which is a gfx1200/gfx1201
kernel. Its Python gate used ext.exl3_gemv_supported() as a proxy for "is
gfx12". Widening exl3_gemv_supported() for the RDNA3.5 WMMA GEMV therefore made
Python offer the MoE kernel on gfx1151, where the C++ guard correctly rejects it
-> RuntimeError during prefill.

Fix: gate the MoE route on exl3_gemv_wmma_family(device) == 1 (gfx12 only),
falling back to the old predicate when the new symbol is absent.
"""
import pathlib, sys

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


OLD = """        hip_grouped_device = False
        if (
            bool(torch.version.hip) and hasattr(ext, "exl3_moe_gfx12_k3") and
            hasattr(ext, "exl3_gemv_supported")
        ):
            device_index = torch.device(self.device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            hip_grouped_device = ext.exl3_gemv_supported(device_index)"""

NEW = """        hip_grouped_device = False
        if (
            bool(torch.version.hip) and hasattr(ext, "exl3_moe_gfx12_k3") and
            hasattr(ext, "exl3_gemv_supported")
        ):
            device_index = torch.device(self.device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            # exl3_moe_gfx12_k3 is a gfx1200/gfx1201 kernel. exl3_gemv_supported()
            # is TRUE on gfx11.5 as well (the RDNA3.5 WMMA GEMV), so it cannot be
            # used as a "is gfx12" proxy here -- require family 1 explicitly.
            if hasattr(ext, "exl3_gemv_wmma_family"):
                hip_grouped_device = (ext.exl3_gemv_wmma_family(device_index) == 1)
            else:
                hip_grouped_device = ext.exl3_gemv_supported(device_index)"""

print("=== block_sparse_mlp.py: gate grouped MoE on gfx12 only ===")
edit(ROOT / "exllamav3" / "modules" / "block_sparse_mlp.py", OLD, NEW,
     "hip_grouped_device requires wmma_family == 1")

# ext_fallbacks.py and block_sparse_mlp_routing.py already hardcode
# ("gfx1200", "gfx1201") so they are correct as-is. Verify that assumption.
print()
print("=== verifying the other gfx12 gates are arch-string based (correct) ===")
for rel in ("exllamav3/ext_fallbacks.py", "exllamav3/modules/block_sparse_mlp_routing.py"):
    p = ROOT / rel
    s = p.read_text()
    if 'arch in ("gfx1200", "gfx1201")' in s:
        print(f"  OK: {rel} uses explicit arch strings")
    else:
        print(f"  NOTE: {rel} -- inspect manually")

print()
print("=== SUMMARY ===")
for c in sorted(set(changed)):
    print("  modified:", c)
if not changed:
    print("  no changes")
