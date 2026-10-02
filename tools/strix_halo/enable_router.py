#!/usr/bin/env python3
"""Enable the fused gfx12 router + dsa_topk on gfx11.5 (RDNA3.5).

Audit first (same method that validated the grouped-MoE widening):
  exllamav3_ext/routing_std_gfx12.cu  -- arch-specific instruction count
      wmma              0
      __builtin_amdgcn  0
      mma_              0
      dp4a              0
      v_sad             0
  present: __shfl_down (4), __syncwarp (1), warpSize (1)

__shfl_down / __syncwarp / warpSize are wave-width primitives, correct on
gfx1151 (wave32, same as gfx1200/1201). There is no RDNA4-only instruction in
the file, so the gfx1200/1201 allowlist is a conservative default rather than a
hardware requirement -- exactly like the grouped-MoE gate.

Profiling motivation: routing_std_gfx12_bsz1 recorded ZERO calls during decode
while hgemm recorded 2,376 (74 per token), i.e. every router gate matmul is
going the slow generic route.

Three gates to widen:
  1. exllamav3_ext/routing_std_gfx12.cu  -- the C++ TORCH_CHECK
  2. exllamav3/modules/block_sparse_mlp_routing.py -- _hip_router_device_supported
  3. exllamav3/ext_fallbacks.py -- the dsa_topk device check

Run from the repo root. Idempotent.
"""
import pathlib
import sys

ROOT = pathlib.Path.home() / "kyojin"
changed = []

WAVE32_ARCHS = '("gfx1200", "gfx1201", "gfx1150", "gfx1151", "gfx1152")'


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


print("=== 1. C++: routing_std_gfx12.cu arch guard ===")
rcu = ROOT / "exllamav3" / "exllamav3_ext" / "routing_std_gfx12.cu"
s = rcu.read_text()
# find the guard line
import re
m = re.search(r'([ \t]*)TORCH_CHECK\(([^;]*?)"routing_std_gfx12_bsz1 requires gfx1200/gfx1201 with wave32"\);',
              s, re.S)
if "gfx1151" in s:
    print("  SKIP (already applied): C++ arch guard")
elif m:
    whole = m.group(0)
    indent = m.group(1)
    new_block = (
        f'{indent}// The kernel uses only wave-width primitives (__shfl_down, __syncwarp)\n'
        f'{indent}// and no RDNA4-only instruction, so every wave32 RDNA3.5/RDNA4 part works.\n'
        f'{indent}{{\n'
        f'{indent}    const std::string a(arch ? arch : "");\n'
        f'{indent}    const bool wave32_wmma_arch =\n'
        f'{indent}        a.rfind("gfx1200", 0) == 0 || a.rfind("gfx1201", 0) == 0 ||\n'
        f'{indent}        a.rfind("gfx1150", 0) == 0 || a.rfind("gfx1151", 0) == 0 ||\n'
        f'{indent}        a.rfind("gfx1152", 0) == 0;\n'
        f'{indent}    TORCH_CHECK(wave32_wmma_arch && props->warpSize == 32,\n'
        f'{indent}                "routing_std_gfx12_bsz1 requires a wave32 gfx11.5/gfx12 part, got ", a);\n'
        f'{indent}}}'
    )
    print("  found guard; inspecting its surroundings")
    start = max(0, m.start() - 600)
    print("  --- context ---")
    for line in s[start:m.end()].splitlines()[-14:]:
        print("   |", line)
else:
    print("  NOTE: guard pattern not matched; will print the region for manual edit")
    i = s.find("requires gfx1200/gfx1201 with wave32")
    print(s[max(0, i - 700):i + 120])

print()
print("=== 2. Python: _hip_router_device_supported ===")
rpy = ROOT / "exllamav3" / "modules" / "block_sparse_mlp_routing.py"
edit(rpy,
'    return arch in ("gfx1200", "gfx1201") and getattr(props, "warp_size", 0) == 32',
'    # routing_std_gfx12 uses only wave-width primitives (__shfl_down/__syncwarp),\n'
'    # no RDNA4-only instruction, so any wave32 gfx11.5/gfx12 part qualifies.\n'
f'    return arch in {WAVE32_ARCHS} and getattr(props, "warp_size", 0) == 32',
"router device check accepts wave32 gfx11.5")

print()
print("=== 3. Python: ext_fallbacks dsa_topk device check ===")
efb = ROOT / "exllamav3" / "ext_fallbacks.py"
edit(efb,
'    return arch in ("gfx1200", "gfx1201") and getattr(props, "warp_size", 0) == 32',
f'    return arch in {WAVE32_ARCHS} and getattr(props, "warp_size", 0) == 32',
"dsa_topk device check accepts wave32 gfx11.5")

print()
print("=== SUMMARY ===")
for c in sorted(set(changed)):
    print("  modified:", c)
if not changed:
    print("  no python changes")
