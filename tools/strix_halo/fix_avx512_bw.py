#!/usr/bin/env python3
"""Fix a genuine fork bug blocking MoE CPU offload: has_avx512_bw is never exported.

exllamav3/model/moe_cpu_host.py calls cext/ext.exl3_moe_cpu_has_avx512_bw() in
three places, but the extension exports only:

    exl3_moe_cpu_has_avx2
    exl3_moe_cpu_has_avx512_vnni
    exl3_moe_cpu_has_avx512_vbmi

so --moe_cpu_split / --moe_cpu_offload die with
    AttributeError: module 'exllamav3_ext' has no attribute
                    'exl3_moe_cpu_has_avx512_bw'

The code's own comment states the intent: "has_avx512_bw is true for the bw,
vnni and vbmi tiers alike". vnni and vbmi both imply AVX512-BW, so deriving it
as (vnni or vbmi) is exactly right for every CPU that reports either, and stays
false on an AVX2-only part. Patch it as a module-level helper with getattr so it
keeps working if a later build does export the symbol.
"""
import pathlib
import sys

p = pathlib.Path.home() / "kyojin" / "exllamav3" / "model" / "moe_cpu_host.py"
s = p.read_text()

if "_has_avx512_bw(" in s:
    print("SKIP: already applied")
    sys.exit(0)

# 1. Insert the helper after the imports.
anchor = None
for cand in ("\nTUNING", "\nclass MoeCpuTuning", "\ndef "):
    i = s.find(cand)
    if i > 0:
        anchor = i
        break
if anchor is None:
    print("FAIL: no insertion point found")
    sys.exit(1)

HELPER = '''

def _has_avx512_bw(mod) -> bool:
    """Whether the CPU has the AVX512-BW-or-better expert kernel tiers.

    The extension exports has_avx2 / has_avx512_vnni / has_avx512_vbmi but not
    has_avx512_bw, even though the host code asks for it. Both VNNI and VBMI
    imply AVX512-BW, so either one is sufficient; an AVX2-only CPU stays False.
    """
    fn = getattr(mod, "exl3_moe_cpu_has_avx512_bw", None)
    if fn is not None:
        return bool(fn())
    vnni = getattr(mod, "exl3_moe_cpu_has_avx512_vnni", None)
    vbmi = getattr(mod, "exl3_moe_cpu_has_avx512_vbmi", None)
    return bool((vnni and vnni()) or (vbmi and vbmi()))

'''
s = s[:anchor] + HELPER + s[anchor:]

# 2. Replace the three call sites.
subs = [
    ("cext.exl3_moe_cpu_has_avx512_bw()", "_has_avx512_bw(cext)"),
    ("ext.exl3_moe_cpu_has_avx512_bw()", "_has_avx512_bw(ext)"),
]
total = 0
for old, new in subs:
    n = s.count(old)
    if n:
        s = s.replace(old, new)
        total += n
        print(f"  replaced {n}x: {old} -> {new}")

if total == 0:
    print("FAIL: no call sites replaced")
    sys.exit(1)

p.write_text(s)
print(f"OK: patched {p} ({total} call sites)")
