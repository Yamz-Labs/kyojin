#!/usr/bin/env python3
"""Widen the C++ is_gfx12_wave32() helper to accept wave32 gfx11.5.

Audited both files that define it -- ZERO matches for wmma, __builtin_amdgcn,
mma_, dp4a, v_sad. They use only __shfl* / __syncwarp / warpSize, which behave
identically on gfx1151 (wave32) and gfx1200/1201. The arch allowlist is a
conservative default, not a hardware requirement.

  routing_std_gfx12.cu : router gate matmul + top-k (0 calls on gfx1151 today,
                         while hgemm takes 2,376 calls/run doing the slow path)
  dsa_topk_gfx12.cu    : sparse-attention top-k

Renames nothing -- keeps the symbol name so both call sites are unaffected.
"""
import pathlib
import sys

ROOT = pathlib.Path.home() / "kyojin"
EXT = ROOT / "exllamav3" / "exllamav3_ext"

OLD = """    const char* arch = prop.gcnArchName;
    const bool gfx12 = (!std::strncmp(arch, "gfx1200", 7) ||
                        !std::strncmp(arch, "gfx1201", 7)) &&
                       (arch[7] == '\\0' || arch[7] == ':');
    return gfx12 && prop.warpSize == WAVE_SIZE;"""

NEW = """    const char* arch = prop.gcnArchName;
    // These kernels use only wave-width primitives (__shfl*, __syncwarp) and no
    // RDNA4-only instruction, so every wave32 gfx11.5 / gfx12 part is valid.
    // gfx1150/1151/1152 = RDNA3.5 (Strix Halo), gfx1200/1201 = RDNA4.
    auto is_arch = [arch](const char* name, size_t n) {
        return !std::strncmp(arch, name, n) && (arch[n] == '\\0' || arch[n] == ':');
    };
    const bool wave32_arch =
        is_arch("gfx1200", 7) || is_arch("gfx1201", 7) ||
        is_arch("gfx1150", 7) || is_arch("gfx1151", 7) || is_arch("gfx1152", 7);
    return wave32_arch && prop.warpSize == WAVE_SIZE;"""

targets = ["routing_std_gfx12.cu", "dsa_topk_gfx12.cu"]
changed = []
for t in targets:
    p = EXT / t
    s = p.read_text()
    if "wave32_arch" in s:
        print(f"  SKIP (already applied): {t}")
        continue
    if s.count(OLD) != 1:
        print(f"  FAIL: {t} -- anchor found {s.count(OLD)} times")
        sys.exit(1)
    p.write_text(s.replace(OLD, NEW, 1))
    print(f"  OK: widened is_gfx12_wave32 in {t}")
    changed.append(t)

# Also relax the now-misleading error strings.
for t, fn in (("routing_std_gfx12.cu", "routing_std_gfx12_bsz1"),
              ("dsa_topk_gfx12.cu", "dsa_topk_gfx12")):
    p = EXT / t
    s = p.read_text()
    old_msg = f'"{fn} requires gfx1200/gfx1201 with wave32"'
    new_msg = f'"{fn} requires a wave32 gfx11.5/gfx12 part"'
    if new_msg in s:
        continue
    if old_msg in s:
        p.write_text(s.replace(old_msg, new_msg))
        print(f"  OK: updated error text in {t}")

print()
print("changed:", changed or "none")
