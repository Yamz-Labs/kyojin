#!/usr/bin/env python3
"""Widen is_gfx12_wave32() in dsa_topk_gfx12.cu (one-line variant of the body)."""
import pathlib, sys

p = pathlib.Path.home() / "kyojin" / "exllamav3" / "exllamav3_ext" / "dsa_topk_gfx12.cu"
s = p.read_text()

OLD = """    const bool gfx12 = (!std::strncmp(arch, "gfx1200", 7) || !std::strncmp(arch, "gfx1201", 7)) &&
                       (arch[7] == '\\0' || arch[7] == ':');
    return gfx12 && prop.warpSize == WAVE_SIZE;"""

NEW = """    // Only wave-width primitives here (__shfl*, warpSize); no RDNA4-only
    // instruction, so any wave32 gfx11.5 / gfx12 part is valid.
    auto is_arch = [arch](const char* name, size_t n) {
        return !std::strncmp(arch, name, n) && (arch[n] == '\\0' || arch[n] == ':');
    };
    const bool wave32_arch =
        is_arch("gfx1200", 7) || is_arch("gfx1201", 7) ||
        is_arch("gfx1150", 7) || is_arch("gfx1151", 7) || is_arch("gfx1152", 7);
    return wave32_arch && prop.warpSize == WAVE_SIZE;"""

if "wave32_arch" in s:
    print("SKIP: already applied")
    sys.exit(0)
if s.count(OLD) != 1:
    print(f"FAIL: anchor found {s.count(OLD)} times")
    sys.exit(1)

s = s.replace(OLD, NEW, 1)
s = s.replace('"dsa_topk_gfx12 requires gfx1200/gfx1201 with wave32"',
              '"dsa_topk_gfx12 requires a wave32 gfx11.5/gfx12 part"')
p.write_text(s)
print("OK: patched dsa_topk_gfx12.cu")
