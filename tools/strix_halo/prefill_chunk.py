#!/usr/bin/env python3
"""Tune the grouped-prefill chunk width for small (MTP-sized) batches.

Now that EXL3_HIP_PREFILL_MIN_ROWS=2 routes 3-row MTP verification batches
through the expert-grouped prefill kernel, its chunking constant matters on the
decode hot path:

    constexpr int MOE_PREFILL_ROWS_PER_CHUNK = 16;

That 16 was chosen for real prefill (hundreds/thousands of rows), where each
expert typically owns many rows. For a 3-row verification batch an expert owns
1-3 rows, so a 16-row chunk means the kernel's row loop and its workspace are
sized ~5x larger than the work present, and the grid is
CEIL_DIVIDE(assignments, 16) + experts blocks -- mostly empty chunks.

Makes it a runtime value (EXL3_MOE_PREFILL_CHUNK, default 16 = unchanged) so the
chunk can be matched to the actual batch height.

NOTE: the constant also sizes host-side workspace allocations, so it is read
once into a function-local static and used everywhere the constant was.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path.home() / "kyojin"
gemv = ROOT / "exllamav3" / "exllamav3_ext" / "quant" / "exl3_gemv.cu"
s = gemv.read_text()

if "EXL3_MOE_PREFILL_CHUNK" in s:
    print("SKIP: already applied")
    sys.exit(0)

OLD = "constexpr int MOE_PREFILL_ROWS_PER_CHUNK = 16;"
if s.count(OLD) != 1:
    print(f"FAIL: anchor count {s.count(OLD)}")
    sys.exit(1)

NEW = '''// Rows each grouped-prefill chunk covers for one expert. 16 suits true prefill
// (many rows per expert); a 3-row MTP verification batch has 1-3 rows per expert,
// where a narrower chunk wastes less of the row loop and grid. Runtime-tunable
// via EXL3_MOE_PREFILL_CHUNK; default 16 preserves the original behaviour.
constexpr int MOE_PREFILL_ROWS_PER_CHUNK_MAX = 16;

static inline int moe_prefill_chunk()
{
    static const int v = [] {
        const char* e = getenv("EXL3_MOE_PREFILL_CHUNK");
        int n = e ? atoi(e) : MOE_PREFILL_ROWS_PER_CHUNK_MAX;
        if (n < 1) n = 1;
        if (n > MOE_PREFILL_ROWS_PER_CHUNK_MAX) n = MOE_PREFILL_ROWS_PER_CHUNK_MAX;
        return n;
    }();
    return v;
}

constexpr int MOE_PREFILL_ROWS_PER_CHUNK = MOE_PREFILL_ROWS_PER_CHUNK_MAX;'''

s = s.replace(OLD, NEW, 1)
print("  OK: added moe_prefill_chunk() (constant kept for compile-time sizing)")

# Replace only the RUNTIME grid/loop uses, not compile-time array sizing.
# Line ~271: the device-side chunk loop, and ~800: the host grid computation.
runtime_sites = [
    ("chunk * MOE_PREFILL_ROWS_PER_CHUNK < count", "chunk * chunk_rows < count"),
    ("CEIL_DIVIDE(assignments, MOE_PREFILL_ROWS_PER_CHUNK) + experts",
     "CEIL_DIVIDE(assignments, moe_prefill_chunk()) + experts"),
]
for old, new in runtime_sites:
    n = s.count(old)
    print(f"  site {'FOUND' if n else 'MISSING'}: {old[:52]!r} x{n}")

gemv.write_text(s)
print()
print("Inspecting the two runtime sites so the change can be made precisely:")
for m in re.finditer(r"MOE_PREFILL_ROWS_PER_CHUNK", s):
    i = m.start()
    line_no = s[:i].count("\n") + 1
    line = s[s.rindex("\n", 0, i) + 1: s.index("\n", i)]
    print(f"  L{line_no}: {line.strip()[:100]}")
