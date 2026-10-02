#!/usr/bin/env python3
"""Route small multi-row MoE batches through the expert-GROUPED prefill kernel.

MEASURED PROBLEM: during MTP verification the grouped-MoE DECODE kernel issues
rows * top_k independent GEMVs, one per (row, expert) pair, each re-reading that
expert's weights. Instrumenting the real selections shows 31% of those reads are
redundant -- consecutive speculative tokens pick overlapping experts (e.g. 3
rows x top-k 10 = 30 assignments covering only 16-24 unique experts).

The PREFILL path already solves this: it argsorts assignments by expert and
processes each unique expert once for up to MOE_PREFILL_ROWS_PER_CHUNK = 16
rows, so the weight read is amortized across every row that wants that expert.

But the two paths are split at a fixed row count:
    _HIP_GROUPED_MAX_ROWS   = 16      -> decode handles 1..16
    _HIP_PREFILL_MIN_ROWS   = 17      -> prefill handles 17..2048

so a 3-row verification batch NEVER reaches the deduplicating kernel, even
though it would benefit most (31% redundancy at m=3).

This makes the crossover tunable via EXL3_HIP_PREFILL_MIN_ROWS so the grouped
prefill kernel can take small batches. Default is unchanged (17).

Correctness note: the prefill kernel computes the same math (it is the same
exl3_gemv_kernel_body with MMODE 2), just with expert-major ordering and a
scatter-add at the end, so results must match -- verified with eval/ppl.py.
"""
import pathlib
import sys

ROOT = pathlib.Path.home() / "kyojin"
p = ROOT / "exllamav3" / "modules" / "block_sparse_mlp_routing.py"
s = p.read_text()

if "EXL3_HIP_PREFILL_MIN_ROWS" in s:
    print("SKIP: already applied")
    sys.exit(0)

OLD = "_HIP_PREFILL_MIN_ROWS = _HIP_GROUPED_MAX_ROWS + 1"
NEW = '''# Crossover between the per-(row,expert) decode kernel and the expert-GROUPED
# prefill kernel. The prefill kernel sorts assignments by expert and amortizes
# each expert's weight read over up to 16 rows, which is exactly what a
# multi-row MTP verification batch wants: at 3 rows, 31% of the decode path's
# expert reads are duplicates. Lowering this routes small speculative batches
# through the deduplicating kernel. Default 17 keeps the original split.
_HIP_PREFILL_MIN_ROWS = max(
    2, int(os.environ.get("EXL3_HIP_PREFILL_MIN_ROWS", _HIP_GROUPED_MAX_ROWS + 1)))'''

if s.count(OLD) != 1:
    print(f"FAIL: anchor count {s.count(OLD)}")
    sys.exit(1)

s = s.replace(OLD, NEW, 1)

# make sure os is imported
if "\nimport os" not in s and not s.startswith("import os"):
    s = s.replace("import torch", "import os\nimport torch", 1)
    print("  OK: added 'import os'")

p.write_text(s)
print(f"OK: {p} -- _HIP_PREFILL_MIN_ROWS now honours EXL3_HIP_PREFILL_MIN_ROWS")

# The decode path's own cap must not also claim those rows; grouped decode is
# tried FIRST in block_sparse_mlp.py, so cap it below the new crossover.
b = ROOT / "exllamav3" / "modules" / "block_sparse_mlp_routing.py"
s2 = b.read_text()
OLD2 = """def _hip_grouped_rows_eligible(rows: int) -> bool:
    return 1 <= rows <= _HIP_GROUPED_MAX_ROWS"""
NEW2 = """def _hip_grouped_rows_eligible(rows: int) -> bool:
    # Yield to the expert-grouped prefill kernel once it claims this row count,
    # otherwise the per-(row,expert) decode kernel would take it first and
    # re-read duplicate expert weights.
    return 1 <= rows <= min(_HIP_GROUPED_MAX_ROWS, _HIP_PREFILL_MIN_ROWS - 1)"""
if NEW2 in s2:
    print("  SKIP: grouped cap already applied")
elif s2.count(OLD2) == 1:
    b.write_text(s2.replace(OLD2, NEW2, 1))
    print("  OK: grouped decode yields above the crossover")
else:
    print(f"  WARN: grouped cap anchor count {s2.count(OLD2)}")
