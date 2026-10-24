"""Job groups for a row-invariant multi-job decode step.

With several jobs decoding together, one forward carries (jobs x rows) token rows. Most weight kernels pick their
arithmetic by the total row count, so a job's row gets other bits than the same row decoded alone. The kernels that
serve up to ROW_INV_MAX_ROWS rows (one-row GEMV, R-row GEMV, fused MoE, ...) give every row the bits of the one-row
launch whatever the row count. A step with more rows than that is therefore run as several forwards, each one with at
most ROW_INV_MAX_ROWS rows: jobs are packed in order, and a job always stays whole in one group. The attention layers
of a group still run one sequence at a time (Attention.forward, params["row_inv_per_seq"]): the split plan of the paged
decode, the sparse selection and the graph path all follow the batch size and the longest sequence of the batch.

The grouped step is opt-in: models declare caps["row_inv_groups"], and EXL3_ROW_INV_GROUPS=1 (GROUPS["on"], mutable at
run time for one-load A/B) turns it on. It also needs ROW_INV["on"]. Off, all jobs run in one forward as before.
EXL3_ROW_INV_GROUP_ROWS overrides the row cap (1 = one forward per job, the slowest and most obviously exact plan).
"""
import os

from ..util.row_inv import ROW_INV_MAX_ROWS

GROUPS = {"on": os.environ.get("EXL3_ROW_INV_GROUPS", "0") not in ("", "0")}


def group_rows_cap() -> int:
    v = os.environ.get("EXL3_ROW_INV_GROUP_ROWS", "")
    return max(1, int(v)) if v.strip() else ROW_INV_MAX_ROWS


def plan_groups(seq_counts: list[int], width: int, cap: int) -> list[tuple[int, int]]:
    """Consecutive job ranges (j0, j1) of one forward each.

    seq_counts: sequences (batch rows) per job; width: token rows per sequence in this step (1 + drafted tokens);
    cap: most token rows a group may carry. A job whose own rows exceed the cap still forms a group of its own."""
    spans = []
    j0 = 0
    rows = 0
    n = len(seq_counts)
    for j in range(n):
        r = seq_counts[j] * width
        if j > j0 and rows + r > cap:
            spans.append((j0, j))
            j0, rows = j, 0
        rows += r
    if n:
        spans.append((j0, n))
    return spans
