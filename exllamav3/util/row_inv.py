"""Row invariance of multi-row verify forwards (MTP / DFlash).

ROW_INV["on"] (default on, EXL3_ROW_INV=0 turns it off, mutable at run time for one-load A/B):
every row of a verify forward, for any number of jobs decoded together, gets the same bits as the
same row would get when its job is decoded alone. Callers that bake the choice into a captured graph
must purge the graphs when the flag flips (block_graph.purge()).
"""
import os

import torch

ROW_INV = {"on": os.environ.get("EXL3_ROW_INV", "1") not in ("", "0")}
# Rows a verify forward may have before the per-row exact paths give way to the batched ones. ROCm (exl3_dec.cu, gfx11.5): the
# R-row GEMV, the union MoE and the row loops carry up to 32 rows, every row with the bits of the one-row launch (rows 9..32 are
# row chunks of the same arithmetic); other builds keep 8. EXL3_ROW_INV_MAX_ROWS overrides (8 restores the previous ROCm cap).
ROW_INV_MAX_ROWS = max(1, int(os.environ.get("EXL3_ROW_INV_MAX_ROWS", "32" if torch.version.hip else "8")))


# GGUF / foreign-quant body (modules/foreign_linear.py): rows 2..GGUF_ROW_INV_MAX_ROWS of one forward (verify rows, a few sessions) run the same
# kernel arithmetic as the one-row step, so a greedy verify matches the plain step bit for bit. Off by default (the few-row int8 MMA kernel
# and the 5..8 column mmvq launch are faster); serve.py --gguf-mtp turns it on. Mutable at run time for one-load A/B (purge the graphs).
GGUF_ROW_INV = {"on": os.environ.get("EXL3_GGUF_ROW_INV", "0") not in ("", "0")}
GGUF_ROW_INV_MAX_ROWS = ROW_INV_MAX_ROWS
# EXL3_GGUF_RI_PARTS (A/B bisect only): comma list of the row-invariant parts to keep, from fl (ForeignLinear), fp16 (dense fp16 projections),
# router (per-row routing), moe (BlockSparseMLP row loop), attn (per-row verify attention). Default: all.
_RI_PARTS = set(os.environ.get("EXL3_GGUF_RI_PARTS", "fl,fp16,router,moe,attn").split(","))


# The recurrent GDN kernel keeps the state of a multi-row call in registers in fp32 between rows (EXL3_GDN_RS_REG_GDN=1), the one-row step
# reloads it rounded to bf16 (EXL3_GDN_STATE_BF16): rows 2..R of a verify step then differ from the plain steps. The reload path
# rounds like the plain step, so row-invariant mode defaults to it (the caller's environment wins).
if GGUF_ROW_INV["on"]:
    os.environ.setdefault("EXL3_GDN_RS_REG_GDN", "0")


def gguf_ri(part: str) -> bool:
    return GGUF_ROW_INV["on"] and part in _RI_PARTS


def job_slices(rows: int, w: int, cap: int) -> list:
    """Row ranges (lo, hi) that cut `rows` verify rows into calls of at most `cap` rows. `w` is the rows of one job (consecutive rows of
    one sequence share most of their experts, so whole jobs stay together when they fit, w > 1 and rows % w == 0). The slices are
    balanced, and none is a single row unless rows == 1 (a one-row call takes the batch-1 kernel)."""
    if rows <= cap:
        return [(0, rows)]
    if 1 < w <= cap and rows % w == 0:
        jobs = rows // w
        n = -(-jobs // max(1, cap // w))
        base, extra = divmod(jobs, n)                    # balanced: the first `extra` slices carry one more job
        out, j = [], 0
        for i in range(n):
            k = base + (1 if i < extra else 0)
            out.append((j * w, (j + k) * w))
            j += k
        return out
    n = -(-rows // cap)
    base, extra = divmod(rows, n)
    out, r = [], 0
    for i in range(n):
        k = base + (1 if i < extra else 0)
        out.append((r, r + k))
        r += k
    return out
