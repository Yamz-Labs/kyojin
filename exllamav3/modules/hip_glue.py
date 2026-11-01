"""KYOJIN_HIP_GLUE: one run-time switch for the decode launch-reduction levers on ROCm.

The first lever is the decode-width site fusion of the hyper-connection pair that every HC
transformer runs twice per layer: the sublayer's residual apply (ext.hc_apply) is deferred
into the next site's fused mix+norm (ext.hc_pf_apply_mix_norm), so one launch replaces two.
The kernel is the prefill one (EXL3_PF_HC_FUSE), whose phase A replays hc_apply_kernel's exact
per-thread loop and whose phase P replays hc_mix_norm_1pass's partials loop verbatim: at a row
count in the single-chunk regime it is bit-identical to the unfused pair. Prefill takes it on
both sides of the knob; decode rows (1 .. 256) only when the knob is on, because that regime
was not measured for the prefill path.

Tests: tests/test_hip_glue.py (no GPU: the range decision and the knob wiring only).
"""
import os

import torch

from ..ext import exllamav3_ext as ext
from ..util.knobs import knob_on

HC_FUSE_MIN_ROWS_PREFILL = 257     # the measured prefill regime (single-chunk, R > 256)
HC_FUSE_MIN_ROWS_DECODE = 1        # decode/verify rows, KYOJIN_HIP_GLUE=1 only
HC_FUSE_MAX_ROWS = 65536           # = HC_MAX_TICKETS in hc_mix.cu


def glue_on() -> bool:
    """KYOJIN_HIP_GLUE, read per call (default off until the lead's gate)."""
    return knob_on("KYOJIN_HIP_GLUE", "0")


def hc_fuse_rows_ok(rows: int, glue: bool | None = None) -> bool:
    """Row counts the fused apply+mix_norm kernel takes. Prefill rows keep the shipped range;
    decode rows (up to 256) join it only when the glue knob is on. Zero rows never fuse."""
    if glue is None:
        glue = glue_on()
    lo = HC_FUSE_MIN_ROWS_DECODE if glue else HC_FUSE_MIN_ROWS_PREFILL
    return lo <= rows <= HC_FUSE_MAX_ROWS


def available() -> bool:
    return hasattr(ext, "hc_pf_apply_mix_norm")
