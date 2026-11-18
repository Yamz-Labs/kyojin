"""EXL3_GGUF_FAST: one switch for the generic-GGUF fast paths (default off).

EXL3_GGUF_FAST=1 turns on, for every flag below that is not set explicitly:
  EXL3_FQ_MMVQ_LEGACY, EXL3_MXFP4_MOE_TUNED, EXL3_FQ_MMVQ_MXFP4, EXL3_MXFP4_PREFILL_GROUPED, EXL3_FQ_MMVQ_EPI, EXL3_LEGACY_PREFILL_DQ.
EXL3_MOE_FOLD_FUSED is NOT part of it (3.6 % < 4 %, stays opt-in). A flag that is set in the environment keeps its own value
(EXL3_FQ_MMVQ_MXFP4=0 wins over EXL3_GGUF_FAST=1). Every reader calls flag() at call time.

All six paths act on ForeignLinear / foreign MXFP4 expert weights, i.e. on weights kept in their stored GGUF format. This tree has no generic
GGUF loader (exllamav3/architecture/gguf_generic.py is not in it), so there is no per-model attribute to key on: the switch is the env alone."""
from __future__ import annotations

import os

FAST_FLAGS = frozenset({
    "EXL3_FQ_MMVQ_LEGACY",
    "EXL3_MXFP4_MOE_TUNED",
    "EXL3_FQ_MMVQ_MXFP4",
    "EXL3_MXFP4_PREFILL_GROUPED",
    "EXL3_FQ_MMVQ_EPI",
    "EXL3_LEGACY_PREFILL_DQ",
})


def flag(name: str) -> bool:
    """True if `name` is on: set explicitly -> its own value == "1"; unset or empty -> EXL3_GGUF_FAST=1 and name in FAST_FLAGS."""
    v = os.environ.get(name)
    if v:
        return v == "1"
    return name in FAST_FLAGS and os.environ.get("EXL3_GGUF_FAST", "0") == "1"
