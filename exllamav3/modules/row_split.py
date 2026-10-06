"""Row split of the speculative verify (EXL3_VERIFY_ROW_SPLIT, default off): 1 = split the dense GEMV and hc-mixer launches here and the MoE
in moe_fused.py; 2 = split only the MoE, the GEMV and hc mixer take 5..8 rows natively (see MODE below).

Several verify kernels have a fast path for up to 4 rows (one-pass hc mixer: GRR_MAX_RT = 4) or are fast only up to
4 rows (exl3_dec_gemv_r / _multi: the 8-row instantiation costs about 4x per row). With 5..8 rows (one sequence,
q_len = R) they fall to a slow launch. These kernels treat every row independently (one weight decode shared across
rows, one reduction order per row), so a launch over R rows is split here into launches of at most 4 rows: row r
runs through exactly the instantiation it uses in a 1..4 row round, same arithmetic, flat cost per row.
"""
from __future__ import annotations
import os
from ..ext import exllamav3_ext as ext

CHUNK = 4
# "1": the GEMV and hc-mixer launches are split here into <= 4-row launches; "2": only the MoE is split (moe_fused.py), the GEMV and hc mixer
# take 5..8 rows natively (EXL3_GEMV_R_RPB=4 EXL3_GEMV_R_CHUNK_FAST=1 EXL3_GR_MIX_Q8_MAXR=8, one launch, bit-identical)
MODE = os.environ.get("EXL3_VERIFY_ROW_SPLIT", "0")
ON = MODE == "1"
if MODE == "2":
    # read by the extension at launch (GEMV) or at the first hc-mixer call; the caller's own values win
    for _k, _v in (("EXL3_GEMV_R_RPB", "4"), ("EXL3_GEMV_R_CHUNK_FAST", "1"), ("EXL3_GR_MIX_Q8_MAXR", "8")):
        os.environ.setdefault(_k, _v)
_installed = False


def _split(fn, nrows_of, slicer):
    def wrapped(*a, **kw):
        R = nrows_of(*a, **kw)
        if R <= CHUNK:
            return fn(*a, **kw)
        for lo in range(0, R, CHUNK):
            hi = min(lo + CHUNK, R)
            fn(*slicer(lo, hi, *a, **kw))
        return None
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    wrapped.__wrapped__ = fn
    return wrapped


def _gemv_r(lo, hi, x, trellis, suh, svh, out, scratch, counters, K, mcg = False):
    return (x[lo:hi], trellis, suh, svh, out[lo:hi], scratch, counters, K, mcg)


def _gemv_r_multi(lo, hi, x, trellis, suh, svh, outs, scratch, counters, K, mcg = False):
    return (x[lo:hi], trellis, suh, svh, [o[lo:hi] for o in outs], scratch, counters, K, mcg)


def _gr_mix_q8(lo, hi, s3, fq, fsc, uq, usc, w, eps, dots, post, mixed):
    return (s3[lo:hi], fq, fsc, uq, usc, w, eps, dots[lo:hi], None if post is None else post[lo:hi], mixed[lo:hi])


def install():
    global _installed
    if _installed or not ON:
        return
    _installed = True
    if hasattr(ext, "exl3_dec_gemv_r"):
        ext.exl3_dec_gemv_r = _split(ext.exl3_dec_gemv_r, lambda x, *a, **k: x.shape[0], _gemv_r)
    if hasattr(ext, "exl3_dec_gemv_r_multi"):
        ext.exl3_dec_gemv_r_multi = _split(ext.exl3_dec_gemv_r_multi, lambda x, *a, **k: x.shape[0], _gemv_r_multi)
    if hasattr(ext, "gr_mix_q8"):
        ext.gr_mix_q8 = _split(ext.gr_mix_q8, lambda s3, *a, **k: s3.shape[0], _gr_mix_q8)
