"""DS4.1 decode, batch 1: fewer launches for the dense EXL3 linears (bit-identical to the separate calls).

Three launch merges, all through `exl3_dec_gemv_x` (one launch, up to 8 matrices):
  * WOA  the 8 grouped o_a projections (each its own input row) -> 1 launch, output written straight into [1, 1, G, r];
  * QKV  wq_a + wkv (same input) -> 1 launch;
  * SHW  shared expert gate + up (same input) -> 1 launch.
The k-split (ktw / kbs) of a launch decides the float summation order. A merged launch would normally pick it from the sum of
the strips of all matrices, so every merge passes `strips_hint` = the strips of ONE equivalent single call, and `eligible()`
checks (with a host replica of the kernel's pick_ktw) that all matrices would have picked the same ktw alone.

Switches: DSV41_DEC2=0 turns everything off; DSV41_DEC2_<WOA|QKV|SHW>=0 turns one off. `FLAGS` can be edited at run time (A/B tests).
"""
from __future__ import annotations
import os
import torch

_NAMES = ("WOA", "QKV", "SHW", "STREAM", "ROUTER", "HC")
FLAGS = {n: os.environ.get("DSV41_DEC2_" + n, os.environ.get("DSV41_DEC2", "1")) != "0" for n in _NAMES}
_state = {}


def _ext():
    e = _state.get("ext")
    if e is None:
        try:
            from ..ext import exllamav3_ext as ext
            e = ext if hasattr(ext, "exl3_dec_gemv_x") and torch.version.hip else False
        except Exception:
            e = False
        # the kernel's k-split can be overridden from the environment: the host replica below assumes the defaults
        if any(os.environ.get(k) for k in ("EXL3_DEC_KTW", "EXL3_DEC_KBS", "EXL3_DEC_MIN_BLOCKS")):
            e = False
        _state["ext"] = e
    return e


def pick_ktw(kt: int, strips: int, kb2: int) -> int:
    """Host replica of exl3_dec.cu pick_ktw (min_blocks 32, nothing forced)."""
    ring = 1 if 4 * kb2 <= 32 else 2
    ok = lambda k: k >= ring and k % ring == 0 and k <= 32 and kt % (8 * k) == 0
    k = 32
    while k >= 4:
        if ok(k) and strips * (kt // (8 * k)) >= 32:
            return k
        k >>= 1
    for k in (8, 4, 16, 2, 32):
        if ok(k):
            return k
    return 0


def _inner(lin):
    """LinearEXL3 under a PackLin when it can take the dec GEMV without padding / bias, else None."""
    m = getattr(lin, "lin", None)
    i = getattr(m, "inner", None)
    if i is None or getattr(i, "quant_type", None) == "fp16" or type(i).__name__ != "LinearEXL3":
        return None
    if not (i.dec_ok and i.bias is None and i.default_out_dtype == torch.half and m.in_features == m.in_features_unpadded
            and m.out_features == m.out_features_unpadded and i.in_features == m.in_features and i.out_features == m.out_features):
        return None
    return i


class Merge:
    """A set of PackLins that can share one `exl3_dec_gemv_x` launch; None-like (`ok` False) otherwise."""

    def __init__(self, lins, per_job_x: bool):
        self.ok = False
        ins = [_inner(l) for l in lins]
        if not _ext() or not ins or len(ins) > 8 or any(i is None for i in ins) or (not per_job_x and len(ins) > 4):
            return
        i0 = ins[0]
        kb2 = int(round(i0.K * 2))
        if any(int(round(i.K * 2)) != kb2 or bool(i.mcg) != bool(i0.mcg) or i.in_features != i0.in_features for i in ins):
            return
        kt = i0.in_features // 16
        alone = {pick_ktw(kt, (i.out_features + 511) // 512, kb2) for i in ins}
        if len(alone) != 1 or 0 in alone:
            return
        # the launch picks from `hint` strips: take the strips of one matrix whose single-call ktw is the common one
        self.hint = (ins[0].out_features + 511) // 512
        if pick_ktw(kt, self.hint, kb2) != next(iter(alone)):
            return
        self.ins, self.n, self.k_in = ins, len(ins), i0.in_features
        self.tr = [i.trellis for i in ins]
        self.suh = [i.suh for i in ins]
        self.svh = [i.svh for i in ins]
        self.K = [float(i.K) for i in ins]
        self.mcg = bool(i0.mcg)
        self.outs = [i.out_features for i in ins]
        self.stride = i0.in_features if per_job_x else 0
        self.ok = True

    def run(self, x, outs):
        from ..modules.quant.exl3 import dec_workspace
        scratch, counters = dec_workspace(x.device)
        _ext().exl3_dec_gemv_x(x, self.tr, self.suh, self.svh, outs, scratch, counters, self.K, self.mcg, self.stride, self.hint)


def _merge(owner, key, lins, per_job_x):
    c = owner.__dict__.setdefault("_dec2", {})
    m = c.get(key)
    if m is None:
        m = c[key] = Merge(lins, per_job_x)
    return m


def wo_a(handle, o):
    """o [B, L, G, d] fp16 contiguous, B = L = 1 -> [1, 1, G, r] fp16, or None (caller runs the per-group path).
    handle: the object that owns the slices (`fn.slices`, list of PackLin)."""
    if not FLAGS["WOA"] or o.dtype != torch.half or not o.is_contiguous() or o.numel() != o.size(2) * o.size(3) or o.size(0) != 1:
        return None
    sl = getattr(handle, "slices", None)
    if sl is None or len(sl) != o.size(2):
        return None
    m = _merge(handle, "woa", sl, True)
    if not m.ok or m.k_in != o.size(3) or len({n for n in m.outs}) != 1:
        return None
    G, r = o.size(2), m.outs[0]
    y = torch.empty(1, 1, G, r, dtype=torch.half, device=o.device)
    y2 = y.view(G, r)
    m.run(o.view(G, o.size(3)), [y2[g] for g in range(G)])
    return y


def pair(owner, key, lins, x):
    """Two or more PackLins over the same one-row fp16 input x [1, K] -> list of [1, N_i] fp16 outputs, or None."""
    if not (FLAGS[key] and x.dtype == torch.half and x.is_contiguous() and x.numel() == x.size(-1)):
        return None
    m = _merge(owner, key, lins, False)
    if not m.ok or m.k_in != x.numel():
        return None
    outs = [torch.empty(1, n, dtype=torch.half, device=x.device) for n in m.outs]
    m.run(x.view(1, -1), outs)
    return outs


# ------------------------------------------------------------------------------------------------ second stream
class Side:
    """One side stream per device + reusable events. Work that does not feed the next kernel of the main stream (the hyper-connection
    mix, the shared expert) runs there while the main stream streams the big weights. Every consumer waits on the event of its
    producer before it reads; every writer of a buffer the side stream reads waits on the side stream's last event first."""

    def __init__(self, dev):
        self.stream = torch.cuda.Stream(device=dev)
        self.fork = torch.cuda.Event()      # main -> side: inputs of the side work are ready
        self.mix_done = torch.cuda.Event()  # side -> main: hc mix outputs ready / hf no longer read
        self.sh_done = torch.cuda.Event()   # side -> main: shared expert output ready


def side(dev):
    """The Side of `dev` when stream overlap is enabled, else None."""
    if not FLAGS["STREAM"] or not torch.cuda.is_available():
        return None
    k = ("side", str(dev))
    sd = _state.get(k)
    if sd is None:
        sd = _state[k] = Side(dev)
    return sd
