"""Grouped decode GEMV over routed MXFP4 experts (GGUF block_mxfp4 layout: per 32-block [E8M0][16 bytes, elements j | j + 16 << 4]).

Program (tile, slot) computes y[slot, tile] = x[slot // XDIV] . W[sel[slot]]^T + b[sel[slot]], the expert's bytes and bias found
through int64 pointer tables (data_ptr of each expert's packed rows / fp16 bias). One launch per projection for all (row, expert) slots,
instead of one GEMV launch per expert. Same decode as triton_gap._gemv32 FMT 2 (fp4 value x2, scale 2^(e - 128)).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .triton_gap import _fp4half

# Strix sweep 11/11: BN 16 101 -> BN 32 115 GB/s, served gpt-oss 24.4 -> 25.7 tok/s
BN = int(os.environ.get("EXL3_MXFP4_MOE_BN", "32"))
NW = int(os.environ.get("EXL3_MXFP4_MOE_NW", "2"))


@triton.jit
def _mxfp4_moe(x_ptr, wtab, btab, sel_ptr, y_ptr, K, N, XDIV, HAS_B: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    e = tl.load(sel_ptr + slot).to(tl.int64)
    w_ptr = tl.load(wtab + e).to(tl.pointer_type(tl.uint8))
    xr = x_ptr + (slot // XDIV).to(tl.int64) * K
    Kb = K // 32
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    j = tl.arange(0, 16)
    row = rn.to(tl.int64) * (Kb * 17)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for kb in range(0, Kb):
        base = row + kb * 17
        d = tl.exp2(tl.load(w_ptr + base, mask=mn, other=0).to(tl.float32) - 128.0)
        qs = tl.load(w_ptr + base[:, None] + 1 + j[None, :], mask=mn[:, None], other=0).to(tl.int32)
        xl = tl.load(xr + kb * 32 + j).to(tl.float32)
        xh = tl.load(xr + kb * 32 + 16 + j).to(tl.float32)
        part = tl.sum(_fp4half(qs & 15) * xl[None, :] + _fp4half(qs >> 4) * xh[None, :], axis=1)
        acc += part * d
    if HAS_B:
        b_ptr = tl.load(btab + e).to(tl.pointer_type(tl.float16))
        acc += tl.load(b_ptr + rn, mask=mn, other=0.0).to(tl.float32)
    tl.store(y_ptr + slot.to(tl.int64) * N + rn, acc, mask=mn)


MMVQ_FIRES = 0     # python-level count of EXL3_FQ_MMVQ_MXFP4 branch calls (eager / capture; a graph replay does not run python)


def mmvq_mxfp4() -> bool:
    """EXL3_FQ_MMVQ_MXFP4=1 on CUDA (not HIP): llama.cpp-style MXFP4 mmvq decode MoE (fq_mxfp4_moe_gu / fq_mxfp4_moe_down). Read at call time."""
    return os.environ.get("EXL3_FQ_MMVQ_MXFP4", "0") == "1" and bool(torch.version.cuda) and not torch.version.hip


def moe_gemv(x: torch.Tensor, wtab: torch.Tensor, btab: torch.Tensor | None, sel: torch.Tensor, N: int, K: int, xdiv: int) -> torch.Tensor:
    """x (rows, K) fp16 contiguous; sel (slots,) int32 expert ids; returns (slots, N) fp32."""
    slots = sel.numel()
    y = torch.empty((slots, N), dtype=torch.float32, device=x.device)
    bn, nw = (TUNED_BN, TUNED_NW) if tuned() else (BN, NW)
    _mxfp4_moe[(triton.cdiv(N, bn), slots)](x, wtab, btab if btab is not None else wtab, sel, y, K, N, xdiv,
                                            HAS_B=btab is not None, BN=bn, num_warps=nw)
    return y


def tuned() -> bool:
    """EXL3_MXFP4_MOE_TUNED=1 on CUDA (not HIP): fused gate+up launch + BN16/NW4 grids. Read at call time."""
    return os.environ.get("EXL3_MXFP4_MOE_TUNED", "0") == "1" and not torch.version.hip


TUNED_BN = 16
TUNED_NW = 4


@triton.jit
def _gu_fused(x_ptr, gtab, gbtab, utab, ubtab, sel_ptr, a_ptr, K, N, LIM, BN: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    e = tl.load(sel_ptr + slot).to(tl.int64)
    gp = tl.load(gtab + e).to(tl.pointer_type(tl.uint8))
    up = tl.load(utab + e).to(tl.pointer_type(tl.uint8))
    Kb = K // 32
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    j = tl.arange(0, 16)
    row = rn.to(tl.int64) * (Kb * 17)
    ag = tl.zeros((BN,), dtype=tl.float32)
    au = tl.zeros((BN,), dtype=tl.float32)
    for kb in range(0, Kb):
        base = row + kb * 17
        xl = tl.load(x_ptr + kb * 32 + j).to(tl.float32)
        xh = tl.load(x_ptr + kb * 32 + 16 + j).to(tl.float32)
        dg = tl.exp2(tl.load(gp + base, mask=mn, other=0).to(tl.float32) - 128.0)
        du = tl.exp2(tl.load(up + base, mask=mn, other=0).to(tl.float32) - 128.0)
        qg = tl.load(gp + base[:, None] + 1 + j[None, :], mask=mn[:, None], other=0).to(tl.int32)
        qu = tl.load(up + base[:, None] + 1 + j[None, :], mask=mn[:, None], other=0).to(tl.int32)
        ag += tl.sum(_fp4half(qg & 15) * xl[None, :] + _fp4half(qg >> 4) * xh[None, :], axis=1) * dg
        au += tl.sum(_fp4half(qu & 15) * xl[None, :] + _fp4half(qu >> 4) * xh[None, :], axis=1) * du
    bg = tl.load(gbtab + e).to(tl.pointer_type(tl.float16))
    bu = tl.load(ubtab + e).to(tl.pointer_type(tl.float16))
    ag += tl.load(bg + rn, mask=mn, other=0.0).to(tl.float32)
    au += tl.load(bu + rn, mask=mn, other=0.0).to(tl.float32)
    g = ag.to(tl.float16).to(tl.float32)        # the 3-launch path rounds g,u to fp16 before the activation
    u = au.to(tl.float16).to(tl.float32)
    g = tl.minimum(g, LIM)
    u = tl.minimum(tl.maximum(u, -LIM), LIM)
    a = (u + 1.0) * g * (1.0 / (1.0 + tl.exp(-1.702 * g)))     # swiglu_oai
    tl.store(a_ptr + slot.to(tl.int64) * N + rn, a.to(tl.float16), mask=mn)


def gu_fused(h, gw, gb, uw, ub, sel, I: int, K: int, lim: float) -> torch.Tensor:
    """Fused gate+up+bias+swiglu_oai: h (rows=1.., K) fp16, returns (slots, I) fp16. Single-row input only (x row 0 for all slots)."""
    a = torch.empty((sel.numel(), I), dtype=torch.float16, device=h.device)
    _gu_fused[(triton.cdiv(I, TUNED_BN), sel.numel())](h, gw, gb, uw, ub, sel, a, K, I, lim, BN=TUNED_BN, num_warps=TUNED_NW)
    return a
