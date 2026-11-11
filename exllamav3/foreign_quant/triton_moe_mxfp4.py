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

BN = int(os.environ.get("EXL3_MXFP4_MOE_BN", "32")  # Strix sweep 11/11: BN 16 101 -> BN 32 115 GB/s, served gpt-oss 24.4 -> 25.7 tok/s)
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


def moe_gemv(x: torch.Tensor, wtab: torch.Tensor, btab: torch.Tensor | None, sel: torch.Tensor, N: int, K: int, xdiv: int) -> torch.Tensor:
    """x (rows, K) fp16 contiguous; sel (slots,) int32 expert ids; returns (slots, N) fp32."""
    slots = sel.numel()
    y = torch.empty((slots, N), dtype=torch.float32, device=x.device)
    _mxfp4_moe[(triton.cdiv(N, BN), slots)](x, wtab, btab if btab is not None else wtab, sel, y, K, N, xdiv,
                                            HAS_B=btab is not None, BN=BN, num_warps=NW)
    return y
