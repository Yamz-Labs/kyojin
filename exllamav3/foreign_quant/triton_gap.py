"""Fused decode GEMV (<= 16 rows) for the 32-element GGUF blocks Q4_0, Q5_0, IQ4_NL, MXFP4, read in their native layout.

One Triton kernel for CUDA and ROCm (no extension build): per block the 16 qs bytes give elements 0..15 (low nibble)
and 16..31 (high nibble), so the activation slices are contiguous and each half is one tl.dot. Output matches
gap_formats.dequant + matmul up to fp32 accumulation order.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

FUSED = {"q4_0": 0, "iq4_nl": 1, "mxfp4": 2, "q5_0": 3}
_BB = {"q4_0": 18, "iq4_nl": 18, "mxfp4": 17, "q5_0": 22}
MAX_ROWS = 16
# (BN, num_warps) per format, from a 3090 sweep at N=12288, K=4096, 4 rows
_CFG = {"q4_0": (32, 1), "q5_0": (32, 1), "mxfp4": (32, 1), "iq4_nl": (16, 1)}


@triton.jit
def _iq4v(q):
    b0 = (q & 1) != 0
    b1 = (q & 2) != 0
    b2 = (q & 4) != 0
    b3 = (q & 8) != 0
    t0 = tl.where(b0, -104.0, -127.0); t1 = tl.where(b0, -65.0, -83.0); t2 = tl.where(b0, -35.0, -49.0); t3 = tl.where(b0, -10.0, -22.0)
    t4 = tl.where(b0, 13.0, 1.0); t5 = tl.where(b0, 38.0, 25.0); t6 = tl.where(b0, 69.0, 53.0); t7 = tl.where(b0, 113.0, 89.0)
    u0 = tl.where(b1, t1, t0); u1 = tl.where(b1, t3, t2); u2 = tl.where(b1, t5, t4); u3 = tl.where(b1, t7, t6)
    return tl.where(b3, tl.where(b2, u3, u2), tl.where(b2, u1, u0))


@triton.jit
def _fp4half(q):
    m = q & 7
    v = tl.where(m < 5, m.to(tl.float32), tl.where(m == 5, 6.0, tl.where(m == 6, 8.0, 12.0)))
    return tl.where(q >= 8, -v, v)


@triton.jit
def _gemv32(x_ptr, w_ptr, o_ptr, R, K, N, FMT: tl.constexpr, BB: tl.constexpr, BN: tl.constexpr, RP: tl.constexpr):
    pid = tl.program_id(0)
    Kb = K // 32
    rn = pid * BN + tl.arange(0, BN)
    rr = tl.arange(0, RP)
    j = tl.arange(0, 16)
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    row = rn.to(tl.int64) * (Kb * BB)
    for kb in range(0, Kb):
        base = row + kb * BB
        if FMT == 2:
            e = tl.load(w_ptr + base).to(tl.float32)
            d = tl.exp2(e - 128.0)
            qoff = 1
        else:
            b0 = tl.load(w_ptr + base).to(tl.int32)
            b1 = tl.load(w_ptr + base + 1).to(tl.int32)
            d = (b0 | (b1 << 8)).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
            qoff = 2
            if FMT == 3:
                qoff = 6
        qs = tl.load(w_ptr + base[:, None] + qoff + j[None, :]).to(tl.int32)
        lo = qs & 15
        hi = qs >> 4
        if FMT == 0:
            wl = (lo - 8).to(tl.float32)
            wh = (hi - 8).to(tl.float32)
        elif FMT == 1:
            wl = _iq4v(lo)
            wh = _iq4v(hi)
        elif FMT == 2:
            wl = _fp4half(lo)
            wh = _fp4half(hi)
        else:
            qh = tl.load(w_ptr + base + 2).to(tl.int32) | (tl.load(w_ptr + base + 3).to(tl.int32) << 8) \
                | (tl.load(w_ptr + base + 4).to(tl.int32) << 16) | (tl.load(w_ptr + base + 5).to(tl.int32) << 24)
            wl = ((lo | (((qh[:, None] >> j[None, :]) & 1) << 4)) - 16).to(tl.float32)
            wh = ((hi | (((qh[:, None] >> (j[None, :] + 16)) & 1) << 4)) - 16).to(tl.float32)
        wl = (wl * d[:, None]).to(tl.float16)
        wh = (wh * d[:, None]).to(tl.float16)
        a_lo = tl.load(x_ptr + rr[:, None] * K + kb * 32 + j[None, :], mask=rr[:, None] < R, other=0.0)
        a_hi = tl.load(x_ptr + rr[:, None] * K + kb * 32 + 16 + j[None, :], mask=rr[:, None] < R, other=0.0)
        acc = tl.dot(a_lo, tl.trans(wl), acc)
        acc = tl.dot(a_hi, tl.trans(wh), acc)
    tl.store(o_ptr + rr[:, None] * N + rn[None, :], acc.to(tl.float16), mask=rr[:, None] < R)


def supported(fmt: str, rows: int, k: int, n: int) -> bool:
    return fmt in FUSED and 1 <= rows <= MAX_ROWS and k % 32 == 0 and n % _CFG[fmt][0] == 0


def gemv(fmt: str, x: torch.Tensor, raw: torch.Tensor, out: torch.Tensor | None = None, BN: int | None = None, num_warps: int | None = None):
    """x fp16 [R, K] contiguous, raw uint8 [N, K/32*bb] -> fp16 [R, N]."""
    R, K = x.shape
    N = raw.shape[0]
    BN = BN or _CFG[fmt][0]
    num_warps = num_warps or _CFG[fmt][1]
    assert R <= MAX_ROWS and K % 32 == 0 and raw.shape[1] == K // 32 * _BB[fmt] and N % BN == 0
    if out is None:
        out = torch.empty((R, N), dtype=torch.float16, device=x.device)
    RP = 16
    _gemv32[(N // BN,)](x, raw, out, R, K, N, FUSED[fmt], _BB[fmt], BN, RP, num_warps=num_warps)
    return out
