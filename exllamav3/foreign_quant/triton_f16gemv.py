"""Decode-row GEMV for fp16 weights stored (K, N) (LinearFP16 orientation), <= 16 rows, for ROCm where hipBLAS GEMV on small / skinny
shapes runs far under the read roof (Strix: 2880 x 512 at 47 GB/s, 4096 x 2880 at 153). One program = BN output columns, K walked in
BK chunks with coalesced (BK, BN) weight tiles; fp32 accumulation, fp16 out."""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

MAX_ROWS = 16
# Strix sweep 11/11 (run_f16gemv.py, gpt-oss shapes, R1): BN 32 BK 64 nw 2: 2880x512 47 -> 138 GB/s, 4096x2880 153 -> 213 vs hipBLAS
BN = int(os.environ.get("EXL3_F16GEMV_BN", "32"))
BK = int(os.environ.get("EXL3_F16GEMV_BK", "64"))
NW = int(os.environ.get("EXL3_F16GEMV_NW", "2"))


@triton.jit
def _f16_gemv(x_ptr, w_ptr, y_ptr, R, K, N, BN: tl.constexpr, BK: tl.constexpr, RP: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    rr = tl.arange(0, RP)
    mr = rr < R
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        w = tl.load(w_ptr + rk[:, None].to(tl.int64) * N + rn[None, :], mask=mk[:, None] & mn[None, :], other=0.0)
        xv = tl.load(x_ptr + rr[:, None] * K + rk[None, :], mask=mr[:, None] & mk[None, :], other=0.0)
        acc = tl.dot(xv, w, acc)
    tl.store(y_ptr + rr[:, None] * N + rn[None, :], acc.to(tl.float16), mask=mr[:, None] & mn[None, :])


def supported(R: int, K: int, N: int) -> bool:
    # hipBLAS stays faster for wide outputs (2880 x 4096: 196 vs 167 GB/s): only N in [128, 3072]
    return 1 <= R <= MAX_ROWS and 128 <= N <= 3072 and os.environ.get("EXL3_F16GEMV", "1") != "0"


def gemv(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """x fp16 (R, K) contiguous, w fp16 (K, N) contiguous -> fp16 (R, N)."""
    R, K = x.shape
    N = w.shape[1]
    y = out if out is not None else torch.empty((R, N), dtype=torch.float16, device=x.device)
    _f16_gemv[(triton.cdiv(N, BN),)](x, w, y, R, K, N, BN=BN, BK=BK, RP=16, num_warps=NW)
    return y
