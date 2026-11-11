"""FP8 (E4M3) linears served as stored: per-row / per-tensor scale (`weight_scale`) or the 2D block grid (`weight_scale_inv`,
DeepSeek / Qwen-FP8 / GLM / MiniMax layout). One byte per weight is read instead of the two of the fp16 dequant.

Decode GEMV (<= MAX_ROWS rows): one Triton kernel for ROCm and CUDA (no extension build). E4M3 -> fp16 is exact by bit placement:
half(bits = sign << 8 | (b & 0x7F) << 7) * 256 (normals and subnormals; the 0x7F NaN code is never stored by real checkpoints).
With the block grid, BN rows of a tile share one block row and each BK chunk one block column, so the block scale is one
scalar per (tile, chunk) applied to the chunk's partial sum. Wider batches: fp16 dequant of the matrix + matmul.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROWS = 16
BN, BK = 64, 128                 # Strix 27B shapes, 1 row (fp8_tiles.py 11/11): 170-190 GB/s vs 102-162 at BN 32


@triton.jit
def _gemv_fp8(x_ptr, q_ptr, s_ptr, y_ptr, R, K, N, S_COLS, GRID: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr, RP: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    rr = tl.arange(0, RP)
    mr = rr < R
    acc = tl.zeros((RP, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        b = tl.load(q_ptr + rn[:, None].to(tl.int64) * K + rk[None, :], mask=mn[:, None] & mk[None, :], other=0).to(tl.int16)
        h = (((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.float16, bitcast=True) * 256.0
        xv = tl.load(x_ptr + rr[:, None] * K + rk[None, :], mask=mr[:, None] & mk[None, :], other=0.0)
        part = tl.dot(xv, tl.trans(h), out_dtype=tl.float32)
        if GRID:
            part = part * tl.load(s_ptr + ((pid * BN) // BO) * S_COLS + k0 // BI)
        acc += part
    if not GRID:
        acc = acc * tl.load(s_ptr + rn, mask=mn, other=0.0)[None, :]
    tl.store(y_ptr + rr[:, None] * N + rn[None, :], acc, mask=mr[:, None] & mn[None, :])


def gemv(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, grid: bool, block: tuple[int, int]) -> torch.Tensor:
    """x half [R, K] (R <= MAX_ROWS), q uint8 [N, K], s float32 [N] (per row) or [ceil(N/bo), ceil(K/bi)] (grid) -> float32 [R, N]."""
    R, K = x.shape
    N = q.shape[0]
    y = torch.empty((R, N), dtype=torch.float32, device=x.device)
    _gemv_fp8[(triton.cdiv(N, BN),)](x, q, s, y, R, K, N, s.shape[1] if grid else 1, GRID=grid, BO=block[0], BI=block[1],
                                     BN=BN, BK=BK, RP=16, num_warps=4)
    return y


def ok(N: int, K: int, grid: bool, block: tuple[int, int]) -> bool:
    """The kernel's tiling fits this matrix: a tile's rows share one block row, a K chunk one block column."""
    return not grid or (block[0] % BN == 0 and block[1] % BK == 0)


def dequant(q: torch.Tensor, s: torch.Tensor, grid: bool, block: tuple[int, int]) -> torch.Tensor:
    """fp16 [N, K] of the stored matrix (reference and the wide-batch path)."""
    w = q.view(torch.float8_e4m3fn).to(torch.float32)
    if grid:
        N, K = q.shape
        s = s.repeat_interleave(block[0], 0)[:N].repeat_interleave(block[1], 1)[:, :K]
        w = w * s
    else:
        w = w * s[:, None]
    return w.half()


class LinearFP8:
    """Inner of a Linear (quant_type "fp8") over the stored E4M3 bytes, checkpoint orientation [out, in]."""

    def __init__(self, in_features, out_features, q, s, grid, block, bias, out_dtype, key=None):
        self.in_features, self.out_features = in_features, out_features
        self.q, self.s, self.grid, self.block = q.contiguous(), s.float().contiguous(), grid, tuple(block)
        self.bias, self.out_dtype, self.key = bias, out_dtype, key
        self.bc = None
        self._pinned_store = None
        self.swap_device = None

    def unload(self):
        pass

    def get_tensors(self, key: str):
        t = {f"{key}.weight": self.q, f"{key}.weight_scale{'_inv' if self.grid else ''}": self.s}
        if self.bias is not None:
            t[f"{key}.bias"] = self.bias
        return t

    def get_weight_tensor(self) -> torch.Tensor:
        return dequant(self.q, self.s, self.grid, self.block).T

    def get_bias_tensor(self):
        return self.bias

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        dtype = out_dtype or self.out_dtype or torch.half
        xh = x if x.dtype == torch.half else x.half()
        if xh.shape[0] <= MAX_ROWS:
            y = gemv(xh.contiguous(), self.q, self.s, self.grid, self.block)
        else:
            y = torch.matmul(xh, dequant(self.q, self.s, self.grid, self.block).T)
        if self.bias is not None:
            y = y + self.bias
        return y.to(dtype).view(out_shape)
