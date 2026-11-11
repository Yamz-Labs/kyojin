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
MOE_BN = 8                        # grouped expert GEMV (moe_tiles.py 11/11, Qwen3-30B-A3B shapes, 8 slots): 79 / 185 GB/s vs 66 / 146 at BN 64
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


@triton.jit
def _gemv_fp8_moe(x_ptr, qtab, stab, sel_ptr, y_ptr, K, N, S_COLS, XDIV, GRID: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    """Grouped decode GEMV over routed experts: program (tile, slot) computes y[slot, tile] = x[slot // XDIV] . W[sel[slot]]^T,
    the expert's bytes and scales found through pointer tables (qtab / stab int64 = data_ptr of each expert)."""
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    e = tl.load(sel_ptr + slot).to(tl.int64)
    q_ptr = tl.load(qtab + e).to(tl.pointer_type(tl.uint8))
    s_ptr = tl.load(stab + e).to(tl.pointer_type(tl.float32))
    xr = x_ptr + (slot // XDIV).to(tl.int64) * K
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        b = tl.load(q_ptr + rn[:, None].to(tl.int64) * K + rk[None, :], mask=mn[:, None] & mk[None, :], other=0).to(tl.int16)
        h = ((((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.float16, bitcast=True) * 256.0).to(tl.float32)
        xv = tl.load(xr + rk, mask=mk, other=0.0).to(tl.float32)
        part = tl.sum(h * xv[None, :], axis=1)
        if GRID:
            part = part * tl.load(s_ptr + ((pid * BN) // BO) * S_COLS + k0 // BI)
        acc += part
    if not GRID:
        acc = acc * tl.load(s_ptr + rn, mask=mn, other=0.0)
    tl.store(y_ptr + slot.to(tl.int64) * N + rn, acc, mask=mn)


def moe_gemv(x, qtab, stab, sel, N, K, grid, block, xdiv, s_cols):
    """x half [rows, K]; sel int [slots] (expert of each slot); input row of slot = slot // xdiv -> float32 [slots, N]."""
    S = sel.numel()
    y = torch.empty((S, N), dtype=torch.float32, device=x.device)
    _gemv_fp8_moe[(triton.cdiv(N, MOE_BN), S)](x, qtab, stab, sel, y, K, N, s_cols, xdiv, GRID=grid, BO=block[0], BI=block[1],
                                               BN=MOE_BN, BK=BK, num_warps=2)
    return y


@triton.jit
def _fp8_moe_gu(x_ptr, gq, gs, uq, us, sel_ptr, y_ptr, K, N, S_COLS, XDIV, GRID: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """gate and up of every routed slot in ONE launch: tiles [0, T) are gate, [T, 2T) up; y [slots, 2N] fp32 (gate | up)."""
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    T = tl.cdiv(N, BN)
    is_up = pid >= T
    tile = pid - T * is_up.to(tl.int32)
    e = tl.load(sel_ptr + slot).to(tl.int64)
    q_ptr = tl.where(is_up, tl.load(uq + e), tl.load(gq + e)).to(tl.pointer_type(tl.uint8))
    s_ptr = tl.where(is_up, tl.load(us + e), tl.load(gs + e)).to(tl.pointer_type(tl.float32))
    xr = x_ptr + (slot // XDIV).to(tl.int64) * K
    rn = tile * BN + tl.arange(0, BN)
    mn = rn < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        b = tl.load(q_ptr + rn[:, None].to(tl.int64) * K + rk[None, :], mask=mn[:, None] & mk[None, :], other=0).to(tl.int16)
        h = ((((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.float16, bitcast=True) * 256.0).to(tl.float32)
        xv = tl.load(xr + rk, mask=mk, other=0.0).to(tl.float32)
        part = tl.sum(h * xv[None, :], axis=1)
        if GRID:
            part = part * tl.load(s_ptr + ((tile * BN) // BO) * S_COLS + k0 // BI)
        acc += part
    if not GRID:
        acc = acc * tl.load(s_ptr + rn, mask=mn, other=0.0)
    tl.store(y_ptr + slot.to(tl.int64) * (2 * N) + N * is_up.to(tl.int64) + rn, acc, mask=mn)


@triton.jit
def _fp8_moe_down(gu_ptr, dq, ds, sel_ptr, w_ptr, y_ptr, I, N, S_COLS, S, GRID: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    """down projection of the S slots of row r with silu(gate) * up staged on the fly and the routing-weighted fold: y [R, N] fp32."""
    pid = tl.program_id(0)
    r = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    out = tl.zeros((BN,), dtype=tl.float32)
    for j in range(0, S):
        slot = r * S + j
        e = tl.load(sel_ptr + slot).to(tl.int64)
        wt = tl.load(w_ptr + slot).to(tl.float32)
        q_ptr = tl.load(dq + e).to(tl.pointer_type(tl.uint8))
        s_ptr = tl.load(ds + e).to(tl.pointer_type(tl.float32))
        gr = gu_ptr + slot.to(tl.int64) * (2 * I)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, I, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < I
            g = tl.load(gr + rk, mask=mk, other=0.0)
            u = tl.load(gr + I + rk, mask=mk, other=0.0)
            a = (g / (1.0 + tl.exp(-g)) * u).to(tl.float16).to(tl.float32)      # same fp16 rounding of the activation as the torch path
            b = tl.load(q_ptr + rn[:, None].to(tl.int64) * I + rk[None, :], mask=mn[:, None] & mk[None, :], other=0).to(tl.int16)
            h = ((((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.float16, bitcast=True) * 256.0).to(tl.float32)
            part = tl.sum(h * a[None, :], axis=1)
            if GRID:
                part = part * tl.load(s_ptr + ((pid * BN) // BO) * S_COLS + k0 // BI)
            acc += part
        if not GRID:
            acc = acc * tl.load(s_ptr + rn, mask=mn, other=0.0)
        out += wt * acc
    tl.store(y_ptr + r.to(tl.int64) * N + rn, out, mask=mn)


def moe_fused(h, tabs, sel, wts, I, K, grid, block, S, sc_gu, sc_d):
    """Routed sum of a SwiGLU FP8 MoE layer in two launches: h half [R, K], sel / wts [R * S] -> float32 [R, K]."""
    gq, gs, uq, us, dq, ds = tabs
    R = h.shape[0]
    gu = torch.empty((R * S, 2 * I), dtype=torch.float32, device=h.device)
    _fp8_moe_gu[(2 * triton.cdiv(I, MOE_BN), R * S)](h, gq, gs, uq, us, sel, gu, K, I, sc_gu, S, GRID=grid, BO=block[0], BI=block[1],
                                                     BN=MOE_BN, BK=BK, num_warps=2)
    y = torch.empty((R, K), dtype=torch.float32, device=h.device)
    _fp8_moe_down[(triton.cdiv(K, MOE_BN), R)](gu, dq, ds, sel, wts, y, I, K, sc_d, S, GRID=grid, BO=block[0], BI=block[1],
                                               BN=MOE_BN, BK=BK, num_warps=2)
    return y


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
