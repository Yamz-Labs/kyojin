"""HY4_MLA_BG=1: the two per-head absorbed MLA matrices (attn_k_b [H, R, nope], attn_v_b [H, vd, R]) read from their STORED Q8_0 bytes at decode.

The old path keeps them as fp32 (`_dense3`, 4.25x the stored size) and runs `einsum` per token: 2.1 GiB read per token (all layers), ~10 ms.
Here int8 q [H, N, K] + fp16 block scales [H, N, K/32] (1.0625 B / weight, exactly the stored values) are read by ONE Triton batched mat-vec per
matrix (grid H x N tiles), fp32 accumulate: out[h, n] = sum_k (q * d)[h, n, k] * x[h, k]. Same products as the fp32 einsum (q * d is the
same fp32 value), only the summation order differs. T > 1 (prefill) keeps the fp32 einsum. Generic by shape (H, N, K % 32 == 0)."""
import os

import numpy as np
import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("HY4_MLA_BG", "0") == "1"
BN = int(os.environ.get("HY4_BG_BN", "16"))
BK = int(os.environ.get("HY4_BG_BK", "128"))
ROW_TILES = os.environ.get("HY4_ROW_TILES", "0") == "1"     # 1 = tile sizes tuned on the Strix for verify rows (T >= 4: BN 32, BK 32; box microbench 19/11)
MAXT = 8                                   # rows served by the batched kernels (verify rows)


@triton.jit
def _k_bg(q_ptr, d_ptr, x_ptr, o_ptr, N, K, NB, sxh, BN: tl.constexpr, BK: tl.constexpr):
    h = tl.program_id(0)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    rm = rows < N
    acc = tl.zeros([BN], dtype=tl.float32)
    qrow = (h * N + rows)[:, None] * K
    drow = (h * N + rows)[:, None] * NB
    for k0 in range(0, K, BK):
        cols = k0 + tl.arange(0, BK)
        cm = cols < K
        m2 = rm[:, None] & cm[None, :]
        q = tl.load(q_ptr + qrow + cols[None, :], mask=m2, other=0).to(tl.float32)
        d = tl.load(d_ptr + drow + (cols // 32)[None, :], mask=m2, other=0.0).to(tl.float32)
        xv = tl.load(x_ptr + h * sxh + cols, mask=cm, other=0.0)
        acc += tl.sum(q * d * xv[None, :], axis=1)
    tl.store(o_ptr + h * N + rows, acc, mask=rm)


@triton.jit
def _k_bgt(q_ptr, d_ptr, x_ptr, o_ptr, N, K, NB, sxt, sxh, T, BT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """T rows (verify): each weight tile is read ONCE and applied to all T rows. out[t, h, n] = sum_k (q * d)[h, n, k] * x[t, h, k]"""
    h = tl.program_id(0)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    rm = rows < N
    tt = tl.arange(0, BT)
    tm = tt < T
    acc = tl.zeros([BT, BN], dtype=tl.float32)
    qrow = (h * N + rows)[:, None] * K
    drow = (h * N + rows)[:, None] * NB
    for k0 in range(0, K, BK):
        cols = k0 + tl.arange(0, BK)
        cm = cols < K
        m2 = rm[:, None] & cm[None, :]
        w = tl.load(q_ptr + qrow + cols[None, :], mask=m2, other=0).to(tl.float32) * tl.load(d_ptr + drow + (cols // 32)[None, :], mask=m2, other=0.0).to(tl.float32)
        xv = tl.load(x_ptr + tt[:, None] * sxt + h * sxh + cols[None, :], mask=tm[:, None] & cm[None, :], other=0.0)
        acc += tl.sum(w[None, :, :] * xv[:, None, :], axis=2)
    tl.store(o_ptr + (tt[:, None] * (tl.num_programs(0) * N)) + h * N + rows[None, :], acc, mask=tm[:, None] & rm[None, :])


class Q8Batched:
    """[H, N, K] Q8_0 matrix as int8 + fp16 scales on a device"""

    def __init__(self, raw, H, N, K, dev):
        b = torch.from_numpy(np.ascontiguousarray(raw)).reshape(H * N, K // 32, 34).to(dev)
        self.d = b[:, :, 0:2].contiguous().view(torch.float16).reshape(H, N, K // 32).contiguous()
        self.q = b[:, :, 2:].contiguous().view(torch.int8).reshape(H, N, K).contiguous()
        self.H, self.N, self.K = H, N, K

    def mv(self, x):
        """x [T, H, K] fp32 -> [T, H, N] fp32 (T == 1: the single-row kernel; T > 1: one kernel, weights read once)"""
        if x.dim() == 3 and x.shape[0] > 1 and x.stride(-1) == 1:
            T = x.shape[0]
            out = torch.empty(T, self.H, self.N, dtype=torch.float32, device=x.device)
            bn, bk = (32, 32) if (ROW_TILES and T >= 4) else (BN, min(BK, 64))
            _k_bgt[(self.H, triton.cdiv(self.N, bn))](self.q, self.d, x, out, self.N, self.K, self.K // 32, x.stride(0), x.stride(1), T,
                                                      BT=triton.next_power_of_2(T), BN=bn, BK=bk, num_warps=4)
            return out
        out = torch.empty(self.H, self.N, dtype=torch.float32, device=x.device)
        if x.dim() == 3 and x.shape[0] == 1 and x.stride(-1) == 1:       # [1, H, K] view with a head stride (e.g. the nope part of q): no copy
            xv, sxh = x, x.stride(1)
        else:
            xv, sxh = x.reshape(self.H, self.K).contiguous(), self.K
        _k_bg[(self.H, triton.cdiv(self.N, BN))](self.q, self.d, xv, out, self.N, self.K, self.K // 32, sxh, BN=BN, BK=BK, num_warps=4)
        return out.reshape(1, self.H, self.N)


def load(g, name, H, dev):
    """GGUF 3-D Q8_0 tensor (ne0 = K, ne1 = N, ne2 = H) -> Q8Batched"""
    ty, dims, off = g.infos[name]
    assert ty == 8 and dims[2] == H and dims[0] % 32 == 0, (name, ty, dims)
    K, N = dims[0], dims[1]
    return Q8Batched(g._pread(off, N * H * (K // 32) * 34), H, N, K, dev)
