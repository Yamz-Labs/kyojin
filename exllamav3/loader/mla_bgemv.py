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


@triton.jit
def _k_bg(q_ptr, d_ptr, x_ptr, o_ptr, N, K, NB, BN: tl.constexpr, BK: tl.constexpr):
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
        xv = tl.load(x_ptr + h * K + cols, mask=cm, other=0.0)
        acc += tl.sum(q * d * xv[None, :], axis=1)
    tl.store(o_ptr + h * N + rows, acc, mask=rm)


class Q8Batched:
    """[H, N, K] Q8_0 matrix as int8 + fp16 scales on a device"""

    def __init__(self, raw, H, N, K, dev):
        b = torch.from_numpy(np.ascontiguousarray(raw)).reshape(H * N, K // 32, 34).to(dev)
        self.d = b[:, :, 0:2].contiguous().view(torch.float16).reshape(H, N, K // 32).contiguous()
        self.q = b[:, :, 2:].contiguous().view(torch.int8).reshape(H, N, K).contiguous()
        self.H, self.N, self.K = H, N, K

    def mv(self, x):
        """x [1, H, K] fp32 -> [1, H, N] fp32"""
        out = torch.empty(self.H, self.N, dtype=torch.float32, device=x.device)
        _k_bg[(self.H, triton.cdiv(self.N, BN))](self.q, self.d, x.reshape(self.H, self.K).contiguous(), out, self.N, self.K, self.K // 32, BN=BN, BK=BK, num_warps=4)
        return out.reshape(1, self.H, self.N)


def load(g, name, H, dev):
    """GGUF 3-D Q8_0 tensor (ne0 = K, ne1 = N, ne2 = H) -> Q8Batched"""
    ty, dims, off = g.infos[name]
    assert ty == 8 and dims[2] == H and dims[0] % 32 == 0, (name, ty, dims)
    K, N = dims[0], dims[1]
    return Q8Batched(g._pread(off, N * H * (K // 32) * 34), H, N, K, dev)
