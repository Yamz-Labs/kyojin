"""HY4_Q8_SOA=1: dense Q8_0 decode GEMV on a structure-of-arrays repack (int8 q [N, K] + fp16 block scales [N, K/32]), Triton.

The stored Q8_0 layout is 34-byte blocks (fp16 d + 32 int8): not 4-byte aligned, so the C++ kernel (fq_gemv_q8_0) reads every weight through
16-bit loads (16 loads per 34 bytes). Repacked, a weight row is one contiguous int8 run: the kernel below reads 16-byte vectors and keeps
many bytes in flight. Same maths as fq_gemv_q8_0 (llama.cpp mmvq order): x is quantised to Q8_1 per 32-block (d = amax / 127 in fp32, q =
round-half-away(x / d), the scale USED in the dot is d rounded to fp16), integer block dot, acc += (dw * da) * isum. Only the summation
order over blocks differs. Rows M == 1 only (decode); callers keep the old path for M > 1."""
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("HY4_Q8_SOA", "0") == "1"
BN = int(os.environ.get("HY4_Q8_BN", "16"))
BB = int(os.environ.get("HY4_Q8_BB", "16"))


@triton.jit
def _k_xq(x_ptr, q_ptr, d_ptr, NB):
    b = tl.program_id(0)
    offs = b * 32 + tl.arange(0, 32)
    x = tl.load(x_ptr + offs)
    amax = tl.max(tl.abs(x), axis=0)
    d = amax / 127.0
    ds = tl.where(amax == 0.0, 1.0, d)
    v = x / ds
    a = tl.abs(v)
    r = tl.floor(a)
    r = tl.where(a - r >= 0.5, r + 1.0, r)                  # roundf: half away from zero
    r = tl.where(v < 0, -r, r)
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)
    r = tl.where(amax == 0.0, 0.0, r)
    tl.store(q_ptr + offs, r.to(tl.int8))
    tl.store(d_ptr + b, d.to(tl.float16).to(tl.float32))


@triton.jit
def _k_q8(q_ptr, d_ptr, xq_ptr, xd_ptr, o_ptr, N, K, NB, BN: tl.constexpr, BB: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    rm = rows < N
    acc = tl.zeros([BN], dtype=tl.float32)
    lane = tl.arange(0, 32)
    for b0 in range(0, NB, BB):
        blks = b0 + tl.arange(0, BB)
        bm = blks < NB
        cols = blks[:, None] * 32 + lane[None, :]            # [BB, 32]
        w = tl.load(q_ptr + rows[:, None, None] * K + cols[None, :, :], mask=rm[:, None, None] & bm[None, :, None], other=0)
        xq = tl.load(xq_ptr + cols, mask=bm[:, None], other=0)
        isum = tl.sum(w.to(tl.int32) * xq.to(tl.int32)[None, :, :], axis=2)          # [BN, BB]
        dw = tl.load(d_ptr + rows[:, None] * NB + blks[None, :], mask=rm[:, None] & bm[None, :], other=0.0).to(tl.float32)
        da = tl.load(xd_ptr + blks, mask=bm, other=0.0)
        acc += tl.sum(dw * da[None, :] * isum.to(tl.float32), axis=1)
    tl.store(o_ptr + rows, acc, mask=rm)


class Meta:
    """shape / device stand-in for freed stored bytes (callers only read .shape / .device)"""

    def __init__(self, shape, device):
        self.shape, self.device = tuple(shape), device

    def __getitem__(self, sl):
        n = len(range(*sl.indices(self.shape[0])))
        return Meta((n,) + self.shape[1:], self.device)


class Q8Soa:
    """a [N, K] Q8_0 matrix as int8 q [N, K] + fp16 d [N, K/32] on a device (from the stored [N, nb, 34] bytes)"""

    def __init__(self, packed):
        N, nb, _ = packed.shape
        self.d = packed[:, :, 0:2].contiguous().view(torch.float16).reshape(N, nb).contiguous()
        self.q = packed[:, :, 2:].contiguous().view(torch.int8).reshape(N, nb * 32).contiguous()
        self.N, self.K, self.nb = N, nb * 32, nb

    @staticmethod
    def cat(parts):
        o = object.__new__(Q8Soa)
        o.q = torch.cat([p.q for p in parts], 0)
        o.d = torch.cat([p.d for p in parts], 0)
        o.N, o.K, o.nb = o.q.shape[0], parts[0].K, parts[0].nb
        return o

    def slice(self, o, n):
        r = object.__new__(Q8Soa)
        r.q, r.d, r.N, r.K, r.nb = self.q[o:o + n], self.d[o:o + n], n, self.K, self.nb
        return r

    def dense(self):
        return (self.q.float().reshape(self.N, self.nb, 32) * self.d.float()[:, :, None]).reshape(self.N, self.K)

    def mv(self, x, out=None):
        """x [1, K] fp32 contiguous -> [1, N] fp32"""
        xq = torch.empty(self.K, dtype=torch.int8, device=x.device)
        xd = torch.empty(self.nb, dtype=torch.float32, device=x.device)
        _k_xq[(self.nb,)](x, xq, xd, self.nb, num_warps=1)
        if out is None:
            out = torch.empty(1, self.N, dtype=torch.float32, device=x.device)
        _k_q8[(triton.cdiv(self.N, BN),)](self.q, self.d, xq, xd, out, self.N, self.K, self.nb, BN=BN, BB=BB, num_warps=4)
        return out
