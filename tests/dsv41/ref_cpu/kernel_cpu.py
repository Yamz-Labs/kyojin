"""
Pure-torch CPU emulation of the tilelang kernels in DeepSeek-V4.1-Flash inference/kernel.py
(vendored as official/kernel_tilelang_UNUSED.py.txt). Same function names / signatures so the
official model.py runs unmodified on CPU. Written the "obvious" way (frexp, tables), deliberately
different from exllamav3/dsv41/quant.py (bit tricks) so the two cross-check each other.
Emulation is exact for the elementwise quantisers; the GEMMs dequantise then multiply in fp32
(the kernels accumulate per-32 block in fp32 then scale; differences are accumulation order only).
"""
import torch
from typing import Optional

FP4_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_ALL = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _pow2_ceil_scale(amax, qmax_inv):
    t = amax * torch.tensor(qmax_inv, dtype=torch.float32)   # f32 multiply, as in the kernel
    m, e = torch.frexp(t)                                     # t = m * 2**e, m in [0.5, 1)
    ce = torch.where(m == 0.5, e - 1, e)                      # ceil(log2 t)
    return torch.ldexp(torch.ones_like(t), ce)


def _fp4_nearest_even(v):
    """v in [-6, 6] -> nearest e2m1 value, ties to the even code."""
    out = torch.empty_like(v)
    a = v.abs()
    codes = torch.arange(8)
    d = (a.unsqueeze(-1) - FP4_POS).abs()                      # [..., 8]
    mn = d.amin(-1, keepdim=True)
    cand = (d == mn)
    # among tied candidates prefer the even code
    even = (codes % 2 == 0)
    score = cand.float() * (1 + even.float())
    idx = score.argmax(-1)
    return torch.copysign(FP4_POS[idx], v)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float().view(-1, N // block_size, block_size)
    amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _pow2_ceil_scale(amax, 1.0 / 448)
    else:
        s = amax * torch.tensor(1.0 / 448, dtype=torch.float32)
    q = (z / s).clamp(-448.0, 448.0)
    if inplace:
        y = q.to(torch.float8_e4m3fn).float() * s
        x.copy_(y.reshape(x.shape).to(x.dtype))
        return x
    return (q.reshape(x.shape).to(torch.float8_e4m3fn),
            s.squeeze(-1).reshape(*x.shape[:-1], N // block_size).to(scale_dtype))


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float().view(-1, N // block_size, block_size)
    amax = z.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0 ** -9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0 ** -126)
        s = _pow2_ceil_scale(amax, 1.0 / 6.0)
    q = _fp4_nearest_even((z / s).clamp(-6.0, 6.0)) * s
    assert inplace, "not needed by the model"
    x.copy_(q.reshape(x.shape).to(x.dtype))
    return x


def _deq_fp8_w(b, b_s, block):
    N, K = b.shape
    s = b_s.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return b.float() * s


def _deq_a(a, a_s, block):
    K = a.size(-1)
    s = a_s.float().reshape(-1, K // block).repeat_interleave(block, -1)
    return a.float().reshape(-1, K) * s


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    A = _deq_a(a, a_s, block_size)
    B = _deq_fp8_w(b, b_s, block_size)
    c = (A @ B.T).to(torch.get_default_dtype())
    return c.reshape(*a.shape[:-1], b.size(0))


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    A = _deq_a(a, a_s, act_block_size)
    bb = b.view(torch.uint8)
    lo, hi = bb & 0x0F, bb >> 4
    B = torch.stack([FP4_ALL[lo.long()], FP4_ALL[hi.long()]], -1).flatten(-2)
    B = B * b_s.float().repeat_interleave(32, -1)
    c = (A @ B.T).to(torch.get_default_dtype())
    return c.reshape(*a.shape[:-1], b.size(0))


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps                              # rows over k
    comb = comb / (comb.sum(-2, keepdim=True) + eps)           # columns
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, s, h, d = q.shape
    valid = topk_idxs >= 0                                      # [b,s,k]
    idx = topk_idxs.clamp_min(0).long()
    g = torch.gather(kv.unsqueeze(1).expand(b, s, -1, d), 2, idx.unsqueeze(-1).expand(-1, -1, -1, d))  # [b,s,k,d]
    sc = torch.einsum("bshd,bskd->bshk", q.float(), g.float()) * softmax_scale
    sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = sc.amax(-1, keepdim=True).clamp_min(-1e30)
    p = torch.exp(sc - mx)
    den = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, h, 1) - mx)
    o = torch.einsum("bshk,bskd->bshd", p, g.float()) / den
    return o.to(q.dtype)
