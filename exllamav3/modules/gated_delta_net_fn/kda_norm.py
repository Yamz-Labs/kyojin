"""
Fused KDA gated RMS norm (EXL3_KDA_NORM_FUSED). Replaces the torch fallback of ext.gated_rms_norm
(norm.cu is not built on ROCm) for the KDA form (sigmoid gate applied after the weighted norm):

    h = x.float() * rsqrt(mean(x.float()^2) + eps)
    y = (h * (w.float() + constant_bias) * sigmoid(g.float())).to(y.dtype)

~12 torch kernels with fp32 intermediates become one pass that reads x (bf16) and g once and writes y once.
Bit-exact vs the fallback: exp via libdevice without FMA contraction, rsqrt as fp64 1/sqrt (torch's is correctly
rounded), the row sum in torch's reduce_kernel order (NT = 32 threads x float4), and an explicit RNE fp16 rounding
that keeps the sign of zero (bitwise check via int16 views, scratch/dg/s6_check.py).
Imported lazily: Triton compiles on first use.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _pair(q, BR: tl.constexpr, N: tl.constexpr):
    return tl.sum(tl.reshape(q, (BR, N // 2, 2)), axis = 2)


@triton.jit
def _torch_rowsum(x_ptr, o_r, m, D: tl.constexpr, BR: tl.constexpr, NT: tl.constexpr):
    t = tl.arange(0, NT)
    a0 = tl.zeros((BR, NT), tl.float32); a1 = tl.zeros((BR, NT), tl.float32)
    a2 = tl.zeros((BR, NT), tl.float32); a3 = tl.zeros((BR, NT), tl.float32)
    for i in tl.static_range(D // (4 * NT)):
        b = o_r[:, None] * D + (i * NT + t[None, :]) * 4
        v = tl.load(x_ptr + b + 0, mask = m, other = 0.0).to(tl.float32); a0 += v * v
        v = tl.load(x_ptr + b + 1, mask = m, other = 0.0).to(tl.float32); a1 += v * v
        v = tl.load(x_ptr + b + 2, mask = m, other = 0.0).to(tl.float32); a2 += v * v
        v = tl.load(x_ptr + b + 3, mask = m, other = 0.0).to(tl.float32); a3 += v * v
    q = ((a0 + a1) + a2) + a3
    if NT >= 64: q = _pair(q, BR, 64)
    if NT >= 32: q = _pair(q, BR, 32)
    if NT >= 16: q = _pair(q, BR, 16)
    if NT >= 8: q = _pair(q, BR, 8)
    if NT >= 4: q = _pair(q, BR, 4)
    if NT >= 2: q = _pair(q, BR, 2)
    return tl.reshape(q, (BR,))


@triton.jit(do_not_specialize = ["R"])
def kda_norm_kernel(
    x_ptr, g_ptr, w_ptr, y_ptr, R, eps, cbias,
    D: tl.constexpr, BR: tl.constexpr, NT: tl.constexpr, SILU: tl.constexpr = False,
):
    i_r = tl.program_id(0).to(tl.int64)
    o_r = i_r * BR + tl.arange(0, BR)
    o_d = tl.arange(0, D)
    m = o_r[:, None] < R
    p = o_r[:, None] * D + o_d[None, :]
    xf = tl.load(x_ptr + p, mask = m, other = 0.0).to(tl.float32)
    if NT > 0:
        # torch's reduce_kernel order for a contiguous 128-wide row: NT threads, each loads float4 vectors
        # (4 accumulators, one per lane), folds them ((a0 + a1) + a2) + a3, then a pairwise tree across threads
        ss = _torch_rowsum(x_ptr, o_r, m, D, BR, NT)
    else:
        ss = tl.sum(xf * xf, axis = 1)
    var = ss / D + eps
    # torch.rsqrt is correctly rounded on ROCm; libdevice.rsqrt (v_rsq_f32) differs by an ulp on ~11% of inputs.
    # 1/sqrt in fp64 rounded to fp32 matches it bit for bit (scratch/dg/s6_exact.py); one per row, so cheap
    r = (1.0 / tl.sqrt(var.to(tl.float64))).to(tl.float32)
    h = xf * r[:, None]
    w = tl.load(w_ptr + o_d).to(tl.float32) + cbias
    h = h * w[None, :]
    gf = tl.load(g_ptr + p, mask = m, other = 0.0).to(tl.float32)
    if SILU:
        h = h * (gf / (1.0 + libdevice.exp(-gf)))  # torch silu: x / (1 + exp(-x))
    else:
        h = h * (1.0 / (1.0 + libdevice.exp(-gf)))
    if y_ptr.dtype.element_ty == tl.float16:
        # Triton's fp32 -> fp16 cast on gfx1151 is not round-to-nearest-even on ties (~1e3 of 16.7M values off by
        # an ulp, fp_downcast_rounding = "rtne" does not change it). Round in fp32 first so the cast is exact:
        # normal range via the integer RNE trick on the low 13 mantissa bits, subnormal range via the 0.5 add/sub trick
        u = h.to(tl.uint32, bitcast = True)
        u = ((u + 0xFFF + ((u >> 13) & 1)) >> 13) << 13
        hn = u.to(tl.float32, bitcast = True)
        a = tl.abs(h)
        hs = (a + 0.5) - 0.5  # ulp(0.5) = 2^-24: RNE to the fp16 subnormal grid
        # restore the sign from the bits, not via h < 0: -0.0 must stay -0.0 (x == 0 with w < 0, or an underflowed gate)
        hs = (hs.to(tl.uint32, bitcast = True) | ((u >> 31) << 31)).to(tl.float32, bitcast = True)
        h = tl.where(a < 6.103515625e-05, hs, hn)
    tl.store(y_ptr + p, h.to(y_ptr.dtype.element_ty), mask = m)


def kda_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    eps: float,
    constant_bias: float = 0.0,
    br: int = 8,
    num_warps: int = 2,
    nt: int = 32,
    silu: bool = False,
) -> None:
    """
    x: (..., D) bf16 contiguous; g: same shape, fp16/bf16/fp32 contiguous; w: (D,) bf16/fp32;
    y: same shape as x, fp16/fp32 (written in place). One weight row (groups = 1), gate after the norm.
    """
    D = x.shape[-1]
    assert x.is_contiguous() and g.is_contiguous() and y.is_contiguous()
    assert x.shape == g.shape == y.shape and w.numel() == D and (D & (D - 1)) == 0
    R = x.numel() // D
    grid = (triton.cdiv(R, br),)
    # enable_fp_fusion = False: no FMA contraction, so x*x, the sum and the scaling match torch's separate kernels
    kda_norm_kernel[grid](
        x, g, w, y, R, float(eps), float(constant_bias), D, br, nt, silu,
        num_warps = num_warps, enable_fp_fusion = False,
    )
