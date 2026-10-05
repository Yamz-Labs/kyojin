"""
Fused prefill pieces of GatedResidual._mix (EXL3_GR_PF_FUSE). On ROCm the grouped rms_norm
(w_groups = hc_mult) falls back to a chunked torch chain (~8 fp32 kernels over R*H*D elements),
and the gate application (sigmoid(g) * normed, mean over streams) is 5 more. Two Triton passes:

    gr_norm : normed (R*H, D) fp16 = x.float() * rsqrt(mean(x^2) + eps) * w[row % H]
    gr_gate : mixed  (R, D)   fp16 = mean_h(sigmoid(g[r, h*D + d]) * normed[r*H + h, d])

fp32 math, RNE fp16 rounding (Triton's fp32 -> fp16 cast is not RNE on gfx1151 ties, see kda_norm.py).
"""
import os
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rne16(h):
    u = h.to(tl.uint32, bitcast = True)
    u = ((u + 0xFFF + ((u >> 13) & 1)) >> 13) << 13
    hn = u.to(tl.float32, bitcast = True)
    a = tl.abs(h)
    hs = (a + 0.5) - 0.5
    hs = (hs.to(tl.uint32, bitcast = True) | ((u >> 31) << 31)).to(tl.float32, bitcast = True)
    return tl.where(a < 6.103515625e-05, hs, hn)


@triton.jit
def _gr_norm_kernel(x_ptr, w_ptr, y_ptr, eps, YP, D: tl.constexpr, H: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BD)
    m = o < D
    x = tl.load(x_ptr + r * D + o, mask = m, other = 0.0).to(tl.float32)
    ss = tl.sum(x * x, axis = 0)
    rs = (1.0 / tl.sqrt((ss / D + eps).to(tl.float64))).to(tl.float32)
    w = tl.load(w_ptr + (r % H) * D + o, mask = m, other = 0.0).to(tl.float32)
    h = _rne16(x * rs * w)
    # y row r = token * H + stream lands at token * YP + stream * D (YP = H * D: contiguous (R*H, D))
    tl.store(y_ptr + (r // H) * YP + (r % H) * D + o, h.to(tl.float16), mask = m)


@triton.jit
def _gr_apply_norm_kernel(x_ptr, y_ptr, post_ptr, w_ptr, n_ptr, eps, YP, D: tl.constexpr, H: tl.constexpr,
                          BD: tl.constexpr, FMA: tl.constexpr):
    # EXL3_PF_GR_FUSE: x[r] += post[r] * y[r // H] (hc_apply, no comb) fused with gr_norm on the new x row.
    # Same row program, BD, num_warps and reduction as _gr_norm_kernel, so the norm is bit-identical.
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BD)
    m = o < D
    x = tl.load(x_ptr + r * D + o, mask = m, other = 0.0)
    y = tl.load(y_ptr + (r // H) * D + o, mask = m, other = 0.0).to(tl.float32)
    p = tl.load(post_ptr + r)
    if FMA:
        x = tl.fma(p, y, x)
    else:
        x = x + p * y
    tl.store(x_ptr + r * D + o, x, mask = m)
    ss = tl.sum(x * x, axis = 0)
    rs = (1.0 / tl.sqrt((ss / D + eps).to(tl.float64))).to(tl.float32)
    w = tl.load(w_ptr + (r % H) * D + o, mask = m, other = 0.0).to(tl.float32)
    h = _rne16(x * rs * w)
    tl.store(n_ptr + (r // H) * YP + (r % H) * D + o, h.to(tl.float16), mask = m)


@triton.jit
def _gr_gate_kernel(g_ptr, n_ptr, y_ptr, D: tl.constexpr, H: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    o = tl.program_id(1) * BD + tl.arange(0, BD)
    m = o < D
    acc = tl.zeros((BD,), tl.float32)
    for h in tl.static_range(H):
        g = tl.load(g_ptr + r * (H * D) + h * D + o, mask = m, other = 0.0).to(tl.float32)
        n = tl.load(n_ptr + (r * H + h) * D + o, mask = m, other = 0.0).to(tl.float32)
        acc += (1.0 / (1.0 + libdevice.exp(-g))) * n
    tl.store(y_ptr + r * D + o, _rne16(acc / H).to(tl.float16), mask = m)


def gr_norm(x: torch.Tensor, w_h: torch.Tensor, y: torch.Tensor, eps: float, H: int, y_pitch: int = 0):
    """x (R*H, D) fp32 contiguous, w_h (H*D,) fp16, y (R*H, D) fp16 contiguous, or y_pitch = token row pitch (>= H*D)
    of a (R, y_pitch) fp16 buffer y that receives the rows of one token side by side."""
    rows, D = x.shape
    if y_pitch:
        assert y_pitch >= H * D and y.dtype == torch.float16 and y.dim() == 2 and y.shape[0] == rows // H and y.shape[1] == y_pitch and y.is_contiguous()
    else:
        y_pitch = H * D
    _gr_norm_kernel[(rows,)](x, w_h, y, float(eps), y_pitch, D, H, triton.next_power_of_2(D),
                             num_warps = 8, enable_fp_fusion = False)


def gr_gate(g: torch.Tensor, normed: torch.Tensor, y: torch.Tensor, H: int):
    """g (R, H*D) fp16 contiguous, normed (R*H, D) fp16, y (R, D) fp16."""
    R, D = y.shape
    BD = 1024
    _gr_gate_kernel[(R, triton.cdiv(D, BD))](g, normed, y, D, H, BD, num_warps = 4, enable_fp_fusion = False)


def gr_apply_norm(x: torch.Tensor, y: torch.Tensor, post: torch.Tensor, w_h: torch.Tensor, normed: torch.Tensor,
                  eps: float, H: int, fma: bool = True, y_pitch: int = 0):
    """x (R*H, D) fp32 contiguous, updated in place (x += post * y); y (R, D) fp16/fp32, post (R*H,) fp32,
    normed (R*H, D) fp16 = gr_norm of the new x."""
    rows, D = x.shape
    if y_pitch:
        assert y_pitch >= H * D and normed.dtype == torch.float16 and normed.dim() == 2 and normed.shape[0] == rows // H and normed.shape[1] == y_pitch and normed.is_contiguous()
    else:
        y_pitch = H * D
    # fma=True is bit-identical to ext.hc_apply (hipcc contracts post * y + x; checked on GPU, R=1..2048)
    _gr_apply_norm_kernel[(rows,)](x, y, post, w_h, normed, float(eps), y_pitch, D, H, triton.next_power_of_2(D), fma,
                                   num_warps = 8, enable_fp_fusion = False)
