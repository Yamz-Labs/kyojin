"""Shared-expert gate tail for MoE decode: out[r, k] = routed[r, k] + float(se[r, k]) * sigmoid(sum_j x[r, j] * w[j]), fp32,
one program per row (decode rows 1..16). Replaces cast + mul + sum + sigmoid + mul + add (6 launches). Written in place into routed."""
import torch
import triton
import triton.language as tl


@triton.jit
def _shgate_tail(x_ptr, w_ptr, se_ptr, out_ptr, K, BK: tl.constexpr):
    r = tl.program_id(0)
    acc = tl.zeros((BK,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        o = k0 + tl.arange(0, BK)
        m = o < K
        acc += tl.load(x_ptr + r * K + o, mask=m, other=0.0).to(tl.float32) * tl.load(w_ptr + o, mask=m, other=0.0)
    g = tl.sigmoid(tl.sum(acc, axis=0))
    for k0 in range(0, K, BK):
        o = k0 + tl.arange(0, BK)
        m = o < K
        y = tl.load(out_ptr + r * K + o, mask=m, other=0.0) + tl.load(se_ptr + r * K + o, mask=m, other=0.0).to(tl.float32) * g
        tl.store(out_ptr + r * K + o, y, mask=m)


def shgate_tail_(routed: torch.Tensor, se: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """routed (R, K) fp32 contiguous, updated in place and returned; se (R, K) half; x (R, K) half; w (K,) fp32."""
    R, K = routed.shape
    _shgate_tail[(R,)](x, w, se, routed, K, BK=1024)
    return routed
