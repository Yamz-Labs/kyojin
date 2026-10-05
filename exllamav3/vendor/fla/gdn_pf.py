# EXL3_GDN_PF=1 (default 1): GDN prefill entry that takes q, k, v as strided slices of the conv output (no input_guard copies of q, k, v:
# three full passes, 168 MB per layer-chunk at 4096 rows) and no single-tensor cat copy of the result. The q / k l2norm reads the strided rows
# directly (same Triton body and the same pinned (BT, num_warps) as l2norm_fwd_kernel: only the load address differs, so the bits are the same),
# and the fused HIP kernel reads v through its row pitch. Falls back to the stock chain on the same inputs made contiguous.
import torch
import triton
import triton.language as tl
from .l2norm import L2_BT, L2_WARPS

BT_LIST = [8, 16, 32, 64, 128]


@triton.jit(do_not_specialize=["T"])
def l2norm_strided_kernel(x, y, eps, T, S, D: tl.constexpr, H: tl.constexpr, BD: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr):
    # rows o_t = token * H + head; input row address token * S + head * D (S = token pitch in elements), output contiguous
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T * H
    m_x = m_t[:, None] & (o_d[None, :] < D)
    p_x = x + (o_t // H)[:, None] * S + (o_t % H)[:, None] * D + o_d[None, :]
    p_y = y + o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)


def strided_ok(x):
    B, T, H, D = x.shape
    return x.stride(3) == 1 and x.stride(2) == D and x.stride(1) >= H * D and (B == 1 or x.stride(0) == T * x.stride(1)) \
        and x.stride(1) % 8 == 0 and x.storage_offset() % 8 == 0


def l2norm_strided(x, eps=1e-6):
    B, T, H, D = x.shape
    assert strided_ok(x) and D == 128
    y = torch.empty((B, T, H, D), dtype=x.dtype, device=x.device)
    rows = B * T * H
    # rows of batch item b start at b * T * S; for B > 1 the flat row -> (token, head) map still holds because stride(0) == T * stride(1)
    l2norm_strided_kernel[(triton.cdiv(rows, L2_BT),)](
        x=x, y=y, eps=eps, T=B * T, S=x.stride(1), D=D, H=H, BD=triton.next_power_of_2(D), NB=triton.cdiv(B * T, 2048 * 32),
        BT=L2_BT, num_warps=L2_WARPS)
    return y
