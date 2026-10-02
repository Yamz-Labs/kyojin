"""
Fused KDA forget gate (EXL3_KDA_G_FUSED). Replaces the torch chain

    gf = f + dt_bias                         (f fp16 -> fp32)
    g  = lower_bound * sigmoid(exp(A_log) * gf)          (safe gate)
    g  = -exp(A_log) * softplus(gf), threshold 20        (no lower bound)

with one Triton pass that reads f once and writes g once. Mode "cumsum" also folds in fla's
chunk_local_cumsum(g, chunk_size, scale = RCP_LN2), so chunk_kda can skip its own cumsum pass.
exp/log1p go through libdevice (ocml on ROCm), compiled without FMA contraction, to match torch.
Imported lazily: Triton compiles on first use.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gate(f, dtb, dec, LOWER: tl.constexpr, lower_bound):
    gf = f.to(tl.float32) + dtb[None, :]
    if LOWER:
        x = dec[None, :] * gf
        return lower_bound * (1.0 / (1.0 + libdevice.exp(-x)))
    else:
        sp = tl.where(gf > 20.0, gf, libdevice.log1p(libdevice.exp(gf)))
        return -dec[None, :] * sp


@triton.jit(do_not_specialize = ["T"])
def kda_gate_kernel(
    f_ptr, dtb_ptr, dec_ptr, g_ptr, lower_bound, scale,
    T, D: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, LOWER: tl.constexpr, CUMSUM: tl.constexpr,
):
    i_s, i_t, i_b = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m = (o_t[:, None] < T) & (o_s[None, :] < D)
    dtb = tl.load(dtb_ptr + o_s, mask = o_s < D, other = 0.0)
    dec = tl.load(dec_ptr + o_s // K, mask = o_s < D, other = 0.0)
    p = (i_b * T + o_t[:, None]) * D + o_s[None, :]
    f = tl.load(f_ptr + p, mask = m, other = 0.0)
    g = _gate(f, dtb, dec, LOWER, lower_bound)
    if CUMSUM:
        # Rows past T load f = 0 but must add 0 to the scan (fla's cumsum loads g = 0 there)
        g = tl.where(m, g, 0.0)
        g = tl.cumsum(g, axis = 0) * scale
    tl.store(g_ptr + p, g, mask = m)


def kda_gate(
    f: torch.Tensor,
    dt_bias: torch.Tensor,
    decay: torch.Tensor,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
    cumsum_chunk: int | None = None,
    cumsum_scale: float = 1.0,
    bs: int = 32,
    num_warps: int = 2,
) -> torch.Tensor:
    """
    f: (B, T, H*K) fp16/fp32 contiguous; dt_bias: (H*K,) fp32; decay: (H,) fp32 = exp(A_log).
    Returns g (B, T, H, K) fp32, chunk-local cumsummed (times cumsum_scale) if cumsum_chunk is set.
    bs = 32 keeps the scan bit-equal to fla's chunk_local_cumsum on gfx1151 (bs = 64: ~1e-4 abs drift).
    """
    assert f.is_contiguous()
    B, T, D = f.shape
    assert D == num_heads * head_dim
    g = torch.empty((B, T, num_heads, head_dim), dtype = torch.float, device = f.device)
    lower = lower_bound is not None
    lb = float(lower_bound) if lower else 0.0
    BT = cumsum_chunk or 64
    grid = (triton.cdiv(D, bs), triton.cdiv(T, BT), B)
    # enable_fp_fusion = False: with FMA contraction libdevice.exp drifts from torch.exp (HIP expf) by an ulp
    # on ~70% of inputs; without it the gate is bit-equal to the torch chain
    kda_gate_kernel[grid](
        f, dt_bias, decay, g, lb, float(cumsum_scale), T, D, head_dim, BT, bs, lower, cumsum_chunk is not None,
        num_warps = num_warps, enable_fp_fusion = False,
    )
    return g
