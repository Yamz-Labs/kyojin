# KDA inter-solve split (exllamav3, kdaC6; not from fla).
#
# fla's chunk_kda_fwd_kernel_inter_solve_fused keeps 12 fp32 16x16 accumulators (Aqk and Akk for the 6 sub-chunk
# pairs) live across its whole K loop: on gfx1151 that is 256 VGPR and ~177 spilled registers per 2-wave program.
# Here the work runs in two kernels:
#  1. inter_off: row block by row block (I = 1, 2, 3), one [16, K] x [K, NJ] product of the gated rows of block I
#     against all earlier rows of the chunk (NJ = 16, 32, 64; the unused 16 columns of I = 3 are masked to zero), so
#     only two [16, NJ] accumulators are live at a time (no spills). Per element the math and the K order are the
#     vendor's: (x * exp2(g_x - g_n)) . (k_j * exp2(g_n - g_j)), BK tiles summed in order. Stores the off-diagonal
#     Aqk (scaled) and the beta-scaled off-diagonal Akk blocks in an fp32 scratch.
#  2. inter_merge: the vendor's forward substitution on the four diagonal blocks and its block merge, unchanged,
#     reading the off-diagonal Akk from the scratch.
# With the solve at 2 warps the result is bitwise equal to the vendor kernel at BK16 w2 (the pinned engine config).
# Fixed-length batches, BT = 64, BC = 16, USE_SAFE_GATE = False only.

import os
import torch
import triton
import triton.language as tl
from .op import exp2
from .kda_chunk_intra import SOLVE_TRIL_DOT_PRECISION


@triton.jit
def _row(q, k, g, beta, Aqk, Aoff, scale, T, i_tc0, I: tl.constexpr, NJ: tl.constexpr, H: tl.constexpr,
         HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr):
    i_tcx = i_tc0 + I * BC
    if i_tcx < T:
        o_i = tl.arange(0, BC)
        o_n = tl.arange(0, NJ)
        o_x = i_tcx + o_i
        m_x = o_x < T
        o_j = i_tc0 + o_n
        m_j = (o_n < I * BC) & (o_j < T)
        b_aq = tl.zeros([BC, NJ], dtype=tl.float32)
        b_ak = tl.zeros([BC, NJ], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_xk = m_x[:, None] & m_k[None, :]
            m_jk = m_j[:, None] & m_k[None, :]
            b_q = tl.load(q + o_x[:, None] * (H*K) + o_k[None, :], mask=m_xk, other=0.0).to(tl.float32)
            b_k = tl.load(k + o_x[:, None] * (H*K) + o_k[None, :], mask=m_xk, other=0.0).to(tl.float32)
            b_g = tl.load(g + o_x[:, None] * (HV*K) + o_k[None, :], mask=m_xk, other=0.0).to(tl.float32)
            b_gn = tl.load(g + i_tcx * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
            b_kj = tl.load(k + o_j[:, None] * (H*K) + o_k[None, :], mask=m_jk, other=0.0).to(tl.float32)
            b_gj = tl.load(g + o_j[:, None] * (HV*K) + o_k[None, :], mask=m_jk, other=0.0).to(tl.float32)
            b_gq = tl.where(m_x[:, None], exp2(b_g - b_gn[None, :]), 0)
            b_kgt = tl.trans(tl.where(m_j[:, None], b_kj * exp2(b_gn[None, :] - b_gj), 0))
            b_aq += tl.dot(b_q * b_gq, b_kgt)
            b_ak += tl.dot(b_k * b_gq, b_kgt)
        m_s = m_x[:, None] & (o_n < I * BC)[None, :]
        tl.store(Aqk + o_x[:, None] * (HV*BT) + o_n[None, :], (b_aq * scale).to(Aqk.dtype.element_ty), mask=m_s)
        b_b = tl.load(beta + o_x * HV, mask=m_x, other=0.0).to(tl.float32)
        tl.store(Aoff + o_x[:, None] * (HV*BT) + o_n[None, :], b_ak * b_b[:, None], mask=m_s)


@triton.jit(do_not_specialize=['T'])
def inter_off_kernel(q, k, g, beta, Aqk, Aoff, scale, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                     BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, GSW: tl.constexpr = False):
    if GSW:  # head-fastest grid: concurrent programs spread over heads (row strides are 16/32 KB -> channel camping)
        i_t, i_bh = tl.program_id(1).to(tl.int64), tl.program_id(0)
    else:
        i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    bos = i_b * T
    i_tc0 = i_t * BT
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    Aqk += (bos * HV + i_hv) * BT
    Aoff += (bos * HV + i_hv) * BT
    beta += bos * HV + i_hv
    _row(q, k, g, beta, Aqk, Aoff, scale, T, i_tc0, 1, 16, H, HV, K, BT, BC, BK)
    _row(q, k, g, beta, Aqk, Aoff, scale, T, i_tc0, 2, 32, H, HV, K, BT, BC, BK)
    _row(q, k, g, beta, Aqk, Aoff, scale, T, i_tc0, 3, 64, H, HV, K, BT, BC, BK)


@triton.jit(do_not_specialize=['T'])
def inter_merge_kernel(Akkd, Akk, Aoff, T, HV: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                       GSW: tl.constexpr = False):
    # vendor forward substitution + block merge (NC = 4, USE_SAFE_GATE = False); off-diagonal Akk from Aoff
    if GSW:
        i_t, i_bh = tl.program_id(1).to(tl.int64), tl.program_id(0)
    else:
        i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    bos = i_b * T
    i_tc0 = i_t * BT
    Akkd += (bos * HV + i_hv) * BC
    Akk += (bos * HV + i_hv) * BT
    Aoff += (bos * HV + i_hv) * BT
    o_i = tl.arange(0, BC)
    o_c0 = i_tc0 + o_i
    o_c1 = i_tc0 + BC + o_i
    o_c2 = i_tc0 + 2*BC + o_i
    o_c3 = i_tc0 + 3*BC + o_i
    m_A0 = (o_c0 < T)[:, None] & (o_i[None, :] < BT)
    m_A1 = (o_c1 < T)[:, None] & (o_i[None, :] < BT)
    m_A2 = (o_c2 < T)[:, None] & (o_i[None, :] < BT)
    m_A3 = (o_c3 < T)[:, None] & (o_i[None, :] < BT)
    b_Akk10 = tl.load(Aoff + o_c1[:, None] * (HV*BT) + o_i[None, :], mask=m_A1, other=0.0)
    b_Akk20 = tl.load(Aoff + o_c2[:, None] * (HV*BT) + o_i[None, :], mask=m_A2, other=0.0)
    b_Akk21 = tl.load(Aoff + o_c2[:, None] * (HV*BT) + (o_i + BC)[None, :], mask=m_A2, other=0.0)
    b_Akk30 = tl.load(Aoff + o_c3[:, None] * (HV*BT) + o_i[None, :], mask=m_A3, other=0.0)
    b_Akk31 = tl.load(Aoff + o_c3[:, None] * (HV*BT) + (o_i + BC)[None, :], mask=m_A3, other=0.0)
    b_Akk32 = tl.load(Aoff + o_c3[:, None] * (HV*BT) + (o_i + 2*BC)[None, :], mask=m_A3, other=0.0)
    b_Ai00 = tl.load(Akkd + o_c0[:, None] * (HV*BC) + o_i[None, :], mask=m_A0, other=0.0).to(tl.float32)
    b_Ai11 = tl.load(Akkd + o_c1[:, None] * (HV*BC) + o_i[None, :], mask=m_A1, other=0.0).to(tl.float32)
    b_Ai22 = tl.load(Akkd + o_c2[:, None] * (HV*BC) + o_i[None, :], mask=m_A2, other=0.0).to(tl.float32)
    b_Ai33 = tl.load(Akkd + o_c3[:, None] * (HV*BC) + o_i[None, :], mask=m_A3, other=0.0).to(tl.float32)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_Ai00 = -tl.where(m_A, b_Ai00, 0)
    b_Ai11 = -tl.where(m_A, b_Ai11, 0)
    b_Ai22 = -tl.where(m_A, b_Ai22, 0)
    b_Ai33 = -tl.where(m_A, b_Ai33, 0)
    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
        b_a00 = tl.where(o_i < i, b_a00, 0.)
        b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(BC + 2, min(2*BC, T - i_tc0)):
        b_a11 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
        b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
        b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
    for i in range(2*BC + 2, min(3*BC, T - i_tc0)):
        b_a22 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
        b_a22 = tl.where(o_i < i - 2*BC, b_a22, 0.)
        b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i - 2*BC)[:, None], b_a22, b_Ai22)
    for i in range(3*BC + 2, min(4*BC, T - i_tc0)):
        b_a33 = -tl.load(Akkd + (i_tc0 + i) * HV*BC + o_i)
        b_a33 = tl.where(o_i < i - 3*BC, b_a33, 0.)
        b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i - 3*BC)[:, None], b_a33, b_Ai33)
    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    P: tl.constexpr = SOLVE_TRIL_DOT_PRECISION
    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_Akk10, input_precision=P), b_Ai00, input_precision=P)
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_Akk21, input_precision=P), b_Ai11, input_precision=P)
    b_Ai20 = -tl.dot(b_Ai22, tl.dot(b_Akk20, b_Ai00, input_precision=P) +
                     tl.dot(b_Akk21, b_Ai10, input_precision=P), input_precision=P)
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_Akk32, input_precision=P), b_Ai22, input_precision=P)
    b_Ai31 = -tl.dot(b_Ai33, tl.dot(b_Akk31, b_Ai11, input_precision=P) +
                     tl.dot(b_Akk32, b_Ai21, input_precision=P), input_precision=P)
    b_Ai30 = -tl.dot(b_Ai33, tl.dot(b_Akk30, b_Ai00, input_precision=P) +
                     tl.dot(b_Akk31, b_Ai10, input_precision=P) +
                     tl.dot(b_Akk32, b_Ai20, input_precision=P), input_precision=P)

    ty = Akk.dtype.element_ty
    p0 = Akk + o_c0[:, None] * (HV*BT) + o_i[None, :]
    p1 = Akk + o_c1[:, None] * (HV*BT) + o_i[None, :]
    p2 = Akk + o_c2[:, None] * (HV*BT) + o_i[None, :]
    p3 = Akk + o_c3[:, None] * (HV*BT) + o_i[None, :]
    tl.store(p0, b_Ai00.to(ty), mask=m_A0)
    tl.store(p1, b_Ai10.to(ty), mask=m_A1)
    tl.store(p1 + BC, b_Ai11.to(ty), mask=m_A1)
    tl.store(p2, b_Ai20.to(ty), mask=m_A2)
    tl.store(p2 + BC, b_Ai21.to(ty), mask=m_A2)
    tl.store(p2 + 2*BC, b_Ai22.to(ty), mask=m_A2)
    tl.store(p3, b_Ai30.to(ty), mask=m_A3)
    tl.store(p3 + BC, b_Ai31.to(ty), mask=m_A3)
    tl.store(p3 + 2*BC, b_Ai32.to(ty), mask=m_A3)
    tl.store(p3 + 3*BC, b_Ai33.to(ty), mask=m_A3)


# (off BK, off warps, merge warps, head-fastest grid, HIP off kernel) per knob value; kdaC6/kdaC7 sweeps on GLM
# td205 captures (scratch/kdac6, scratch/kdac7)
SPLIT_CFG = {1: (16, 2, 2, False, False),
             2: (16, 2, 4, False, False),  # merge at 4 warps, 0 spills, not bitwise (bench only)
             3: (16, 2, 2, True, True),    # kdaC7: HIP off kernel (exllamav3_ext.kda_inter_off) + head-fastest merge
             4: (16, 2, 2, True, False)}   # kdaC7: both Triton kernels on the head-fastest grid


def inter_split_mode(T, BT, BC, cu_seqlens=None, safe_gate=False):
    """EXL3_KDA_IS_SPLIT (read per call): 0 = fla inter_solve_fused, 1 = split kernels, merge at 2 warps
    (bitwise equal to the pinned BK16 w2 vendor launch), 2 = merge at 4 warps (no spills, not bitwise; bench -66 us at T2048, not engine-checked),
    3 (default) = HIP off-diagonal kernel + merge on a head-fastest grid, 4 = both Triton kernels on a head-fastest grid (3 and 4
    bitwise equal to 1; the head-fastest grid avoids memory-channel camping from the 16/32 KB row strides)."""
    m = int(os.environ.get("EXL3_KDA_IS_SPLIT", "3") or 0)
    if m and (cu_seqlens is not None or safe_gate or BT != 64 or BC != 16):
        return 0
    return m


def inter_solve_split(q, k, gk, beta, Aqk, Akkd, Akk, scale, mode):
    """Off-diagonal Aqk into Aqk, full Akk^-1 into Akk (same contract as fla's inter_solve_fused launch)."""
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT, BC = 64, 16
    bk, nw, mnw, gsw, hip = SPLIT_CFG[mode]
    Aoff = torch.empty(B, T, HV, BT, device=k.device, dtype=torch.float32)
    grid = (B * HV, triton.cdiv(T, BT)) if gsw else (triton.cdiv(T, BT), B * HV)
    if hip and K == 128 and all(x.is_contiguous() for x in (q, k, gk, beta, Aqk)) and \
            q.dtype == k.dtype == beta.dtype == Aqk.dtype == torch.bfloat16 and gk.dtype == torch.float32:
        from ...ext import exllamav3_ext as ext
        ext.kda_inter_off(q, k, gk, beta, Aqk, Aoff, float(scale))
        k1 = None
    else:
        k1 = inter_off_kernel[grid](q, k, gk, beta, Aqk, Aoff, scale, T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=bk,
                                    GSW=gsw, num_warps=nw, num_stages=1)
    k2 = inter_merge_kernel[grid](Akkd, Akk, Aoff, T, HV=HV, BT=BT, BC=BC, GSW=gsw, num_warps=mnw, num_stages=1)
    return k1, k2
