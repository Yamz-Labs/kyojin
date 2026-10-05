# Shared-Gram variant of the fused kkt + solve_tril kernel (gdn_chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel).
# GENERATED from that kernel's source by a kernel generator script (not published) (do not edit by hand). One program handles one (chunk, k-head) and
# loops over the HV // H value heads that share the k-head: the 10 raw k k^T blocks (fp32 tl.dot chains) are computed once instead of
# once per value head. Everything per value head (gate, beta, exp2 scaling, forward substitution, block merge, store) is the original text,
# so the stored A is bitwise equal. ABL (timing experiments only, 0 in service): 1 = skip forward substitution.
import torch
import triton
import triton.language as tl
from .op import exp2
from .utils import autotune_cache_kwargs
from .gdn_chunk_fwd import SOLVE_TRIL_DOT_PRECISION


@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=['H', 'HV', 'K', 'BC'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def kkt_solve_shared_kernel(
    k,
    g,
    beta,
    A,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    ABL: tl.constexpr,
):
    i_t, i_bk = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hk = i_bk // H, i_bk % H
    bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_hk) * K

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # Step 1 (once per k-head): the 10 raw lower-triangular [BC, BC] blocks of K @ K^T
    # 4 diagonal blocks
    r_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    r_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    r_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    r_A33 = tl.zeros([BC, BC], dtype=tl.float32)

    # 6 off-diagonal blocks
    r_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    r_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    r_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    r_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    r_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    r_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k0 = k + (i_tc0 + o_i)[:, None] * (H*K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_tc0[:, None] & (o_k[None, :] < K), other=0.0)
        # diagonal block 0
        r_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = k + (i_tc1 + o_i)[:, None] * (H*K) + o_k[None, :]
            b_k1 = tl.load(p_k1, mask=m_tc1[:, None] & (o_k[None, :] < K), other=0.0)
            # diagonal block 1
            r_A11 += tl.dot(b_k1, tl.trans(b_k1))
            # off-diagonal (1,0)
            r_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = k + (i_tc2 + o_i)[:, None] * (H*K) + o_k[None, :]
                b_k2 = tl.load(p_k2, mask=m_tc2[:, None] & (o_k[None, :] < K), other=0.0)
                # diagonal block 2
                r_A22 += tl.dot(b_k2, tl.trans(b_k2))
                # off-diagonal (2,0), (2,1)
                r_A20 += tl.dot(b_k2, tl.trans(b_k0))
                r_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = k + (i_tc3 + o_i)[:, None] * (H*K) + o_k[None, :]
                    b_k3 = tl.load(p_k3, mask=m_tc3[:, None] & (o_k[None, :] < K), other=0.0)
                    # diagonal block 3
                    r_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    # off-diagonal (3,0), (3,1), (3,2)
                    r_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    r_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    r_A32 += tl.dot(b_k3, tl.trans(b_k2))

    for i_g in range(HV // H):
        i_h = i_hk * (HV // H) + i_g
        A_h = A + (bos * HV + i_h) * BT

        # load beta / gate for each sub-chunk
        p_b0 = beta + bos * HV + i_h + (i_tc0 + o_i) * HV
        p_b1 = beta + bos * HV + i_h + (i_tc1 + o_i) * HV
        p_b2 = beta + bos * HV + i_h + (i_tc2 + o_i) * HV
        p_b3 = beta + bos * HV + i_h + (i_tc3 + o_i) * HV
        b_b0 = tl.load(p_b0, mask=m_tc0, other=0.0).to(tl.float32)
        b_b1 = tl.load(p_b1, mask=m_tc1, other=0.0).to(tl.float32)
        b_b2 = tl.load(p_b2, mask=m_tc2, other=0.0).to(tl.float32)
        b_b3 = tl.load(p_b3, mask=m_tc3, other=0.0).to(tl.float32)

        p_g0 = g + bos * HV + i_h + (i_tc0 + o_i) * HV
        p_g1 = g + bos * HV + i_h + (i_tc1 + o_i) * HV
        p_g2 = g + bos * HV + i_h + (i_tc2 + o_i) * HV
        p_g3 = g + bos * HV + i_h + (i_tc3 + o_i) * HV

        b_g0 = tl.load(p_g0, mask=m_tc0, other=0.0).to(tl.float32)
        b_g1 = tl.load(p_g1, mask=m_tc1, other=0.0).to(tl.float32)
        b_g2 = tl.load(p_g2, mask=m_tc2, other=0.0).to(tl.float32)
        b_g3 = tl.load(p_g3, mask=m_tc3, other=0.0).to(tl.float32)

        # apply gate, beta scaling, and masking
        # m_d: strictly lower triangular mask for diagonal blocks
        # m_tc: boundary mask to prevent NaN from 0 * inf (IEEE 754) when
        #   out-of-bounds g loads as 0 via boundary_check and exp2(0 - g_inbounds) overflows
        m_d = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_A00 = r_A00 * tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.)
        b_A11 = r_A11 * tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], exp2(b_g1[:, None] - b_g1[None, :]), 0.)
        b_A22 = r_A22 * tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], exp2(b_g2[:, None] - b_g2[None, :]), 0.)
        b_A33 = r_A33 * tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], exp2(b_g3[:, None] - b_g3[None, :]), 0.)

        b_A10 = r_A10 * tl.where(m_tc1[:, None] & m_tc0[None, :], exp2(b_g1[:, None] - b_g0[None, :]), 0.)
        b_A20 = r_A20 * tl.where(m_tc2[:, None] & m_tc0[None, :], exp2(b_g2[:, None] - b_g0[None, :]), 0.)
        b_A21 = r_A21 * tl.where(m_tc2[:, None] & m_tc1[None, :], exp2(b_g2[:, None] - b_g1[None, :]), 0.)
        b_A30 = r_A30 * tl.where(m_tc3[:, None] & m_tc0[None, :], exp2(b_g3[:, None] - b_g0[None, :]), 0.)
        b_A31 = r_A31 * tl.where(m_tc3[:, None] & m_tc1[None, :], exp2(b_g3[:, None] - b_g1[None, :]), 0.)
        b_A32 = r_A32 * tl.where(m_tc3[:, None] & m_tc2[None, :], exp2(b_g3[:, None] - b_g2[None, :]), 0.)

        # diagonal blocks: scaled by beta
        b_A00 = b_A00 * b_b0[:, None]
        b_A11 = b_A11 * b_b1[:, None]
        b_A22 = b_A22 * b_b2[:, None]
        b_A33 = b_A33 * b_b3[:, None]

        # off-diagonal blocks: full block, scaled by beta
        b_A10 = b_A10 * b_b1[:, None]
        b_A20 = b_A20 * b_b2[:, None]
        b_A21 = b_A21 * b_b2[:, None]
        b_A30 = b_A30 * b_b3[:, None]
        b_A31 = b_A31 * b_b3[:, None]
        b_A32 = b_A32 * b_b3[:, None]

        b_Ai00 = -b_A00
        b_Ai11 = -b_A11
        b_Ai22 = -b_A22
        b_Ai33 = -b_A33

        for i in range(2, tl.minimum(BC, T - i_tc0) * (1 - (ABL & 1))):
            b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(2, tl.minimum(BC, T - i_tc1) * (1 - (ABL & 1))):
            b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.), 0)
            b_a11 = tl.where(o_i < i, b_a11, 0.)
            b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
        for i in range(2, tl.minimum(BC, T - i_tc2) * (1 - (ABL & 1))):
            b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.), 0)
            b_a22 = tl.where(o_i < i, b_a22, 0.)
            b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
        for i in range(2, tl.minimum(BC, T - i_tc3) * (1 - (ABL & 1))):
            b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.), 0)
            b_a33 = tl.where(o_i < i, b_a33, 0.)
            b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

        b_Ai10 = -tl.dot(
            tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai00,
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
        b_Ai21 = -tl.dot(
            tl.dot(b_Ai22, b_A21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai11,
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )
        b_Ai32 = -tl.dot(
            tl.dot(b_Ai33, b_A32, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai22,
            input_precision=SOLVE_TRIL_DOT_PRECISION
        )

        b_Ai20 = -tl.dot(
            b_Ai22,
            tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai31 = -tl.dot(
            b_Ai33,
            tl.dot(b_A31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_A32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai30 = -tl.dot(
            b_Ai33,
            tl.dot(b_A30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_A31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
            tl.dot(b_A32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )

        p_A00 = A_h + (i_tc0 + o_i)[:, None] * (HV*BT) + o_i[None, :]
        p_A10 = A_h + (i_tc1 + o_i)[:, None] * (HV*BT) + o_i[None, :]
        p_A11 = A_h + (i_tc1 + o_i)[:, None] * (HV*BT) + (BC + o_i)[None, :]
        p_A20 = A_h + (i_tc2 + o_i)[:, None] * (HV*BT) + o_i[None, :]
        p_A21 = A_h + (i_tc2 + o_i)[:, None] * (HV*BT) + (BC + o_i)[None, :]
        p_A22 = A_h + (i_tc2 + o_i)[:, None] * (HV*BT) + (2*BC + o_i)[None, :]
        p_A30 = A_h + (i_tc3 + o_i)[:, None] * (HV*BT) + o_i[None, :]
        p_A31 = A_h + (i_tc3 + o_i)[:, None] * (HV*BT) + (BC + o_i)[None, :]
        p_A32 = A_h + (i_tc3 + o_i)[:, None] * (HV*BT) + (2*BC + o_i)[None, :]
        p_A33 = A_h + (i_tc3 + o_i)[:, None] * (HV*BT) + (3*BC + o_i)[None, :]

        m_A0 = m_tc0[:, None] & (o_i[None, :] < BT)
        m_A1 = m_tc1[:, None] & (o_i[None, :] < BT)
        m_A2 = m_tc2[:, None] & (o_i[None, :] < BT)
        m_A3 = m_tc3[:, None] & (o_i[None, :] < BT)
        m_A11 = m_tc1[:, None] & ((BC + o_i)[None, :] < BT)
        m_A21 = m_tc2[:, None] & ((BC + o_i)[None, :] < BT)
        m_A22 = m_tc2[:, None] & ((2*BC + o_i)[None, :] < BT)
        m_A31 = m_tc3[:, None] & ((BC + o_i)[None, :] < BT)
        m_A32 = m_tc3[:, None] & ((2*BC + o_i)[None, :] < BT)
        m_A33 = m_tc3[:, None] & ((3*BC + o_i)[None, :] < BT)

        tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), mask=m_A0)
        tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), mask=m_A1)
        tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), mask=m_A11)
        tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), mask=m_A2)
        tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), mask=m_A21)
        tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), mask=m_A22)
        tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), mask=m_A3)
        tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), mask=m_A31)
        tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), mask=m_A32)
        tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), mask=m_A33)


def kkt_shared(k, g, beta, chunk_size=64, abl=0):
    """Same result as gdn_fused.gdn_kkt_solve (bitwise): A [B, T, HV, 64] = (I + tril(beta k k^T exp2(dg)))^-1, bf16."""
    B, T, H, K = k.shape
    HV = beta.shape[2]
    assert chunk_size == 64 and K % 32 == 0 and HV % H == 0
    A = torch.zeros(B, T, HV, chunk_size, device=k.device, dtype=k.dtype)
    kkt_solve_shared_kernel[(triton.cdiv(T, chunk_size), B * H)](k=k, g=g, beta=beta, A=A, T=T, H=H, HV=HV, K=K, BT=chunk_size, BC=16, ABL=abl)
    return A
