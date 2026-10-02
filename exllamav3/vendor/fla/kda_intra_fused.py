# KDA intra-chunk A matrices in one kernel (exllamav3, not from fla).
#
# fla's chunk_kda_fwd_intra runs two kernels: token_parallel (one program per token, loops over the up to 16
# earlier tokens of its sub-chunk, reloading k_j and g_j every time) writes the fp32 diagonal 16x16 Akk blocks
# to DRAM (Akkd), then inter_solve_fused (one program per chunk and head) computes the off-diagonal blocks,
# reloads Akkd, inverts (I + Akk) and stores Aqk and Akk^-1. Here one program per (chunk, head) computes the
# diagonal blocks too, from the k, q, g rows it already touches, so Akkd, the token_parallel launch and the
# Akk memset go away (upper blocks are stored as zeros). Per-element math mirrors token_parallel
# (q_i * (k_j exp2(g_i - g_j)), (k_i beta_i) * (k_j exp2(g_i - g_j))); only the K summation order differs.
# Fixed-length batches, BT = 64, BC = 16, K <= 256; g is the chunk-local cumsum in log2 space (fp32).

import os
import torch
import triton
import triton.language as tl
from .op import exp2
from .utils import IS_TF32_SUPPORTED

if IS_TF32_SUPPORTED:
    DOT_PREC = tl.constexpr('tf32')
else:
    DOT_PREC = tl.constexpr('ieee')


@triton.jit
def _diag_block(q, k, g, beta, o_c, m_c, scale, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                BC: tl.constexpr, BKD: tl.constexpr):
    # q, k, g, beta already offset to (bos, head). Returns the [BC, BC] diagonal blocks (fp32):
    # Aqk (j <= i, scaled) and Akk (j < i, beta_i applied).
    o_i = tl.arange(0, BC)
    m_le = o_i[:, None] >= o_i[None, :]
    b_b = tl.load(beta + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
    b_Aq = tl.zeros([BC, BC], dtype=tl.float32)
    b_Ak = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BKD)):
        o_k = i_k * BKD + tl.arange(0, BKD)
        m_k = o_k < K
        m = m_c[:, None] & m_k[None, :]
        b_q = tl.load(q + o_c[:, None] * (H*K) + o_k[None, :], mask=m, other=0.0).to(tl.float32)
        b_k = tl.load(k + o_c[:, None] * (H*K) + o_k[None, :], mask=m, other=0.0).to(tl.float32)
        b_g = tl.load(g + o_c[:, None] * (HV*K) + o_k[None, :], mask=m, other=0.0).to(tl.float32)
        # [i, j, k]; j > i is masked out after the sum, keep its exponent finite
        b_d = tl.where(m_le[:, :, None], b_g[:, None, :] - b_g[None, :, :], 0.)
        b_kg = b_k[None, :, :] * exp2(b_d)
        b_Aq += tl.sum(b_q[:, None, :] * b_kg, 2)
        b_Ak += tl.sum((b_k * b_b[:, None])[:, None, :] * b_kg, 2)
    b_Aq = tl.where(m_le, b_Aq * scale, 0.)
    b_Ak = tl.where(o_i[:, None] > o_i[None, :], b_Ak, 0.)
    return b_Aq, b_Ak


@triton.jit
def _fwd_subst(b_A, n, BC: tl.constexpr):
    # b_A = strictly lower Akk block (fp32). Returns (I + A)^-1 by forward substitution; row extraction
    # from registers gives the same values fla reloads from Akkd.
    o_i = tl.arange(0, BC)
    b_Ai = -b_A
    for i in range(2, n):
        b_a = tl.sum(tl.where((o_i == i)[:, None], b_Ai, 0.), 0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    return b_Ai + (o_i[:, None] == o_i[None, :])


@triton.jit
def _inter_pair(b_x, b_k, b_gx, b_gn, b_ky, b_gy, b_mx, F16: tl.constexpr):
    # rows x (later sub-chunk) against rows y (earlier): x * exp2(g_x - g_n) . (k_y exp2(g_n - g_y))^T.
    # Both factors are bounded by |x|, |k| (g is non-increasing), so F16 (fp16 WMMA, fp32 accumulate) is
    # safe in range; its 10-bit mantissa matches the tf32 fla uses on NVIDIA.
    b_gq = tl.where(b_mx[:, None], exp2(b_gx - b_gn[None, :]), 0)
    b_kgt = tl.trans(b_ky * exp2(b_gn[None, :] - b_gy))
    b_xg = b_x * b_gq
    b_kg = b_k * b_gq
    if F16:
        b_a = tl.dot(b_xg.to(tl.float16), b_kgt.to(tl.float16))
        b_b = tl.dot(b_kg.to(tl.float16), b_kgt.to(tl.float16))
    else:
        b_a = tl.dot(b_xg, b_kgt)
        b_b = tl.dot(b_kg, b_kgt)
    return b_a, b_b


@triton.jit
def _sdot(a, b, S16: tl.constexpr):
    # solve-phase 16x16 products: fp32 ieee (bitwise-safe default) or fp16 inputs with fp32 accumulate
    if S16:
        return tl.dot(a.to(tl.float16), b.to(tl.float16))
    return tl.dot(a, b, input_precision=DOT_PREC)


@triton.autotune(
    # gfx1151 sweep (step 11, tools/prefill/kda_intra_r2.py): BK 16 BKD 32 w8 best; BK >= 32 spills
    configs=[triton.Config({'BK': 16, 'BKD': 32}, num_warps=8, num_stages=1)],
    key=['H', 'HV', 'K', 'BT', 'BC'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_intra_fused(
    q, k, g, beta, Aqk, Akk, scale, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    BK: tl.constexpr, BKD: tl.constexpr, F16: tl.constexpr = False, S16: tl.constexpr = False,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    bos = i_b * T
    if i_t * BT >= T:
        return
    i_tc0 = i_t * BT
    i_tc1 = i_tc0 + BC
    i_tc2 = i_tc0 + 2 * BC
    i_tc3 = i_tc0 + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT

    o_i = tl.arange(0, BC)
    o_c0 = i_tc0 + o_i
    o_c1 = i_tc1 + o_i
    o_c2 = i_tc2 + o_i
    o_c3 = i_tc3 + o_i
    m_tc0 = o_c0 < T
    m_tc1 = o_c1 < T
    m_tc2 = o_c2 < T
    m_tc3 = o_c3 < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    # off-diagonal blocks (same math and order as fla's inter_solve_fused)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_k0 = tl.load(k + o_c0[:, None] * (H*K) + o_k[None, :], mask=m_tc0[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_g0 = tl.load(g + o_c0[:, None] * (HV*K) + o_k[None, :], mask=m_tc0[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        if i_tc1 < T:
            m1 = m_tc1[:, None] & m_k[None, :]
            b_q1 = tl.load(q + o_c1[:, None] * (H*K) + o_k[None, :], mask=m1, other=0.0).to(tl.float32)
            b_k1 = tl.load(k + o_c1[:, None] * (H*K) + o_k[None, :], mask=m1, other=0.0).to(tl.float32)
            b_g1 = tl.load(g + o_c1[:, None] * (HV*K) + o_k[None, :], mask=m1, other=0.0).to(tl.float32)
            b_gn1 = tl.load(g + i_tc1 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
            a, b = _inter_pair(b_q1, b_k1, b_g1, b_gn1, b_k0, b_g0, m_tc1, F16)
            b_Aqk10 += a
            b_Akk10 += b
            if i_tc2 < T:
                m2 = m_tc2[:, None] & m_k[None, :]
                b_q2 = tl.load(q + o_c2[:, None] * (H*K) + o_k[None, :], mask=m2, other=0.0).to(tl.float32)
                b_k2 = tl.load(k + o_c2[:, None] * (H*K) + o_k[None, :], mask=m2, other=0.0).to(tl.float32)
                b_g2 = tl.load(g + o_c2[:, None] * (HV*K) + o_k[None, :], mask=m2, other=0.0).to(tl.float32)
                b_gn2 = tl.load(g + i_tc2 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
                a, b = _inter_pair(b_q2, b_k2, b_g2, b_gn2, b_k0, b_g0, m_tc2, F16)
                b_Aqk20 += a
                b_Akk20 += b
                a, b = _inter_pair(b_q2, b_k2, b_g2, b_gn2, b_k1, b_g1, m_tc2, F16)
                b_Aqk21 += a
                b_Akk21 += b
                if i_tc3 < T:
                    m3 = m_tc3[:, None] & m_k[None, :]
                    b_q3 = tl.load(q + o_c3[:, None] * (H*K) + o_k[None, :], mask=m3, other=0.0).to(tl.float32)
                    b_k3 = tl.load(k + o_c3[:, None] * (H*K) + o_k[None, :], mask=m3, other=0.0).to(tl.float32)
                    b_g3 = tl.load(g + o_c3[:, None] * (HV*K) + o_k[None, :], mask=m3, other=0.0).to(tl.float32)
                    b_gn3 = tl.load(g + i_tc3 * HV*K + o_k, mask=m_k, other=0).to(tl.float32)
                    a, b = _inter_pair(b_q3, b_k3, b_g3, b_gn3, b_k0, b_g0, m_tc3, F16)
                    b_Aqk30 += a
                    b_Akk30 += b
                    a, b = _inter_pair(b_q3, b_k3, b_g3, b_gn3, b_k1, b_g1, m_tc3, F16)
                    b_Aqk31 += a
                    b_Akk31 += b
                    a, b = _inter_pair(b_q3, b_k3, b_g3, b_gn3, b_k2, b_g2, m_tc3, F16)
                    b_Aqk32 += a
                    b_Akk32 += b

    b_b1 = tl.load(beta + o_c1 * HV, mask=m_tc1, other=0.0).to(tl.float32)
    b_b2 = tl.load(beta + o_c2 * HV, mask=m_tc2, other=0.0).to(tl.float32)
    b_b3 = tl.load(beta + o_c3 * HV, mask=m_tc3, other=0.0).to(tl.float32)
    b_Akk10 = b_Akk10 * b_b1[:, None]
    b_Akk20 = b_Akk20 * b_b2[:, None]
    b_Akk21 = b_Akk21 * b_b2[:, None]
    b_Akk30 = b_Akk30 * b_b3[:, None]
    b_Akk31 = b_Akk31 * b_b3[:, None]
    b_Akk32 = b_Akk32 * b_b3[:, None]

    # Aqk rows: [A_c0 .. A_cc, 0 ...]; upper blocks are masked downstream (fla leaves them unwritten)
    ty_q = Aqk.dtype.element_ty
    p0 = Aqk + o_c0[:, None] * (HV*BT) + o_i[None, :]
    p1 = Aqk + o_c1[:, None] * (HV*BT) + o_i[None, :]
    p2 = Aqk + o_c2[:, None] * (HV*BT) + o_i[None, :]
    p3 = Aqk + o_c3[:, None] * (HV*BT) + o_i[None, :]
    tl.store(p1, (b_Aqk10 * scale).to(ty_q), mask=m_tc1[:, None])
    tl.store(p2, (b_Aqk20 * scale).to(ty_q), mask=m_tc2[:, None])
    tl.store(p2 + BC, (b_Aqk21 * scale).to(ty_q), mask=m_tc2[:, None])
    tl.store(p3, (b_Aqk30 * scale).to(ty_q), mask=m_tc3[:, None])
    tl.store(p3 + BC, (b_Aqk31 * scale).to(ty_q), mask=m_tc3[:, None])
    tl.store(p3 + 2*BC, (b_Aqk32 * scale).to(ty_q), mask=m_tc3[:, None])

    # diagonal blocks (token_parallel's job in fla), computed after the inter loop so only one pair is live,
    # then forward substitution on each; the merged inverse follows in fla's order
    n = T - i_tc0
    b_Aq, b_Ak = _diag_block(q, k, g, beta, o_c0, m_tc0, scale, H, HV, K, BC, BKD)
    tl.store(p0, b_Aq.to(ty_q), mask=m_tc0[:, None])
    b_Ai00 = _fwd_subst(b_Ak, min(BC, n), BC)
    b_Aq, b_Ak = _diag_block(q, k, g, beta, o_c1, m_tc1, scale, H, HV, K, BC, BKD)
    tl.store(p1 + BC, b_Aq.to(ty_q), mask=m_tc1[:, None])
    b_Ai11 = _fwd_subst(b_Ak, min(2*BC, n) - BC, BC)
    b_Aq, b_Ak = _diag_block(q, k, g, beta, o_c2, m_tc2, scale, H, HV, K, BC, BKD)
    tl.store(p2 + 2*BC, b_Aq.to(ty_q), mask=m_tc2[:, None])
    b_Ai22 = _fwd_subst(b_Ak, min(3*BC, n) - 2*BC, BC)
    b_Aq, b_Ak = _diag_block(q, k, g, beta, o_c3, m_tc3, scale, H, HV, K, BC, BKD)
    tl.store(p3 + 3*BC, b_Aq.to(ty_q), mask=m_tc3[:, None])
    b_Ai33 = _fwd_subst(b_Ak, min(4*BC, n) - 3*BC, BC)

    b_Ai10 = -_sdot(_sdot(b_Ai11, b_Akk10, S16), b_Ai00, S16)
    b_Ai21 = -_sdot(_sdot(b_Ai22, b_Akk21, S16), b_Ai11, S16)
    b_Ai20 = -_sdot(
        b_Ai22,
        _sdot(b_Akk20, b_Ai00, S16) + _sdot(b_Akk21, b_Ai10, S16),
        S16)
    b_Ai32 = -_sdot(_sdot(b_Ai33, b_Akk32, S16), b_Ai22, S16)
    b_Ai31 = -_sdot(
        b_Ai33,
        _sdot(b_Akk31, b_Ai11, S16) + _sdot(b_Akk32, b_Ai21, S16),
        S16)
    b_Ai30 = -_sdot(
        b_Ai33,
        _sdot(b_Akk30, b_Ai00, S16) + _sdot(b_Akk31, b_Ai10, S16)
        + _sdot(b_Akk32, b_Ai20, S16),
        S16)

    ty_k = Akk.dtype.element_ty
    b_z = tl.zeros([BC, BC], dtype=tl.float32).to(ty_k)
    p0 = Akk + o_c0[:, None] * (HV*BT) + o_i[None, :]
    p1 = Akk + o_c1[:, None] * (HV*BT) + o_i[None, :]
    p2 = Akk + o_c2[:, None] * (HV*BT) + o_i[None, :]
    p3 = Akk + o_c3[:, None] * (HV*BT) + o_i[None, :]
    tl.store(p0, b_Ai00.to(ty_k), mask=m_tc0[:, None])
    tl.store(p0 + BC, b_z, mask=m_tc0[:, None])
    tl.store(p0 + 2*BC, b_z, mask=m_tc0[:, None])
    tl.store(p0 + 3*BC, b_z, mask=m_tc0[:, None])
    tl.store(p1, b_Ai10.to(ty_k), mask=m_tc1[:, None])
    tl.store(p1 + BC, b_Ai11.to(ty_k), mask=m_tc1[:, None])
    tl.store(p1 + 2*BC, b_z, mask=m_tc1[:, None])
    tl.store(p1 + 3*BC, b_z, mask=m_tc1[:, None])
    tl.store(p2, b_Ai20.to(ty_k), mask=m_tc2[:, None])
    tl.store(p2 + BC, b_Ai21.to(ty_k), mask=m_tc2[:, None])
    tl.store(p2 + 2*BC, b_Ai22.to(ty_k), mask=m_tc2[:, None])
    tl.store(p2 + 3*BC, b_z, mask=m_tc2[:, None])
    tl.store(p3, b_Ai30.to(ty_k), mask=m_tc3[:, None])
    tl.store(p3 + BC, b_Ai31.to(ty_k), mask=m_tc3[:, None])
    tl.store(p3 + 2*BC, b_Ai32.to(ty_k), mask=m_tc3[:, None])
    tl.store(p3 + 3*BC, b_Ai33.to(ty_k), mask=m_tc3[:, None])


def chunk_kda_fwd_intra_fused(q, k, gk, beta, scale, chunk_size=64):
    """Returns (Aqk, Akk) like fla's chunk_kda_fwd_intra(skip_wu=True). Fixed-length only, BT=64."""
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT, BC = chunk_size, 16
    assert BT == 64 and K <= 256
    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    grid = (triton.cdiv(T, BT), B * HV)
    chunk_kda_fwd_kernel_intra_fused[grid](q, k, gk, beta, Aqk, Akk, scale, T, H=H, HV=HV, K=K, BT=BT, BC=BC)
    return Aqk, Akk


def chunk_kda_fwd_intra_pinned(q, k, gk, beta, scale, chunk_size=64):
    """fla's two intra kernels (token_parallel + inter_solve_fused) launched with fixed gfx1151 configs,
    bypassing their autotune. Bitwise equal to chunk_kda_fwd_intra(skip_wu=True) (its autotune picks
    token_parallel BH8 w8, inter_solve BK32 w2; the K summation order of the pinned configs gives the same
    bits). Step 11 sweep on GLM captures, T=4096: 19.5 -> 12.9 ms. Returns (Aqk, Akk)."""
    from .kda_chunk_intra import chunk_kda_fwd_kernel_inter_solve_fused as _is
    from .kda_chunk_intra_token_parallel import chunk_kda_fwd_kernel_intra_token_parallel as _tp
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT, BC = chunk_size, 16
    assert BT == 64
    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)
    nw_is, ns_is = 2, 1
    # EXL3_KDA_INTRA_CFG: optional launch-only override, "warps,stages" (e.g. 8,1 or 4,1).
    # Empty/unset is the existing pinned default, preserving the bitwise baseline.
    cfg = os.environ.get("EXL3_KDA_INTRA_CFG", "")
    if cfg and cfg != "0":
        nw, ns = (int(x) for x in cfg.replace(",", " ").replace("x", " ").split())
        if nw not in (4, 8) or ns not in (1, 2):
            raise ValueError("EXL3_KDA_INTRA_CFG must be '4,1', '4,2', '8,1', or '8,2'")
    else:
        nw, ns = 16, 1
        nw_is, ns_is = 2, 1
    if cfg:
        nw_is, ns_is = nw, ns
    _tp.fn.fn[(B * T, triton.cdiv(HV, 16))](q, k, gk, beta, Aqk, Akkd, scale, None, B, T, H=H, HV=HV, K=K, BT=BT,
                                            BC=BC, BH=16, IS_VARLEN=False, num_warps=nw, num_stages=ns)
    # EXL3_KDA_IS_SPLIT (default 3 = HIP off + head-fastest merge, kdaC7): spill-free two-kernel inter-solve (kdaC6), see kda_inter_split.py
    from .kda_inter_split import inter_split_mode, inter_solve_split
    m = inter_split_mode(T, BT, BC)
    if m:
        inter_solve_split(q, k, gk, beta, Aqk, Akkd, Akk, scale, m)
        return Aqk, Akk
    _is.fn.fn[(triton.cdiv(T, BT), B * HV)](q, k, gk, beta, Aqk, Akkd, Akk, scale, None, None, T, H=H, HV=HV, K=K,
                                            BT=BT, BC=BC, NC=BT // BC, BK=16, IS_VARLEN=False,
                                            USE_SAFE_GATE=False, num_warps=nw_is, num_stages=ns_is)
    return Aqk, Akk
