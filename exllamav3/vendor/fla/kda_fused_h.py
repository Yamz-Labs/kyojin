# KDA chunk recurrence with the WY products fused in (exllamav3, not from fla).
#
# fla's chunk_kda runs recompute_w_u (w = Akk (k beta exp2 g), u = Akk (v beta), kg = k exp2(g_last - g))
# and stores w, u, kg to DRAM, then chunk_gated_delta_rule_fwd_h reads them back. This kernel computes
# w, u and kg per chunk inside the sequential state loop, so they never leave registers. With FUSE_O it also
# emits the output (fla's chunk_gla_fwd_o_gk: o = scale (q exp2 g) h + tril(Aqk) v_new) in the same loop,
# so neither h per chunk nor v_new is stored. Casts mirror the fla kernels (w, u, kg, v_new, h rounded to the
# input dtype where fla stores them), so results match the three-kernel path to rounding of the dot order.
# Supports K <= 128 (key head dim), any V; g (cumsum, log2 space) in fp32.

import torch
import triton
import triton.language as tl
from .op import exp2
from .utils import autotune_cache_kwargs


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
})
@triton.autotune(
    # gfx1151 sweep (tools/prefill/kda_sweep.py, T=255..4096): BV=128 w8 s2 wins every shape by 20-50 %;
    # one program per head computes w once instead of V/BV times.
    configs=[triton.Config({'BV': 128}, num_warps=8, num_stages=2)],
    key=['H', 'HV', 'K', 'V', 'BT', 'FUSE_O'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_h_fused(
    q, k, v, gk, beta, Akk, Aqk,
    h, v_new, o, h0, ht,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    FUSE_O: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = (i_n * T).to(tl.int64)
    NT = tl.cdiv(T, BT)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    gk += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    Akk += (bos * HV + i_hv) * BT
    Aqk += (bos * HV + i_hv) * BT
    if FUSE_O:
        o += (bos * HV + i_hv) * V
    else:
        v_new += (bos * HV + i_hv) * V
        h += ((i_n * NT) * HV + i_hv).to(tl.int64) * K * V

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        h0 += i_nh * K * V
        b_h1 += tl.load(h0 + o_k1[:, None] * V + o_v[None, :], mask=m_k1[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            b_h2 += tl.load(h0 + o_k2[:, None] * V + o_v[None, :], mask=m_k2[:, None] & m_v[None, :], other=0.0).to(tl.float32)

    for i_t in range(NT):
        o_t = i_t * BT + o_i
        m_t = o_t < T
        last_idx = min((i_t + 1) * BT, T) - 1
        if not FUSE_O:
            p_h = h + i_t.to(tl.int64) * HV * K * V
            tl.store(p_h + o_k1[:, None] * V + o_v[None, :], b_h1.to(h.dtype.element_ty), mask=m_k1[:, None] & m_v[None, :])
            if K > 64:
                tl.store(p_h + o_k2[:, None] * V + o_v[None, :], b_h2.to(h.dtype.element_ty), mask=m_k2[:, None] & m_v[None, :])

        b_b = tl.load(beta + o_t * HV, mask=m_t, other=0.0)
        b_A = tl.load(Akk + o_t[:, None] * (HV * BT) + o_i[None, :], mask=m_t[:, None], other=0.0)

        # key block 1: w1 = Akk (k beta exp2 g), kg1 = k exp2(g_last - g), qg1 = q exp2 g
        m_tk1 = m_t[:, None] & m_k1[None, :]
        b_k1 = tl.load(k + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_tk1, other=0.0)
        b_g1 = tl.load(gk + o_t[:, None] * (HV * K) + o_k1[None, :], mask=m_tk1, other=0.0).to(tl.float32)
        b_kb1 = b_k1 * b_b[:, None]
        b_kb1 *= exp2(b_g1)
        b_w1 = tl.dot(b_A, b_kb1.to(b_k1.dtype)).to(b_k1.dtype)
        b_vn = tl.dot(b_w1, b_h1.to(b_k1.dtype))
        b_gn1 = tl.load(gk + last_idx * (HV * K) + o_k1, mask=m_k1, other=0.).to(tl.float32)
        b_kg1 = (b_k1 * tl.where(m_t[:, None], exp2(b_gn1[None, :] - b_g1), 0)).to(b_k1.dtype)
        if FUSE_O:
            b_q1 = tl.load(q + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_tk1, other=0.0)
            b_qg1 = (b_q1 * exp2(b_g1)).to(b_q1.dtype)
            b_o = tl.dot(b_qg1, b_h1.to(b_k1.dtype).to(b_q1.dtype))
        if K > 64:
            m_tk2 = m_t[:, None] & m_k2[None, :]
            b_k2 = tl.load(k + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_tk2, other=0.0)
            b_g2 = tl.load(gk + o_t[:, None] * (HV * K) + o_k2[None, :], mask=m_tk2, other=0.0).to(tl.float32)
            b_kb2 = b_k2 * b_b[:, None]
            b_kb2 *= exp2(b_g2)
            b_w2 = tl.dot(b_A, b_kb2.to(b_k2.dtype)).to(b_k2.dtype)
            b_vn += tl.dot(b_w2, b_h2.to(b_k2.dtype))
            b_gn2 = tl.load(gk + last_idx * (HV * K) + o_k2, mask=m_k2, other=0.).to(tl.float32)
            b_kg2 = (b_k2 * tl.where(m_t[:, None], exp2(b_gn2[None, :] - b_g2), 0)).to(b_k2.dtype)
            if FUSE_O:
                b_q2 = tl.load(q + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_tk2, other=0.0)
                b_qg2 = (b_q2 * exp2(b_g2)).to(b_q2.dtype)
                b_o += tl.dot(b_qg2, b_h2.to(b_k2.dtype).to(b_q2.dtype))

        # u = Akk (v beta); v_new = u - w h
        m_tv = m_t[:, None] & m_v[None, :]
        b_vv = tl.load(v + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_tv, other=0.0)
        b_u = tl.dot(b_A, (b_vv * b_b[:, None]).to(b_vv.dtype)).to(b_vv.dtype)
        b_vn = b_u - b_vn
        b_vs = b_vn.to(b_vv.dtype)
        if FUSE_O:
            b_o *= scale
            b_Aq = tl.load(Aqk + o_t[:, None] * (HV * BT) + o_i[None, :], mask=m_t[:, None], other=0.0)
            b_Aq = tl.where(m_s, b_Aq, 0.).to(b_vv.dtype)
            b_o += tl.dot(b_Aq, b_vs)
            tl.store(o + o_t[:, None] * (HV * V) + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_tv)
        else:
            tl.store(v_new + o_t[:, None] * (HV * V) + o_v[None, :], b_vs.to(v_new.dtype.element_ty), mask=m_tv)

        # state update: h = h exp2(g_last) + kg^T v_new
        b_vk = b_vn.to(b_k1.dtype)
        b_h1 *= exp2(b_gn1)[:, None]
        b_h1 += tl.dot(tl.trans(b_kg1), b_vk)
        if K > 64:
            b_h2 *= exp2(b_gn2)[:, None]
            b_h2 += tl.dot(tl.trans(b_kg2), b_vk)

    if STORE_FINAL_STATE:
        ht += i_nh * K * V
        tl.store(ht + o_k1[:, None] * V + o_v[None, :], b_h1.to(ht.dtype.element_ty), mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            tl.store(ht + o_k2[:, None] * V + o_v[None, :], b_h2.to(ht.dtype.element_ty), mask=m_k2[:, None] & m_v[None, :])


def chunk_kda_fwd_h_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Akk: torch.Tensor,
    Aqk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    fuse_o: bool = False,
):
    """
    Returns (h [B, NT, HV, K, V] or None, v_new [B, T, HV, V] or None, o [B, T, HV, V] or None, final_state).
    fuse_o=False: h and v_new for chunk_gla_fwd_o_gk. fuse_o=True: o directly (h, v_new not stored).
    """
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    assert K <= 128, "fused KDA h kernel supports K <= 128"
    final_state = k.new_zeros(B, HV, K, V, dtype=torch.float32) if output_final_state else None
    if fuse_o:
        h = v_new = None
        o = torch.empty_like(v)
    else:
        h = k.new_empty(B, NT, HV, K, V)
        v_new = torch.empty_like(v)
        o = None

    def grid(meta): return (triton.cdiv(V, meta['BV']), B * HV)
    chunk_kda_fwd_kernel_h_fused[grid](
        q=q, k=k, v=v, gk=gk, beta=beta, Akk=Akk, Aqk=Aqk,
        h=h, v_new=v_new, o=o, h0=initial_state, ht=final_state,
        scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT, FUSE_O=fuse_o,
    )
    return h, v_new, o, final_state
