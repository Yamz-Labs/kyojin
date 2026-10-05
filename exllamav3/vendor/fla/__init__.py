"""
Forward-only chunked linear-attention kernels vendored from flash-linear-attention
(https://github.com/fla-org/flash-linear-attention), v0.5.2, MIT license: see LICENSE in this
directory. Only the Triton forward paths exllamav3 uses for prefill are included (the gated delta
rule, KDA and simple GLA chunk recurrences), without autograd, backward kernels, context
parallelism, in-kernel gate activation or the fla backend dispatch. The kernel sources are
verbatim apart from import paths; the three entry points below replace fla's autograd wrappers.

Inputs are `[B, T, H, K]` / `[B, T, HV, V]`, equal-length sequences only (no cu_seqlens).
"""
from __future__ import annotations
import os
import sys
import torch

from .utils import input_guard
from .l2norm import l2norm_fwd
from . import gdn_pf as _gdn_pf
from .cumsum import chunk_local_cumsum
from .gdn_chunk_fwd import chunk_gated_delta_rule_fwd_intra
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_h import chunk_fwd_h
from .kda_chunk_intra import chunk_kda_fwd_intra
from .kda_intra_fused import chunk_kda_fwd_intra_fused, chunk_kda_fwd_intra_pinned
from .gla_chunk import chunk_gla_fwd_o_gk
from .kda_fused_h import chunk_kda_fwd_h_fused

_HIP_FAIL = []  # first failure disables the HIP path for the process


def _hip_ok(q, k, v, beta, Akk, Aqk, h0):
    if _HIP_FAIL:
        return False
    bf = torch.bfloat16
    ts = (q, k, v, beta, Akk, Aqk)
    return (all(t.dtype == bf and t.is_contiguous() for t in ts)
            and (h0 is None or h0.is_contiguous()))

# 1 / ln(2) as fla defines it (best fp32 approximation); gates are pre-scaled so the kernels use exp2
RCP_LN2 = 1.4426950216


@input_guard
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunked gated delta rule (Gated DeltaNet prefill).

    q, k: [B, T, H, K]; v: [B, T, HV, V] (HV a multiple of H); g: [B, T, HV] log-space decay;
    beta: [B, T, HV] post-sigmoid; initial_state: [B, HV, K, V] fp32 or None.
    Returns (o [B, T, HV, V] in q's dtype, final_state [B, HV, K, V] fp32 or None).
    """
    H, HV = q.shape[2], v.shape[2]
    assert q.shape[2] == k.shape[2], "q and k must have the same number of heads"
    assert HV % H == 0, f"num_v_heads ({HV}) must be a multiple of num_heads ({H})"
    assert chunk_size in (16, 32, 64), f"chunk_size must be 16, 32 or 64, got {chunk_size}"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    g = chunk_local_cumsum(g, chunk_size = chunk_size, scale = RCP_LN2)
    if os.environ.get("EXL3_GDN_FUSE", "0") == "1":   # hand-written HIP chunk kernel (gdn_fused.py), bitwise equal, default off
        from .gdn_fused import chunk_gdn_fused
        res = chunk_gdn_fused(q, k, v, g, beta, scale, initial_state, output_final_state, chunk_size)
        if res is not None:
            return res[0].to(q.dtype), res[1]
    # WY representation: fused kkt + solve_tril + recompute_w_u. u is the new v
    w, u, _ = chunk_gated_delta_rule_fwd_intra(k = k, v = v, g = g, beta = beta, chunk_size = chunk_size)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k = k, w = w, u = u, g = g,
        initial_state = initial_state,
        output_final_state = output_final_state,
        chunk_size = chunk_size,
    )
    o = chunk_fwd_o(q = q, k = k, v = v_new, h = h, g = g, scale = scale, chunk_size = chunk_size)
    return o.to(q.dtype), final_state


def chunk_gated_delta_rule_pf(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    """chunk_gated_delta_rule(use_qk_l2norm_in_kernel=True, chunk_size=64) for q, k, v given as (possibly strided) views of the conv
    output; g (fp32) and beta (bf16) contiguous. Bitwise equal to the guarded entry; no copies of q, k, v when the views are plain
    token-pitch slices."""
    if not (_gdn_pf.strided_ok(q) and _gdn_pf.strided_ok(k) and q.shape[-1] == 128 and q.dtype == torch.bfloat16):
        return chunk_gated_delta_rule(q, k, v, g=g, beta=beta, scale=scale, initial_state=initial_state,
                                      output_final_state=output_final_state, use_qk_l2norm_in_kernel=True)
    g = g.contiguous(); beta = beta.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    chunk_size = 64
    if scale is None:
        scale = k.shape[-1] ** -0.5
    with torch.cuda.device(q.device.index):
        qn = _gdn_pf.l2norm_strided(q)
        kn = _gdn_pf.l2norm_strided(k)
        g = chunk_local_cumsum(g, chunk_size = chunk_size, scale = RCP_LN2)
        if os.environ.get("EXL3_GDN_FUSE", "0") == "1":
            from .gdn_fused import chunk_gdn_fused
            res = chunk_gdn_fused(qn, kn, v, g, beta, scale, initial_state, output_final_state, chunk_size)
            if res is not None:
                return res[0].to(qn.dtype), res[1]
        v = v.contiguous()
        w, u, _ = chunk_gated_delta_rule_fwd_intra(k = kn, v = v, g = g, beta = beta, chunk_size = chunk_size)
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k = kn, w = w, u = u, g = g, initial_state = initial_state, output_final_state = output_final_state, chunk_size = chunk_size)
        o = chunk_fwd_o(q = qn, k = kn, v = v_new, h = h, g = g, scale = scale, chunk_size = chunk_size)
        return o.to(qn.dtype), final_state


@input_guard
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    g_cumsum: bool = False,
    ckpt: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunked Kimi Delta Attention (per-channel decay) prefill.

    q, k: [B, T, H, K] with K <= 256; v: [B, T, HV, V]; g: [B, T, HV, K] log-space decay;
    beta: [B, T, HV] post-sigmoid; initial_state: [B, HV, K, V] fp32 or None.
    Returns (o [B, T, HV, V] in q's dtype, final_state [B, HV, K, V] fp32 or None).
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    assert q.shape == k.shape, f"q and k must have the same shape, got {q.shape} vs {k.shape}"
    assert K <= 256, f"KDA supports key head dims up to 256, got {K}"
    assert HV % H == 0, f"num_v_heads ({HV}) must be a multiple of num_heads ({H})"
    assert g.shape == (B, T, HV, K), f"g must have shape {[B, T, HV, K]}, got {list(g.shape)}"
    assert beta.shape == (B, T, HV), f"beta must have shape {[B, T, HV]}, got {list(beta.shape)}"
    assert chunk_size in (32, 64), f"chunk_size must be 32 or 64, got {chunk_size}"
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be float32"
    if scale is None:
        scale = K ** -0.5
    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    # g_cumsum: the caller already applied chunk_local_cumsum(g, chunk_size, RCP_LN2) (exllamav3 kda_gate)
    if not g_cumsum:
        g = chunk_local_cumsum(g, chunk_size = chunk_size, scale = RCP_LN2)
    # EXL3_KDA_FUSE (read per call): 0 = fla chain (recompute_w_u -> fwd_h -> gla_o), 1 = w/u/kg fused
    # into the state loop, 2 = output fused too (no w, u, kg, h, v_new in DRAM), 3 = the 2 path as a hand-written
    # HIP kernel (vendor/fla/hip, bitwise equal to 2, ~2.3x faster; falls back to 2 on unsupported shapes/dtypes
    # or if hipcc/module load fails; default 3, step 12). Chunks under 512 tokens
    # keep the fla chain: the one-program-per-head fused loop is ~0.14 ms slower there (T=255 sweep).
    fuse = int(os.environ.get("EXL3_KDA_FUSE", "3")) if K <= 128 and chunk_size == 64 and T >= 512 else 0
    if fuse:
        # EXL3_KDA_INTRA (read per call): 2 = fla's two intra kernels with pinned gfx1151 configs (bitwise
        # equal to 0), 1 = one-kernel intra (diagonal blocks computed in the inter/solve program, no
        # token_parallel pass, no Akkd round trip; ties 2 on gfx1151, not bitwise), 0 = fla autotune. Default 2 (step 11)
        intra = int(os.environ.get("EXL3_KDA_INTRA", "2"))
        if intra == 2:
            Aqk, Akk = chunk_kda_fwd_intra_pinned(q = q, k = k, gk = g, beta = beta, scale = scale, chunk_size = chunk_size)
        elif intra == 1:
            Aqk, Akk = chunk_kda_fwd_intra_fused(q = q, k = k, gk = g, beta = beta, scale = scale, chunk_size = chunk_size)
        else:
            _, _, _, _, Aqk, Akk = chunk_kda_fwd_intra(
                q = q, k = k, v = v, gk = g, beta = beta, scale = scale, chunk_size = chunk_size, skip_wu = True,
            )
        o = None
        if fuse == 3 and K == 128 and v.shape[-1] == 128 and _hip_ok(q, k, v, beta, Akk, Aqk, initial_state):
            try:
                from .hip.kda_fused_h_hip import chunk_kda_fwd_h_fused_hip
                h, v_new, o, final_state = chunk_kda_fwd_h_fused_hip(
                    q = q, k = k, v = v, gk = g, beta = beta, Akk = Akk, Aqk = Aqk, scale = scale,
                    initial_state = initial_state, output_final_state = output_final_state, chunk_size = chunk_size,
                    ckpt = ckpt,  # mid-chunk state side-write (HIP path only; sets ckpt["ok"])
                )
            except Exception as e:  # hipcc missing, compile or module-load failure: fall back to Triton
                _HIP_FAIL.append(repr(e))
                print(f"chunk_kda: HIP fused_h disabled, Triton fallback: {e!r}", file = sys.stderr)
                o = None
        if o is None:
            h, v_new, o, final_state = chunk_kda_fwd_h_fused(
                q = q, k = k, v = v, gk = g, beta = beta, Akk = Akk, Aqk = Aqk, scale = scale,
                initial_state = initial_state, output_final_state = output_final_state,
                chunk_size = chunk_size, fuse_o = fuse >= 2,
            )
        if o is None:
            o = chunk_gla_fwd_o_gk(q = q, v = v_new, g = g, A = Aqk, h = h, scale = scale, chunk_size = chunk_size)
        return o.to(q.dtype), final_state
    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
        q = q, k = k, v = v, gk = g, beta = beta, scale = scale, chunk_size = chunk_size,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k = kg, w = w, u = u, gk = g,
        initial_state = initial_state,
        output_final_state = output_final_state,
        chunk_size = chunk_size,
    )
    o = chunk_gla_fwd_o_gk(q = q, v = v_new, g = g, A = Aqk, h = h, scale = scale, chunk_size = chunk_size)
    return o.to(q.dtype), final_state


@input_guard
def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunked simple GLA (head-wise scalar decay, no delta correction), used for the Mamba2 SSD
    prefill.

    q, k: [B, T, H, K]; v: [B, T, H, V]; g: [B, T, H] log-space decay or None;
    initial_state: [B, H, K, V] or None.
    Returns (o [B, T, H, V] in q's dtype, final_state [B, H, K, V] or None).
    """
    assert chunk_size & (chunk_size - 1) == 0, f"chunk_size must be a power of 2, got {chunk_size}"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if g is not None:
        g = chunk_local_cumsum(g, chunk_size = chunk_size, scale = RCP_LN2)
    h, ht = chunk_fwd_h(
        k = k, v = v, g = g,
        h0 = initial_state,
        output_final_state = output_final_state,
        chunk_size = chunk_size,
        states_in_fp32 = False,
    )
    o = chunk_fwd_o(q = q, k = k, v = v, g = g, h = h, scale = scale, chunk_size = chunk_size)
    return o.to(q.dtype), ht


__all__ = ["chunk_gated_delta_rule", "chunk_kda", "chunk_simple_gla"]
