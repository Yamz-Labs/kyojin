# EXL3_GDN_FUSE=1: GDN prefill chunk path with the hand-written HIP kernel (vendor/fla/hip/gdn_fused_h.hip) replacing
# recompute_w_u + chunk_h + chunk_o. kkt + solve_tril stay on the Triton kernel (it produces A). Bitwise equal to the Triton chain
# (w, u, v_new, state, o, final state), microbench Qwen shape (H 16, HV 48) on gfx1151. Default off; falls back to the Triton chain
# on unsupported shapes / dtypes and disables itself for the process if the HIP build or launch fails.
import os
import torch
import triton
from .gdn_chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel

_FAIL = []


def gdn_fuse_ok(q, k, v, g, beta, initial_state, chunk_size):
    if _FAIL or chunk_size != 64:
        return False
    bf = torch.bfloat16
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    return (K == 128 and V == 128 and HV % H == 0 and q.shape == k.shape and g.shape == (B, T, HV) and beta.shape == (B, T, HV)
            and all(t.dtype == bf and t.is_contiguous() for t in (q, k, beta)) and v.dtype == bf and v.stride(3) == 1 and v.stride(2) == V
            and v.stride(1) >= HV * V and v.stride(1) % 2 == 0 and v.storage_offset() % 2 == 0 and (B == 1 or v.stride(0) == T * v.stride(1)) and g.dtype == torch.float32 and g.is_contiguous()
            and (initial_state is None or (initial_state.dtype == torch.float32 and initial_state.is_contiguous()
                                           and initial_state.shape == (B, HV, K, V))))


def gdn_kkt_solve(k, g, beta, chunk_size=64):
    B, T, H, K = k.shape
    HV = beta.shape[2]
    # EXL3_GDN_KKT_SHARED (default 1): one program per (chunk, k-head) computes the k k^T Gram once for the HV // H value heads that share it
    # (stock: once per value head); everything per value head is the stock text, A is bitwise equal. 0 = stock kernel.
    if os.environ.get("EXL3_GDN_KKT_SHARED", "1") == "1" and chunk_size == 64 and K % 32 == 0:
        from .gdn_kkt_shared import kkt_shared
        return kkt_shared(k, g, beta, chunk_size)
    A = torch.zeros(B, T, HV, chunk_size, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(triton.cdiv(T, chunk_size), B * HV)](
        k=k, g=g, beta=beta, A=A, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=chunk_size, BC=16)
    return A


def chunk_gdn_fused(q, k, v, g, beta, scale, initial_state, output_final_state, chunk_size=64):
    """Returns (o, final_state) or None when the caller must run the Triton chain."""
    if not gdn_fuse_ok(q, k, v, g, beta, initial_state, chunk_size):
        return None
    try:
        from .hip.gdn_fused_h_hip import chunk_gdn_fwd_fused_hip
        A = gdn_kkt_solve(k, g, beta, chunk_size)
        return chunk_gdn_fwd_fused_hip(q, k, v, g, beta, A, scale, initial_state=initial_state, output_final_state=output_final_state)
    except Exception:
        _FAIL.append(1)
        return None
