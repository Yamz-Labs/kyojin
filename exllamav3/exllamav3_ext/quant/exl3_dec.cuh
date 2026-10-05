#pragma once

#include <ATen/Tensor.h>

// Batch-1 EXL3 decode kernels for RDNA3.5 (gfx1151): one lane owns one 16x16 trellis tile, the
// bit windows sit at compile-time positions (fully unrolled decode, no LDS staging, no shuffles),
// fp32 accumulation through v_dot2_f32_f16. The input Hadamard (suh, H128) runs in
// the block prologue, the output Hadamard (H128, svh) in the epilogue of the last block to finish
// a 512-column strip, so one launch covers what the generic path does in three.
//
// K (bits per weight) in {2, 2.5, 3, 4, 5, 6}. `mcg` selects the codebook for all matrices of a
// launch (uniform per layer): false = mul1, true = mcg (GLM-5.3-Flash packs, see REPORT-22).
// The two templates decode bit-identically to their codebook's reference decode; only the
// summation order differs from the reconstruct+hgemm path.

#if defined(USE_ROCM)

// y[N] = x[K] @ W, x fp16 [1, K], out fp16 or fp32 [1, N]. `scratch` is a float32 workspace and
// `counters` an int32 workspace (zeroed once, left zeroed by every call), see exl3_dec_workspace.
void exl3_dec_gemv
(
    const at::Tensor& x,
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    at::Tensor& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    double K,
    bool mcg = false
);

// Same, input read in 128-wide groups at a stride: group g (inputs 128g ..) at x + g * x_gstride
// (e.g. o_proj over the first v_head_dim = 128 lanes of each 192-wide head, no slice copy).
void exl3_dec_gemv_strided
(
    const at::Tensor& x,
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    at::Tensor& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    double K,
    int64_t in_features,
    int64_t x_gstride,
    bool mcg = false
);

// Up to 4 GEMVs sharing one input x in one launch (e.g. q/k/v projections). All matrices must
// share K and the codebook.
void exl3_dec_gemv_multi
(
    const at::Tensor& x,
    const std::vector<at::Tensor>& trellis,
    const std::vector<at::Tensor>& suh,
    const std::vector<at::Tensor>& svh,
    const std::vector<at::Tensor>& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    const std::vector<double>& K,
    bool mcg = false
);

// R-row dense GEMV (REPORT-17): y[R, N] = x[R, K] @ W, one launch, each weight tile decoded once
// and applied to every row -- bit-exact vs R independent exl3_dec_gemv calls PROVIDED the caller
// doesn't touch EXL3_DEC_KTW/EXL3_DEC_MIN_BLOCKS/EXL3_DEC_KBS differently between the two (same
// pick_ktw/kbs_of call as batch-1). x must be contiguous [R, K]; no strided (x_gstride) form yet.
// mcg: codebook as in exl3_dec_gemv (glm-mtp step 8); row j stays bit-equal to exl3_dec_gemv(.., mcg).
// scratch/counters sized like exl3_dec_gemv_multi's but R times over (R * kbs * sum(N_i) floats,
// R * sum(strips_i) ints) -- pass a dedicated workspace, not the shared dec_workspace().
void exl3_dec_gemv_r
(
    const at::Tensor& x,
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    at::Tensor& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    double K,
    bool mcg = false
);

void exl3_dec_gemv_r_multi
(
    const at::Tensor& x,
    const std::vector<at::Tensor>& trellis,
    const std::vector<at::Tensor>& suh,
    const std::vector<at::Tensor>& svh,
    const std::vector<at::Tensor>& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    const std::vector<double>& K,
    bool mcg = false
);

// Routed MoE for 1 token: out[H] (fp32) = sum_s w[s] * down_e(silu(gate_e(x)) * up_e(x)), e = sel[s].
// Pointer tables (int64 device, one entry per expert) as built by MultiLinear. Two launches.
// act: fp16 [topk, I] workspace. accumulate: out += sum (out holds the fp32 residual).
// act_limit > 0 clamps silu(gate) to +act_limit and up to +/-act_limit (GLM swiglu_limit).
// mcg: the three projections use the mcg codebook (uniform per layer) instead of mul1.
void exl3_dec_moe
(
    const at::Tensor& x,
    at::Tensor& out,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    at::Tensor& act,
    at::Tensor& scratch,
    at::Tensor& counters,
    int64_t intermediate,
    double K_gu,
    double K_down,
    bool accumulate,
    double act_limit,
    bool mcg
);

// exl3_dec_moe with the shared expert folded in as slot topk (weight 1, own K / codebook),
// decB8 (EXL3_DEC_SHARED_FOLD). act: fp16 [topk + 1, I]. Routed K 2 / 2.5, shared K 4.
void exl3_dec_moe_shared
(
    const at::Tensor& x,
    at::Tensor& out,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    const at::Tensor& sh_gate_trellis,
    const at::Tensor& sh_gate_suh,
    const at::Tensor& sh_gate_svh,
    const at::Tensor& sh_up_trellis,
    const at::Tensor& sh_up_suh,
    const at::Tensor& sh_up_svh,
    const at::Tensor& sh_down_trellis,
    const at::Tensor& sh_down_suh,
    const at::Tensor& sh_down_svh,
    at::Tensor& act,
    at::Tensor& scratch,
    at::Tensor& counters,
    int64_t intermediate,
    double K_gu,
    double K_down,
    double K_sh_gu,
    double K_sh_down,
    bool accumulate,
    double act_limit,
    bool mcg
);

// R-row union MoE (REPORT-16), R<=8: x [R,H] fp16, out [R,H] fp32, selected/weights [R,topk].
// Each unique expert across the R rows' selections is decoded once and applied to every row that
// picked it, in the exact batch-1 exl3_dec_moe per-lane arithmetic order (row r bit-identical to
// a fresh batch-1 call). act: fp16 [R*topk, I], down_part: fp32 [R*topk, H] workspaces.
void exl3_dec_moe_union
(
    const at::Tensor& x,
    at::Tensor& out,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    at::Tensor& act,
    at::Tensor& down_part,
    at::Tensor& scratch,
    at::Tensor& counters,
    int64_t intermediate,
    double K_gu,
    double K_down,
    bool accumulate,
    double act_limit,
    bool mcg,
    bool device_table
);

// Sigmoid top-k router for 1 token (routing_dots semantics, fp16 logits), one launch.
void exl3_dec_router
(
    const at::Tensor& x,
    const at::Tensor& gate,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& selected,
    at::Tensor& weights,
    at::Tensor& scratch,
    at::Tensor& counters,
    double scale
);

// NR = 2..4 verify rows of exl3_dec_router in one launch (bitwise per row).
void exl3_dec_router_rows
(
    const at::Tensor& x,
    const at::Tensor& gate,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& selected,
    at::Tensor& weights,
    at::Tensor& scratch,
    at::Tensor& counters,
    double scale
);

// Fused MLP pre-norm + router: y = norm(r + xa) (fp16), routing on y, and r += xa, one launch
// (rms_norm_res_in + exl3_dec_router semantics).
void exl3_dec_router_norm
(
    const at::Tensor& xa,
    at::Tensor& r,
    const c10::optional<at::Tensor>& w,
    double eps,
    double constant_bias,
    double constant_scale,
    at::Tensor& y,
    const at::Tensor& gate,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& selected,
    at::Tensor& weights,
    at::Tensor& scratch,
    at::Tensor& counters,
    double scale
);

// RMSNorm for decode rows (ext_fallbacks.rms_norm / rms_norm_res_in semantics, w_groups == 1).
// mode 0: y = norm(x); 1: y += norm(x); 2: r += x, y = norm(r).
void exl3_dec_rms_norm
(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& w,
    at::Tensor& y,
    const c10::optional<at::Tensor>& r,
    double eps,
    double constant_bias,
    double constant_scale,
    int64_t mode
);

#endif
