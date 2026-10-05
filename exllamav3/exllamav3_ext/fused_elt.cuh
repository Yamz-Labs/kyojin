#pragma once
#include <ATen/Tensor.h>

// ROCm-only fused elementwise ops (fused_elt_rocm.cu). Each replaces a torch kernel chain
// from ext_fallbacks.py with ONE launch, same arithmetic per element.

// y = rmsnorm(x [* act(g)]) * (w[row % w_groups] + constant_bias) [* act(g)]
// x bf16, g bf16/f32, w bf16/f32, y f16/f32; gate_act 0 = SiLU, 1 = sigmoid
void fused_gated_rms_norm
(
    const at::Tensor& x,
    const at::Tensor& w,
    at::Tensor& y,
    const at::Tensor& g,
    double eps,
    double constant_bias,
    int64_t w_groups,
    bool gate_first,
    int64_t gate_act,
    int64_t y_pitch,   // 0 = y contiguous; else y is [tokens, y_pitch] storage
    int64_t hpt        // heads per token row when y_pitch != 0
);

// out[row, 0:k] = x[row, 0:k] for 2-byte elements, out at row pitch `pitch` (consumer-side row padding of a GEMM input)
void pad_copy
(
    const at::Tensor& x,
    at::Tensor& out,
    int64_t pitch
);

// out = x * sigmoid(g) (fp16; the ext_fallbacks.mul_sigmoid_ arithmetic: sigmoid rounded to half, then the half product)
// written to a row-padded view: x, g contiguous [..., k], out columns contiguous at row pitch `pitch`
void mul_sigmoid_pad
(
    const at::Tensor& x,
    const at::Tensor& g,
    at::Tensor& out,
    int64_t pitch
);

// z = silu(x) * y (ext_fallbacks._act_mul contract incl. act_limit clamps); x/y f16 or f32, z f16
void fused_silu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    double act_limit
);

// KDA gates: beta = bf16(sigmoid(b * beta_scale)), g = lb * sigmoid(exp(a_log) * (f + dt_bias))
// (or -exp(a_log) * softplus20(f + dt_bias) when has_lb is false). All inputs fp32.
void kda_gate
(
    const at::Tensor& b,        // (T, nv)
    const at::Tensor& f,        // (T, nv * dk)
    const at::Tensor& dt_bias,  // (nv * dk)
    const at::Tensor& a_log,    // (nv)
    double beta_scale,
    double lower_bound,
    bool has_lb,
    at::Tensor& beta,           // (T, nv) bf16
    at::Tensor& g               // (T, nv * dk) fp32
);

// kda-dec step 2 (decode, rows <= 8): f = fa @ wfb (K = 128, hipBLAS order) + kda_gate math, one launch
void kda_fb_gate
(
    const at::Tensor& fa,       // (rows, 128) fp16
    const at::Tensor& wfb,      // (128, nv * dk) fp16
    const at::Tensor& b,        // (rows, nv) fp32
    const at::Tensor& dt_bias,  // (nv * dk) fp32
    const at::Tensor& a_log,    // (nv) fp32
    double beta_scale,
    double lower_bound,
    bool has_lb,
    at::Tensor& beta,           // (rows, nv) bf16
    at::Tensor& g               // (rows, nv * dk) fp32
);

// kda-dec step 2: z = ga @ wgb per head, then fused_gated_rms_norm (fp32 gate) on x (rows * nv, 128)
void kda_gb_norm
(
    const at::Tensor& x,
    const at::Tensor& ga,
    const at::Tensor& wgb,
    const at::Tensor& w,
    at::Tensor& y,
    double eps,
    double constant_bias,
    int64_t w_groups,
    bool gate_first,
    int64_t gate_act
);
