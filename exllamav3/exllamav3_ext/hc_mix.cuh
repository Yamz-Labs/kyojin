#pragma once

#include <ATen/Tensor.h>

class Graph;

// Fused mHC HyperConnection mix / HyperHead collapse

bool hc_mix_supported(int device);
int hc_mix_num_chunks(int R, int row_len);

void hc_mix
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor collapsed
);

void hc_mix_set_onepass(int64_t on);

void hc_mix_norm
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor y,
    const c10::optional<at::Tensor>& w,
    double norm_eps,
    double constant_bias,
    double constant_scale
);
void hc_head
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    at::Tensor partials,
    at::Tensor collapsed
);

void hc_apply
(
    at::Tensor x,
    const at::Tensor& y,
    const at::Tensor& post,
    const c10::optional<at::Tensor>& comb
);

// Prefill apply+mix_norm fusion (pffuse1, EXL3_PF_HC_FUSE, default off)
void hc_pf_apply_mix_norm
(
    at::Tensor x,
    const at::Tensor& ya,
    const at::Tensor& post_a,
    const at::Tensor& comb_a,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor y,
    const c10::optional<at::Tensor>& w,
    double norm_eps,
    double constant_bias,
    double constant_scale
);

void gr_mix
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& upt,
    const at::Tensor& w,
    double rms_eps,
    at::Tensor dots,
    c10::optional<at::Tensor> post,
    at::Tensor mixed
);

// Int8 variant (per-row fn scale, per-output-column up scale); see hc_mix.cu
void gr_mix_q8
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& fn_scale,
    const at::Tensor& upt,
    const at::Tensor& up_scale,
    const at::Tensor& w,
    double rms_eps,
    at::Tensor dots,
    c10::optional<at::Tensor> post,
    at::Tensor mixed
);
