#pragma once

#if defined(USE_ROCM)

#include <ATen/Tensor.h>

// Decode-only standard MoE router for up to eight gfx1200/gfx1201 rows. The gate is
// supplied as a persistent contiguous (experts, hidden) transpose so each GEMV is coalesced.
// The bsz1 name is retained as a compatibility binding.
void gm_routing_std_bsz1
(
    const at::Tensor& hidden,
    const at::Tensor& gate_t,
    at::Tensor& scores,
    at::Tensor& topk_indices,
    at::Tensor& topk_weights
);

void gm_route(const at::Tensor& x, const at::Tensor& gate, const at::Tensor& bias, at::Tensor& scores, at::Tensor& sel, at::Tensor& w,
              int64_t gating, bool norm, double scale, double eps);

#endif
