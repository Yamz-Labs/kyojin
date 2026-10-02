#pragma once

#if defined(USE_ROCM)

#include <ATen/Tensor.h>

// QSA selection: fp16 (R, T) with 128-aligned row stride, top-k (1 <= k <= 512) into
// contiguous int32 (R, W), k <= W <= 512, ascending indices then -1 padding.
void dsa_topk_gfx12(const at::Tensor& scores, at::Tensor& indices, int64_t k);

#endif
