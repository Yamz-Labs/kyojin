#pragma once
#if defined(USE_ROCM)
#include <ATen/Tensor.h>
bool gm_dense_q8(const at::Tensor& qb, const at::Tensor& packed, at::Tensor& out, int64_t cpb);
#endif
