#pragma once
#include <ATen/Tensor.h>

// glmsmall1: MLA decode absorb / unfold on VALU for R <= 8 rows, bit-identical to the Triton kernels (ROCm only)
void mla_absorb_dec(const at::Tensor& q, const at::Tensor& w_uk_flat, at::Tensor& out);
void mla_unfold_dec(const at::Tensor& o_lat, const at::Tensor& w_uv_flat, at::Tensor& out);
