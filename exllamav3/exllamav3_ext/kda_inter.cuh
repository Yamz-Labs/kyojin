#pragma once
#include <ATen/Tensor.h>

// KDA inter-solve off-diagonal dots (kdaC7, EXL3_KDA_IS_SPLIT=3): Aqk off-diagonal 16x16 blocks (bf16, scaled)
// and Aoff = (k-dots) * beta (fp32), bitwise equal to vendor/fla/kda_inter_split.py inter_off_kernel (BK16, 2 warps).
void kda_inter_off
(
    const at::Tensor& q,        // [B,T,H,128] bfloat16, contiguous
    const at::Tensor& k,        // [B,T,H,128] bfloat16, contiguous
    const at::Tensor& g,        // [B,T,HV,128] float, contiguous
    const at::Tensor& beta,     // [B,T,HV] bfloat16, contiguous
    at::Tensor& Aqk,            // [B,T,HV,64] bfloat16 (off-diagonal blocks written)
    at::Tensor& Aoff,           // [B,T,HV,64] float (off-diagonal blocks written)
    double scale
);
