// fp32 MMA accumulation (no EXL3_GEMM_H_ACC): the fused kernel's expert groups give each block long k spans, where fp16
// accumulation of 3INST experts measured ~1.4x the KLD of the unfused path on a 3090; no speed cost
#define EXL3_GEMM_NO_H_ACC
#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k3_n128_cb0() { return exl3_moe_kernel<3, 128, 0>; }
fp_exl3_moe_kernel exl3_moe_kernel_k3_n256_cb0() { return exl3_moe_kernel<3, 256, 0>; }
