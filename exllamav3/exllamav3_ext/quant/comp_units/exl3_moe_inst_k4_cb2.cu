#define EXL3_GEMM_NO_H_ACC   // fp32 MMA accumulation: see exl3_moe_inst_k1_cb0.cu (halves the block error, no speed cost)
#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k4_n128_cb2() { return exl3_moe_kernel<4, 128, 2>; }
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_cb2() { return exl3_moe_kernel<4, 256, 2>; }
