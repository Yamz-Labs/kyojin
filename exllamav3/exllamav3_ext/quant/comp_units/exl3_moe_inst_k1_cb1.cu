#define EXL3_GEMM_NO_H_ACC   // fp32 MMA accumulation: see exl3_moe_inst_k1_cb0.cu (halves the block error, no speed cost)
#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k1_n128_cb1() { return exl3_moe_kernel<1, 128, 1>; }
fp_exl3_moe_kernel exl3_moe_kernel_k1_n256_cb1() { return exl3_moe_kernel<1, 256, 1>; }
