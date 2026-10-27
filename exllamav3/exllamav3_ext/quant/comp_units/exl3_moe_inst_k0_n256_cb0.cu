// fp32 MMA accumulation: see exl3_moe_inst_k1_cb0.cu
#define EXL3_GEMM_NO_H_ACC
#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k0_n256_cb0() { return exl3_moe_kernel<0, 256, 0>; }
