// SPDX-License-Identifier: MIT
// Kyojin: mul_mat_q_case<GGML_TYPE_Q5_1> instance for the routed (MoE) HIP MMQ. See gmmq_case.cuh.
#if defined(USE_ROCM)
#include "gmmq_case.cuh"
GMMQ_INST_CASE(GGML_TYPE_Q5_1)
#endif
