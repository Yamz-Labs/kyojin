// SPDX-License-Identifier: MIT
// Kyojin: routed (MoE) F16-WMMA prefill torch op (rocm/routed_f16/routed_f16.cu, from Gufo; see THIRD_PARTY_NOTICES.md). HIP gfx1151 only.
#pragma once
#include <torch/extension.h>

// out[t * top_k + s, :K] (half) = down[e] @ (silu(gate[e] @ x[t]) * (up[e] @ x[t])), e = ids[t, s]. counts int32 [E], tiles int32 (expert | 128-row
// tile << 16); *_ptrs int64 [E] device pointers to each expert's raw GGUF rows. Formats: gate/up Q4_K 12 / Q5_K 13, down Q5_1 7 / Q8_0 8.
void fq_routed_f16_moe(torch::Tensor x, torch::Tensor ids, torch::Tensor counts, torch::Tensor tiles,
                       torch::Tensor gate_ptrs, torch::Tensor up_ptrs, torch::Tensor down_ptrs, int64_t gate_fmt,
                       int64_t down_fmt, int64_t I, torch::Tensor out);

// out[t, :] (float) = sum_s weights[t, s] * rows[t * top_k + s, :], s in order (rows half [T * top_k, K], weights float [T, top_k]); the weighted
// top-k reduction after fq_routed_f16_moe (Gufo MoeEpilogueVec4Kernel<__half> without the shared expert).
void fq_routed_f16_combine(torch::Tensor rows, torch::Tensor weights, torch::Tensor out);
