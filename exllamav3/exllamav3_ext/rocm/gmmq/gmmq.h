// SPDX-License-Identifier: MIT
// Kyojin: routed (MoE) HIP MMQ torch op (rocm/gmmq/gmmq_torch.cu). HIP only.
#pragma once
#include <torch/extension.h>

// out[t * n_used + s, :] = dequant(W[ids[t, s]]) @ x[t, :] in fp32, activations quantized to q8_1 (llama.cpp MMQ).
// x: fp16/fp32 [T, K]; ids: int32 [T, n_used]; w_ptrs: int64 [E] device pointers to each expert's raw GGUF blocks
// (M rows of K / blck blocks); fmt: ggml type id (Q4_K 12, Q5_K 13, Q5_1 7, Q8_0 8); out: fp32 [T * n_used, M].
// When K % 256 != 0 the last row of each expert is read up to 256 values past its end; those blocks get zero scales (see gmmq_api.h).
void fq_mmq_moe(torch::Tensor x, torch::Tensor ids, torch::Tensor w_ptrs, int64_t fmt, int64_t M, torch::Tensor out,
                int64_t tile_cols, int64_t max_expert_rows);
