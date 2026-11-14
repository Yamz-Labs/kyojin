// SPDX-License-Identifier: MIT
// Kyojin: host entry points of the routed (MoE) HIP MMQ (rocm/gmmq). Plain header: no vendored ggml names, so the
// torch translation unit (gmmq_torch.cu) can include it next to the torch headers.
#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

// Bytes of device scratch gmmq_moe() needs for these shapes (id maps + q8_1 activations).
size_t gmmq_moe_scratch_bytes(int type, int K, int n_tokens, int n_experts, int n_expert_used);

// out[t * n_expert_used + s, :M] = W[ids[t, s]] (M x K, raw GGUF blocks of `type`) @ x[t, :K], fp32.
//   type: ggml type id (Q4_K 12, Q5_K 13, Q5_1 7, Q8_0 8).
//   w_ptrs: device array of n_experts pointers, each to one expert's M rows of K / blck raw blocks (row-major).
//     MMQ loads K in steps of 256 values: when K % 256 != 0, the tail of each row reads the next row and the last row
//     of an expert reads up to (256 - K % 256) / blck blocks past its end. That memory must be readable; the Q8_0 / Q5_1
//     tile loads zero the scales of blocks past the row end, so their contents never reach the result.
//   x: device fp32 [n_tokens, K]; ids: device int32 [n_tokens, n_expert_used] (ids outside [0, n_experts) are skipped
//     and leave their output row untouched).
//   tile_cols / max_expert_rows: 0 = llama.cpp's default column tile (by n_tokens * n_expert_used); otherwise the
//     MMQ column tile width / the largest per-expert row count when the caller knows it (Gufo's routed knobs).
// Returns 0, or < 0 on a bad argument / launch error (message on stderr).
int gmmq_moe(int type, const void * const * w_ptrs, const float * x, const int32_t * ids, float * out,
             int M, int K, int n_tokens, int n_experts, int n_expert_used,
             void * scratch, size_t scratch_bytes, int tile_cols, int max_expert_rows, hipStream_t stream);
