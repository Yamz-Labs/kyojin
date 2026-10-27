#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <vector>

// Grouped EXL3 MoE prefill for gfx11.5 (RDNA3.5 WMMA): tokens are sorted by expert and every
// projection runs as ONE launch over all experts. Trellis tiles are decoded in the GEMM (to LDS),
// no fp16 weight materialization. Gate+up share a launch. Any hidden / intermediate width that
// is a multiple of 128, any top-k, K = 2, 2.5, 3, 4, 5, 6, both codebooks (mcg / mul1).
void exl3_moe_prefill_wmma
(
    const at::Tensor& A,              // fp16 [rows, H]
    at::Tensor& output,               // fp32 [rows, H]
    const at::Tensor& selected,       // int64 [rows, top_k]
    const at::Tensor& weights,        // fp16 [rows, top_k]
    const at::Tensor& order,          // int64 [rows * top_k], stable argsort of selected
    const at::Tensor& expert_count,   // int64 [E + 1]
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    double K_gu,
    double K_down,
    at::Tensor& gu_had,               // fp16 [2 * A, H]
    at::Tensor& gu_out,               // fp16 [2 * A, I]
    at::Tensor& down_out,             // fp32 [A, H]
    at::Tensor& expert_offsets,       // int64 [E + 1]
    at::Tensor& inverse_order,        // int64 [A]
    at::Tensor& tiles,                // int32 [>= ceil(A / 64) + E]
    at::Tensor& tile_count,           // int32 [1]
    double act_limit,                 // clamped SwiGLU limit, 0 = unclamped
    bool mcg,                         // trellis codebook: true = mcg, false = mul1
    const c10::optional<at::Tensor>& addend   // fp16 [<= rows, H] widened and added to output rows (shared expert), or none
);

// Mixed-K layer (no pruned experts) in one pipeline: E-length pointer tables, one gather/act/reduce pass, the two GEMMs
// launched once per K group over that group's tile list.
void exl3_moe_prefill_wmma_mixed
(
    const at::Tensor& A, at::Tensor& output, const at::Tensor& selected, const at::Tensor& weights,
    const at::Tensor& order, const at::Tensor& expert_count,
    const at::Tensor& gate_trellis, const at::Tensor& gate_suh, const at::Tensor& gate_svh,
    const at::Tensor& up_trellis, const at::Tensor& up_suh, const at::Tensor& up_svh,
    const at::Tensor& down_trellis, const at::Tensor& down_suh, const at::Tensor& down_svh,
    const at::Tensor& egroup,
    const std::vector<double>& K_gu, const std::vector<double>& K_down,
    const std::vector<int64_t>& mcg, const std::vector<int64_t>& group_experts,
    at::Tensor& gu_had, at::Tensor& gu_out, at::Tensor& down_out,
    at::Tensor& expert_offsets, at::Tensor& inverse_order, at::Tensor& tiles, at::Tensor& tile_count,
    double act_limit, const c10::optional<at::Tensor>& addend
);

// Standalone grouped GEMM on pre-sorted rows (test / benchmark entry): C[sorted rows] =
// A[sorted rows] @ W_e (had domain, no input/output transforms), one launch.
void exl3_moe_prefill_gemm_test
(
    const at::Tensor& A,              // fp16 [A_rows, K]
    at::Tensor& C,                    // fp16 or fp32 [A_rows, N]
    const at::Tensor& expert_count,   // int64 [E + 1]
    const at::Tensor& trellis_table,  // int64 [E]
    double K,
    int n_size,
    at::Tensor& expert_offsets,
    at::Tensor& tiles,
    at::Tensor& tile_count
);

// Dense EXL3 GEMM (one group) through the expert prefill kernel: C = A @ W_inner, had domain, K 3 or 4, N % 256 == 0
void exl3_dense_mpw2
(
    const at::Tensor& A,              // fp16 [rows, K], unit column stride
    at::Tensor& C,                    // fp16 or fp32 [rows, N], contiguous
    const at::Tensor& trellis,        // int16 [K / 16, N / 16, 16 * bits]
    double K
);
