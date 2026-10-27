#pragma once

#include <ATen/Tensor.h>

bool exl3_moe_grouped_supported(int K_gate, int K_up, int K_down);

// Grouped prefill tier: experts with count in [count_lo, count_hi] (m_tile 32 or 64 row tiles). Their rows live in compact
// work buffers at group_base[e] + row (n_rows in all); the weighted outputs are summed per token in k order and added to output_state
void exl3_moe_grouped
(
    const at::Tensor& hidden_state,
    const at::Tensor& expert_count,
    const at::Tensor& expert_start,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const int act_function,
    const int K_gate,
    const int K_up,
    const int K_down,
    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,
    const float act_limit,
    const at::Tensor& output_state,
    const at::Tensor& flat_expert,
    const at::Tensor& inv_order,
    const at::Tensor& group_base,
    const int64_t n_rows,
    const int count_lo,
    const int count_hi,
    const int m_tile,
    const int intermediate_dim
);
