// llama.cpp mmvq-style IQ decode GEMV (fq_mmvq.cu, CUDA only): q8_1 activations qb [M, K/32*36] (fq_quant_q8_1_rows),
// packed = raw GGUF bytes [N, K/QK*block_bytes], out [M, N] half or float. Any M (launches of <= 8 columns).
#pragma once
#include <torch/extension.h>

void fq_mmvq_iq2_xxs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_mmvq_iq2_xs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_mmvq_iq2_s(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_mmvq_iq3_xxs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_mmvq_iq3_s(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_mmvq_iq4_nl(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_mmvq_iq4_xs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_mmvq_q5_k(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
// Grouped decode MoE for IQ experts: format ids 0 iq2_xxs, 1 iq2_xs, 2 iq2_s, 3 iq3_xxs, 4 iq3_s, 5 iq4_nl, 6 iq4_xs, 7 q5_k (gate/up and down independent).
// Arguments and results as fq_moe_kq_grouped_mixed_q8 (fq_api.h).
void fq_moe_mmvq_grouped(torch::Tensor h, torch::Tensor sel, torch::Tensor wts, torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                         torch::Tensor down_ptrs, int64_t gate_stride, int64_t up_stride, int64_t down_stride, int64_t type_ug, int64_t type_down,
                         torch::Tensor partial, torch::Tensor out);
// Dequantize IQ blocks to fp16 [N, K] (K % 256 == 0); fmt = MQ type id as above (0..6). Port of ggml convert.cu dequantize_block_iq*.
void fq_dequant_iq(torch::Tensor packed, torch::Tensor out, int64_t fmt);
