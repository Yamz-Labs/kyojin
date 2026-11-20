// llama.cpp mmvq-style IQ decode GEMV (fq_mmvq.cu, CUDA and HIP): q8_1 activations qb [M, K/32*36] (fq_quant_q8_1_rows),
// packed = raw GGUF bytes [N, K/QK*block_bytes], out [M, N] half or float. Any M (launches of <= 8 columns).
#pragma once
#include <torch/extension.h>

void fq_mmvq_iq2_xxs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_iq2_xs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_iq2_s(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_iq3_xxs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_iq3_s(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_iq4_nl(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_iq4_xs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_q5_k(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_q4_k(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_q6_k(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_q3_k(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_q4_0(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_q4_1(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_q5_0(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
void fq_mmvq_q5_1(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias = c10::nullopt);
// Grouped decode MoE for IQ experts: format ids 0 iq2_xxs, 1 iq2_xs, 2 iq2_s, 3 iq3_xxs, 4 iq3_s, 5 iq4_nl, 6 iq4_xs, 7 q5_k (gate/up and down independent).
// Arguments and results as fq_moe_kq_grouped_mixed_q8 (fq_api.h).
void fq_moe_mmvq_grouped(torch::Tensor h, torch::Tensor sel, torch::Tensor wts, torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                         torch::Tensor down_ptrs, int64_t gate_stride, int64_t up_stride, int64_t down_stride, int64_t type_ug, int64_t type_down,
                         torch::Tensor partial, torch::Tensor out);
// EXL3_FQ_MMVQ_MXFP4 (CUDA, R == 1): llama.cpp-shaped MXFP4 MoE decode on the stacked experts through int64 pointer tables (fq_mmvq.cu).
// gu: act [S, I] half = swiglu_oai(gate + bias, up + bias, lim); qb = ONE q8_1 row of the input. down: out [S, N] float = q8_1(act)[slot] . W^T + bias.
void fq_mxfp4_moe_gu(torch::Tensor qb, torch::Tensor sel, torch::Tensor gate_ptrs, torch::Tensor gate_bias, torch::Tensor up_ptrs,
                     torch::Tensor up_bias, torch::Tensor act, int64_t I, int64_t K, double lim);
void fq_mxfp4_moe_down(torch::Tensor qb, torch::Tensor sel, torch::Tensor down_ptrs, torch::Tensor down_bias, torch::Tensor out, int64_t N,
                       int64_t K);
// Dequantize IQ blocks to fp16 [N, K] (K % 256 == 0); fmt = MQ type id as above (0..6, 10 q3_k). Port of ggml convert.cu dequantize_block_iq*.
void fq_dequant_iq(torch::Tensor packed, torch::Tensor out, int64_t fmt);
#if !defined(USE_ROCM)
// Few-row MMQ (fq_mmq_few.cu, CUDA sm_80+, port of llama.cpp PR #31533): out[R, N] = x[R, K] (half) @ W^T, R 1..24, raw GGUF rows
// packed [N, K/256 * unit bytes], fmt 0 Q8_0, 1 Q4_K, 2 Q5_K, 3 Q6_K, 4 Q3_K, 5 IQ4_XS; scratch >= ceil(R/8)*K*9 bytes. False = not served (nothing launched).
bool fq_mmq_few(torch::Tensor x, torch::Tensor scratch, torch::Tensor packed, torch::Tensor out, int fmt);
#endif

// ABI marker: the fq_mmvq_* functions above take an optional half bias applied in the store (EXL3_FQ_MMVQ_EPI).
inline int fq_mmvq_epi_abi() { return 1; }

// EXL3_MOE_FOLD_FUSED: out[H] (half or float) = sum over the S slots of d[s, :H] * float(rw[s]), torch's reduce order (fq_mmvq.cu).
void fq_moe_fold(torch::Tensor d, torch::Tensor rw, torch::Tensor out);
