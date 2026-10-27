// Host entry points of the foreign-quant kernels (defined in fq_dequant.cu /
// fq_gemm.cu, bound in bindings.cpp). Portable CUDA/HIP.
#pragma once

#include <torch/extension.h>

void fq_dequant_q8_0(torch::Tensor packed, torch::Tensor out);
void fq_dequant_iq2_xxs(torch::Tensor packed, torch::Tensor out);
void fq_dequant_nvfp4(torch::Tensor packed, torch::Tensor scales, double global_scale, torch::Tensor out);
void fq_gemm_q8_0(torch::Tensor x, torch::Tensor packed, torch::Tensor out);
void fq_gemm_iq2_xxs(torch::Tensor x, torch::Tensor packed, torch::Tensor out);
void fq_gemm_nvfp4(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, double global_scale,
                   torch::Tensor out);
void fq_gemv_q8_0(torch::Tensor x, torch::Tensor packed, torch::Tensor out);
void fq_gemv_iq2_xxs(torch::Tensor x, torch::Tensor packed, torch::Tensor out);
void fq_gemv_nvfp4(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, double global_scale,
                   torch::Tensor out, bool pair = true);
void fq_moe_iq2xxs_grouped(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor up_ptrs, torch::Tensor down_ptrs,
                           int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out);
void fq_mlp_shared_iq2xxs(torch::Tensor x, torch::Tensor up_packed, torch::Tensor h);
// q8_1-activation path (fq_gemv_q8.cu): quantize + integer-dp4a dots.
torch::Tensor fq_q8_scratch(int dev, int M, int K);
void fq_quant_q8_1_rows(torch::Tensor x, torch::Tensor qb);
void fq_gemv_iq2_xxs_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_gemv_nvfp4_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor scales,
                      double global_scale, torch::Tensor out);
void fq_gemv_q8_0_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
// q8_1-activation grouped MoE (fq_moe_q8.cu), same args/contract as above.
bool fq_use_q8moe();
void fq_moe_iq2xxs_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                              torch::Tensor up_ptrs, torch::Tensor down_ptrs,
                              int64_t up_stride, int64_t down_stride,
                              torch::Tensor partial, torch::Tensor out);
// Unfused dequant-to-fp16 for the K-quants + IQ4_XS (fq_dequant_kq.cu):
// packed = raw GGUF bytes [N, Kb*bb] (bb = 144/176/210/136 B per
// 256-block), out = fp16 [N, K] with K = Kb*256. Bitwise-identical to
// dequant_row(blocks, fmt).astype(np.float16) in foreign_quant/k_quants.py.
void fq_dequant_q4_k(torch::Tensor packed, torch::Tensor out);
void fq_dequant_q5_k(torch::Tensor packed, torch::Tensor out);
void fq_dequant_q6_k(torch::Tensor packed, torch::Tensor out);
void fq_dequant_iq4_xs(torch::Tensor packed, torch::Tensor out);
// q8_1-activation K-quant + IQ4_XS decode GEMV (fq_gemv_kq.cu).
// qb = caller-quantized activations (fq_quant_q8_1_rows); packed = raw GGUF
// bytes [N, nb256*block_bytes] (144/176/210/136 B per 256-block).
void fq_gemv_q4_k_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_gemv_q5_k_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_gemv_q6_k_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
void fq_gemv_iq4_xs_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out);
// q8_1-activation grouped MoE for KQ/IQ4_XS SwiGLU experts (fq_moe_kq.cu):
// gate/up/down_ptrs = per-expert base addresses (E int64);
// strides are per-row bytes; partial = [S, K] float scratch.
void fq_moe_q4k_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                           torch::Tensor down_ptrs,
                           int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out);
void fq_moe_q5k_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                           torch::Tensor down_ptrs,
                           int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out);
void fq_moe_q6k_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                           torch::Tensor down_ptrs,
                           int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out);
void fq_moe_iq4xs_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                             torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                             torch::Tensor down_ptrs,
                             int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                             torch::Tensor partial, torch::Tensor out);
