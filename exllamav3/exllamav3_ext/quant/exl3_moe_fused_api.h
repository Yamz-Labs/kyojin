#pragma once
// Entry points of the fused MoE half-layer (EXL3_MOE_FUSED), ROCm only. Implemented in exl3_moe_fused.cu (own translation unit since core5).
#include <ATen/ATen.h>
#include <vector>
#include <cstdint>
bool exl3_moe_fused_supported(int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB);
std::vector<int64_t> exl3_moe_fused_ws_offsets(int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB);
int64_t exl3_moe_fused_grid(int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB);
// wt: int64 [3 * (NEXP + 1)] device pointers of the gate/up/down trellis matrices (unit NEXP = shared expert); svt: int64 [6 * (NEXP + 1)] device pointers of
// gsuh gsvh usuh usvh dsuh dsvh. native = 1: engine trellis layout, 0: repacked layout (exl3_moe_fused_idx.h).
void exl3_moe_fused_half
(
    at::Tensor& x, const at::Tensor& fn, const at::Tensor& fn_scale, const at::Tensor& upt, const at::Tensor& up_scale, const at::Tensor& w_h,
    const at::Tensor& router, const at::Tensor& sgate, const at::Tensor& wt, const at::Tensor& svt,
    at::Tensor& ws, double rms_eps, int64_t native, int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB
);
// test entry: C = matvec(A, B) with the fused kernel's tile body. A half [R, K] (Hadamard domain), B int16 trellis [K/16, N/16, 16 bits] (native) or the repacked
// layout of the same matrix, C half or float [R, N]. R 1..4; (bits, K) from a fixed table (see exl3_moe_fused.cu).
void exl3_moe_fused_tile_test(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int64_t bits, int64_t native);
