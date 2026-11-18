// IQ-format decode GEMV, llama.cpp mmvq style (CUDA and HIP, wave32): q8_1 activations, integer dp4a dots of the ggml vec_dot_iq*_q8_1
// functions (ggml_mmvq/ggml_vecdotq_iq.cuh), 1..8 activation columns per launch (weights read once for all columns).
//
// The kernel structure follows llama.cpp ggml/src/ggml-cuda/mmvq.cu (mul_mat_vec_q; https://github.com/ggml-org/llama.cpp,
// commit 7182ef7). MIT License, Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.md.
// Per block: nwarps warps of 32 lanes, RPB output rows; a lane takes (qi / vdr)-lane groups per weight block and walks the
// row with stride blocks_per_iter; per-warp partials meet in shared memory, then a warp shuffle reduction.
//
// Output [M, N] row-major (half or float); qb = fq_quant_q8_1_rows output [M, K/32 * 36] bytes (block_q8_1: half d, half s, 32 x i8);
// packed = raw GGUF bytes [N, K/QK * block_bytes].
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <map>
#include <tuple>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_api.h"
#include "ggml_mmvq/ggml_vecdotq_iq.cuh"
#include "ggml_mmvq/ggml_vecdotq_k46.cuh"
#include "ggml_mmvq/ggml_vecdotq_legacy.cuh"
#include "ggml_mmvq/ggml_vecdotq_mxfp4.cuh"

#if defined(USE_ROCM)
// HIP: maskless shuffles (RDNA runs these kernels as wave32, MQ_WARP = 32)
#define __shfl_xor_sync(mask, var, ...) __shfl_xor(var, __VA_ARGS__)
#endif

namespace {

using namespace ggml_iq;

enum { MQ_IQ2_XXS = 0, MQ_IQ2_XS = 1, MQ_IQ2_S = 2, MQ_IQ3_XXS = 3, MQ_IQ3_S = 4, MQ_IQ4_NL = 5, MQ_IQ4_XS = 6, MQ_Q5_K = 7, MQ_Q4_K = 8, MQ_Q6_K = 9, MQ_Q3_K = 10, MQ_Q4_0 = 11, MQ_Q4_1 = 12, MQ_Q5_0 = 13, MQ_Q5_1 = 14, MQ_MXFP4 = 15, MQ_NTYPES = 16 };

// qk = weight elements per block, qi = ints per block (ggml QI*), vdr = ints per lane per call, bb = block bytes
template <int T> struct MqTraits;
#define MQ_TRAITS(TYPE, QK, QI, VDR, BB, FN)                                                                                    \
    template <> struct MqTraits<TYPE> {                                                                                         \
        static constexpr int qk = QK, qi = QI, vdr = VDR, bb = BB;                                                              \
        static __device__ __forceinline__ float dot(const void* vbq, const block_q8_1* y, int kbx, int iqs)                   \
        { return FN(vbq, y, kbx, iqs); }                                                                                        \
    }
MQ_TRAITS(MQ_IQ2_XXS, 256, 16, VDR_IQ2_XXS_Q8_1_MMVQ, 66, vec_dot_iq2_xxs_q8_1);
MQ_TRAITS(MQ_IQ2_XS, 256, 16, VDR_IQ2_XS_Q8_1_MMVQ, 74, vec_dot_iq2_xs_q8_1);
MQ_TRAITS(MQ_IQ2_S, 256, 16, VDR_IQ2_S_Q8_1_MMVQ, 82, vec_dot_iq2_s_q8_1);
MQ_TRAITS(MQ_IQ3_XXS, 256, 16, VDR_IQ3_XXS_Q8_1_MMVQ, 98, vec_dot_iq3_xxs_q8_1);
MQ_TRAITS(MQ_IQ3_S, 256, 16, VDR_IQ3_S_Q8_1_MMVQ, 110, vec_dot_iq3_s_q8_1);
MQ_TRAITS(MQ_IQ4_NL, 32, 4, VDR_IQ4_NL_Q8_1_MMVQ, 18, vec_dot_iq4_nl_q8_1);
MQ_TRAITS(MQ_IQ4_XS, 256, 32, VDR_IQ4_XS_Q8_1_MMVQ, 136, vec_dot_iq4_xs_q8_1);
MQ_TRAITS(MQ_Q5_K, 256, 32, VDR_Q5_K_Q8_1_MMVQ, 176, vec_dot_q5_K_q8_1);
MQ_TRAITS(MQ_Q4_K, 256, 32, VDR_Q4_K_Q8_1_MMVQ, 144, vec_dot_q4_K_q8_1);
MQ_TRAITS(MQ_Q6_K, 256, 32, VDR_Q6_K_Q8_1_MMVQ, 210, vec_dot_q6_K_q8_1);
MQ_TRAITS(MQ_Q3_K, 256, 16, VDR_Q3_K_Q8_1_MMVQ, 110, vec_dot_q3_K_q8_1);
// legacy 32-element GGUF blocks (EXL3_FQ_MMVQ_LEGACY in foreign_linear.py): qi = QI4_0 = QI5_0 = 4 ints per block
MQ_TRAITS(MQ_Q4_0, 32, 4, VDR_Q4_0_Q8_1_MMVQ, 18, vec_dot_q4_0_q8_1);
MQ_TRAITS(MQ_Q4_1, 32, 4, VDR_Q4_1_Q8_1_MMVQ, 20, vec_dot_q4_1_q8_1);
MQ_TRAITS(MQ_Q5_0, 32, 4, VDR_Q5_0_Q8_1_MMVQ, 22, vec_dot_q5_0_q8_1);
MQ_TRAITS(MQ_Q5_1, 32, 4, VDR_Q5_1_Q8_1_MMVQ, 24, vec_dot_q5_1_q8_1);
MQ_TRAITS(MQ_MXFP4, 32, 4, VDR_MXFP4_Q8_1_MMVQ, 17, vec_dot_mxfp4_q8_1);
#undef MQ_TRAITS

constexpr int MQ_WARP = 32;
constexpr int MQ_MAX_COLS = 8;

// llama.cpp calc_nwarps / calc_rows_per_block (generic table, Ampere): 1..4 columns 4 warps, 5..8 columns 2 warps; 2 rows per block when batched
constexpr int mq_nwarps(int ncols) { return ncols <= 4 ? 4 : 2; }
constexpr int mq_rows(int ncols) { return ncols == 1 ? 1 : 2; }

__device__ __forceinline__ float mq_warp_sum(float x)
{
#pragma unroll
    for (int o = MQ_WARP / 2; o > 0; o >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, o, MQ_WARP);
    return x;
}

template <typename OUT> __device__ __forceinline__ void mq_store(OUT* p, float v);
template <> __device__ __forceinline__ void mq_store<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void mq_store<__half>(__half* p, float v) { *p = __float2half_rn(v); }

// EXL3_FQ_MMVQ_EPI: optional bias in the store. half out: round the dot to half FIRST, then half(float(h) + float(b)) (torch's `y + bias` on half);
// float out: v + float(b). bias == nullptr is the plain store.
template <typename OUT> __device__ __forceinline__ void mq_store_b(OUT* p, float v, const __half* __restrict__ bias, int64_t n);
template <> __device__ __forceinline__ void mq_store_b<float>(float* p, float v, const __half* __restrict__ bias, int64_t n)
{
    *p = bias ? v + __half2float(bias[n]) : v;
}
template <> __device__ __forceinline__ void mq_store_b<__half>(__half* p, float v, const __half* __restrict__ bias, int64_t n)
{
    __half h = __float2half_rn(v);
    if (bias) h = __float2half_rn(__half2float(h) + __half2float(bias[n]));
    *p = h;
}

// One launch: out[j][row] = dot(weights row, activation column j), j < NC. Weight blocks of row r start at vx + r * row_bytes.
template <int T, int NC, typename OUT>
__global__ void __launch_bounds__(MQ_WARP * mq_nwarps(NC), 1)
fq_mmvq_kernel(const uint8_t* __restrict__ vx, const block_q8_1* __restrict__ vy, OUT* __restrict__ dst,
               int ncols_x, int nrows, int64_t row_blocks, int64_t y_col_blocks, int64_t dst_col_stride,
               const __half* __restrict__ bias)
{
    using Tr = MqTraits<T>;
    constexpr int NW = mq_nwarps(NC);
    constexpr int RPB = mq_rows(NC);
    constexpr int blocks_per_iter = Tr::vdr * NW * MQ_WARP / Tr::qi;

    const int tid = MQ_WARP * threadIdx.y + threadIdx.x;
    const int row0 = RPB * blockIdx.x;
    const int blocks_per_row_x = ncols_x / Tr::qk;

    float tmp[NC][RPB];
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int i = 0; i < RPB; ++i)
            tmp[j][i] = 0.0f;

    // rows past the end read the last valid row (their result is not stored)
    int64_t rbase[RPB];
#pragma unroll
    for (int i = 0; i < RPB; ++i)
        rbase[i] = (int64_t)min(row0 + i, nrows - 1) * row_blocks;

    for (int kbx = tid / (Tr::qi / Tr::vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter)
    {
        const int kby = kbx * (Tr::qk / 32);
        const int kqs = Tr::vdr * (tid % (Tr::qi / Tr::vdr));
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int i = 0; i < RPB; ++i)
                tmp[j][i] += Tr::dot(vx + rbase[i] * Tr::bb, &vy[j * y_col_blocks + kby], kbx, kqs);
    }

    __shared__ float tmp_shared[NW > 1 ? NW - 1 : 1][NC][RPB][MQ_WARP];
    if constexpr (NW > 1)
    {
        if (threadIdx.y > 0)
        {
#pragma unroll
            for (int j = 0; j < NC; ++j)
#pragma unroll
                for (int i = 0; i < RPB; ++i)
                    tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
        }
        __syncthreads();
        if (threadIdx.y > 0)
            return;
    }
#pragma unroll
    for (int j = 0; j < NC; ++j)
    {
#pragma unroll
        for (int i = 0; i < RPB; ++i)
        {
            float v = tmp[j][i];
            if constexpr (NW > 1)
            {
#pragma unroll
                for (int l = 0; l < NW - 1; ++l)
                    v += tmp_shared[l][j][i][threadIdx.x];
            }
            v = mq_warp_sum(v);
            if (threadIdx.x == 0 && row0 + i < nrows)
                mq_store_b<OUT>(&dst[(int64_t)j * dst_col_stride + row0 + i], v, bias, row0 + i);
        }
    }
}

template <int T, int NC, typename OUT>
void mq_launch(const uint8_t* w, const uint8_t* qb, OUT* out, int K, int N, int64_t y_col_blocks, int64_t dst_col_stride, const __half* bias, cudaStream_t stream)
{
    using Tr = MqTraits<T>;
    constexpr int NW = mq_nwarps(NC);
    constexpr int RPB = mq_rows(NC);
    dim3 grid((N + RPB - 1) / RPB, 1, 1), block(MQ_WARP, NW, 1);
    fq_mmvq_kernel<T, NC, OUT><<<grid, block, 0, stream>>>(w, reinterpret_cast<const block_q8_1*>(qb), out, K, N,
                                                           (int64_t)(K / Tr::qk), y_col_blocks, dst_col_stride, bias);
}

template <int T, typename OUT>
void mq_cols(int nc, const uint8_t* w, const uint8_t* qb, OUT* out, int K, int N, int64_t y_col_blocks, const __half* bias, cudaStream_t stream)
{
    switch (nc)
    {
        case 1: mq_launch<T, 1, OUT>(w, qb, out, K, N, y_col_blocks, N, bias, stream); break;
        case 2: mq_launch<T, 2, OUT>(w, qb, out, K, N, y_col_blocks, N, bias, stream); break;
        case 3: mq_launch<T, 3, OUT>(w, qb, out, K, N, y_col_blocks, N, bias, stream); break;
        case 4: mq_launch<T, 4, OUT>(w, qb, out, K, N, y_col_blocks, N, bias, stream); break;
        case 5: mq_launch<T, 5, OUT>(w, qb, out, K, N, y_col_blocks, N, bias, stream); break;
        case 6: mq_launch<T, 6, OUT>(w, qb, out, K, N, y_col_blocks, N, bias, stream); break;
        case 7: mq_launch<T, 7, OUT>(w, qb, out, K, N, y_col_blocks, N, bias, stream); break;
        default: mq_launch<T, 8, OUT>(w, qb, out, K, N, y_col_blocks, N, bias, stream); break;
    }
}

template <int T>
void mq_run(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, const std::optional<torch::Tensor>& bias_opt)
{
    using Tr = MqTraits<T>;
    TORCH_CHECK(qb.is_cuda() && packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(qb.is_contiguous() && packed.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(out.dtype() == torch::kHalf || out.dtype() == torch::kFloat, "out must be half or float");
    TORCH_CHECK_DIM(qb, 2);
    TORCH_CHECK_DIM(packed, 2);
    TORCH_CHECK_DIM(out, 2);
    const int M = (int)qb.size(0);
    const int N = (int)packed.size(0);
    const int K = (int)(qb.size(1) / 36 * 32);
    TORCH_CHECK((int)out.size(0) == M && (int)out.size(1) == N, "out must be [M, N]");
    TORCH_CHECK(K % Tr::qk == 0, "K must be a multiple of the format block size");
    TORCH_CHECK(packed.size(1) == (int64_t)(K / Tr::qk) * Tr::bb, "packed width must match K");
    TORCH_CHECK((uintptr_t)packed.data_ptr() % 4 == 0 && ((int64_t)(K / Tr::qk) * Tr::bb) % 2 == 0, "weights must be 2-byte aligned (4 for IQ4_XS)");
    TORCH_CHECK(Tr::bb % 4 == 2 || ((int64_t)(K / Tr::qk) * Tr::bb) % 4 == 0, "this format needs 4-byte aligned weight rows");
    TORCH_CHECK((uintptr_t)qb.data_ptr() % 4 == 0, "qb must be 4-byte aligned");
    const __half* bias = nullptr;
    if (bias_opt.has_value() && bias_opt->defined())
    {
        TORCH_CHECK(bias_opt->is_cuda() && bias_opt->is_contiguous() && bias_opt->dtype() == torch::kHalf, "bias must be a contiguous CUDA half tensor");
        TORCH_CHECK(bias_opt->numel() == N, "bias must have N elements");
        bias = reinterpret_cast<const __half*>(bias_opt->data_ptr<at::Half>());
    }
    at::cuda::CUDAGuard guard(qb.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t y_col_blocks = K / 32;
    const uint8_t* w = packed.data_ptr<uint8_t>();
    const uint8_t* q = qb.data_ptr<uint8_t>();
    for (int m0 = 0; m0 < M; m0 += MQ_MAX_COLS)
    {
        const int nc = std::min(MQ_MAX_COLS, M - m0);
        const uint8_t* qm = q + (int64_t)m0 * y_col_blocks * 36;
        if (out.dtype() == torch::kHalf)
            mq_cols<T, __half>(nc, w, qm, reinterpret_cast<__half*>(out.data_ptr<at::Half>()) + (int64_t)m0 * N, K, N, y_col_blocks, bias, stream);
        else
            mq_cols<T, float>(nc, w, qm, out.data_ptr<float>() + (int64_t)m0 * N, K, N, y_col_blocks, bias, stream);
    }
}


// ---------------------------------------------------------------------------------------------------------------------------------
// Grouped decode MoE (slot-parallel): assignment s = row * topk + j -> expert sel[s]. Up+gate kernel: act[s, n] = silu(gate . h) * (up . h)
// for the q8_1 activation row s / topk; down kernel: partial[s, n] = down[sel[s]] row n . q8_1(act row s). One warp = 32 / LPR rows
// side by side (LPR lanes per row; 16 when a row has only 2 weight blocks, e.g. the 512-wide down projection), each warp 2 such row
// groups in sequence, 4 warps per block. Same dots and the same half-rounded SwiGLU epilogue as fq_moe_kq.cu.
// ---------------------------------------------------------------------------------------------------------------------------------
constexpr int MQM_WARPS = 4;
constexpr int MQM_ITER = 2;

template <int LPR> __device__ __forceinline__ float mq_group_sum(float x)
{
#pragma unroll
    for (int o = LPR / 2; o > 0; o >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, o, MQ_WARP);
    return x;
}

template <int T, int LPR>
__global__ void __launch_bounds__(MQ_WARP * MQM_WARPS)
fq_mmvq_moe_upgate_kernel(const uint8_t* __restrict__ qb_h, const int64_t* __restrict__ sel,
                          const uint8_t* const* __restrict__ gate_bases, const uint8_t* const* __restrict__ up_bases,
                          int64_t gate_stride, int64_t up_stride, int64_t qb_stride, __half* __restrict__ act,
                          int I, int K, int E, int topk)
{
    using Tr = MqTraits<T>;
    constexpr int GPW = MQ_WARP / LPR;                       // row groups per warp
    constexpr int ROWS_BLOCK = MQM_WARPS * MQM_ITER * GPW;
    const int s = blockIdx.y;
    int64_t e = sel[s];
    e = (e < 0 || e >= E) ? 0 : e;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int grp = lane / LPR, ll = lane % LPR;
    const block_q8_1* y = reinterpret_cast<const block_q8_1*>(qb_h + (int64_t)(s / topk) * qb_stride);
    const int nb = K / Tr::qk;
    constexpr int per_iter = LPR * Tr::vdr / Tr::qi;
    const uint8_t* gb = gate_bases[e];
    const uint8_t* ub = up_bases[e];
#pragma unroll
    for (int it = 0; it < MQM_ITER; ++it)
    {
        const int n = blockIdx.x * ROWS_BLOCK + (warp * MQM_ITER + it) * GPW + grp;
        const int nr = min(n, I - 1);
        const uint8_t* gr = gb + (int64_t)nr * gate_stride;
        const uint8_t* ur = ub + (int64_t)nr * up_stride;
        float g = 0.0f, u = 0.0f;
        for (int kbx = ll / (Tr::qi / Tr::vdr); kbx < nb; kbx += per_iter)
        {
            const int kqs = Tr::vdr * (ll % (Tr::qi / Tr::vdr));
            const block_q8_1* yb = y + kbx * (Tr::qk / 32);
            g += Tr::dot(gr, yb, kbx, kqs);
            u += Tr::dot(ur, yb, kbx, kqs);
        }
        g = mq_group_sum<LPR>(g);
        u = mq_group_sum<LPR>(u);
        if (ll == 0 && n < I)
        {
            const float gh = __half2float(__float2half_rn(g));
            const float uh = __half2float(__float2half_rn(u));
            const float sig = 1.0f / (1.0f + expf(-gh));
            act[(int64_t)s * I + n] = __float2half_rn(sig * gh * uh);
        }
    }
}

template <int T, int LPR>
__global__ void __launch_bounds__(MQ_WARP * MQM_WARPS)
fq_mmvq_moe_down_kernel(const uint8_t* __restrict__ qb_a, int64_t qb_stride, const int64_t* __restrict__ sel,
                        const uint8_t* const* __restrict__ down_bases, int64_t down_stride, float* __restrict__ partial,
                        int I, int K, int E)
{
    using Tr = MqTraits<T>;
    constexpr int GPW = MQ_WARP / LPR;
    constexpr int ROWS_BLOCK = MQM_WARPS * MQM_ITER * GPW;
    const int s = blockIdx.y;
    int64_t e = sel[s];
    e = (e < 0 || e >= E) ? 0 : e;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int grp = lane / LPR, ll = lane % LPR;
    const block_q8_1* y = reinterpret_cast<const block_q8_1*>(qb_a + (int64_t)s * qb_stride);
    const int nb = I / Tr::qk;
    constexpr int per_iter = LPR * Tr::vdr / Tr::qi;
    const uint8_t* db = down_bases[e];
#pragma unroll
    for (int it = 0; it < MQM_ITER; ++it)
    {
        const int n = blockIdx.x * ROWS_BLOCK + (warp * MQM_ITER + it) * GPW + grp;
        const uint8_t* dr = db + (int64_t)min(n, K - 1) * down_stride;
        float p = 0.0f;
        for (int kbx = ll / (Tr::qi / Tr::vdr); kbx < nb; kbx += per_iter)
        {
            const int kqs = Tr::vdr * (ll % (Tr::qi / Tr::vdr));
            p += Tr::dot(dr, y + kbx * (Tr::qk / 32), kbx, kqs);
        }
        p = mq_group_sum<LPR>(p);
        if (ll == 0 && n < K)
            partial[(int64_t)s * K + n] = p;
    }
}

template <int T>
void mq_moe_upgate_t(const uint8_t* qb_h, const int64_t* sel, const uint8_t* const* gb, const uint8_t* const* ub, int64_t gs, int64_t us,
                     int64_t qs, __half* act, int S, int I, int K, int E, int topk, cudaStream_t st)
{
    using Tr = MqTraits<T>;
    const int lanes = (K / Tr::qk) * (Tr::qi / Tr::vdr);
    auto go = [&](auto lpr_c)
    {
        constexpr int LPR = decltype(lpr_c)::value;
        constexpr int RB = MQM_WARPS * MQM_ITER * (MQ_WARP / LPR);
        dim3 grid((I + RB - 1) / RB, S, 1), block(MQ_WARP, MQM_WARPS, 1);
        fq_mmvq_moe_upgate_kernel<T, LPR><<<grid, block, 0, st>>>(qb_h, sel, gb, ub, gs, us, qs, act, I, K, E, topk);
    };
    if (lanes >= 32) go(std::integral_constant<int, 32>());
    else if (lanes >= 16) go(std::integral_constant<int, 16>());
    else go(std::integral_constant<int, 8>());
}

template <int T>
void mq_moe_down_t(const uint8_t* qb_a, int64_t qs, const int64_t* sel, const uint8_t* const* db, int64_t ds, float* partial, int S, int I, int K,
                   int E, cudaStream_t st)
{
    using Tr = MqTraits<T>;
    const int lanes = (I / Tr::qk) * (Tr::qi / Tr::vdr);
    auto go = [&](auto lpr_c)
    {
        constexpr int LPR = decltype(lpr_c)::value;
        constexpr int RB = MQM_WARPS * MQM_ITER * (MQ_WARP / LPR);
        dim3 grid((K + RB - 1) / RB, S, 1), block(MQ_WARP, MQM_WARPS, 1);
        fq_mmvq_moe_down_kernel<T, LPR><<<grid, block, 0, st>>>(qb_a, qs, sel, db, ds, partial, I, K, E);
    };
    if (lanes >= 32) go(std::integral_constant<int, 32>());
    else if (lanes >= 16) go(std::integral_constant<int, 16>());
    else go(std::integral_constant<int, 8>());
}


// Weighted fold: out[r, k] = sum_j w[r, j] * partial[r * topk + j, k] (same order and rounding as fq_moe_kq_combine_kernel)
__global__ void fq_mmvq_moe_combine_kernel(const float* __restrict__ partial, const __half* __restrict__ wts, float* __restrict__ out, int S, int K)
{
    partial += (int64_t)blockIdx.y * S * K;
    wts += (int64_t)blockIdx.y * S;
    out += (int64_t)blockIdx.y * K;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < K; k += gridDim.x * blockDim.x)
    {
        float acc = 0.0f;
        for (int s = 0; s < S; s++)
            acc = __fadd_rn(acc, __fmul_rn(partial[(int64_t)s * K + k], __half2float(wts[s])));
        out[k] = acc;
    }
}


// ---------------------------------------------------------------------------------------------------------------------------------
// MXFP4 decode MoE, llama.cpp launch shape (EXL3_FQ_MMVQ_MXFP4): one output row per block, MXM_WARPS = 4 warps (128 threads) per row
// (mmvq.cu:452-460 nwarps = 4 for ncols_dst = 1, :579-583 rows_per_block = 1), blocks_per_iter = vdr * 4 * 32 / qi = 64, per-warp partials
// through shared memory (mmvq.cu:763-840), grid (rows, slots). Existing grouped IQ kernels above use 1 warp per row (LPR 32), 8 rows per block.
// x is one q8_1 row shared by every slot (R == 1); expert weights / biases through the same int64 pointer tables the Triton path uses.
// gate and up of the same row are computed in one launch (mmvq.cu:661-760), + bias, fp16 rounding, swiglu_oai with clamp, fp16 out [S, I].
// ---------------------------------------------------------------------------------------------------------------------------------
constexpr int MXM_WARPS = 4;

template <bool GU>
__global__ void __launch_bounds__(MQ_WARP * MXM_WARPS, 1)
fq_mxfp4_moe_kernel(const uint8_t* __restrict__ qb, int64_t qb_slot_stride, const int* __restrict__ sel, const int64_t* __restrict__ wtab,
                    const int64_t* __restrict__ btab, const int64_t* __restrict__ wtab2, const int64_t* __restrict__ btab2,
                    void* __restrict__ outp, int N, int K, int E, float lim)
{
    using Tr = MqTraits<MQ_MXFP4>;
    constexpr int blocks_per_iter = Tr::vdr * MXM_WARPS * MQ_WARP / Tr::qi;
    const int row = blockIdx.x, slot = blockIdx.y;
    int e = sel[slot];
    e = (e < 0 || e >= E) ? 0 : e;
    const int tid = MQ_WARP * threadIdx.y + threadIdx.x;
    const int nb = K / Tr::qk;
    const block_q8_1* y = reinterpret_cast<const block_q8_1*>(qb + (int64_t)slot * qb_slot_stride);
    const uint8_t* w0 = reinterpret_cast<const uint8_t*>(wtab[e]) + (int64_t)row * nb * Tr::bb;
    const uint8_t* w1 = GU ? reinterpret_cast<const uint8_t*>(wtab2[e]) + (int64_t)row * nb * Tr::bb : nullptr;
    float a0 = 0.0f, a1 = 0.0f;
    for (int kbx = tid / (Tr::qi / Tr::vdr); kbx < nb; kbx += blocks_per_iter)
    {
        const int kqs = Tr::vdr * (tid % (Tr::qi / Tr::vdr));
        a0 += Tr::dot(w0, y + kbx, kbx, kqs);
        if constexpr (GU)
            a1 += Tr::dot(w1, y + kbx, kbx, kqs);
    }
    __shared__ float sh[MXM_WARPS - 1][2][MQ_WARP];
    if (threadIdx.y > 0)
    {
        sh[threadIdx.y - 1][0][threadIdx.x] = a0;
        sh[threadIdx.y - 1][1][threadIdx.x] = a1;
    }
    __syncthreads();
    if (threadIdx.y > 0)
        return;
#pragma unroll
    for (int l = 0; l < MXM_WARPS - 1; ++l)
    {
        a0 += sh[l][0][threadIdx.x];
        a1 += sh[l][1][threadIdx.x];
    }
    a0 = mq_warp_sum(a0);
    if constexpr (GU) a1 = mq_warp_sum(a1);
    if (threadIdx.x != 0)
        return;
    if constexpr (GU)
    {
        // same rounding as triton_moe_mxfp4._gu_fused: acc + bias, rounded to fp16, clamp, (u + 1) * g * sigmoid(1.702 g)
        float g = a0 + (btab ? __half2float(reinterpret_cast<const __half*>(btab[e])[row]) : 0.0f);
        float u = a1 + (btab2 ? __half2float(reinterpret_cast<const __half*>(btab2[e])[row]) : 0.0f);
        g = __half2float(__float2half_rn(g));
        u = __half2float(__float2half_rn(u));
        g = fminf(g, lim);
        u = fminf(fmaxf(u, -lim), lim);
        const float a = (u + 1.0f) * g * (1.0f / (1.0f + expf(-1.702f * g)));
        reinterpret_cast<__half*>(outp)[(int64_t)slot * N + row] = __float2half_rn(a);
    }
    else
    {
        if (btab) a0 += __half2float(reinterpret_cast<const __half*>(btab[e])[row]);
        reinterpret_cast<float*>(outp)[(int64_t)slot * N + row] = a0;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Dequantize to fp16 [N, K] (prefill / large-batch path: dequant + GEMM). Port of ggml/src/ggml-cuda/convert.cu dequantize_block_iq*
// (MIT, The ggml authors): one 32-thread block per 256-element super block, 8 outputs per thread (4 + 4 for the 4-bit formats).
// ---------------------------------------------------------------------------------------------------------------------------------
template <int T> __global__ void fq_deq_iq_kernel(const uint8_t* __restrict__ packed, __half* __restrict__ out, int64_t row_bytes, int64_t K, int nb)
{
    const int blk = blockIdx.x;                       // block id in [0, N * nb)
    const int64_t row = blk / nb, bi = blk % nb;
    const uint8_t* base = packed + row * row_bytes;
    __half* y0 = out + row * K + bi * 256;
    const int tid = threadIdx.x;
    const int il = tid / 8, ib = tid % 8;
    if constexpr (T == MQ_IQ2_XXS)
    {
        const block_iq2_xxs* x = (const block_iq2_xxs*)base + bi;
        __half* y = y0 + 32 * ib + 8 * il;
        const uint16_t* q2 = x->qs + 4 * ib;
        const uint8_t* aux8 = (const uint8_t*)q2;
        const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[il]);
        const uint32_t aux32 = q2[2] | (q2[3] << 16);
        const float d = __half2float(x->d) * (0.5f + (aux32 >> 28)) * 0.25f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
        for (int j = 0; j < 8; ++j) y[j] = __float2half_rn(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
    else if constexpr (T == MQ_IQ2_XS)
    {
        const block_iq2_xs* x = (const block_iq2_xs*)base + bi;
        __half* y = y0 + 32 * ib + 8 * il;
        const uint16_t* q2 = x->qs + 4 * ib;
        const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (q2[il] & 511));
        const float d = __half2float(x->d) * (0.5f + ((x->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
        for (int j = 0; j < 8; ++j) y[j] = __float2half_rn(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
    else if constexpr (T == MQ_IQ2_S)
    {
        const block_iq2_s* x = (const block_iq2_s*)base + bi;
        __half* y = y0 + 32 * ib + 8 * il;
        const uint8_t* grid = (const uint8_t*)(iq2s_grid + (x->qs[4 * ib + il] | ((x->qh[ib] << (8 - 2 * il)) & 0x300)));
        const float d = __half2float(x->d) * (0.5f + ((x->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        const uint8_t signs = x->qs[QK_K / 8 + 4 * ib + il];
        for (int j = 0; j < 8; ++j) y[j] = __float2half_rn(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
    else if constexpr (T == MQ_IQ3_XXS)
    {
        const block_iq3_xxs* x = (const block_iq3_xxs*)base + bi;
        __half* y = y0 + 32 * ib + 8 * il;
        const uint8_t* q3 = x->qs + 8 * ib;
        const uint16_t* gas = (const uint16_t*)(x->qs + QK_K / 4) + 2 * ib;
        const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + q3[2 * il + 1]);
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float d = __half2float(x->d) * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
        for (int j = 0; j < 4; ++j)
        {
            y[j + 0] = __float2half_rn(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
            y[j + 4] = __float2half_rn(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
    }
    else if constexpr (T == MQ_IQ3_S)
    {
        const block_iq3_s* x = (const block_iq3_s*)base + bi;
        __half* y = y0 + 32 * ib + 8 * il;
        const uint8_t* qs = x->qs + 8 * ib;
        const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * il + 0] | ((x->qh[ib] << (8 - 2 * il)) & 256)));
        const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * il + 1] | ((x->qh[ib] << (7 - 2 * il)) & 256)));
        const float d = __half2float(x->d) * (1 + 2 * ((x->scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
        const uint8_t signs = x->signs[4 * ib + il];
        for (int j = 0; j < 4; ++j)
        {
            y[j + 0] = __float2half_rn(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
            y[j + 4] = __float2half_rn(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
        }
    }
    else if constexpr (T == MQ_IQ4_NL)
    {
        const block_iq4_nl* x = (const block_iq4_nl*)base + bi * (QK_K / QK4_NL);
        __half* y = y0 + 32 * ib + 4 * il;
        const uint8_t* q4 = x[ib].qs + 4 * il;
        const float d = __half2float(x[ib].d);
        for (int j = 0; j < 4; ++j)
        {
            y[j + 0] = __float2half_rn(d * kvalues_iq4nl[q4[j] & 0xf]);
            y[j + 16] = __float2half_rn(d * kvalues_iq4nl[q4[j] >> 4]);
        }
    }
    else if constexpr (T == MQ_Q3_K)
    {
        // ggml convert.cu dequantize_block_q3_K: thread = 8 values of one 16-value scale group
        const block_q3_K* x = (const block_q3_K*)base + bi;
        const int n = tid / 16, j = (tid / 4) % 4, half_ = (tid / 2) % 2, l8 = (tid % 2) * 8;
        const int is = 8 * n + 2 * j + half_;
        const int sc = (((x->scales[is % 8] >> (4 * (is / 8))) & 0xF) | (((x->scales[8 + is % 4] >> (2 * (is / 4))) & 3) << 4)) - 32;
        const float dl = __half2float(x->d) * sc;
        const int l = 16 * half_ + l8;
        __half* y = y0 + 128 * n + 32 * j + l;
        for (int e = 0; e < 8; ++e)
        {
            const int q = (x->qs[32 * n + l + e] >> (2 * j)) & 3;
            const int hb = (x->hmask[l + e] >> (4 * n + j)) & 1;
            y[e] = __float2half_rn(dl * (float)(q - (hb ? 0 : 4)));
        }
    }
    else  // MQ_IQ4_XS
    {
        const block_iq4_xs* x = (const block_iq4_xs*)base + bi;
        __half* y = y0 + 32 * ib + 4 * il;
        const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
        const float d = __half2float(x->d) * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
        for (int j = 0; j < 4; ++j)
        {
            y[j + 0] = __float2half_rn(d * kvalues_iq4nl[q4[j] & 0xf]);
            y[j + 16] = __float2half_rn(d * kvalues_iq4nl[q4[j] >> 4]);
        }
    }
}

template <int T> void mq_dequant_t(torch::Tensor packed, torch::Tensor out)
{
    using Tr = MqTraits<T>;
    TORCH_CHECK(packed.is_cuda() && out.is_cuda() && packed.is_contiguous() && out.is_contiguous(), "tensors must be contiguous CUDA");
    TORCH_CHECK(out.dtype() == torch::kHalf, "out must be half");
    TORCH_CHECK_DIM(packed, 2);
    TORCH_CHECK_DIM(out, 2);
    const int64_t N = packed.size(0), K = out.size(1);
    TORCH_CHECK(out.size(0) == N && K % 256 == 0 && packed.size(1) == (K / Tr::qk) * Tr::bb, "shape mismatch");
    TORCH_CHECK((uintptr_t)packed.data_ptr() % 4 == 0, "weights must be 4-byte aligned");
    at::cuda::CUDAGuard guard(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int nb = (int)(K / 256);
    fq_deq_iq_kernel<T><<<(unsigned)(N * nb), 32, 0, stream>>>(packed.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                                                                  packed.size(1), K, nb);
}

}  // namespace

void fq_mmvq_iq2_xxs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_IQ2_XXS>(qb, packed, out, bias); }
void fq_mmvq_iq2_xs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_IQ2_XS>(qb, packed, out, bias); }
void fq_mmvq_iq2_s(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_IQ2_S>(qb, packed, out, bias); }
void fq_mmvq_iq3_xxs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_IQ3_XXS>(qb, packed, out, bias); }
void fq_mmvq_iq3_s(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_IQ3_S>(qb, packed, out, bias); }
void fq_mmvq_iq4_nl(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_IQ4_NL>(qb, packed, out, bias); }
void fq_mmvq_iq4_xs(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_IQ4_XS>(qb, packed, out, bias); }
void fq_mmvq_q5_k(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_Q5_K>(qb, packed, out, bias); }
void fq_mmvq_q4_k(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_Q4_K>(qb, packed, out, bias); }
void fq_mmvq_q6_k(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_Q6_K>(qb, packed, out, bias); }
void fq_mmvq_q3_k(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_Q3_K>(qb, packed, out, bias); }
void fq_mmvq_q4_0(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_Q4_0>(qb, packed, out, bias); }
void fq_mmvq_q4_1(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_Q4_1>(qb, packed, out, bias); }
void fq_mmvq_q5_0(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_Q5_0>(qb, packed, out, bias); }
void fq_mmvq_q5_1(torch::Tensor qb, torch::Tensor packed, torch::Tensor out, std::optional<torch::Tensor> bias) { mq_run<MQ_Q5_1>(qb, packed, out, bias); }

// Standalone grouped decode MoE for IQ experts (any of the 7 IQ formats for gate/up and for down; type ids = MQ_* order:
// 0 iq2_xxs, 1 iq2_xs, 2 iq2_s, 3 iq3_xxs, 4 iq3_s, 5 iq4_nl, 6 iq4_xs). Same arguments and results as fq_moe_kq_grouped_mixed_q8:
// h [R, K] half, sel [R * topk] int64, wts [R * topk] half, per-expert base pointers (E int64), strides = bytes per row, partial [R * topk, K] float,
// out [R, K] float.
void fq_moe_mmvq_grouped(torch::Tensor h, torch::Tensor sel, torch::Tensor wts, torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                         torch::Tensor down_ptrs, int64_t gate_stride, int64_t up_stride, int64_t down_stride, int64_t type_ug, int64_t type_down,
                         torch::Tensor partial, torch::Tensor out)
{
    static const int kQk[16] = {256, 256, 256, 256, 256, 32, 256, 256, 0, 0, 0, 0, 0, 0, 0, 32};
    static const int kBb[16] = {66, 74, 82, 98, 110, 18, 136, 176, 0, 0, 0, 0, 0, 0, 0, 17};
    // 15 = MQ_MXFP4 (plain silu(gate) * up, no bias: the gpt-oss swiglu_oai + bias path is fq_mxfp4_moe_gu / fq_mxfp4_moe_down below)
    auto ok_t = [](int64_t t) { return (t >= 0 && t < 8) || t == MQ_MXFP4; };
    TORCH_CHECK(ok_t(type_ug) && ok_t(type_down), "fq_moe_mmvq_grouped: bad format id");
    const int K = (int)h.size(1), R = (int)h.size(0), S = (int)sel.numel();
    TORCH_CHECK(S % R == 0 && R >= 1, "sel must be R x topk");
    TORCH_CHECK(h.dtype() == torch::kHalf && h.is_cuda() && h.is_contiguous(), "h must be contiguous half CUDA");
    const int topk = S / R;
    const int E = (int)up_ptrs.numel();
    const int I = (int)(down_stride / kBb[type_down]) * kQk[type_down];
    TORCH_CHECK(K % kQk[type_ug] == 0 && I % kQk[type_down] == 0 && I % 32 == 0, "K / I must match the format block sizes");
    at::cuda::CUDAGuard guard(h.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    torch::Tensor qb_h = fq_q8_scratch(h.get_device(), R, K);
    fq_quant_q8_1_rows(h, qb_h);
    static thread_local std::map<std::tuple<int, int, int>, torch::Tensor> act_cache;
    auto key = std::make_tuple((int)h.get_device(), S, I);
    auto it = act_cache.find(key);
    torch::Tensor act;
    if (it != act_cache.end())
        act = it->second;
    else
    {
        act = torch::empty({S, I}, torch::dtype(torch::kHalf).device(h.device()));
        act_cache[key] = act;
    }
    const int64_t qb_stride_h = (int64_t)(K / 32) * 36;
    const uint8_t* const* gb = reinterpret_cast<const uint8_t* const*>(gate_ptrs.data_ptr<int64_t>());
    const uint8_t* const* ub = reinterpret_cast<const uint8_t* const*>(up_ptrs.data_ptr<int64_t>());
    const uint8_t* const* db = reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr<int64_t>());
    __half* actp = reinterpret_cast<__half*>(act.data_ptr<at::Half>());
    switch (type_ug)
    {
#define MQ_UG(T) case T: mq_moe_upgate_t<T>(qb_h.data_ptr<uint8_t>(), sel.data_ptr<int64_t>(), gb, ub, gate_stride, up_stride, qb_stride_h, actp, S, I, K, E, topk, stream); break;
        MQ_UG(MQ_IQ2_XXS) MQ_UG(MQ_IQ2_XS) MQ_UG(MQ_IQ2_S) MQ_UG(MQ_IQ3_XXS) MQ_UG(MQ_IQ3_S) MQ_UG(MQ_IQ4_NL) MQ_UG(MQ_IQ4_XS) MQ_UG(MQ_Q5_K) MQ_UG(MQ_MXFP4)
#undef MQ_UG
    }
    torch::Tensor qb_act = fq_q8_scratch(h.get_device(), S, I);
    fq_quant_q8_1_rows(act, qb_act);
    const int64_t qb_stride_a = (int64_t)(I / 32) * 36;
    switch (type_down)
    {
#define MQ_DN(T) case T: mq_moe_down_t<T>(qb_act.data_ptr<uint8_t>(), qb_stride_a, sel.data_ptr<int64_t>(), db, down_stride, partial.data_ptr<float>(), S, I, K, E, stream); break;
        MQ_DN(MQ_IQ2_XXS) MQ_DN(MQ_IQ2_XS) MQ_DN(MQ_IQ2_S) MQ_DN(MQ_IQ3_XXS) MQ_DN(MQ_IQ3_S) MQ_DN(MQ_IQ4_NL) MQ_DN(MQ_IQ4_XS) MQ_DN(MQ_Q5_K) MQ_DN(MQ_MXFP4)
#undef MQ_DN
    }
    fq_mmvq_moe_combine_kernel<<<dim3((K + 255) / 256, R), 256, 0, stream>>>(partial.data_ptr<float>(), reinterpret_cast<const __half*>(wts.data_ptr<at::Half>()),
                                                                                out.data_ptr<float>(), topk, K);
}

// EXL3_FQ_MMVQ_MXFP4 decode MoE (R == 1). qb: fq_quant_q8_1_rows of the input. Pointer tables = int64 data_ptr of each expert's packed rows
// [rows, K/32*17] / fp16 bias [rows] (as triton_moe_mxfp4); sel int32 [S]; bias tables may be empty (numel 0) = no bias.
// gu: act [S, I] fp16 = swiglu_oai(gate(x) + bg, up(x) + bu, lim) with fp16 rounding of both before the activation; qb is ONE row (all slots).
void fq_mxfp4_moe_gu(torch::Tensor qb, torch::Tensor sel, torch::Tensor gate_ptrs, torch::Tensor gate_bias, torch::Tensor up_ptrs,
                     torch::Tensor up_bias, torch::Tensor act, int64_t I, int64_t K, double lim)
{
    TORCH_CHECK(sel.dtype() == torch::kInt && qb.dtype() == torch::kUInt8 && act.dtype() == torch::kHalf && act.is_contiguous() && K % 32 == 0,
                "fq_mxfp4_moe_gu: bad arguments");
    TORCH_CHECK(qb.numel() >= (K / 32) * 36 && act.numel() >= sel.numel() * I, "fq_mxfp4_moe_gu: sizes");
    at::cuda::CUDAGuard guard(qb.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid((unsigned)I, (unsigned)sel.numel(), 1), block(MQ_WARP, MXM_WARPS, 1);
    fq_mxfp4_moe_kernel<true><<<grid, block, 0, stream>>>(
        qb.data_ptr<uint8_t>(), 0, sel.data_ptr<int>(), gate_ptrs.data_ptr<int64_t>(), gate_bias.numel() ? gate_bias.data_ptr<int64_t>() : nullptr,
        up_ptrs.data_ptr<int64_t>(), up_bias.numel() ? up_bias.data_ptr<int64_t>() : nullptr, act.data_ptr(), (int)I, (int)K,
        (int)gate_ptrs.numel(), (float)lim);
}

// down: out [S, N] fp32 = act_q8[slot] . W_down[sel[slot]]^T + bias; qb = S rows (one per slot) of K/32*36 bytes.
void fq_mxfp4_moe_down(torch::Tensor qb, torch::Tensor sel, torch::Tensor down_ptrs, torch::Tensor down_bias, torch::Tensor out, int64_t N,
                       int64_t K)
{
    TORCH_CHECK(sel.dtype() == torch::kInt && qb.dtype() == torch::kUInt8 && out.dtype() == torch::kFloat && out.is_contiguous() && K % 32 == 0,
                "fq_mxfp4_moe_down: bad arguments");
    TORCH_CHECK(qb.numel() >= sel.numel() * (K / 32) * 36 && out.numel() >= sel.numel() * N, "fq_mxfp4_moe_down: sizes");
    at::cuda::CUDAGuard guard(qb.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid((unsigned)N, (unsigned)sel.numel(), 1), block(MQ_WARP, MXM_WARPS, 1);
    fq_mxfp4_moe_kernel<false><<<grid, block, 0, stream>>>(
        qb.data_ptr<uint8_t>(), (K / 32) * 36, sel.data_ptr<int>(), down_ptrs.data_ptr<int64_t>(),
        down_bias.numel() ? down_bias.data_ptr<int64_t>() : nullptr, nullptr, nullptr, out.data_ptr(), (int)N, (int)K, (int)down_ptrs.numel(), 0.0f);
}

// out [N, K] half from raw GGUF bytes; fmt = MQ type id (0 iq2_xxs, 1 iq2_xs, 2 iq2_s, 3 iq3_xxs, 4 iq3_s, 5 iq4_nl, 6 iq4_xs), K % 256 == 0
void fq_dequant_iq(torch::Tensor packed, torch::Tensor out, int64_t fmt)
{
    switch (fmt)
    {
        case MQ_IQ2_XXS: mq_dequant_t<MQ_IQ2_XXS>(packed, out); break;
        case MQ_IQ2_XS: mq_dequant_t<MQ_IQ2_XS>(packed, out); break;
        case MQ_IQ2_S: mq_dequant_t<MQ_IQ2_S>(packed, out); break;
        case MQ_IQ3_XXS: mq_dequant_t<MQ_IQ3_XXS>(packed, out); break;
        case MQ_IQ3_S: mq_dequant_t<MQ_IQ3_S>(packed, out); break;
        case MQ_IQ4_NL: mq_dequant_t<MQ_IQ4_NL>(packed, out); break;
        case MQ_IQ4_XS: mq_dequant_t<MQ_IQ4_XS>(packed, out); break;
        case MQ_Q3_K: mq_dequant_t<MQ_Q3_K>(packed, out); break;
        default: TORCH_CHECK(false, "fq_dequant_iq: bad format id ", fmt);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// EXL3_MOE_FOLD_FUSED: out[n] = sum_s (d[s, n] * float(rw[s])) in ONE launch, replacing routing_weights.float(), the broadcast multiply,
// .sum(1) and the final .to(out_dtype). Order = torch's CUDA reduce for a size-S non-fastest dim (Reduce.cuh thread_reduce_impl, no warp split:
// ((0+v0)+v1)+v2)+v3, one rounded fp32 multiply then one rounded add, no FMA), so the fp32 result is bit-identical to the torch chain.
// ---------------------------------------------------------------------------------------------------------------------------------
template <typename RW> __device__ __forceinline__ float fold_to_f(RW v);
template <> __device__ __forceinline__ float fold_to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float fold_to_f<__half>(__half v) { return __half2float(v); }

template <typename RW, typename OUT>
__global__ void fq_moe_fold_kernel(const float* __restrict__ d, const RW* __restrict__ rw, OUT* __restrict__ out, int S, int64_t N, int H)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= H) return;
    float acc = 0.0f;
    for (int s = 0; s < S; ++s)
        acc = __fadd_rn(acc, __fmul_rn(d[(int64_t)s * N + n], fold_to_f<RW>(rw[s])));
    mq_store<OUT>(&out[n], acc);
}

void fq_moe_fold(torch::Tensor d, torch::Tensor rw, torch::Tensor out)
{
    TORCH_CHECK(d.is_cuda() && rw.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(d.is_contiguous() && rw.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(d.dtype() == torch::kFloat && d.dim() == 2, "d must be float [S, N]");
    TORCH_CHECK(rw.dtype() == torch::kHalf || rw.dtype() == torch::kFloat, "rw must be half or float");
    TORCH_CHECK(out.dtype() == torch::kHalf || out.dtype() == torch::kFloat, "out must be half or float");
    const int S = (int)d.size(0);
    const int64_t N = d.size(1);
    const int H = (int)out.numel();
    TORCH_CHECK(rw.numel() == S && H <= N, "shape mismatch");
    at::cuda::CUDAGuard guard(d.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int blocks = (H + 127) / 128;
    const float* dp = d.data_ptr<float>();
    if (rw.dtype() == torch::kHalf)
    {
        const __half* r = reinterpret_cast<const __half*>(rw.data_ptr<at::Half>());
        if (out.dtype() == torch::kHalf) fq_moe_fold_kernel<__half, __half><<<blocks, 128, 0, stream>>>(dp, r, reinterpret_cast<__half*>(out.data_ptr<at::Half>()), S, N, H);
        else fq_moe_fold_kernel<__half, float><<<blocks, 128, 0, stream>>>(dp, r, out.data_ptr<float>(), S, N, H);
    }
    else
    {
        const float* r = rw.data_ptr<float>();
        if (out.dtype() == torch::kHalf) fq_moe_fold_kernel<float, __half><<<blocks, 128, 0, stream>>>(dp, r, reinterpret_cast<__half*>(out.data_ptr<at::Half>()), S, N, H);
        else fq_moe_fold_kernel<float, float><<<blocks, 128, 0, stream>>>(dp, r, out.data_ptr<float>(), S, N, H);
    }
}
