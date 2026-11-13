// q8_1-activation decode kernels for K-quant + IQ4_XS weights
// port of the llama.cpp mmvq vmmq arms (ggml/src/ggml-cuda/vecdotq.cuh).
// Activations are quantized ONCE per row to q8_1 blocks by the shared
// fq_quant_q8_1_rows (fq_gemv_q8.cu) into the caller-cached global scratch,
// then row-per-block dots (fq_dot_q4_k_q8 / fq_dot_q5_k_q8 / fq_dot_q6_k_q8 /
// fq_dot_iq4_xs_q8 in fq_gemv.cuh) run INTEGER dp4a dots.
//
// Same portability contract as fq_gemv_q8.cu (no PTX, no warp shuffles,
// shared-memory reduction, fq_dp4a + fq_vsub4 builtins with scalar
// fallbacks off-CUDA).
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
#include "fq_kq_item.cuh"
#include "fq_api.h"

namespace {

// Row-per-block KQ q8 dot: block bidx computes output row n of x-row m
// against the packed K-quant matrix. Threads take sub-chunks round-robin;
// partials reduce sequentially (portable, same grid as the IQ2/NVFP4/Q8 row
// kernels in fq_gemv_q8.cu). FMT selects the dot (0 = Q4_K, 1 = Q5_K,
// 2 = Q6_K, 3 = IQ4_XS).
template <typename OUT, int FMT>
__global__ void fq_gemv_kq_kernel(const uint8_t* __restrict__ aqb,
                                  const uint8_t* __restrict__ packed,
                                  OUT* __restrict__ out,
                                  int M, int N, int K, int64_t packed_stride,
                                  int64_t qb_stride)
{
    int bidx = blockIdx.x;
    int m = bidx / N;
    int n = bidx % N;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    extern __shared__ float red[];

    float acc = 0.0f;
    if (n < N)
    {
        const uint8_t* wr = packed + (int64_t)n * packed_stride;
        const uint8_t* aq = aqb + (int64_t)m * qb_stride;
        if constexpr (FMT == 0)
            acc = fq_dot_q4_k_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 1)
            acc = fq_dot_q5_k_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 2)
            acc = fq_dot_q6_k_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 3)
            acc = fq_dot_iq4_xs_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 4)
            acc = fq_dot_iq3_xxs_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 5)
            acc = fq_dot_iq2_s_q8(wr, aq, K, tid, nt);
        else
            acc = fq_dot_iq3_s_q8(wr, aq, K, tid, nt);
    }
    red[tid] = acc;
    __syncthreads();
    if (n < N && tid == 0)
    {
        float s = 0.0f;
        for (int l = 0; l < nt; l++)
            s = __fadd_rn(s, red[l]);
        if constexpr (std::is_same<OUT, __half>::value)
            out[(int64_t)m * N + n] = __float2half_rn(s);
        else
            out[(int64_t)m * N + n] = s;
    }
}

// ---------------------------------------------------------------------------
// Q4_K / Q6_K decode GEMV v2 ("rows" kernel): same float result bit for bit as
// fq_gemv_kq_kernel above, but built for memory-level parallelism.
//
// What the v1 kernel above does wrong on a GPU with a deep memory pipeline (RTX 3090: Q4_K ran at ~36 % of
// the DRAM roof, llama.cpp mmvq at ~70 %):
//  - its weight loads are memcpy() from a uint8_t*, which the compiler must treat as byte-aligned: LDG.U8
//    chains instead of one 16-byte load per thread;
//  - one 2.3 KB row per 256-thread block, and thread 0 then adds the 256 partials one after the other
//    (about 1 us of dependent adds), so a block spends half of its life with no load in flight.
//
// v2: an "item" is the work one thread does with ONE aligned wide load pair (Q4_K: two 16-byte loads = 4 of
// the v1 chunks; Q6_K: 4 chunks from u16 pairs, the block is 210 bytes so only 2-byte aligned, like ggml
// get_int_b2). Items of ROWS output rows go to shared memory as per-chunk float values; then ROWS threads
// (one per row) add them in EXACTLY the order v1 does (per-lane partial c = l, l + nt, ..., then lane by lane),
// so every output is bit-identical to v1 (the integer dp4a parts are exact; only the float order matters and
// it is kept). The serial add chain is now shared by ROWS rows per block and overlaps with the loads of the
// other resident blocks.
//
// Dot structure after ggml-cuda mmvq vec_dot_q4_K_q8_1 / vec_dot_q6_K_q8_1 (llama.cpp ggml/src/ggml-cuda/
// vecdotq.cuh, MIT, Copyright (c) 2023-2024 The ggml authors). Portable like the rest of fq_*: no PTX, no
// shuffles (HIP builds the same source; fq_dp4a is sudot4 there).
// ---------------------------------------------------------------------------

// fq_h2f_lo, fq_ld_u16x2, fq_kq_item_q4_k, fq_kq_item_q6_k: fq_kq_item.cuh (shared with fq_moe_kq.cu)

// One block = ROWS consecutive output rows of x-row m = blockIdx.x / nbn, 64 threads per row. Thread j of a row owns
// the items j, j + 64, j + 128, ... (one 256-chunk window of v1 per step); those items are chunks whose v1 lane
// (c mod nt, nt = 256 when K >= 4096 / 2048) is fixed, so thread j keeps four v1 lane partials in registers, adding
// the windows in increasing order exactly as v1's per-thread loop does. The nt lane partials go to shared memory
// (row stride 260 floats: the ROWS chain threads hit different banks), then thread r < ROWS adds row r lane by lane
// in v1's order. Lanes >= nsub (K small, nt padded to a multiple of 32) are written as 0.0f like v1's idle threads.
template <typename OUT, int FMT, int ROWS>
__global__ void __launch_bounds__(64 * ROWS) fq_gemv_kq_rows_kernel(const uint8_t* __restrict__ aqb,
                                                                    const uint8_t* __restrict__ packed,
                                                                    OUT* __restrict__ out,
                                                                    int N, int K, int64_t packed_stride, int64_t qb_stride,
                                                                    int nbn, int nt)
{
    constexpr int PS = 260;
    __shared__ __align__(16) float part[ROWS * PS];
    constexpr int IPB = (FMT == 0) ? 4 : 8;                    // items per 256-block
    const int tid = threadIdx.x;
    const int r = tid >> 6, j = tid & 63;
    const int m = blockIdx.x / nbn;
    const int n0 = (blockIdx.x - m * nbn) * ROWS;
    const int n = n0 + r;
    const int nitems = (K / 256) * IPB;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (n < N)
    {
        const uint8_t* aq = aqb + (int64_t)m * qb_stride;
        const uint8_t* wr = packed + (int64_t)n * packed_stride;
        for (int it = j; it < nitems; it += 64)
        {
            float t[4];
            if constexpr (FMT == 0)
                fq_kq_item_q4_k(wr, aq, it, t);
            else
                fq_kq_item_q6_k(wr, aq, it, t);
            #pragma unroll
            for (int q = 0; q < 4; q++)
                acc[q] = __fadd_rn(acc[q], t[q]);
        }
    }
    #pragma unroll
    for (int q = 0; q < 4; q++)
    {
        const int lane = (FMT == 0) ? (4 * j + q) : (32 * (j >> 3) + (j & 7) + 8 * q);
        if (lane < nt)
            part[r * PS + lane] = acc[q];
    }
    __syncthreads();
    if (tid < ROWS && n0 + tid < N)
    {
        const float4* pv = reinterpret_cast<const float4*>(part + tid * PS);
        float s = 0.0f;
        #pragma unroll 8
        for (int l = 0; l < nt / 4; l++)
        {
            const float4 v = pv[l];
            s = __fadd_rn(s, v.x);
            s = __fadd_rn(s, v.y);
            s = __fadd_rn(s, v.z);
            s = __fadd_rn(s, v.w);
        }
        if constexpr (std::is_same<OUT, __half>::value)
            out[(int64_t)m * N + n0 + tid] = __float2half_rn(s);
        else
            out[(int64_t)m * N + n0 + tid] = s;
    }
}

}  // namespace

// BOLT_FQ_KQ_ROWS: output rows per block of the v2 Q4_K / Q6_K kernel (1, 2, 4). 0 = the v1 kernel (A/B); unset = the per-format
// default measured on the RTX 3090 (Q4_K 4 rows, Q6_K 1 row: its item has ~2x the load instructions, 64-thread blocks keep more
// of them resident: down 4096x12288 Q6_K 85 % of the DRAM roof at 1 row vs 63 % at 4). HIP default 1 row (Strix Halo, Qwen3-8B
// Q4_K_M served N1 v1 36.9 -> 1 row 39.1 / 4 rows 39.3 tok/s, same output; 35B-A3B unchanged: 53.3 / 53.3 / 53.1).
static int fq_kq_rows_env(int fmt)
{
    static const char* e = std::getenv("BOLT_FQ_KQ_ROWS");
    static const int forced = e ? std::atoi(e) : -1;
    if (forced >= 0)
        return forced;
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
    return 1;
#else
    return fmt == 0 ? 4 : 1;
#endif
}

template <typename OUT, int FMT>
static void fq_gemv_kq_rows_dispatch(int rows, int M, const uint8_t* qb, const uint8_t* packed, OUT* out, int N, int K,
                                     int64_t pstride, int64_t qb_stride, int nt, cudaStream_t stream)
{
    auto run = [&](auto rows_c)
    {
        constexpr int R = decltype(rows_c)::value;
        const int nbn = (N + R - 1) / R;
        fq_gemv_kq_rows_kernel<OUT, FMT, R><<<(unsigned)((int64_t)M * nbn), 64 * R, 0, stream>>>(
            qb, packed, out, N, K, pstride, qb_stride, nbn, nt);
    };
    if (rows >= 4) run(std::integral_constant<int, 4>());
    else if (rows >= 2) run(std::integral_constant<int, 2>());
    else run(std::integral_constant<int, 1>());
}

// Shared row-per-block q8 launch sizing: threads take sub-chunks
// round-robin (same as fq_q8_nt in fq_gemv_q8.cu, generalized).
static inline int fq_kq_nt(int K, int sub)
{
    int nsub = K / sub;
    return nsub >= 256 ? 256 : ((nsub + 31) / 32) * 32;
}

template <int FMT, int SUB>
static void fq_gemv_kq_launch(torch::Tensor qb, torch::Tensor packed, torch::Tensor out,
                              int64_t block_bytes)
{
    TORCH_CHECK(qb.is_cuda() && packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(qb.is_contiguous() && packed.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(out.dtype() == torch::kHalf || out.dtype() == torch::kFloat,
                "out must be half or float");
    TORCH_CHECK_DIM(qb, 2);
    TORCH_CHECK_DIM(packed, 2);
    TORCH_CHECK_DIM(out, 2);
    int M = (int)qb.size(0);
    int N = (int)packed.size(0);
    TORCH_CHECK((int)out.size(0) == M, "out rows must match qb rows");
    TORCH_CHECK((int)out.size(1) == N, "out cols must match N");
    int K = (int)qb.size(1) / FQ_Q8_1_BYTES * FQ_Q8_1_GROUP;
    TORCH_CHECK(K % SUB == 0, "K must be a multiple of the dot sub-chunk");
    TORCH_CHECK((int)packed.size(1) == (K / FQ_Q4_K_GROUP) * block_bytes, "packed width must match K");
    int64_t pstride = (int64_t)(K / FQ_Q4_K_GROUP) * block_bytes;
    int64_t qb_stride = (int64_t)(K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
    int nt = fq_kq_nt(K, SUB);
    size_t shmem = (size_t)nt * sizeof(float);
    at::cuda::CUDAGuard guard(qb.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if constexpr (FMT == 0 || FMT == 2)
    {
        // v2 rows kernel (bit-identical to the v1 kernel below): 16-byte aligned Q4_K rows, 2-byte aligned Q6_K rows
        const int rows = fq_kq_rows_env(FMT);
        const uintptr_t align = (FMT == 0) ? 16 : 2;
        if (rows > 0 && K % 256 == 0 && (uintptr_t)packed.data_ptr() % align == 0 && (uintptr_t)qb.data_ptr() % 4 == 0 &&
            pstride % align == 0)
        {
            auto go = [&](auto* op)
            {
                fq_gemv_kq_rows_dispatch<std::decay_t<decltype(*op)>, FMT>(
                    rows, M, qb.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(), op, N, K, pstride, qb_stride, nt, stream);
            };
            if (out.dtype() == torch::kHalf)
                go(reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
            else
                go(out.data_ptr<float>());
            return;
        }
    }
    auto launch = [&](auto* op)
    {
        fq_gemv_kq_kernel<std::decay_t<decltype(*op)>, FMT> <<<M * N, nt, shmem, stream>>>(
            qb.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(),
            op, M, N, K, pstride, qb_stride);
    };
    if (out.dtype() == torch::kHalf)
        launch(reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
    else
        launch(out.data_ptr<float>());
}

void fq_gemv_q4_k_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<0, 16>(qb, packed, out, FQ_Q4_K_BYTES);
}

void fq_gemv_q5_k_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<1, 16>(qb, packed, out, FQ_Q5_K_BYTES);
}

void fq_gemv_q6_k_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<2, 8>(qb, packed, out, FQ_Q6_K_BYTES);
}

void fq_gemv_iq4_xs_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<3, 32>(qb, packed, out, FQ_IQ4_XS_BYTES);
}

void fq_gemv_iq3_xxs_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<4, 32>(qb, packed, out, FQ_IQ3_XXS_BYTES);
}

void fq_gemv_iq2_s_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<5, 32>(qb, packed, out, FQ_IQ2_S_BYTES);
}

void fq_gemv_iq3_s_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<6, 32>(qb, packed, out, FQ_IQ3_S_BYTES);
}
