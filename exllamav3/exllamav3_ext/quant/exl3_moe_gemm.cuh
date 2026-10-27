#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"

// Grouped prefill GEMM for the exl3 MoE tier: C[e rows] = A[e rows] @ W_e for every expert e of
// [count_lo, count_hi] rows, A / C addressed by the expert-sorted row position. One CTA owns one
// (expert, BM-row tile, 128-column tile) output block and runs the whole K loop itself (no split-K
// across CTAs, no group barriers): 8 warps, each decoding one 16x16 trellis block per k16 and feeding
// it to BM/16 row fragments, fp32 accumulation, cp.async multi-stage pipeline. Same B decode
// (dq_dispatch) and A fragment layout as exl3_gemm_kernel_inner. Output is rounded to fp16 once.
#define MOEG_THREADS 256
#define MOEG_BN 128

template<int BITS, int BM>
constexpr int moeg_stage_bytes() { return BM * 64 + 512 * BITS; }

template<int BITS, int BM, int STAGES, int MINB>
__global__ __launch_bounds__(MOEG_THREADS, MINB)
void exl3_moe_gemm_kernel
(
    const half* __restrict__ A0, const half* __restrict__ A1,
    half* __restrict__ C0, half* __restrict__ C1,
    const uint16_t** __restrict__ B0, const uint16_t** __restrict__ B1,
    const int64_t* __restrict__ counts,     // [E]
    const int64_t* __restrict__ starts,     // [E] first sorted row of expert
    const int num_experts,
    const int K, const int N,
    const int count_lo, const int count_hi
)
{
    const int z = blockIdx.z;
    const int which = z >= num_experts;
    const int e = which ? z - num_experts : z;
    const int cnt = (int) counts[e];
    if (cnt < count_lo || cnt > count_hi) return;
    const int m0 = blockIdx.y * BM;
    if (m0 >= cnt) return;
    const int rows = min(BM, cnt - m0);
    const int n0 = blockIdx.x * MOEG_BN;

    const half* A = (which ? A1 : A0) + (size_t) (starts[e] + m0) * K;
    half* C = (which ? C1 : C0) + (size_t) (starts[e] + m0) * N + n0;
    const uint16_t* B = (which ? B1 : B0)[e];

    extern __shared__ __align__(16) uint8_t moeg_smem[];
    constexpr int A_BYTES = BM * 64;
    constexpr int STAGE_BYTES = moeg_stage_bytes<BITS, BM>();

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int KT = K / 32;
    const int NBf = N / 16;           // 16-col blocks across the full matrix
    const int nb0 = n0 / 16;

    // per-thread copy descriptors
    constexpr int A_ITERS = BM * 4 / MOEG_THREADS > 0 ? BM * 4 / MOEG_THREADS : 1;
    constexpr bool A_PART = BM * 4 < MOEG_THREADS;
    const half* a_src[A_ITERS];
    uint32_t a_dst[A_ITERS];
    bool a_ok[A_ITERS];
    #pragma unroll
    for (int i = 0; i < A_ITERS; ++i)
    {
        int c = tid + i * MOEG_THREADS;
        int row = c >> 2, col = c & 3;
        int colsw = col ^ ((row >> 1) & 3);
        a_ok[i] = (!A_PART || c < BM * 4) && row < rows;
        a_src[i] = A + (size_t) row * K + col * 8;
        a_dst[i] = row * 64 + colsw * 16;
    }
    constexpr int B_CHUNKS = 32 * BITS;            // 16B chunks per stage (2 k16 rows)
    const bool b_ok = tid < B_CHUNKS;
    const int b_kr = tid / (16 * BITS), b_j = tid % (16 * BITS);
    const uint16_t* b_src = B + ((size_t) b_kr * NBf + nb0) * (16 * BITS) + b_j * 8;
    const size_t b_step = (size_t) 2 * NBf * (16 * BITS);   // per k32 tile, in u16
    const uint32_t b_dst = A_BYTES + b_kr * (8 * 16 * BITS * 2) + b_j * 16;

    const uint32_t smem_base = (uint32_t) __cvta_generic_to_shared(moeg_smem);

    auto load_stage = [&] (int kt, int st)
    {
        uint32_t sb = smem_base + st * STAGE_BYTES;
        #pragma unroll
        for (int i = 0; i < A_ITERS; ++i)
            if (a_ok[i])
            {
                const half* src = a_src[i] + kt * 32;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + a_dst[i]), "l"(src));
            }
        if (b_ok)
        {
            const uint16_t* src = b_src + (size_t) kt * b_step;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + b_dst), "l"(src));
        }
    };

    FragC acc[BM / 16][2];
    #pragma unroll
    for (int m = 0; m < BM / 16; ++m)
        #pragma unroll
        for (int n = 0; n < 2; ++n) acc[m][n] = {};

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < KT) load_stage(s, s);
        cp_async_fence();
    }

    // ldmatrix lane addressing (same as exl3_gemm_kernel_inner)
    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int lc = lane >> 4;

    for (int kt = 0; kt < KT; ++kt)
    {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KT) load_stage(nk, nk % STAGES);
            cp_async_fence();
        }
        const int st = kt % STAGES;
        const uint32_t sa = smem_base + st * STAGE_BYTES;
        const uint16_t* sb16 = (const uint16_t*) (moeg_smem + st * STAGE_BYTES + A_BYTES);
        #pragma unroll
        for (int s = 0; s < 2; ++s)
        {
            FragB fb0, fb1;
            const uint32_t* shb = (const uint32_t*) (sb16 + (s * 8 + warp) * (16 * BITS));
            dq_dispatch<BITS, 2>(shb, lane << 3, fb0, fb1);
            FragA fa[BM / 16];
            #pragma unroll
            for (int m = 0; m < BM / 16; ++m)
            {
                int R = m * 16 + lr;
                int c = (lc + s * 2) ^ ((R >> 1) & 3);
                uint32_t addr = sa + R * 64 + c * 16;
                uint32_t* a = reinterpret_cast<uint32_t*>(&fa[m]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
            }
            #pragma unroll
            for (int m = 0; m < BM / 16; ++m)
            {
                ptx_mma_m16n8k16(fa[m], fb0, acc[m][0]);
                ptx_mma_m16n8k16(fa[m], fb1, acc[m][1]);
            }
        }
    }
    cp_async_wait<0>();

    // epilogue: fp32 -> fp16, direct from fragments
    #pragma unroll
    for (int m = 0; m < BM / 16; ++m)
    {
        const int r0 = m * 16 + (lane >> 2), r1 = r0 + 8;
        #pragma unroll
        for (int n = 0; n < 2; ++n)
        {
            const int col = warp * 16 + n * 8 + (lane & 3) * 2;
            if (r0 < rows) *(half2*) (C + (size_t) r0 * N + col) = __floats2half2_rn(acc[m][n][0], acc[m][n][1]);
            if (r1 < rows) *(half2*) (C + (size_t) r1 * N + col) = __floats2half2_rn(acc[m][n][2], acc[m][n][3]);
        }
    }
}

template<int BITS, int BM, int STAGES, int MINB>
static void exl3_moe_gemm_launch
(
    const half* A0, const half* A1, half* C0, half* C1,
    const uint16_t** B0, const uint16_t** B1,
    const int64_t* counts, const int64_t* starts, int num_experts, int nmat,
    int K, int N, int count_lo, int count_hi, int max_rows, cudaStream_t stream
)
{
    constexpr int smem = STAGES * moeg_stage_bytes<BITS, BM>();
    auto kern = exl3_moe_gemm_kernel<BITS, BM, STAGES, MINB>;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    dim3 grid(N / MOEG_BN, (max_rows + BM - 1) / BM, num_experts * nmat);
    kern<<<grid, MOEG_THREADS, smem, stream>>>(A0, A1, C0, C1, B0, B1, counts, starts, num_experts, K, N, count_lo, count_hi);
}
