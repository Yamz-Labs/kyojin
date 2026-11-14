// FP8 (E4M3, stored as is) decode GEMV in C++ (CUDA and HIP, wave32 on RDNA). Replaces the Triton _gemv_fp8 for 1..16 activation rows:
// the Triton tile read 64 rows x 128 B per iteration and streamed DRAM at 42-55 GB/s on gfx1151 once the weights fall out of the 32 MB
// MALL (every served step). Here one warp owns one output column n and walks its K bytes in 16-byte lanes (512 contiguous bytes per warp
// iteration), so a weight row is one long burst. Each weight chunk is decoded once and dotted against up to FQ_FP8_MR activation rows.
// E4M3 -> fp16 by the same bit move as quant/fp8.py (sign | exponent-mantissa << 7, x 256), fp32 accumulation; scales per output row
// (s[N], applied at the end) or per [bo x bi] block (s[ceil(N/bo), s_cols], bi % 16 == 0, applied per 16-byte chunk) as the Triton kernel.
#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_api.h"

#if defined(USE_ROCM)
#define __shfl_xor_sync(mask, var, ...) __shfl_xor(var, __VA_ARGS__)
#endif

#define FQ_FP8_WARP 32
#define FQ_FP8_WARPS 4
#define FQ_FP8_MR 8

__device__ __forceinline__ float fq_e4m3_f(uint32_t b)
{
    uint16_t h = (uint16_t)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    return __half2float(__ushort_as_half(h)) * 256.0f;
}

template <int MR>
__global__ void fq_gemv_fp8_kernel(const __half* __restrict__ x, const uint8_t* __restrict__ q, const float* __restrict__ s,
                                   float* __restrict__ out, int M, int N, int K, int s_cols, int bo, int bi, bool grid)
{
    int lane = threadIdx.x % FQ_FP8_WARP;
    int n = blockIdx.x * FQ_FP8_WARPS + threadIdx.x / FQ_FP8_WARP;
    int m0 = blockIdx.y * MR;
    int mc = min(MR, M - m0);
    if (n >= N) return;                                   // whole warp leaves together (no block-wide sync below)

    const uint8_t* wr = q + (int64_t)n * K;
    const float* srow = grid ? s + (int64_t)(n / bo) * s_cols : nullptr;
    float acc[MR];
    #pragma unroll
    for (int r = 0; r < MR; r++) acc[r] = 0.0f;

    for (int k = lane * 16; k < K; k += FQ_FP8_WARP * 16)
    {
        uint4 w = *reinterpret_cast<const uint4*>(wr + k);
        float wf[16];
        uint32_t wv[4] = {w.x, w.y, w.z, w.w};
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            #pragma unroll
            for (int b = 0; b < 4; b++) wf[j * 4 + b] = fq_e4m3_f((wv[j] >> (8 * b)) & 0xFFu);
        }
        float sc = grid ? srow[k / bi] : 1.0f;
        #pragma unroll
        for (int r = 0; r < MR; r++)
        {
            if (r < mc)
            {
                const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)(m0 + r) * K + k);
                uint4 xa = xr[0], xb = xr[1];
                const __half2* h2a = reinterpret_cast<const __half2*>(&xa);
                const __half2* h2b = reinterpret_cast<const __half2*>(&xb);
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    float2 fa = __half22float2(h2a[j]);
                    float2 fb = __half22float2(h2b[j]);
                    sum = fmaf(wf[2 * j], fa.x, sum);
                    sum = fmaf(wf[2 * j + 1], fa.y, sum);
                    sum = fmaf(wf[8 + 2 * j], fb.x, sum);
                    sum = fmaf(wf[8 + 2 * j + 1], fb.y, sum);
                }
                acc[r] = fmaf(sum, sc, acc[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < MR; r++)
    {
        float v = acc[r];
        #pragma unroll
        for (int o = FQ_FP8_WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, FQ_FP8_WARP);
        if (lane == 0 && r < mc) out[(int64_t)(m0 + r) * N + n] = grid ? v : v * s[n];
    }
}

void fq_gemv_fp8(torch::Tensor x, torch::Tensor q, torch::Tensor s, torch::Tensor out, int64_t s_cols, int64_t bo, int64_t bi, bool grid)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    TORCH_CHECK(x.dtype() == torch::kHalf && x.dim() == 2 && x.is_contiguous(), "fq_gemv_fp8: x must be contiguous half [M, K]");
    TORCH_CHECK(q.dtype() == torch::kUInt8 && q.dim() == 2 && q.is_contiguous(), "fq_gemv_fp8: q must be contiguous uint8 [N, K]");
    TORCH_CHECK(s.dtype() == torch::kFloat && s.is_contiguous(), "fq_gemv_fp8: s must be contiguous float32");
    TORCH_CHECK(out.dtype() == torch::kFloat && out.is_contiguous(), "fq_gemv_fp8: out must be contiguous float32 [M, N]");
    int M = x.size(0), K = x.size(1), N = q.size(0);
    TORCH_CHECK(q.size(1) == K && out.size(0) == M && out.size(1) == N, "fq_gemv_fp8: shape mismatch");
    TORCH_CHECK(K % 16 == 0 && (!grid || bi % 16 == 0), "fq_gemv_fp8: K and the scale block width must be multiples of 16");
    TORCH_CHECK(((uintptr_t)x.data_ptr() % 16) == 0 && ((uintptr_t)q.data_ptr() % 16) == 0, "fq_gemv_fp8: x and q must be 16-byte aligned");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
    dim3 block(FQ_FP8_WARP * FQ_FP8_WARPS);
    if (M == 1)
    {
        dim3 g((N + FQ_FP8_WARPS - 1) / FQ_FP8_WARPS, 1);
        fq_gemv_fp8_kernel<1><<<g, block, 0, stream>>>(xp, q.data_ptr<uint8_t>(), s.data_ptr<float>(), out.data_ptr<float>(),
                                                       M, N, K, (int)s_cols, (int)bo, (int)bi, grid);
    }
    else
    {
        dim3 g((N + FQ_FP8_WARPS - 1) / FQ_FP8_WARPS, (M + FQ_FP8_MR - 1) / FQ_FP8_MR);
        fq_gemv_fp8_kernel<FQ_FP8_MR><<<g, block, 0, stream>>>(xp, q.data_ptr<uint8_t>(), s.data_ptr<float>(), out.data_ptr<float>(),
                                                               M, N, K, (int)s_cols, (int)bo, (int)bi, grid);
    }
}
