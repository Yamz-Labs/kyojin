// Fused dequant-GEMV/GEMM for foreign-quant weights (decode path, M = 1..16).
// out[m, n] = sum_k x[m, k] * W[n, k], fp32 accumulate, fp16 out. Weights are
// dequanted on the fly with the bit-exact fq_*_block() arithmetic, so versus
// dequant()+matmul only the accumulation order differs (fp16 tolerance).
// Portable CUDA/HIP: one thread per (row, out-col), sequential K loop, no
// cross-lane traffic, no wave-size assumptions.
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_dequant.cuh"
#include "fq_api.h"

namespace {

// W layout [N, Kb, B] with group G (Q8_0: G=32/B=34, IQ2_XXS: G=256/B=66).
template <int G, int B>
__global__ void fq_gemm_kernel(const __half* __restrict__ x, const uint8_t* __restrict__ packed,
                               __half* __restrict__ out, int M, int N, int K)
{
    int64_t total = (int64_t)M * N;
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (int64_t)gridDim.x * blockDim.x)
    {
        int n = (int)(t % N);
        int m = (int)(t / N);
        const __half* xr = x + (int64_t)m * K;
        const uint8_t* wr = packed + (int64_t)n * (K / G) * B;
        float acc = 0.0f;
        if constexpr (G == FQ_Q8_0_GROUP)
        {
            for (int k0 = 0; k0 < K; k0 += 32)
            {
                const uint8_t* blk = wr + (k0 / 32) * 34;
                __half dh;
                memcpy(&dh, blk, 2);
                float d = __half2float(dh);
                for (int j = 0; j < 32; j++)
                {
                    float w = __fmul_rn((float)((int8_t)blk[2 + j]), d);
                    acc = __fadd_rn(acc, __fmul_rn(__half2float(xr[k0 + j]), w));
                }
            }
        }
        else
        {
            for (int k0 = 0; k0 < K; k0 += 256)
            {
                const uint8_t* blk = wr + (k0 / 256) * 66;
                for (int ib = 0; ib < 8; ib++)
                {
                    const uint8_t* q = blk + 2 + 8 * ib;
                    uint32_t q0 = (uint32_t)q[0] | ((uint32_t)q[1] << 8);
                    uint32_t q1 = (uint32_t)q[2] | ((uint32_t)q[3] << 8);
                    uint32_t q2 = (uint32_t)q[4] | ((uint32_t)q[5] << 8);
                    uint32_t q3 = (uint32_t)q[6] | ((uint32_t)q[7] << 8);
                    uint32_t a0 = q0 | (q1 << 16);
                    uint32_t a1 = q2 | (q3 << 16);
                    __half dh;
                    memcpy(&dh, blk, 2);
                    float db = __fmul_rn(__fmul_rn(__half2float(dh), 0.5f + (float)(a1 >> 28)), 0.25f);
                    for (int elle = 0; elle < 4; elle++)
                    {
                        uint64_t grid = FQ_IQ2XXS_GRID[(a0 >> (8 * elle)) & 0xFF];
                        uint8_t signs = FQ_KSIGNS_IQ2XS[(a1 >> (7 * elle)) & 127];
                        int kb = k0 + 32 * ib + 8 * elle;
                        for (int j = 0; j < 8; j++)
                        {
                            float mag = (float)((grid >> (8 * j)) & 0xFF);
                            float sgn = (signs & FQ_KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
                            float w = __fmul_rn(__fmul_rn(db, mag), sgn);
                            acc = __fadd_rn(acc, __fmul_rn(__half2float(xr[kb + j]), w));
                        }
                    }
                }
            }
        }
        out[t] = __float2half_rn(acc);
    }
}

__global__ void fq_gemm_nvfp4_kernel(const __half* __restrict__ x, const uint8_t* __restrict__ packed,
                                     const uint8_t* __restrict__ scales, __half* __restrict__ out,
                                     int M, int N, int K, float global)
{
    // packed [N, K/2], scales [N, K/16].
    int64_t total = (int64_t)M * N;
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (int64_t)gridDim.x * blockDim.x)
    {
        int n = (int)(t % N);
        int m = (int)(t / N);
        const __half* xr = x + (int64_t)m * K;
        const uint8_t* pr = packed + (int64_t)n * (K / 2);
        const uint8_t* sr = scales + (int64_t)n * (K / 16);
        float acc = 0.0f;
        for (int k0 = 0; k0 < K; k0 += 16)
        {
            float gs = fq_e4m3(sr[k0 / 16]);
            for (int j = 0; j < 16; j += 2)
            {
                uint8_t byte = pr[(k0 + j) / 2];
                float w0 = __fmul_rn(__fmul_rn(FQ_E2M1_LUT[byte & 0xF], gs), global);
                float w1 = __fmul_rn(__fmul_rn(FQ_E2M1_LUT[(byte >> 4) & 0xF], gs), global);
                acc = __fadd_rn(acc, __fmul_rn(__half2float(xr[k0 + j]), w0));
                acc = __fadd_rn(acc, __fmul_rn(__half2float(xr[k0 + j + 1]), w1));
            }
        }
        out[t] = __float2half_rn(acc);
    }
}

template <int G, int B>
static void fq_gemm_q(torch::Tensor x, torch::Tensor packed, torch::Tensor out)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(packed, kByte);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK_DIM(x, 2);
    TORCH_CHECK_DIM(packed, 3);
    TORCH_CHECK_DIM(out, 2);
    TORCH_CHECK_SIZE(packed, 2, B);
    int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
    TORCH_CHECK((int)out.size(0) == M, "out rows must match x rows");
    TORCH_CHECK(K % G == 0, "K must be a multiple of the quant group");
    TORCH_CHECK((int)packed.size(0) == N && (int)packed.size(1) == K / G, "packed must be [N, K/G, B]");
    TORCH_CHECK(M >= 1 && M <= 16, "fused path covers M = 1..16; larger rows take dequant+matmul");
    at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t total = (int64_t)M * N;
    fq_gemm_kernel<G, B><<<CEIL_DIVIDE(total, 256), 256, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), packed.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), M, N, K);
}

}  // namespace

void fq_gemm_q8_0(torch::Tensor x, torch::Tensor packed, torch::Tensor out)
{
    fq_gemm_q<FQ_Q8_0_GROUP, FQ_Q8_0_BYTES>(x, packed, out);
}

void fq_gemm_iq2_xxs(torch::Tensor x, torch::Tensor packed, torch::Tensor out)
{
    fq_gemm_q<FQ_IQ2_XXS_GROUP, FQ_IQ2_XXS_BYTES>(x, packed, out);
}

void fq_gemm_nvfp4(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, double global_scale,
                   torch::Tensor out)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(packed, kByte);
    TORCH_CHECK_DTYPE(scales, kByte);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && scales.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && scales.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK_DIM(x, 2);
    TORCH_CHECK_DIM(packed, 2);
    TORCH_CHECK_DIM(scales, 2);
    TORCH_CHECK_DIM(out, 2);
    int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
    TORCH_CHECK((int)out.size(0) == M, "out rows must match x rows");
    TORCH_CHECK(K % FQ_NVFP4_GROUP == 0, "K must be a multiple of 16");
    TORCH_CHECK((int)packed.size(0) == N && (int)packed.size(1) == K / 2, "packed must be [N, K/2]");
    TORCH_CHECK((int)scales.size(0) == N && (int)scales.size(1) == K / 16, "scales must be [N, K/16]");
    TORCH_CHECK(M >= 1 && M <= 16, "fused path covers M = 1..16; larger rows take dequant+matmul");
    at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t total = (int64_t)M * N;
    fq_gemm_nvfp4_kernel<<<CEIL_DIVIDE(total, 256), 256, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), packed.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        M, N, K, (float)global_scale);
}
