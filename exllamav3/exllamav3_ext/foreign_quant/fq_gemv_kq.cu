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
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
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
        else
            acc = fq_dot_iq4_xs_q8(wr, aq, K, tid, nt);
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

}  // namespace

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
