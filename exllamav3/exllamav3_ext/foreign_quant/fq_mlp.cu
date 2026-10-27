// Fused shared-expert up + relu2 for foreign-quant MLPs (Bolt MoE layers,
// dense IQ2_XXS shared expert): batch-1 up GEMV with the torch-exact relu2
// epilogue, writing the [1, I] half activations to a caller-owned scratch
// the down GEMV (plain fq_gemv_iq2_xxs, half in / float out) then consumes.
//
// Why: the shared expert runs on the torch path (up gemv + torch relu +
// torch square + down gemv, 2 x 86 us + elementwise launches per layer at
// Bolt shapes). The dots dominate and are unchanged here; this fuses only
// the activation into the up epilogue, killing the relu/square launches,
// the intermediate u allocation, and u's global round-trip. Same dot and
// same epilogue arithmetic as fq_moe_up_kernel (fq_moe.cu), so output is
// bit-identical to the torch path (not just fp16-tolerant).
//
// Portable CUDA/HIP: same contract as fq_gemv.cu (no PTX, no shuffles,
// shared-memory reduction, __fmul_rn/__fadd_rn only).
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
#include "fq_api.h"

namespace {

// Dense S=1 up kernel: block t computes up to 8 rows of up against x,
// relu2 (rounded to half exactly like torch square(relu(half)): the sum is
// rounded to half first, then the square of two half values is exact in
// fp32 and rounds to half identically to a hardware hmul) into h.
__global__ void fq_mlp_up_kernel(const __half* __restrict__ x,
                                 const uint8_t* __restrict__ up,
                                 int64_t up_stride,
                                 __half* __restrict__ h,
                                 int I, int K, int ntiles)
{
    int t = blockIdx.x;
    int grp = threadIdx.x / FQ_GEMV_LANES;   // 0..7 output slot in this block
    int lane = threadIdx.x % FQ_GEMV_LANES;
    int n = t * FQ_GEMV_OUT_PER_BLOCK + grp;

    extern __shared__ __half sh[];
    __half* xs = sh;                                            // K halves: staged x
    uint32_t* hdrs = reinterpret_cast<uint32_t*>(xs + K);       // 8 x K/256*16 u32
    float* red = reinterpret_cast<float*>(hdrs + 8 * (K / FQ_IQ2_XXS_GROUP) * 16);

    __shared__ fq_tables_sh tab;
    fq_tables_load(&tab, 0, 1);
    fq_stage_x<__half>(x, xs, K);
    fq_stage_iq2_hdrs(up, up_stride, hdrs, K / FQ_IQ2_XXS_GROUP,
                      t * FQ_GEMV_OUT_PER_BLOCK, I);
    __syncthreads();

    float u = 0.0f;
    if (n < I)
        u = fq_dot_iq2_xxs(up + (int64_t)n * up_stride, xs, &tab,
                            hdrs + (int64_t)grp * (K / FQ_IQ2_XXS_GROUP) * 16, K, lane);
    red[threadIdx.x] = u;
    __syncthreads();
    if (n < I && lane == 0)
    {
        float su = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
            su = __fadd_rn(su, red[grp * 32 + l]);
        float uh = __half2float(__float2half_rn(su));
        float au = uh > 0.0f ? uh * uh : 0.0f;
        h[n] = __float2half_rn(au);
    }
}

// q8_1-activation variant (Bolt phase 2): row-per-block integer-dp4a dot
// over the globally quantized x, same torch-exact relu2 epilogue.
// V2 template (default on): the v2 dot needs no prologue tables.
template <bool V2>
__global__ void fq_mlp_up_kernel_q8(const uint8_t* __restrict__ aqb,
                                    const uint8_t* __restrict__ up,
                                    int64_t up_stride,
                                    __half* __restrict__ h,
                                    int I, int K)
{
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    int nb = K / FQ_IQ2_XXS_GROUP;

    extern __shared__ uint8_t sh8[];
    float* red;
    uint32_t* hdrs = nullptr;
    uint32_t* signs32 = nullptr;
    __shared__ fq_tables_sh tab;
    const uint8_t* grid8 = nullptr;
    if constexpr (!V2)
    {
        hdrs = reinterpret_cast<uint32_t*>(sh8);
        signs32 = hdrs + (size_t)nb * 16;
        red = reinterpret_cast<float*>(signs32 + 256);
        fq_tables_load(&tab, 0, 1);
        if (n < I)
            fq_stage_iq2_hdrs_row(up, up_stride, hdrs, nb, n, I);
        __syncthreads();   // staging race guard (see fq_gemv_q8.cu)
        fq_signs32_build(&tab, signs32);
        __syncthreads();
        grid8 = reinterpret_cast<const uint8_t*>(tab.grid);
    }
    else
    {
        red = reinterpret_cast<float*>(sh8);
    }

    float u = 0.0f;
    if (n < I)
    {
        if constexpr (V2)
            u = fq_dot_iq2_xxs_q8v2(up + (int64_t)n * up_stride, aqb, K, tid, nt);
        else
            u = fq_dot_iq2_xxs_q8(up + (int64_t)n * up_stride, hdrs, grid8,
                                  signs32, aqb, K, tid, nt);
    }
    red[tid] = u;
    __syncthreads();
    if (n < I && tid == 0)
    {
        float su = 0.0f;
        for (int l = 0; l < nt; l++)
            su = __fadd_rn(su, red[l]);
        float uh = __half2float(__float2half_rn(su));
        float au = uh > 0.0f ? uh * uh : 0.0f;
        h[n] = __float2half_rn(au);
    }
}

}  // namespace

void fq_mlp_shared_iq2xxs(torch::Tensor x, torch::Tensor up_packed, torch::Tensor h)
{
    TORCH_CHECK(x.is_cuda() && up_packed.is_cuda() && h.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.is_contiguous() && up_packed.is_contiguous() && h.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(h, kHalf);
    TORCH_CHECK_DIM(x, 2);
    TORCH_CHECK_DIM(up_packed, 3);
    TORCH_CHECK_DIM(h, 2);
    TORCH_CHECK((int)x.size(0) == 1 && (int)h.size(0) == 1, "batch-1 decode only");
    int K = (int)x.size(1);
    int I = (int)h.size(1);
    TORCH_CHECK(K % FQ_IQ2_XXS_GROUP == 0, "K must be a multiple of 256");
    TORCH_CHECK(K <= 8192, "IQ2 header scratch covers K <= 8192");
    TORCH_CHECK((int)up_packed.size(0) == I && (int)up_packed.size(1) == K / FQ_IQ2_XXS_GROUP,
                "up_packed must be [I, K/256, 66]");
    TORCH_CHECK_SIZE(up_packed, 2, FQ_IQ2_XXS_BYTES);
    at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (fq_use_q8act())
    {
        torch::Tensor aqb = fq_q8_scratch((int)x.get_device(), 1, K);
        fq_quant_q8_1_rows(x, aqb);
        int64_t up_stride = (int64_t)(K / FQ_IQ2_XXS_GROUP) * FQ_IQ2_XXS_BYTES;
        int nchunks = K / 32;
        int nt = fq_q8_nt_iq2(K);
        bool v2 = fq_use_q8v2();
        size_t shmem = v2 ? (size_t)nt * sizeof(float)
                          : (size_t)(K / FQ_IQ2_XXS_GROUP) * 16 * sizeof(uint32_t) +
                            256 * sizeof(uint32_t) + (size_t)nt * sizeof(float);
        if (v2)
            fq_mlp_up_kernel_q8<true><<<I, nt, shmem, stream>>>(
                aqb.data_ptr<uint8_t>(),
                up_packed.data_ptr<uint8_t>(), up_stride,
                reinterpret_cast<__half*>(h.data_ptr<at::Half>()), I, K);
        else
            fq_mlp_up_kernel_q8<false><<<I, nt, shmem, stream>>>(
                aqb.data_ptr<uint8_t>(),
                up_packed.data_ptr<uint8_t>(), up_stride,
                reinterpret_cast<__half*>(h.data_ptr<at::Half>()), I, K);
        return;
    }
    int64_t up_stride = (int64_t)(K / FQ_IQ2_XXS_GROUP) * FQ_IQ2_XXS_BYTES;
    int ntiles = (I + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    size_t shmem = ((size_t)K) * sizeof(__half) + FQ_GEMV_THREADS * sizeof(float) +
                   ((size_t)8 * (K / FQ_IQ2_XXS_GROUP)) * 16 * sizeof(uint32_t);
    fq_mlp_up_kernel<<<ntiles, FQ_GEMV_THREADS, shmem, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        up_packed.data_ptr<uint8_t>(), up_stride,
        reinterpret_cast<__half*>(h.data_ptr<at::Half>()), I, K, ntiles);
}
