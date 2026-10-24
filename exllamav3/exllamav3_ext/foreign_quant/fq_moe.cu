// Grouped decode MoE for foreign-quant experts (Bolt latent MoE, gateless
// relu2, IQ2_XXS): the selected experts' up + down GEMVs run in two launches
// (plus a weighted-fold launch), replacing the per-expert torch loop (9 up +
// 9 down launches, plus the argsort / bincount+tolist host sync /
// index_select / index_add_ per MoE layer that the dense dispatch pays at
// decode).
//
// Grid (BOTH kernels): one 256-thread block per (slot s, 8-row tile t) --
// the same parallelization as fq_gemv (one block must NOT loop all tiles:
// a single block running every tile sequentially leaves 73 of 82 SMs idle
// and measured 6x slower than the per-expert loop). The up kernel writes
// the relu2-half activations to a global [S, I] scratch (46 KB at Bolt
// shapes); the down kernel stages one scratch row and writes partial[s].
// A third single-block kernel folds partial with the routing weights in
// slot order (fp32). Element arithmetic is the same fq_dot_* as fq_gemv.cu
// (shared header fq_gemv.cuh); only accumulation/summation ORDERS differ
// from the dense torch path (fp16 tolerance).
//
// Portable CUDA/HIP: same contract as fq_gemv.cu (no PTX, no shuffles,
// shared-memory reduction, __fmul_rn/__fadd_rn only).
#include <cuda_fp16.h>
#include <map>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
#include "fq_api.h"

namespace {

// Up kernel: block (s, t) computes up to 8 rows of up[sel[s]] against h,
// relu2 (rounded to half exactly like torch square(relu(half)): the sum is
// rounded to half first, then the square of two half values is exact in
// fp32 and rounds to half identically to a hardware hmul) into act[s].
__global__ void fq_moe_up_kernel(const __half* __restrict__ h,
                                 const int64_t* __restrict__ sel,
                                 const uint8_t* const* __restrict__ up_bases,
                                 int64_t up_stride,
                                 __half* __restrict__ act,
                                 int S, int I, int K, int E, int ntiles_u)
{
    int b = blockIdx.x;
    int s = b / ntiles_u;
    int t = b % ntiles_u;
    int grp = threadIdx.x / FQ_GEMV_LANES;   // 0..7 output slot in this block
    int lane = threadIdx.x % FQ_GEMV_LANES;
    int64_t e = sel[s];
    e = (e < 0 || e >= E) ? 0 : e;
    int n = t * FQ_GEMV_OUT_PER_BLOCK + grp;

    extern __shared__ __half sh[];
    __half* xs = sh;                                            // K halves: staged h
    uint32_t* hdrs = reinterpret_cast<uint32_t*>(xs + K);       // 8 x K/256*16 u32
    float* red = reinterpret_cast<float*>(hdrs + 8 * (K / FQ_IQ2_XXS_GROUP) * 16);

    __shared__ fq_tables_sh tab;
    fq_tables_load(&tab, 0, 1);
    fq_stage_x<__half>(h, xs, K);
    fq_stage_iq2_hdrs(up_bases[e], up_stride, hdrs, K / FQ_IQ2_XXS_GROUP,
                       t * FQ_GEMV_OUT_PER_BLOCK, I);
    __syncthreads();

    float u = 0.0f;
    if (n < I)
        u = fq_dot_iq2_xxs(up_bases[e] + (int64_t)n * up_stride, xs, &tab,
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
        act[(int64_t)s * I + n] = __float2half_rn(au);
    }
}

// Down kernel: block (s, t) stages act[s] and computes up to 8 rows of
// down[sel[s]] into partial[s].
__global__ void fq_moe_down_kernel(const __half* __restrict__ act,
                                   const int64_t* __restrict__ sel,
                                   const uint8_t* const* __restrict__ down_bases,
                                   int64_t down_stride,
                                   float* __restrict__ partial,
                                   int S, int I, int K, int E, int ntiles_d)
{
    int b = blockIdx.x;
    int s = b / ntiles_d;
    int t = b % ntiles_d;
    int grp = threadIdx.x / FQ_GEMV_LANES;
    int lane = threadIdx.x % FQ_GEMV_LANES;
    int64_t e = sel[s];
    e = (e < 0 || e >= E) ? 0 : e;
    int n = t * FQ_GEMV_OUT_PER_BLOCK + grp;

    extern __shared__ __half sh[];
    __half* xs = sh;                                            // I halves: staged act row
    uint32_t* hdrs = reinterpret_cast<uint32_t*>(xs + I);       // 8 x I/256*16 u32
    float* red = reinterpret_cast<float*>(hdrs + 8 * (I / FQ_IQ2_XXS_GROUP) * 16);

    __shared__ fq_tables_sh tab;
    fq_tables_load(&tab, 0, 1);
    fq_stage_x<__half>(act + (int64_t)s * I, xs, I);
    fq_stage_iq2_hdrs(down_bases[e], down_stride, hdrs, I / FQ_IQ2_XXS_GROUP,
                       t * FQ_GEMV_OUT_PER_BLOCK, K);
    __syncthreads();

    float p = 0.0f;
    if (n < K)
        p = fq_dot_iq2_xxs(down_bases[e] + (int64_t)n * down_stride, xs, &tab,
                            hdrs + (int64_t)grp * (I / FQ_IQ2_XXS_GROUP) * 16, I, lane);
    red[threadIdx.x] = p;
    __syncthreads();
    if (n < K && lane == 0)
    {
        float sp = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
            sp = __fadd_rn(sp, red[grp * 32 + l]);
        partial[(int64_t)s * K + n] = sp;
    }
}

// Weighted fold: out[k] = sum_s w[s] * partial[s, k]. One block, 256
// threads, each a strided subset of K.
__global__ void fq_moe_combine_kernel(const float* __restrict__ partial,
                                      const __half* __restrict__ wts,
                                      float* __restrict__ out, int S, int K)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < K; k += gridDim.x * blockDim.x)
    {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 32; s++)
        {
            if (s < S)
                acc = __fadd_rn(acc, __fmul_rn(partial[(int64_t)s * K + k],
                                              __half2float(wts[s])));
        }
    out[k] = acc;
    }
}

}  // namespace

void fq_moe_iq2xxs_grouped(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor up_ptrs, torch::Tensor down_ptrs,
                           int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out)
{
    TORCH_CHECK(h.is_cuda() && sel.is_cuda() && wts.is_cuda() && partial.is_cuda() && out.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(h.is_contiguous() && sel.is_contiguous() && wts.is_contiguous() &&
                partial.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK_DTYPE(h, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(wts, kHalf);
    TORCH_CHECK_DTYPE(up_ptrs, kLong);
    TORCH_CHECK_DTYPE(down_ptrs, kLong);
    TORCH_CHECK_DTYPE(partial, kFloat);
    TORCH_CHECK_DTYPE(out, kFloat);
    TORCH_CHECK_DIM(h, 2);
    TORCH_CHECK((int)h.size(0) == 1, "grouped MoE covers batch-1 decode");
    int K = (int)h.size(1);
    int S = (int)sel.numel();
    TORCH_CHECK((int)wts.numel() == S, "sel/wts must agree");
    TORCH_CHECK((int)partial.size(0) == S && (int)partial.size(1) == K, "partial must be [S, K]");
    TORCH_CHECK((int)out.size(0) == 1 && (int)out.size(1) == K, "out must be [1, K]");
    TORCH_CHECK(S >= 1 && S <= 32, "top-k must be 1..32");
    TORCH_CHECK(K % FQ_IQ2_XXS_GROUP == 0, "expert width K must be a multiple of 256");
    TORCH_CHECK(K <= 4096, "expert width K <= 4096 (shared scratch)");
    int E = (int)up_ptrs.numel();
    TORCH_CHECK(E > 0 && (int)down_ptrs.numel() == E, "expert pointer tables must agree");
    if (fq_use_q8moe())
    {
        // q8_1-activation path (fq_moe_q8.cu): quantize once, integer dp4a
        // dots. Same contract (fp16 tolerance vs the dense torch path).
        fq_moe_iq2xxs_grouped_q8(h, sel, wts, up_ptrs, down_ptrs,
                                 up_stride, down_stride, partial, out);
        return;
    }
    at::cuda::CUDAGuard guard(h.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Widths from the row strides: up rows are [K/256, 66] (K = expert
    // input width), down rows are [I/256, 66] (I = intermediate width).
    TORCH_CHECK(up_stride == (int64_t)(K / FQ_IQ2_XXS_GROUP) * FQ_IQ2_XXS_BYTES,
                "up stride must match K");
    TORCH_CHECK(down_stride % FQ_IQ2_XXS_BYTES == 0, "bad down stride");
    int I = (int)(down_stride / FQ_IQ2_XXS_BYTES) * FQ_IQ2_XXS_GROUP;
    TORCH_CHECK(I % FQ_IQ2_XXS_GROUP == 0 && I <= 4096, "intermediate width 256..4096");
    // The [S, I] half activation scratch lives in the caller's partial
    // buffer tail? No: separate cached workspace below (partial stays [S,K]).
    int ntiles_u = (I + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    int ntiles_d = (K + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    // Activation scratch: cached per (device, S, I) on the CUDA device.
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
    size_t shmem_u = ((size_t)K) * sizeof(__half) + FQ_GEMV_THREADS * sizeof(float) +
                     ((size_t)8 * (K / FQ_IQ2_XXS_GROUP)) * 16 * sizeof(uint32_t);
    fq_moe_up_kernel<<<S * ntiles_u, FQ_GEMV_THREADS, shmem_u, stream>>>(
        reinterpret_cast<const __half*>(h.data_ptr<at::Half>()),
        sel.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(up_ptrs.data_ptr<int64_t>()),
        up_stride,
        reinterpret_cast<__half*>(act.data_ptr<at::Half>()),
        S, I, K, E, ntiles_u);
    size_t shmem_d = ((size_t)I) * sizeof(__half) + FQ_GEMV_THREADS * sizeof(float) +
                     ((size_t)8 * (I / FQ_IQ2_XXS_GROUP)) * 16 * sizeof(uint32_t);
    fq_moe_down_kernel<<<S * ntiles_d, FQ_GEMV_THREADS, shmem_d, stream>>>(
        reinterpret_cast<const __half*>(act.data_ptr<at::Half>()),
        sel.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr<int64_t>()),
        down_stride,
        partial.data_ptr<float>(), S, I, K, E, ntiles_d);
    int nblocks = (K + 255) / 256;
    fq_moe_combine_kernel<<<nblocks, 256, 0, stream>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<const __half*>(wts.data_ptr<at::Half>()),
        out.data_ptr<float>(), S, K);
}
