// Grouped decode MoE with q8_1 activations (Bolt phase 2): same grid as
// fq_moe.cu (one 256-thread block per (slot s, 8-row tile t), up + down +
// fold launches) but integer-dp4a dots over once-per-step q8_1 activations
// (llama.cpp mmvq design). h [1, K] is quantized once for the up dots;
// the [S, I] act scratch is quantized per slot row for the down dots.
// Opt-in via BOLT_FQ_Q8MOE=1 (branch inside fq_moe_iq2xxs_grouped).
// Portable (no PTX/shuffles; __dp4a via fq_dp4a with HIP fallback).
#include <cuda_fp16.h>
#include <map>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
#include "fq_api.h"

namespace {

// Up kernel: block (s, t) computes up to 8 rows of up[sel[s]] against the
// pre-quantized h (aqb, one row), relu2 epilogue torch-exact, into act[s].
// Lane mapping: group grp owns output n; lane l takes 32-chunks round-robin
// (the fq_dot_iq2_xxs_q8 (lane, 32) split), 32 partials reduce in shared.
// V2 template (default on): the v2 dot needs no prologue tables.
template <bool V2>
__global__ void fq_moe_up_kernel_q8(const uint8_t* __restrict__ aqb,
                                    const int64_t* __restrict__ sel,
                                    const uint8_t* const* __restrict__ up_bases,
                                    int64_t up_stride,
                                    int64_t qb_stride,
                                    __half* __restrict__ act,
                                    int S, int I, int K, int E, int ntiles_u)
{
    int b = blockIdx.x;
    int s = b / ntiles_u;
    int t = b % ntiles_u;
    int grp = threadIdx.x / FQ_GEMV_LANES;
    int lane = threadIdx.x % FQ_GEMV_LANES;
    int64_t e = sel[s];
    e = (e < 0 || e >= E) ? 0 : e;
    int n = t * FQ_GEMV_OUT_PER_BLOCK + grp;

    extern __shared__ uint8_t sh[];
    float* red;
    uint32_t* hdrs = nullptr;
    uint32_t* signs32 = nullptr;
    __shared__ fq_tables_sh tab;
    const uint8_t* grid8 = nullptr;
    int nb = K / FQ_IQ2_XXS_GROUP;
    if constexpr (!V2)
    {
        // Layout: hdrs (8 x nb*16 u32), signs32 (256 u32), red (256 floats).
        hdrs = reinterpret_cast<uint32_t*>(sh);
        signs32 = hdrs + (size_t)8 * nb * 16;
        red = reinterpret_cast<float*>(signs32 + 256);
        fq_tables_load(&tab, 0, 1);
        fq_stage_iq2_hdrs(up_bases[e], up_stride, hdrs, nb,
                          t * FQ_GEMV_OUT_PER_BLOCK, I);
        // Barrier BEFORE signs32_build consumes the staged tab (same race as
        // the row-per-block kernel: staging is cooperative, the build reads
        // other threads' entries).
        __syncthreads();
        fq_signs32_build(&tab, signs32);
        __syncthreads();
        grid8 = reinterpret_cast<const uint8_t*>(tab.grid);
    }
    else
    {
        red = reinterpret_cast<float*>(sh);
    }

    float u = 0.0f;
    if (n < I)
    {
        if constexpr (V2)
            u = fq_dot_iq2_xxs_q8v2(up_bases[e] + (int64_t)n * up_stride,
                                    aqb, K, lane, FQ_GEMV_LANES);
        else
            u = fq_dot_iq2_xxs_q8(up_bases[e] + (int64_t)n * up_stride,
                                  hdrs + (int64_t)grp * nb * 16, grid8, signs32,
                                  aqb, K, lane, FQ_GEMV_LANES);
    }
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

// Down kernel: block (s, t) dots down[sel[s]] rows against the
// pre-quantized act row s into partial[s]. V2 template (default on).
template <bool V2>
__global__ void fq_moe_down_kernel_q8(const uint8_t* __restrict__ aqb,
                                      int64_t qb_stride,
                                      const int64_t* __restrict__ sel,
                                      const uint8_t* __restrict__ const* down_bases,
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

    extern __shared__ uint8_t sh[];
    float* red;
    uint32_t* hdrs = nullptr;
    uint32_t* signs32 = nullptr;
    __shared__ fq_tables_sh tab;
    const uint8_t* grid8 = nullptr;
    int nb = I / FQ_IQ2_XXS_GROUP;
    if constexpr (!V2)
    {
        hdrs = reinterpret_cast<uint32_t*>(sh);
        signs32 = hdrs + (size_t)8 * nb * 16;
        red = reinterpret_cast<float*>(signs32 + 256);
        fq_tables_load(&tab, 0, 1);
        fq_stage_iq2_hdrs(down_bases[e], down_stride, hdrs, nb,
                          t * FQ_GEMV_OUT_PER_BLOCK, K);
        // Barrier BEFORE signs32_build (see up kernel above).
        __syncthreads();
        fq_signs32_build(&tab, signs32);
        __syncthreads();
        grid8 = reinterpret_cast<const uint8_t*>(tab.grid);
    }
    else
    {
        red = reinterpret_cast<float*>(sh);
    }

    float p = 0.0f;
    if (n < K)
    {
        if constexpr (V2)
            p = fq_dot_iq2_xxs_q8v2(down_bases[e] + (int64_t)n * down_stride,
                                    aqb + (int64_t)s * qb_stride, I, lane, FQ_GEMV_LANES);
        else
            p = fq_dot_iq2_xxs_q8(down_bases[e] + (int64_t)n * down_stride,
                                  hdrs + (int64_t)grp * nb * 16, grid8, signs32,
                                  aqb + (int64_t)s * qb_stride, I, lane, FQ_GEMV_LANES);
    }
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

}  // namespace

// Weighted fold (duplicate of fq_moe.cu's: that TU keeps it in an
// anonymous namespace, so this TU has its own copy): out[k] = sum_s
// w[s] * partial[s, k]. One block, 256 threads, strided subset of K.
__global__ void fq_moe_combine_kernel_q8(const float* __restrict__ partial,
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

bool fq_use_q8moe()
{
    const char* e = std::getenv("BOLT_FQ_Q8MOE");
    return !e || e[0] != '0';
}

void fq_moe_iq2xxs_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                              torch::Tensor up_ptrs, torch::Tensor down_ptrs,
                              int64_t up_stride, int64_t down_stride,
                              torch::Tensor partial, torch::Tensor out)
{
    int K = (int)h.size(1);
    int S = (int)sel.numel();
    // Entry checks (S range, K/I multiples, dtypes) are done by the caller
    // (fq_moe_iq2xxs_grouped branches here after validating); K%256 and
    // I%256 imply the q8 path's K%32/I%32.
    TORCH_CHECK_DTYPE(h, kHalf);
    int E = (int)up_ptrs.numel();
    int I = (int)(down_stride / FQ_IQ2_XXS_BYTES) * FQ_IQ2_XXS_GROUP;
    at::cuda::CUDAGuard guard(h.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int ntiles_u = (I + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    int ntiles_d = (K + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    // Quantize once: h row (up dots) + per-slot act rows (down dots).
    torch::Tensor qb_h = fq_q8_scratch(h.get_device(), 1, K);
    fq_quant_q8_1_rows(h, qb_h);
    int64_t qb_stride_h = (int64_t)(K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
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
    int nb_u = K / FQ_IQ2_XXS_GROUP;
    bool v2 = fq_use_q8v2();
    size_t shmem_u = v2 ? FQ_GEMV_THREADS * sizeof(float)
                        : (size_t)8 * nb_u * 16 * sizeof(uint32_t) +
                          256 * sizeof(uint32_t) + FQ_GEMV_THREADS * sizeof(float);
    if (v2)
        fq_moe_up_kernel_q8<true><<<S * ntiles_u, FQ_GEMV_THREADS, shmem_u, stream>>>(
            qb_h.data_ptr<uint8_t>(), sel.data_ptr<int64_t>(),
            reinterpret_cast<const uint8_t* const*>(up_ptrs.data_ptr<int64_t>()),
            up_stride, qb_stride_h,
            reinterpret_cast<__half*>(act.data_ptr<at::Half>()),
            S, I, K, E, ntiles_u);
    else
        fq_moe_up_kernel_q8<false><<<S * ntiles_u, FQ_GEMV_THREADS, shmem_u, stream>>>(
            qb_h.data_ptr<uint8_t>(), sel.data_ptr<int64_t>(),
            reinterpret_cast<const uint8_t* const*>(up_ptrs.data_ptr<int64_t>()),
            up_stride, qb_stride_h,
            reinterpret_cast<__half*>(act.data_ptr<at::Half>()),
            S, I, K, E, ntiles_u);
    torch::Tensor qb_act = fq_q8_scratch(h.get_device(), S, I);
    fq_quant_q8_1_rows(act, qb_act);
    int64_t qb_stride_a = (int64_t)(I / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
    int nb_d = I / FQ_IQ2_XXS_GROUP;
    size_t shmem_d = v2 ? FQ_GEMV_THREADS * sizeof(float)
                        : (size_t)8 * nb_d * 16 * sizeof(uint32_t) +
                          256 * sizeof(uint32_t) + FQ_GEMV_THREADS * sizeof(float);
    if (v2)
        fq_moe_down_kernel_q8<true><<<S * ntiles_d, FQ_GEMV_THREADS, shmem_d, stream>>>(
            qb_act.data_ptr<uint8_t>(), qb_stride_a, sel.data_ptr<int64_t>(),
            reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr<int64_t>()),
            down_stride, partial.data_ptr<float>(), S, I, K, E, ntiles_d);
    else
        fq_moe_down_kernel_q8<false><<<S * ntiles_d, FQ_GEMV_THREADS, shmem_d, stream>>>(
            qb_act.data_ptr<uint8_t>(), qb_stride_a, sel.data_ptr<int64_t>(),
            reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr<int64_t>()),
            down_stride, partial.data_ptr<float>(), S, I, K, E, ntiles_d);
    int nblocks = (K + 255) / 256;
    fq_moe_combine_kernel_q8<<<nblocks, 256, 0, stream>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<const __half*>(wts.data_ptr<at::Half>()),
        out.data_ptr<float>(), S, K);
}
