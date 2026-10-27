// Grouped decode MoE with q8_1 activations for K-quant/IQ4_XS experts
// same (slot, tile) grid as fq_moe_q8.cu (one 256-thread
// block per (slot s, 8-row tile t), gate+up + down + fold launches) but the
// KQ/IQ4_XS integer-dp4a dots (fq_dot_*_q8 in fq_gemv.cuh) and a SwiGLU
// epilogue (MiniCPM MoE is SwiGLU, unlike Bolt's gateless latent MoE).
// h [1, K] is quantized once; the [S, I] act scratch is quantized per slot
// row for the down dots. FMT selects the dot (0 = Q4_K, 1 = Q5_K,
// 2 = Q6_K, 3 = IQ4_XS). Portable (no PTX/shuffles).
#include <cuda_fp16.h>
#include <map>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
#include "fq_api.h"

namespace {

// Up+gate kernel: block (s, t) computes up to 8 gate rows and 8 up rows of
// expert sel[s] against the pre-quantized h (aqb, one row), SwiGLU epilogue
// (silu(gate) * up, torch half path) into act[s]. Lane mapping: group grp
// owns output n; lane l takes sub-chunks round-robin (the dot's (lane, 32)
// split), 32 partials reduce in shared.
template <int FMT>
__global__ void fq_moe_kq_upgate_kernel(const uint8_t* __restrict__ aqb,
                                        const int64_t* __restrict__ sel,
                                        const uint8_t* const* __restrict__ gate_bases,
                                        const uint8_t* const* __restrict__ up_bases,
                                        int64_t gate_stride,
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

    extern __shared__ float red[];

    float g = 0.0f, u = 0.0f;
    if (n < I)
    {
        const uint8_t* gr = gate_bases[e] + (int64_t)n * gate_stride;
        const uint8_t* ur = up_bases[e] + (int64_t)n * up_stride;
        if constexpr (FMT == 0)
        {
            g = fq_dot_q4_k_q8(gr, aqb, K, lane, FQ_GEMV_LANES);
            u = fq_dot_q4_k_q8(ur, aqb, K, lane, FQ_GEMV_LANES);
        }
        else if constexpr (FMT == 1)
        {
            g = fq_dot_q5_k_q8(gr, aqb, K, lane, FQ_GEMV_LANES);
            u = fq_dot_q5_k_q8(ur, aqb, K, lane, FQ_GEMV_LANES);
        }
        else if constexpr (FMT == 2)
        {
            g = fq_dot_q6_k_q8(gr, aqb, K, lane, FQ_GEMV_LANES);
            u = fq_dot_q6_k_q8(ur, aqb, K, lane, FQ_GEMV_LANES);
        }
        else
        {
            g = fq_dot_iq4_xs_q8(gr, aqb, K, lane, FQ_GEMV_LANES);
            u = fq_dot_iq4_xs_q8(ur, aqb, K, lane, FQ_GEMV_LANES);
        }
    }
    // Two reductions in one shared row: red[2*tid] = g, red[2*tid+1] = u.
    red[2 * threadIdx.x] = g;
    red[2 * threadIdx.x + 1] = u;
    __syncthreads();
    if (n < I && lane == 0)
    {
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
        {
            sg = __fadd_rn(sg, red[(grp * 32 + l) * 2]);
            su = __fadd_rn(su, red[(grp * 32 + l) * 2 + 1]);
        }
        // Torch half path: g/u round to half (Linear out_dtype = half),
        // silu in float, product back to half.
        float gh = __half2float(__float2half_rn(sg));
        float uh = __half2float(__float2half_rn(su));
        float sig = 1.0f / (1.0f + expf(-gh));
        act[(int64_t)s * I + n] = __float2half_rn(sig * gh * uh);
    }
}

// Down kernel: block (s, t) dots down[sel[s]] rows against the
// pre-quantized act row s into partial[s]. Same lane split as above.
template <int FMT>
__global__ void fq_moe_kq_down_kernel(const uint8_t* __restrict__ aqb,
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

    extern __shared__ float red[];

    float p = 0.0f;
    if (n < K)
    {
        const uint8_t* dr = down_bases[e] + (int64_t)n * down_stride;
        const uint8_t* aq = aqb + (int64_t)s * qb_stride;
        if constexpr (FMT == 0)
            p = fq_dot_q4_k_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else if constexpr (FMT == 1)
            p = fq_dot_q5_k_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else if constexpr (FMT == 2)
            p = fq_dot_q6_k_q8(dr, aq, I, lane, FQ_GEMV_LANES);
        else
            p = fq_dot_iq4_xs_q8(dr, aq, I, lane, FQ_GEMV_LANES);
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

// Weighted fold (same as fq_moe_q8.cu's): out[k] = sum_s w[s]*partial[s,k].
__global__ void fq_moe_kq_combine_kernel(const float* __restrict__ partial,
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

template <int FMT>
static void fq_moe_kq_grouped_launch(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                                     torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                                     torch::Tensor down_ptrs,
                                     int64_t gate_stride, int64_t up_stride,
                                     int64_t down_stride, int64_t block_bytes,
                                     torch::Tensor partial, torch::Tensor out)
{
    int K = (int)h.size(1);
    int S = (int)sel.numel();
    TORCH_CHECK_DTYPE(h, kHalf);
    int E = (int)up_ptrs.numel();
    // I from the down row stride (down rows are I-wide, like Bolt's
    // derivation from down_stride).
    int I = (int)(down_stride / block_bytes) * FQ_Q4_K_GROUP;
    at::cuda::CUDAGuard guard(h.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int ntiles_u = (I + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    int ntiles_d = (K + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
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
    // Up+gate: 2 floats per thread in shared (g and u side by side).
    size_t shmem_u = (size_t)FQ_GEMV_THREADS * 2 * sizeof(float);
    fq_moe_kq_upgate_kernel<FMT><<<S * ntiles_u, FQ_GEMV_THREADS, shmem_u, stream>>>(
        qb_h.data_ptr<uint8_t>(), sel.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(gate_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint8_t* const*>(up_ptrs.data_ptr<int64_t>()),
        gate_stride, up_stride, qb_stride_h,
        reinterpret_cast<__half*>(act.data_ptr<at::Half>()),
        S, I, K, E, ntiles_u);
    torch::Tensor qb_act = fq_q8_scratch(h.get_device(), S, I);
    fq_quant_q8_1_rows(act, qb_act);
    int64_t qb_stride_a = (int64_t)(I / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
    size_t shmem_d = (size_t)FQ_GEMV_THREADS * sizeof(float);
    fq_moe_kq_down_kernel<FMT><<<S * ntiles_d, FQ_GEMV_THREADS, shmem_d, stream>>>(
        qb_act.data_ptr<uint8_t>(), qb_stride_a, sel.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t* const*>(down_ptrs.data_ptr<int64_t>()),
        down_stride, partial.data_ptr<float>(), S, I, K, E, ntiles_d);
    int nblocks = (K + 255) / 256;
    fq_moe_kq_combine_kernel<<<nblocks, 256, 0, stream>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<const __half*>(wts.data_ptr<at::Half>()),
        out.data_ptr<float>(), S, K);
}

void fq_moe_q4k_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                           torch::Tensor down_ptrs,
                           int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out)
{
    fq_moe_kq_grouped_launch<0>(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs,
                                gate_stride, up_stride, down_stride,
                                FQ_Q4_K_BYTES, partial, out);
}

void fq_moe_q5k_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                           torch::Tensor down_ptrs,
                           int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out)
{
    fq_moe_kq_grouped_launch<1>(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs,
                                gate_stride, up_stride, down_stride,
                                FQ_Q5_K_BYTES, partial, out);
}

void fq_moe_q6k_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                           torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                           torch::Tensor down_ptrs,
                           int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                           torch::Tensor partial, torch::Tensor out)
{
    fq_moe_kq_grouped_launch<2>(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs,
                                gate_stride, up_stride, down_stride,
                                FQ_Q6_K_BYTES, partial, out);
}

void fq_moe_iq4xs_grouped_q8(torch::Tensor h, torch::Tensor sel, torch::Tensor wts,
                             torch::Tensor gate_ptrs, torch::Tensor up_ptrs,
                             torch::Tensor down_ptrs,
                             int64_t gate_stride, int64_t up_stride, int64_t down_stride,
                             torch::Tensor partial, torch::Tensor out)
{
    fq_moe_kq_grouped_launch<3>(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs,
                                gate_stride, up_stride, down_stride,
                                FQ_IQ4_XS_BYTES, partial, out);
}
