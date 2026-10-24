// DS4.1 decode: routed experts of one layer in 5 launches, all K groups at once (batch 1, top-6). Replaces, per layer, the
// ~8 launches x (number of K groups) of exl3_moe_gfx12_k3 (+ dedup metadata, index_select, ...). Same arithmetic as that path:
// the Hadamard / silu / VALU matvec bodies are the engine's own (exl3_moe_valu.cuh, hadamard_inner.cuh), only the launch
// structure changes: the per-expert trellis bitrate is read per block instead of per launch, and the weighted reduction keeps
// the grouped order (per K group, experts ascending, groups summed later by ds41_moe_tail), so outputs are bit-identical.
// Loader: dsv41_moe.py.
#define USE_ROCM 1
#define HIPBLAS_USE_HIP_HALF 1
#define EXL3_HIP_STG_PAD 1
#define __nv_bfloat162 __hip_bfloat162
#define __nv_bfloat16 __hip_bfloat16
#define CEIL_DIVIDE(x, a) (((x) + (a) - 1) / (a))
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <hip/hip_bf16.h>
#include "../../exllamav3_ext/util.cuh"
#include "../../exllamav3_ext/quant/exl3_gemv_kernel.cuh"
#include "../../exllamav3_ext/quant/exl3_moe_valu.cuh"

#ifndef MV_VU
#define MV_VU 5   // k-slices per pipeline stage of the matvec
#endif
#define TOPK 6
constexpr float HAD_SCALE_M = 0.088388347648f;

// ---- 1. x -> gate/up Hadamard input rows (12 matrices x H/128 chunks): moe_had_rows_kernel<true,false>, token input
extern "C" __global__ __launch_bounds__(32)
void ds41m_had_in(const half* __restrict__ x, half* __restrict__ gu_had, const int64_t* __restrict__ sel,
                  const int64_t* __restrict__ suh_g, const int64_t* __restrict__ suh_u, int H, int experts)
{
    const int matrix = blockIdx.x, slot = matrix % TOPK, proj = matrix / TOPK;
    const int64_t e = sel[slot];
    const int64_t sp = (e >= 0 && e < experts) ? (proj == 0 ? suh_g : suh_u)[e] : 0;
    const size_t off = (size_t) matrix * H + blockIdx.y * 128;
    if (!sp)
    {
        for (int i = threadIdx.x; i < 128; i += blockDim.x) gu_had[off + i] = __float2half_rn(0.0f);
        return;
    }
    had_hf_r_128_inner<true, false>(x + blockIdx.y * 128, gu_had + off, reinterpret_cast<const half*>(sp), HAD_SCALE_M);
}

// ---- 2. mixed-K VALU matvec: one launch for all experts; BITS chosen per block from kbits[expert]
template <bool CF32, bool TWO, int VU>
__device__ __forceinline__ void mixed_body
(
    const half* A, const int64_t* selected, const int* kbits, const int64_t* T0, const int64_t* T1, void* C,
    int size_k, int size_n, int experts, int assignments
)
{
    constexpr int W = EXL3_MOE_VALU_W;
    __shared__ float red[mv::red_size(W)];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ntiles = size_n / 16;
    const int tile = mv::tile_of(blockIdx.x, wave, W);
    if (tile >= ntiles) return;
    const int p0 = blockIdx.y;
    const int projection = TWO ? blockIdx.z : 0;
    const int64_t expert = selected[p0];
    const int64_t* B_table = projection == 0 ? T0 : T1;
    const size_t matrix = (size_t) projection * assignments + p0;
    void* C_rows = CF32 ? static_cast<void*>(reinterpret_cast<float*>(C) + matrix * size_n)
                        : static_cast<void*>(reinterpret_cast<half*>(C) + matrix * size_n);
    const bool valid = expert >= 0 && expert < experts;
    const int64_t B_ptr = valid ? B_table[expert] : 0;
    if (!B_ptr)
    {
        if (lane < 16)
        {
            const size_t o = mv::c_idx(0, size_n, tile, lane);
            if constexpr (CF32) reinterpret_cast<float*>(C_rows)[o] = 0.0f;
            else reinterpret_cast<half*>(C_rows)[o] = __float2half_rn(0.0f);
        }
        return;
    }
    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B_ptr);
    const uint32_t* Ap = reinterpret_cast<const uint32_t*>(A + matrix * size_k);
    switch (kbits[expert])
    {
    case 1: mv_tile<1, CF32, 1, VU>(Ap, size_k, B32, ntiles, C_rows, size_n, tile, red, wave, lane); break;
    case 2: mv_tile<2, CF32, 1, VU>(Ap, size_k, B32, ntiles, C_rows, size_n, tile, red, wave, lane); break;
    case 4: mv_tile<4, CF32, 1, VU>(Ap, size_k, B32, ntiles, C_rows, size_n, tile, red, wave, lane); break;
    default: mv_tile<3, CF32, 1, VU>(Ap, size_k, B32, ntiles, C_rows, size_n, tile, red, wave, lane); break;
    }
}

extern "C" __global__ __launch_bounds__(EXL3_MOE_VALU_W * 32)
void ds41m_valu_gu(const half* A, const int64_t* sel, const int* kbits, const int64_t* T0, const int64_t* T1, void* C,
                   int size_k, int size_n, int experts)
{ mixed_body<false, true, MV_VU>(A, sel, kbits, T0, T1, C, size_k, size_n, experts, TOPK); }

extern "C" __global__ __launch_bounds__(EXL3_MOE_VALU_W * 32)
void ds41m_valu_down(const half* A, const int64_t* sel, const int* kbits, const int64_t* T0, const int64_t* T1, void* C,
                     int size_k, int size_n, int experts)
{ mixed_body<true, false, MV_VU>(A, sel, kbits, T0, T1, C, size_k, size_n, experts, TOPK); }

// ---- 3. svh Hadamard of gate and up, silu*up (swiglu limit), suh Hadamard of the down input: one (slot, 128-chunk) per warp
extern "C" __global__ __launch_bounds__(32)
void ds41m_mid(half* __restrict__ gu_out, half* __restrict__ down_had, const int64_t* __restrict__ sel,
               const int64_t* __restrict__ svh_g, const int64_t* __restrict__ svh_u, const int64_t* __restrict__ suh_d,
               int I, int experts, float limit)
{
    const int slot = blockIdx.x;
    const int64_t e = sel[slot];
    const bool ok = e >= 0 && e < experts;
    const int64_t pg = ok ? svh_g[e] : 0, pu = ok ? svh_u[e] : 0, pd = ok ? suh_d[e] : 0;
    const size_t c0 = blockIdx.y * 128;
    half* g = gu_out + (size_t) slot * I + c0;
    half* u = gu_out + (size_t) (TOPK + slot) * I + c0;
    half* d = down_had + (size_t) slot * I + c0;
    if (pg) had_hf_r_128_inner<false, true>(g, g, reinterpret_cast<const half*>(pg), HAD_SCALE_M);
    else for (int i = threadIdx.x; i < 128; i += 32) g[i] = __float2half_rn(0.0f);
    if (pu) had_hf_r_128_inner<false, true>(u, u, reinterpret_cast<const half*>(pu), HAD_SCALE_M);
    else for (int i = threadIdx.x; i < 128; i += 32) u[i] = __float2half_rn(0.0f);
    for (int i = threadIdx.x; i < 128; i += 32)
    {
        float gv = __half2float(g[i]);
        half uv = u[i];
        if (limit > 0.0f)
        {
            gv = fminf(gv, limit);
            uv = __float2half_rn(fminf(fmaxf(__half2float(uv), -limit), limit));
        }
        const half act = __float2half_rn(gv / (1.0f + expf(-gv)));
        d[i] = __hmul(act, uv);
    }
    if (pd) had_hf_r_128_inner<true, false>(d, d, reinterpret_cast<const half*>(pd), HAD_SCALE_M);
    else for (int i = threadIdx.x; i < 128; i += 32) d[i] = __float2half_rn(0.0f);
}

// ---- 5. svh Hadamard of the down outputs + weighted reduction per K group. Group g sums its picks in ascending expert id
// (ties: routing slot), exactly the order of moe_weighted_reduce_kernel on the group-masked picks. out_g = fp32 [H].
extern "C" __global__ __launch_bounds__(32)
void ds41m_final(float* __restrict__ down_out, const int64_t* __restrict__ sel, const half* __restrict__ wts,
                 const int64_t* __restrict__ svh_d, const int64_t* __restrict__ kgroup,
                 float* __restrict__ o0, float* __restrict__ o1, float* __restrict__ o2, int H, int experts, int G)
{
    const int t = threadIdx.x & 31;
    const size_t c0 = (size_t) blockIdx.y * 128;
    float4 rowv[TOPK];
    int64_t ex[TOPK];
    int grp[TOPK];
    #pragma unroll
    for (int s = 0; s < TOPK; ++s)
    {
        ex[s] = sel[s];
        const bool ok = ex[s] >= 0 && ex[s] < experts;
        grp[s] = ok ? (int) kgroup[ex[s]] : -1;
        const int64_t p = ok ? svh_d[ex[s]] : 0;
        float* r = down_out + (size_t) s * H + c0;
        if (p) had_ff_r_128_inner<false, true>(r, r, reinterpret_cast<const half*>(p), HAD_SCALE_M);
        else ((float4*) r)[t] = make_float4(0.f, 0.f, 0.f, 0.f);
        rowv[s] = ((const float4*) r)[t];
    }
    #pragma unroll
    for (int g = 0; g < 3; ++g)
    {
        if (g >= G) break;
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        unsigned used = 0;
        #pragma unroll
        for (int rank = 0; rank < TOPK; ++rank)
        {
            int best = -1;
            int64_t be = INT64_MAX;
            #pragma unroll
            for (int s = 0; s < TOPK; ++s)
                if (grp[s] == g && !(used & (1u << s)) && (ex[s] < be || (ex[s] == be && (best < 0 || s < best)))) { be = ex[s]; best = s; }
            if (best < 0) break;
            used |= 1u << best;
            const float w = __half2float(wts[best]);
            const float4 v = rowv[best];
            sum.x = __fadd_rn(sum.x, __fmul_rn(v.x, w));
            sum.y = __fadd_rn(sum.y, __fmul_rn(v.y, w));
            sum.z = __fadd_rn(sum.z, __fmul_rn(v.z, w));
            sum.w = __fadd_rn(sum.w, __fmul_rn(v.w, w));
        }
        float* o = g == 0 ? o0 : (g == 1 ? o1 : o2);
        ((float4*) (o + c0))[t] = sum;
    }
}
