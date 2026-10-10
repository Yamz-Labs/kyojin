// Grouped EXL3 MoE prefill, gfx11.5 WMMA (see exl3_moe_prefill.cuh).
//
// Pipeline per MoE layer (8 launches, independent of the expert count):
//   1. metadata: expert offsets, inverse order, compacted (expert, 64-row tile) list
//   2. gather + input Hadamard (gate suh, up suh) into expert-sorted rows
//   3. grouped GEMM gate+up: trellis tiles decoded into LDS inside the GEMM, v_wmma 16x16x16
//   4. output Hadamard (svh) of gate and up, SiLU(g) * u, input Hadamard of down (suh)
//   5. grouped GEMM down (fp32 out)
//   6. output Hadamard of down (svh)
//   7. weighted per-token reduce over top-k (fixed slot order, deterministic)
// The per-row math of 2/4/6 matches LinearEXL3.reconstruct_hgemm (had_r_128 before/after the
// GEMM, fp16 intermediates); only the GEMM accumulation order differs.

#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "hadamard_inner.cuh"
#if defined(USE_ROCM)
#include "../hip/hip_mma.cuh"
#endif
#include "exl3_dq.cuh"
#include "exl3_moe_prefill.cuh"
#include <cstdlib>
#include <type_traits>

#if defined(USE_ROCM)

namespace mpw
{
constexpr int MT = 128;                 // rows per tile (one expert per tile)
constexpr int NT = 128;                 // output columns per block
constexpr int KS = 2;                   // 16-wide k slices per main-loop iteration
constexpr int THREADS = 256;            // 8 waves: 2 (row groups, interleaved) x 4 (cols, 32 each)
constexpr int RG = MT / 32;             // 16-row groups per wave
constexpr int NTILES = NT / 16;
constexpr int A_STRIDE = KS * 16 + 8;   // halves per LDS A row (padded)
constexpr int TILE_SHIFT = 16;          // tile id = expert << 16 | m-tile
constexpr float HAD_SCALE = 0.088388347648f;
}

// ---------------------------------------------------------------------------------------------
// Metadata: exclusive scan of counts, inverse order, tile list

__global__ void mpw_metadata_kernel
(
    const int64_t* expert_count,
    const int64_t* order,
    int64_t* expert_offsets,
    int64_t* inverse_order,
    int* tiles,
    int* num_tiles_out,
    int experts,
    int assignments,
    int mt
)
{
    if (order)
    {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < assignments; idx += blockDim.x * gridDim.x)
        {
            const int64_t original_slot = order[idx];
            if (original_slot >= 0 && original_slot < assignments)
                inverse_order[original_slot] = idx;
        }
    }
    if (blockIdx.x != 0) return;

    __shared__ int64_t warp_off[32];
    __shared__ int warp_chk[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    int64_t run_off = 0;
    int run_chk = 0;
    if (threadIdx.x == 0) expert_offsets[0] = 0;
    for (int base = 0; base < experts; base += blockDim.x)
    {
        const int e = base + threadIdx.x;
        const int64_t count = e < experts ? expert_count[e] : 0;
        const int nch = (int) ((count + mt - 1) / mt);
        int64_t so = count; int sc = nch;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            const int64_t to = __shfl_up_sync(0xffffffffu, so, o);
            const int tc = __shfl_up_sync(0xffffffffu, sc, o);
            if (lane >= o) { so += to; sc += tc; }
        }
        if (lane == 31) { warp_off[warp] = so; warp_chk[warp] = sc; }
        __syncthreads();
        int64_t wbase_o = run_off; int wbase_c = run_chk;
        for (int w = 0; w < warp; ++w) { wbase_o += warp_off[w]; wbase_c += warp_chk[w]; }
        const int64_t excl_o = wbase_o + so - count;
        const int excl_c = wbase_c + sc - nch;
        if (e < experts)
        {
            expert_offsets[e + 1] = excl_o + count;
            for (int c = 0; c < nch; ++c)
                tiles[excl_c + c] = (e << mpw::TILE_SHIFT) | c;
        }
        __syncthreads();
        for (int w = 0; w < nwarps; ++w) { run_off += warp_off[w]; run_chk += warp_chk[w]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) *num_tiles_out = run_chk;
}

// ---------------------------------------------------------------------------------------------
// Warp-level 128-wide Hadamard helpers with explicit scale pointers (scale already offset to the
// 128-block). Same arithmetic as had_hf_r_128_inner / had_ff_r_128_inner.

__device__ __forceinline__ void mpw_had_h(half4& v, const half* pre, const half* post, int t)
{
    if (pre)
    {
        half4 s = ((const half4*) pre)[t];
        v.x = __hmul2(v.x, s.x);
        v.y = __hmul2(v.y, s.y);
    }
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    const float r = mpw::HAD_SCALE;
    v.x = __floats2half2_rn(h0 * r, h1 * r);
    v.y = __floats2half2_rn(h2 * r, h3 * r);
    if (post)
    {
        half4 s = ((const half4*) post)[t];
        v.x = __hmul2(v.x, s.x);
        v.y = __hmul2(v.y, s.y);
    }
}

// Gather token rows into expert-sorted order and apply the input transform of gate and up.
// One wave per (projection, token row, 128-block): the A row is read ONCE and scattered to the
// row's top_k sorted slots (before, each (proj, slot, blk) wave re-read A -> top_k x redundant
// row reads). Per-element arithmetic is unchanged (same A bytes, same per-expert suh).
__global__ __launch_bounds__(256)
void mpw_gather_had_kernel
(
    const half* __restrict__ A,
    half* __restrict__ out,
    const int64_t* __restrict__ selected,
    const int64_t* __restrict__ inverse_order,
    const int64_t* __restrict__ suh_0,
    const int64_t* __restrict__ suh_1,
    int assignments,
    int top_k,
    int width,
    int projections
)
{
    const int blocks = width / 128;
    const int rows = assignments / top_k;
    const int64_t job = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job >= (int64_t) projections * rows * blocks) return;
    const int t = threadIdx.x & 31;
    const int blk = job % blocks;
    const int r = (job / blocks) % rows;
    const int proj = job / ((int64_t) rows * blocks);
    const half4 v0 = ((const half4*) (A + (int64_t) r * width + blk * 128))[t];
    for (int k = 0; k < top_k; ++k)
    {
        const int64_t orig = (int64_t) r * top_k + k;
        const int64_t slot = inverse_order[orig];
        const int64_t e = selected[orig];
        const half* suh = (const half*) (proj ? suh_1 : suh_0)[e];
        half4 v = v0;
        mpw_had_h(v, suh + blk * 128, nullptr, t);
        ((half4*) (out + ((int64_t) proj * assignments + slot) * width + blk * 128))[t] = v;
    }
}

// Output transform of gate and up, SiLU(g) * u, input transform of down. In place into the
// gate half of gu (rows 0..A-1). act_limit > 0 applies the GLM-5.3-Flash clamped SwiGLU,
// min(silu(g), L) * clamp(u, -L, L), bit-for-bit the expression exl3_dec.cu's swiglu_act()
// and activation_kernels.cuh's ACT_SILU use; act_limit == 0 keeps the unclamped form.
__global__ __launch_bounds__(256)
void mpw_act_kernel
(
    half* __restrict__ gu,
    const int64_t* __restrict__ selected,
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ g_svh,
    const int64_t* __restrict__ u_svh,
    const int64_t* __restrict__ d_suh,
    int assignments,
    int width,
    float act_limit
)
{
    const int blocks = width / 128;
    const int64_t job = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job >= (int64_t) assignments * blocks) return;
    const int t = threadIdx.x & 31;
    const int blk = job % blocks;
    const int slot = job / blocks;
    const int64_t e = selected[order[slot]];
    half* gp = gu + (int64_t) slot * width + blk * 128;
    half* up = gu + ((int64_t) assignments + slot) * width + blk * 128;
    half4 g = ((half4*) gp)[t];
    half4 u = ((half4*) up)[t];
    mpw_had_h(g, nullptr, (const half*) g_svh[e] + blk * 128, t);
    mpw_had_h(u, nullptr, (const half*) u_svh[e] + blk * 128, t);
    auto act = [] (half x) -> half
    {
        const float f = __half2float(x);
        return __float2half_rn(f / (1.0f + expf(-f)));
    };
    // Unclamped: exactly the half2 multiply the pre-act_limit kernel did. Clamped: the fp16
    // min/max order of activation_kernels.cuh's ACT_SILU / exl3_dec.cu's swiglu_act
    const bool clamp = act_limit != 0.0f;
    const half lim = __float2half_rn(act_limit);
    const half nlim = __float2half_rn(-act_limit);
    auto swiglu = [&] (half2 u2, half g0, half g1) -> half2
    {
        half s0 = act(g0), s1 = act(g1);
        if (clamp)
        {
            s0 = __hmin(s0, lim);
            s1 = __hmin(s1, lim);
            u2 = __halves2half2(
                __hmin(__hmax(__low2half(u2), nlim), lim),
                __hmin(__hmax(__high2half(u2), nlim), lim));
        }
        return __hmul2(__halves2half2(s0, s1), u2);
    };
    half4 a;
    a.x = swiglu(u.x, __low2half(g.x), __high2half(g.x));
    a.y = swiglu(u.y, __low2half(g.y), __high2half(g.y));
    mpw_had_h(a, (const half*) d_suh[e] + blk * 128, nullptr, t);
    ((half4*) gp)[t] = a;
}

// Down output transform fused into the reduce: output[token, blk] = sum_k w_k * (H128(d_k) * svh_e_k),
// d_k = fp32 down GEMM row of the token's k-th pick, w_k = fp32 routing weight: the weighted
// combine runs entirely in fp32. One wave per (token, 128-block), k ascending.
__global__ __launch_bounds__(256)
void mpw_reduce_had_kernel
(
    const float* __restrict__ sorted_rows,
    const int64_t* __restrict__ selected,
    const float* __restrict__ weights,
    const int64_t* __restrict__ inverse_order,
    const int64_t* __restrict__ svh_table,
    float* __restrict__ output,
    int rows,
    int top_k,
    int width
)
{
    const int blocks = width / 128;
    const int64_t job = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job >= (int64_t) rows * blocks) return;
    const int t = threadIdx.x & 31;
    const int blk = job % blocks;
    const int64_t row = job / blocks;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < top_k; ++k)
    {
        const int64_t slot = inverse_order[row * top_k + k];
        const int64_t e = selected[row * top_k + k];
        const float w = weights[row * top_k + k];
        const float4 fv = ((const float4*) (sorted_rows + slot * width + blk * 128))[t];
        float v0 = fv.x, v1 = fv.y, v2 = fv.z, v3 = fv.w;
        float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
        float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
        shuffle_had_f4x32(h0, h1, h2, h3, t);
        const half4 sc = ((const half4*) ((const half*) svh_table[e] + blk * 128))[t];
        const float r = mpw::HAD_SCALE * w;
        sum.x += h0 * r * __low2float(sc.x);
        sum.y += h1 * r * __high2float(sc.x);
        sum.z += h2 * r * __low2float(sc.y);
        sum.w += h3 * r * __high2float(sc.y);
    }
    ((float4*) (output + row * width + blk * 128))[t] = sum;
}

// ---------------------------------------------------------------------------------------------
// Glue v2 (EXL3_MPW_GLUE=1, default off). Same per-element arithmetic as the kernels above, same
// output bytes. Two changes: (1) the top_k index chain (inverse_order -> selected -> table pointer
// -> data) is fetched ONCE per wave, lane k holding pick k, and broadcast with shuffles, instead of
// a serial dependent load chain per pick; (2) the picks are unrolled so loads of different picks
// are in flight together. Gather also serves gate and up from one wave (shares the A row and the
// index chain). Needs top_k <= 32 (host check) and projections == 2.
#define MPW_GLUE_TK_MAX 32

__device__ __forceinline__ int64_t mpw_bcast64(int64_t v, int k)
{
    const int lo = __shfl((int) (uint32_t) v, k, 32);
    const int hi = __shfl((int) (uint32_t) ((uint64_t) v >> 32), k, 32);
    return (int64_t) (((uint64_t) (uint32_t) hi << 32) | (uint64_t) (uint32_t) lo);
}

__global__ __launch_bounds__(256)
void mpw_gather_had2_kernel
(
    const half* __restrict__ A,
    half* __restrict__ out,
    const int64_t* __restrict__ selected,
    const int64_t* __restrict__ inverse_order,
    const int64_t* __restrict__ suh_0,
    const int64_t* __restrict__ suh_1,
    int assignments,
    int top_k,
    int width
)
{
    const int blocks = width / 128;
    const int rows = assignments / top_k;
    const int64_t job = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job >= (int64_t) rows * blocks) return;
    const int t = threadIdx.x & 31;
    const int blk = job % blocks;
    const int64_t r = job / blocks;
    int64_t slot_l = 0, p0_l = 0, p1_l = 0;
    if (t < top_k)
    {
        const int64_t orig = r * top_k + t;
        slot_l = inverse_order[orig];
        const int64_t e = selected[orig];
        p0_l = suh_0[e];
        p1_l = suh_1[e];
    }
    const half4 v0 = ((const half4*) (A + r * width + blk * 128))[t];
    #pragma unroll 5
    for (int k = 0; k < top_k; ++k)
    {
        const int64_t slot = mpw_bcast64(slot_l, k);
        const half* s0 = (const half*) mpw_bcast64(p0_l, k);
        const half* s1 = (const half*) mpw_bcast64(p1_l, k);
        half4 va = v0, vb = v0;
        mpw_had_h(va, s0 + blk * 128, nullptr, t);
        mpw_had_h(vb, s1 + blk * 128, nullptr, t);
        ((half4*) (out + (int64_t) slot * width + blk * 128))[t] = va;
        ((half4*) (out + ((int64_t) assignments + slot) * width + blk * 128))[t] = vb;
    }
}

template <int TK>
__global__ __launch_bounds__(256)
void mpw_reduce_had2_kernel
(
    const float* __restrict__ sorted_rows,
    const int64_t* __restrict__ selected,
    const float* __restrict__ weights,
    const int64_t* __restrict__ inverse_order,
    const int64_t* __restrict__ svh_table,
    float* __restrict__ output,
    int rows,
    int width
)
{
    const int blocks = width / 128;
    const int64_t job = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job >= (int64_t) rows * blocks) return;
    const int t = threadIdx.x & 31;
    const int blk = job % blocks;
    const int64_t row = job / blocks;
    int64_t slot_l = 0, p_l = 0;
    float w_l = 0.f;
    if (t < TK)
    {
        const int64_t orig = row * TK + t;
        slot_l = inverse_order[orig];
        p_l = svh_table[selected[orig]];
        w_l = weights[orig];
    }
    float4 fv[TK];
    #pragma unroll
    for (int k = 0; k < TK; ++k)
    {
        const int64_t slot = mpw_bcast64(slot_l, k);
        fv[k] = ((const float4*) (sorted_rows + slot * width + blk * 128))[t];
    }
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int k = 0; k < TK; ++k)
    {
        const float w = __shfl(w_l, k, 32);
        const half* svh = (const half*) mpw_bcast64(p_l, k);
        float v0 = fv[k].x, v1 = fv[k].y, v2 = fv[k].z, v3 = fv[k].w;
        float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
        float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
        shuffle_had_f4x32(h0, h1, h2, h3, t);
        const half4 sc = ((const half4*) (svh + blk * 128))[t];
        const float r = mpw::HAD_SCALE * w;
        sum.x += h0 * r * __low2float(sc.x);
        sum.y += h1 * r * __high2float(sc.x);
        sum.z += h2 * r * __low2float(sc.y);
        sum.w += h3 * r * __high2float(sc.y);
    }
    ((float4*) (output + row * width + blk * 128))[t] = sum;
}

// ---------------------------------------------------------------------------------------------
// Grouped GEMM: blockIdx.x = 128-column block, blockIdx.y = tile slot, blockIdx.z = projection

// Fragment loads (LDM): 0 = each lane reads its full 16-k row (lanes 16-31 re-read lanes 0-15's
// bytes: half the LDS read traffic is duplicate); 1 = each lane reads one 8-k half (lanes 0-15 k 0-7,
// lanes 16-31 k 8-15) and v_permlanex16 fetches the other half from lane ^ 16. Lanes 16-31 then hold
// k 8-15 in the low 4 dwords and k 0-7 in the high 4, for A and B alike: v_wmma_f32_16x16x16_f16 on
// gfx1151 computes the odd output rows from the upper half-wave's own A/B, and a k permutation applied
// to both operands leaves the dot product unchanged (scratch/probe_lanes.cu, mode up-kswap).
#if defined(EXL3_HIP_WMMA_GFX115)
__device__ __forceinline__ HipFp16x16 mpw_frag_x16(const half* p)
{
    const uint4 o = *((const uint4*) p);
    uint4 q;
    q.x = __builtin_amdgcn_permlanex16(o.x, o.x, 0x76543210, 0xfedcba98, false, false);
    q.y = __builtin_amdgcn_permlanex16(o.y, o.y, 0x76543210, 0xfedcba98, false, false);
    q.z = __builtin_amdgcn_permlanex16(o.z, o.z, 0x76543210, 0xfedcba98, false, false);
    q.w = __builtin_amdgcn_permlanex16(o.w, o.w, 0x76543210, 0xfedcba98, false, false);
    HipFp16x16 f;
    ((uint4*) &f)[0] = o;
    ((uint4*) &f)[1] = q;
    return f;
}
#endif  // EXL3_HIP_WMMA_GFX115

template <int bits, bool HALF, bool OUT_FP32, int DBG = 0, int PFD = 2, int CB = 2, int LDM = 0>
__global__ __launch_bounds__(mpw::THREADS)
void mpw_gemm_kernel
(
    const half* __restrict__ A,
    int64_t a_proj_stride,
    const int64_t* __restrict__ expert_offsets,
    const int* __restrict__ tiles,
    const int* __restrict__ num_tiles,
    const int64_t* __restrict__ B_table_0,
    const int64_t* __restrict__ B_table_1,
    void* __restrict__ C,
    int64_t c_proj_stride,
    int size_k,
    int size_n,
    int a_ld                        // A row pitch in elements (>= size_k); the old mpw kernel needs a_ld == size_k
)
{
#if defined(EXL3_HIP_WMMA_GFX115)
    using namespace mpw;
    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;
    constexpr int SEG = NTILES * TWORDS;
    constexpr int WORDS = KS * SEG;
    constexpr int WLOADS = (WORDS + THREADS - 1) / THREADS;
    constexpr int ALOADS = MT * KS * 16 / 8 / THREADS;
    static_assert(ALOADS * THREADS * 8 == MT * KS * 16, "whole 16-byte A loads");

    if (blockIdx.y >= *num_tiles) return;
    const int tile = tiles[blockIdx.y];
    const int expert = tile >> TILE_SHIFT;
    const int mtile = tile & ((1 << TILE_SHIFT) - 1);
    const int64_t row0 = expert_offsets[expert] + (int64_t) mtile * MT;
    const int rows = (int) min((int64_t) MT, expert_offsets[expert + 1] - row0);
    const int proj = blockIdx.z;
    const uint32_t* B32 = (const uint32_t*) (proj ? B_table_1 : B_table_0)[expert];
    A += proj * a_proj_stride + row0 * size_k;

    const int n0 = blockIdx.x * NT;
    const size_t slice_stride = (size_t) (size_n / 16) * TWORDS;
    const uint32_t* Bblk = B32 + (size_t) (n0 / 16) * TWORDS;

    // Double-buffered LDS, one barrier per iteration (KS k-slices):
    //   iteration it: barrier; store A(it+1) -> sh_a[(it+1)&1], W(it+2) -> sh_w[it&1];
    //   issue global loads A(it+2), W(it+3); WMMA on sh_a[it&1] x sh_b[it&1];
    //   decode sh_w[(it+1)&1] -> sh_b[(it+1)&1]
    __shared__ __align__(16) half sh_a[2][MT * A_STRIDE];
    __shared__ __align__(16) uint32_t sh_w[2][WORDS];
    __shared__ __align__(16) half sh_b[2][KS * NTILES * 256];

    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int iters = size_k / (KS * 16);

    // Global -> register prefetch
    const int a_r = t >> 2;             // + i * (THREADS / 4)
    const int a_c = (t & 3) * 8;
    const half* a_src = A + (int64_t) a_r * size_k + a_c;
    // Register ring: set r holds P(j) = {A(j), W(j + 1)} for j & 1 == r (PFD == 2), loaded two
    // iterations before it is stored to LDS; PFD == 1 keeps the single set (one iteration)
    uint4 ra_[2][ALOADS];
    uint32_t rw_[2][WLOADS];
    auto load_a = [&] (int it, auto set)
    {
        uint4* ra = ra_[decltype(set)::value];
        const int k0 = it * KS * 16;
        #pragma unroll
        for (int i = 0; i < ALOADS; ++i)
            ra[i] = a_r + i * (THREADS / 4) < rows
                ? *((const uint4*) (a_src + (int64_t) i * (THREADS / 4) * size_k + k0))
                : make_uint4(0, 0, 0, 0);
    };
    auto load_w = [&] (int it, auto set)
    {
        uint32_t* rw = rw_[decltype(set)::value];
        const uint32_t* bsrc = Bblk + (size_t) (it * KS) * slice_stride;
        #pragma unroll
        for (int i = 0; i < WLOADS; ++i)
        {
            const int idx = i * THREADS + t;
            if (WORDS % THREADS == 0 || idx < WORDS)
            {
                const int s = idx / SEG;
                const int w = idx - s * SEG;
                rw[i] = bsrc[(size_t) s * slice_stride + w];
            }
        }
    };
    auto store_a = [&] (int buf, auto set)
    {
        const uint4* ra = ra_[decltype(set)::value];
        #pragma unroll
        for (int i = 0; i < ALOADS; ++i)
            *((uint4*) (sh_a[buf] + (a_r + i * (THREADS / 4)) * A_STRIDE + a_c)) = ra[i];
    };
    auto store_w = [&] (int buf, auto set)
    {
        const uint32_t* rw = rw_[decltype(set)::value];
        #pragma unroll
        for (int i = 0; i < WLOADS; ++i)
        {
            const int idx = i * THREADS + t;
            if (WORDS % THREADS == 0 || idx < WORDS) sh_w[buf][idx] = rw[i];
        }
    };
    auto decode = [&] (int buf)
    {
        #pragma unroll
        for (int j = 0; j < KS * NTILES / 8; ++j)
        {
            const int tidx = warp * (KS * NTILES / 8) + j;
            const int s = tidx / NTILES;
            const int nt = tidx % NTILES;
            FragB f0, f1;
            if constexpr (DBG == 1)
            {
                f0[0] = f0[1] = f1[0] = f1[1] = *((const half2*) (sh_w[buf] + s * SEG + nt * TWORDS + (lane & 15)));
            }
            else
                dq_dispatch<bits, CB, HALF>(sh_w[buf] + s * SEG + nt * TWORDS, lane * 8, f0, f1);
            // CUDA m16n8k16 B fragment: col g = lane >> 2 (f0) and g + 8 (f1),
            // k = 2q, 2q+1 (elem 0) and 2q+8, 2q+9 (elem 1), q = lane & 3.
            // LDS layout: [s][nt][col][k]
            half* bt = sh_b[buf] + tidx * 256;
            const int g = lane >> 2;
            const int q = (lane & 3) * 2;
            *((half2*) (bt + g * 16 + q)) = f0[0];
            *((half2*) (bt + g * 16 + q + 8)) = f0[1];
            *((half2*) (bt + (g + 8) * 16 + q)) = f1[0];
            *((half2*) (bt + (g + 8) * 16 + q + 8)) = f1[1];
        }
    };

    const int wm = warp >> 2;           // row groups wm, wm + 2, wm + 4, ... (interleaved)
    const int wn = warp & 3;            // column quarter: n-tiles wn*2, wn*2+1
    bool g_act[RG];
    #pragma unroll
    for (int i = 0; i < RG; ++i) g_act[i] = (wm + 2 * i) * 16 < rows;

    HipFp32x8 acc[RG][2];
    #pragma unroll
    for (int i = 0; i < RG; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.0f;

    using S0 = std::integral_constant<int, 0>;
    using S1 = std::integral_constant<int, 1>;

    auto mma = [&] (int cur)
    {
        #pragma unroll
        for (int s = 0; s < KS; ++s)
        {
            HipFp16x16 bf[2];
            #pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                const half* p = sh_b[cur] + (s * NTILES + wn * 2 + j) * 256 + (lane & 15) * 16;
                if constexpr (LDM == 1) bf[j] = mpw_frag_x16(p + (lane >> 4) * 8);
                else bf[j] = *((const HipFp16x16*) p);
            }
            #pragma unroll
            for (int i = 0; i < RG; ++i)
            {
                if (!g_act[i]) continue;
                const half* p = sh_a[cur] + ((wm + 2 * i) * 16 + (lane & 15)) * A_STRIDE + s * 16;
                HipFp16x16 af;
                if constexpr (LDM == 1) af = mpw_frag_x16(p + (lane >> 4) * 8);
                else
                {
                    ((uint4*) &af)[0] = ((const uint4*) p)[0];
                    ((uint4*) &af)[1] = ((const uint4*) p)[1];
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                {
                    if constexpr (DBG == 2)
                        acc[i][j][0] += (float) af[0] * (float) bf[j][1];
                    else
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf[j], acc[i][j]);
                }
            }
        }
    };

    if constexpr (PFD == 1)
    {
        // Prologue: A(0), W(0), W(1) in LDS, W(0) decoded, A(1) and W(2) in flight
        load_a(0, S0{}); load_w(0, S0{});
        store_a(0, S0{}); store_w(0, S0{});
        if (iters > 1) { load_w(1, S0{}); store_w(1, S0{}); }
        __syncthreads();
        decode(0);
        if (iters > 1) load_a(1, S0{});
        if (iters > 2) load_w(2, S0{});

        for (int it = 0; it < iters; ++it)
        {
            const int cur = it & 1;
            __syncthreads();
            if (it + 1 < iters) store_a(cur ^ 1, S0{});
            if (it + 2 < iters) store_w(cur, S0{});
            if (it + 2 < iters) load_a(it + 2, S0{});
            if (it + 3 < iters) load_w(it + 3, S0{});
            mma(cur);
            if (it + 1 < iters) decode(cur ^ 1);
        }
    }
    else
    {
        // Prologue: W(0), A(0), W(1) in LDS, W(0) decoded, P(1) -> set 1, P(2) -> set 0
        load_a(0, S0{}); load_w(0, S0{});
        store_a(0, S0{}); store_w(0, S0{});
        if (iters > 1) { load_w(1, S1{}); store_w(1, S1{}); }
        __syncthreads();
        decode(0);
        if (iters > 1) { load_a(1, S1{}); if (iters > 2) load_w(2, S1{}); }
        if (iters > 2) { load_a(2, S0{}); if (iters > 3) load_w(3, S0{}); }

        // LDS-only barrier: __syncthreads() would drain vmcnt and with it the two iterations
        // of global prefetch in flight
        #define MPW_LDS_BARRIER asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
        auto step = [&] (int it, auto set)     // set = (it + 1) & 1
        {
            const int cur = it & 1;
            MPW_LDS_BARRIER;
            if (it + 1 < iters) store_a(cur ^ 1, set);
            if (it + 2 < iters) store_w(cur, set);
            if (it + 3 < iters) { load_a(it + 3, set); if (it + 4 < iters) load_w(it + 4, set); }
            mma(cur);
            if (it + 1 < iters) decode(cur ^ 1);
        };
        int it = 0;
        for (; it + 1 < iters; it += 2)
        {
            step(it, S1{});
            step(it + 1, S0{});
        }
        if (it < iters) step(it, S1{});
        #undef MPW_LDS_BARRIER
    }

    // Epilogue: C(lane, r) = C[row = 2r + (lane >> 4)][col = lane & 15]
    #pragma unroll
    for (int i = 0; i < RG; ++i)
    {
        if (!g_act[i]) continue;
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const int col = n0 + (wn * 2 + j) * 16 + (lane & 15);
            #pragma unroll
            for (int r = 0; r < 8; ++r)
            {
                const int row = (wm + 2 * i) * 16 + 2 * r + (lane >> 4);
                if (row >= rows) continue;
                const int64_t off = proj * c_proj_stride + (row0 + row) * size_n + col;
                if constexpr (OUT_FP32)
                    ((float*) C)[off] = acc[i][j][r];
                else
                    ((half*) C)[off] = __float2half_rn(acc[i][j][r]);
            }
        }
    }
#endif
}

// ---------------------------------------------------------------------------------------------
// mpw2 grouped GEMM (MPW_KERN=2, default). Measured on mpw_gemm (gemm_bench, HANDOFF step 7): a fixed
// per-tile cost of ~7.7 ms per 4096x2048 launch at 8 rows/expert and no overlap of WMMA with the rest.
// On gfx1151 WMMA, trellis-decode VALU and permlanes share the SIMD, so time >= WMMA + decode: the decode
// alone is ~3 ms per 4096x2048 call (no-decode ablation), paid once per m-tile. mpw2 changes the data flow:
//   - each wave owns NW adjacent 16-column n-tiles for all MT rows: it decodes its own B tiles into a
//     wave-private LDS stage and reads them back as WMMA fragments (no block barrier guards B). NW = 2
//     halves the A fragment reads per WMMA;
//   - A is the only block-shared operand: double-buffered, unpadded, 16-byte chunks XOR-swizzled
//     (conflict-free for the chunk stores and the fragment reads);
//   - PF iterations of global prefetch for A and W in registers (clamped addresses: one uniform
//     load stream, so vmcnt waits stay counted).
// Default MT 128, NW 2 (241-256 VGPR, 34-37 KB LDS, no spills). MT 64 NW 2 wins on uniform <= 64
// rows/expert but loses on skewed routing (experts of 65-128 rows decode twice); both bit-exact vs mpw_gemm.
// Per iteration and wave: KS x (NW B fragments + up to RG A fragments, RG * NW WMMAs), RG = MT / 16.

namespace mpw2
{
constexpr int NT = 128;                 // 8 waves x one 16-column n-tile
constexpr int KS = 2;                   // 16-wide k slices per iteration (one block barrier)
constexpr int THREADS = 256;
constexpr int WAVES = THREADS / 32;
}

// A tile in LDS: row r = 32 halves = 4 chunks of 16 bytes, chunk c stored at slot c ^ ((r >> 1) & 3).
// 8 lanes reading the same chunk of rows r..r+7 then hit 8 distinct 16-byte bank groups.
__device__ __forceinline__ int mpw2_a_off(int r, int c)
{
    return r * 32 + ((c ^ ((r >> 1) & 3)) << 3);
}

#if defined(EXL3_CB_HAVE_SAD)
// x * C mod 2^32 for x < 2^16 as two full-rate v_mul_u32_u24 and one v_lshl_add_u32. codebook.cuh's
// mul_const_w16 writes the same split in C, but LLVM folds it back into one quarter-rate v_mul_lo_u32
// (seen in the mpw2 ISA, scratch/step8); the asm keeps the split.
template <uint32_t C>
__device__ __forceinline__ uint32_t mpw2_mul_w16(uint32_t x)
{
    uint32_t lo, hi;
    asm("v_mul_u32_u24 %0, %1, %2" : "=v"(lo) : "s"(C & 0xffffu), "v"(x));
    asm("v_mul_u32_u24 %0, %1, %2" : "=v"(hi) : "s"(C >> 16), "v"(x));
    return lo + (hi << 16);
}

// dq8_aligned_2bits (exl3_dq.cuh) for CB 2 (mul1) with mpw2_mul_w16: same words, same bits
__device__ __forceinline__ void mpw2_dq8_2bits_mul1(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i1 = t_offset >> 4;
    uint32_t i0 = (i1 + 15) & 15;
    uint32_t b = fshift(ptr[i1], ptr[i0], ((~t_offset) & 8) << 1);
    uint32_t w[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) w[j] = mpw2_mul_w16<0x83DCD12Du>((b >> (14 - 2 * j)) & 0xffffu);
    frag0[0] = decode_mul1_product_2_sad(w[0], w[1]);
    frag0[1] = decode_mul1_product_2_sad(w[2], w[3]);
    frag1[0] = decode_mul1_product_2_sad(w[4], w[5]);
    frag1[1] = decode_mul1_product_2_sad(w[6], w[7]);
}
#endif

// DBG (ablation, bench only): bit 0 = no decode (raw words as B), bit 1 = no WMMA, bit 2 = no weight loads
// OPT (MPW_OPT, bit-exact, default 2): bit 0 = B fragments as two direct LDS reads in the LDM=1 k order
// (lanes 16-31 read k 8-15 first), no permlanes; bit 1 = same for A; bit 2 = 2-bit mul1 decode with
// forced 24-bit multiplies (mpw2_dq8_2bits_mul1)
template <int bits, bool HALF, bool OUT_FP32, int CB, int LDM, int MT, int PF, int NW, int DBG = 0, int OPT = 0>
__global__ __launch_bounds__(mpw2::THREADS)
void mpw2_gemm_kernel
(
    const half* __restrict__ A,
    int64_t a_proj_stride,
    const int64_t* __restrict__ expert_offsets,
    const int* __restrict__ tiles,
    const int* __restrict__ num_tiles,
    const int64_t* __restrict__ B_table_0,
    const int64_t* __restrict__ B_table_1,
    void* __restrict__ C,
    int64_t c_proj_stride,
    int size_k,
    int size_n,
    int a_ld                        // A row pitch in elements (>= size_k); the old mpw kernel needs a_ld == size_k
)
{
#if defined(EXL3_HIP_WMMA_GFX115)
    constexpr int KS = mpw2::KS;
    constexpr int NT = mpw2::WAVES * 16 * NW;
    constexpr int THREADS = mpw2::THREADS;
    constexpr int RG = MT / 16;
    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;
    constexpr int WW = KS * NW * TWORDS;            // trellis words per wave per iteration
    constexpr int WL = (WW + 31) / 32;              // W loads per lane per iteration
    constexpr int ALOADS = MT * 4 / THREADS;        // 16-byte A chunks per thread per iteration
    static_assert(ALOADS * THREADS == MT * 4, "whole A passes");
    static_assert(PF >= 1 && PF <= 4, "prefetch depth");

    if (blockIdx.y >= *num_tiles) return;
    const int tile = tiles[blockIdx.y];
    const int expert = tile >> mpw::TILE_SHIFT;
    const int mtile = tile & ((1 << mpw::TILE_SHIFT) - 1);
    const int64_t row0 = expert_offsets[expert] + (int64_t) mtile * MT;
    const int rows = (int) min((int64_t) MT, expert_offsets[expert + 1] - row0);
    const int proj = blockIdx.z;
    const uint32_t* B32 = (const uint32_t*) (proj ? B_table_1 : B_table_0)[expert];
    A += proj * a_proj_stride + row0 * a_ld;

    const int n0 = blockIdx.x * NT;
    const size_t slice_stride = (size_t) (size_n / 16) * TWORDS;

    __shared__ __align__(16) half sh_a[2][MT * 32];
    __shared__ __align__(16) uint32_t sh_w[mpw2::WAVES][WW];
    __shared__ __align__(16) half sh_b[mpw2::WAVES][KS * NW * 256];

    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int iters = size_k / (KS * 16);
    const int nrg = (rows + 15) >> 4;               // active 16-row groups (block-uniform)

    const int a_r = t >> 2;                         // + i * (THREADS / 4)
    const int a_c = t & 3;                          // 16-byte chunk (8 k)
    const half* a_src = A + (int64_t) a_r * a_ld + a_c * 8;
    // Tail block (size_n not a multiple of NT): waves past the last n-tile read the last valid NW tiles
    // (duplicate, never stored: the epilogue skips them), so no trellis read leaves the tensor.
    const int ntile_n = size_n / 16;
    const int first_tile = min(n0 / 16 + warp * NW, ntile_n - NW);
    const uint32_t* w_src = B32 + (size_t) first_tile * TWORDS;   // NW adjacent tiles: contiguous

    uint4 ra[PF][ALOADS];
    uint32_t rw[PF][WL];

    // Loads past the last iteration re-read the last one (never stored): the load stream stays uniform
    auto load_a = [&] (int it, uint4* r)
    {
        const int k0 = min(it, iters - 1) * KS * 16;
        #pragma unroll
        for (int i = 0; i < ALOADS; ++i)
            r[i] = a_r + i * (THREADS / 4) < rows
                ? *((const uint4*) (a_src + (int64_t) i * (THREADS / 4) * a_ld + k0))
                : make_uint4(0, 0, 0, 0);
    };
    auto load_w = [&] (int it, uint32_t* r)
    {
        const uint32_t* src = w_src + (size_t) (min(it, iters - 1) * KS) * slice_stride;
        #pragma unroll
        for (int i = 0; i < WL; ++i)
        {
            const int idx = i * 32 + lane;
            if constexpr (DBG & 4) r[i] = idx * 0x9e3779b9u + it;
            else if (WW % 32 == 0 || idx < WW)
            {
                const int s = idx / (NW * TWORDS);
                r[i] = src[(size_t) s * slice_stride + (idx - s * NW * TWORDS)];
            }
        }
    };
    auto store_a = [&] (int buf, const uint4* r)
    {
        #pragma unroll
        for (int i = 0; i < ALOADS; ++i)
            *((uint4*) (sh_a[buf] + mpw2_a_off(a_r + i * (THREADS / 4), a_c))) = r[i];
    };
    auto store_w = [&] (const uint32_t* r)
    {
        #pragma unroll
        for (int i = 0; i < WL; ++i)
        {
            const int idx = i * 32 + lane;
            if (WW % 32 == 0 || idx < WW) sh_w[warp][idx] = r[i];
        }
    };
    // Decode this wave's KS x NW tiles through its private stage. LDS ops of one wave execute in order,
    // so the cross-lane RAW (store -> fragment read) and WAR (next iteration's stores) need no barrier.
    auto decode = [&] (HipFp16x16* bf)
    {
        #pragma unroll
        for (int s = 0; s < KS * NW; ++s)           // tile s = slice s / NW, n-tile s % NW
        {
            FragB f0, f1;
            if constexpr (DBG & 1)
                f0[0] = f0[1] = f1[0] = f1[1] = *((const half2*) (sh_w[warp] + s * TWORDS + (lane & 15)));
#if defined(EXL3_CB_HAVE_SAD)
            else if constexpr ((OPT & 4) && bits == 2 && !HALF && CB == 2)
                mpw2_dq8_2bits_mul1(sh_w[warp] + s * TWORDS, lane * 8, f0, f1);
#endif
            else
                dq_dispatch<bits, CB, HALF>(sh_w[warp] + s * TWORDS, lane * 8, f0, f1);
            // CUDA m16n8k16 B fragment: col g = lane >> 2 (f0) and g + 8 (f1), k = 2q, 2q+1 and
            // 2q+8, 2q+9, q = lane & 3. Stage layout [s][col][k]
            half* bt = sh_b[warp] + s * 256;
            const int g = lane >> 2;
            const int q = (lane & 3) * 2;
            *((half2*) (bt + g * 16 + q)) = f0[0];
            *((half2*) (bt + g * 16 + q + 8)) = f0[1];
            *((half2*) (bt + (g + 8) * 16 + q)) = f1[0];
            *((half2*) (bt + (g + 8) * 16 + q + 8)) = f1[1];
        }
        __builtin_amdgcn_wave_barrier();
        #pragma unroll
        for (int s = 0; s < KS * NW; ++s)
        {
            const half* p = sh_b[warp] + s * 256 + (lane & 15) * 16;
            if constexpr (OPT & 1)
            {
                ((uint4*) &bf[s])[0] = *((const uint4*) (p + (lane >> 4) * 8));
                ((uint4*) &bf[s])[1] = *((const uint4*) (p + ((lane >> 4) ^ 1) * 8));
            }
            else if constexpr (LDM == 1) bf[s] = mpw_frag_x16(p + (lane >> 4) * 8);
            else bf[s] = *((const HipFp16x16*) p);
        }
    };

    HipFp32x8 acc[RG][NW];
    #pragma unroll
    for (int i = 0; i < RG; ++i)
        #pragma unroll
        for (int j = 0; j < NW; ++j)
            #pragma unroll
            for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.0f;

    auto mma = [&] (int cur, const HipFp16x16* bf)
    {
        #pragma unroll
        for (int s = 0; s < KS; ++s)
        {
            #pragma unroll
            for (int i = 0; i < RG; ++i)
            {
                if (i >= nrg) continue;
                const int r = i * 16 + (lane & 15);
                HipFp16x16 af;
                if constexpr (DBG & 8) af = bf[s * NW];     // bench only: A operand from registers (no LDS read)
                else if constexpr (OPT & 2)
                {
                    ((uint4*) &af)[0] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2 + (lane >> 4))));
                    ((uint4*) &af)[1] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2 + ((lane >> 4) ^ 1))));
                }
                else if constexpr (LDM == 1) af = mpw_frag_x16(sh_a[cur] + mpw2_a_off(r, s * 2 + (lane >> 4)));
                else
                {
                    ((uint4*) &af)[0] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2)));
                    ((uint4*) &af)[1] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2 + 1)));
                }
                #pragma unroll
                for (int j = 0; j < NW; ++j)
                {
                    if constexpr (DBG & 2) acc[i][j][0] += (float) af[0] + (float) bf[s * NW + j][lane & 15];
                    else acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf[s * NW + j], acc[i][j]);
                }
            }
        }
    };

    // Ring: set j % PF holds A(j) and W(j) until consumed
    #pragma unroll
    for (int j = 0; j < PF; ++j) { load_a(j, ra[j]); load_w(j, rw[j]); }
    store_a(0, ra[0]);
    load_a(PF, ra[0]);
    __syncthreads();

    #define MPW_LDS_BARRIER asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
    auto step = [&] (int it, auto ic)               // ic = it % PF
    {
        constexpr int ws = decltype(ic)::value;
        constexpr int as = (ws + 1) % PF;
        MPW_LDS_BARRIER;                            // sh_a[(it + 1) & 1] free, A(it) visible
        store_w(rw[ws]);
        load_w(it + PF, rw[ws]);
        if (it + 1 < iters) store_a((it + 1) & 1, ra[as]);
        load_a(it + 1 + PF, ra[as]);
        if constexpr (OPT & 256)
        {
            // OPT bit 8: waves whose n-tiles all lie past size_n (tail block) skip decode and WMMA: their
            // results are never stored (the epilogue skips them), so the output is unchanged. Wave-uniform.
            if (n0 + warp * NW * 16 >= size_n) return;
        }
        HipFp16x16 bf[KS * NW];
        decode(bf);
        mma(it & 1, bf);
    };
    using I0 = std::integral_constant<int, 0>;
    using I1 = std::integral_constant<int, 1 % PF>;
    using I2 = std::integral_constant<int, 2 % PF>;
    using I3 = std::integral_constant<int, 3 % PF>;
    int it = 0;
    for (; it + PF <= iters; it += PF)
    {
        step(it, I0{});
        if constexpr (PF > 1) step(it + 1, I1{});
        if constexpr (PF > 2) step(it + 2, I2{});
        if constexpr (PF > 3) step(it + 3, I3{});
    }
    if constexpr (PF > 1) if (it < iters) step(it, I0{});
    if constexpr (PF > 2) if (it + 1 < iters) step(it + 1, I1{});
    if constexpr (PF > 3) if (it + 2 < iters) step(it + 2, I2{});
    #undef MPW_LDS_BARRIER

    // Epilogue: C(lane, r) = C[row = 2r + (lane >> 4)][col = lane & 15]
    #pragma unroll
    for (int i = 0; i < RG; ++i)
    {
        if (i >= nrg) continue;
        #pragma unroll
        for (int j = 0; j < NW; ++j)
        {
            if (n0 + (warp * NW + j) * 16 >= size_n) continue;      // tail block: tile past size_n (wave-uniform)
            const int col = n0 + (warp * NW + j) * 16 + (lane & 15);
            #pragma unroll
            for (int r = 0; r < 8; ++r)
            {
                const int row = i * 16 + 2 * r + (lane >> 4);
                if (row >= rows) continue;
                const int64_t off = proj * c_proj_stride + (row0 + row) * size_n + col;
                if constexpr (OUT_FP32)
                    ((float*) C)[off] = acc[i][j][r];
                else
                    ((half*) C)[off] = __float2half_rn(acc[i][j][r]);
            }
        }
    }
#endif
}

// ---------------------------------------------------------------------------------------------
// mpw2x (EXL3_MPW2X, default off): mpw2 with the same math and summation order (bit-exact), plus
//   OPT bit 3 (8): trellis words read through a global (addrspace 1) pointer. The table-loaded B pointer
//     is generic in mpw2, so its W prefetch compiles to flat loads, which also count on lgkmcnt: every
//     LDS wait (lgkmcnt(0)) in the loop then drains the whole prefetch and exposes DRAM latency per step;
//   OPT bit 4 (16): A fragments software-pipelined one row group ahead;
//   OPT bit 5 (32): K loop specialized on the block-uniform row-group count 6/7/8 (branch-free chain);
//   OPT bit 7 (128): with bit 4, a scheduling barrier per row-group step (caps live A fragments at two).
// The K loop runs through always-inlined lambdas (ring/step/mma): different register allocation than
// mpw2 (224 vs 256 VGPR, no spill at PF 2), which is why this is a separate kernel and mpw2 is untouched.

template <int bits, bool HALF, bool OUT_FP32, int CB, int LDM, int MT, int PF, int NW, int DBG = 0, int OPT = 0>
__global__ __launch_bounds__(mpw2::THREADS)
void mpw2x_gemm_kernel
(
    const half* __restrict__ A,
    int64_t a_proj_stride,
    const int64_t* __restrict__ expert_offsets,
    const int* __restrict__ tiles,
    const int* __restrict__ num_tiles,
    const int64_t* __restrict__ B_table_0,
    const int64_t* __restrict__ B_table_1,
    void* __restrict__ C,
    int64_t c_proj_stride,
    int size_k,
    int size_n,
    int a_ld                        // A row pitch in elements (>= size_k); the old mpw kernel needs a_ld == size_k
)
{
#if defined(EXL3_HIP_WMMA_GFX115)
    constexpr int KS = mpw2::KS;
    constexpr int NT = mpw2::WAVES * 16 * NW;
    constexpr int THREADS = mpw2::THREADS;
    constexpr int RG = MT / 16;
    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;
    constexpr int WW = KS * NW * TWORDS;            // trellis words per wave per iteration
    constexpr int WL = (WW + 31) / 32;              // W loads per lane per iteration
    constexpr int ALOADS = MT * 4 / THREADS;        // 16-byte A chunks per thread per iteration
    static_assert(ALOADS * THREADS == MT * 4, "whole A passes");
    static_assert(PF >= 1 && PF <= 4, "prefetch depth");

    if (blockIdx.y >= *num_tiles) return;
    const int tile = tiles[blockIdx.y];
    const int expert = tile >> mpw::TILE_SHIFT;
    const int mtile = tile & ((1 << mpw::TILE_SHIFT) - 1);
    const int64_t row0 = expert_offsets[expert] + (int64_t) mtile * MT;
    const int rows = (int) min((int64_t) MT, expert_offsets[expert + 1] - row0);
    const int proj = blockIdx.z;
    const uint32_t* B32 = (const uint32_t*) (proj ? B_table_1 : B_table_0)[expert];
    A += proj * a_proj_stride + row0 * a_ld;

    const int n0 = blockIdx.x * NT;
    const size_t slice_stride = (size_t) (size_n / 16) * TWORDS;

    __shared__ __align__(16) half sh_a[2][MT * 32];
    __shared__ __align__(16) uint32_t sh_w[mpw2::WAVES][WW];
    __shared__ __align__(16) half sh_b[mpw2::WAVES][KS * NW * 256];

    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int iters = size_k / (KS * 16);
    const int nrg = (rows + 15) >> 4;               // active 16-row groups (block-uniform)

    const int a_r = t >> 2;                         // + i * (THREADS / 4)
    const int a_c = t & 3;                          // 16-byte chunk (8 k)
    const half* a_src = A + (int64_t) a_r * a_ld + a_c * 8;
    const int ntile_n = size_n / 16;                // tail block: clamp like mpw2_gemm_kernel (duplicate, never stored)
    const int first_tile = min(n0 / 16 + warp * NW, ntile_n - NW);
    const uint32_t* w_src = B32 + (size_t) first_tile * TWORDS;   // NW adjacent tiles: contiguous

    uint4 ra[PF][ALOADS];
    uint32_t rw[PF][WL];

    // Loads past the last iteration re-read the last one (never stored): the load stream stays uniform
    auto load_a = [&] (int it, uint4* r)
    {
        const int k0 = min(it, iters - 1) * KS * 16;
        #pragma unroll
        for (int i = 0; i < ALOADS; ++i)
            r[i] = a_r + i * (THREADS / 4) < rows
                ? *((const uint4*) (a_src + (int64_t) i * (THREADS / 4) * a_ld + k0))
                : make_uint4(0, 0, 0, 0);
    };
    auto load_w = [&] (int it, uint32_t* r)
    {
        const uint32_t* src = w_src + (size_t) (min(it, iters - 1) * KS) * slice_stride;
        #pragma unroll
        for (int i = 0; i < WL; ++i)
        {
            const int idx = i * 32 + lane;
            if constexpr (DBG & 4) r[i] = idx * 0x9e3779b9u + it;
            else if (WW % 32 == 0 || idx < WW)
            {
                const int s = idx / (NW * TWORDS);
                const size_t o = (size_t) s * slice_stride + (idx - s * NW * TWORDS);
                if constexpr (OPT & 8) r[i] = ((const __attribute__((address_space(1))) uint32_t*) src)[o];
                else r[i] = src[o];
            }
        }
    };
    auto store_a = [&] (int buf, const uint4* r)
    {
        #pragma unroll
        for (int i = 0; i < ALOADS; ++i)
            *((uint4*) (sh_a[buf] + mpw2_a_off(a_r + i * (THREADS / 4), a_c))) = r[i];
    };
    auto store_w = [&] (const uint32_t* r)
    {
        #pragma unroll
        for (int i = 0; i < WL; ++i)
        {
            const int idx = i * 32 + lane;
            if (WW % 32 == 0 || idx < WW) sh_w[warp][idx] = r[i];
        }
    };
    // Decode this wave's KS x NW tiles through its private stage. LDS ops of one wave execute in order,
    // so the cross-lane RAW (store -> fragment read) and WAR (next iteration's stores) need no barrier.
    auto decode = [&] (HipFp16x16* bf)
    {
        #pragma unroll
        for (int s = 0; s < KS * NW; ++s)           // tile s = slice s / NW, n-tile s % NW
        {
            FragB f0, f1;
            if constexpr (DBG & 1)
                f0[0] = f0[1] = f1[0] = f1[1] = *((const half2*) (sh_w[warp] + s * TWORDS + (lane & 15)));
#if defined(EXL3_CB_HAVE_SAD)
            else if constexpr ((OPT & 4) && bits == 2 && !HALF && CB == 2)
                mpw2_dq8_2bits_mul1(sh_w[warp] + s * TWORDS, lane * 8, f0, f1);
#endif
            else
                dq_dispatch<bits, CB, HALF>(sh_w[warp] + s * TWORDS, lane * 8, f0, f1);
            // CUDA m16n8k16 B fragment: col g = lane >> 2 (f0) and g + 8 (f1), k = 2q, 2q+1 and
            // 2q+8, 2q+9, q = lane & 3. Stage layout [s][col][k]
            half* bt = sh_b[warp] + s * 256;
            const int g = lane >> 2;
            const int q = (lane & 3) * 2;
            *((half2*) (bt + g * 16 + q)) = f0[0];
            *((half2*) (bt + g * 16 + q + 8)) = f0[1];
            *((half2*) (bt + (g + 8) * 16 + q)) = f1[0];
            *((half2*) (bt + (g + 8) * 16 + q + 8)) = f1[1];
        }
        __builtin_amdgcn_wave_barrier();
        #pragma unroll
        for (int s = 0; s < KS * NW; ++s)
        {
            const half* p = sh_b[warp] + s * 256 + (lane & 15) * 16;
            if constexpr (OPT & 1)
            {
                ((uint4*) &bf[s])[0] = *((const uint4*) (p + (lane >> 4) * 8));
                ((uint4*) &bf[s])[1] = *((const uint4*) (p + ((lane >> 4) ^ 1) * 8));
            }
            else if constexpr (LDM == 1) bf[s] = mpw_frag_x16(p + (lane >> 4) * 8);
            else bf[s] = *((const HipFp16x16*) p);
        }
    };

    HipFp32x8 acc[RG][NW];
    #pragma unroll
    for (int i = 0; i < RG; ++i)
        #pragma unroll
        for (int j = 0; j < NW; ++j)
            #pragma unroll
            for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.0f;

    auto mma = [&] (int cur, const HipFp16x16* bf, auto nrc) __attribute__((always_inline))
    {
        constexpr int NR = decltype(nrc)::value;    // > 0: row-group count known at compile time (OPT bit 5)
        const int nr = NR ? NR : nrg;
        if constexpr ((OPT & 16) && (OPT & 2) && !(DBG & 2))
        {
            // OPT bit 4: A fragments software-pipelined one step ahead (two live fragments, no per-group
            // lgkmcnt(0) drain, no spills). Loads are unconditional (block-uniform guard only around the
            // WMMAs), so the unrolled chain alternates two registers without phis. Same order: s outer,
            // i inner, j innermost -> every acc sees the same k sequence (bit-exact).
            auto lda = [&] (int q)
            {
                const int s = q / RG;
                const int r = (q % RG) * 16 + (lane & 15);
                HipFp16x16 af;
                ((uint4*) &af)[0] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2 + (lane >> 4))));
                ((uint4*) &af)[1] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2 + ((lane >> 4) ^ 1))));
                return af;
            };
            HipFp16x16 a0 = lda(0);
            #pragma unroll
            for (int q = 0; q < KS * RG; ++q)
            {
                const int s = q / RG;
                const int i = q % RG;
                HipFp16x16 a1;
                if (q + 1 < KS * RG) a1 = lda(q + 1);
                if (i < nr)
                {
                    #pragma unroll
                    for (int j = 0; j < NW; ++j)
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, bf[s * NW + j], acc[i][j]);
                }
                a0 = a1;
                if constexpr (OPT & 128) __builtin_amdgcn_sched_barrier(0);   // bit 7: keep one step in flight
            }
            return;
        }
        #pragma unroll
        for (int s = 0; s < KS; ++s)
        {
            #pragma unroll
            for (int i = 0; i < RG; ++i)
            {
                if (i >= nr) continue;
                const int r = i * 16 + (lane & 15);
                HipFp16x16 af;
                if constexpr (OPT & 2)
                {
                    ((uint4*) &af)[0] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2 + (lane >> 4))));
                    ((uint4*) &af)[1] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2 + ((lane >> 4) ^ 1))));
                }
                else if constexpr (LDM == 1) af = mpw_frag_x16(sh_a[cur] + mpw2_a_off(r, s * 2 + (lane >> 4)));
                else
                {
                    ((uint4*) &af)[0] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2)));
                    ((uint4*) &af)[1] = *((const uint4*) (sh_a[cur] + mpw2_a_off(r, s * 2 + 1)));
                }
                #pragma unroll
                for (int j = 0; j < NW; ++j)
                {
                    if constexpr (DBG & 2) acc[i][j][0] += (float) af[0] + (float) bf[s * NW + j][lane & 15];
                    else acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf[s * NW + j], acc[i][j]);
                }
            }
        }
    };

    // Ring: set j % PF holds A(j) and W(j) until consumed
    #pragma unroll
    for (int j = 0; j < PF; ++j) { load_a(j, ra[j]); load_w(j, rw[j]); }
    store_a(0, ra[0]);
    load_a(PF, ra[0]);
    __syncthreads();

    #define MPW_LDS_BARRIER asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
    auto step = [&] (int it, auto ic, auto nrc) __attribute__((always_inline))     // ic = it % PF
    {
        constexpr int ws = decltype(ic)::value;
        constexpr int as = (ws + 1) % PF;
        MPW_LDS_BARRIER;                            // sh_a[(it + 1) & 1] free, A(it) visible
        store_w(rw[ws]);
        load_w(it + PF, rw[ws]);
        if (it + 1 < iters) store_a((it + 1) & 1, ra[as]);
        load_a(it + 1 + PF, ra[as]);
        HipFp16x16 bf[KS * NW];
        decode(bf);
        mma(it & 1, bf, nrc);
    };
    using I0 = std::integral_constant<int, 0>;
    using I1 = std::integral_constant<int, 1 % PF>;
    using I2 = std::integral_constant<int, 2 % PF>;
    using I3 = std::integral_constant<int, 3 % PF>;
    auto ring = [&] (auto nrc) __attribute__((always_inline))
    {
        int it = 0;
        for (; it + PF <= iters; it += PF)
        {
            step(it, I0{}, nrc);
            if constexpr (PF > 1) step(it + 1, I1{}, nrc);
            if constexpr (PF > 2) step(it + 2, I2{}, nrc);
            if constexpr (PF > 3) step(it + 3, I3{}, nrc);
        }
        if constexpr (PF > 1) if (it < iters) step(it, I0{}, nrc);
        if constexpr (PF > 2) if (it + 1 < iters) step(it + 1, I1{}, nrc);
        if constexpr (PF > 3) if (it + 2 < iters) step(it + 2, I2{}, nrc);
    };
    using N0 = std::integral_constant<int, 0>;
    // OPT bit 5: the K loop specialized on the block-uniform row-group count (typical expert sizes), so
    // the row-group guards fold away and the WMMA chain is branch-free; other counts take the runtime loop
    if constexpr ((OPT & 32) && RG == 8)
    {
        switch (nrg)
        {
            case 8: ring(std::integral_constant<int, 8>{}); break;
            case 7: ring(std::integral_constant<int, 7>{}); break;
            case 6: ring(std::integral_constant<int, 6>{}); break;
            default: ring(std::integral_constant<int, 0>{}); break;
        }
    }
    else ring(N0{});
    #undef MPW_LDS_BARRIER

    // Epilogue: C(lane, r) = C[row = 2r + (lane >> 4)][col = lane & 15]
    #pragma unroll
    for (int i = 0; i < RG; ++i)
    {
        if (i >= nrg) continue;
        #pragma unroll
        for (int j = 0; j < NW; ++j)
        {
            if (n0 + (warp * NW + j) * 16 >= size_n) continue;      // tail block: tile past size_n (wave-uniform)
            const int col = n0 + (warp * NW + j) * 16 + (lane & 15);
            #pragma unroll
            for (int r = 0; r < 8; ++r)
            {
                const int row = i * 16 + 2 * r + (lane >> 4);
                if (row >= rows) continue;
                const int64_t off = proj * c_proj_stride + (row0 + row) * size_n + col;
                if constexpr (OUT_FP32)
                    ((float*) C)[off] = acc[i][j][r];
                else
                    ((half*) C)[off] = __float2half_rn(acc[i][j][r]);
            }
        }
    }
#endif
}

// ---------------------------------------------------------------------------------------------
// Host side

namespace
{

// MPW_KERN: 1 = mpw_gemm (old), 2 = mpw2_gemm (default).
// EXL3_MPW2_VARIANT selects compiled mpw2 shapes: 0 = old PF4/MT128, 1 = PF2 (default),
// 2 = PF3, 3 = PF1, 4 = PF4/MT64. MPW_MT/MPW_NW remain compatibility overrides for variant 0.
inline int mpw_variant()
{
    const char* e = getenv("EXL3_MPW2_VARIANT");
    return e ? atoi(e) : 1;
}

inline int mpw_kern(int64_t assignments, int64_t experts)
{
    const char* e = getenv("MPW_KERN");
    return e ? atoi(e) : 2;
}

// MPW_NW (mpw2 only): n-tiles per wave, 2 (default, needs N % 32 == 0, else 1) or 1
inline int mpw_nw(int size_n)
{
    const char* e = getenv("MPW_NW");
    const int nw = e ? atoi(e) : 2;
    return (nw == 2 && size_n % 32 == 0) ? 2 : 1;      // NW 2 pairs n-tiles; the tail block is handled in-kernel
}

inline int mpw_mt(int kern)
{
    if (kern != 2) return mpw::MT;
    if (mpw_variant() == 4) return 64;
    const char* e = getenv("MPW_MT");
    return (e && atoi(e) == 64) ? 64 : mpw::MT;
}

template <bool OUT_FP32, int CB>
void launch_gemm_cb
(
    float K, dim3 grid, hipStream_t stream,
    const half* A, int64_t a_proj_stride, const int64_t* offsets, const int* tiles,
    const int* num_tiles, const int64_t* t0, const int64_t* t1, void* C, int64_t c_proj_stride,
    int size_k, int size_n, int kern, int a_ld
)
{
    TORCH_CHECK(a_ld >= size_k && (a_ld * 2) % 16 == 0, "exl3_moe_prefill_wmma: A pitch must be >= K and 16-byte aligned");
    TORCH_CHECK(a_ld == size_k || kern == 2, "exl3_moe_prefill_wmma: padded A pitch needs the mpw2 kernel");
    #define MPW_ARGS A, a_proj_stride, offsets, tiles, num_tiles, t0, t1, C, c_proj_stride, size_k, size_n, a_ld
    const char* ldm_env = getenv("MPW_LDM");   // read per launch: A/B harnesses flip it in-process
    const int ldm = ldm_env ? atoi(ldm_env) : 1;   // LDM=1 default: +3.4% e2e prefill@4K (HANDOFF step 2)
    const int k2 = (int) (K * 2.0f + 0.5f);
    if (kern == 2)
    {
        // grid.x arrives for NT = 128; mpw2 blocks span 128 * NW columns
        const int mt = mpw_mt(kern);
        const int nw = mpw_nw(size_n);
        grid.x = CEIL_DIVIDE(size_n, 128 * nw);        // last block may be a tail (mpw2_gemm_kernel clamps it)
        TORCH_CHECK(size_n % 16 == 0, "exl3_moe_prefill_wmma: size_n must be a multiple of 16");
        const char* pf_env = getenv("MPW_PF");
        const int pf = pf_env ? atoi(pf_env) : 4;
        const char* dbg_env = getenv("MPW_DBG");
        const int dbg = dbg_env ? atoi(dbg_env) : 0;
        // The kernel's NW template argument must equal nw: grid.x = size_n / (128 * nw) blocks of 128 * NW
        // columns. NW 2 with nw 1 (size_n % 256 != 0, e.g. Qwen3.8 gate/up N = 640) reads trellis tiles and
        // writes C columns past size_n.
        #define MPW2_L(b, h, M, P, W, D) \
            { TORCH_CHECK((W) == nw, "exl3_moe_prefill_wmma: mpw2 NW ", W, " != grid NW ", nw); \
              mpw2_gemm_kernel<b, h, OUT_FP32, CB, 1, M, P, W, D><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
        #define MPW2_V(V, M, P, W) \
            { TORCH_CHECK((W) == nw, "exl3_moe_prefill_wmma: mpw2 NW ", W, " != grid NW ", nw); \
              mpw2_gemm_kernel<2, false, OUT_FP32, CB, 1, M, P, W, 0, 2><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
        // EXL3_MPW2X=<OPT> (default 0 = off): mpw2x candidate (K 2, MT 128, NW 2 only), EXL3_MPW2X_PF=2|1.
        // Read per launch like MPW_LDM (A/B harnesses flip it in-process). Unknown values fall through.
        {
            const char* xe = getenv("EXL3_MPW2X");
            const int xo = xe ? atoi(xe) : 0;
            if (xo && k2 == 4 && nw == 2 && mt == 128)
            {
                const char* xpe = getenv("EXL3_MPW2X_PF");
                const int xpf = xpe ? atoi(xpe) : 2;
                #define MPW2X(O, P) if (xo == O && xpf == P) \
                    { mpw2x_gemm_kernel<2, false, OUT_FP32, CB, 1, 128, P, 2, 0, O><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
                MPW2X(2, 2) MPW2X(6, 2) MPW2X(10, 2) MPW2X(14, 2) MPW2X(2, 1) MPW2X(6, 1)
                #undef MPW2X
            }
        }
        // EXL3_PF7_OPT=<opt> (default 0 = off), EXL3_PF7_PF=1..4: production-class MT 128 NW 2 kernel for any K with
        // an explicit OPT mask (bits 0/1 direct LDS fragment reads, bit 2 2-bit mul1 decode, bit 8 tail-wave skip).
        // Bit-exact against OPT 0 / 2: same index arithmetic, same summation order.
        {
            const char* poe = getenv("EXL3_PF7_OPT");
            const int po = poe ? atoi(poe) : 0;
            if (po && nw == 2 && mt == 128)
            {
                const char* ppe = getenv("EXL3_PF7_PF");
                const int pp = ppe ? atoi(ppe) : 2;
                #define PF7O(b, h, O, P) if (po == O && pp == P) \
                    { mpw2_gemm_kernel<b, h, OUT_FP32, CB, 1, 128, P, 2, 0, O><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
                #define PF7K(b, h) PF7O(b, h, 3, 2) PF7O(b, h, 3, 4) PF7O(b, h, 259, 2) PF7O(b, h, 259, 4) \
                                   PF7O(b, h, 2, 2) PF7O(b, h, 2, 4) PF7O(b, h, 258, 2) PF7O(b, h, 258, 4) \
                                   PF7O(b, h, 6, 2) PF7O(b, h, 6, 4) PF7O(b, h, 262, 2) PF7O(b, h, 262, 4)
                switch (k2)
                {
                    case 4: PF7K(2, false) break;
                    case 5: PF7K(2, true) break;
                    case 6: PF7K(3, false) break;
                    case 7: PF7K(3, true) break;
                    case 8: PF7K(4, false) break;
                    default: break;
                }
                #undef PF7K
                #undef PF7O
            }
        }
        // EXL3_PF7_DBG=<1..7> (bench only, default 0): the production K 2 kernel with the DBG ablation bits
        // (1 no decode, 2 no WMMA, 4 no weight loads). Same index arithmetic as DBG 0 (the bits only drop work).
        {
            const char* pde = getenv("EXL3_PF7_DBG");
            const int pd = pde ? atoi(pde) : 0;
            if (pd && k2 == 4 && nw == 2 && mt == 128)
            {
                #define PF7D(D) if (pd == D) \
                    { mpw2_gemm_kernel<2, false, OUT_FP32, CB, 1, 128, 2, 2, D, 2><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
                PF7D(1) PF7D(2) PF7D(3) PF7D(4) PF7D(5) PF7D(6) PF7D(7) PF7D(8) PF7D(9) PF7D(12) PF7D(13)
                #undef PF7D
            }
        }
        if (k2 == 4 && nw == 2)                      // variants are NW 2 shapes; nw 1 takes MPW2_K below
        {
            switch (mpw_variant())
            {
                case 1:
                    // pf7: tail-wave skip (OPT bit 8) when size_n is not a multiple of 256 (Qwen gate/up N = 640)
                    if (size_n % 256 && !getenv("EXL3_PF7_OFF"))
                    { mpw2_gemm_kernel<2, false, OUT_FP32, CB, 1, 128, 2, 2, 0, 258><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
                    MPW2_V(1, 128, 2, 2)
                case 2: MPW2_V(2, 128, 3, 2)
                case 3: MPW2_V(3, 128, 1, 2)
                case 4: MPW2_V(4, 64, 4, 2)
                default: break;
            }
        }
        #define MPW2_K(b, h) \
            { if (nw == 2) { if (mt == 64) MPW2_L(b, h, 64, 4, 2, 0) MPW2_L(b, h, 128, 4, 2, 0) } \
              if (mt == 64) MPW2_L(b, h, 64, 4, 1, 0) \
              MPW2_L(b, h, 128, 4, 1, 0) }
        #define MPW2_D(D) if (dbg == D) MPW2_L(2, false, 64, 4, 2, D)
        const char* opt_env = getenv("MPW_OPT");   // read per launch (K 2, MT 128, NW 2 only); MPW_OPT=0 = old path
        const int opt = opt_env ? atoi(opt_env) : 2;   // 2 default: bit-exact, +2.3% e2e prefill@4K/16K (HANDOFF step 8)
        #define MPW2_O(O) if (opt == O) \
            { mpw2_gemm_kernel<2, false, OUT_FP32, CB, 1, 128, 4, 2, 0, O><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
        if (k2 == 4 && nw == 2 && mt == 128 && dbg == 0)
        {
            MPW2_O(1) MPW2_O(2) MPW2_O(3) MPW2_O(4) MPW2_O(5) MPW2_O(6) MPW2_O(7)
        }
        #undef MPW2_O
        if (k2 == 4 && nw == 2 && mt == 64)         // ablation instances (MPW_MT=64 MPW_NW=2, K 2 only)
        {
            MPW2_D(1) MPW2_D(2) MPW2_D(3) MPW2_D(4) MPW2_D(5) MPW2_D(6) MPW2_D(7)
            if (pf == 2) MPW2_L(2, false, 64, 2, 2, 0)
            if (ldm == 0)
            { mpw2_gemm_kernel<2, false, OUT_FP32, CB, 0, 64, 4, 2, 0><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
        }
        // pf7: 3-bit experts (MiMo mixed layers) get the direct-LDS fragment reads (OPT bits 0/1) that K 2 already had
        if (k2 == 6 && nw == 2 && mt == 128 && dbg == 0 && !getenv("EXL3_PF7_OFF"))
        {
            if (size_n % 256)
            { mpw2_gemm_kernel<3, false, OUT_FP32, CB, 1, 128, 2, 2, 0, 259><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
            { mpw2_gemm_kernel<3, false, OUT_FP32, CB, 1, 128, 2, 2, 0, 3><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
        }
        // 4-bit experts (Qwen3.8 K4 layers) take the same direct-LDS + tail-wave-skip variant as the 3-bit block
        // above (measured bit-identical to the default instance, +2 % prefill; the instance is already compiled for the
        // EXL3_PF7_OPT override)
        if (k2 == 8 && nw == 2 && mt == 128 && dbg == 0 && !getenv("EXL3_PF7_OFF"))
        {
            if (size_n % 256)
            { mpw2_gemm_kernel<4, false, OUT_FP32, CB, 1, 128, 2, 2, 0, 259><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
            { mpw2_gemm_kernel<4, false, OUT_FP32, CB, 1, 128, 2, 2, 0, 3><<<grid, mpw2::THREADS, 0, stream>>>(MPW_ARGS); return; }
        }
        switch (k2)
        {
            case 4: MPW2_K(2, false)
            case 5: MPW2_K(2, true)
            case 6: MPW2_K(3, false)
            case 7: MPW2_K(3, true)
            case 8: MPW2_K(4, false)
            case 10: MPW2_K(5, false)
            case 12: MPW2_K(6, false)
            default: TORCH_CHECK(false, "exl3_moe_prefill_wmma: unsupported K ", K);
        }
        #undef MPW2_D
        #undef MPW2_V
        #undef MPW2_K
        #undef MPW2_L
    }
    #define MPW_K(b, h) \
        { if (ldm == 1) { mpw_gemm_kernel<b, h, OUT_FP32, 0, 2, CB, 1><<<grid, mpw::THREADS, 0, stream>>>(MPW_ARGS); } \
          else { mpw_gemm_kernel<b, h, OUT_FP32, 0, 2, CB, 0><<<grid, mpw::THREADS, 0, stream>>>(MPW_ARGS); } break; }
    switch (k2)
    {
        case 4: MPW_K(2, false)
        case 5:
        {
            static const int dbg = getenv("MPW_DBG") ? atoi(getenv("MPW_DBG")) : 0;
            if (dbg == 1) { mpw_gemm_kernel<2, true, OUT_FP32, 1, 2, CB><<<grid, mpw::THREADS, 0, stream>>>(MPW_ARGS); break; }
            if (dbg == 2) { mpw_gemm_kernel<2, true, OUT_FP32, 2, 2, CB><<<grid, mpw::THREADS, 0, stream>>>(MPW_ARGS); break; }
            if (dbg == 3) { mpw_gemm_kernel<2, true, OUT_FP32, 0, 1, CB><<<grid, mpw::THREADS, 0, stream>>>(MPW_ARGS); break; }
            MPW_K(2, true)
        }
        case 6: MPW_K(3, false)
        case 7: MPW_K(3, true)
        case 8: MPW_K(4, false)
        case 10: MPW_K(5, false)
        case 12: MPW_K(6, false)
        default: TORCH_CHECK(false, "exl3_moe_prefill_wmma: unsupported K ", K);
    }
    #undef MPW_K
    #undef MPW_ARGS
}

// CB 1 = mcg, 2 = mul1 (codebook.cuh's decode_3inst_2 convention, same as exl3_dec.cu's CB).
// Nested angle brackets would break a single macro, hence the two-level dispatch.
template <bool OUT_FP32>
void launch_gemm
(
    float K, dim3 grid, hipStream_t stream,
    const half* A, int64_t a_proj_stride, const int64_t* offsets, const int* tiles,
    const int* num_tiles, const int64_t* t0, const int64_t* t1, void* C, int64_t c_proj_stride,
    int size_k, int size_n, bool mcg, int kern, int a_ld = 0
)
{
    if (a_ld == 0) a_ld = size_k;
    if (mcg)
        launch_gemm_cb<OUT_FP32, 1>(K, grid, stream, A, a_proj_stride, offsets, tiles, num_tiles,
                                    t0, t1, C, c_proj_stride, size_k, size_n, kern, a_ld);
    else
        launch_gemm_cb<OUT_FP32, 2>(K, grid, stream, A, a_proj_stride, offsets, tiles, num_tiles,
                                    t0, t1, C, c_proj_stride, size_k, size_n, kern, a_ld);
}

inline int tile_slots(int assignments, int experts, int kern)
{
    return CEIL_DIVIDE(assignments, mpw_mt(kern)) + experts;
}

void check_table(const at::Tensor& t, int64_t experts, const char* name)
{
    TORCH_CHECK(t.dtype() == at::kLong && t.is_contiguous() && t.numel() == experts,
                "exl3_moe_prefill_wmma: ", name, " must be int64[E]");
}

}  // namespace

void exl3_moe_prefill_wmma
(
    const at::Tensor& A,
    at::Tensor& output,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& order,
    const at::Tensor& expert_count,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    double K_gu,
    double K_down,
    at::Tensor& gu_had,
    at::Tensor& gu_out,
    at::Tensor& down_out,
    at::Tensor& expert_offsets,
    at::Tensor& inverse_order,
    at::Tensor& tiles,
    at::Tensor& tile_count,
    double act_limit,
    bool mcg
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(A.dtype() == at::kHalf && A.is_contiguous() && A.dim() == 2, "A must be fp16 [rows, H]");
    const int rows = A.size(0);
    const int H = A.size(1);
    TORCH_CHECK(selected.dtype() == at::kLong && selected.is_contiguous() && selected.dim() == 2 &&
                selected.size(0) == rows, "selected must be int64 [rows, top_k]");
    const int top_k = selected.size(1);
    const int assignments = rows * top_k;
    const int64_t experts = gate_trellis.numel();
    TORCH_CHECK(experts < (1 << (31 - mpw::TILE_SHIFT)), "too many experts");
    TORCH_CHECK(H % 128 == 0, "hidden width must be a multiple of 128");
    TORCH_CHECK(output.dtype() == at::kFloat && output.is_contiguous() &&
                output.size(0) == rows && output.size(1) == H, "output must be fp32 [rows, H]");
    TORCH_CHECK(weights.dtype() == at::kFloat && weights.is_contiguous() && weights.sizes() == selected.sizes(),
                "weights must be fp32 [rows, top_k]");
    TORCH_CHECK(order.dtype() == at::kLong && order.numel() == assignments, "order must be int64 [A]");
    TORCH_CHECK(expert_count.dtype() == at::kLong && expert_count.numel() >= experts, "expert_count");
    check_table(gate_trellis, experts, "gate_trellis"); check_table(gate_suh, experts, "gate_suh");
    check_table(gate_svh, experts, "gate_svh"); check_table(up_trellis, experts, "up_trellis");
    check_table(up_suh, experts, "up_suh"); check_table(up_svh, experts, "up_svh");
    check_table(down_trellis, experts, "down_trellis"); check_table(down_suh, experts, "down_suh");
    check_table(down_svh, experts, "down_svh");
    TORCH_CHECK(gu_had.dtype() == at::kHalf && gu_had.numel() >= (int64_t) 2 * assignments * H, "gu_had too small");
    TORCH_CHECK(gu_out.dtype() == at::kHalf && gu_out.dim() == 2 && gu_out.size(0) >= 2 * assignments, "gu_out too small");
    const int I = gu_out.size(1);
    TORCH_CHECK(I % 128 == 0, "intermediate width must be a multiple of 128");
    TORCH_CHECK(down_out.dtype() == at::kFloat && down_out.numel() >= (int64_t) assignments * H, "down_out too small");
    TORCH_CHECK(expert_offsets.dtype() == at::kLong && expert_offsets.numel() >= experts + 1, "expert_offsets");
    TORCH_CHECK(inverse_order.dtype() == at::kLong && inverse_order.numel() >= assignments, "inverse_order");
    const int kern = mpw_kern(assignments, experts);
    const int slots = tile_slots(assignments, experts, kern);
    TORCH_CHECK(tiles.dtype() == at::kInt && tiles.numel() >= slots, "tiles too small");
    TORCH_CHECK(tile_count.dtype() == at::kInt && tile_count.numel() >= 1, "tile_count");

    const int64_t* sel = (const int64_t*) selected.data_ptr();
    const int64_t* ord = (const int64_t*) order.data_ptr();
    int64_t* offs = (int64_t*) expert_offsets.data_ptr();
    int64_t* inv = (int64_t*) inverse_order.data_ptr();
    int* tl = (int*) tiles.data_ptr();
    int* tc = (int*) tile_count.data_ptr();
    auto P = [] (const at::Tensor& t) { return (const int64_t*) t.data_ptr(); };

    mpw_metadata_kernel<<<CEIL_DIVIDE(assignments, 256), 256, 0, stream>>>
    ((const int64_t*) expert_count.data_ptr(), ord, offs, inv, tl, tc, (int) experts, assignments, mpw_mt(kern));

    const char* glue_env = getenv("EXL3_MPW_GLUE");   // read per launch (A/B harnesses flip it in-process)
    const bool glue2 = glue_env && glue_env[0] == '1' && top_k >= 1 && top_k <= MPW_GLUE_TK_MAX;
    if (glue2)
    {
        const int64_t g2jobs = (int64_t) (assignments / top_k) * (H / 128);
        mpw_gather_had2_kernel<<<CEIL_DIVIDE(g2jobs, 8), 256, 0, stream>>>
        ((const half*) A.data_ptr(), (half*) gu_had.data_ptr(), sel, inv, P(gate_suh), P(up_suh),
         assignments, top_k, H);
    }
    else
    {
    const int64_t gjobs = (int64_t) 2 * (assignments / top_k) * (H / 128);
    mpw_gather_had_kernel<<<CEIL_DIVIDE(gjobs, 8), 256, 0, stream>>>
    ((const half*) A.data_ptr(), (half*) gu_had.data_ptr(), sel, inv, P(gate_suh), P(up_suh),
     assignments, top_k, H, 2);
    }

    launch_gemm<false>((float) K_gu, dim3(I / mpw::NT, slots, 2), stream,
        (const half*) gu_had.data_ptr(), (int64_t) assignments * H, offs, tl, tc,
        P(gate_trellis), P(up_trellis), gu_out.data_ptr(), (int64_t) assignments * I, H, I, mcg, kern);

    const int64_t ajobs = (int64_t) assignments * (I / 128);
    mpw_act_kernel<<<CEIL_DIVIDE(ajobs, 8), 256, 0, stream>>>
        ((half*) gu_out.data_ptr(), sel, ord, P(gate_svh), P(up_svh), P(down_suh), assignments, I,
         (float) act_limit);

    // down_out = gu_had workspace viewed fp32 (same byte count): the down GEMM writes fp32 rows
    // so the weighted reduce consumes fp32 end to end (2x output bytes vs fp16; cost in REPORT-10c)
    launch_gemm<true>((float) K_down, dim3(H / mpw::NT, slots, 1), stream,
        (const half*) gu_out.data_ptr(), 0, offs, tl, tc,
        P(down_trellis), P(down_trellis), down_out.data_ptr(), 0, I, H, mcg, kern);

    const int64_t rjobs = (int64_t) rows * (H / 128);
#define MPW_RED2(TKV) mpw_reduce_had2_kernel<TKV><<<CEIL_DIVIDE(rjobs, 8), 256, 0, stream>>> \
    ((const float*) down_out.data_ptr(), sel, (const float*) weights.data_ptr(), inv, P(down_svh), \
     (float*) output.data_ptr(), rows, H)
    if (glue2 && top_k == 10) { MPW_RED2(10); }
    else if (glue2 && top_k == 8) { MPW_RED2(8); }
    else
    mpw_reduce_had_kernel<<<CEIL_DIVIDE(rjobs, 8), 256, 0, stream>>>
    ((const float*) down_out.data_ptr(), sel, (const float*) weights.data_ptr(), inv, P(down_svh),
     (float*) output.data_ptr(), rows, top_k, H);
    cuda_check(hipPeekAtLastError());
}

void exl3_moe_prefill_gemm_test
(
    const at::Tensor& A,
    at::Tensor& C,
    const at::Tensor& expert_count,
    const at::Tensor& trellis_table,
    double K,
    int n_size,
    at::Tensor& expert_offsets,
    at::Tensor& tiles,
    at::Tensor& tile_count
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    // EXL3_GEMMTEST_Z=2 (bench only) = gate/up form: A and C hold 2 projections of rows each, grid.z = 2.
    // A may be a row-strided view (stride(1) == 1, pitch = stride(0) >= K, 16-byte aligned): the A row pitch bench.
    const char* ze = getenv("EXL3_GEMMTEST_Z");
    const int nz = (ze && atoi(ze) == 2) ? 2 : 1;
    TORCH_CHECK(A.dim() == 2 && A.stride(1) == 1 && A.size(0) % nz == 0, "A must be row-strided [nz * rows, K]");
    const int rows = A.size(0) / nz;
    const int size_k = A.size(1);
    const int a_ld = (int) A.stride(0);
    const int64_t experts = trellis_table.numel();
    TORCH_CHECK(C.size(0) == rows * nz && C.size(1) == n_size && C.is_contiguous(), "C shape");
    TORCH_CHECK(size_k % (mpw::KS * 16) == 0 && n_size % mpw::NT == 0, "unsupported shape");
    const int kern = mpw_kern(rows, experts);
    const int slots = tile_slots(rows, experts, kern);
    TORCH_CHECK(tiles.numel() >= slots, "tiles too small");
    int64_t* offs = (int64_t*) expert_offsets.data_ptr();
    int* tl = (int*) tiles.data_ptr();
    int* tc = (int*) tile_count.data_ptr();
    mpw_metadata_kernel<<<1, 256, 0, stream>>>
    ((const int64_t*) expert_count.data_ptr(), nullptr, offs, nullptr, tl, tc, (int) experts, rows, mpw_mt(kern));
    const int64_t* tt = (const int64_t*) trellis_table.data_ptr();
    if (C.dtype() == at::kFloat)
        launch_gemm<true>((float) K, dim3(n_size / mpw::NT, slots, nz), stream,
            (const half*) A.data_ptr(), (int64_t) rows * a_ld, offs, tl, tc, tt, tt, C.data_ptr(), (int64_t) rows * n_size, size_k, n_size, false, kern, a_ld);
    else
        launch_gemm<false>((float) K, dim3(n_size / mpw::NT, slots, nz), stream,
            (const half*) A.data_ptr(), (int64_t) rows * a_ld, offs, tl, tc, tt, tt, C.data_ptr(), (int64_t) rows * n_size, size_k, n_size, false, kern, a_ld);
    cuda_check(hipPeekAtLastError());
}

#endif  // USE_ROCM
