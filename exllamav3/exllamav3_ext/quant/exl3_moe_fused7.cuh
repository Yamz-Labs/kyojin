#pragma once
// core7: fused MoE half-layer, front end rewritten for memory-level parallelism (same arithmetic order as core5, byte-identical results).
// Phases dots / finalize / router issue all their global loads of a work item before the first use; the dots phase splits (h, j) items
// evenly over the 80 teams (no 5-round tail) and keeps no per-round block barrier. gate/up, down, top-k, combine are core5's.
#include "exl3_moe_fused.cuh"
#include "exl3_moe_fused7_idx.h"

#if defined(USE_ROCM) || defined(__HIPCC__)
#include "exl3_moe_fused7_mv.cuh"
namespace mf7 {
using namespace mf;

template <bool TM> __device__ __forceinline__ void st7(const Params& p, int k)
{
    if constexpr (TM) { if (threadIdx.x == 0 && blockIdx.x < STAMP_MAXB && k < STAMP_STRIDE) p.seldbg[stamp_idx(blockIdx.x, k)] = (int) (unsigned) wall_clock64(); }
}

template <bool TM> __device__ __forceinline__ void gbar7(const Params& p, int& phase)
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    st7<TM>(p, 2 * phase + 1);
    ++phase;
    if (threadIdx.x == 0) gbar_wait(p, phase);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    st7<TM>(p, 2 * phase);
}

// ---------------------------------------------------------------- phase 0: HC dots
template <class S>
__device__ void ph_dots7(const Params& p, Smem<S>& s)
{
    MF_DIMS
    constexpr int D4 = DM::D4, NIT = (D4 / 2 + 127) / 128, NIT2 = (D4 + 127) / 128, MR1 = MR + 1;
    const int R = p.R, team = threadIdx.x >> 7, tid = threadIdx.x & 127, lane = tid & 31, warp = tid >> 5;
    const int NT = p.G * 4, tg = blockIdx.x * 4 + team;
    const int total = H * MR1;
    const int lo = dots_lo(total, NT, tg), hi = dots_lo(total, NT, tg + 1);
    float* scr = s.u.g.red;
    int cur = lo;
    while (cur < hi)
    {
        const int h = cur / MR1, j = cur % MR1;
        const int m0 = cur - lo;
        int n = 1;
        if (j < MR)
        {
            n = min(min(DOTS_GRP, hi - cur), MR - j);
            int2 pk[DOTS_GRP][NIT];
            #pragma unroll
            for (int q = 0; q < DOTS_GRP; ++q)
                #pragma unroll
                for (int it = 0; it < NIT; ++it)
                {
                    const int c = tid + 128 * it;
                    pk[q][it] = (q < n && c < D4 / 2) ? ((const int2*) (p.fn + fn_row<S>(j + q, h)))[c] : make_int2(0, 0);
                }
            float4 xs[NIT][2][MAXR];
            #pragma unroll
            for (int it = 0; it < NIT; ++it)
            {
                const int c = tid + 128 * it;
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                {
                    xs[it][0][r] = make_float4(0.f, 0.f, 0.f, 0.f); xs[it][1][r] = xs[it][0][r];
                    if (r < R && c < D4 / 2)
                    {
                        const float4* s4 = (const float4*) (p.xin + xin_row<S>(r, 0));
                        xs[it][0][r] = s4[(size_t) h * D4 + 2 * c]; xs[it][1][r] = s4[(size_t) h * D4 + 2 * c + 1];
                    }
                }
            }
            float a[DOTS_GRP][MAXR];
            #pragma unroll
            for (int q = 0; q < DOTS_GRP; ++q)
                #pragma unroll
                for (int r = 0; r < MAXR; ++r) a[q][r] = 0.0f;
            #pragma unroll
            for (int it = 0; it < NIT; ++it)
            {
                const int c = tid + 128 * it;
                if (c < D4 / 2)
                {
                    #pragma unroll
                    for (int q = 0; q < DOTS_GRP; ++q)
                    {
                        if (q < n)
                        {
                            float w0, w1, w2, w3, w4, w5, w6, w7;
                            if (p.gs)
                            {
                                const float gsc = p.fn_scale[fngs_idx<S>(j + q, h, c)];
                                gs_unpack_s8x4((uint32_t) pk[q][it].x, gsc, w0, w1, w2, w3);
                                gs_unpack_s8x4((uint32_t) pk[q][it].y, gsc, w4, w5, w6, w7);
                            }
                            else
                            {
                                unpack_s8x4((uint32_t) pk[q][it].x, w0, w1, w2, w3);
                                unpack_s8x4((uint32_t) pk[q][it].y, w4, w5, w6, w7);
                            }
                            #pragma unroll
                            for (int r = 0; r < MAXR; ++r)
                            {
                                if (r < R)
                                {
                                    const float4 s0 = xs[it][0][r], s1 = xs[it][1][r];
                                    float x = a[q][r];
                                    x = fmaf(s0.x, w0, x); x = fmaf(s0.y, w1, x); x = fmaf(s0.z, w2, x); x = fmaf(s0.w, w3, x);
                                    x = fmaf(s1.x, w4, x); x = fmaf(s1.y, w5, x); x = fmaf(s1.z, w6, x); x = fmaf(s1.w, w7, x);
                                    a[q][r] = x;
                                }
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int q = 0; q < DOTS_GRP; ++q)
                if (q < n)
                    #pragma unroll
                    for (int r = 0; r < MAXR; ++r)
                        if (r < R)
                        {
                            float v = a[q][r];
                            for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o, 32);
                            if (lane == 0) scr[dots_scratch(team, m0 + q, r, warp)] = v;
                        }
        }
        else
        {
            float4 sv[NIT2][MAXR];
            #pragma unroll
            for (int it = 0; it < NIT2; ++it)
            {
                const int c = tid + 128 * it;
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                {
                    sv[it][r] = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (r < R && c < D4) sv[it][r] = ((const float4*) (p.xin + xin_row<S>(r, 0)))[(size_t) h * D4 + c];
                }
            }
            float a[MAXR];
            #pragma unroll
            for (int r = 0; r < MAXR; ++r) a[r] = 0.0f;
            #pragma unroll
            for (int it = 0; it < NIT2; ++it)
                if (tid + 128 * it < D4)
                    #pragma unroll
                    for (int r = 0; r < MAXR; ++r)
                        if (r < R) a[r] = fmaf(sv[it][r].x, sv[it][r].x, fmaf(sv[it][r].y, sv[it][r].y, fmaf(sv[it][r].z, sv[it][r].z, fmaf(sv[it][r].w, sv[it][r].w, a[r]))));
            #pragma unroll
            for (int r = 0; r < MAXR; ++r)
                if (r < R)
                {
                    float v = a[r];
                    for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o, 32);
                    if (lane == 0) scr[dots_scratch(team, m0, r, warp)] = v;
                }
        }
        cur += n;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 4 * DOTS_MAXIT * R; idx += THREADS)
    {
        const int tm = idx / (DOTS_MAXIT * R), rem = idx % (DOTS_MAXIT * R), m = rem / R, r = rem % R;
        const int tgx = blockIdx.x * 4 + tm;
        const int it = dots_lo(total, NT, tgx) + m;
        if (it >= dots_lo(total, NT, tgx + 1)) continue;
        const int h = it / MR1, j = it % MR1;
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) v += scr[dots_scratch(tm, m, r, w)];
        if (j < MR && !p.gs) v *= p.fn_scale[j];
        p.dots[dots_idx<S>(r, j, h)] = v;
    }
}

// ---------------------------------------------------------------- phase 1: HC finalize (core5 arithmetic; the int8 up-projection words are loaded before the scalar prologue)
template <class S, bool TM>
__device__ void ph_finalize7(const Params& p, Smem<S>& s)
{
    MF_DIMS
    const int R = p.R, team = threadIdx.x >> 8, tid = threadIdx.x & 255, lane = tid & 31, warp = tid >> 5;
    const int D4 = D / 4;
    constexpr int CH = 32, NI = LR / 32, NTS = (LR * MAXR + 255) / 256;
    static_assert(LR % 32 == 0 && D % CH == 0, "finalize7 shape");
    const int total = DM::FIN_CHUNKS;
    const int rounds = (total + p.G * 2 - 1) / (p.G * 2);
    float* t_s = s.u.h.t[team];
    float* rmr_s = s.u.h.rmr[team];
    const float inv_h = 1.0f / (float) H;
    for (int k = 0; k < rounds; ++k)
    {
        const int cidx = blockIdx.x * 2 + team + k * p.G * 2;
        const bool valid = cidx < total;
        const int c = cidx * (CH / 4) + warp;                  // the warp's single column quad
        // loads in this order (the wait counter is in order): dots of the scalar prologue (L2), then the int8 up words (DRAM)
        static_assert(H == 4, "float4 dots rows");
        float dsum[4];
        float4 dd[NTS];
        #pragma unroll
        for (int h = 0; h < 4; ++h) dsum[h] = 0.0f;
        if (valid && tid < H * R) { const int r = tid / H, h = tid % H; dsum[0] = p.dots[dots_idx<S>(r, MR, h)]; }
        #pragma unroll
        for (int q = 0; q < NTS; ++q)
        {
            const int idx = tid + 256 * q;
            dd[q] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (valid && idx < LR * R) { const int r = idx / LR, ii = idx % LR; dd[q] = *(const float4*) (p.dots + dots_idx<S>(r, 0, 0) + (size_t) ii * H); }
        }
        uint32_t up[H][NI];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            #pragma unroll
            for (int i = 0; i < NI; ++i)
                up[h][i] = valid ? *(const uint32_t*) (p.upt + upt_u32<S>(h, c, lane + 32 * i) * 4) : 0u;
        if (k == 0) st7<TM>(p, 12);
        if (valid && tid < H * R)
        {
            const int r = tid / H, h = tid % H;
            rmr_s[r * 4 + h] = rsqrtf(dsum[0] / (float) D + p.rms_eps);
        }
        __syncthreads();
        if (k == 0) st7<TM>(p, 13);
        if (valid)
        {
            #pragma unroll
            for (int q = 0; q < NTS; ++q)
            {
                const int idx = tid + 256 * q;
                if (idx < LR * R)
                {
                    const int r = idx / LR, ii = idx % LR;
                    float v = 0.0f;
                    v = fmaf(rmr_s[r * 4 + 0], dd[q].x, v); v = fmaf(rmr_s[r * 4 + 1], dd[q].y, v); v = fmaf(rmr_s[r * 4 + 2], dd[q].z, v); v = fmaf(rmr_s[r * 4 + 3], dd[q].w, v);
                    v *= inv_h;
                    t_s[r * LR + ii] = v * sigmoidf_(v);
                }
            }
            if (cidx == 0 && tid < H)
            {
                for (int r = 0; r < R; ++r)
                {
                    const float* dr = p.dots + dots_idx<S>(r, 0, 0);
                    float v = 0.0f;
                    #pragma unroll
                    for (int h = 0; h < H; ++h) v = fmaf(rmr_s[r * 4 + h], dr[(size_t) (LR + tid) * H + h], v);
                    p.post[(size_t) r * H + tid] = 2.0f * sigmoidf_(v * inv_h);
                }
            }
        }
        __syncthreads();
        if (k == 0) st7<TM>(p, 14);
        if (valid)
        {
            float4 g[MAXR][H];
            #pragma unroll
            for (int r = 0; r < MAXR; ++r)
                #pragma unroll
                for (int h = 0; h < H; ++h) g[r][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 gsc[H];
            #pragma unroll
            for (int i = 0; i < NI; ++i)
            {
                const int ii = lane + 32 * i;
                float ti[MAXR];
                #pragma unroll
                for (int r = 0; r < MAXR; ++r) ti[r] = r < R ? t_s[r * LR + ii] : 0.0f;
                if (p.gs && (i & 1) == 0)                           // first 32-rank step of a 64-rank group
                {
                    #pragma unroll
                    for (int h = 0; h < H; ++h) gsc[h] = *(const float4*) (p.up_scale + upgs_idx<S>(h, c, ii));
                }
                #pragma unroll
                for (int h = 0; h < H; ++h)
                {
                    float u0, u1, u2, u3;
                    unpack_s8x4(up[h][i], u0, u1, u2, u3);
                    if (p.gs) { u0 = gs_dq(u0, gsc[h].x); u1 = gs_dq(u1, gsc[h].y); u2 = gs_dq(u2, gsc[h].z); u3 = gs_dq(u3, gsc[h].w); }
                    #pragma unroll
                    for (int r = 0; r < MAXR; ++r)
                        if (r < R)
                        {
                            g[r][h].x = fmaf(ti[r], u0, g[r][h].x); g[r][h].y = fmaf(ti[r], u1, g[r][h].y);
                            g[r][h].z = fmaf(ti[r], u2, g[r][h].z); g[r][h].w = fmaf(ti[r], u3, g[r][h].w);
                        }
                }
            }
            #pragma unroll
            for (int r = 0; r < MAXR; ++r)
                if (r < R)
                    #pragma unroll
                    for (int h = 0; h < H; ++h)
                        for (int o = 16; o > 0; o >>= 1)
                        {
                            g[r][h].x += __shfl_xor(g[r][h].x, o, 32); g[r][h].y += __shfl_xor(g[r][h].y, o, 32);
                            g[r][h].z += __shfl_xor(g[r][h].z, o, 32); g[r][h].w += __shfl_xor(g[r][h].w, o, 32);
                        }
            if (lane == 0)
            {
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                    if (r < R)
                    {
                        const float4* s4 = (const float4*) (p.xin + xin_row<S>(r, 0));
                        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        #pragma unroll
                        for (int h = 0; h < H; ++h)
                        {
                            const float4 sc = p.gs ? make_float4(1.0f, 1.0f, 1.0f, 1.0f) : *(const float4*) (p.up_scale + upsc_idx<S>(h, c));
                            float4 sv = s4[(size_t) h * D4 + c];
                            half4 wq = *(const half4*) (p.w + upsc_idx<S>(h, c));
                            float coef = rmr_s[r * 4 + h] * inv_h;
                            o.x = fmaf(sigmoidf_(g[r][h].x * sc.x) * coef * __low2float(wq.x),  sv.x, o.x);
                            o.y = fmaf(sigmoidf_(g[r][h].y * sc.y) * coef * __high2float(wq.x), sv.y, o.y);
                            o.z = fmaf(sigmoidf_(g[r][h].z * sc.z) * coef * __low2float(wq.y),  sv.z, o.z);
                            o.w = fmaf(sigmoidf_(g[r][h].w * sc.w) * coef * __high2float(wq.y), sv.w, o.w);
                        }
                        half2* out2 = (half2*) (p.mixed + mixed_row<S>(r));
                        out2[c * 2] = __floats2half2_rn(o.x, o.y);
                        out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
                    }
            }
        }
        __syncthreads();
        if (k == 0) st7<TM>(p, 15);
    }
}

// ---------------------------------------------------------------- phase 2: router logits, all row loads of two experts issued before the first fma
template <class S>
__device__ void ph_router7(const Params& p, Smem<S>& s)
{
    MF_DIMS
    constexpr int NIT = D / 64, NE = 2;
    static_assert(D % 64 == 0, "router7 shape");
    const int R = p.R, wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    half* mx = s.u.g.A;
    for (int i = threadIdx.x; i < R * D / 8; i += THREADS) ((uint4*) mx)[i] = ((const uint4*) p.mixed)[i];
    __syncthreads();
    for (int eb = wave * p.G + blockIdx.x; eb < NEXP; eb += NE * 16 * p.G)
    {
        uint32_t g[NE][NIT];
        #pragma unroll
        for (int k = 0; k < NE; ++k)
        {
            const int e = eb + k * 16 * p.G;
            const uint32_t* g2 = (const uint32_t*) (p.router + router_row<S>(e < NEXP ? e : eb));
            #pragma unroll
            for (int i = 0; i < NIT; ++i) g[k][i] = g2[lane + 32 * i];
        }
        #pragma unroll
        for (int k = 0; k < NE; ++k)
        {
            const int e = eb + k * 16 * p.G;
            if (e < NEXP)
            {
                float sum[MAXR];
                #pragma unroll
                for (int r = 0; r < MAXR; ++r) sum[r] = 0.0f;
                #pragma unroll
                for (int i = 0; i < NIT; ++i)
                {
                    const half2 gg = __builtin_bit_cast(half2, g[k][i]);
                    #pragma unroll
                    for (int r = 0; r < MAXR; ++r)
                        if (r < R)
                        {
                            const half2 hh = ((const half2*) (mx + (size_t) r * D))[lane + 32 * i];
                            sum[r] = fmaf(__half2float(__low2half(hh)), __half2float(__low2half(gg)), sum[r]);
                            sum[r] = fmaf(__half2float(__high2half(hh)), __half2float(__high2half(gg)), sum[r]);
                        }
                }
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                {
                    float v = wave_sum_down(sum[r]);
                    if (lane == 0 && r < R) p.scores[scores_idx<S>(r, e)] = __float2half_rn(v);
                }
            }
        }
    }
    if (blockIdx.x == p.G - 1 && wave == 15)
    {
        // replay of the engine's add_sigmoid_gate_proj reduction (virtual thread t = lane + 32 j, strided fma chain, tree strides 512..32 as register folds, then 16..1 by shuffles);
        // all sgate loads are issued first, the fma chain order per (row, j) is unchanged
        constexpr int NT3 = (D + 1023) / 1024;
        float sgf[32][NT3];
        #pragma unroll
        for (int j = 0; j < 32; ++j)
            #pragma unroll
            for (int t = 0; t < NT3; ++t)
            {
                const int i = lane + 32 * j + 1024 * t;
                sgf[j][t] = i < D ? __half2float(p.sgate_w[i]) : 0.0f;
            }
        for (int r = 0; r < R; ++r)
        {
            float pt[32];
            #pragma unroll
            for (int j = 0; j < 32; ++j)
            {
                float a = 0.0f;
                #pragma unroll
                for (int t = 0; t < NT3; ++t)
                {
                    const int i = lane + 32 * j + 1024 * t;
                    if (i < D) a = fmaf(sgf[j][t], __half2float(mx[(size_t) r * D + i]), a);
                }
                pt[j] = a;
            }
            #pragma unroll
            for (int w = 16; w >= 1; w >>= 1)
            {
                #pragma unroll
                for (int j = 0; j < w; ++j) pt[j] += pt[j + w];
            }
            const float v = wave_sum_down(pt[0]);
            if (lane == 0) p.sgl[r] = v;
        }
    }
}

template <class S, bool TM, int ABL>
__global__ __launch_bounds__(THREADS) void moe_fused7_kernel(Params p, int vmask)
{
    __shared__ Smem<S> s;
    int phase = 0;
    st7<TM>(p, 0);
    if (vmask & 1) ph_dots7<S>(p, s); else mf::ph_dots<S>(p, s);
    gbar7<TM>(p, phase);
    if (vmask & 2) ph_finalize7<S, TM>(p, s); else mf::ph_finalize<S>(p, s);
    gbar7<TM>(p, phase);
    if (vmask & 4) ph_router7<S>(p, s); else mf::ph_router<S>(p, s);
    gbar7<TM>(p, phase);
    ph_topk<S>(p, s);
    if (blockIdx.x == 0 && threadIdx.x < p.R * S::TOPK) p.seldbg[threadIdx.x] = s.sel[threadIdx.x];
    if (blockIdx.x == 0 && threadIdx.x < p.R * S::TOPK) p.seldbg[64 + threadIdx.x] = (int) __half_as_ushort(s.wt[threadIdx.x]);
    ph_gateup7<S, ABL>(p, s); gbar7<TM>(p, phase);
    ph_down7<S, ABL>(p, s);   gbar7<TM>(p, phase);
    ph_combine<S>(p, s);
    st7<TM>(p, 11);
    gfinish(p);
}

}  // namespace mf7
#endif
