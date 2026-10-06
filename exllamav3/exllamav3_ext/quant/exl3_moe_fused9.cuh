#pragma once
// core9: mf7 kernel + cache touches (read-only prefetch loads whose results are dropped).
//  - static touch at kernel start / after the first dots group: finalize up-projection words, router rows, shared expert matrices (all known before the router)
//  - down touch after top-k: the down matrices of the selected experts, pulled while gate/up streams
// The arithmetic is mf7's, untouched: results are byte-identical to mf7::half for every vm value.
#include "exl3_moe_fused7.cuh"
#include "exl3_moe_fused9_idx.h"
#include "exl3_moe_fused9_inl.cuh"

#if defined(USE_ROCM) || defined(__HIPCC__)
namespace mf9 {
using namespace mf;
constexpr int V_T0 = 1, V_TMID = 2, V_TDOWN = 4, V_SU = 8, V_SR = 16, V_SS = 32;

// the load is inline asm into a sink register that stays live (chained "+v") until the end of the kernel; vector memory loads return in order, nobody reads the sink.
// The compiler does not know about these loads: its own vmcnt waits count only its loads, so they can only over-wait (safe).
__device__ __forceinline__ void touch_one(uint32_t& sink, const void* base, long bytes, long chunk, int lane)
{
    const char* a = (const char*) base + touch_addr(bytes, chunk, lane);
    asm volatile("global_load_b32 %0, %1, off" : "+v"(sink) : "v"(a));
}
__device__ __forceinline__ void touch_range(uint32_t& sink, const void* base, long bytes, int gw, int nw, int lane)
{
    const long nch = touch_nch(bytes);
    for (long ch = gw; ch < nch; ch += nw)
        touch_one(sink, base, bytes, ch, lane);
}
template <class S>
__device__ __forceinline__ void touch_static(const Params& p, int vm, uint32_t& sink)
{
    using DM = Dm<S>;
    const int gw = blockIdx.x * 16 + (threadIdx.x >> 5), nw = p.G * 16, lane = threadIdx.x & 31;
    const int sel = (vm & (V_SU | V_SR | V_SS)) ? (vm & (V_SU | V_SR | V_SS)) : (V_SU | V_SR | V_SS);
    if (sel & V_SU) touch_range(sink, p.upt, upt_bytes<S>(), gw, nw, lane);
    if (sel & V_SR) touch_range(sink, p.router, router_bytes<S>(), gw, nw, lane);
    if (sel & V_SS)
    {
        touch_range(sink, p.wt[3 * DM::NEXP + 0], sh_gu_bytes<S>(), gw, nw, lane);
        touch_range(sink, p.wt[3 * DM::NEXP + 1], sh_gu_bytes<S>(), gw, nw, lane);
        touch_range(sink, p.wt[3 * DM::NEXP + 2], sh_dn_bytes<S>(), gw, nw, lane);
    }
}
template <class S>
__device__ __forceinline__ void touch_down(const Params& p, const Smem<S>& s, uint32_t& sink)
{
    using DM = Dm<S>;
    const int gw = blockIdx.x * 16 + (threadIdx.x >> 5), nw = p.G * 16, lane = threadIdx.x & 31;
    int issued = 0;
    for (int u = 0; u <= s.ngroups && issued < TOUCH_DOWN_CAP; ++u)
    {
        const bool sh = u == s.ngroups;
        const int e = sh ? DM::NEXP : s.sorted_e[s.gstart[u]];
        const void* base = p.wt[3 * e + 2];
        const long bytes = sh ? sh_dn_bytes<S>() : rt_dn_bytes<S>();
        const long nch = touch_nch(bytes);
        for (long ch = gw; ch < nch && issued < TOUCH_DOWN_CAP; ch += nw)
        {
            touch_one(sink, base, bytes, ch, lane);
            ++issued;
        }
    }
}

// ---------------------------------------------------------------- phase 0: HC dots
template <class S>
__device__ __forceinline__ void ph_dots9(const Params& p, Smem<S>& s, int vm, uint32_t& sink)
{
    MF_DIMS
    using namespace mf7;
    constexpr int D4 = DM::D4, NIT = (D4 / 2 + 127) / 128, NIT2 = (D4 + 127) / 128, MR1 = MR + 1;
    const int R = p.R, team = threadIdx.x >> 7, tid = threadIdx.x & 127, lane = tid & 31, warp = tid >> 5;
    const int NT = p.G * 4, tg = blockIdx.x * 4 + team;
    const int total = H * MR1;
    const int lo = dots_lo(total, NT, tg), hi = dots_lo(total, NT, tg + 1);
    float* scr = s.u.g.red;
    int cur = lo;
    bool touched = false;
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
            if ((vm & V_TMID) && !touched) { touch_static<S>(p, vm, sink); touched = true; }
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
    if ((vm & V_TMID) && !touched) touch_static<S>(p, vm, sink);
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

template <class S, bool TM, bool INL, int ABL = 0>
__device__ __forceinline__ void body9(const Params& p, int vm)
{
    __shared__ Smem<S> s;
    int phase = 0;
    uint32_t sink = 0;
    mf7::st7<TM>(p, 0);
    if (vm & V_T0) touch_static<S>(p, vm, sink);
    ph_dots9<S>(p, s, vm, sink);
    mf7::gbar7<TM>(p, phase);
    if constexpr (INL) ph_finalize9<S, TM>(p, s); else mf7::ph_finalize7<S, TM>(p, s);
    mf7::gbar7<TM>(p, phase);
    if constexpr (INL) ph_router9<S>(p, s); else mf7::ph_router7<S>(p, s);
    mf7::gbar7<TM>(p, phase);
    if constexpr (INL) ph_topk9<S>(p, s, vm); else ph_topk<S>(p, s);
    mf7::st7<TM>(p, 15);
    if (blockIdx.x == 0 && threadIdx.x < p.R * S::TOPK) p.seldbg[threadIdx.x] = s.sel[threadIdx.x];
    if (blockIdx.x == 0 && threadIdx.x < p.R * S::TOPK) p.seldbg[64 + threadIdx.x] = (int) __half_as_ushort(s.wt[threadIdx.x]);
    if (vm & V_TDOWN) touch_down<S>(p, s, sink);
    if constexpr (INL) ph_gateup9<S, ABL>(p, s); else mf7::ph_gateup7<S, 0>(p, s);
    mf7::gbar7<TM>(p, phase);
    if constexpr (INL) ph_down9<S, ABL>(p, s); else mf7::ph_down7<S, 0>(p, s);
    mf7::gbar7<TM>(p, phase);
    if constexpr (INL) ph_combine9<S>(p, s); else ph_combine<S>(p, s);
    asm volatile("" :: "v"(sink));
    mf7::st7<TM>(p, 11);
    gfinish(p);
}

template <class S, bool TM, bool INL, int ABL = 0>
__global__ __launch_bounds__(THREADS) void moe_fused9_kernel(Params p, int vm) { body9<S, TM, INL, ABL>(p, vm); }
// two resident blocks per WGP (8 waves per SIMD, <= 192 VGPR): a grid of 2 * WGPs blocks, so a tile list of 880 tiles meets 640 waves (2 rounds) instead of 320 (3 rounds)
template <class S, bool TM, bool INL, int ABL = 0>
__global__ __launch_bounds__(THREADS) __attribute__((amdgpu_num_vgpr(192))) void moe_fused9o_kernel(Params p, int vm) { body9<S, TM, INL, ABL>(p, vm); }

// Split launches. The same phases and the same block partition as body9 (G = two blocks per WGP), but every grid barrier is a kernel boundary,
// so no block ever waits for another one and residency does not matter (a desktop client sharing the iGPU cannot stall or corrupt the step).
// Stage 0 dots | 1 finalize | 2 router | 3 top-k + gate/up | 4 down | 5 combine. The top-k result crosses the boundaries through seldbg (selected experts and weights, written by block 0 of stage 3).
template <class S, bool INL, int ABL, int STAGE>
__device__ __forceinline__ void body9s(const Params& p, int vm)
{
    static_assert(INL, "split launches exist for the forced-inline phases only");
    __shared__ Smem<S> s;
    uint32_t sink = 0;
    if constexpr (STAGE == 0) { if (vm & V_T0) touch_static<S>(p, vm, sink); ph_dots9<S>(p, s, vm, sink); }
    if constexpr (STAGE == 1) ph_finalize9<S, false>(p, s);
    if constexpr (STAGE == 2) ph_router9<S>(p, s);
    if constexpr (STAGE == 3)
    {
        ph_topk9<S>(p, s, vm);
        if (blockIdx.x == 0 && threadIdx.x < p.R * S::TOPK) p.seldbg[threadIdx.x] = s.sel[threadIdx.x];
        if (blockIdx.x == 0 && threadIdx.x < p.R * S::TOPK) p.seldbg[64 + threadIdx.x] = (int) __half_as_ushort(s.wt[threadIdx.x]);
        if (vm & V_TDOWN) touch_down<S>(p, s, sink);
        ph_gateup9<S, ABL>(p, s);
    }
    if constexpr (STAGE == 4) { topk_restore9<S>(p, s); ph_down9<S, ABL>(p, s); }
    if constexpr (STAGE == 5) { topk_restore9<S>(p, s); ph_combine9<S>(p, s); }
    asm volatile("" :: "v"(sink));
}
template <class S, bool INL, int ABL, int STAGE>
__global__ __launch_bounds__(THREADS) __attribute__((amdgpu_num_vgpr(192))) void moe_fused9s_kernel(Params p, int vm) { body9s<S, INL, ABL, STAGE>(p, vm); }

}  // namespace mf9
#endif
