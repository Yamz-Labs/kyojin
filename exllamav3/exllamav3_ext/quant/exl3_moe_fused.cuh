#pragma once
// Fused MoE half-layer decode (gfx11.5, opt-in EXL3_MOE_FUSED=1): ONE persistent launch per half-layer for R = 1..4 rows.
// hyper-connection dots -> finalize/mix -> router -> top-k + expert grouping -> gate/up (VALU trellis matvec) -> silu*up -> down
// -> Hadamard post + weighted sum + shared expert + gated residual apply (in place on the residual stack).
// Weights are repacked at load in read order (see exl3_moe_fused_idx.h). The matvec body is the core3 VALU one (same reduction order for
// every row count: chunk-ordered 16-chunk sums from 0.0f), so the result is byte-identical to the flag-off engine at R > 1 and to the same
// row of any larger batch at R = 1.
// Shapes: everything is a template parameter S (mf::QwenShape is the one instantiated); K (bits) of routed and shared experts are S::RB / S::SB.
#include "exl3_moe_fused_idx.h"
#include "exl3_moe_valu.cuh"   // mv_dot2

#if defined(USE_ROCM) || defined(__HIPCC__)
namespace mf {

struct Params
{
    int R, G;
    float* xin; float* xout;                       // xout may alias xin (in-place apply)
    const int8_t* fn; const float* fn_scale; const int8_t* upt; const float* up_scale; const half* w;
    const half* router; const half* sgate_w;
    const uint32_t* const* wt;                     // [3 * (NEXP + 1)] matrix base pointers: unit u (expert id, shared = NEXP), proj 0 gate 1 up 2 down
    const half* const* svt;                        // [6 * (NEXP + 1)]: gsuh gsvh usuh usvh dsuh dsvh of unit u
    int native;                                    // 1 = engine trellis layout, 0 = core4 repack layout (idx.h)
    float* dots; float* post; half* mixed; half* scores; float* sgl;
    half* gu; float* dn; float* ydbg; int* seldbg;
    unsigned* bar; unsigned* done; int* err;       // all three zero between launches (the last block resets bar and done)
    float rms_eps;
    int gs;                                        // 1 = group-scaled int8 mixer weights (fn_scale / up_scale hold group scales, see idx.h)
};

template <class S>
struct Smem
{
    union
    {
        struct { float red[RED_SIZE]; half A[MAXR * S::D]; half A2[MAXR * S::D]; } g;
        struct { float rows[(S::TOPK + 1) * 128]; } c;
        struct { float t[2][MAXR * S::LR]; float rmr[2][MAXR * 4]; float dred[4][MAXR][4][4]; } h;
    } u;
    static constexpr int TR = TRows<S>::v;                    // rows of one launch (4, or 8 in the moe8 wide instantiation)
    static constexpr int TA = TR * S::TOPK;                   // top-k slots
    int sel[TA];
    half wt[TA];
    int sorted_e[TA], sorted_a[TA], tok[TA], gstart[TA + 2], head[TA];
    int urows[TA + (TR > MAXR ? 2 : 1)];                      // wide: up to TA routed units + 2 shared units
    int ngroups;                                              // wide: routed units (groups cut to MAXR rows)
};

#define MF_DIMS using DM = Dm<S>; constexpr int D = DM::D, H = DM::H, LR = DM::LR, MR = DM::MR, NEXP = DM::NEXP, TOPK = DM::TOPK, INTER = DM::INTER; \
                (void) D; (void) H; (void) LR; (void) MR; (void) NEXP; (void) TOPK; (void) INTER;

constexpr float HAD_SCALE = 0.088388347648f;

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ void unpack_s8x4(uint32_t v, float& f0, float& f1, float& f2, float& f3)
{
    f0 = (float) ((int) (v << 24) >> 24);
    f1 = (float) ((int) (v << 16) >> 24);
    f2 = (float) ((int) (v << 8) >> 24);
    f3 = (float) ((int) v >> 24);
}

// group scales: weight = half(code * scale) as fp32 (bit-exact twin of the fp16 weights half(code * scale) used by the fp16 kernels)
// The product must be rounded to fp32 BEFORE the half rounding (the fp16 twin sees half(fp32(q * sc))); without the barrier the compiler
// fuses mul + convert into one v_fma_mix and rounds once (differs on exact half ties).
#ifndef GS_NOFUSE
#define GS_NOFUSE(p) asm volatile("" : "+v"(p))
#endif
__device__ __forceinline__ float gs_dq(float q, float sc) { float p = q * sc; GS_NOFUSE(p); return __half2float(__float2half_rn(p)); }
__device__ __forceinline__ void gs_unpack_s8x4(uint32_t v, float sc, float& f0, float& f1, float& f2, float& f3)
{
    unpack_s8x4(v, f0, f1, f2, f3);
    f0 = gs_dq(f0, sc); f1 = gs_dq(f1, sc); f2 = gs_dq(f2, sc); f3 = gs_dq(f3, sc);
}

// ---- optional phase time stamps (-DMF_TIMING): thread 0 of each block writes wall_clock64 (100 MHz) low 32 bits, slot k, to seldbg
#ifdef MF_TIMING
__device__ __forceinline__ void mf_stamp(const Params& p, int k)
{
    if (threadIdx.x == 0 && blockIdx.x < STAMP_MAXB && k < STAMP_SLOTS) p.seldbg[stamp_idx(blockIdx.x, k)] = (int) (unsigned) wall_clock64();
}
#else
__device__ __forceinline__ void mf_stamp(const Params&, int) {}
#endif

// ---- grid barrier (counter starts at 0 each launch)
// ctl words (u32, 256 bytes): 0 bar, 1 done, 2 err = first timed-out phase (0 = none), 3 timeout count, 4 phase / 5 block / 6 bar value seen / 7 wait ticks /
// 8 first block that had not arrived, 11 grid size, of the FIRST timeout; 9 longest wait over WAIT_LOG ticks, 10 count of such waits; [CTL_ARR, CTL_ARR + CTL_MAXB) arrival markers.
constexpr int CTL_ARR = 16, CTL_MAXB = 48;
constexpr unsigned WAIT_LOG = 5000;                                // 50 us at the 100 MHz wall clock
// thread 0 of the block, after ++phase: announce, wait until all p.G blocks arrived. Bounded spin; on timeout the record above is written (first one wins)
__device__ __forceinline__ void gbar_wait(const Params& p, int phase)
{
    unsigned* ctl = p.bar;
    const unsigned target = (unsigned) phase * p.G;
    __hip_atomic_store(ctl + CTL_ARR + blockIdx.x, (unsigned) phase, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_fetch_add(p.bar, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const long long t0 = wall_clock64();
    long long spins = 0;
    while (__hip_atomic_load(p.bar, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target)
    {
        __builtin_amdgcn_s_sleep(1);
        if (++spins > 8000000LL)
        {
            __hip_atomic_fetch_add(ctl + 3, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            unsigned zero = 0u;
            if (__hip_atomic_compare_exchange_strong(ctl + 2, &zero, (unsigned) phase, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
            {
                unsigned miss = 0xffffffffu;
                for (int b = 0; b < p.G; ++b)
                    if (__hip_atomic_load(ctl + CTL_ARR + b, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned) phase) { miss = (unsigned) b; break; }
                ctl[4] = (unsigned) phase; ctl[5] = blockIdx.x; ctl[6] = __hip_atomic_load(p.bar, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                ctl[7] = (unsigned) (wall_clock64() - t0); ctl[8] = miss; ctl[11] = (unsigned) p.G;
            }
            return;
        }
    }
    const unsigned el = (unsigned) (wall_clock64() - t0);
    if (el > WAIT_LOG)
    {
        __hip_atomic_fetch_max(ctl + 9, el, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_fetch_add(ctl + 10, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

__device__ __forceinline__ void gbar(const Params& p, int& phase)
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    mf_stamp(p, 2 * phase + 1);
    ++phase;
    if (threadIdx.x == 0) gbar_wait(p, phase);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    mf_stamp(p, 2 * phase);
}

// the last block to arrive here resets bar, done and the arrival markers: every block is past its last barrier, nobody spins
__device__ __forceinline__ void gfinish(const Params& p)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const unsigned d = __hip_atomic_fetch_add(p.done, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (d == (unsigned) p.G - 1)
        {
            for (int b = 0; b < p.G; ++b) __hip_atomic_store(p.bar + CTL_ARR + b, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(p.bar, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(p.done, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// ---------------------------------------------------------------- Hadamard helpers (register form, same math as hadamard_inner.cuh)
template <bool PRE, bool POST>
__device__ __forceinline__ half4 had_regs_h(half4 v, const half* scale, const int t)
{
    if constexpr (PRE) { half4 sc = ((const half4*) scale)[t]; v.x = __hmul2(v.x, sc.x); v.y = __hmul2(v.y, sc.y); }
    float v0 = __half2float(__low2half(v.x)), v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y)), v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * HAD_SCALE, h1 * HAD_SCALE);
    v.y = __floats2half2_rn(h2 * HAD_SCALE, h3 * HAD_SCALE);
    if constexpr (POST) { half4 sc = ((const half4*) scale)[t]; v.x = __hmul2(v.x, sc.x); v.y = __hmul2(v.y, sc.y); }
    return v;
}

__device__ __forceinline__ float4 had_regs_f_post(float4 v, const half* scale, const int t)
{
    float v0 = v.x, v1 = v.y, v2 = v.z, v3 = v.w;
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    v.x = s0 + s1; v.y = d0 + d1; v.z = s0 - s1; v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= HAD_SCALE; v.y *= HAD_SCALE; v.z *= HAD_SCALE; v.w *= HAD_SCALE;
    half4 sc = ((const half4*) scale)[t];
    v.x *= __low2float(sc.x); v.y *= __high2float(sc.x); v.z *= __low2float(sc.y); v.w *= __high2float(sc.y);
    return v;
}

__device__ __forceinline__ float wave_sum_down(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o, 32);
    return __shfl(v, 0, 32);
}

// K5 decode of one engine lane (t) from the lane set {wrap, own0..own4}: same arithmetic as the engine's dq4<5, cb> twice (values 0..3, 4..7)
__device__ __forceinline__ void mf_dq8_k5(const uint32_t (&w)[6], const int t, FragB& f0, FragB& f1)
{
    uint32_t v[8];
    #pragma unroll
    for (int g = 0; g < 2; ++g)
    {
        const uint32_t a = w[k5_i0(t, g)], b = w[k5_i2(t, g)];
        const int s2 = k5_s2(t, g);
        v[4 * g + 3] = fshift(b, a, s2) & 0xffff;
        v[4 * g + 2] = fshift(b, a, s2 + 5) & 0xffff;
        v[4 * g + 1] = fshift(b, a, s2 + 10) & 0xffff;
        v[4 * g + 0] = fshift(b, a, s2 + 15) & 0xffff;
    }
    exl3_gemv_ns::decode8<2>(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], f0, f1);
}

// ---------------------------------------------------------------- VALU tile on a trellis tile (native or repacked layout)
// One wave = one 16-column tile, all k. Same decode body as core3's mv_tile; Bt = base of the tile's first k-slice, ss = u32 between k-slices.
// Lane (cq, g8): g8 = column pair (g8, g8 + 8), cq = chunk lane. Words: wrap word + `BITS` own words; K3 own words are three scalar loads (3 g8 is not
// 8-byte aligned). K3 decode: lane (g8, t) is the engine's lane 4 g8 + t, whose (a, b, s2) of dq8_regs_3bits are (w[0], w[1], 8), (w[1], w[2], 16),
// (w[2], w[3], 24), (w[2], w[3], 0) in the order wrap, own0, own1, own2 (mf_k3_map in the CPU proof re-derives this from the engine's formulas).
template <int BITS, bool CF32, int RW, int KS>
__device__ __forceinline__ void mf_tile(const uint32_t* __restrict__ A32, const uint32_t* __restrict__ Bt, long ss, void* __restrict__ C,
                                        int size_n, int tile, float* red, int wave, int lane)
{
    static_assert(BITS >= 2 && BITS <= 5, "K1, K6.. need their own per-lane word set and decode body: not implemented");
    constexpr int NWD = BITS + 1, CH = (KS + 15) / 16, NRND = 4, K = KS * 16;
    constexpr int VU = CH > 5 ? 5 : CH;
    const int g8 = lane & 7, cq = lane >> 3;
    float rs0[RW], rs1[RW];
    #pragma unroll
    for (int r = 0; r < RW; ++r) { rs0[r] = 0.f; rs1[r] = 0.f; }
    uint32_t wa[VU][NWD], wb[VU][NWD];
    auto ld = [&](uint32_t (&w)[NWD], int s)
    {
        w[0] = Bt[wrp_i(BITS, g8, s, ss)];
        const uint32_t* q = Bt + own_i(BITS, g8, s, ss);
        if constexpr (BITS == 2) { const uint2 v = *(const uint2*) q; w[1] = v.x; w[2] = v.y; }
        else if constexpr (BITS == 3) { w[1] = q[0]; w[2] = q[1]; w[3] = q[2]; }
        else if constexpr (BITS == 5) { w[1] = q[0]; w[2] = q[1]; w[3] = q[2]; w[4] = q[3]; w[5] = q[4]; }
        else { const uint4 v = *(const uint4*) q; w[1] = v.x; w[2] = v.y; w[3] = v.z; w[4] = v.w; }
    };
    auto ldst = [&](uint32_t (&w)[VU][NWD], int rnd, int sg0)
    {
        const int cc = 4 * rnd + cq, s0n = cc * CH, m = myn(KS, cc);
        #pragma unroll
        for (int u = 0; u < VU; ++u) if (sg0 + u < m) ld(w[u], s0n + sg0 + u);
    };
    ldst(wa, 0, 0);
    #pragma unroll 1
    for (int round = 0; round < NRND; ++round)
    {
        const int c = 4 * round + cq;
        const int s0 = c * CH;
        const int myn_ = myn(KS, c);
        float acc0[RW], acc1[RW];
        #pragma unroll
        for (int r = 0; r < RW; ++r) { acc0[r] = 0.f; acc1[r] = 0.f; }
        #pragma unroll 1
        for (int sb = 0; sb < CH; sb += VU)
        {
            int nr = round, ns = sb + VU;
            if (ns >= CH) { nr = round + 1; ns = 0; }
            if (nr < NRND) ldst(wb, nr, ns);
            #pragma unroll
            for (int u = 0; u < VU; ++u)
            {
                if (sb + u < myn_)
                {
                    FragB f0[4], f1[4];
                    #pragma unroll
                    for (int t = 0; t < 4; ++t)
                    {
                        if constexpr (BITS == 2) exl3_gemv_ns::dq8_regs_2bits<2>(wa[u][t >> 1], wa[u][1 + (t >> 1)], t << 3, f0[t], f1[t]);
                        else if constexpr (BITS == 3) exl3_gemv_ns::dq8_regs_3bits<2>(wa[u][t == 0 ? 0 : (t == 1 ? 1 : 2)], wa[u][t == 0 ? 1 : (t == 1 ? 2 : 3)], t == 0 ? 8 : (t == 1 ? 16 : (t == 2 ? 24 : 0)), f0[t], f1[t]);
                        else if constexpr (BITS == 5) mf_dq8_k5(wa[u], t, f0[t], f1[t]);
                        else exl3_gemv_ns::dq8_regs_4bits<2>(wa[u][t], wa[u][t + 1], f0[t], f1[t]);
                    }
                    #pragma unroll
                    for (int r = 0; r < RW; ++r)
                    {
                        const long ai = a_idx(r, K, s0 + sb + u);
                        const uint4 alo = *(const uint4*) (A32 + ai);
                        const uint4 ahi = *(const uint4*) (A32 + ai + 4);
                        const uint32_t aq[8] = {alo.x, alo.y, alo.z, alo.w, ahi.x, ahi.y, ahi.z, ahi.w};
                        #pragma unroll
                        for (int q = 0; q < 8; ++q)
                        {
                            const int t = q & 3, h = q >> 2;
                            acc0[r] = mv_dot2(__builtin_bit_cast(uint32_t, f0[t][h]), aq[q], acc0[r]);
                            acc1[r] = mv_dot2(__builtin_bit_cast(uint32_t, f1[t][h]), aq[q], acc1[r]);
                        }
                    }
                }
            }
            #pragma unroll
            for (int u = 0; u < VU; ++u)
                #pragma unroll
                for (int w = 0; w < NWD; ++w) wa[u][w] = wb[u][w];
        }
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            red[red_idx(wave, cq, g8, r)] = acc0[r];
            red[red_idx(wave, cq, g8 + 8, r)] = acc1[r];
        }
        __syncwarp();
        if (cq == 0)
        {
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc)
                #pragma unroll
                for (int r = 0; r < RW; ++r)
                {
                    rs0[r] += red[red_idx(wave, cc, g8, r)];
                    rs1[r] += red[red_idx(wave, cc, g8 + 8, r)];
                }
        }
        __syncwarp();
    }
    if (cq == 0)
    {
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            if constexpr (CF32) { ((float*) C)[c_idx(r, size_n, tile, g8)] = rs0[r]; ((float*) C)[c_idx(r, size_n, tile, g8 + 8)] = rs1[r]; }
            else { ((half*) C)[c_idx(r, size_n, tile, g8)] = __float2half_rn(rs0[r]); ((half*) C)[c_idx(r, size_n, tile, g8 + 8)] = __float2half_rn(rs1[r]); }
        }
    }
}

template <int BITS, bool CF32, int KS>
__device__ __forceinline__ void mf_tile_rows(int rows, const uint32_t* A32, const uint32_t* Bt, long ss, void* C, int size_n, int tile, float* red, int wave, int lane)
{
    if (rows == 1) mf_tile<BITS, CF32, 1, KS>(A32, Bt, ss, C, size_n, tile, red, wave, lane);
    else if (rows == 2) mf_tile<BITS, CF32, 2, KS>(A32, Bt, ss, C, size_n, tile, red, wave, lane);
    else if (rows == 3) mf_tile<BITS, CF32, 3, KS>(A32, Bt, ss, C, size_n, tile, red, wave, lane);
    else mf_tile<BITS, CF32, 4, KS>(A32, Bt, ss, C, size_n, tile, red, wave, lane);
}

// ---------------------------------------------------------------- phase 0: HC dots (128-thread teams)
template <class S>
__device__ void ph_dots(const Params& p, Smem<S>& s)
{
    MF_DIMS
    const int R = p.R, team = threadIdx.x >> 7, tid = threadIdx.x & 127, lane = tid & 31, warp = tid >> 5;
    const int D4 = D / 4;
    const int rounds = (MR + 1 + p.G * 4 - 1) / (p.G * 4);
    for (int k = 0; k < rounds; ++k)
    {
        const int j = blockIdx.x * 4 + team + k * p.G * 4;
        const bool valid = j < MR + 1;
        if (valid)
        {
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                float a[MAXR];
                #pragma unroll
                for (int r = 0; r < MAXR; ++r) a[r] = 0.0f;
                if (j < MR)
                {
                    const int2* f8 = (const int2*) (p.fn + fn_row<S>(j, h));
                    for (int c = tid; c < D4 / 2; c += 128)
                    {
                        int2 pk = f8[c];
                        float w0, w1, w2, w3, w4, w5, w6, w7;
                        if (p.gs)
                        {
                            const float gsc = p.fn_scale[fngs_idx<S>(j, h, c)];
                            gs_unpack_s8x4((uint32_t) pk.x, gsc, w0, w1, w2, w3);
                            gs_unpack_s8x4((uint32_t) pk.y, gsc, w4, w5, w6, w7);
                        }
                        else
                        {
                            unpack_s8x4((uint32_t) pk.x, w0, w1, w2, w3);
                            unpack_s8x4((uint32_t) pk.y, w4, w5, w6, w7);
                        }
                        #pragma unroll
                        for (int r = 0; r < MAXR; ++r)
                        {
                            if (r < R)
                            {
                                const float4* s4 = (const float4*) (p.xin + xin_row<S>(r, 0));
                                float4 s0 = s4[(size_t) h * D4 + 2 * c];
                                float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                                float x = a[r];
                                x = fmaf(s0.x, w0, x); x = fmaf(s0.y, w1, x); x = fmaf(s0.z, w2, x); x = fmaf(s0.w, w3, x);
                                x = fmaf(s1.x, w4, x); x = fmaf(s1.y, w5, x); x = fmaf(s1.z, w6, x); x = fmaf(s1.w, w7, x);
                                a[r] = x;
                            }
                        }
                    }
                }
                else
                {
                    for (int c = tid; c < D4; c += 128)
                    {
                        #pragma unroll
                        for (int r = 0; r < MAXR; ++r)
                            if (r < R)
                            {
                                const float4* s4 = (const float4*) (p.xin + xin_row<S>(r, 0));
                                float4 sv = s4[(size_t) h * D4 + c];
                                a[r] = fmaf(sv.x, sv.x, fmaf(sv.y, sv.y, fmaf(sv.z, sv.z, fmaf(sv.w, sv.w, a[r]))));
                            }
                    }
                }
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                {
                    float v = a[r];
                    for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o, 32);
                    if (lane == 0 && r < R) s.u.h.dred[team][r][h][warp] = v;
                }
            }
        }
        __syncthreads();
        if (valid && tid < H * R)
        {
            const int r = tid / H, h = tid % H;
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < 4; ++w) v += s.u.h.dred[team][r][h][w];
            if (j < MR && !p.gs) v *= p.fn_scale[j];
            p.dots[dots_idx<S>(r, j, h)] = v;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- phase 1: HC finalize (256-thread teams, one 32-column chunk per item, all R rows)
// The int8 up-projection words of a chunk are loaded once and used for every row (core4 re-read them R times); per row and output the
// accumulation order is unchanged (ii ascending per lane, then the xor tree).
template <class S>
__device__ void ph_finalize(const Params& p, Smem<S>& s)
{
    MF_DIMS
    const int R = p.R, team = threadIdx.x >> 8, tid = threadIdx.x & 255, lane = tid & 31, warp = tid >> 5;
    const int D4 = D / 4;
    constexpr int CH = 32;                     // chunk_cols
    const int total = DM::FIN_CHUNKS;
    const int rounds = (total + p.G * 2 - 1) / (p.G * 2);
    float* t_s = s.u.h.t[team];                // [MAXR][LR]
    float* rmr_s = s.u.h.rmr[team];            // [MAXR][4]
    const float inv_h = 1.0f / (float) H;
    for (int k = 0; k < rounds; ++k)
    {
        const int cidx = blockIdx.x * 2 + team + k * p.G * 2;
        const bool valid = cidx < total;
        if (valid && tid < H * R)
        {
            const int r = tid / H, h = tid % H;
            rmr_s[r * 4 + h] = rsqrtf(p.dots[dots_idx<S>(r, MR, h)] / (float) D + p.rms_eps);
        }
        __syncthreads();
        if (valid)
        {
            for (int idx = tid; idx < LR * R; idx += 256)
            {
                const int r = idx / LR, ii = idx % LR;
                const float* dr = p.dots + dots_idx<S>(r, 0, 0);
                float v = 0.0f;
                #pragma unroll
                for (int h = 0; h < H; ++h) v = fmaf(rmr_s[r * 4 + h], dr[(size_t) ii * H + h], v);
                v *= inv_h;
                t_s[r * LR + ii] = v * sigmoidf_(v);
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
        if (valid)
        {
            const int c0 = cidx * CH, c1 = min(c0 + CH, D);
            for (int c = c0 / 4 + warp; c < c1 / 4; c += 8)
            {
                float4 g[MAXR][H];
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                    #pragma unroll
                    for (int h = 0; h < H; ++h) g[r][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 gsc[H];
                for (int ii = lane; ii < LR; ii += 32)
                {
                    float ti[MAXR];
                    #pragma unroll
                    for (int r = 0; r < MAXR; ++r) ti[r] = r < R ? t_s[r * LR + ii] : 0.0f;
                    if (p.gs && (((ii - lane) & 32) == 0))      // the 32-rank step that opens a 64-rank group: warp-uniform (lane < 32)
                    {
                        #pragma unroll
                        for (int h = 0; h < H; ++h) gsc[h] = *(const float4*) (p.up_scale + upgs_idx<S>(h, c, ii));
                    }
                    #pragma unroll
                    for (int h = 0; h < H; ++h)
                    {
                        uint32_t u = *(const uint32_t*) (p.upt + upt_u32<S>(h, c, ii) * 4);
                        float u0, u1, u2, u3;
                        unpack_s8x4(u, u0, u1, u2, u3);
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
                if (lane != 0) continue;
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
    }
}

// ---------------------------------------------------------------- phase 2: router logits + shared-gate logit
// The R mixed rows are staged once per block in LDS (the A region, unused until phase 3); every expert row then streams from global once.
template <class S>
__device__ void ph_router(const Params& p, Smem<S>& s)
{
    MF_DIMS
    const int R = p.R, wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    half* mx = s.u.g.A;                                         // [R][D] halves (A holds MAXR * D)
    for (int i = threadIdx.x; i < R * D / 8; i += THREADS) ((uint4*) mx)[i] = ((const uint4*) p.mixed)[i];
    __syncthreads();
    for (int e = wave * p.G + blockIdx.x; e < NEXP; e += 16 * p.G)
    {
        const half2* g2 = (const half2*) (p.router + router_row<S>(e));
        float sum[MAXR];
        #pragma unroll
        for (int r = 0; r < MAXR; ++r) sum[r] = 0.0f;
        #pragma unroll 8
        for (int col = lane; col < D / 2; col += 32)
        {
            const half2 g = g2[col];
            #pragma unroll
            for (int r = 0; r < MAXR; ++r)
                if (r < R)
                {
                    const half2 hh = ((const half2*) (mx + (size_t) r * D))[col];
                    sum[r] = fmaf(__half2float(__low2half(hh)), __half2float(__low2half(g)), sum[r]);
                    sum[r] = fmaf(__half2float(__high2half(hh)), __half2float(__high2half(g)), sum[r]);
                }
        }
        #pragma unroll
        for (int r = 0; r < MAXR; ++r)
        {
            float v = wave_sum_down(sum[r]);
            if (lane == 0 && r < R) p.scores[scores_idx<S>(r, e)] = __float2half_rn(v);
        }
    }
    if (blockIdx.x == p.G - 1 && wave == 15)
    {
        for (int r = 0; r < R; ++r)
        {
            // replay of the engine's add_sigmoid_gate_proj reduction: 1024 virtual threads t = lane + 32 j, strided fma chain,
            // shared-memory tree strides 512..32 (register folds over j), then strides 16..1 (shuffles)
            float pt[32];
            #pragma unroll
            for (int j = 0; j < 32; ++j)
            {
                float a = 0.0f;
                for (int i = lane + 32 * j; i < D; i += 1024)
                    a = fmaf(__half2float(p.sgate_w[i]), __half2float(mx[(size_t) r * D + i]), a);
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

// ---------------------------------------------------------------- top-k (one wave per row), redundant per block
template <class S>
__device__ void ph_topk(const Params& p, Smem<S>& s)
{
    MF_DIMS
    constexpr int KEYS = DM::KEYS;
    const int R = p.R, wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (wave < R)
    {
        float key[KEYS];
        #pragma unroll
        for (int i = 0; i < KEYS; ++i)
        {
            // NaN and -inf scores become the lowest finite key: they stay selectable (lowest index first), so a row always picks TOPK distinct
            // experts and a group never holds more than R rows (the A staging buffers hold MAXR rows)
            const float kv = __half2float(p.scores[scores_idx<S>(wave, i * 32 + lane)]);
            key[i] = (kv > -__FLT_MAX__) ? kv : -__FLT_MAX__;
        }
        float gmax = 0.0f, ek = 0.0f;
        int myidx = 0;
        #pragma unroll
        for (int rank = 0; rank < TOPK; ++rank)
        {
            float bk = -INFINITY; int bi = 0x7fffffff;
            #pragma unroll
            for (int i = 0; i < KEYS; ++i)
                if (key[i] > bk) { bk = key[i]; bi = i * 32 + lane; }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                float ok = __shfl_xor(bk, o, 32); int oi = __shfl_xor(bi, o, 32);
                if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; }
            }
            if (rank == 0) gmax = bk;
            if (lane == rank) { ek = (bk == gmax) ? 1.0f : expf(bk - gmax); myidx = clamp_expert(bi, NEXP); }
            #pragma unroll
            for (int i = 0; i < KEYS; ++i)
                if (i * 32 + lane == bi) key[i] = -INFINITY;
        }
        const float e = lane < TOPK ? ek : 0.0f;
        const float sum = wave_sum_down(e) + 1.0e-20f;
        if (lane < TOPK)
        {
            s.sel[wave * TOPK + lane] = myidx;
            s.wt[wave * TOPK + lane] = __float2half_rn(e / sum);
        }
    }
    __syncthreads();
    const int NA = R * TOPK;
    if (threadIdx.x < NA)
    {
        const int a = threadIdx.x, e = s.sel[a];
        int pp = 0;
        for (int b = 0; b < NA; ++b) { const int eb = s.sel[b]; pp += (eb < e) || (eb == e && b < a); }
        s.sorted_e[pp] = e; s.sorted_a[pp] = a; s.tok[pp] = a / TOPK;
    }
    __syncthreads();
    if (threadIdx.x < NA) s.head[threadIdx.x] = (threadIdx.x == 0 || s.sorted_e[threadIdx.x - 1] != s.sorted_e[threadIdx.x]) ? 1 : 0;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int g = 0;
        for (int q = 0; q < NA; ++q) if (s.head[q]) s.gstart[g++] = q;
        s.gstart[g] = NA; s.ngroups = g;
        for (int u = 0; u < g; ++u) s.urows[u] = min(s.gstart[u + 1] - s.gstart[u], MAXR);
        s.urows[g] = R;                       // the shared expert: R rows
    }
    __syncthreads();
}

// ---------------------------------------------------------------- phases 3/4: one wave per 16-column tile.
// Segments (one (unit, proj) pair for gate/up, one unit for down) are processed two at a time with two A buffers, so the wave-strided
// tile list spans both and the block never idles between segments. The tile list is split between blocks by cost (idx.h split_tile).
struct Seg { int unit, proj, rows, p0, e; bool shared; const half* const* sv; };
template <class S>
__device__ __forceinline__ Seg seginfo(const Params& p, const Smem<S>& s, int seg, bool two_proj)
{
    using DM = Dm<S>;
    Seg g; g.unit = two_proj ? seg >> 1 : seg; g.proj = two_proj ? (seg & 1) : 0; g.shared = g.unit == s.ngroups; g.e = 0;
    if constexpr (TRows<S>::v > MAXR)
    {
        g.shared = g.unit >= s.ngroups;
        if (g.shared) { const int k = g.unit - s.ngroups; g.rows = shared_rows(p.R, k); g.p0 = shared_p0(p.R * DM::TOPK, k); g.sv = p.svt + 6 * DM::NEXP; }
        else { g.p0 = s.gstart[g.unit]; g.rows = min(s.gstart[g.unit + 1] - g.p0, MAXR); g.e = s.sorted_e[g.p0]; g.sv = p.svt + 6 * g.e; }
        return g;
    }
    if (g.shared) { g.rows = p.R; g.p0 = p.R * DM::TOPK; g.sv = p.svt + 6 * DM::NEXP; }
    else { g.p0 = s.gstart[g.unit]; g.rows = min(s.gstart[g.unit + 1] - g.p0, MAXR); g.e = s.sorted_e[g.p0]; g.sv = p.svt + 6 * g.e; }   // min: never active (distinct experts per row)
    return g;
}

template <class S>
__device__ void ph_gateup(const Params& p, Smem<S>& s)
{
    MF_DIMS
    const int R = p.R, NA = R * TOPK, NAR = NA + R;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NU = s.ngroups + 1;
    constexpr int TPP = DM::TPP;
    const bool nat = p.native != 0;
    const long long W = split_total(s.urows, NU, 2, TPP);
    const int lo = split_tile(s.urows, NU, 2, TPP, (long long) blockIdx.x * W / p.G), hi = split_tile(s.urows, NU, 2, TPP, (long long) (blockIdx.x + 1) * W / p.G);
    int i = lo;
    while (i < hi)
    {
        const int segA = i / TPP, endA = min(hi, (segA + 1) * TPP);
        const int iB = endA, segB = iB / TPP;
        const bool hasB = iB < hi;
        const int endB = hasB ? min(hi, (segB + 1) * TPP) : iB;
        const int nA = endA - i, nB = endB - iB;
        __syncthreads();
        for (int w2 = 0; w2 < (hasB ? 2 : 1); ++w2)
        {
            const int seg = w2 ? segB : segA;
            const Seg g = seginfo<S>(p, s, seg, true);
            half* Abuf = w2 ? s.u.g.A2 : s.u.g.A;
            const half* suh = g.sv[g.proj == 0 ? 0 : 2];
            for (int task = wave; task < g.rows * DM::CMB_BLOCKS; task += 16)
            {
                const int row = task / DM::CMB_BLOCKS, blk = task % DM::CMB_BLOCKS;
                const int token = g.shared ? row : s.tok[g.p0 + row];
                half4 v = ((const half4*) (p.mixed + mixed_row<S>(token) + blk * 128))[lane];
                v = had_regs_h<true, false>(v, suh + blk * 128, lane);
                ((half4*) (Abuf + a_row_gu<S>(row, blk)))[lane] = v;
            }
        }
        __syncthreads();
        for (int n = wave; n < nA + nB; n += 16)
        {
            const bool inB = n >= nA;
            const int seg = inB ? segB : segA;
            const int tile = (inB ? iB + (n - nA) : i + n) - seg * TPP;
            const Seg g = seginfo<S>(p, s, seg, true);
            const uint32_t* Ab = (const uint32_t*) (inB ? s.u.g.A2 : s.u.g.A);
            half* C = p.gu + gu_out<S>(g.proj, NAR, g.p0);
            const uint32_t* Wm = p.wt[3 * (g.shared ? DM::NEXP : g.e) + g.proj];
            if (g.shared)
                mf_tile_rows<DM::SB, false, DM::KSG>(g.rows, Ab, Wm + tile_off(nat, DM::SB, DM::KSG, TPP, tile), tile_ss(nat, DM::SB, TPP), C, INTER, tile, s.u.g.red, wave, lane);
            else
                mf_tile_rows<DM::RB, false, DM::KSG>(g.rows, Ab, Wm + tile_off(nat, DM::RB, DM::KSG, TPP, tile), tile_ss(nat, DM::RB, TPP), C, INTER, tile, s.u.g.red, wave, lane);
        }
        i = endB;
    }
}

template <class S>
__device__ void ph_down(const Params& p, Smem<S>& s)
{
    MF_DIMS
    const int R = p.R, NA = R * TOPK, NAR = NA + R;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NU = s.ngroups + 1;
    constexpr int TPU = DM::TPU;
    const bool nat = p.native != 0;
    const long long W = split_total(s.urows, NU, 1, TPU);
    const int lo = split_tile(s.urows, NU, 1, TPU, (long long) blockIdx.x * W / p.G), hi = split_tile(s.urows, NU, 1, TPU, (long long) (blockIdx.x + 1) * W / p.G);
    int i = lo;
    while (i < hi)
    {
        const int segA = i / TPU, endA = min(hi, (segA + 1) * TPU);
        const int iB = endA, segB = iB / TPU;
        const bool hasB = iB < hi;
        const int endB = hasB ? min(hi, (segB + 1) * TPU) : iB;
        const int nA = endA - i, nB = endB - iB;
        __syncthreads();
        for (int w2 = 0; w2 < (hasB ? 2 : 1); ++w2)
        {
            const Seg g = seginfo<S>(p, s, w2 ? segB : segA, false);
            half* Abuf = w2 ? s.u.g.A2 : s.u.g.A;
            const half* const* sv = g.sv;
            for (int task = wave; task < g.rows * DM::INT_BLOCKS; task += 16)
            {
                const int row = task / DM::INT_BLOCKS, blk = task % DM::INT_BLOCKS;
                half4 gg = ((const half4*) (p.gu + gu_read<S>(NAR, g.p0 + row, blk, 0)))[lane];
                half4 uu = ((const half4*) (p.gu + gu_read<S>(NAR, g.p0 + row, blk, 1)))[lane];
                gg = had_regs_h<false, true>(gg, sv[1] + blk * 128, lane);
                uu = had_regs_h<false, true>(uu, sv[3] + blk * 128, lane);
                half4 a;
                {
                    float f0 = __half2float(__low2half(gg.x)), f1 = __half2float(__high2half(gg.x));
                    float f2 = __half2float(__low2half(gg.y)), f3 = __half2float(__high2half(gg.y));
                    half h0 = __float2half_rn(f0 / (1.0f + expf(-f0))), h1 = __float2half_rn(f1 / (1.0f + expf(-f1)));
                    half h2 = __float2half_rn(f2 / (1.0f + expf(-f2))), h3 = __float2half_rn(f3 / (1.0f + expf(-f3)));
                    a.x = __halves2half2(__hmul(h0, __low2half(uu.x)), __hmul(h1, __high2half(uu.x)));
                    a.y = __halves2half2(__hmul(h2, __low2half(uu.y)), __hmul(h3, __high2half(uu.y)));
                }
                a = had_regs_h<true, false>(a, sv[4] + blk * 128, lane);
                ((half4*) (Abuf + a_row_dn<S>(row, blk)))[lane] = a;
            }
        }
        __syncthreads();
        for (int n = wave; n < nA + nB; n += 16)
        {
            const bool inB = n >= nA;
            const int seg = inB ? segB : segA;
            const int tile = (inB ? iB + (n - nA) : i + n) - seg * TPU;
            const Seg g = seginfo<S>(p, s, seg, false);
            const uint32_t* Ab = (const uint32_t*) (inB ? s.u.g.A2 : s.u.g.A);
            float* C = p.dn + dn_row<S>(g.p0);
            const uint32_t* Wm = p.wt[3 * (g.shared ? DM::NEXP : g.e) + 2];
            if (g.shared)
                mf_tile_rows<DM::SB, true, DM::KSD>(g.rows, Ab, Wm + tile_off(nat, DM::SB, DM::KSD, TPU, tile), tile_ss(nat, DM::SB, TPU), C, D, tile, s.u.g.red, wave, lane);
            else
                mf_tile_rows<DM::RB, true, DM::KSD>(g.rows, Ab, Wm + tile_off(nat, DM::RB, DM::KSD, TPU, tile), tile_ss(nat, DM::RB, TPU), C, D, tile, s.u.g.red, wave, lane);
        }
        i = endB;
    }
}

// ---------------------------------------------------------------- phase 5: post had, weighted sum, shared add, hc apply
template <class S>
__device__ void ph_combine(const Params& p, Smem<S>& s)
{
    MF_DIMS
    const int R = p.R, NA = R * TOPK;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int items = DM::CMB_BLOCKS * R;
    for (int i = blockIdx.x; i < items; i += p.G)
    {
        const int cb = i / R, r = i % R;
        if (wave <= TOPK)
        {
            const int a = r * TOPK + (wave < TOPK ? wave : 0);
            int pp = 0, e = 0;
            const half* const* sv;
            if (wave < TOPK)
            {
                for (int q = 0; q < NA; ++q) if (s.sorted_a[q] == a) { pp = q; break; }
                e = s.sorted_e[pp]; sv = p.svt + 6 * e;
            }
            else { pp = NA + r; sv = p.svt + 6 * NEXP; }
            float4 v = ((const float4*) (p.dn + dn_row<S>(pp) + cb * 128))[lane];
            v = had_regs_f_post(v, sv[5] + cb * 128, lane);
            ((float4*) (s.u.c.rows + wave * 128))[lane] = v;
        }
        __syncthreads();
        if (threadIdx.x < 128)
        {
            const int col = cb * 128 + threadIdx.x, c = threadIdx.x;
            float sum = 0.0f;
            for (int q = 0; q < NA; ++q)
            {
                const int a = s.sorted_a[q];
                if (a / TOPK != r) continue;
                const int slot = a % TOPK;
                const float weighted = __fmul_rn(s.u.c.rows[slot * 128 + c], __half2float(s.wt[a]));
                sum = __fadd_rn(sum, weighted);
            }
            const float syw = 1.0f / (1.0f + __expf(-p.sgl[r]));
            if (!(syw < 1e-8f)) sum += s.u.c.rows[TOPK * 128 + c] * syw;
            p.ydbg[(size_t) r * D + col] = sum;
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                const float po = p.post[r * H + h];
                const float xv = p.xin[xout_idx<S>(r, h, col)];
                float o = po * sum;
                o += xv;
                p.xout[xout_idx<S>(r, h, col)] = o;
            }
        }
        __syncthreads();
    }
}

// Test entry (core5): all tiles of one K x N matrix (R rows of A already in the Hadamard domain, row-major R x K halves) through mf_tile, one block.
// Used to compare the tile body, native and repack addressing and K3 against the engine's exl3_gemv on synthetic data of real shapes.
template <int BITS, bool CF32, int KS>
__global__ __launch_bounds__(THREADS) void mf_tile_test_kernel(const uint32_t* A, const uint32_t* B, void* C, int R, int nt, int native)
{
    __shared__ float red[RED_SIZE];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool nat = native != 0;
    for (int tile = wave; tile < nt; tile += 16)
        mf_tile_rows<BITS, CF32, KS>(R, A, B + tile_off(nat, BITS, KS, nt, tile), tile_ss(nat, BITS, nt), C, nt * 16, tile, red, wave, lane);
}

template <class S>
__global__ __launch_bounds__(THREADS) void moe_fused_kernel(Params p)
{
    __shared__ Smem<S> s;
    int phase = 0;
    mf_stamp(p, 0);
    ph_dots<S>(p, s);        gbar(p, phase);
    ph_finalize<S>(p, s);    gbar(p, phase);
    ph_router<S>(p, s);      gbar(p, phase);
    ph_topk<S>(p, s);
    if (blockIdx.x == 0 && threadIdx.x < p.R * S::TOPK) p.seldbg[threadIdx.x] = s.sel[threadIdx.x];
    if (blockIdx.x == 0 && threadIdx.x < p.R * S::TOPK) p.seldbg[64 + threadIdx.x] = (int) __half_as_ushort(s.wt[threadIdx.x]);
    ph_gateup<S>(p, s);      gbar(p, phase);
    ph_down<S>(p, s);        gbar(p, phase);
    ph_combine<S>(p, s);
    mf_stamp(p, 11);
    gfinish(p);
}

}  // namespace mf
#endif
