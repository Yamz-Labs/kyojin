// KDA fused chunk recurrence (o fused), hand-written for gfx11.5 WMMA (Strix Halo, wave32).
// Same math and the same rounding points as chunk_kda_fwd_kernel_h_fused (FUSE_O) in kda_fused_h.py,
// so o and the final state match it bitwise: bf16 x bf16 products truncated, casts RNE, raw v_exp_f32,
// fp32 WMMA accumulation chained in the same k order, no fma contraction (build with -ffp-contract=off).
//
// One workgroup (8 waves) per (batch, value head); K = V = 128, BT = 64. Wave i owns the state columns
// h[:, 16i:16i+16] in registers (8 WMMA C tiles) and computes w[:, 16i:16i+16] and u[:, 16i:16i+16].
// Per chunk: inputs -> LDS (kb^T, vb^T, Akk) -> w, u -> LDS (w, qg, kg^T, Aqk) -> o, v_new, state update.
// All global loads of a chunk issue in one batch (Aqk overlaps the w / u products); 4 barriers per chunk.
// Fragment layout (gfx1151, oracle-verified in exllamav3_ext/hip/hip_mma.cuh):
//   A: lane l, elem r -> A[l & 15][r];  B: lane l, elem r -> B[r][l & 15];  C: lane l, reg r -> C[2r + (l >> 4)][l & 15]

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v8u __attribute__((ext_vector_type(8)));

#ifndef ABL
#define ABL 0  // timing-only ablation bits (wrong results): 1 phase-1 loads, 2 w/u, 4 o/vn chain, 8 tril o + store, 16 state update
#endif
#define BT 64
#define KD 128
#define VD 128
#define LDR 136  // row-major [token][kdim] buffers (bf16 elements per row)
#define LDT 72   // transposed [dim][token] buffers

#ifndef PH1
#define PH1 0  // 1: phase 1 after the barrier: batched raw loads, kb^T / vb^T stored straight to LDS
#endif
#ifndef PF
#define PF 0  // PH1 only: raw loads of chunk c+1 token group 0 issued during chunk c (1: before the state update, 2: before o / vn)
#endif
#ifndef PFG
#define PFG 1  // PF: number of PH1 token groups prefetched
#endif
#if PF && !PH1
#error "PF needs PH1"
#endif
// raw per-token loads of NG_ tokens from row TROW (rows clamped to T - 1: always a valid address)
#define LDG(RK, RQ, RV, RG, RB, TROW)                                                   \
    _Pragma("unroll") for (int j = 0; j < NG; ++j)                                    \
    {                                                                                  \
        const int64_t tc = bos + min((TROW) + j, T - 1);                               \
        RK[j] = *(const uint32_t*) (k + (tc * H + i_h) * KD + 2 * cp);                 \
        RQ[j] = *(const uint32_t*) (q + (tc * H + i_h) * KD + 2 * cp);                 \
        RV[j] = *(const uint32_t*) (v + (tc * HV + i_hv) * VD + 2 * cp);               \
        RG[j] = *(const float2*) (gk + (tc * HV + i_hv) * KD + 2 * cp);                \
        RB[j] = beta[tc * HV + i_hv];                                                  \
    }
#ifndef UNI
#define UNI 0  // 1: wave-uniform token rows in phase 1 and o stores (scalar row address, lane offset only)
#endif
#ifndef ADR
#define ADR 0  // 1: o store / loads addressed from one uniform chunk base + opaque lane offset (same bits)
#endif
#ifndef LB
#define LB 0  // non-PH1 phase 1: raw loads batched LB tokens at a time (2, 4, 8, 16) behind a sched_barrier
#endif
#ifndef SB
#define SB 0  // bit mask of sched_barrier(0) sites (register pressure, same bits): 1 o/vn kt loop, 2 state kt loop, 4 w/u a loop, 8 tril a loop
#endif
#ifndef OPQ
#define OPQ 0  // bit mask: LDS fragment rows from a per-iteration opaque copy of the lane index, so the scheduler
               // cannot hoist a later phase's LDS loads (spills): 1 o/vn kt loop, 2 state kt loop, 4 w/u, 8 tril
#endif
__device__ __forceinline__ int pin(int x) { asm volatile("" : "+v"(x)); return x; }
__device__ __forceinline__ void pinv(v8f& x) { asm volatile("" : "+v"(x)); }
#ifndef OPV
#define OPV 0  // bit mask: pin WMMA accumulators at loop-iteration ends (ISel cannot sink the WMMAs past the next
               // phase's LDS loads): 1 o/vn kt loop, 2 state kt loop, 8 tril a loop
#endif
#define PIN(b, x) ((OPQ & (b)) ? pin(x) : (x))
#define SBAR(b) do { if (SB & (b)) __builtin_amdgcn_sched_barrier(0); } while (0)
#ifndef BRL
#define BRL 0  // 1: branchless f2bf (mask select) + uniform o-store fast path (same bits)
#endif
__device__ __forceinline__ uint32_t f2bf(float f)
{
    uint32_t u = __float_as_uint(f);
#if BRL
    // mask select, no branch: keeps the chunk loop one basic block so loads can batch
    const uint32_t n = ((u >> 16) | 0x40u) & 0xffffu;
    const uint32_t r = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    const uint32_t m = 0u - (uint32_t) ((u & 0x7fffffffu) > 0x7f800000u);
    return (r & ~m) | (n & m);
#else
    if ((u & 0x7fffffffu) > 0x7f800000u) return ((u >> 16) | 0x40u) & 0xffffu;
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
#endif
}
__device__ __forceinline__ float bf2f(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
// Triton rounds explicit .to(bf16) casts to nearest even but truncates the result of bf16 x bf16 arithmetic
// (k * beta, v * beta); oracle-checked against the Triton kernel on a captured GLM chunk.
__device__ __forceinline__ uint32_t f2bf_rtz(float f) { return __float_as_uint(f) >> 16; }
__device__ __forceinline__ float ex2(float x) { return __builtin_amdgcn_exp2f(x); }

__device__ __forceinline__ v8f wmma(v8u a, v8u b, v8f c)
{
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(v16s, a), __builtin_bit_cast(v16s, b), c);
}

// 16 contiguous bf16 at base[row * ld + col] (32 B, 16 B aligned)
__device__ __forceinline__ v8u ld16(const uint16_t* base, int row, int ld, int col)
{
    const uint4* p = (const uint4*) (base + row * ld + col);
    uint4 x = p[0], y = p[1];
    v8u r = {x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
    return r;
}

// p must be a valid address (clamped row); zeroed after the load when !ok
__device__ __forceinline__ v8u ld16g(const uint16_t* p, bool ok)
{
    uint4 x = ((const uint4*) p)[0], y = ((const uint4*) p)[1];
    v8u r = {x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
    if (!ok) r = (v8u) {0, 0, 0, 0, 0, 0, 0, 0};
    return r;
}

// C tile (fp32, rows 2r + half) -> B fragment of the same bf16-rounded matrix (k = row), via permlanex16.
__device__ __forceinline__ v8u c_to_b(const v8f& c, int half)
{
    uint32_t own[4], oth[4];
    #pragma unroll
    for (int m = 0; m < 4; ++m)
    {
        own[m] = f2bf(c[2 * m]) | (f2bf(c[2 * m + 1]) << 16);
        oth[m] = __builtin_amdgcn_permlanex16(own[m], own[m], 0x76543210u, 0xfedcba98u, false, false);
    }
    v8u b;
    #pragma unroll
    for (int d = 0; d < 8; ++d)
    {
        uint32_t o = (own[d >> 1] >> (16 * (d & 1))) & 0xffffu;
        uint32_t t = (oth[d >> 1] >> (16 * (d & 1))) & 0xffffu;
        b[d] = half ? (t | (o << 16)) : (o | (t << 16));
    }
    return b;
}

#ifndef PROF
#define PROF 0  // 1: per-phase shader-cycle sums (20-bit counter deltas) per (WG, wave) into kda_prof
#endif
#if PROF
__device__ uint32_t kda_prof[4096 * 9];
#define PT(n) { const uint32_t _t = (uint32_t) __builtin_readcyclecounter(); pacc[n] += (_t - plast) & 0xfffffu; plast = _t; }
#else
#define PT(n)
#endif

#ifndef PFW
#define PFW 0  // 1..4: a 9th wave touches chunk c+1's rows (L2 / MALL prefetch) after barrier PFW of chunk c
#endif

extern "C" __global__ __launch_bounds__(PFW ? 288 : 256) void kda_fused_h_o(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const float* __restrict__ gk, const uint16_t* __restrict__ beta, const uint16_t* __restrict__ Akk,
    const uint16_t* __restrict__ Aqk, uint16_t* __restrict__ o, const float* __restrict__ h0,
    float* __restrict__ ht, float scale, int T, int H, int HV, float* __restrict__ hc, int ckpt_chunk,
    float* __restrict__ hc2, int ckpt_chunk2)
{
    __shared__ __attribute__((aligned(16))) uint16_t sW[BT * LDR];   // w [token][kdim]
    __shared__ __attribute__((aligned(16))) uint16_t sX[KD * LDT];   // kb^T [kdim][token], then qg [token][kdim]
    __shared__ __attribute__((aligned(16))) uint16_t sY[KD * LDT];   // vb^T [v][token], then kg^T [kdim][token]
    __shared__ __attribute__((aligned(16))) uint16_t sA[BT * LDT];   // Akk [token][token], then Aqk
    __shared__ float sEgn[KD];                                        // exp2(g_last)

    const int tid = threadIdx.x, wv = tid >> 5, lane = tid & 31, ll = lane & 15, half = lane >> 4;
    const int i_nh = blockIdx.x, i_n = i_nh / HV, i_hv = i_nh % HV, i_h = i_hv / (HV / H);
    const int64_t bos = (int64_t) i_n * T;
    const int NT = (T + BT - 1) / BT;
#if PFW
    if (wv == 8)  // prefetch wave: same barrier count as the compute waves (4 per chunk), loads never consumed
    {
        uint32_t d[26];
        #pragma unroll
        for (int n = 0; n < 26; ++n) d[n] = 0;
        for (int i_t = 0; i_t < NT; ++i_t)
            #pragma unroll
            for (int b = 0; b < 4; ++b)
            {
                __syncthreads();
                if (b == PFW - 1 && i_t + 1 < NT)
                {
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");  // previous touches (a chunk old) done
                    #pragma unroll
                    for (int n = 0; n < 26; ++n) asm volatile("" :: "v"(d[n]));
                    #pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        const int64_t tc = bos + min((i_t + 1) * BT + lane + 32 * e, T - 1);
                        const void* pp[13] = {k + (tc * H + i_h) * KD, k + (tc * H + i_h) * KD + 64,
                                              q + (tc * H + i_h) * KD, q + (tc * H + i_h) * KD + 64,
                                              v + (tc * HV + i_hv) * VD, v + (tc * HV + i_hv) * VD + 64,
                                              gk + (tc * HV + i_hv) * KD, gk + (tc * HV + i_hv) * KD + 32,
                                              gk + (tc * HV + i_hv) * KD + 64, gk + (tc * HV + i_hv) * KD + 96,
                                              beta + tc * HV + i_hv, Akk + (tc * HV + i_hv) * BT,
                                              Aqk + (tc * HV + i_hv) * BT};
                        #pragma unroll
                        for (int n = 0; n < 13; ++n)
                            asm volatile("global_load_dword %0, %1, off" : "=&v"(d[13 * e + n]) : "v"(pp[n]));
                    }
                }
            }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        #pragma unroll
        for (int n = 0; n < 26; ++n) asm volatile("" :: "v"(d[n]));
        return;
    }
#endif
#ifdef STG
    // timing experiment: odd WGs start ~STG x 8k clocks late so their DRAM bursts (phase 1) fall in the others' compute
    if (blockIdx.x & 1)
        for (int i = 0; i < STG; ++i) __builtin_amdgcn_s_sleep(127);
#endif
    const int cp = tid & 63, tg = tid >> 6;   // loader: column pair 2cp, tokens 16tg..16tg+15
    const int col = 16 * wv + ll;             // this lane's state / value column
    const int ar = tid >> 2, ac = 16 * (tid & 3);  // A-matrix staging: row, 16-column slice

    v8f h[8];
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt)
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            h[kt][r] = h0 ? h0[(int64_t) i_nh * KD * VD + (16 * kt + 2 * r + half) * VD + col] : 0.0f;

#if PH1
    constexpr int NG = 16 / PH1;
#endif
#if PF
    uint32_t pk[NG * PFG], pq[NG * PFG], pv[NG * PFG], pb[NG * PFG];
    float2 pg[NG * PFG];
    #pragma unroll
    for (int g_ = 0; g_ < PFG; ++g_) LDG((pk + NG * g_), (pq + NG * g_), (pv + NG * g_), (pg + NG * g_), (pb + NG * g_), 16 * tg + NG * g_);
#endif
#if PROF
    uint32_t pacc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t plast = (uint32_t) __builtin_readcyclecounter();
#endif
    for (int i_t = 0; i_t < NT; ++i_t)
    {
        const int t0 = i_t * BT;
        const int last = min(t0 + BT, T) - 1;
        const float2 gn = *(const float2*) (gk + ((bos + last) * HV + i_hv) * KD + 2 * cp);

        // ---- phase 1: per-token products in registers (+ this thread's Akk slice)
        const v8u akk = ld16g(Akk + ((bos + min(t0 + ar, T - 1)) * HV + i_hv) * BT + ac, t0 + ar < T);
        uint32_t kbp[2][8], vbp[2][8], kgp[2][8], qgp[16];
#if UNI
        // tg is wave-uniform: token rows (and their row pointers) live in SGPRs, loads use saddr + one lane offset
        const int tgu = __builtin_amdgcn_readfirstlane(tg);
#else
        const int tgu = tg;
#endif
#if PH1
        PT(0);
        __syncthreads();  // previous chunk done with sX, sY, sEgn, sW
        PT(1);
        // all 16 tokens' raw loads first (one DRAM round trip), then convert and store kb^T / vb^T straight to LDS:
        // no packed kb / vb registers held across the barrier
        // PH1 = number of token groups: each group's raw loads issue back to back, then convert (peak 96 / PH1 VGPRs)
        #pragma unroll
        for (int gi = 0; gi < PH1; ++gi)
        {
            uint32_t rk[NG], rq[NG], rv[NG], rb[NG];
            float2 rg[NG];
#if PF
            if (gi < PFG)  // groups 0..PFG-1 were prefetched during the previous chunk
            {
                #pragma unroll
                for (int j = 0; j < NG; ++j) { rk[j] = pk[NG * gi + j]; rq[j] = pq[NG * gi + j]; rv[j] = pv[NG * gi + j]; rg[j] = pg[NG * gi + j]; rb[j] = pb[NG * gi + j]; }
            }
            else
#endif
            LDG(rk, rq, rv, rg, rb, t0 + 16 * tgu + NG * gi);
            #pragma unroll
            for (int s2 = NG * gi / 2; s2 < NG * (gi + 1) / 2; ++s2)
            {
                uint32_t kb[2][2], vb[2][2], kg[2][2];
                #pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    const int s = 2 * s2 + e, j = s - NG * gi;
                    const bool ok = t0 + 16 * tgu + s < T;
                    uint32_t kk = rk[j], qq = rq[j], vv = rv[j], bb = rb[j];
                    float2 g = rg[j];
                    if (!ok) { kk = qq = vv = bb = 0; g = make_float2(0.f, 0.f); }
                    const float fb = bf2f(bb), eg0 = ex2(g.x), eg1 = ex2(g.y);
                    const float k0 = bf2f(kk & 0xffff), k1 = bf2f(kk >> 16);
                    const float q0 = bf2f(qq & 0xffff), q1 = bf2f(qq >> 16);
                    kb[0][e] = f2bf(bf2f(f2bf_rtz(k0 * fb)) * eg0);
                    kb[1][e] = f2bf(bf2f(f2bf_rtz(k1 * fb)) * eg1);
                    vb[0][e] = f2bf_rtz(bf2f(vv & 0xffff) * fb);
                    vb[1][e] = f2bf_rtz(bf2f(vv >> 16) * fb);
                    kg[0][e] = f2bf(k0 * (ok ? ex2(gn.x - g.x) : 0.0f));
                    kg[1][e] = f2bf(k1 * (ok ? ex2(gn.y - g.y) : 0.0f));
                    qgp[s] = f2bf(q0 * eg0) | (f2bf(q1 * eg1) << 16);
                }
                #pragma unroll
                for (int c = 0; c < 2; ++c)
                {
                    *(uint32_t*) (sX + (2 * cp + c) * LDT + 16 * tg + 2 * s2) = kb[c][0] | (kb[c][1] << 16);
                    *(uint32_t*) (sY + (2 * cp + c) * LDT + 16 * tg + 2 * s2) = vb[c][0] | (vb[c][1] << 16);
                    kgp[c][s2] = kg[c][0] | (kg[c][1] << 16);
                }
            }
        }
#else
#if LB
        // LB: all LB tokens' raw loads issue back to back (one DRAM round trip per LB tokens), then convert
        uint32_t lk[16], lq[16], lv[16], lb_[16];
        float2 lg[16];
#endif
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2)
        {
#if LB
            if ((2 * s2) % LB == 0)
            {
                #pragma unroll
                for (int j = 2 * s2; j < 2 * s2 + LB; ++j)
                {
                    const int64_t tc = bos + min(t0 + 16 * tgu + j, T - 1);
                    lk[j] = *(const uint32_t*) (k + (tc * H + i_h) * KD + 2 * cp);
                    lq[j] = *(const uint32_t*) (q + (tc * H + i_h) * KD + 2 * cp);
                    lv[j] = *(const uint32_t*) (v + (tc * HV + i_hv) * VD + 2 * cp);
                    lg[j] = *(const float2*) (gk + (tc * HV + i_hv) * KD + 2 * cp);
                    lb_[j] = beta[tc * HV + i_hv];
                }
                __builtin_amdgcn_sched_barrier(0);
            }
#endif
            uint32_t kb[2][2], vb[2][2], kg[2][2];
            #pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                const int tt = 16 * tgu + 2 * s2 + e, t = t0 + tt;
                const bool ok = t < T;
#if LB
                uint32_t kk = lk[2 * s2 + e], qq = lq[2 * s2 + e], vv = lv[2 * s2 + e], bb = lb_[2 * s2 + e];
                float2 g = lg[2 * s2 + e];
#else
                // branch-free: clamp the row, zero after the load, so all loads issue back to back
                const int64_t tc = bos + min(t, T - 1);
                uint32_t kk = (ABL & 1) ? 0u : *(const uint32_t*) (k + (tc * H + i_h) * KD + 2 * cp);
                uint32_t qq = (ABL & 1) ? 0u : *(const uint32_t*) (q + (tc * H + i_h) * KD + 2 * cp);
                uint32_t vv = (ABL & 1) ? 0u : *(const uint32_t*) (v + (tc * HV + i_hv) * VD + 2 * cp);
                float2 g = (ABL & 1) ? make_float2(0.f, 0.f) : *(const float2*) (gk + (tc * HV + i_hv) * KD + 2 * cp);
                uint32_t bb = (ABL & 1) ? 0u : beta[tc * HV + i_hv];
#endif
                if (!ok) { kk = qq = vv = bb = 0; g = make_float2(0.f, 0.f); }
                const float fb = bf2f(bb), eg0 = ex2(g.x), eg1 = ex2(g.y);
                const float k0 = bf2f(kk & 0xffff), k1 = bf2f(kk >> 16);
                const float q0 = bf2f(qq & 0xffff), q1 = bf2f(qq >> 16);
                kb[0][e] = f2bf(bf2f(f2bf_rtz(k0 * fb)) * eg0);
                kb[1][e] = f2bf(bf2f(f2bf_rtz(k1 * fb)) * eg1);
                vb[0][e] = f2bf_rtz(bf2f(vv & 0xffff) * fb);
                vb[1][e] = f2bf_rtz(bf2f(vv >> 16) * fb);
                kg[0][e] = f2bf(k0 * (ok ? ex2(gn.x - g.x) : 0.0f));
                kg[1][e] = f2bf(k1 * (ok ? ex2(gn.y - g.y) : 0.0f));
                qgp[2 * s2 + e] = f2bf(q0 * eg0) | (f2bf(q1 * eg1) << 16);
            }
            #pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                kbp[c][s2] = kb[c][0] | (kb[c][1] << 16);
                vbp[c][s2] = vb[c][0] | (vb[c][1] << 16);
                kgp[c][s2] = kg[c][0] | (kg[c][1] << 16);
            }
        }

        PT(0);
        __syncthreads();  // previous chunk done with sX, sY, sEgn, sW
        PT(1);
        #pragma unroll
        for (int c = 0; c < 2; ++c)
            #pragma unroll
            for (int s2 = 0; s2 < 8; ++s2)
            {
                *(uint32_t*) (sX + (2 * cp + c) * LDT + 16 * tg + 2 * s2) = kbp[c][s2];
                *(uint32_t*) (sY + (2 * cp + c) * LDT + 16 * tg + 2 * s2) = vbp[c][s2];
            }
#endif
        if (tg == 0) { sEgn[2 * cp] = ex2(gn.x); sEgn[2 * cp + 1] = ex2(gn.y); }
        {
            uint4* d = (uint4*) (sA + ar * LDT + ac);
            d[0] = make_uint4(akk[0], akk[1], akk[2], akk[3]); d[1] = make_uint4(akk[4], akk[5], akk[6], akk[7]);
        }
        __syncthreads();
        PT(2);
        // Aqk slice: issued here, consumed after the w / u products (latency hidden)
        const v8u aqk = ld16g(Aqk + ((bos + min(t0 + ar, T - 1)) * HV + i_hv) * BT + ac, t0 + ar < T);

        // ---- w[:, kdim block wv] and u[:, v block wv]
        uint32_t up[4][4];
        #pragma unroll
        for (int a = 0; a < 4; ++a)
        {
            const int l_ = PIN(4, ll), c_ = PIN(4, col);
            v8u A[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                A[j] = ld16(sA, 16 * a + l_, LDT, 16 * j);
            v8f w = {0, 0, 0, 0, 0, 0, 0, 0}, u = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                if (!(ABL & 2)) w = wmma(A[j], ld16(sX, c_, LDT, 16 * j), w);
                if (!(ABL & 2)) u = wmma(A[j], ld16(sY, c_, LDT, 16 * j), u);
            }
            #pragma unroll
            for (int r = 0; r < 8; ++r)
                sW[(16 * a + 2 * r + half) * LDR + col] = (uint16_t) f2bf(w[r]);
            #pragma unroll
            for (int m = 0; m < 4; ++m)
                up[a][m] = f2bf(u[2 * m]) | (f2bf(u[2 * m + 1]) << 16);
            SBAR(4);
        }
        PT(3);
        __syncthreads();  // sW complete; sX / sY phase-1 reads done
        PT(4);

        #pragma unroll
        for (int s = 0; s < 16; ++s)
            *(uint32_t*) (sX + (16 * tg + s) * LDR + 2 * cp) = qgp[s];
        #pragma unroll
        for (int c = 0; c < 2; ++c)
            #pragma unroll
            for (int s2 = 0; s2 < 8; ++s2)
                *(uint32_t*) (sY + (2 * cp + c) * LDT + 16 * tg + 2 * s2) = kgp[c][s2];
        {
            uint4* d = (uint4*) (sA + ar * LDT + ac);
            d[0] = make_uint4(aqk[0], aqk[1], aqk[2], aqk[3]); d[1] = make_uint4(aqk[4], aqk[5], aqk[6], aqk[7]);
        }
        __syncthreads();
        PT(5);

#if PF == 2
        #pragma unroll
        for (int g_ = 0; g_ < PFG; ++g_) LDG((pk + NG * g_), (pq + NG * g_), (pv + NG * g_), (pg + NG * g_), (pb + NG * g_), t0 + BT + 16 * tg + NG * g_);
#endif
        // ---- o = qg h, vn = w h (chained over kdim tiles)
        v8f oa[4], vn[4];
        #pragma unroll
        for (int a = 0; a < 4; ++a) { oa[a] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0}; vn[a] = oa[a]; }
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
        {
            const v8u hb = c_to_b(h[kt], half);
            const int l_ = PIN(1, ll);
            #pragma unroll
            for (int a = 0; a < 4; ++a)
            {
                if (!(ABL & 4)) oa[a] = wmma(ld16(sX, 16 * a + l_, LDR, 16 * kt), hb, oa[a]);
                if (!(ABL & 4)) vn[a] = wmma(ld16(sW, 16 * a + l_, LDR, 16 * kt), hb, vn[a]);
            }
            if (OPV & 1)
                #pragma unroll
                for (int a = 0; a < 4; ++a) { pinv(oa[a]); pinv(vn[a]); }
            SBAR(1);
        }

        PT(6);
        // ---- v_new = u - w h; o = scale o + tril(Aqk) v_new
        v8u vs[4];
        #pragma unroll
        for (int a = 0; a < 4; ++a)
        {
            #pragma unroll
            for (int r = 0; r < 8; ++r)
            {
                vn[a][r] = bf2f((up[a][r >> 1] >> (16 * (r & 1))) & 0xffffu) - vn[a][r];
                oa[a][r] *= scale;
            }
            vs[a] = c_to_b(vn[a], half);
        }
        #pragma unroll
        for (int a = 0; a < 4; ++a)
        {
            const int l_ = PIN(8, ll);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                v8u A = {0, 0, 0, 0, 0, 0, 0, 0};
                if (j <= a)
                {
                    A = ld16(sA, 16 * a + l_, LDT, 16 * j);
                    if (j == a)
                    {
                        #pragma unroll
                        for (int m = 0; m < 8; ++m)
                        {
                            if (2 * m > ll) A[m] &= 0xffff0000u;
                            if (2 * m + 1 > ll) A[m] &= 0x0000ffffu;
                        }
                    }
                }
                if (!(ABL & 8)) oa[a] = wmma(A, vs[j], oa[a]);
            }
            if (OPV & 8) pinv(oa[a]);
#if UNI && ADR
            if (t0 + BT <= T)
            {
                // ADR: one uniform chunk base + a per-iteration opaque 32-bit lane offset (no hoisted per-row pointers)
                uint16_t* ob = o + ((bos + t0 + 16 * a) * HV + i_hv) * VD;
                uint32_t lo = (uint32_t) (half * HV * VD + col);
                asm volatile("" : "+v"(lo));
                const uint32_t rs = (uint32_t) (2 * HV * VD);
                #pragma unroll
                for (int r = 0; r < 8; ++r)
                    ob[lo + r * rs] = (uint16_t) f2bf(oa[a][r]);
            }
            else
#elif UNI
            if (t0 + BT <= T)
            {
                #pragma unroll
                for (int r = 0; r < 8; ++r)
                {
                    uint16_t* ob = o + ((bos + t0 + 16 * a + 2 * r) * HV + i_hv) * VD;  // uniform row base
                    ob[half * HV * VD + col] = (uint16_t) f2bf(oa[a][r]);
                }
            }
            else
#elif BRL
            if (t0 + BT <= T)
            {
                uint16_t* ob = o + ((bos + t0 + 16 * a + half) * HV + i_hv) * VD + col;
                #pragma unroll
                for (int r = 0; r < 8; ++r)
                    ob[(int64_t) 2 * r * HV * VD] = (uint16_t) f2bf(oa[a][r]);
            }
            else
#endif
            #pragma unroll
            for (int r = 0; r < 8; ++r)
            {
                const int t = t0 + 16 * a + 2 * r + half;
                if (t < T) o[((bos + t) * HV + i_hv) * VD + col] = (uint16_t) f2bf(oa[a][r]);
            }
            SBAR(8);
        }

#if PF == 1
        #pragma unroll
        for (int g_ = 0; g_ < PFG; ++g_) LDG((pk + NG * g_), (pq + NG * g_), (pv + NG * g_), (pg + NG * g_), (pb + NG * g_), t0 + BT + 16 * tg + NG * g_);
#endif
        PT(7);
        // ---- h = h exp2(g_last) + kg^T v_new
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
        {
            const int l_ = PIN(2, ll);
            #pragma unroll
            for (int r = 0; r < 8; ++r)
                h[kt][r] *= sEgn[16 * kt + 2 * r + half];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (!(ABL & 16)) h[kt] = wmma(ld16(sY, 16 * kt + l_, LDT, 16 * j), vs[j], h[kt]);
            if (OPV & 2) pinv(h[kt]);
            SBAR(2);
        }
        PT(8);
        // mid-chunk checkpoint (EXL3_MIDCHUNK_CKPT=2): state after chunk ckpt_chunk - 1 (row 64 * ckpt_chunk), same layout as ht
        if (hc && i_t + 1 == ckpt_chunk)
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt)
                #pragma unroll
                for (int r = 0; r < 8; ++r)
                    hc[(int64_t) i_nh * KD * VD + (16 * kt + 2 * r + half) * VD + col] = h[kt][r];
        // second checkpoint (EXL3_PF_NO_TAIL: the prompt's last full page, stashed inside the chunk)
        if (hc2 && i_t + 1 == ckpt_chunk2)
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt)
                #pragma unroll
                for (int r = 0; r < 8; ++r)
                    hc2[(int64_t) i_nh * KD * VD + (16 * kt + 2 * r + half) * VD + col] = h[kt][r];
    }

#if PROF
    if (lane == 0)
        for (int n = 0; n < 9; ++n) kda_prof[(blockIdx.x * 8 + wv) * 9 + n] = pacc[n];
#endif
    if (ht)
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            #pragma unroll
            for (int r = 0; r < 8; ++r)
                ht[(int64_t) i_nh * KD * VD + (16 * kt + 2 * r + half) * VD + col] = h[kt][r];
}
