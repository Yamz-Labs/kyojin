// GDN (gated delta rule) fused chunk kernel, hand-written for gfx11.5 WMMA (Strix Halo, wave32).
// Fuses recompute_w_u + chunk_gated_delta_rule_fwd_h + chunk_fwd_o of the vendored fla Triton chain:
//   w = A (k beta exp2 g), u = A (v beta), v_new = u - w h, o = scale (exp2(g) q h + tril(q k^T exp2(g_i - g_j)) v_new),
//   h <- exp2(g_last) h + k^T (v_new exp2(g_last - g)).
// Same rounding points as the Triton kernels (bf16 products truncated, explicit casts RNE, raw v_exp_f32,
// fp32 WMMA accumulation chained in the same k order, no fma contraction: build with -ffp-contract=off).
// One workgroup (8 waves) per (batch, value head); K = V = 128, BT = 64. Wave i owns the state columns
// h[:, 16i:16i+16] in registers (8 WMMA C tiles). A (the solved kkt matrix, bf16) comes from the Triton kkt_solve kernel.
// Fragment layout (gfx1151, oracle-verified in exllamav3_ext/hip/hip_mma.cuh):
//   A: lane l, elem r -> A[l & 15][r];  B: lane l, elem r -> B[r][l & 15];  C: lane l, reg r -> C[2r + (l >> 4)][l & 15]
//
// v is read through the row pitch PV (elements; HV * VD when contiguous), so the conv output slice can be used in place.
// MEMORY SAFETY: every global address comes from the IX_* macros below with the token row clamped by TCL (rows >= T
// re-read row T - 1 and are zeroed in registers), every store is guarded by t < T. check_gdn_bounds.py parses these
// macros and replays all accesses over the real shapes (it must fail when TCL or a guard is mutated).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v8u __attribute__((ext_vector_type(8)));

#define BT 64
#define KD 128
#define VD 128
#define LDR 136  // row-major [token][kdim] buffers (bf16 elements per row)
#define LDT 72   // transposed [dim][token] buffers

#ifndef OFMA
#define OFMA 1   // 1 (Triton contracts it; bit-exact): o = fma(o_inter, scale, o_intra * scale)  (Triton may contract this add)
#endif
#ifndef DBG
#define DBG 0    // 1: dump w, u, v_new (bf16) and per-chunk state (bf16) for stage-by-stage checks (B = 1)
#endif
#ifndef RELOAD
#define RELOAD 0 // 1: q and k rows are re-read from global (L2) after the w/u phase instead of being held in registers across it
#endif
#ifndef SB
#define SB 0     // bit mask of sched_barrier(0) sites (stops the scheduler hoisting later LDS fragment loads: spills): 1 chain kt loop, 2 state kt loop, 4 w/u a loop, 8 o2 a loop, 16 Aqk kt loop
#endif
#ifndef ABL
#define ABL 0    // timing ablations (output is garbage when nonzero; addresses unchanged or fewer): 1 w/u chain, 2 Aqk chain, 4 o1/vn chain, 8 o2 chain, 16 state update chain, 32 phase-1 global loads, 64 barriers 3-6, 128 barriers 1-2, 256 o stores (result folded into a sink instead), 512 o stores all to row 0, 1024 / 2048 / 4096 zero the v / k+q / g+beta loads
#endif
#define SYNCA(b) do { if (!(ABL & (b))) __syncthreads(); } while (0)
#define SBAR(b) do { if (SB & (b)) __builtin_amdgcn_sched_barrier(0); } while (0)
#ifndef OST
#define OST 0    // 1: o tile staged through LDS (sW) and written as 16 B vectors, full 256 B rows (same bits; replaces 2 B stores of 32 B row pieces)
#endif
#ifndef PF1
#define PF1 0    // 1: phase 1 writes its LDS tiles directly after a barrier at the top of the chunk (no register staging across a barrier; same bits)
#endif
#if PF1 && RELOAD
#error PF1 and RELOAD are exclusive
#endif
#ifndef BRL
#define BRL 1    // branchless f2bf (same bits)
#endif

// ---- global index macros (checked on CPU). tc = absolute token row (bos + clamped t), results in elements ----
#define TCL(bos_, t_, T_) ((bos_) + min((t_), (T_) - 1))
#define IX_QK(tc, hh, c) ((((int64_t) (tc)) * H + (hh)) * KD + (c))
#define IX_V(tc, hv, c) ((((int64_t) (tc)) * HV + (hv)) * VD + (c))
#define IX_VR(tc, hv, c) ((((int64_t) (tc)) * PV) + ((int64_t) (hv)) * VD + (c))   // v read with row pitch PV (elements)
#define IX_G(tc, hv) (((int64_t) (tc)) * HV + (hv))
#define IX_A(tc, hv, c) ((((int64_t) (tc)) * HV + (hv)) * BT + (c))
#define IX_S(nh, kd, cl) (((int64_t) (nh)) * KD * VD + (kd) * VD + (cl))

__device__ __forceinline__ uint32_t f2bf(float f)
{
    uint32_t u = __float_as_uint(f);
#if BRL
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
// Triton rounds explicit .to(bf16) casts to nearest even but truncates the result of bf16 x bf16 arithmetic
__device__ __forceinline__ uint32_t f2bf_rtz(float f) { return __float_as_uint(f) >> 16; }
__device__ __forceinline__ float ex2(float x) { return __builtin_amdgcn_exp2f(x); }

__device__ __forceinline__ v8f wmma(v8u a, v8u b, v8f c)
{
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(v16s, a), __builtin_bit_cast(v16s, b), c);
}

// 16 contiguous bf16 at base[row * ld + col] (32 B, 16 B aligned) from LDS
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

extern "C" __global__ __launch_bounds__(256) void gdn_fused_h_o(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const float* __restrict__ g, const uint16_t* __restrict__ beta, const uint16_t* __restrict__ A,
    uint16_t* __restrict__ o, const float* __restrict__ h0, float* __restrict__ ht, float scale, int T, int H, int HV, int PV,
    uint16_t* __restrict__ dw, uint16_t* __restrict__ du, uint16_t* __restrict__ dvn, uint16_t* __restrict__ dh)
{
    __shared__ __attribute__((aligned(16))) uint16_t sW[BT * LDR];   // w [token][kdim]
    __shared__ __attribute__((aligned(16))) uint16_t sX[KD * LDT];   // kb^T [kdim][token], then q [token][kdim]
    __shared__ __attribute__((aligned(16))) uint16_t sY[KD * LDT];   // vb^T [v][token], then k [token][kdim], then k^T [kdim][token]
    __shared__ __attribute__((aligned(16))) uint16_t sA[BT * LDT];   // A [token][token], then tril(Aqk) bf16
    __shared__ float sG[BT], sGE[BT], sRS[BT];                        // g, exp2(g), exp2(g_last - g) (0 for rows >= T)

    const int tid = threadIdx.x, wv = tid >> 5, lane = tid & 31, ll = lane & 15, half = lane >> 4;
    const int i_nh = blockIdx.x, i_n = i_nh / HV, i_hv = i_nh % HV, i_h = i_hv / (HV / H);
    const int64_t bos = (int64_t) i_n * T;
    const int NT = (T + BT - 1) / BT;
    const int cp = tid & 63, tg = tid >> 6;   // loader: column pair 2cp, tokens 16tg..16tg+15
    const int col = 16 * wv + ll;             // this lane's state / value column
    const int ar = tid >> 2, ac = 16 * (tid & 3);  // A-matrix staging: row, 16-column slice

    uint32_t sink = 0;   // ABL 256 only
    v8f h[8];
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt)
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            h[kt][r] = h0 ? h0[IX_S(i_nh, 16 * kt + 2 * r + half, col)] : 0.0f;

    for (int i_t = 0; i_t < NT; ++i_t)
    {
        const int t0 = i_t * BT;
        const int last = min(t0 + BT, T) - 1;
        const float gl = g[IX_G(bos + last, i_hv)];
        const float egl = ex2(gl);

#if PF1
        // ---- phase 1 (PF1): barrier first (previous chunk done with sX, sY, sW, sA, sG), then every per-token product goes straight
        // into LDS, so no phase-1 result is held in registers across a barrier (that held ~64 registers next to the 64-register state
        // and made the compiler spill about 50 per chunk to scratch)
        SYNCA(128);
        if (tid < BT)
        {
            const int t = t0 + tid;
            const float gv = (t < T) ? g[IX_G(TCL(bos, t, T), i_hv)] : 0.0f;
            sG[tid] = gv; sGE[tid] = ex2(gv); sRS[tid] = (t < T) ? ex2(gl - gv) : 0.0f;
        }
        {
            const v8u aa = ld16g(A + IX_A(TCL(bos, t0 + ar, T), i_hv, ac), t0 + ar < T);
            uint4* d = (uint4*) (sA + ar * LDT + ac);
            d[0] = make_uint4(aa[0], aa[1], aa[2], aa[3]); d[1] = make_uint4(aa[4], aa[5], aa[6], aa[7]);
        }
        uint32_t qraw[16], kraw[16];
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2)
        {
            uint32_t kb[2][2], vb[2][2];
            #pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                const int tt = 16 * tg + 2 * s2 + e, t = t0 + tt;
                const bool ok = t < T;
                const int64_t tc = TCL(bos, t, T);
                uint32_t kk = *(const uint32_t*) (k + IX_QK(tc, i_h, 2 * cp));
                uint32_t qq = *(const uint32_t*) (q + IX_QK(tc, i_h, 2 * cp));
                uint32_t vv = *(const uint32_t*) (v + IX_VR(tc, i_hv, 2 * cp));
                float gv = g[IX_G(tc, i_hv)];
                uint32_t bb = beta[IX_G(tc, i_hv)];
                if (!ok || (ABL & 32)) { kk = qq = vv = bb = 0; gv = 0.f; }
                if (ABL & 1024) vv = 0;
                if (ABL & 2048) { kk = 0; qq = 0; }
                if (ABL & 4096) { bb = 0; gv = 0.f; }
                const float fb = bf2f(bb), eg = ex2(gv);
                const float k0 = bf2f(kk & 0xffff), k1 = bf2f(kk >> 16);
                kb[0][e] = f2bf(bf2f(f2bf_rtz(k0 * fb)) * eg);
                kb[1][e] = f2bf(bf2f(f2bf_rtz(k1 * fb)) * eg);
                vb[0][e] = f2bf_rtz(bf2f(vv & 0xffff) * fb);
                vb[1][e] = f2bf_rtz(bf2f(vv >> 16) * fb);
                qraw[2 * s2 + e] = qq;
                kraw[2 * s2 + e] = kk;
            }
            #pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                *(uint32_t*) (sX + (2 * cp + c) * LDT + 16 * tg + 2 * s2) = kb[c][0] | (kb[c][1] << 16);
                *(uint32_t*) (sY + (2 * cp + c) * LDT + 16 * tg + 2 * s2) = vb[c][0] | (vb[c][1] << 16);
            }
        }
        SYNCA(128);

#else
        // ---- phase 1: per-token products in registers (+ this thread's A slice)
        const v8u aa = ld16g(A + IX_A(TCL(bos, t0 + ar, T), i_hv, ac), t0 + ar < T);
        uint32_t kbp[2][8], vbp[2][8];
#if !RELOAD
        uint32_t qraw[16], kraw[16];
#else
        uint32_t kraw[16];
#endif
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2)
        {
            uint32_t kb[2][2], vb[2][2];
            #pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                const int tt = 16 * tg + 2 * s2 + e, t = t0 + tt;
                const bool ok = t < T;
                const int64_t tc = TCL(bos, t, T);
                uint32_t kk = *(const uint32_t*) (k + IX_QK(tc, i_h, 2 * cp));
                uint32_t qq = *(const uint32_t*) (q + IX_QK(tc, i_h, 2 * cp));
                uint32_t vv = *(const uint32_t*) (v + IX_VR(tc, i_hv, 2 * cp));
                float gv = g[IX_G(tc, i_hv)];
                uint32_t bb = beta[IX_G(tc, i_hv)];
                if (!ok || (ABL & 32)) { kk = qq = vv = bb = 0; gv = 0.f; }
                if (ABL & 1024) vv = 0;
                if (ABL & 2048) { kk = 0; qq = 0; }
                if (ABL & 4096) { bb = 0; gv = 0.f; }
                const float fb = bf2f(bb), eg = ex2(gv);
                const float k0 = bf2f(kk & 0xffff), k1 = bf2f(kk >> 16);
                kb[0][e] = f2bf(bf2f(f2bf_rtz(k0 * fb)) * eg);
                kb[1][e] = f2bf(bf2f(f2bf_rtz(k1 * fb)) * eg);
                vb[0][e] = f2bf_rtz(bf2f(vv & 0xffff) * fb);
                vb[1][e] = f2bf_rtz(bf2f(vv >> 16) * fb);
#if !RELOAD
                qraw[2 * s2 + e] = qq;
                kraw[2 * s2 + e] = kk;
#endif
            }
            #pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                kbp[c][s2] = kb[c][0] | (kb[c][1] << 16);
                vbp[c][s2] = vb[c][0] | (vb[c][1] << 16);
            }
        }

        SYNCA(128);  // previous chunk done with sX, sY, sW, sA, sG
        if (tid < BT)
        {
            const int t = t0 + tid;
            const float gv = (t < T) ? g[IX_G(TCL(bos, t, T), i_hv)] : 0.0f;
            sG[tid] = gv; sGE[tid] = ex2(gv); sRS[tid] = (t < T) ? ex2(gl - gv) : 0.0f;
        }
        #pragma unroll
        for (int c = 0; c < 2; ++c)
            #pragma unroll
            for (int s2 = 0; s2 < 8; ++s2)
            {
                *(uint32_t*) (sX + (2 * cp + c) * LDT + 16 * tg + 2 * s2) = kbp[c][s2];
                *(uint32_t*) (sY + (2 * cp + c) * LDT + 16 * tg + 2 * s2) = vbp[c][s2];
            }
        {
            uint4* d = (uint4*) (sA + ar * LDT + ac);
            d[0] = make_uint4(aa[0], aa[1], aa[2], aa[3]); d[1] = make_uint4(aa[4], aa[5], aa[6], aa[7]);
        }
        SYNCA(128);

#endif

        // ---- w[:, kdim block wv] and u[:, v block wv]
        uint32_t up[4][4];
        #pragma unroll
        for (int a = 0; a < 4; ++a)
        {
            v8u Af[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                Af[j] = ld16(sA, 16 * a + ll, LDT, 16 * j);
            v8f w = {0, 0, 0, 0, 0, 0, 0, 0}, u = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                if (!(ABL & 1))
                {
                    w = wmma(Af[j], ld16(sX, col, LDT, 16 * j), w);
                    u = wmma(Af[j], ld16(sY, col, LDT, 16 * j), u);
                }
            }
            #pragma unroll
            for (int r = 0; r < 8; ++r)
                sW[(16 * a + 2 * r + half) * LDR + col] = (uint16_t) f2bf(w[r]);
            #pragma unroll
            for (int m = 0; m < 4; ++m)
                up[a][m] = f2bf(u[2 * m]) | (f2bf(u[2 * m + 1]) << 16);
            SBAR(4);
#if DBG
            #pragma unroll
            for (int r = 0; r < 8; ++r)
            {
                const int t = t0 + 16 * a + 2 * r + half;
                if (t < T)
                {
                    dw[IX_V(bos + t, i_hv, col)] = (uint16_t) f2bf(w[r]);
                    du[IX_V(bos + t, i_hv, col)] = (uint16_t) ((up[a][r >> 1] >> (16 * (r & 1))) & 0xffffu);
                }
            }
#endif
        }
        SYNCA(64);  // sW complete; sX / sY phase-1 reads done

        // q and k row-major for the intra-chunk product
        #pragma unroll
        for (int s = 0; s < 16; ++s)
        {
#if RELOAD
            const int t = t0 + 16 * tg + s;
            const int64_t tc = TCL(bos, t, T);
            uint32_t kk = *(const uint32_t*) (k + IX_QK(tc, i_h, 2 * cp));
            uint32_t qq = *(const uint32_t*) (q + IX_QK(tc, i_h, 2 * cp));
            if (t >= T) kk = qq = 0;
            kraw[s] = kk;
            *(uint32_t*) (sX + (16 * tg + s) * LDR + 2 * cp) = qq;
            *(uint32_t*) (sY + (16 * tg + s) * LDR + 2 * cp) = kk;
#else
            *(uint32_t*) (sX + (16 * tg + s) * LDR + 2 * cp) = qraw[s];
            *(uint32_t*) (sY + (16 * tg + s) * LDR + 2 * cp) = kraw[s];
#endif
        }
        SYNCA(64);

        // ---- Aqk = tril(q k^T exp2(g_i - g_j)) -> bf16 into sA (10 lower tiles over 8 waves)
        #pragma unroll
        for (int ti = 0; ti < 2; ++ti)
        {
            const int tix = wv + 8 * ti;
            if (tix < 10)
            {
                const int a = (tix >= 6) ? 3 : (tix >= 3) ? 2 : (tix >= 1) ? 1 : 0;
                const int j = tix - (a * (a + 1)) / 2;
                v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
                #pragma unroll
                for (int kt = 0; kt < 8; ++kt)
                {
                    if (!(ABL & 2)) acc = wmma(ld16(sX, 16 * a + ll, LDR, 16 * kt), ld16(sY, 16 * j + ll, LDR, 16 * kt), acc);
                    SBAR(16);
                }
                const int cj = 16 * j + ll;
                const float gc = sG[cj];
                #pragma unroll
                for (int r = 0; r < 8; ++r)
                {
                    const int ri = 16 * a + 2 * r + half;
                    const float x = acc[r] * ex2(sG[ri] - gc);
                    const bool keep = (t0 + ri >= t0 + cj) && (t0 + ri < T) && (t0 + cj < T);
                    sA[ri * LDT + cj] = (uint16_t) (keep ? f2bf(x) : 0u);
                }
            }
        }
        SYNCA(64);  // tril(Aqk) complete; sY (k row-major) reads done

        // k^T [kdim][token] for the state update
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2)
        {
            const uint32_t k0 = kraw[2 * s2], k1 = kraw[2 * s2 + 1];
            *(uint32_t*) (sY + (2 * cp) * LDT + 16 * tg + 2 * s2) = (k0 & 0xffffu) | (k1 << 16);
            *(uint32_t*) (sY + (2 * cp + 1) * LDT + 16 * tg + 2 * s2) = (k0 >> 16) | (k1 & 0xffff0000u);
        }

        // ---- o1 = q h, vn = w h (chained over kdim tiles)
        v8f oa[4], vn[4];
        #pragma unroll
        for (int a = 0; a < 4; ++a) { oa[a] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0}; vn[a] = oa[a]; }
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
        {
            const v8u hb = c_to_b(h[kt], half);
#if DBG
            #pragma unroll
            for (int r = 0; r < 8; ++r)
                dh[((int64_t) i_t * HV + i_hv) * KD * VD + (16 * kt + 2 * r + half) * VD + col] = (uint16_t) f2bf(h[kt][r]);
#endif
            #pragma unroll
            for (int a = 0; a < 4; ++a)
            {
                if (!(ABL & 4))
                {
                    oa[a] = wmma(ld16(sX, 16 * a + ll, LDR, 16 * kt), hb, oa[a]);
                    vn[a] = wmma(ld16(sW, 16 * a + ll, LDR, 16 * kt), hb, vn[a]);
                }
            }
            SBAR(1);
        }

#if OST
        __syncthreads();  // every wave is done reading sW (w) before the o tile is staged into it
#endif
        // ---- v_new = u - w h; o = scale (exp2(g) o1) + scale (Aqk v_new)
        v8u vs[4], vg[4];
        #pragma unroll
        for (int a = 0; a < 4; ++a)
        {
            v8f vsc;
            #pragma unroll
            for (int r = 0; r < 8; ++r)
            {
                vn[a][r] = bf2f((up[a][r >> 1] >> (16 * (r & 1))) & 0xffffu) - vn[a][r];
                vsc[r] = vn[a][r] * sRS[16 * a + 2 * r + half];
            }
            vs[a] = c_to_b(vn[a], half);
            vg[a] = c_to_b(vsc, half);
#if DBG
            #pragma unroll
            for (int r = 0; r < 8; ++r)
            {
                const int t = t0 + 16 * a + 2 * r + half;
                if (t < T) dvn[IX_V(bos + t, i_hv, col)] = (uint16_t) f2bf(vn[a][r]);
            }
#endif
        }
        #pragma unroll
        for (int a = 0; a < 4; ++a)
        {
            v8f o2 = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (j <= a && !(ABL & 8)) o2 = wmma(ld16(sA, 16 * a + ll, LDT, 16 * j), vs[j], o2);
            #pragma unroll
            for (int r = 0; r < 8; ++r)
            {
                const int t = t0 + 16 * a + 2 * r + half;
                const float o1 = oa[a][r] * sGE[16 * a + 2 * r + half];
#if OFMA
                const float res = __builtin_fmaf(o1, scale, o2[r] * scale);
#else
                const float res = o1 * scale + o2[r] * scale;
#endif
#if OST
                sW[(16 * a + 2 * r + half) * LDR + col] = (uint16_t) f2bf(res);
#else
                if (ABL & 256) sink ^= f2bf(res);
                else if (ABL & 512) { o[IX_V(bos, i_hv, col)] = (uint16_t) f2bf(res); }   // timing only: every row lands on token row 0 of this item (same instruction count, no write traffic)
                else
                if (t < T) o[IX_V(bos + t, i_hv, col)] = (uint16_t) f2bf(res);
#endif
            }
            SBAR(8);
        }

        SYNCA(64);  // k^T visible
#if OST
        // o tile (sW, written above) -> global: thread (row rr, 8 columns cc), 4 rows per thread, 16 B per store
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int rr = (tid >> 4) + 16 * i, cc = 8 * (tid & 15), t = t0 + rr;
            const uint4 x = *(const uint4*) (sW + rr * LDR + cc);
            if (t < T) *(uint4*) (o + IX_V(bos + t, i_hv, cc)) = x;
        }
#endif
        // ---- h = h exp2(g_last) + k^T (v_new exp2(g_last - g))
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
        {
            #pragma unroll
            for (int r = 0; r < 8; ++r)
                h[kt][r] *= egl;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (!(ABL & 16)) h[kt] = wmma(ld16(sY, 16 * kt + ll, LDT, 16 * j), vg[j], h[kt]);
            SBAR(2);
        }
    }

    if (ABL & 256) { if (sink == 0x9e3779b9u && tid == 0) o[0] = 0; }   // keeps the ablated o path alive
    if (ht)
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            #pragma unroll
            for (int r = 0; r < 8; ++r)
                ht[IX_S(i_nh, 16 * kt + 2 * r + half, col)] = h[kt][r];
}
