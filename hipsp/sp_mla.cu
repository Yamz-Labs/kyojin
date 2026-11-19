// Sparse MLA prefill kernel for gfx1151 (WMMA f16, wave32). One workgroup (16 waves) = one query x 64 heads.
// Latent rows [N, 576] (512 content + 64 rope), Q [H, T, 576], O [H, T, 512] fp16 (saturated). Visible keys = selected groups + own open group.
// Layout: wave w = (mt = w & 3: 16 heads, qq = w >> 2: quarter). QK: S^T = K Q^T over a 144-dim slice (Q B fragments resident, 72 VGPR), partial S through LDS,
// every wave of a head tile sums the 4 partials and runs the same online softmax (lazy rescale); PV: O^T = V^T P^T for 128 dims (acc 64 VGPR).
// K tile (16 keys) is staged in LDS row-major (QK) and transposed with a 16-key slot permutation [evens | odds] (PV), P^T is built with one permlanex16 per 4 regs.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "sp_idx.h"
typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 h16;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float f8;
#define LOG2E 1.4426950408889634f
#define D_ 576
#define R_ 512
#define KPITCH 1168   // bytes per K row in LDS (73 * 16: conflict free b128 rows)
#define VPITCH 48     // bytes per V^T row (16 keys * 2 + 16 pad)
#define LAZY 8.0f
#ifdef NO_SYNC
#define SYNC() ((void)0)
#else
#define SYNC() __syncthreads()
#endif
#ifndef HALFK
#define HALFK 0
#endif

__device__ __forceinline__ uint32_t perm16(uint32_t x) { return __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false); }
__device__ __forceinline__ float permf(float x) { return __builtin_bit_cast(float, perm16(__builtin_bit_cast(uint32_t, x))); }
__device__ __forceinline__ uint32_t pk2(float a, float b)
{
    return (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)a) | ((uint32_t)__builtin_bit_cast(uint16_t, (_Float16)b) << 16);
}
__device__ __forceinline__ h16 mk16(uint4 a, uint4 b) { h16 f; ((uint4*)&f)[0] = a; ((uint4*)&f)[1] = b; return f; }
__device__ __forceinline__ _Float16 satf(float x) { return (_Float16)fminf(fmaxf(x, -65504.f), 65504.f); }
__device__ __forceinline__ uint32_t pk2s(float a, float b)
{
    return (uint32_t)__builtin_bit_cast(uint16_t, satf(a)) | ((uint32_t)__builtin_bit_cast(uint16_t, satf(b)) << 16);
}

template <int KP, bool SINK>
__global__ __launch_bounds__(512) void sp_mla(const _Float16* __restrict__ Q, const _Float16* __restrict__ KC, const float* __restrict__ sink, const int* __restrict__ gidx,
    const int* __restrict__ nsel, _Float16* __restrict__ O, int T, int qstart, int H, int GS, int N, float scale, int tail, int fixed_tiles)
{
    __shared__ __attribute__((aligned(16))) char Kn[16 * KPITCH];
    __shared__ __attribute__((aligned(16))) char Vt[512 * VPITCH];
    __shared__ __attribute__((aligned(16))) float Sp[4 * 4 * 16 * 16];
    __shared__ int gi[512];
    const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, n = l & 15, hh = l >> 4, mt = w & 3, qq = w >> 2;
    const int t = blockIdx.x, hg = blockIdx.y * 64;
    const int pos = qstart + t;
    const int ns = fixed_tiles ? 0 : min(nsel[t], min(GS, 512));
    const int lo_t = sp_lo_t(pos, KP, tail);
    const int ntail = pos - lo_t + 1;
    const int nv = fixed_tiles ? fixed_tiles * 16 : ns * KP + ntail;
    const int ntiles = (nv + 15) >> 4;
    if (tid < ns) gi[tid] = gidx[(long)t * GS + tid];
    const int head = hg + mt * 16 + n;
    const long qrow = ((long)head * T + t) * D_;
    // resident Q B fragments (both lane halves hold the full 16 dims)
    h16 qf[9];
#pragma unroll
    for (int s = 0; s < 9; s++) {
        const uint4* p = (const uint4*)(Q + qrow + 144 * qq + 16 * s);
        qf[s] = mk16(p[0], p[1]);
    }
    f8 acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = (f8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = SINK ? sink[head] * LOG2E : -1e30f;
    float ls = SINK ? (hh == 0 ? 1.f : 0.f) : 0.f;
    const float c = scale * LOG2E;
    // gather task: V part (dim chunk cT, group g, parity e) = 2 keys x 8 dims; rope part for tid < 128
    const int cT = tid >> 3, tg = (tid >> 1) & 3, te = tid & 1;
    const int rk = tid >> 3, rc = tid & 7;
    uint4 ga, gb, gr;
    auto gload = [&](int tau) {
        const int v0 = tau * 16 + 4 * tg + te;
        int r0, r1;
        if (fixed_tiles) { r0 = v0 & 15; r1 = (v0 + 2) & 15; }
        else { r0 = sp_row<KP>(v0, nv, ns, lo_t, gi, N); r1 = sp_row<KP>(v0 + 2, nv, ns, lo_t, gi, N); }
        ga = *(const uint4*)(KC + (long)r0 * D_ + 8 * cT);
        gb = *(const uint4*)(KC + (long)r1 * D_ + 8 * cT);
        if (tid < 128) {
            const int rr = fixed_tiles ? ((tau * 16 + rk) & 15) : sp_row<KP>(tau * 16 + rk, nv, ns, lo_t, gi, N);
            gr = *(const uint4*)(KC + (long)rr * D_ + R_ + 8 * rc);
        }
    };
    auto wK = [&]() {
        const int k0 = 4 * tg + te;
        *(uint4*)(Kn + k0 * KPITCH + 16 * cT) = ga;
        *(uint4*)(Kn + (k0 + 2) * KPITCH + 16 * cT) = gb;
        if (tid < 128) *(uint4*)(Kn + rk * KPITCH + 2 * (R_ + 8 * rc)) = gr;
    };
    auto wV = [&]() {
        const uint32_t av[4] = {ga.x, ga.y, ga.z, ga.w}, bv[4] = {gb.x, gb.y, gb.z, gb.w};
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t a = av[j >> 1], b = bv[j >> 1];
            const uint32_t pr = (j & 1) ? ((a >> 16) | (b & 0xffff0000u)) : ((a & 0xffffu) | (b << 16));
            const int d = (8 * cT + j) ^ (cT & 3);
            *(uint32_t*)(Vt + d * VPITCH + 16 * te + 4 * tg) = pr;
        }
    };
    __syncthreads();
    if (ntiles > 0) { gload(0); wK(); }
    __syncthreads();
    const char* kb = Kn + n * KPITCH + 2 * 144 * qq;
    float* spw = Sp + ((mt * 4 + qq) * 16 + hh) * 16 + n;
    for (int tau = 0; tau < ntiles; tau++) {
        const bool more = tau + 1 < ntiles;
#ifndef NO_GATHER
        wV();
#endif
        //  Vt_tau (free: every wave passed the end barrier of tau-1)
#ifndef NO_GATHER
        if (more) gload(tau + 1);
#endif
        f8 s = (f8){0, 0, 0, 0, 0, 0, 0, 0};
#ifndef NO_QK
        f8 s1 = s, s2 = s;
#pragma unroll
        for (int k = 0; k < 9; k++) {
            const uint4* kp = (const uint4*)(kb + 32 * k);
            #ifdef NO_LDSRD
            const h16 af = qf[(k + 1) % 9];
#else
            const h16 af = mk16(kp[0], HALFK ? kp[0] : kp[1]);
#endif
            if (k % 3 == 0) s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, qf[k], s);
            else if (k % 3 == 1) s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, qf[k], s1);
            else s2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, qf[k], s2);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] = (s[i] + s1[i]) + s2[i];
#endif
#pragma unroll
#ifndef NO_SP
        for (int i = 0; i < 8; i++) spw[2 * i * 16] = s[i];
#endif
        SYNC();
#ifndef NO_GATHER
        if (more) wK();
#endif
        //  Kn_{tau+1}: every wave finished QK(tau)
        float sv[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            float p[4];
#pragma unroll
#ifdef NO_SP
            for (int q2 = 0; q2 < 4; q2++) p[q2] = s[i];
#else
            for (int q2 = 0; q2 < 4; q2++) p[q2] = (q2 == qq) ? s[i] : Sp[(((mt * 4 + q2) * 16 + 2 * i + hh) * 16) + n];
#endif
            sv[i] = (((p[0] + p[1]) + p[2]) + p[3]) * c;
        }
#ifdef NO_SOFT
        uint32_t pw[4]; const float alpha = 1.f; const bool upd = false;
        for (int i = 0; i < 8; i += 2) pw[i >> 1] = pk2(sv[i], sv[i + 1]);
#else
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const bool ok = tau * 16 + 2 * i + hh < nv;
            sv[i] = ok ? sv[i] : -INFINITY;
            mx = fmaxf(mx, sv[i]);
        }
        mx = fmaxf(mx, permf(mx));
        const bool upd = mx > m + LAZY;
        const float mn = upd ? mx : m;
        const float alpha = upd ? __builtin_amdgcn_exp2f(m - mn) : 1.f;
        m = mn;
        float ps = 0.f;
        uint32_t pw[4];
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const float p0 = __builtin_amdgcn_exp2f(sv[i] - mn), p1 = __builtin_amdgcn_exp2f(sv[i + 1] - mn);
            ps += p0 + p1;
            pw[i >> 1] = pk2(p0, p1);
        }
        ls = ls * alpha + ps;
        if (__any(upd)) {
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] *= alpha;
        }
#endif
        #ifdef DEPFREE
        for (int i = 0; i < 8; i++) ls += s[i];
        const h16 pb = mk16(make_uint4(tau, n, hh, 7), make_uint4(3, 4, tau, 5));
#else
        const h16 pb = mk16(make_uint4(pw[0], pw[1], pw[2], pw[3]), make_uint4(perm16(pw[0]), perm16(pw[1]), perm16(pw[2]), perm16(pw[3])));
#endif
#ifndef NO_PV
#pragma unroll
        for (int tt = 0; tt < 8; tt++) {
            const int d = (16 * (8 * qq + tt) + n);
            const int pd = d ^ ((d >> 3) & 3);
            const char* vp = Vt + pd * VPITCH;
            #ifdef NO_LDSRD
            const uint4 a0 = make_uint4(pw[0] + tt, pw[1], pw[2], pw[3] ^ tt), a1 = make_uint4(pw[1], pw[2] + tt, pw[3], pw[0]);
#else
            const uint4 a0 = *(const uint4*)(vp + 16 * hh), a1 = HALFK ? a0 : *(const uint4*)(vp + 16 * (hh ^ 1));
#endif
            acc[tt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(mk16(a0, a1), pb, acc[tt]);
        }
#endif
        SYNC();
    }
    ls += permf(ls);
    const float inv = 1.f / fmaxf(ls, 1e-30f);
    _Float16* op = O + ((long)head * T + t) * R_ + 128 * qq + 8 * hh;
#pragma unroll
    for (int tt = 0; tt < 8; tt++) {
        float a[8], b[8];
#pragma unroll
        for (int i = 0; i < 8; i++) { a[i] = acc[tt][i] * inv; b[i] = permf(a[i]); }
        uint4 o;
        if (hh == 0) o = make_uint4(pk2s(a[0], b[0]), pk2s(a[1], b[1]), pk2s(a[2], b[2]), pk2s(a[3], b[3]));
        else o = make_uint4(pk2s(b[4], a[4]), pk2s(b[5], a[5]), pk2s(b[6], a[6]), pk2s(b[7], a[7]));
        *(uint4*)(op + 16 * tt) = o;
    }
}

extern "C" int sp_launch(const void* Q, const void* KC, const void* sink, const void* gidx, const void* nsel, void* O, int T, int qstart, int H, int GS, int N, float scale, int tail,
    int KP, int fixed_tiles, void* stream)
{
    hipStream_t st = (hipStream_t)stream;
    if (GS > 512 || H % 64) return -2;
    dim3 g(T, H / 64);
#define L(KPV, SK) sp_mla<KPV, SK><<<g, 512, 0, st>>>((const _Float16*)Q, (const _Float16*)KC, (const float*)sink, (const int*)gidx, (const int*)nsel, (_Float16*)O, T, qstart, H, GS, N, scale, tail, fixed_tiles)
    if (KP == 4) { if (sink) L(4, true); else L(4, false); }
    else if (KP == 2) { if (sink) L(2, true); else L(2, false); }
    else if (KP == 8) { if (sink) L(8, true); else L(8, false); }
    else return -1;
    return (int)hipGetLastError();
}
