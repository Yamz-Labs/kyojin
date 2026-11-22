// Inkling 2 global-layer decode attention + fixed KV cache for gfx1151 (WMMA f16, wave32).
// Cache (per global slot, head-major, capacity CAP rows, CAP % 64 == 0):
//   K  [NKV][CAP][128]            fp16 (Q = 1: int8) row-major: one key = 256 B (128 B)
//   V  [NKV][CAP/64][128][64]     fp16 (Q = 1: int8) dim-major inside a 64-key tile, keys of each 16-group stored [evens | odds]
//   ksc, vsc [NKV][CAP]           fp16 per-(key, head) scale (Q != 0 only): value = int8 * scale
// Decode attention: grid (NKV, SMAX), block 128 * RT threads (4 key-waves x RT row tiles). One block = one kv head x one key chunk, all 3 * T query rows
// (row r = t * 3 + g, head 3 * kvh + g). Wave = 16 keys per step: S^T = K Q^T (Q from LDS), online softmax (log2, lazy rescale), O^T = V^T P^T, no block barrier in the loop.
// Split-K over the context; chunk and number of active splits come from the device-side length (HIP-graph safe). Combine kernel merges the splits,
// adds the sink once and applies the optional output gate. Scores = q.k * HD^-0.5 + bias(dist) for 0 <= dist < 1024 (bias table [T][NH][1024] fp32).
#include <hip/hip_runtime.h>
#include <stdint.h>
typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 h16;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float f8;
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
#define LOG2E 1.4426950408889634f
#define NKV 8
#define NH 24
#define HD 128
#define NREL 1024
#define LAZY 8.0f
#define QP 136   // halves per Q row in LDS (272 B)
#define RP 68    // floats per row of the merge buffer

__device__ __forceinline__ uint32_t perm16(uint32_t x) { return __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false); }
__device__ __forceinline__ float permf(float x) { return __builtin_bit_cast(float, perm16(__builtin_bit_cast(uint32_t, x))); }
__device__ __forceinline__ uint32_t pk2(float a, float b)
{
    return (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)a) | ((uint32_t)__builtin_bit_cast(uint16_t, (_Float16)b) << 16);
}
__device__ __forceinline__ h16 mk16(uint4 a, uint4 b) { h16 f; ((uint4*)&f)[0] = a; ((uint4*)&f)[1] = b; return f; }
__device__ __forceinline__ _Float16 satf(float x) { return (_Float16)fminf(fmaxf(x, -65504.f), 65504.f); }
__device__ __forceinline__ float bf2f(uint16_t b) { return __builtin_bit_cast(float, (uint32_t)b << 16); }

// 4 int8 -> 4 fp16 (exact integers). mode 1: perm + packed subtract; mode 2: plain conversions
template <int DQ>
__device__ __forceinline__ void deq4(uint32_t w, uint32_t& o0, uint32_t& o1)
{
    if (DQ == 1) {
        const uint32_t x = w ^ 0x80808080u;
        const uint32_t a = __builtin_amdgcn_perm(0x64646464u, x, 0x05010400u);
        const uint32_t b = __builtin_amdgcn_perm(0x64646464u, x, 0x05030402u);
        const h2v k = (h2v){(_Float16)1152.f, (_Float16)1152.f};
        o0 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2v, a) - k);
        o1 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2v, b) - k);
    } else {
        const int b0 = (int8_t)(w & 0xff), b1 = (int8_t)((w >> 8) & 0xff), b2 = (int8_t)((w >> 16) & 0xff), b3 = (int8_t)(w >> 24);
        o0 = pk2((float)b0, (float)b1);
        o1 = pk2((float)b2, (float)b3);
    }
}
template <int DQ>
__device__ __forceinline__ h16 deq16(uint4 u)
{
    uint4 a, b;
    deq4<DQ>(u.x, a.x, a.y); deq4<DQ>(u.y, a.z, a.w); deq4<DQ>(u.z, b.x, b.y); deq4<DQ>(u.w, b.z, b.w);
    return mk16(a, b);
}

struct AP {
    const _Float16* Q;      // [T][NH][HD] fp16
    const void* K;          // slot base
    const void* V;
    const _Float16* ksc;
    const _Float16* vsc;
    const int* pos0p;       // device scalar: keys before the T new tokens (the T new rows are already appended)
    const float* rel;       // [T][NH][NREL] or null
    float* wm; float* wl; float* wo;
    int T, smax, chunk_min;
    long cap;
    float scale;
};

__device__ __forceinline__ void split_geom(int len, int smax, int chunk_min, int& chunk, int& nsp)
{
    int c = (len + smax - 1) / smax;
    c = (c + 63) & ~63;
    chunk = c > chunk_min ? c : chunk_min;
    nsp = (len + chunk - 1) / chunk;
}

template <int RT, int Q>
__global__ __launch_bounds__(128 * RT) void gkv_attend_k(const AP p)
{
    constexpr int ES = Q ? 1 : 2;
    constexpr int DQ = Q;   // Q = 1 fast dequant, Q = 2 plain dequant
    __shared__ __attribute__((aligned(16))) _Float16 qs[16 * RT * QP];
    __shared__ __attribute__((aligned(16))) float red[RT * 4 * 16 * RP];
    __shared__ float mlds[RT * 4 * 16], lsl[RT * 4 * 16];
    const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, n = l & 15, hh = l >> 4, kw = w & 3, rt = w >> 2;
    const int kvh = blockIdx.x, sp = blockIdx.y;
    const int pos0 = *p.pos0p, T = p.T, len = pos0 + T;
    int chunk, nsp;
    split_geom(len, p.smax, p.chunk_min, chunk, nsp);
    if (sp >= nsp) return;
    const int c0 = sp * chunk, c1 = min(c0 + chunk, len);
    for (int i = tid; i < 16 * RT * 16; i += 128 * RT) {
        const int row = i >> 4, cc = i & 15, t = row / 3, g = row - 3 * t;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (row < 3 * T) v = *(const uint4*)(p.Q + ((size_t)t * NH + 3 * kvh + g) * HD + 8 * cc);
        *(uint4*)(qs + row * QP + 8 * cc) = v;
    }
    __syncthreads();
    const int row = 16 * rt + n, tq = row / 3, gq = row - 3 * tq;
    const bool valid = row < 3 * T;
    const int head = 3 * kvh + gq;
    const int lim = valid ? pos0 + tq : pos0;
    const float* relb = (p.rel && valid) ? p.rel + ((size_t)tq * NH + head) * NREL : nullptr;
    const _Float16* qrow = qs + row * QP;
    const char* Kh = (const char*)p.K + (size_t)kvh * p.cap * HD * ES;
    const char* Vh = (const char*)p.V + (size_t)kvh * p.cap * HD * ES;
    const _Float16* kscp = p.ksc + (size_t)kvh * p.cap;
    const _Float16* vscp = p.vsc + (size_t)kvh * p.cap;
    f8 acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = (f8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -1e30f, ls = 0.f;
    const float c = p.scale * LOG2E;
    for (int sub = kw; c0 + sub * 16 < c1; sub += 4) {
        const int kbase = c0 + sub * 16;
        const char* kr = Kh + (size_t)(kbase + n) * (HD * ES);
        const int tl = kbase >> 6, s16 = (kbase & 63) >> 4;
        const char* vb = Vh + (((size_t)tl * HD + n) * 64 + s16 * 16) * ES;     // + tt * 16 * 64 * ES
        uint4 kf[8][Q ? 1 : 2], vf[8][Q ? 1 : 2];
        if constexpr (Q) {
#pragma unroll
            for (int ks = 0; ks < 8; ks++) kf[ks][0] = *(const uint4*)(kr + 16 * ks);
#pragma unroll
            for (int tt = 0; tt < 8; tt++) {
                const uint4 x = *(const uint4*)(vb + (size_t)tt * 16 * 64);
                vf[tt][0] = hh ? make_uint4(x.z, x.w, x.x, x.y) : x;
            }
        } else {
#pragma unroll
            for (int ks = 0; ks < 8; ks++) { kf[ks][0] = *(const uint4*)(kr + 32 * ks); kf[ks][1] = *(const uint4*)(kr + 32 * ks + 16); }
#pragma unroll
            for (int tt = 0; tt < 8; tt++) {
                const char* vp = vb + (size_t)tt * 16 * 64 * 2;
                vf[tt][0] = *(const uint4*)(vp + 16 * hh);
                vf[tt][1] = *(const uint4*)(vp + 16 * (hh ^ 1));
            }
        }
        uint4 su0 = make_uint4(0, 0, 0, 0), su1 = su0, vu0 = su0, vu1 = su0;
        if constexpr (Q) {
            su0 = *(const uint4*)(kscp + kbase); su1 = *(const uint4*)(kscp + kbase + 8);
            vu0 = *(const uint4*)(vscp + kbase); vu1 = *(const uint4*)(vscp + kbase + 8);
        }
        f8 s0 = (f8){0, 0, 0, 0, 0, 0, 0, 0}, s1 = s0, s2 = s0;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const uint4* qp = (const uint4*)(qrow + 16 * ks);
            const h16 bq = mk16(qp[0], qp[1]);
            h16 af;
            if constexpr (Q) af = deq16<DQ>(kf[ks][0]); else af = mk16(kf[ks][0], kf[ks][1]);
            if (ks % 3 == 0) s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bq, s0);
            else if (ks % 3 == 1) s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bq, s1);
            else s2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bq, s2);
        }
        float sv[8], vsc8[8];
        const uint32_t sk[8] = {su0.x, su0.y, su0.z, su0.w, su1.x, su1.y, su1.z, su1.w};
        const uint32_t vk[8] = {vu0.x, vu0.y, vu0.z, vu0.w, vu1.x, vu1.y, vu1.z, vu1.w};
        const bool biasable = relb && (kbase + 15 >= pos0 - (NREL - 1));
#pragma unroll
        for (int i = 0; i < 8; i++) {
            float x = ((s0[i] + s1[i]) + s2[i]) * c;
            if constexpr (Q) {
                const uint32_t a = hh ? (sk[i] >> 16) : (sk[i] & 0xffffu), b = hh ? (vk[i] >> 16) : (vk[i] & 0xffffu);
                x *= (float)__builtin_bit_cast(_Float16, (uint16_t)a);
                vsc8[i] = (float)__builtin_bit_cast(_Float16, (uint16_t)b);
            } else vsc8[i] = 1.f;
            const int key = kbase + 2 * i + hh, dist = lim - key;
            if (biasable && dist >= 0 && dist < NREL) x += relb[dist] * LOG2E;
            sv[i] = dist >= 0 ? x : -INFINITY;
        }
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; i++) mx = fmaxf(mx, sv[i]);
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
            pw[i >> 1] = pk2(p0 * vsc8[i], p1 * vsc8[i + 1]);
        }
        ls = ls * alpha + ps;
        if (__any(upd)) {
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] *= alpha;
        }
        const h16 pb = mk16(make_uint4(pw[0], pw[1], pw[2], pw[3]), make_uint4(perm16(pw[0]), perm16(pw[1]), perm16(pw[2]), perm16(pw[3])));
#pragma unroll
        for (int tt = 0; tt < 8; tt++) {
            h16 af;
            if constexpr (Q) af = deq16<DQ>(vf[tt][0]); else af = mk16(vf[tt][0], vf[tt][1]);
            acc[tt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, pb, acc[tt]);
        }
    }
    ls += permf(ls);
    if (hh == 0) mlds[(rt * 4 + kw) * 16 + n] = m;
    __syncthreads();
    float M = mlds[(rt * 4) * 16 + n];
#pragma unroll
    for (int k = 1; k < 4; k++) M = fmaxf(M, mlds[(rt * 4 + k) * 16 + n]);
    const float scl = __builtin_amdgcn_exp2f(m - M);
    if (hh == 0) lsl[(rt * 4 + kw) * 16 + n] = ls * scl;
    const size_t pbase = ((size_t)kvh * p.smax + sp) * 32 + 16 * rt;
    const int tr = tid & 127;
#pragma unroll
    for (int half = 0; half < 2; half++) {
#pragma unroll
        for (int tt = 0; tt < 4; tt++)
#pragma unroll
            for (int i = 0; i < 8; i++) red[((rt * 4 + kw) * 16 + n) * RP + 16 * tt + 2 * i + hh] = acc[half * 4 + tt][i] * scl;
        __syncthreads();
        for (int e = tr; e < 16 * 64; e += 128) {
            const int r = e >> 6, d = e & 63;
            const float v = red[((rt * 4 + 0) * 16 + r) * RP + d] + red[((rt * 4 + 1) * 16 + r) * RP + d] + red[((rt * 4 + 2) * 16 + r) * RP + d] + red[((rt * 4 + 3) * 16 + r) * RP + d];
            p.wo[(pbase + r) * HD + 64 * half + d] = v;
        }
        if (half == 0 && tr < 16) {
            float Mr = mlds[(rt * 4) * 16 + tr];
            for (int k = 1; k < 4; k++) Mr = fmaxf(Mr, mlds[(rt * 4 + k) * 16 + tr]);
            p.wm[pbase + tr] = Mr;
            p.wl[pbase + tr] = lsl[(rt * 4) * 16 + tr] + lsl[(rt * 4 + 1) * 16 + tr] + lsl[(rt * 4 + 2) * 16 + tr] + lsl[(rt * 4 + 3) * 16 + tr];
        }
        __syncthreads();
    }
}

// combine: one block per (t, head); sink joins the denominator once; optional gate (raw logits, sigmoid applied here)
__global__ __launch_bounds__(128) void gkv_combine_k(const float* __restrict__ wm, const float* __restrict__ wl, const float* __restrict__ wo, const float* __restrict__ sink,
    const void* __restrict__ gate, void* __restrict__ out, const int* __restrict__ pos0p, int T, int smax, int chunk_min, int out_bf16, int gate_bf16)
{
    const int b = blockIdx.x, t = b / NH, h = b % NH, kvh = h / 3, g = h % 3, row = t * 3 + g, d = threadIdx.x;
    const int len = *pos0p + T;
    int chunk, nsp;
    split_geom(len, smax, chunk_min, chunk, nsp);
    const size_t base = (size_t)kvh * smax * 32 + row;
    float M = -1e30f;
    for (int s = 0; s < nsp; s++) M = fmaxf(M, wm[base + (size_t)s * 32]);
    float L = 0.f;
    if (sink) { const float sk = sink[t * NH + h] * LOG2E; M = fmaxf(M, sk); L = __builtin_amdgcn_exp2f(sk - M); }
    float o = 0.f;
    for (int s = 0; s < nsp; s++) {
        const size_t pi = base + (size_t)s * 32;
        const float a = __builtin_amdgcn_exp2f(wm[pi] - M);
        L += wl[pi] * a;
        o += wo[pi * HD + d] * a;
    }
    o /= L;
    if (gate) {
        const size_t gi = ((size_t)t * NH + h) * HD + d;
        const float gv = gate_bf16 ? bf2f(((const uint16_t*)gate)[gi]) : (float)((const _Float16*)gate)[gi];
        o *= 1.f / (1.f + __expf(-gv));
    }
    const size_t oi = ((size_t)t * NH + h) * HD + d;
    if (out_bf16) ((uint16_t*)out)[oi] = (uint16_t)(__builtin_bit_cast(uint32_t, o) >> 16);   // truncation: only for shape-compat; the engine converts fp16 -> bf16 itself
    else ((_Float16*)out)[oi] = satf(o);
}

// bias table: rel[t][h][e] = sum_d r[t][h][d] * proj[d][e]  (r already l2-clipped and tau-scaled, proj [16][1024] fp32)
__global__ __launch_bounds__(256) void gkv_rel_k(const float* __restrict__ r, const float* __restrict__ proj, float* __restrict__ out)
{
    const int th = blockIdx.x;
    float rv[16];
#pragma unroll
    for (int d = 0; d < 16; d++) rv[d] = r[(size_t)th * 16 + d];
    for (int e = threadIdx.x; e < NREL; e += 256) {
        float a = 0.f;
#pragma unroll
        for (int d = 0; d < 16; d++) a += rv[d] * proj[d * NREL + e];
        out[(size_t)th * NREL + e] = a;
    }
}

// append T new K / V rows (in place, O(T)): K [T][NKV][HD], V same, fp16 or bf16 input; Q != 0 quantizes per (token, head) row: int8 + fp16 scale = absmax / 127
template <int Q>
__global__ __launch_bounds__(128) void gkv_append_k(void* __restrict__ Kc, void* __restrict__ Vc, _Float16* __restrict__ ksc, _Float16* __restrict__ vsc,
    const void* __restrict__ kn, const void* __restrict__ vn, const int* __restrict__ pos0p, long cap, int in_bf16)
{
    __shared__ float red[2][4];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long pos = *pos0p + t;
    const size_t si = ((size_t)t * NKV + h) * HD + d;
    float kx, vx;
    if (in_bf16) { kx = bf2f(((const uint16_t*)kn)[si]); vx = bf2f(((const uint16_t*)vn)[si]); }
    else { kx = (float)((const _Float16*)kn)[si]; vx = (float)((const _Float16*)vn)[si]; }
    const long tl = pos >> 6, r = pos & 63, g = r >> 4, rr = r & 15, sl = ((rr & 1) << 3) + (rr >> 1);
    const size_t vi = (((size_t)h * (cap >> 6) + tl) * HD + d) * 64 + g * 16 + sl;
    const size_t ki = ((size_t)h * cap + pos) * HD + d;
    if constexpr (!Q) {
        ((_Float16*)Kc)[ki] = satf(kx);
        ((_Float16*)Vc)[vi] = satf(vx);
    } else {
        float ak = fabsf(kx), av = fabsf(vx);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { ak = fmaxf(ak, __shfl_xor(ak, o)); av = fmaxf(av, __shfl_xor(av, o)); }
        if ((d & 31) == 0) { red[0][d >> 5] = ak; red[1][d >> 5] = av; }
        __syncthreads();
        ak = fmaxf(fmaxf(red[0][0], red[0][1]), fmaxf(red[0][2], red[0][3]));
        av = fmaxf(fmaxf(red[1][0], red[1][1]), fmaxf(red[1][2], red[1][3]));
        const _Float16 sk = (_Float16)fmaxf(ak / 127.f, 1e-8f), sv = (_Float16)fmaxf(av / 127.f, 1e-8f);
        const float qk = rintf(kx / (float)sk), qv = rintf(vx / (float)sv);
        ((int8_t*)Kc)[ki] = (int8_t)fminf(fmaxf(qk, -127.f), 127.f);
        ((int8_t*)Vc)[vi] = (int8_t)fminf(fmaxf(qv, -127.f), 127.f);
        if (d == 0) { ksc[(size_t)h * cap + pos] = sk; vsc[(size_t)h * cap + pos] = sv; }
    }
}

// self-test of the int8 -> fp16 conversion (both modes): in [N] u32 -> out [2][N][2] u32
template <int DQ>
__global__ void gkv_deq_test_k(const uint32_t* in, uint32_t* out, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { uint32_t a, b; deq4<DQ>(in[i], a, b); out[2 * i] = a; out[2 * i + 1] = b; }
}

extern "C" int gkv_attend(const void* Q, const void* K, const void* V, const void* ksc, const void* vsc, const int* pos0, const float* rel, float* ws, int T, int smax, int chunk_min,
    long cap, float scale, int q, void* stream)
{
    hipStream_t st = (hipStream_t)stream;
    if (T < 1 || T > 8 || smax < 1 || smax > 256 || chunk_min < 64 || (chunk_min & 63) || (cap & 63) || q < 0 || q > 2) return -2;
    AP p;
    p.Q = (const _Float16*)Q; p.K = K; p.V = V; p.ksc = (const _Float16*)ksc; p.vsc = (const _Float16*)vsc; p.pos0p = pos0; p.rel = rel;
    p.wm = ws; p.wl = ws + (size_t)NKV * smax * 32; p.wo = ws + 2 * (size_t)NKV * smax * 32; p.T = T; p.smax = smax; p.chunk_min = chunk_min; p.cap = cap; p.scale = scale;
    dim3 g(NKV, smax);
    const int rt = (3 * T + 15) / 16;
#define LA(R_, Q_) gkv_attend_k<R_, Q_><<<g, 128 * R_, 0, st>>>(p)
    if (rt == 1) { if (q == 0) LA(1, 0); else if (q == 1) LA(1, 1); else LA(1, 2); }
    else { if (q == 0) LA(2, 0); else if (q == 1) LA(2, 1); else LA(2, 2); }
    return (int)hipGetLastError();
}
extern "C" size_t gkv_ws_floats(int smax) { return (size_t)NKV * smax * 32 * (2 + HD); }
extern "C" int gkv_combine(const float* ws, const float* sink, const void* gate, void* out, const int* pos0, int T, int smax, int chunk_min, int out_bf16, int gate_bf16, void* stream)
{
    const float* wm = ws; const float* wl = ws + (size_t)NKV * smax * 32; const float* wo = ws + 2 * (size_t)NKV * smax * 32;
    gkv_combine_k<<<T * NH, 128, 0, (hipStream_t)stream>>>(wm, wl, wo, sink, gate, out, pos0, T, smax, chunk_min, out_bf16, gate_bf16);
    return (int)hipGetLastError();
}
extern "C" int gkv_rel(const float* r, const float* proj, float* out, int T, void* stream)
{
    gkv_rel_k<<<T * NH, 256, 0, (hipStream_t)stream>>>(r, proj, out);
    return (int)hipGetLastError();
}
extern "C" int gkv_append(void* K, void* V, void* ksc, void* vsc, const void* kn, const void* vn, const int* pos0, int T, long cap, int q, int in_bf16, void* stream)
{
    dim3 g(T, NKV);
    if (q) gkv_append_k<1><<<g, 128, 0, (hipStream_t)stream>>>(K, V, (_Float16*)ksc, (_Float16*)vsc, kn, vn, pos0, cap, in_bf16);
    else gkv_append_k<0><<<g, 128, 0, (hipStream_t)stream>>>(K, V, (_Float16*)ksc, (_Float16*)vsc, kn, vn, pos0, cap, in_bf16);
    return (int)hipGetLastError();
}
extern "C" int gkv_deq_test(const void* in, void* out, int N, int mode, void* stream)
{
    if (mode == 1) gkv_deq_test_k<1><<<(N + 255) / 256, 256, 0, (hipStream_t)stream>>>((const uint32_t*)in, (uint32_t*)out, N);
    else gkv_deq_test_k<2><<<(N + 255) / 256, 256, 0, (hipStream_t)stream>>>((const uint32_t*)in, (uint32_t*)out, N);
    return (int)hipGetLastError();
}
