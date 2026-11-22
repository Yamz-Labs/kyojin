// Inkling 2 Mini decode attention for the RTX 3090 (sm_86): CUDA port of the Strix kernels (gkv global layers + lattn local ring), mma.sync m16n8k16.
// One kernel template serves both ops (LOCAL = 0: global slot, keys 0..len-1, bias 0 beyond 1024; LOCAL = 1: ring R, window 1024, keys at dist >= 1024 masked).
// Cache layout (per slot, element type ET = fp16 or bf16, CAP % 16 == 0; CAP = R for a ring):
//   K [NKV][CAP][128]            row-major (one key = 256 B)
//   V [NKV][CAP/16][128 dims][16 keys]   dim-major 16-key tiles; the key at tile position j = 8*nt + 2*tg + e is stored at p = 4*tg + 2*nt + e, so the PV B fragment
//                                        of one lane (keys 2tg, 2tg+1 | 2tg+8, 2tg+9) is ONE 8-byte load.
// Attention orientation = FlashAttention-2 style: S = Q K^T with the 3*T query rows of a kv head as the M = 16 rows of the mma (RT = 1 tile for T <= 5, 2 tiles for T = 6..8),
// keys as N = 8 columns (2 n-tiles per 16-key step), P stays in registers as the A operand of O = P V (the C fragment of S IS the A fragment of PV).
// Dim permutation: the 128 dims are consumed in an order D(ks, j) chosen so that each lane loads 16 contiguous bytes of a key row per 32 dims (Q fragments use the same order).
// Grid (NKV, smax), block 128 * RT threads = 4 key-waves x RT row tiles; split-K by the device length (graph safe); merge of the 4 waves in smem; fp32 workspace (m, l, O) per
// (kv head, split, row); combine kernel merges the splits, adds the sink once, applies the optional gate.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define NKV 8
#define NH 24
#define HD 128
#define NREL 1024
#define LOG2E 1.4426950408889634f
#define LAZY 8.0f
#define RP 68    // floats per row of the merge buffer

__device__ unsigned int g_fire[8];   // 0 gkv attend blocks, 1 lattn attend blocks, 2 combine blocks, 3 append blocks, 4 rel blocks

__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

template <int BF> __device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    if (BF)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <int BF> __device__ __forceinline__ uint32_t pack2(float lo, float hi)
{
    if (BF) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *(uint32_t*)&v; }
    __half2 v = __floats2half2_rn(lo, hi); return *(uint32_t*)&v;
}
template <int BF> __device__ __forceinline__ uint16_t cvt1(float x)
{
    if (BF) { __nv_bfloat16 v = __float2bfloat16_rn(x); return *(uint16_t*)&v; }
    x = fminf(fmaxf(x, -65504.f), 65504.f);   // fp16 saturating clamp on write
    __half v = __float2half_rn(x); return *(uint16_t*)&v;
}
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ float h2f(uint16_t b) { return __half2float(*(__half*)&b); }

struct AP {
    const uint16_t* Q;      // [T][NH][HD] (ET)
    const char* K;          // slot base
    const char* V;
    const int* pos0p;       // device scalar: keys before the T new tokens (the T new rows are already appended)
    const float* rel;       // [T][NH][NREL] or null
    float* wm; float* wl; float* wo;
    int T, smax, chunk_min, R;   // R = ring rows (LOCAL) ; CAP for global is only needed by the K / V base offsets
    long cap;
    float scale;
    unsigned int* cnt;      // fused combine: per-kv-head arrival counters (zero at start, self-resetting) or null
    const float* sink; const void* gate; void* out; int out_kind, gate_kind;
};

__device__ __forceinline__ void geom(int pos0, int T, int local, int smax, int chunk_min, int& kb0, int& chunk, int& nsp, int& len)
{
    len = pos0 + T;
    kb0 = local ? (max(0, pos0 - (NREL - 1)) & ~15) : 0;
    const int n = len - kb0, gran = local ? 16 : 64;
    int c = (n + smax - 1) / smax;
    c = (c + gran - 1) & ~(gran - 1);
    chunk = c > chunk_min ? c : chunk_min;
    nsp = (n + chunk - 1) / chunk;
}

// one (t, head, dim) output element from the split partials (L2 reads: __ldcg, partials written by other blocks of the same launch)
__device__ __forceinline__ void comb_one(const float* wm, const float* wl, const float* wo, const float* sink, const void* gate, void* out,
    int kvh, int t, int h, int nsp, int smax, int d, int out_kind, int gate_kind)
{
    const int gq = h % 3, row = t * 3 + gq;
    const size_t base = (size_t)kvh * smax * 32 + row;
    float M = -1e30f;
    for (int s = 0; s < nsp; s++) M = fmaxf(M, __ldcg(wm + base + (size_t)s * 32));
    float L = 0.f;
    if (sink) { const float sk = sink[t * NH + h] * LOG2E; M = fmaxf(M, sk); L = ex2(sk - M); }
    float o = 0.f;
    for (int s = 0; s < nsp; s++) {
        const size_t pi = base + (size_t)s * 32;
        const float a = ex2(__ldcg(wm + pi) - M);
        L += __ldcg(wl + pi) * a;
        o += __ldcg(wo + pi * HD + d) * a;
    }
    o /= L;
    const size_t oi = ((size_t)t * NH + h) * HD + d;
    if (gate) {
        float gv;
        if (gate_kind == 0) gv = h2f(((const uint16_t*)gate)[oi]); else if (gate_kind == 1) gv = bf2f(((const uint16_t*)gate)[oi]); else gv = ((const float*)gate)[oi];
        o *= 1.f / (1.f + __expf(-gv));
    }
    if (out_kind) ((uint16_t*)out)[oi] = cvt1<1>(o); else ((uint16_t*)out)[oi] = cvt1<0>(o);
}

template <int RT, int BF, int LOCAL>
__global__ __launch_bounds__(128 * RT) void attn_k(const AP p)
{
    __shared__ __align__(16) float red[RT * 4 * 16 * RP];
    __shared__ float mlds[RT * 4 * 16], lsl[RT * 4 * 16];
    __shared__ int islast;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, tg = lane & 3, kw = w & 3, rt = w >> 2;
    const int kvh = blockIdx.x, sp = blockIdx.y;
    const int pos0 = *p.pos0p, T = p.T;
    int kb0, chunk, nsp, len;
    geom(pos0, T, LOCAL, p.smax, p.chunk_min, kb0, chunk, nsp, len);
    if (sp >= nsp) return;
    if (tid == 0) atomicAdd(&g_fire[LOCAL], 1u);
    const int c0 = kb0 + sp * chunk, c1 = min(c0 + chunk, len);
    // rows of this lane: ra = 16 rt + g, rb = ra + 8 ; row = 3 t + gq
    const int ra = 16 * rt + g, rb = ra + 8;
    const int ta = ra / 3, ga = ra - 3 * ta, tb = rb / 3, gb = rb - 3 * tb;
    const bool va = ra < 3 * T, vb = rb < 3 * T;
    const int lima = va ? pos0 + ta : pos0, limb = vb ? pos0 + tb : pos0;
    const float* rela = (p.rel && va) ? p.rel + ((size_t)ta * NH + 3 * kvh + ga) * NREL : nullptr;
    const float* relb = (p.rel && vb) ? p.rel + ((size_t)tb * NH + 3 * kvh + gb) * NREL : nullptr;
    // Q fragments (A operand), dim order D(ks, j): 32 c + 8 tg + 4 sub + 2 breg + e
    uint32_t qa[8][4];
    {
        const uint16_t* qra = p.Q + ((size_t)ta * NH + 3 * kvh + ga) * HD;
        const uint16_t* qrb = p.Q + ((size_t)tb * NH + 3 * kvh + gb) * HD;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int d0 = 32 * (ks >> 1) + 8 * tg + 4 * (ks & 1);
            uint2 xa = make_uint2(0, 0), xb = xa;
            if (va) xa = *(const uint2*)(qra + d0);
            if (vb) xb = *(const uint2*)(qrb + d0);
            qa[ks][0] = xa.x; qa[ks][1] = xb.x; qa[ks][2] = xa.y; qa[ks][3] = xb.y;
        }
    }
    const char* Kh = p.K + (size_t)kvh * p.cap * HD * 2;
    const char* Vh = p.V + (size_t)kvh * p.cap * HD * 2;
    float acc[16][4];
#pragma unroll
    for (int d = 0; d < 16; d++) { acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f; }
    float ma = -1e30f, mb = -1e30f, lsa = 0.f, lsb = 0.f;
    const float c = p.scale * LOG2E;
    for (int sub = kw; c0 + sub * 16 < c1; sub += 4) {
        const int kbase = c0 + sub * 16;
        const int krow = LOCAL ? kbase % p.R : kbase;
        const char* kr = Kh + (size_t)(krow + g) * (HD * 2) + 16 * tg;
        const char* vt = Vh + (size_t)(krow >> 4) * (HD * 16 * 2) + (size_t)g * 32 + 8 * tg;
        uint4 kf[2][4];
        uint2 vf[16];
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int cc = 0; cc < 4; cc++) kf[nt][cc] = *(const uint4*)(kr + (size_t)nt * 8 * (HD * 2) + 64 * cc);
#pragma unroll
        for (int d = 0; d < 16; d++) vf[d] = *(const uint2*)(vt + (size_t)d * 8 * 32);
        float s[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; nt++) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                const uint4 u = kf[nt][ks >> 1];
                if (ks & 1) mma16816<BF>(s[nt], qa[ks], u.z, u.w);
                else mma16816<BF>(s[nt], qa[ks], u.x, u.y);
            }
        }
        // scores: s[nt][0,1] row ra, keys j = 8 nt + 2 tg + {0,1}; s[nt][2,3] row rb
        const bool biasable = LOCAL ? true : (kbase + 15 >= pos0 - (NREL - 1));
        float xa[4], xb[4];
        float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = kbase + 8 * nt + 2 * tg + e;
                const int da = lima - key, db = limb - key;
                float va_ = s[nt][e] * c, vb_ = s[nt][2 + e] * c;
                const bool oka = LOCAL ? (da >= 0 && da < NREL) : (da >= 0);
                const bool okb = LOCAL ? (db >= 0 && db < NREL) : (db >= 0);
                if (biasable) {
                    if (rela && da >= 0 && da < NREL) va_ += rela[da] * LOG2E;
                    if (relb && db >= 0 && db < NREL) vb_ += relb[db] * LOG2E;
                }
                va_ = oka ? va_ : -INFINITY;
                vb_ = okb ? vb_ : -INFINITY;
                xa[2 * nt + e] = va_; xb[2 * nt + e] = vb_;
                mxa = fmaxf(mxa, va_); mxb = fmaxf(mxb, vb_);
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1)); mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1)); mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const bool upa = mxa > ma + LAZY, upb = mxb > mb + LAZY;
        const float mna = upa ? mxa : ma, mnb = upb ? mxb : mb;
        const float ala = upa ? ex2(ma - mna) : 1.f, alb = upb ? ex2(mb - mnb) : 1.f;
        ma = mna; mb = mnb;
        float pa_[4], pb_[4];
#pragma unroll
        for (int i = 0; i < 4; i++) { pa_[i] = ex2(xa[i] - mna); pb_[i] = ex2(xb[i] - mnb); }
        lsa = lsa * ala + ((pa_[0] + pa_[1]) + (pa_[2] + pa_[3]));
        lsb = lsb * alb + ((pb_[0] + pb_[1]) + (pb_[2] + pb_[3]));
        if (__any_sync(0xffffffffu, upa || upb)) {
#pragma unroll
            for (int d = 0; d < 16; d++) { acc[d][0] *= ala; acc[d][1] *= ala; acc[d][2] *= alb; acc[d][3] *= alb; }
        }
        uint32_t pf[4];
        pf[0] = pack2<BF>(pa_[0], pa_[1]);   // row ra, keys 2tg, 2tg+1 (nt 0)
        pf[1] = pack2<BF>(pb_[0], pb_[1]);   // row rb
        pf[2] = pack2<BF>(pa_[2], pa_[3]);   // row ra, keys 8 + 2tg.. (nt 1)
        pf[3] = pack2<BF>(pb_[2], pb_[3]);
#pragma unroll
        for (int d = 0; d < 16; d++) mma16816<BF>(acc[d], pf, vf[d].x, vf[d].y);
    }
    lsa += __shfl_xor_sync(0xffffffffu, lsa, 1); lsa += __shfl_xor_sync(0xffffffffu, lsa, 2);
    lsb += __shfl_xor_sync(0xffffffffu, lsb, 1); lsb += __shfl_xor_sync(0xffffffffu, lsb, 2);
    if (tg == 0) { mlds[(rt * 4 + kw) * 16 + g] = ma; mlds[(rt * 4 + kw) * 16 + g + 8] = mb; }
    __syncthreads();
    float Ma = mlds[(rt * 4) * 16 + g], Mb = mlds[(rt * 4) * 16 + g + 8];
#pragma unroll
    for (int k = 1; k < 4; k++) { Ma = fmaxf(Ma, mlds[(rt * 4 + k) * 16 + g]); Mb = fmaxf(Mb, mlds[(rt * 4 + k) * 16 + g + 8]); }
    const float scla = ex2(ma - Ma), sclb = ex2(mb - Mb);
    if (tg == 0) { lsl[(rt * 4 + kw) * 16 + g] = lsa * scla; lsl[(rt * 4 + kw) * 16 + g + 8] = lsb * sclb; }
    const size_t pbase = ((size_t)kvh * p.smax + sp) * 32 + 16 * rt;
    const int tr = tid & 127;
    float* rw = red + ((rt * 4 + kw) * 16) * RP;
#pragma unroll
    for (int half = 0; half < 2; half++) {
#pragma unroll
        for (int dd = 0; dd < 8; dd++) {
            const int d = half * 8 + dd;
            *(float2*)(rw + g * RP + 8 * dd + 2 * tg) = make_float2(acc[d][0] * scla, acc[d][1] * scla);
            *(float2*)(rw + (g + 8) * RP + 8 * dd + 2 * tg) = make_float2(acc[d][2] * sclb, acc[d][3] * sclb);
        }
        __syncthreads();
        for (int e = tr; e < 16 * 64; e += 128) {
            const int r = e >> 6, dc = e & 63;
            const float* rb4 = red + (rt * 4) * 16 * RP + r * RP + dc;
            p.wo[(pbase + r) * HD + 64 * half + dc] = rb4[0] + rb4[16 * RP] + rb4[2 * 16 * RP] + rb4[3 * 16 * RP];
        }
        if (half == 0 && tr < 16) {
            float Mr = mlds[(rt * 4) * 16 + tr];
            for (int k = 1; k < 4; k++) Mr = fmaxf(Mr, mlds[(rt * 4 + k) * 16 + tr]);
            p.wm[pbase + tr] = Mr;
            p.wl[pbase + tr] = lsl[(rt * 4) * 16 + tr] + lsl[(rt * 4 + 1) * 16 + tr] + lsl[(rt * 4 + 2) * 16 + tr] + lsl[(rt * 4 + 3) * 16 + tr];
        }
        __syncthreads();
    }
    if (p.cnt) {
        __threadfence();
        __syncthreads();
        if (tid == 0) islast = (atomicAdd(p.cnt + kvh, 1u) == (unsigned)(nsp - 1));
        __syncthreads();
        if (islast) {
            if (tid == 0) p.cnt[kvh] = 0;
            __threadfence();
            if (tid == 0) atomicAdd(&g_fire[5], 1u);
            const float* wm = p.wm; const float* wl = p.wl; const float* wo = p.wo;
            for (int row = tid >> 7; row < 3 * T; row += RT) comb_one(wm, wl, wo, p.sink, p.gate, p.out, kvh, row / 3, 3 * kvh + row % 3, nsp, p.smax, tid & 127, p.out_kind, p.gate_kind);
        }
    }
}

// combine: one block per (t, head); sink joins the denominator once; optional gate (raw logits, sigmoid applied here); out_kind 0 fp16, 1 bf16; gate_kind 0 fp16, 1 bf16, 2 fp32
__global__ __launch_bounds__(128) void combine_k(const float* __restrict__ wm, const float* __restrict__ wl, const float* __restrict__ wo, const float* __restrict__ sink,
    const void* __restrict__ gate, void* __restrict__ out, const int* __restrict__ pos0p, int T, int smax, int chunk_min, int local, int out_kind, int gate_kind)
{
    const int b = blockIdx.x, t = b / NH, h = b % NH, kvh = h / 3, d = threadIdx.x;
    if (d == 0) atomicAdd(&g_fire[2], 1u);
    int kb0, chunk, nsp, len;
    geom(*pos0p, T, local, smax, chunk_min, kb0, chunk, nsp, len);
    comb_one(wm, wl, wo, sink, gate, out, kvh, t, h, nsp, smax, d, out_kind, gate_kind);
}

// bias table: rel[t][h][e] = sum_d r[t][h][d] * proj[d][e]  (r already l2-clipped and tau-scaled, proj [16][1024] fp32)
__global__ __launch_bounds__(256) void rel_k(const float* __restrict__ r, const float* __restrict__ proj, float* __restrict__ out)
{
    const int th = blockIdx.x;
    if (threadIdx.x == 0) atomicAdd(&g_fire[4], 1u);
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

// append T new K / V rows in place: K [T][NKV][HD], V same; in_kind 0 fp16, 1 bf16, 2 fp32; stored as ET (BF). LOCAL: row = pos % R, a chunk longer than R writes only its last R rows
template <int BF>
__global__ __launch_bounds__(128) void append_k(char* __restrict__ Kc, char* __restrict__ Vc, const void* __restrict__ kn, const void* __restrict__ vn,
    const int* __restrict__ pos0p, long cap, int T, int local, int in_kind)
{
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (local && t + (int)cap < T) return;
    if (d == 0 && h == 0) atomicAdd(&g_fire[3], 1u);
    const long pos = *pos0p + t;
    const long row = local ? pos % cap : pos;
    const size_t si = ((size_t)t * NKV + h) * HD + d;
    float kx, vx;
    if (in_kind == 0) { kx = h2f(((const uint16_t*)kn)[si]); vx = h2f(((const uint16_t*)vn)[si]); }
    else if (in_kind == 1) { kx = bf2f(((const uint16_t*)kn)[si]); vx = bf2f(((const uint16_t*)vn)[si]); }
    else { kx = ((const float*)kn)[si]; vx = ((const float*)vn)[si]; }
    const int j = (int)(row & 15), nt = j >> 3, tgv = (j & 7) >> 1, e = j & 1, pp = 4 * tgv + 2 * nt + e;
    const size_t vi = (((size_t)h * (cap >> 4) + (row >> 4)) * HD + d) * 16 + pp;
    const size_t ki = ((size_t)h * cap + row) * HD + d;
    ((uint16_t*)Kc)[ki] = cvt1<BF>(kx);
    ((uint16_t*)Vc)[vi] = cvt1<BF>(vx);
}

// sustained mma.sync m16n8k16 rate (f16 in, f32 acc): 8 independent chains per warp
template <int BF>
__global__ void mma_peak_k(float* out, int iters)
{
    float c[8][4];
    uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
#pragma unroll
    for (int i = 0; i < 8; i++) { c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f; }
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int i = 0; i < 8; i++) mma16816<BF>(c[i], a, 0x3c003c00u, 0x3c003c00u);
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (s == 123456.f) out[0] = s;
}

extern "C" int ink2attn_attend(const void* Q, const void* K, const void* V, const int* pos0, const float* rel, float* ws, int T, int smax, int chunk_min,
    long cap, float scale, int local, int bf, int R, unsigned int* cnt, const float* sink, const void* gate, void* out, int out_kind, int gate_kind, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    if (T < 1 || T > 8 || smax < 1 || smax > 256 || chunk_min < 16 || (chunk_min & 15) || (cap & 15) || (!local && (chunk_min & 63))) return -2;
    AP p;
    p.Q = (const uint16_t*)Q; p.K = (const char*)K; p.V = (const char*)V; p.pos0p = pos0; p.rel = rel;
    p.wm = ws; p.wl = ws + (size_t)NKV * smax * 32; p.wo = ws + 2 * (size_t)NKV * smax * 32; p.T = T; p.smax = smax; p.chunk_min = chunk_min; p.R = R; p.cap = cap; p.scale = scale;
    p.cnt = cnt; p.sink = sink; p.gate = gate; p.out = out; p.out_kind = out_kind; p.gate_kind = gate_kind;
    dim3 g(NKV, smax);
    const int rt = (3 * T + 15) / 16;
#define LA(R_, B_, L_) attn_k<R_, B_, L_><<<g, 128 * R_, 0, st>>>(p)
    if (rt == 1) { if (local) { bf ? LA(1, 1, 1) : LA(1, 0, 1); } else { bf ? LA(1, 1, 0) : LA(1, 0, 0); } }
    else { if (local) { bf ? LA(2, 1, 1) : LA(2, 0, 1); } else { bf ? LA(2, 1, 0) : LA(2, 0, 0); } }
    return (int)cudaGetLastError();
}
extern "C" size_t ink2attn_ws_floats(int smax) { return (size_t)NKV * smax * 32 * (2 + HD); }
extern "C" int ink2attn_combine(const float* ws, const float* sink, const void* gate, void* out, const int* pos0, int T, int smax, int chunk_min, int local, int out_kind, int gate_kind, void* stream)
{
    const float* wm = ws; const float* wl = ws + (size_t)NKV * smax * 32; const float* wo = ws + 2 * (size_t)NKV * smax * 32;
    combine_k<<<T * NH, 128, 0, (cudaStream_t)stream>>>(wm, wl, wo, sink, gate, out, pos0, T, smax, chunk_min, local, out_kind, gate_kind);
    return (int)cudaGetLastError();
}
extern "C" int ink2attn_rel(const float* r, const float* proj, float* out, int T, void* stream)
{
    rel_k<<<T * NH, 256, 0, (cudaStream_t)stream>>>(r, proj, out);
    return (int)cudaGetLastError();
}
extern "C" int ink2attn_append(void* K, void* V, const void* kn, const void* vn, const int* pos0, int T, long cap, int local, int in_kind, int bf, void* stream)
{
    dim3 g(T, NKV);
    if (bf) append_k<1><<<g, 128, 0, (cudaStream_t)stream>>>((char*)K, (char*)V, kn, vn, pos0, cap, T, local, in_kind);
    else append_k<0><<<g, 128, 0, (cudaStream_t)stream>>>((char*)K, (char*)V, kn, vn, pos0, cap, T, local, in_kind);
    return (int)cudaGetLastError();
}
extern "C" int ink2attn_mma_peak(float* out, int iters, int blocks, int bf, void* stream)
{
    if (bf) mma_peak_k<1><<<blocks, 128, 0, (cudaStream_t)stream>>>(out, iters);
    else mma_peak_k<0><<<blocks, 128, 0, (cudaStream_t)stream>>>(out, iters);
    return (int)cudaGetLastError();
}
// device path counters (fires-check): out[0..4]; synchronous, call outside graph capture
extern "C" int ink2attn_fires(unsigned int* host_out)
{
    cudaDeviceSynchronize();
    return (int)cudaMemcpyFromSymbol(host_out, g_fire, 6 * sizeof(unsigned int));
}
