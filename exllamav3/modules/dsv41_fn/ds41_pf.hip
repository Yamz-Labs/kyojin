// DS4.1 prefill kernels for gfx1151: sparse attention (WMMA), routed-expert gather-sum, rope rows.
//
// ds41_sparse_attn_pf: sparse prefill attention (MQA, 64 heads x 512, one shared latent kv, per-row key lists, per-head sink) for gfx1151 WMMA f16.
// Replaces ops.sparse_attention (gather + fp32 einsums) for L > 1. Math of the reference: scores = (q . k) * scale in fp32, softmax over the
// listed keys plus a sink logit that only enters the denominator, o = (p @ v) / den, v == k. Keys with idx < 0 are masked; a row without
// valid key gives 0. Differences to the reference: the fp32 accumulation order (WMMA chains), p is rounded to fp16 for the PV product.
// One block of 16 waves per row: wave (head group hg = w & 3, d quarter dq = w >> 2). The 32 keys of a chunk are staged once in LDS for all
// waves (K and V are the same latent). Waves dq = 0, 1 of a group compute the QK tile of keys 16 dq .. 16 dq + 15 and exchange the scores through
// LDS; all four waves of a group then run the identical softmax and PV over their own quarter of d.
// Fragment layout (gfx1151): A: lane l, elem r -> A[l & 15][r]; B: lane l, elem r -> B[r][l & 15]; C: lane l, reg r -> C[2r + (l >> 4)][l & 15]
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <hip/hip_fp16.h>

typedef _Float16 half_t;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

#define HD 512
#define NHEAD 64
#define DQ 128            // d per wave (quarter)
#define BN 32             // keys per chunk
#define LDK 520           // LDS pitch of a staged key row in halves

__device__ __forceinline__ v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }
__device__ __forceinline__ v16h ld16h(const half_t* p) {
    v4u a = *(const v4u*) p, b = *(const v4u*) (p + 8);
    union { v4u u[2]; v16h h; } x; x.u[0] = a; x.u[1] = b; return x.h;
}

// q [R, 64, 512] fp16, kv [N, 512] fp16, idx [R, K] int32, sink [64] fp32, o [R, 64, 512] fp16. grid = R, block 512.
extern "C" __global__ __launch_bounds__(512) void ds41_sparse_attn_pf(
    const half_t* __restrict__ q, const half_t* __restrict__ kv, const int* __restrict__ idx, const float* __restrict__ sink,
    half_t* __restrict__ o, int K, int N, float scale)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x, wv = tid >> 5, lane = tid & 31;
    const int hg = wv & 3, dq = wv >> 2;
    const int n = lane & 15, hh = lane >> 4;
    __shared__ half_t sK[BN * LDK];
    __shared__ float sS[4 * 2 * 8 * 32];
    __shared__ int sTok[BN];

    v8f acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -__builtin_inff(), l = 0.0f;
    const half_t* qh = q + ((int64_t) row * NHEAD + hg * 16 + n) * HD;
    const int nch = (K + BN - 1) / BN;
    // The keys of chunk c + 1 are loaded into registers while chunk c computes (key j = tid >> 4, 16 threads per key, 4 x 16 B each), and
    // written to LDS at the top of the next iteration: the global-load latency hides behind QK / softmax / PV. Math unchanged.
    const int kj = tid >> 4, kpart = tid & 15;
    const half_t* kvrow = kv;
    int ptok = -1;
    v4u pre[4];
#define DS41_PF_FETCH(cc) { \
        const int kk_ = (cc) * BN + kj; \
        const int id_ = (cc) < nch && kk_ < K ? idx[(int64_t) row * K + kk_] : -1; \
        ptok = id_ >= 0 ? min(id_, N - 1) : -1; \
        const v4u* src_ = (const v4u*) (kvrow + (int64_t) max(ptok, 0) * HD) + kpart * 4; \
        _Pragma("unroll") for (int e_ = 0; e_ < 4; ++e_) pre[e_] = ptok >= 0 ? src_[e_] : (v4u) {0, 0, 0, 0}; }
    DS41_PF_FETCH(0)
    for (int c = 0; c < nch; ++c) {
        if (kpart == 0) sTok[kj] = ptok;
        {
            v4u* dst = (v4u*) (sK + kj * LDK) + kpart * 4;
#pragma unroll
            for (int e = 0; e < 4; ++e) dst[e] = pre[e];
        }
        __syncthreads();
        DS41_PF_FETCH(c + 1)
        const unsigned vmask = (unsigned) __ballot(sTok[lane] >= 0);
        if (vmask == 0u) { __syncthreads(); continue; }
        if (dq < 2) {
            v8f S = {0, 0, 0, 0, 0, 0, 0, 0};
            const half_t* kp = sK + (16 * dq + n) * LDK;
#pragma unroll 4
            for (int ks = 0; ks < 32; ++ks) S = wmma(ld16h(kp + 16 * ks), ld16h(qh + 16 * ks), S);
            float* ps = sS + ((hg * 2 + dq) * 8) * 32 + lane;
#pragma unroll
            for (int r = 0; r < 8; ++r) ps[r * 32] = S[r];
        }
        __syncthreads();
        float s[2][8], mx = -__builtin_inff();
        bool ok[2][8];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                ok[t][r] = (vmask >> (16 * t + 2 * r + hh)) & 1u;
                s[t][r] = ok[t][r] ? sS[((hg * 2 + t) * 8 + r) * 32 + lane] * scale : -__builtin_inff();
                mx = fmaxf(mx, s[t][r]);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16));
        const float m_new = fmaxf(m, mx);
        const float m_exp = (m_new == -__builtin_inff()) ? 0.0f : m_new;
        float p[2][8], sum = 0.0f;
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                p[t][r] = ok[t][r] ? __expf(s[t][r] - m_exp) : 0.0f;
                sum += p[t][r];
            }
        const float alpha = (m == -__builtin_inff()) ? 0.0f : __expf(m - m_exp);
        sum += __shfl_xor(sum, 16);
        l = l * alpha + sum;
        m = m_new;
        v16h pf[2];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const half_t own = (half_t) p[t][r];
                union { half_t h; uint16_t u; } cv; cv.h = own;
                const uint16_t other = (uint16_t) __shfl_xor((int) cv.u, 16);
                union { uint16_t u; half_t h; } oc; oc.u = other;
                pf[t][2 * r] = hh == 0 ? own : oc.h;
                pf[t][2 * r + 1] = hh == 0 ? oc.h : own;
            }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[nt][i] *= alpha;
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                v16h A;
#pragma unroll
                for (int r = 0; r < 16; ++r) A[r] = sK[(16 * t + r) * LDK + dq * DQ + 16 * nt + n];
                acc[nt] = wmma(A, pf[t], acc[nt]);
            }
        __syncthreads();
    }
    const float sk = sink[hg * 16 + n];
    const float m_f = fmaxf(m, sk);
    const float w = __expf(m - m_f);                 // m == -inf -> 0
    const float den = l * w + __expf(sk - m_f);
    const float scl = w / den;
    // output: lane (head n, parity hh) holds d = 16 nt + 2 r + hh; pair with the other half-wave and store 4 B (alternating r)
    half_t* ob = o + ((int64_t) row * NHEAD + hg * 16 + n) * HD + dq * DQ;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float own = acc[nt][r] * scl;
            const float other = __shfl_xor(own, 16);
            if ((r & 1) == hh) {
                const half_t lo = (half_t) (hh == 0 ? own : other), hi = (half_t) (hh == 0 ? other : own);
                union { half_t h[2]; uint32_t u; } pk; pk.h[0] = lo; pk.h[1] = hi;
                *(uint32_t*) (ob + 16 * nt + 2 * r) = pk.u;
            }
        }
}

// ds41_gather_sum: out[t] = sum over slots k of res[inv[t * K + k]] in slot order (rows with inv < 0 contribute nothing). res fp32 [N, H] holds the
// routed-expert results in group-sorted order, inv maps a (token, slot) pick to its row. Same fp32 additions in the same order as the
// reference's slot-order reduction. grid = T, block 256, H % 4 == 0.
extern "C" __global__ __launch_bounds__(256) void ds41_gather_sum(
    const float* __restrict__ res, const int* __restrict__ inv, float* __restrict__ out, int K, int H)
{
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < H / 4; i += 256) {
        float4 a = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = 0; k < K; ++k) {
            const int r = inv[(int64_t) t * K + k];
            if (r >= 0) {
                const float4 x = *(const float4*) (res + (int64_t) r * H + i * 4);
                a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w;
            }
        }
        *(float4*) (out + (int64_t) t * H + i * 4) = a;
    }
}

// ds41_rope_rows: rotate the last 64 dims (32 adjacent pairs) of every (row, head) 512-wide vector of fp16 x in place, by the table entry of
// position pos[row] (fc fp32 [max_pos, 32, 2] = cos, sin). inverse conjugates. The reference's complex product in fp32, no fma.
// One thread per (row, head, pair).
extern "C" __global__ __launch_bounds__(256) void ds41_rope_rows(
    uint16_t* __restrict__ x, const float* __restrict__ fc, const int* __restrict__ pos, long total, int heads, int inverse)
{
    const long g = (long) blockIdx.x * 256 + threadIdx.x;
    if (g >= total) return;
    const int pr = (int) (g & 31);
    const long rh = g >> 5;
    const long r = rh / heads;
    uint16_t* p = x + rh * HD + 448 + 2 * pr;
    const float* t = fc + ((int64_t) pos[r] * 32 + pr) * 2;
    const float c = t[0];
    const float s = inverse ? -t[1] : t[1];
    const float a = __half2float(__ushort_as_half(p[0])), b = __half2float(__ushort_as_half(p[1]));
    const float re = a * c - b * s, im = a * s + b * c;
    p[0] = __half_as_ushort(__float2half_rn(re));
    p[1] = __half_as_ushort(__float2half_rn(im));
}
