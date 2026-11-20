// Decode MoE (grouped, q8_1 activations) kernels for GGUF k-quants, gfx1151 device-only code object (build: hip_kq/build_moe.sh).
// Same arithmetic and accumulation order as fq_moe_kq.cu (fq_moe_kq_upgate_kernel<8> / fq_moe_kq_upgate_v2_kernel / fq_moe_kq_down_v2_kernel<0>):
// outputs are bit-equal. Kernels with the suffix _ref are verbatim copies kept for the A/B; _v3 are the rewrites:
//   * K / I are template parameters (NB = K / 256): every weight load of a thread is issued before its first dot (full unroll),
//   * the q8_1 activation row of the slot is staged once in LDS (the reference re-reads it from global per item),
//   * Q3_K (110-byte blocks, 2-byte aligned): one thread per 256-block, whole block loaded as 28 dwords (alignbyte shift for odd blocks),
//     the 16 scale values are decoded once per block, gate and up share the activation reads.
// Assignment s = row * topk + j reads activation row s / topk; perm (optional) is the expert-major slot order of the sort kernel.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define Q8_1_BYTES 36
#define Q4_K_BYTES 144
#define Q3_K_BYTES 110
#define OUT_PER_BLOCK 8
#define LANES 32
#define ROWS 32
#define PS 33

__device__ __forceinline__ int dp4a_(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }
__device__ __forceinline__ float h2f_lo(uint32_t w) { uint16_t lo = (uint16_t)(w & 0xFFFFu); __half h; __builtin_memcpy(&h, &lo, 2); return __half2float(h); }
__device__ __forceinline__ uint32_t vsub4_(uint32_t a, uint32_t b)
{
    uint32_t v = 0;
    #pragma unroll
    for (int m = 0; m < 4; m++) v |= (((((a >> (8 * m)) & 0xFFu) - ((b >> (8 * m)) & 0xFFu)) & 0xFFu) << (8 * m));
    return v;
}

__device__ __forceinline__ void slot_tile(int b, int ntiles, int S, const int32_t* __restrict__ perm, int nslots, int& s, int& t)
{
    if (perm != nullptr) { const int p = b % nslots; t = b / nslots; s = (p < S) ? (int)perm[p] : p; }
    else { s = b / ntiles; t = b % ntiles; }
}

extern "C" {

// ---- sort: perm[rank of assignment i by (expert id, i)] = i (verbatim)
__global__ void moe_sort(const int64_t* __restrict__ sel, int32_t* __restrict__ perm, int S, int E)
{
    const int i = threadIdx.x;
    if (i < S)
    {
        int64_t k = sel[i]; k = (k < 0 || k >= E) ? 0 : k;
        int rank = 0;
        for (int j = 0; j < S; j++) { int64_t kj = sel[j]; kj = (kj < 0 || kj >= E) ? 0 : kj; rank += (kj < k) || (kj == k && j < i); }
        perm[rank] = i;
    }
}

// ---- q8_1 quantisation of fp16 rows (verbatim fq_quant_q8_1_kernel<__half>): grid M * K/32, 32 threads
__global__ void moe_quant_q8_1(const __half* __restrict__ x, uint8_t* __restrict__ qb, int M, int K)
{
    const int blk = blockIdx.x, nb32 = K / 32, m = blk / nb32, b = blk % nb32, tid = threadIdx.x;
    if (m >= M) return;
    const __half* xr = x + (int64_t)m * K + (int64_t)b * 32;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(__half2float(xr[i])));
    float d = amax / 127.0f;
    float v = __half2float(xr[tid]);
    int q = (amax == 0.0f) ? 0 : (int)roundf(v / d);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    uint8_t* qbblk = qb + ((int64_t)m * nb32 + b) * Q8_1_BYTES;
    qbblk[4 + tid] = (uint8_t)(int8_t)q;
    if (tid == 0)
    {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) sum += __half2float(xr[i]);
        __half dh = __float2half_rn(d), sh = __float2half_rn(sum);
        uint16_t dhb, shb;
        __builtin_memcpy(&dhb, &dh, 2); __builtin_memcpy(&shb, &sh, 2);
        *reinterpret_cast<uint16_t*>(qbblk) = dhb;
        *reinterpret_cast<uint16_t*>(qbblk + 2) = shb;
    }
}

// ---- combine (verbatim, no shared expert): out[r] = sum_s partial[r*S+s] * wts[r*S+s]
__global__ void moe_combine(const float* __restrict__ partial, const __half* __restrict__ wts, float* __restrict__ out, int S, int K)
{
    partial += (int64_t)blockIdx.y * S * K; wts += (int64_t)blockIdx.y * S; out += (int64_t)blockIdx.y * K;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < K; k += gridDim.x * blockDim.x)
    {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 32; s++) if (s < S) acc = __fadd_rn(acc, __fmul_rn(partial[(int64_t)s * K + k], __half2float(wts[s])));
        out[k] = acc;
    }
}

}  // extern "C"

// ================================================================== Q3_K v1 dot (verbatim fq_dot_q3_k_q8) and reference gate/up
__device__ float dot_q3_k_ref(const uint8_t* __restrict__ wr, const uint8_t* __restrict__ aqb, int K, int tid, int nthreads)
{
    int nsub = K / 16; float acc = 0.0f;
    for (int c = tid; c < nsub; c += nthreads)
    {
        int b = c >> 4, iqs = c & 15;
        const uint8_t* blk = wr + (size_t)b * Q3_K_BYTES;
        int bq8_offset = 4 * (iqs / 8), scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        int32_t vl, vm;
        __builtin_memcpy(&vl, blk + 32 + 4 * iqs, 4); __builtin_memcpy(&vm, blk + 4 * (iqs % 8), 4);
        int32_t vh = ~vm >> bq8_offset;
        const uint8_t* scales = blk + 96;
        __half dh; __builtin_memcpy(&dh, blk + 108, 2);
        float d = __half2float(dh), sumf = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int isc = scale_offset + 2 * i;
            int sc_low = (scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
            int sc_high = ((scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
            int sc = (sc_low | sc_high) - 32;
            uint32_t vil = ((uint32_t)vl >> (2 * i)) & 0x03030303u;
            uint32_t vih = (uint32_t)((vh >> i) << 2) & 0x04040404u;
            uint32_t vi = vsub4_(vil, vih);
            const uint8_t* aqb_blk = aqb + (size_t)(b * 8 + bq8_offset + i) * Q8_1_BYTES;
            __half dah; __builtin_memcpy(&dah, aqb_blk, 2);
            float d8 = __half2float(dah);
            int32_t u; __builtin_memcpy(&u, aqb_blk + 4 + 4 * (iqs % 8), 4);
            sumf = __fadd_rn(sumf, __fmul_rn(d8, (float)(dp4a_((int32_t)vi, u, 0) * sc)));
        }
        acc = __fadd_rn(acc, __fmul_rn(d, sumf));
    }
    return acc;
}

extern "C" {

// reference: fq_moe_kq_upgate_kernel<8> (256 threads, 8 rows per block, 32 lanes per row), dynamic LDS 256 * 2 * 4 bytes
__global__ void ug_q3_ref(const uint8_t* __restrict__ aqb, const int64_t* __restrict__ sel, const uint8_t* const* __restrict__ gate_bases,
                          const uint8_t* const* __restrict__ up_bases, int64_t gate_stride, int64_t up_stride, int64_t qb_stride,
                          __half* __restrict__ act, int S, int I, int K, int E, int ntiles_u, int topk, float lim,
                          const int32_t* __restrict__ perm, int nslots)
{
    int s, t; slot_tile(blockIdx.x, ntiles_u, S, perm, nslots, s, t);
    int grp = threadIdx.x / LANES, lane = threadIdx.x % LANES;
    int64_t e = sel[s]; e = (e < 0 || e >= E) ? 0 : e;
    int n = t * OUT_PER_BLOCK + grp;
    const uint8_t* aq = aqb + (int64_t)(s / topk) * qb_stride;
    extern __shared__ float red[];
    float g = 0.0f, u = 0.0f;
    if (n < I)
    {
        g = dot_q3_k_ref(gate_bases[e] + (int64_t)n * gate_stride, aq, K, lane, LANES);
        u = dot_q3_k_ref(up_bases[e] + (int64_t)n * up_stride, aq, K, lane, LANES);
    }
    red[2 * threadIdx.x] = g; red[2 * threadIdx.x + 1] = u;
    __syncthreads();
    if (n < I && lane == 0)
    {
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++) { sg = __fadd_rn(sg, red[(grp * 32 + l) * 2]); su = __fadd_rn(su, red[(grp * 32 + l) * 2 + 1]); }
        float gh = __half2float(__float2half_rn(sg)), uh = __half2float(__float2half_rn(su));
        if (lim > 0.0f) { gh = fminf(gh, lim); uh = fminf(fmaxf(uh, -lim), lim); }
        float sig = 1.0f / (1.0f + expf(-gh));
        act[(int64_t)s * I + n] = __float2half_rn(sig * gh * uh);
    }
}

}  // extern "C"

// ================================================================== Q4_K items
struct Q4It { uint32_t hd, s0, s1, s2; uint4 va, vb; };

__device__ __forceinline__ void q4_load(const uint8_t* __restrict__ wr, int it, Q4It& r)
{
    const int b = it >> 2, g = it & 3;
    const uint8_t* blk = wr + (size_t)b * Q4_K_BYTES;
    r.hd = *reinterpret_cast<const uint32_t*>(blk);
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(blk + 4);
    r.s0 = sp[0]; r.s1 = sp[1]; r.s2 = sp[2];
    r.va = *reinterpret_cast<const uint4*>(blk + 16 + 32 * g);
    r.vb = *reinterpret_cast<const uint4*>(blk + 16 + 32 * g + 16);
}

// verbatim fq_kq_item_q4_k arithmetic; the q8_1 row comes from LDS words aql (9 words per 32-block)
__device__ __forceinline__ void q4_comp(const Q4It& r, int it, const uint32_t* __restrict__ aql, float* __restrict__ dst)
{
    const int b = it >> 2, g = it & 3;
    const uint32_t* a0 = aql + (b * 8 + 2 * g) * 9;
    const uint32_t* a1 = a0 + 9;
    const float dm0 = h2f_lo(r.hd), dm1 = h2f_lo(r.hd >> 16);
    const float d8[2] = {h2f_lo(a0[0]), h2f_lo(a1[0])};
    const int sh = 16 * (g & 1);
    const uint32_t s0 = (r.s0 >> sh) & 0xFFFFu, s2 = (r.s1 >> sh) & 0xFFFFu, s4 = (r.s2 >> sh) & 0xFFFFu;
    const uint32_t hi = (g >= 2) ? 0xFFFFFFFFu : 0u;
    const uint16_t aux0 = (uint16_t)(((s0 & 0x3F3Fu) & ~hi) | ((((s4 >> 0) & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2)) & hi));
    const uint16_t aux1 = (uint16_t)(((s2 & 0x3F3Fu) & ~hi) | ((((s4 >> 4) & 0x0F0Fu) | ((s2 & 0xC0C0u) >> 2)) & hi));
    const int sc[2] = {(int)(aux0 & 0xFFu), (int)((aux0 >> 8) & 0xFFu)};
    const int mn[2] = {(int)(aux1 & 0xFFu), (int)((aux1 >> 8) & 0xFFu)};
    const uint32_t w0[4] = {r.va.x, r.va.y, r.va.z, r.va.w};
    const uint32_t w1[4] = {r.vb.x, r.vb.y, r.vb.z, r.vb.w};
    #pragma unroll
    for (int qo = 0; qo < 4; qo++)
    {
        float sumf_d = 0.0f, sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            const uint32_t* a = i ? a1 : a0;
            const uint32_t v0i = (w0[qo] >> (4 * i)) & 0x0F0F0F0Fu;
            const uint32_t v1i = (w1[qo] >> (4 * i)) & 0x0F0F0F0Fu;
            const int32_t u0 = (int32_t)a[1 + qo], u1 = (int32_t)a[5 + qo];
            int dot1 = dp4a_((int32_t)v0i, u0, 0);
            dot1 = dp4a_((int32_t)v1i, u1, dot1);
            int dot2 = dp4a_(0x01010101, u0, 0);
            dot2 = dp4a_(0x01010101, u1, dot2);
            sumf_d = __fadd_rn(sumf_d, __fmul_rn(d8[i], (float)(dot1 * sc[i])));
            sumf_m = __fadd_rn(sumf_m, __fmul_rn(d8[i], (float)(dot2 * mn[i])));
        }
        dst[qo] = __fadd_rn(__fmul_rn(dm0, sumf_d), -__fmul_rn(dm1, sumf_m));
    }
}

template <int NB>
__device__ __forceinline__ void stage_act(uint32_t* __restrict__ aql, const uint8_t* __restrict__ aq)
{
    const uint32_t* src = reinterpret_cast<const uint32_t*>(aq);
    for (int i = threadIdx.x; i < NB * 8 * 9; i += blockDim.x) aql[i] = src[i];
    __syncthreads();
}

// Q4_K gate+up, v3: 32 rows per block, 8 threads per row, items j, j + 8, ... all loaded before the first dot. grid Sg * (I / 32) x 256
template <int NB>
__device__ __forceinline__ void ug_q4_v3_body(const uint8_t* __restrict__ aqb, const int64_t* __restrict__ sel,
        const uint8_t* const* __restrict__ gate_bases, const uint8_t* const* __restrict__ up_bases, int64_t gate_stride, int64_t up_stride,
        int64_t qb_stride, __half* __restrict__ act, int S, int I, int E, int ntiles_u, int topk, float lim,
        const int32_t* __restrict__ perm, int nslots)
{
    constexpr int NIT = NB * 4, PER = (NIT + 7) / 8;
    __shared__ float pg[ROWS * PS];
    __shared__ float pu[ROWS * PS];
    __shared__ uint32_t aql[NB * 8 * 9];
    int s, t; slot_tile(blockIdx.x, ntiles_u, S, perm, nslots, s, t);
    const int r = threadIdx.x >> 3, j = threadIdx.x & 7;
    int64_t e = sel[s]; e = (e < 0 || e >= E) ? 0 : e;
    const int n = t * ROWS + r;
    const uint8_t* gr = gate_bases[e] + (int64_t)n * gate_stride;
    const uint8_t* ur = up_bases[e] + (int64_t)n * up_stride;
    Q4It G[PER], U[PER];
    #pragma unroll
    for (int p = 0; p < PER; p++) if (NIT % 8 == 0 || j + 8 * p < NIT) { q4_load(gr, j + 8 * p, G[p]); q4_load(ur, j + 8 * p, U[p]); }
    stage_act<NB>(aql, aqb + (int64_t)(s / topk) * qb_stride);
    float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int p = 0; p < PER; p++)
        if (NIT % 8 == 0 || j + 8 * p < NIT)
        {
            float tg[4], tu[4];
            q4_comp(G[p], j + 8 * p, aql, tg);
            q4_comp(U[p], j + 8 * p, aql, tu);
            #pragma unroll
            for (int q = 0; q < 4; q++) { ag[q] = __fadd_rn(ag[q], tg[q]); au[q] = __fadd_rn(au[q], tu[q]); }
        }
    #pragma unroll
    for (int q = 0; q < 4; q++) { pg[r * PS + 4 * j + q] = ag[q]; pu[r * PS + 4 * j + q] = au[q]; }
    __syncthreads();
    if (j == 0)
    {
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++) { sg = __fadd_rn(sg, pg[r * PS + l]); su = __fadd_rn(su, pu[r * PS + l]); }
        float gh = __half2float(__float2half_rn(sg)), uh = __half2float(__float2half_rn(su));
        if (lim > 0.0f) { gh = fminf(gh, lim); uh = fminf(fmaxf(uh, -lim), lim); }
        float sig = 1.0f / (1.0f + expf(-gh));
        act[(int64_t)s * I + n] = __float2half_rn(sig * gh * uh);
    }
}

extern "C" {
#define UG_Q4(NBV) \
__global__ void __launch_bounds__(256) ug_q4_v3_##NBV(const uint8_t* __restrict__ aqb, const int64_t* __restrict__ sel, \
        const uint8_t* const* __restrict__ gb, const uint8_t* const* __restrict__ ub, int64_t gs, int64_t us, int64_t qs, \
        __half* __restrict__ act, int S, int I, int E, int ntiles, int topk, float lim, const int32_t* __restrict__ perm, int nslots) \
{ ug_q4_v3_body<NBV>(aqb, sel, gb, ub, gs, us, qs, act, S, I, E, ntiles, topk, lim, perm, nslots); }
UG_Q4(6)
UG_Q4(8)
UG_Q4(16)

}  // extern "C"

// Q4_K down, v3 (act row of slot s from the q8_1 act scratch): partial[s, n] = sum over the 32 lane partials in order. grid S * (K / 32) x 256
template <int NB>
__device__ __forceinline__ void dn_q4_v3_body(const uint8_t* __restrict__ aqb, int64_t qb_stride, const int64_t* __restrict__ sel,
        const uint8_t* const* __restrict__ down_bases, int64_t down_stride, float* __restrict__ partial, int S, int K, int E, int ntiles_d,
        const int32_t* __restrict__ perm, int nslots)
{
    constexpr int NIT = NB * 4, PER = (NIT + 7) / 8;
    __shared__ float pp[ROWS * PS];
    __shared__ uint32_t aql[NB * 8 * 9];
    int s, t; slot_tile(blockIdx.x, ntiles_d, S, perm, nslots, s, t);
    const int r = threadIdx.x >> 3, j = threadIdx.x & 7;
    int64_t e = sel[s]; e = (e < 0 || e >= E) ? 0 : e;
    const int n = t * ROWS + r;
    const uint8_t* dr = down_bases[e] + (int64_t)n * down_stride;
    Q4It D[PER];
    #pragma unroll
    for (int p = 0; p < PER; p++) if (NIT % 8 == 0 || j + 8 * p < NIT) q4_load(dr, j + 8 * p, D[p]);
    stage_act<NB>(aql, aqb + (int64_t)s * qb_stride);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int p = 0; p < PER; p++)
        if (NIT % 8 == 0 || j + 8 * p < NIT)
        {
            float tv[4];
            q4_comp(D[p], j + 8 * p, aql, tv);
            #pragma unroll
            for (int q = 0; q < 4; q++) acc[q] = __fadd_rn(acc[q], tv[q]);
        }
    #pragma unroll
    for (int q = 0; q < 4; q++) pp[r * PS + 4 * j + q] = acc[q];
    __syncthreads();
    if (j == 0)
    {
        float sp = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++) sp = __fadd_rn(sp, pp[r * PS + l]);
        partial[(int64_t)s * K + n] = sp;
    }
}

extern "C" {
#define DN_Q4(NBV) \
__global__ void __launch_bounds__(256) dn_q4_v3_##NBV(const uint8_t* __restrict__ aqb, int64_t qs, const int64_t* __restrict__ sel, \
        const uint8_t* const* __restrict__ db, int64_t ds, float* __restrict__ partial, int S, int K, int E, int ntiles, \
        const int32_t* __restrict__ perm, int nslots) \
{ dn_q4_v3_body<NBV>(aqb, qs, sel, db, ds, partial, S, K, E, ntiles, perm, nslots); }
DN_Q4(6)
DN_Q4(8)
DN_Q4(16)

}  // extern "C"

// ================================================================== Q3_K gate+up v3: one thread per (row, 256-block), NB blocks per row
struct Q3B { uint32_t w[28]; };

__device__ __forceinline__ void q3_load(const uint8_t* __restrict__ blk, Q3B& r)
{
    const uintptr_t a = (uintptr_t)blk;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(a & ~(uintptr_t)3);
    uint32_t t[28];
    #pragma unroll
    for (int k = 0; k < 28; k++) t[k] = p[k];
    const bool odd = (a & 2) != 0;
    #pragma unroll
    for (int k = 0; k < 27; k++) r.w[k] = odd ? __builtin_amdgcn_alignbyte(t[k + 1], t[k], 2) : t[k];
    r.w[27] = odd ? (t[27] >> 16) : t[27];
}

__device__ __forceinline__ void q3_scales(const Q3B& B, int* __restrict__ sc)
{
    #pragma unroll
    for (int k = 0; k < 16; k++)
    {
        const int i0 = k % 8, i1 = 8 + k % 4;
        const int s0 = (B.w[24 + (i0 >> 2)] >> (8 * (i0 & 3))) & 0xFF;
        const int s1 = (B.w[24 + (i1 >> 2)] >> (8 * (i1 & 3))) & 0xFF;
        sc[k] = (((s0 >> (4 * (k / 8))) & 0xF) | (((s1 >> (2 * (k / 4))) & 3) << 4)) - 32;
    }
}

template <int NB>
__device__ __forceinline__ void ug_q3_v3_body(const uint8_t* __restrict__ aqb, const int64_t* __restrict__ sel,
        const uint8_t* const* __restrict__ gate_bases, const uint8_t* const* __restrict__ up_bases, int64_t gate_stride, int64_t up_stride,
        int64_t qb_stride, __half* __restrict__ act, int S, int I, int E, int ntiles_u, int topk, float lim,
        const int32_t* __restrict__ perm, int nslots)
{
    constexpr int RS = NB * 16 + 1;
    extern __shared__ float lds[];                       // pg[ROWS * RS], pu[ROWS * RS], aql[NB * 72]
    float* pg = lds; float* pu = lds + ROWS * RS;
    float* dgs = lds + 2 * ROWS * RS; float* dus = dgs + ROWS * NB;          // block scales d (the lane accumulation is fma(d, sumf, acc), as the compiler fuses it in the reference)
    uint32_t* aql = reinterpret_cast<uint32_t*>(lds + 2 * ROWS * RS + 2 * ROWS * NB);
    int s, t; slot_tile(blockIdx.x, ntiles_u, S, perm, nslots, s, t);
    const int r = threadIdx.x / NB, b = threadIdx.x % NB;
    int64_t e = sel[s]; e = (e < 0 || e >= E) ? 0 : e;
    const int n = t * ROWS + r;
    Q3B G, U;
    q3_load(gate_bases[e] + (int64_t)n * gate_stride + (size_t)b * Q3_K_BYTES, G);
    q3_load(up_bases[e] + (int64_t)n * up_stride + (size_t)b * Q3_K_BYTES, U);
    {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(aqb + (int64_t)(s / topk) * qb_stride);
        for (int i = threadIdx.x; i < NB * 72; i += blockDim.x) aql[i] = src[i];
        __syncthreads();
    }
    int scg[16], scu[16];
    q3_scales(G, scg); q3_scales(U, scu);
    const float dg = h2f_lo(G.w[27]), du = h2f_lo(U.w[27]);
    float d8[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) d8[k] = h2f_lo(aql[(b * 8 + k) * 9]);
    dgs[r * NB + b] = dg; dus[r * NB + b] = du;
    #pragma unroll
    for (int iqs = 0; iqs < 16; iqs++)
    {
        const int bq8 = 4 * (iqs / 8), so = (iqs & 8) + ((iqs & 7) >> 2);
        const int32_t vlg = (int32_t)G.w[8 + iqs], vmg = (int32_t)G.w[iqs & 7];
        const int32_t vlu = (int32_t)U.w[8 + iqs], vmu = (int32_t)U.w[iqs & 7];
        const int32_t vhg = ~vmg >> bq8, vhu = ~vmu >> bq8;
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            const int32_t u = (int32_t)aql[(b * 8 + bq8 + i) * 9 + 1 + (iqs & 7)];
            const float d8i = d8[bq8 + i];
            const uint32_t vilg = ((uint32_t)vlg >> (2 * i)) & 0x03030303u, vihg = (uint32_t)((vhg >> i) << 2) & 0x04040404u;
            const uint32_t vilu = ((uint32_t)vlu >> (2 * i)) & 0x03030303u, vihu = (uint32_t)((vhu >> i) << 2) & 0x04040404u;
            // bytes of vil are 0..3 and of vih 0 / 4: a per-byte wrapping subtraction without borrow
            const uint32_t vig = ((vilg | 0x80808080u) - vihg) ^ 0x80808080u;
            const uint32_t viu = ((vilu | 0x80808080u) - vihu) ^ 0x80808080u;
            sg = __fadd_rn(sg, __fmul_rn(d8i, (float)(dp4a_((int32_t)vig, u, 0) * scg[so + 2 * i])));
            su = __fadd_rn(su, __fmul_rn(d8i, (float)(dp4a_((int32_t)viu, u, 0) * scu[so + 2 * i])));
        }
        pg[r * RS + b * 16 + iqs] = sg;
        pu[r * RS + b * 16 + iqs] = su;
    }
    __syncthreads();
    if (b == 0)
    {
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
        {
            const int h = l >> 4, iq = l & 15;
            float ag = 0.0f, au = 0.0f;
            #pragma unroll
            for (int k = 0; 2 * k + h < NB; k++) { ag = fmaf(dgs[r * NB + 2 * k + h], pg[r * RS + (2 * k + h) * 16 + iq], ag); au = fmaf(dus[r * NB + 2 * k + h], pu[r * RS + (2 * k + h) * 16 + iq], au); }
            sg = __fadd_rn(sg, ag); su = __fadd_rn(su, au);
        }
        float gh = __half2float(__float2half_rn(sg)), uh = __half2float(__float2half_rn(su));
        if (lim > 0.0f) { gh = fminf(gh, lim); uh = fminf(fmaxf(uh, -lim), lim); }
        float sig = 1.0f / (1.0f + expf(-gh));
        act[(int64_t)s * I + n] = __float2half_rn(sig * gh * uh);
    }
}

extern "C" {
#define UG_Q3(NBV) \
__global__ void __launch_bounds__(ROWS * NBV) ug_q3_v3_##NBV(const uint8_t* __restrict__ aqb, const int64_t* __restrict__ sel, \
        const uint8_t* const* __restrict__ gb, const uint8_t* const* __restrict__ ub, int64_t gs, int64_t us, int64_t qs, \
        __half* __restrict__ act, int S, int I, int E, int ntiles, int topk, float lim, const int32_t* __restrict__ perm, int nslots) \
{ ug_q3_v3_body<NBV>(aqb, sel, gb, ub, gs, us, qs, act, S, I, E, ntiles, topk, lim, perm, nslots); }
UG_Q3(6)
UG_Q3(8)
}  // extern "C"

// ================================================================== references copied from fq_moe_kq.cu (v2 Q4_K gate/up and down, 8 threads per row)
__device__ __forceinline__ void q4_item_ref(const uint8_t* __restrict__ wr, const uint8_t* __restrict__ aq, int it, float* __restrict__ dst)
{
    Q4It r; q4_load(wr, it, r);
    const int b = it >> 2, g = it & 3;
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(aq + (size_t)(b * 8 + 2 * g) * Q8_1_BYTES);
    // same arithmetic as q4_comp, reading the row from global
    const uint32_t* a1 = a0 + 9;
    const float dm0 = h2f_lo(r.hd), dm1 = h2f_lo(r.hd >> 16);
    const float d8[2] = {h2f_lo(a0[0]), h2f_lo(a1[0])};
    const int sh = 16 * (g & 1);
    const uint32_t s0 = (r.s0 >> sh) & 0xFFFFu, s2 = (r.s1 >> sh) & 0xFFFFu, s4 = (r.s2 >> sh) & 0xFFFFu;
    const uint32_t hi = (g >= 2) ? 0xFFFFFFFFu : 0u;
    const uint16_t aux0 = (uint16_t)(((s0 & 0x3F3Fu) & ~hi) | ((((s4 >> 0) & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2)) & hi));
    const uint16_t aux1 = (uint16_t)(((s2 & 0x3F3Fu) & ~hi) | ((((s4 >> 4) & 0x0F0Fu) | ((s2 & 0xC0C0u) >> 2)) & hi));
    const int sc[2] = {(int)(aux0 & 0xFFu), (int)((aux0 >> 8) & 0xFFu)};
    const int mn[2] = {(int)(aux1 & 0xFFu), (int)((aux1 >> 8) & 0xFFu)};
    const uint32_t w0[4] = {r.va.x, r.va.y, r.va.z, r.va.w};
    const uint32_t w1[4] = {r.vb.x, r.vb.y, r.vb.z, r.vb.w};
    #pragma unroll
    for (int qo = 0; qo < 4; qo++)
    {
        float sumf_d = 0.0f, sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            const uint32_t* a = i ? a1 : a0;
            const uint32_t v0i = (w0[qo] >> (4 * i)) & 0x0F0F0F0Fu, v1i = (w1[qo] >> (4 * i)) & 0x0F0F0F0Fu;
            const int32_t u0 = (int32_t)a[1 + qo], u1 = (int32_t)a[5 + qo];
            int dot1 = dp4a_((int32_t)v0i, u0, 0);
            dot1 = dp4a_((int32_t)v1i, u1, dot1);
            int dot2 = dp4a_(0x01010101, u0, 0);
            dot2 = dp4a_(0x01010101, u1, dot2);
            sumf_d = __fadd_rn(sumf_d, __fmul_rn(d8[i], (float)(dot1 * sc[i])));
            sumf_m = __fadd_rn(sumf_m, __fmul_rn(d8[i], (float)(dot2 * mn[i])));
        }
        dst[qo] = __fadd_rn(__fmul_rn(dm0, sumf_d), -__fmul_rn(dm1, sumf_m));
    }
}

extern "C" {

__global__ void __launch_bounds__(256) ug_q4_ref(const uint8_t* __restrict__ aqb, const int64_t* __restrict__ sel,
        const uint8_t* const* __restrict__ gate_bases, const uint8_t* const* __restrict__ up_bases, int64_t gate_stride, int64_t up_stride,
        int64_t qb_stride, __half* __restrict__ act, int S, int I, int K, int E, int ntiles_u, int topk, float lim,
        const int32_t* __restrict__ perm, int nslots)
{
    __shared__ float pg[ROWS * PS];
    __shared__ float pu[ROWS * PS];
    int s, t; slot_tile(blockIdx.x, ntiles_u, S, perm, nslots, s, t);
    const int r = threadIdx.x >> 3, j = threadIdx.x & 7;
    int64_t e = sel[s]; e = (e < 0 || e >= E) ? 0 : e;
    const int n = t * ROWS + r;
    const uint8_t* aq = aqb + (int64_t)(s / topk) * qb_stride;
    float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (n < I)
    {
        const uint8_t* gr = gate_bases[e] + (int64_t)n * gate_stride;
        const uint8_t* ur = up_bases[e] + (int64_t)n * up_stride;
        const int nitems = (K / 256) * 4;
        for (int it = j; it < nitems; it += 8)
        {
            float tg[4], tu[4];
            q4_item_ref(gr, aq, it, tg); q4_item_ref(ur, aq, it, tu);
            #pragma unroll
            for (int q = 0; q < 4; q++) { ag[q] = __fadd_rn(ag[q], tg[q]); au[q] = __fadd_rn(au[q], tu[q]); }
        }
    }
    #pragma unroll
    for (int q = 0; q < 4; q++) { pg[r * PS + 4 * j + q] = ag[q]; pu[r * PS + 4 * j + q] = au[q]; }
    __syncthreads();
    if (n < I && j == 0)
    {
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++) { sg = __fadd_rn(sg, pg[r * PS + l]); su = __fadd_rn(su, pu[r * PS + l]); }
        float gh = __half2float(__float2half_rn(sg)), uh = __half2float(__float2half_rn(su));
        if (lim > 0.0f) { gh = fminf(gh, lim); uh = fminf(fmaxf(uh, -lim), lim); }
        float sig = 1.0f / (1.0f + expf(-gh));
        act[(int64_t)s * I + n] = __float2half_rn(sig * gh * uh);
    }
}

__global__ void __launch_bounds__(256) dn_q4_ref(const uint8_t* __restrict__ aqb, int64_t qb_stride, const int64_t* __restrict__ sel,
        const uint8_t* const* __restrict__ down_bases, int64_t down_stride, float* __restrict__ partial, int S, int I, int K, int E, int ntiles_d,
        const int32_t* __restrict__ perm, int nslots)
{
    __shared__ float pp[ROWS * PS];
    int s, t; slot_tile(blockIdx.x, ntiles_d, S, perm, nslots, s, t);
    const int r = threadIdx.x >> 3, j = threadIdx.x & 7;
    int64_t e = sel[s]; e = (e < 0 || e >= E) ? 0 : e;
    const int n = t * ROWS + r;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (n < K)
    {
        const uint8_t* dr = down_bases[e] + (int64_t)n * down_stride;
        const uint8_t* aq = aqb + (int64_t)s * qb_stride;
        const int nitems = (I / 256) * 4;
        for (int it = j; it < nitems; it += 8)
        {
            float tv[4];
            q4_item_ref(dr, aq, it, tv);
            #pragma unroll
            for (int q = 0; q < 4; q++) acc[q] = __fadd_rn(acc[q], tv[q]);
        }
    }
    #pragma unroll
    for (int q = 0; q < 4; q++) pp[r * PS + 4 * j + q] = acc[q];
    __syncthreads();
    if (n < K && j == 0)
    {
        float sp = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++) sp = __fadd_rn(sp, pp[r * PS + l]);
        partial[(int64_t)s * K + n] = sp;
    }
}

}  // extern "C"
