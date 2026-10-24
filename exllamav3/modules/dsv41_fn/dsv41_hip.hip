// DeepSeek-V4.1-Flash small-op kernels for gfx1151 (wave32). One code object, loaded by dsv41_hip.py.
//
// Every kernel replaces a chain of torch ops of the CPU-exact reference (exllamav3/dsv41/{ops,quant,engram}.py) with one launch.
// The number formats (fp8/fp4 fake-quant, e4m3 rounding, MXFP4, Engram hash) come from dsv41_num.h, which is bit-identical to the
// reference on CPU (tests/dsv41/test_dsv41_num.py). Reductions (rms, dot products) use a fixed wave/block tree that differs from
// torch's, so kernel-vs-reference results are within fp32 rounding before the fp16 store, not bit-exact (tests/dsv41/test_dsv41_hip.py
// states the tolerance per kernel). Built with -ffp-contract=off: no fused multiply-add changes the rounding of the formulas.
//
// Conventions: activations fp16 unless said; hyper-connection streams fp32 [R, 4, 5120]; all small weights fp32 (cast once at load);
// rope tables are the reference's complex table as interleaved fp32 (cos, sin) [max_pos, 32, 2] (torch.polar, so identical entries).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "dsv41_num.h"

#define WARP 32
#define HD 512                  // attention head dim / compressed latent width
#define RD 64                   // rope dims (last 64 of the 512)
#define HCM 4                   // hc_mult
#define DIM 5120                // hidden size

static __device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }
static __device__ __forceinline__ uint16_t f2h(float v) { return __half_as_ushort(__float2half_rn(v)); }
static __device__ __forceinline__ float wsum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, WARP); return v; }
static __device__ __forceinline__ float wmax(float v) { for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WARP)); return v; }
static __device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
static __device__ __forceinline__ float rsqrt_ieee(float x) { return 1.0f / sqrtf(x); }

// rotate the adjacent pair (a, b) by the table entry (c, s): the reference's complex product (a + bi)(c + si), fp32, no fma
static __device__ __forceinline__ void rot(float& a, float& b, float c, float s, bool inv)
{
    if (inv) s = -s;
    float re = a * c - b * s, im = a * s + b * c;
    a = re; b = im;
}

// -------------------------------------------------------------------------------------------------------------------
// ds41_kv_prep: wkv output -> kv_norm -> rope(last 64) -> fp8 fake-quant (ue8m0 / 32) -> window ring slot + out row.
// One wave per row, 16 contiguous elements per lane (a 32-group spans lane pairs). kv [R, 512] fp16, w [512] fp32,
// fc [max_pos, 32, 2], pos [R] int32 absolute positions, slot [R] int32 ring slot (-1 = none), out [R, 512] fp16 (may be null),
// ring [win, 512] fp16 (may be null).
extern "C" __global__ __launch_bounds__(WARP)
void ds41_kv_prep(const uint16_t* __restrict__ kv, const float* __restrict__ w, const float* __restrict__ fc,
                  const int* __restrict__ pos, const int* __restrict__ slot, uint16_t* __restrict__ out,
                  uint16_t* __restrict__ ring, float eps)
{
    const int r = blockIdx.x, lane = threadIdx.x;
    float v[16];
    const uint16_t* src = kv + (size_t) r * HD + lane * 16;
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) { v[i] = h2f(src[i]); ss += v[i] * v[i]; }
    ss = wsum(ss);
    const float rstd = rsqrt_ieee(ss / (float) HD + eps);
    #pragma unroll
    for (int i = 0; i < 16; i++) v[i] = h2f(f2h(w[lane * 16 + i] * (v[i] * rstd)));
    if (lane >= (HD - RD) / 16)                               // lanes 28..31 hold the 64 rope dims
    {
        const float* t = fc + (size_t) pos[r] * 64;
        #pragma unroll
        for (int i = 0; i < 16; i += 2)
        {
            const int pr = (lane * 16 + i - (HD - RD)) / 2;
            rot(v[i], v[i + 1], t[2 * pr], t[2 * pr + 1], false);
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) v[i] = h2f(f2h(v[i]));
    }
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor(amax, 1, WARP));
    const float s = ds_fp8_scale(amax);
    uint16_t o[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) o[i] = f2h(ds_fp8_q(v[i], s));
    if (out) { uint16_t* d = out + (size_t) r * HD + lane * 16; for (int i = 0; i < 16; i++) d[i] = o[i]; }
    if (ring && slot[r] >= 0) { uint16_t* d = ring + (size_t) slot[r] * HD + lane * 16; for (int i = 0; i < 16; i++) d[i] = o[i]; }
}

// ds41_rmsnorm_rows: y = fp16(w * (x * rsqrt(mean(x^2) + eps))), x fp16 [R, N], w fp32 [N], N % 32 == 0 (the reference's rmsnorm)
extern "C" __global__ __launch_bounds__(WARP)
void ds41_rmsnorm_rows(const uint16_t* __restrict__ x, const float* __restrict__ w, uint16_t* __restrict__ y, int N, float eps)
{
    const int r = blockIdx.x, lane = threadIdx.x;
    const uint16_t* p = x + (size_t) r * N;
    float ss = 0.0f;
    for (int i = lane; i < N; i += WARP) { float t = h2f(p[i]); ss += t * t; }
    ss = wsum(ss);
    const float rstd = rsqrt_ieee(ss / (float) N + eps);
    for (int i = lane; i < N; i += WARP) y[(size_t) r * N + i] = f2h(w[i] * (h2f(p[i]) * rstd));
}

// ds41_fq_rows: in-place fake-quant of fp16 [R, N] (N % group == 0). mode 0: fp8 e4m3 / ue8m0, group 32 (o before wo_a);
// mode 1: fp4 e2m1 / ue8m0, group 32; mode 2: fp4 e2m1 / e4m3 scale, group 16. One thread per group.
extern "C" __global__ __launch_bounds__(256)
void ds41_fq_rows(uint16_t* __restrict__ x, long n_groups, int mode)
{
    const long g = (long) blockIdx.x * 256 + threadIdx.x;
    if (g >= n_groups) return;
    const int gs = mode == 2 ? 16 : 32;
    uint16_t* p = x + g * gs;
    float v[32], amax = 0.0f;
    for (int i = 0; i < gs; i++) { v[i] = h2f(p[i]); amax = fmaxf(amax, fabsf(v[i])); }
    const float s = mode == 0 ? ds_fp8_scale(amax) : (mode == 1 ? ds_fp4_scale_ue8m0(amax) : ds_fp4_scale_e4m3(amax));
    for (int i = 0; i < gs; i++) p[i] = f2h(mode == 0 ? ds_fp8_q(v[i], s) : ds_fp4_q(v[i], s));
}

// ds41_fq_rows_v2: bit-identical to ds41_fq_rows, built for the prefill row counts (o [R, 32768] fp16 is 134 MB per layer at
// R = 2048; v1 ran at 4.4 ms = 61 GB/s: one thread per group, 2-byte strided loads, two IEEE divisions per element).
// One thread per 8 contiguous halves (one 16-byte load and store), a group of 32 (modes 0, 1) is 4 neighbouring lanes and a
// group of 16 (mode 2) is 2; the group amax is a __shfl_xor butterfly. The scale of modes 0 and 1 is a power of two (ue8m0), so
// x / s == x * (1 / s) exactly (the reciprocal is a power of two as well) and the division of the e4m3 grid step (a / step,
// step = 2^k) is the same multiplication: same real value, same single rounding, so the result is bit-identical.
// Mode 2 keeps the true division (its e4m3 scale is not a power of two). n_vec = numel / 8, x 16-byte aligned. Every lane of
// the wave takes part in the shuffles (lanes past n_vec carry amax 0 and write nothing).
static __device__ __forceinline__ float ds_inv_pow2(float s) { return ds_u2f((uint32_t) (254 - (int) ((ds_f2u(s) >> 23) & 0xFF)) << 23); }
static __device__ __forceinline__ float ds_e4m3_round_mul(float v)
{
    const float a = fabsf(v);
    float q;
    if (a < 0.015625f)
        q = rintf(a * 512.0f) * (1.0f / 512.0f);
    else
    {
        const int32_t e = (int32_t) ((ds_f2u(a) >> 23) & 0xFF) - 127;
        const float step = ds_u2f((uint32_t) (e - 3 + 127) << 23);
        const float rstep = ds_u2f((uint32_t) (127 - (e - 3)) << 23);
        q = rintf(a * rstep) * step;
    }
    return copysignf(q, v);
}
extern "C" __global__ __launch_bounds__(256)
void ds41_fq_rows_v2(uint16_t* __restrict__ x, long n_vec, int mode)
{
    const long t = (long) blockIdx.x * 256 + threadIdx.x;
    const bool ok = t < n_vec;
    uint32_t q[4] = {0, 0, 0, 0};
    if (ok) { const uint4 u = *reinterpret_cast<const uint4*>(x + t * 8); q[0] = u.x; q[1] = u.y; q[2] = u.z; q[3] = u.w; }
    float v[8], amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) { v[2 * i] = h2f(q[i] & 0xFFFF); v[2 * i + 1] = h2f(q[i] >> 16); }
    #pragma unroll
    for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor(amax, 1, WARP));
    if (mode != 2) amax = fmaxf(amax, __shfl_xor(amax, 2, WARP));
    float o[8];
    if (mode == 0)
    {
        const float s = ds_fp8_scale(amax), r = ds_inv_pow2(s);
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = ds_e4m3_round_mul(fminf(fmaxf(v[i] * r, -448.0f), 448.0f)) * s;
    }
    else if (mode == 1)
    {
        const float s = ds_fp4_scale_ue8m0(amax), r = ds_inv_pow2(s);
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = ds_fp4_round(fminf(fmaxf(v[i] * r, -6.0f), 6.0f)) * s;
    }
    else
    {
        const float s = ds_fp4_scale_e4m3(amax);
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = ds_fp4_q(v[i], s);
    }
    if (ok)
    {
        uint4 pk;
        pk.x = (uint32_t) f2h(o[0]) | ((uint32_t) f2h(o[1]) << 16); pk.y = (uint32_t) f2h(o[2]) | ((uint32_t) f2h(o[3]) << 16);
        pk.z = (uint32_t) f2h(o[4]) | ((uint32_t) f2h(o[5]) << 16); pk.w = (uint32_t) f2h(o[6]) | ((uint32_t) f2h(o[7]) << 16);
        *reinterpret_cast<uint4*>(x + t * 8) = pk;
    }
}

// ds41_idx_q_prep: indexer q [R, 32 * 128] fp16 in place: rope on dims 64..127 of every head, then fp4 / ue8m0 per 32.
// One thread per (row, head, 32-group).
extern "C" __global__ __launch_bounds__(256)
void ds41_idx_q_prep(uint16_t* __restrict__ q, const float* __restrict__ fc, const int* __restrict__ pos, int R, int heads)
{
    const long g = (long) blockIdx.x * 256 + threadIdx.x;
    const long total = (long) R * heads * 4;
    if (g >= total) return;
    const int r = (int) (g / (heads * 4));
    const int grp = (int) (g % 4);
    uint16_t* p = q + g * 32;
    float v[32], amax = 0.0f;
    for (int i = 0; i < 32; i++) v[i] = h2f(p[i]);
    if (grp >= 2)
    {
        const float* t = fc + (size_t) pos[r] * 64;
        for (int i = 0; i < 32; i += 2)
        {
            const int pr = ((grp - 2) * 32 + i) / 2;
            rot(v[i], v[i + 1], t[2 * pr], t[2 * pr + 1], false);
        }
        for (int i = 0; i < 32; i++) v[i] = h2f(f2h(v[i]));
    }
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
    const float s = ds_fp4_scale_ue8m0(amax);
    for (int i = 0; i < 32; i++) p[i] = f2h(ds_fp4_q(v[i], s));
}

// ds41_idx_k_store: indexer key = fp4_ue8m0(rope(rmsnorm(wk(latent)))) -> pool row dst[r]. k [R, 128] fp16, w [128] fp32,
// ck_pos [R] rope position (compressed index * ratio), dst [R] int32 pool row (-1 = skip), pool [rows, 128] fp16.
// One wave per row, 4 elements per lane; a 32-group is 8 lanes.
extern "C" __global__ __launch_bounds__(WARP)
void ds41_idx_k_store(const uint16_t* __restrict__ k, const float* __restrict__ w, const float* __restrict__ fc,
                      const int* __restrict__ ck_pos, const int* __restrict__ dst, uint16_t* __restrict__ pool, float eps)
{
    const int r = blockIdx.x, lane = threadIdx.x;
    if (dst[r] < 0) return;
    float v[4], ss = 0.0f;
    for (int i = 0; i < 4; i++) { v[i] = h2f(k[(size_t) r * 128 + lane * 4 + i]); ss += v[i] * v[i]; }
    ss = wsum(ss);
    const float rstd = rsqrt_ieee(ss / 128.0f + eps);
    for (int i = 0; i < 4; i++) v[i] = h2f(f2h(w[lane * 4 + i] * (v[i] * rstd)));
    if (lane >= 16)
    {
        const float* t = fc + (size_t) ck_pos[r] * 64;
        for (int i = 0; i < 4; i += 2)
        {
            const int pr = (lane * 4 + i - 64) / 2;
            rot(v[i], v[i + 1], t[2 * pr], t[2 * pr + 1], false);
        }
        for (int i = 0; i < 4; i++) v[i] = h2f(f2h(v[i]));
    }
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor(amax, o, WARP));
    const float s = ds_fp4_scale_ue8m0(amax);
    for (int i = 0; i < 4; i++) pool[(size_t) dst[r] * 128 + lane * 4 + i] = f2h(ds_fp4_q(v[i], s));
}

// ds41_pool_store: compressed latent (post compressor norm, pre rope) [R, 512] fp16 -> rope(last 64) -> fp4 / e4m3 scale per 16
// -> pool row dst[r] [rows, 512] fp16. One wave per row, 16 elements per lane (one fp4 group per lane).
extern "C" __global__ __launch_bounds__(WARP)
void ds41_pool_store(const uint16_t* __restrict__ lat, const float* __restrict__ fc, const int* __restrict__ ck_pos,
                     const int* __restrict__ dst, uint16_t* __restrict__ pool)
{
    const int r = blockIdx.x, lane = threadIdx.x;
    if (dst[r] < 0) return;
    float v[16];
    for (int i = 0; i < 16; i++) v[i] = h2f(lat[(size_t) r * HD + lane * 16 + i]);
    if (lane >= (HD - RD) / 16)
    {
        const float* t = fc + (size_t) ck_pos[r] * 64;
        for (int i = 0; i < 16; i += 2)
        {
            const int pr = (lane * 16 + i - (HD - RD)) / 2;
            rot(v[i], v[i + 1], t[2 * pr], t[2 * pr + 1], false);
        }
        for (int i = 0; i < 16; i++) v[i] = h2f(f2h(v[i]));
    }
    float amax = 0.0f;
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));
    const float s = ds_fp4_scale_e4m3(amax);
    for (int i = 0; i < 16; i++) pool[(size_t) dst[r] * HD + lane * 16 + i] = f2h(ds_fp4_q(v[i], s));
}

// ds41_compress: softmax-pool `ratio` rows (ratio <= 8) per completed group -> rmsnorm. kv, sc fp32 [R, 512] (wkv / wgate outputs of
// the new rows at absolute positions start .. start + R - 1); kst, sst fp32 [ratio, 512] hold the rows of the previous incomplete
// group (slot = position % ratio). Output group g (g = start / ratio + blockIdx.x) -> out [n_groups, 512] fp16 = fp16(rmsnorm(fp16(
// sum_j kv_j * softmax_j(sc_j)))). The state of the new trailing incomplete group is written by ds41_compress_state (separate
// launch: first-group reads of the state must not race the trailing writes).
extern "C" __global__ __launch_bounds__(WARP)
void ds41_compress(const float* __restrict__ kv, const float* __restrict__ sc, const float* __restrict__ kst,
                   const float* __restrict__ sst, const float* __restrict__ w, uint16_t* __restrict__ out,
                   int start, int ratio, float eps)
{
    const int lane = threadIdx.x;
    const int g = start / ratio + blockIdx.x;
    float v[16];
    for (int i = 0; i < 16; i++)
    {
        const int c = lane * 16 + i;
        float k[8], s[8];
        for (int j = 0; j < ratio; j++)
        {
            const int p = g * ratio + j;
            if (p >= start) { k[j] = kv[(size_t) (p - start) * HD + c]; s[j] = sc[(size_t) (p - start) * HD + c]; }
            else { k[j] = kst[(size_t) j * HD + c]; s[j] = sst[(size_t) j * HD + c]; }
        }
        float m = s[0];
        for (int j = 1; j < ratio; j++) m = fmaxf(m, s[j]);
        float e[8], den = 0.0f;
        for (int j = 0; j < ratio; j++) { e[j] = expf(s[j] - m); den += e[j]; }
        float acc = 0.0f;
        for (int j = 0; j < ratio; j++) { float p = k[j] * (e[j] / den); acc = j == 0 ? p : acc + p; }
        v[i] = h2f(f2h(acc));
    }
    float ss = 0.0f;
    for (int i = 0; i < 16; i++) ss += v[i] * v[i];
    ss = wsum(ss);
    const float rstd = rsqrt_ieee(ss / (float) HD + eps);
    for (int i = 0; i < 16; i++) out[(size_t) blockIdx.x * HD + lane * 16 + i] = f2h(w[lane * 16 + i] * (v[i] * rstd));
}

// copies the rows of the trailing incomplete group (positions end - end % ratio .. end - 1, those >= start) into the state slots
extern "C" __global__ __launch_bounds__(256)
void ds41_compress_state(const float* __restrict__ kv, const float* __restrict__ sc, float* __restrict__ kst,
                         float* __restrict__ sst, int start, int R, int ratio)
{
    const int end = start + R;
    const int first = end - end % ratio;
    const int j = blockIdx.x;                       // slot
    const int p = first + j;
    if (j >= end % ratio || p < start) return;
    for (int c = threadIdx.x; c < HD; c += 256)
    {
        kst[(size_t) j * HD + c] = kv[(size_t) (p - start) * HD + c];
        sst[(size_t) j * HD + c] = sc[(size_t) (p - start) * HD + c];
    }
}

// -------------------------------------------------------------------------------------------------------------------
// Hyper-connections with the delayed pre of DS4.1 (the pre of one site collapses the streams two sites later).
// ds41_hc_mix_part: grid (NCH, R), 256 threads. Streams h [R, 4 * 5120] fp32, fn [24, 20480] fp32 or fp16 (FN_HALF).
// Partials per row and chunk: part[(r * NCH + c) * 25 + m], m < 24 dot products, m = 24 sum of squares.
#define HC_COLS (HCM * DIM)       // 20480
#define HC_M 24

template <bool FN_HALF>
static __device__ __forceinline__ void hc_part_body(const float* __restrict__ h, const void* __restrict__ fnv,
                                                    float* __restrict__ part, int nch)
{
    __shared__ float red[8][25];
    const int r = blockIdx.y, c = blockIdx.x, t = threadIdx.x, wv = t / WARP, lane = t % WARP;
    const int chunk = HC_COLS / nch;
    const int col = c * chunk + t * 4;                       // chunk == 1024 or a multiple of 1024 handled by the loop below
    float acc[HC_M], ssq = 0.0f;
    #pragma unroll
    for (int m = 0; m < HC_M; m++) acc[m] = 0.0f;
    for (int cc = col; cc < (c + 1) * chunk; cc += 1024)
    {
        const float4 x = *reinterpret_cast<const float4*>(h + (size_t) r * HC_COLS + cc);
        ssq += x.x * x.x; ssq += x.y * x.y; ssq += x.z * x.z; ssq += x.w * x.w;
        #pragma unroll
        for (int m = 0; m < HC_M; m++)
        {
            float f0, f1, f2, f3;
            if (FN_HALF)
            {
                const uint2 u = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(fnv) + (size_t) m * HC_COLS + cc);
                f0 = h2f(u.x & 0xFFFF); f1 = h2f(u.x >> 16); f2 = h2f(u.y & 0xFFFF); f3 = h2f(u.y >> 16);
            }
            else
            {
                const float4 f = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(fnv) + (size_t) m * HC_COLS + cc);
                f0 = f.x; f1 = f.y; f2 = f.z; f3 = f.w;
            }
            float a = acc[m];
            a += x.x * f0; a += x.y * f1; a += x.z * f2; a += x.w * f3;
            acc[m] = a;
        }
    }
    #pragma unroll
    for (int m = 0; m < HC_M; m++) acc[m] = wsum(acc[m]);
    ssq = wsum(ssq);
    if (lane == 0)
    {
        #pragma unroll
        for (int m = 0; m < HC_M; m++) red[wv][m] = acc[m];
        red[wv][24] = ssq;
    }
    __syncthreads();
    if (t < 25)
    {
        float s = 0.0f;
        for (int i = 0; i < 8; i++) s += red[i][t];
        part[((size_t) r * nch + c) * 25 + t] = s;
    }
}
extern "C" __global__ __launch_bounds__(256)
void ds41_hc_mix_part(const float* h, const float* fn, float* part, int nch) { hc_part_body<false>(h, fn, part, nch); }
extern "C" __global__ __launch_bounds__(256)
void ds41_hc_mix_part_h(const float* h, const uint16_t* fn, float* part, int nch) { hc_part_body<true>(h, fn, part, nch); }

// ds41_hc_mix_fin: grid R, one wave. mixes = dot * rsqrt(mean(h^2) + norm_eps); pre = sigmoid(m s0 + b) + hc_eps,
// post = 2 sigmoid(m s1 + b), comb = sinkhorn(softmax(m s2 + b)) (ops.hc_split_sinkhorn). scale [3], base [24] fp32.
// Outputs pre [R, 4], post [R, 4], comb [R, 16] fp32.
extern "C" __global__ __launch_bounds__(WARP)
void ds41_hc_mix_fin(const float* __restrict__ part, const float* __restrict__ scale, const float* __restrict__ base,
                     float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb, int nch, int iters,
                     float norm_eps, float hc_eps)
{
    __shared__ float mix[HC_M];
    const int r = blockIdx.x, lane = threadIdx.x;
    float tot = 0.0f, ssq = 0.0f;
    if (lane < 25)
    {
        for (int c = 0; c < nch; c++) tot += part[((size_t) r * nch + c) * 25 + lane];
    }
    ssq = __shfl(tot, 24, WARP);
    const float rstd = rsqrt_ieee(ssq / (float) HC_COLS + norm_eps);
    if (lane < HC_M) mix[lane] = tot * rstd;
    __syncthreads();
    if (lane == 0)
    {
        for (int i = 0; i < 4; i++)
        {
            pre[r * 4 + i] = sigmoidf_(mix[i] * scale[0] + base[i]) + hc_eps;
            post[r * 4 + i] = 2.0f * sigmoidf_(mix[4 + i] * scale[1] + base[4 + i]);
        }
        float cb[4][4];
        for (int a = 0; a < 4; a++)
        {
            float x[4], m = -INFINITY, den = 0.0f;
            for (int b = 0; b < 4; b++) { x[b] = mix[8 + a * 4 + b] * scale[2] + base[8 + a * 4 + b]; m = fmaxf(m, x[b]); }
            for (int b = 0; b < 4; b++) { x[b] = expf(x[b] - m); den += x[b]; }
            for (int b = 0; b < 4; b++) cb[a][b] = x[b] / den + hc_eps;
        }
        for (int b = 0; b < 4; b++)                                   // column normalise (sum over rows)
        {
            float s = cb[0][b]; s += cb[1][b]; s += cb[2][b]; s += cb[3][b];
            for (int a = 0; a < 4; a++) cb[a][b] = cb[a][b] / (s + hc_eps);
        }
        for (int it = 0; it < iters - 1; it++)
        {
            for (int a = 0; a < 4; a++)                               // row normalise
            {
                float s = cb[a][0]; s += cb[a][1]; s += cb[a][2]; s += cb[a][3];
                for (int b = 0; b < 4; b++) cb[a][b] = cb[a][b] / (s + hc_eps);
            }
            for (int b = 0; b < 4; b++)
            {
                float s = cb[0][b]; s += cb[1][b]; s += cb[2][b]; s += cb[3][b];
                for (int a = 0; a < 4; a++) cb[a][b] = cb[a][b] / (s + hc_eps);
            }
        }
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) comb[r * 16 + a * 4 + b] = cb[a][b];
    }
}

// ds41_hc_cn: [optional hc_post] + collapse with an EXTERNAL pre + rmsnorm, one block (256 threads) per row.
//   apply:  h[c] = post[c] * y + sum_j comb[j][c] * h[j]   (in place on the fp32 streams, hc_post of ops.py)
//   always: x = fp16(sum_c pre[c] * h[c]) (kept in fp32 where fp16 would overflow);  out = fp16(w * (x * rsqrt(mean(x^2) + eps)))   (hc_pre, then the next norm)
// y is fp16 (attention) or fp32 (MoE) [R, 5120]; pre may be a different site's pre than post/comb belong to (delayed pre).
extern "C" __global__ __launch_bounds__(256)
void ds41_hc_cn(float* __restrict__ h, const void* __restrict__ y, int y_fp32, const float* __restrict__ post,
                const float* __restrict__ comb, const float* __restrict__ pre, const float* __restrict__ w,
                uint16_t* __restrict__ out, int apply, float eps)
{
    __shared__ float red[8];
    const int r = blockIdx.x, t = threadIdx.x, wv = t / WARP, lane = t % WARP;
    float* hr = h + (size_t) r * HC_COLS;
    float pr[4], po[4], cm[16];
    for (int i = 0; i < 4; i++) { pr[i] = pre[r * 4 + i]; po[i] = apply ? post[r * 4 + i] : 0.0f; }
    for (int i = 0; i < 16; i++) cm[i] = apply ? comb[r * 16 + i] : 0.0f;
    float xv[20], ss = 0.0f;
    #pragma unroll
    for (int u = 0; u < 5; u++)
    {
        const int d = (u * 256 + t) * 4;                    // DIM / 4 = 1280 float4 per stream = 5 per thread
        float s[4][4];
        for (int c = 0; c < 4; c++)
        {
            const float4 q = *reinterpret_cast<const float4*>(hr + c * DIM + d);
            s[c][0] = q.x; s[c][1] = q.y; s[c][2] = q.z; s[c][3] = q.w;
        }
        if (apply)
        {
            float yv[4];
            if (y_fp32)
            {
                const float4 q = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(y) + (size_t) r * DIM + d);
                yv[0] = q.x; yv[1] = q.y; yv[2] = q.z; yv[3] = q.w;
            }
            else
            {
                const uint2 q = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(y) + (size_t) r * DIM + d);
                yv[0] = h2f(q.x & 0xFFFF); yv[1] = h2f(q.x >> 16); yv[2] = h2f(q.y & 0xFFFF); yv[3] = h2f(q.y >> 16);
            }
            float n[4][4];
            for (int c = 0; c < 4; c++)
                for (int e = 0; e < 4; e++)
                {
                    float b = cm[0 * 4 + c] * s[0][e]; b += cm[1 * 4 + c] * s[1][e]; b += cm[2 * 4 + c] * s[2][e];
                    b += cm[3 * 4 + c] * s[3][e];
                    n[c][e] = po[c] * yv[e] + b;
                }
            for (int c = 0; c < 4; c++)
            {
                *reinterpret_cast<float4*>(hr + c * DIM + d) = make_float4(n[c][0], n[c][1], n[c][2], n[c][3]);
                for (int e = 0; e < 4; e++) s[c][e] = n[c][e];
            }
        }
        for (int e = 0; e < 4; e++)
        {
            float x = pr[0] * s[0][e]; x += pr[1] * s[1][e]; x += pr[2] * s[2][e]; x += pr[3] * s[3][e];
            // fp16 rounding of the collapsed stream, except where it would overflow: the residual stream carries massive
            // activations (|x| > 65504, e.g. 68278 at column 2455 of layer 30), which fp16 turns into inf, then
            // rsqrt(inf) = 0 and inf * 0 = NaN in the norm output; the NaN row made the router emit expert id E and
            // the MoE kernels read their pointer tables out of range (GPU illegal memory access). The squares sum in fp32.
            { const float xr = h2f(f2h(x)); if (fabsf(xr) <= 65504.0f) x = xr; }
            xv[u * 4 + e] = x;
            ss += x * x;
        }
    }
    ss = wsum(ss);
    if (lane == 0) red[wv] = ss;
    __syncthreads();
    float tot = red[0];
    for (int i = 1; i < 8; i++) tot += red[i];
    const float rstd = rsqrt_ieee(tot / (float) DIM + eps);
    #pragma unroll
    for (int u = 0; u < 5; u++)
    {
        const int d = (u * 256 + t) * 4;
        for (int e = 0; e < 4; e++) out[(size_t) r * DIM + d + e] = f2h(w[d + e] * (xv[u * 4 + e] * rstd));
    }
}

// -------------------------------------------------------------------------------------------------------------------
// Router: sqrt(softplus(x @ W^T / temp)) scores, then top-k of score + bias.
// ds41_router_scores: grid (E / 8, ceil(R / 4)), 256 threads = 8 waves, wave per (expert, up to 4 rows). x fp16 [R, 5120];
// W [E, 5120] fp32 (wtype 0), fp16 (1) or bf16 (2); scores fp32 [R, E].
template <int RT>
__device__ __forceinline__ void router_scores_t(const uint16_t* __restrict__ x, const void* __restrict__ W, int wtype, float* __restrict__ scores,
                        int R, int E, float inv_temp)
{
    const int e = blockIdx.x * 8 + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int r0 = blockIdx.y * RT;
    if (e >= E) return;
    float acc[RT];
    #pragma unroll
    for (int j = 0; j < RT; j++) acc[j] = 0.0f;
    #pragma unroll 2
    for (int it = 0; it < DIM / 256; it++)
    {
        const int k = it * 256 + lane * 8;
        float wv[8];
        if (wtype == 0)
        {
            const float4 a = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(W) + (size_t) e * DIM + k);
            const float4 b = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(W) + (size_t) e * DIM + k + 4);
            wv[0] = a.x; wv[1] = a.y; wv[2] = a.z; wv[3] = a.w; wv[4] = b.x; wv[5] = b.y; wv[6] = b.z; wv[7] = b.w;
        }
        else
        {
            const uint4 u = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(W) + (size_t) e * DIM + k);
            const uint32_t q[4] = {u.x, u.y, u.z, u.w};
            for (int i = 0; i < 4; i++)
            {
                const uint32_t lo = q[i] & 0xFFFF, hi = q[i] >> 16;
                wv[2 * i] = wtype == 1 ? h2f(lo) : ds_u2f(lo << 16);
                wv[2 * i + 1] = wtype == 1 ? h2f(hi) : ds_u2f(hi << 16);
            }
        }
        #pragma unroll
        for (int j = 0; j < RT; j++)
        {
            if (r0 + j >= R) break;
            const uint4 u = *reinterpret_cast<const uint4*>(x + (size_t) (r0 + j) * DIM + k);
            const uint32_t q[4] = {u.x, u.y, u.z, u.w};
            float a = acc[j];
            for (int i = 0; i < 4; i++) { a += wv[2 * i] * h2f(q[i] & 0xFFFF); a += wv[2 * i + 1] * h2f(q[i] >> 16); }
            acc[j] = a;
        }
    }
    #pragma unroll
    for (int j = 0; j < RT; j++)
    {
        const float s = wsum(acc[j]);
        if (lane == 0 && r0 + j < R)
        {
            const float l = s * inv_temp;
            const float sp = l > 20.0f ? l : log1pf(expf(l));          // F.softplus: beta 1, threshold 20
            scores[(size_t) (r0 + j) * E + e] = sqrtf(sp);
        }
    }
}


extern "C" __global__ __launch_bounds__(256)
void ds41_router_scores(const uint16_t* __restrict__ x, const void* __restrict__ W, int wtype, float* __restrict__ scores,
                        int R, int E, float inv_temp) { router_scores_t<4>(x, W, wtype, scores, R, E, inv_temp); }
// Prefill form: a wave takes 4 consecutive experts x 8 rows, so each x load is shared by 4 experts and each W load by 8 rows.
// Per (row, expert) the accumulation order is the one of ds41_router_scores (bitwise identical scores).
extern "C" __global__ __launch_bounds__(256)
void ds41_router_scores16(const uint16_t* __restrict__ x, const void* __restrict__ W, int wtype, float* __restrict__ scores,
                          int R, int E, float inv_temp)
{
    constexpr int RT = 8, ET = 4;
    const int e0 = (blockIdx.x * 8 + threadIdx.x / WARP) * ET, lane = threadIdx.x % WARP;
    const int r0 = blockIdx.y * RT;
    if (e0 >= E) return;
    float acc[ET][RT];
    #pragma unroll
    for (int a = 0; a < ET; a++)
        #pragma unroll
        for (int j = 0; j < RT; j++) acc[a][j] = 0.0f;
    #pragma unroll 1
    for (int it = 0; it < DIM / 256; it++)
    {
        const int k = it * 256 + lane * 8;
        float wv[ET][8];
        #pragma unroll
        for (int a = 0; a < ET; a++)
        {
            const int e = e0 + a;
            if (wtype == 0)
            {
                const float4 p = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(W) + (size_t) e * DIM + k);
                const float4 q = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(W) + (size_t) e * DIM + k + 4);
                wv[a][0] = p.x; wv[a][1] = p.y; wv[a][2] = p.z; wv[a][3] = p.w; wv[a][4] = q.x; wv[a][5] = q.y; wv[a][6] = q.z; wv[a][7] = q.w;
            }
            else
            {
                const uint4 u = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(W) + (size_t) e * DIM + k);
                const uint32_t q[4] = {u.x, u.y, u.z, u.w};
                #pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    const uint32_t lo = q[i] & 0xFFFF, hi = q[i] >> 16;
                    wv[a][2 * i] = wtype == 1 ? h2f(lo) : ds_u2f(lo << 16);
                    wv[a][2 * i + 1] = wtype == 1 ? h2f(hi) : ds_u2f(hi << 16);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < RT; j++)
        {
            if (r0 + j >= R) break;
            const uint4 u = *reinterpret_cast<const uint4*>(x + (size_t) (r0 + j) * DIM + k);
            const uint32_t q[4] = {u.x, u.y, u.z, u.w};
            float xv[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) { xv[2 * i] = h2f(q[i] & 0xFFFF); xv[2 * i + 1] = h2f(q[i] >> 16); }
            #pragma unroll
            for (int a = 0; a < ET; a++)
            {
                float t = acc[a][j];
                #pragma unroll
                for (int i = 0; i < 8; i++) t += wv[a][i] * xv[i];
                acc[a][j] = t;
            }
        }
    }
    #pragma unroll
    for (int a = 0; a < ET; a++)
        #pragma unroll
        for (int j = 0; j < RT; j++)
        {
            const float s = wsum(acc[a][j]);
            if (lane == 0 && r0 + j < R)
            {
                const float l = s * inv_temp;
                const float sp = l > 20.0f ? l : log1pf(expf(l));
                scores[(size_t) (r0 + j) * E + e0 + a] = sqrtf(sp);
            }
        }
}

// ds41_router_topk: grid R, one wave. key = score + bias; K rounds of wave argmax (lowest index wins ties);
// weights = score / (sum + 1e-20) * scale over the selected set in selection order. sel int64 [R, K], wts fp16 [R, K]
// (wts_f32 fp32 [R, K] optional). E <= 512.
extern "C" __global__ __launch_bounds__(WARP)
void ds41_router_topk(const float* __restrict__ scores, const float* __restrict__ bias, int64_t* __restrict__ sel,
                      uint16_t* __restrict__ wts, float* __restrict__ wts_f32, int E, int K, int norm, float scale)
{
    const int r = blockIdx.x, lane = threadIdx.x;
    float key[16];
    for (int i = 0; i < 16; i++) { const int e = lane + i * WARP; key[i] = e < E ? scores[(size_t) r * E + e] + bias[e] : -INFINITY; }
    int chosen[16];
    for (int t = 0; t < K; t++)
    {
        float bv = -INFINITY; int bi = 0x7FFFFFFF;
        for (int i = 0; i < 16; i++) { const int e = lane + i * WARP; if (key[i] > bv || (key[i] == bv && e < bi)) { bv = key[i]; bi = e; } }
        for (int o = 16; o > 0; o >>= 1)
        {
            const float ov = __shfl_xor(bv, o, WARP); const int oi = __shfl_xor(bi, o, WARP);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        chosen[t] = bi;
        if ((bi % WARP) == lane) key[bi / WARP] = -INFINITY;
    }
    if (lane == 0)
    {
        float s[16], sum = 0.0f;
        for (int t = 0; t < K; t++) { s[t] = scores[(size_t) r * E + chosen[t]]; sum = t == 0 ? s[t] : sum + s[t]; }
        for (int t = 0; t < K; t++)
        {
            float wv = norm && K > 1 ? s[t] / (sum + 1e-20f) : s[t];
            wv *= scale;
            sel[(size_t) r * K + t] = chosen[t];
            wts[(size_t) r * K + t] = f2h(wv);
            if (wts_f32) wts_f32[(size_t) r * K + t] = wv;
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------
// Engram. ds41_engram_comp: cache[start + r] = token_map[ids[r]] (or -1 where mask[r] == 0). ids int64 [R], tmap int32,
// cache int64 [max_seq], mask uint8 [R] or null.
extern "C" __global__ __launch_bounds__(256)
void ds41_engram_comp(const int64_t* __restrict__ ids, const int* __restrict__ tmap, int64_t* __restrict__ cache,
                      const uint8_t* __restrict__ mask, int start, int R)
{
    const int r = blockIdx.x * 256 + threadIdx.x;
    if (r >= R) return;
    cache[start + r] = (mask && !mask[r]) ? -1 : (int64_t) tmap[ids[r]];
}

// ds41_engram_hash: grid R, one thread per (layer, column). cache int64, mult [nl, ng], primes [nl, (ng-1)*nh],
// offsets [nl, (ng-1)*nh] (all int64) -> rows int64 [R, nl, (ng-1)*nh]. cache must already hold this chunk's ids (ds41_engram_comp).
extern "C" __global__ __launch_bounds__(64)
void ds41_engram_hash(const int64_t* __restrict__ cache, const int64_t* __restrict__ mult, const int64_t* __restrict__ primes,
                      const int64_t* __restrict__ offsets, int64_t* __restrict__ rows, int start, int nl, int ng, int nh,
                      int64_t pad)
{
    const int r = blockIdx.x, nc = (ng - 1) * nh;
    const int t = threadIdx.x;
    if (t >= nl * nc) return;
    const int l = t / nc, c = t % nc;
    int64_t toks[8];
    ds_engram_toks(cache, (int64_t) start + r, ng, pad, toks);
    rows[((size_t) r * nl + l) * nc + c] =
        ds_engram_hash_col(toks, nh, c, mult + (size_t) l * ng, primes + (size_t) l * nc, offsets + (size_t) l * nc);
}

// ds41_mxfp4_decode: raw uint8 [N, 136] (128 bytes of e2m1 pairs, low nibble first, then 8 e8m0 scales per 32 values) ->
// out fp16 [M, 256]. inv int32 [M] picks the source row (dedup'ed fetch), or null for M == N. One thread per 8 values.
extern "C" __global__ __launch_bounds__(256)
void ds41_mxfp4_decode(const uint8_t* __restrict__ raw, const int* __restrict__ inv, uint16_t* __restrict__ out, long M)
{
    const long g = (long) blockIdx.x * 256 + threadIdx.x;
    if (g >= M * 32) return;
    const long m = g / 32;
    const int t = (int) (g % 32);
    const long src = inv ? inv[m] : m;
    const uint8_t* row = raw + src * 136;
    const uint32_t q = *reinterpret_cast<const uint32_t*>(row + t * 4);
    const float s = ds_e8m0(row[128 + t / 4]);
    uint16_t o[8];
    for (int i = 0; i < 4; i++)
    {
        const uint32_t byte = (q >> (8 * i)) & 0xFF;
        o[2 * i] = f2h(ds_e2m1(byte & 15) * s);
        o[2 * i + 1] = f2h(ds_e2m1(byte >> 4) * s);
    }
    uint4 pk;
    pk.x = (uint32_t) o[0] | ((uint32_t) o[1] << 16); pk.y = (uint32_t) o[2] | ((uint32_t) o[3] << 16);
    pk.z = (uint32_t) o[4] | ((uint32_t) o[5] << 16); pk.w = (uint32_t) o[6] | ((uint32_t) o[7] << 16);
    *reinterpret_cast<uint4*>(out + m * 256 + t * 8) = pk;
}

// ds41_fp8_decode: raw uint8 [N, 264] (256 e4m3 values, then 8 e8m0 scales per 32 values; the official Engram tables) ->
// out fp16 [M, 256]; inv as in ds41_mxfp4_decode. One thread per 8 values. Same math as engram.dequant_rows (float multiply, round to half).
static __device__ __forceinline__ float ds_e4m3(uint32_t b)
{
    const uint32_t e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 15 && m == 7) v = __uint_as_float(0x7FC00000u);
    else if (e == 0) v = (float) m * 0.001953125f;                      // m * 2^-9
    else v = (1.0f + (float) m * 0.125f) * __uint_as_float((e + 120u) << 23);   // 2^(e-7)
    return v;                                                       // magnitude; the sign is applied on the half bits (keeps -0)
}

static __device__ __forceinline__ uint16_t ds_e4m3_h(uint32_t b, float s)
{
    return (uint16_t) (f2h(ds_e4m3(b) * s) | ((b & 128) << 8));
}

extern "C" __global__ __launch_bounds__(256)
void ds41_fp8_decode(const uint8_t* __restrict__ raw, const int* __restrict__ inv, uint16_t* __restrict__ out, long M)
{
    const long g = (long) blockIdx.x * 256 + threadIdx.x;
    if (g >= M * 32) return;
    const long m = g / 32;
    const int t = (int) (g % 32);
    const long src = inv ? inv[m] : m;
    const uint8_t* row = raw + src * 264;
    const uint2 q = *reinterpret_cast<const uint2*>(row + t * 8);
    const uint32_t sb = row[256 + t / 4];
    const float s = sb == 255 ? __uint_as_float(0x7FC00000u) : ds_e8m0(sb);
    uint16_t o[8];
    for (int i = 0; i < 4; i++)
    {
        o[i] = ds_e4m3_h((q.x >> (8 * i)) & 0xFF, s);
        o[4 + i] = ds_e4m3_h((q.y >> (8 * i)) & 0xFF, s);
    }
    uint4 pk;
    pk.x = (uint32_t) o[0] | ((uint32_t) o[1] << 16); pk.y = (uint32_t) o[2] | ((uint32_t) o[3] << 16);
    pk.z = (uint32_t) o[4] | ((uint32_t) o[5] << 16); pk.w = (uint32_t) o[6] | ((uint32_t) o[7] << 16);
    *reinterpret_cast<uint4*>(out + m * 256 + t * 8) = pk;
}

// ds41_engram_gate: grid R, 128 threads (wave per stream). h fp32 [R, 4, 5120] updated in place; kv fp16 [R, 5 * 5120]
// (key [4, 5120] then value [5120], the wkv output); qk fp32 [4, 5120] = q_weight * k_weight.
//   rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps); dot = sum(h * qk * key) * rstd * dim^-0.5
//   gate = sigmoid(copysign(sqrt(max(|dot|, clamp)), dot));  h += gate * value
extern "C" __global__ __launch_bounds__(128)
void ds41_engram_gate(float* __restrict__ h, const uint16_t* __restrict__ kv, const float* __restrict__ qk,
                      float eps, float dim_scale, float clamp_value)
{
    const int r = blockIdx.x, c = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    float* hp = h + ((size_t) r * HCM + c) * DIM;
    const uint16_t* key = kv + (size_t) r * (HCM + 1) * DIM + (size_t) c * DIM;
    const uint16_t* val = kv + (size_t) r * (HCM + 1) * DIM + (size_t) HCM * DIM;
    const float* q = qk + (size_t) c * DIM;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (int d = lane * 4; d < DIM; d += WARP * 4)
    {
        const float4 hv = *reinterpret_cast<const float4*>(hp + d);
        const float4 qv = *reinterpret_cast<const float4*>(q + d);
        const uint2 ku = *reinterpret_cast<const uint2*>(key + d);
        const float k0 = h2f(ku.x & 0xFFFF), k1 = h2f(ku.x >> 16), k2 = h2f(ku.y & 0xFFFF), k3 = h2f(ku.y >> 16);
        sh += hv.x * hv.x; sh += hv.y * hv.y; sh += hv.z * hv.z; sh += hv.w * hv.w;
        sk += k0 * k0; sk += k1 * k1; sk += k2 * k2; sk += k3 * k3;
        sd += hv.x * qv.x * k0; sd += hv.y * qv.y * k1; sd += hv.z * qv.z * k2; sd += hv.w * qv.w * k3;
    }
    sh = wsum(sh); sk = wsum(sk); sd = wsum(sd);
    const float rstd = rsqrt_ieee(sh / (float) DIM + eps) * rsqrt_ieee(sk / (float) DIM + eps);
    const float dot = sd * rstd * dim_scale;
    const float gate = sigmoidf_(copysignf(sqrtf(fmaxf(fabsf(dot), clamp_value)), dot));
    for (int d = lane * 4; d < DIM; d += WARP * 4)
    {
        float4 hv = *reinterpret_cast<const float4*>(hp + d);
        const uint2 vu = *reinterpret_cast<const uint2*>(val + d);
        hv.x += gate * h2f(vu.x & 0xFFFF); hv.y += gate * h2f(vu.x >> 16);
        hv.z += gate * h2f(vu.y & 0xFFFF); hv.w += gate * h2f(vu.y >> 16);
        *reinterpret_cast<float4*>(hp + d) = hv;
    }
}

// -------------------------------------------------------------------------------------------------------------------
// ds41_attn_dec: decode (1 token) sparse attention of one layer, all 64 heads, one block (256 threads) per head.
//   q [64, 512] fp16 straight from wq_b (rope on the last 64 dims is applied here at pos, rounded to fp16 like the reference),
//   keys = the sliding window (ring [128, 512] fp16, already holding the current token; position p lives in slot p & 127,
//   oldest first, p < 0 masked) ++ nc compressed rows comp[cidx[i] - coff] (cidx -1 = masked, order as given),
//   softmax(scale * q.k) with the per-head sink logit in the denominator only, o = p.k / den rounded to fp16,
//   inverse rope on the last 64 dims, fp8 fake-quant (ue8m0 / 32), written to out [64, 512] fp16 (= the wo_a input).
// Replaces rope_apply_(q), window_idxs, cat, sparse_attention, rope_apply_(o, inverse), act_quant_fp8_(o) of the decode step.
#define ATT_MAXK 1152
#define ATT_HB 2                 // heads per block: every key row is read once for ATT_HB heads
extern "C" __global__ __launch_bounds__(256)
void ds41_attn_dec(const uint16_t* __restrict__ q, const uint16_t* __restrict__ ring, const uint16_t* __restrict__ comp,
                   const int* __restrict__ cidx, int nc, int coff, const float* __restrict__ sink,
                   const float* __restrict__ fc, const int* __restrict__ pos, uint16_t* __restrict__ out, float scale)
{
    __shared__ float qs[ATT_HB][HD];
    __shared__ float sc[ATT_HB][ATT_MAXK];
    __shared__ float part[8][ATT_HB][HD];
    __shared__ float red[ATT_HB][8];
    const int t = threadIdx.x, w = t >> 5, lane = t & 31, h0 = blockIdx.x * ATT_HB;
    const int p = pos[0];
    const int n_tot = 128 + nc;
    const float* ft = fc + (size_t) p * 64;
    #pragma unroll
    for (int hh = 0; hh < ATT_HB; hh++)
    {
        float a = h2f(q[(size_t) (h0 + hh) * HD + 2 * t]), b = h2f(q[(size_t) (h0 + hh) * HD + 2 * t + 1]);
        if (t >= (HD - RD) / 2)
        {
            const int pr = t - (HD - RD) / 2;
            rot(a, b, ft[2 * pr], ft[2 * pr + 1], false);
            a = h2f(f2h(a)); b = h2f(f2h(b));
        }
        qs[hh][2 * t] = a; qs[hh][2 * t + 1] = b;
    }
    __syncthreads();
    float qr[ATT_HB][16];
    #pragma unroll
    for (int hh = 0; hh < ATT_HB; hh++)
        #pragma unroll
        for (int i = 0; i < 16; i++) qr[hh][i] = qs[hh][lane * 16 + i];
    // scores: 4 keys in flight per wave
    for (int i0 = w; i0 < n_tot; i0 += 32)
    {
        uint4 u[4][2]; bool va[4];
        #pragma unroll
        for (int k = 0; k < 4; k++)
        {
            const int i = i0 + 8 * k;
            va[k] = false;
            if (i < n_tot)
            {
                const uint16_t* row;
                if (i < 128) { const int pp = p - 127 + i; va[k] = pp >= 0; row = ring + (size_t) (pp & 127) * HD; }
                else { const int c = cidx[i - 128]; va[k] = c >= 0; row = comp + (size_t) (va[k] ? c - coff : 0) * HD; }
                if (va[k]) { const uint4* r4 = (const uint4*) (row + lane * 16); u[k][0] = r4[0]; u[k][1] = r4[1]; }
            }
        }
        #pragma unroll
        for (int k = 0; k < 4; k++)
        {
            const int i = i0 + 8 * k;
            if (i >= n_tot) break;
            float d[ATT_HB];
            #pragma unroll
            for (int hh = 0; hh < ATT_HB; hh++) d[hh] = 0.0f;
            if (va[k])
            {
                const uint16_t* e0 = (const uint16_t*) &u[k][0]; const uint16_t* e1 = (const uint16_t*) &u[k][1];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    const float k0 = h2f(e0[j]), k1 = h2f(e1[j]);
                    #pragma unroll
                    for (int hh = 0; hh < ATT_HB; hh++) { d[hh] += qr[hh][j] * k0; d[hh] += qr[hh][8 + j] * k1; }
                }
            }
            #pragma unroll
            for (int hh = 0; hh < ATT_HB; hh++)
            {
                const float v = wsum(d[hh]);
                if (lane == 0) sc[hh][i] = va[k] ? v * scale : -INFINITY;
            }
        }
    }
    __syncthreads();
    float mx[ATT_HB];
    #pragma unroll
    for (int hh = 0; hh < ATT_HB; hh++)
    {
        float m = -INFINITY;
        for (int i = t; i < n_tot; i += 256) m = fmaxf(m, sc[hh][i]);
        m = wmax(m);
        if (lane == 0) red[hh][w] = m;
    }
    __syncthreads();
    #pragma unroll
    for (int hh = 0; hh < ATT_HB; hh++)
    {
        float m = red[hh][0];
        #pragma unroll
        for (int i = 1; i < 8; i++) m = fmaxf(m, red[hh][i]);
        mx[hh] = fmaxf(m, -1e30f);
    }
    __syncthreads();
    #pragma unroll
    for (int hh = 0; hh < ATT_HB; hh++)
    {
        float ds = 0.0f;
        for (int i = t; i < n_tot; i += 256) { const float e = expf(sc[hh][i] - mx[hh]); sc[hh][i] = e; ds += e; }
        ds = wsum(ds);
        if (lane == 0) red[hh][w] = ds;
    }
    __syncthreads();
    float den[ATT_HB];
    #pragma unroll
    for (int hh = 0; hh < ATT_HB; hh++)
    {
        float d = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) d += red[hh][i];
        den[hh] = d + expf(sink[h0 + hh] - mx[hh]);
    }
    float acc[ATT_HB][16];
    #pragma unroll
    for (int hh = 0; hh < ATT_HB; hh++)
        #pragma unroll
        for (int i = 0; i < 16; i++) acc[hh][i] = 0.0f;
    for (int i0 = w; i0 < n_tot; i0 += 32)
    {
        uint4 u[4][2]; float pw[4][ATT_HB]; bool ok[4];
        #pragma unroll
        for (int k = 0; k < 4; k++)
        {
            const int i = i0 + 8 * k;
            ok[k] = false;
            if (i < n_tot)
            {
                bool any = false;
                #pragma unroll
                for (int hh = 0; hh < ATT_HB; hh++) { pw[k][hh] = sc[hh][i]; any = any || pw[k][hh] != 0.0f; }
                if (any)
                {
                    ok[k] = true;
                    const uint16_t* row = i < 128 ? ring + (size_t) ((p - 127 + i) & 127) * HD
                                                  : comp + (size_t) (cidx[i - 128] - coff) * HD;
                    const uint4* r4 = (const uint4*) (row + lane * 16); u[k][0] = r4[0]; u[k][1] = r4[1];
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < 4; k++)
        {
            if (!ok[k]) continue;
            const uint16_t* e0 = (const uint16_t*) &u[k][0]; const uint16_t* e1 = (const uint16_t*) &u[k][1];
            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                const float k0 = h2f(e0[j]), k1 = h2f(e1[j]);
                #pragma unroll
                for (int hh = 0; hh < ATT_HB; hh++) { acc[hh][j] += pw[k][hh] * k0; acc[hh][8 + j] += pw[k][hh] * k1; }
            }
        }
    }
    #pragma unroll
    for (int hh = 0; hh < ATT_HB; hh++)
        #pragma unroll
        for (int i = 0; i < 16; i++) part[w][hh][lane * 16 + i] = acc[hh][i];
    __syncthreads();
    #pragma unroll
    for (int hh = 0; hh < ATT_HB; hh++)
    {
        float a = 0.0f, b = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) { a += part[i][hh][2 * t]; b += part[i][hh][2 * t + 1]; }
        a = h2f(f2h(a / den[hh])); b = h2f(f2h(b / den[hh]));
        if (t >= (HD - RD) / 2)
        {
            const int pr = t - (HD - RD) / 2;
            rot(a, b, ft[2 * pr], ft[2 * pr + 1], true);
            a = h2f(f2h(a)); b = h2f(f2h(b));
        }
        float am = fmaxf(fabsf(a), fabsf(b));
        am = fmaxf(am, __shfl_xor(am, 8, WARP)); am = fmaxf(am, __shfl_xor(am, 4, WARP));
        am = fmaxf(am, __shfl_xor(am, 2, WARP)); am = fmaxf(am, __shfl_xor(am, 1, WARP));
        const float s = ds_fp8_scale(am);
        out[(size_t) (h0 + hh) * HD + 2 * t] = f2h(ds_fp8_q(a, s));
        out[(size_t) (h0 + hh) * HD + 2 * t + 1] = f2h(ds_fp8_q(b, s));
    }
}

// -------------------------------------------------------------------------------------------------------------------
// ds41_dgemv_<w><x>: y[r, n] = sum_k W[n, k] * x[r, k] for R <= 8 rows, K % 256 == 0. One wave per output row n (8 waves per
// block), 8 contiguous k per lane per step, fp32 accumulation. W: f = fp32, b = bf16 (a lossless storage of bf16-origin weights),
// h = fp16; x: f = fp32, h = fp16; y has the dtype of x. Replaces the Tensile fp32 GEMMs of the compressor and the indexer
// projections at decode, which run at ~45 GB/s.
template <int WT, int XT>
static __device__ __forceinline__ void dgemv_body(const void* __restrict__ W, const void* __restrict__ X, void* __restrict__ Y,
                                                  int K, int N, int R)
{
    const int wv = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + wv;
    if (n >= N) return;
    float acc[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) acc[r] = 0.0f;
    for (int k0 = lane * 8; k0 < K; k0 += 256)
    {
        float wf[8];
        if (WT == 0)
        {
            const float4* p = (const float4*) ((const float*) W + (size_t) n * K + k0);
            float4 a = p[0], b = p[1];
            wf[0] = a.x; wf[1] = a.y; wf[2] = a.z; wf[3] = a.w; wf[4] = b.x; wf[5] = b.y; wf[6] = b.z; wf[7] = b.w;
        }
        else
        {
            uint4 u = *(const uint4*) ((const uint16_t*) W + (size_t) n * K + k0);
            const uint16_t* e = (const uint16_t*) &u;
            #pragma unroll
            for (int i = 0; i < 8; i++) wf[i] = WT == 1 ? ds_u2f((uint32_t) e[i] << 16) : h2f(e[i]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++)
        {
            if (r >= R) break;
            float xf[8];
            if (XT == 0)
            {
                const float4* p = (const float4*) ((const float*) X + (size_t) r * K + k0);
                float4 a = p[0], b = p[1];
                xf[0] = a.x; xf[1] = a.y; xf[2] = a.z; xf[3] = a.w; xf[4] = b.x; xf[5] = b.y; xf[6] = b.z; xf[7] = b.w;
            }
            else
            {
                uint4 u = *(const uint4*) ((const uint16_t*) X + (size_t) r * K + k0);
                const uint16_t* e = (const uint16_t*) &u;
                #pragma unroll
                for (int i = 0; i < 8; i++) xf[i] = h2f(e[i]);
            }
            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) s += wf[i] * xf[i];
            acc[r] += s;
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r++)
    {
        if (r >= R) break;
        const float v = wsum(acc[r]);
        if (lane == 0)
        {
            if (XT == 0) ((float*) Y)[(size_t) r * N + n] = v;
            else ((uint16_t*) Y)[(size_t) r * N + n] = f2h(v);
        }
    }
}
#define DGEMV(name, WT, XT) extern "C" __global__ __launch_bounds__(256) \
    void name(const void* W, const void* X, void* Y, int K, int N, int R) { dgemv_body<WT, XT>(W, X, Y, K, N, R); }
DGEMV(ds41_dgemv_ff, 0, 0)
DGEMV(ds41_dgemv_bf, 1, 0)
DGEMV(ds41_dgemv_hf, 2, 0)
DGEMV(ds41_dgemv_fh, 0, 1)
DGEMV(ds41_dgemv_bh, 1, 1)
DGEMV(ds41_dgemv_hh, 2, 1)

// ds41_rmsnorm_wide: ds41_rmsnorm_rows with a 256-thread block per row (N % 8 == 0, N <= 8192): 8 contiguous elements per thread,
// the row read once into registers. Same formula; the sum of squares is reduced wave-first then across the 8 waves.
extern "C" __global__ __launch_bounds__(256)
void ds41_rmsnorm_wide(const uint16_t* __restrict__ x, const float* __restrict__ w, uint16_t* __restrict__ y, int N, float eps)
{
    __shared__ float red[8];
    const int r = blockIdx.x, t = threadIdx.x, wv = t >> 5, lane = t & 31;
    const uint16_t* p = x + (size_t) r * N;
    float v[8]; float ss = 0.0f;
    const bool on = t * 8 < N;
    if (on)
    {
        uint4 u = *(const uint4*) (p + t * 8);
        const uint16_t* e = (const uint16_t*) &u;
        #pragma unroll
        for (int i = 0; i < 8; i++) { v[i] = h2f(e[i]); ss += v[i] * v[i]; }
    }
    ss = wsum(ss);
    if (lane == 0) red[wv] = ss;
    __syncthreads();
    float tot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) tot += red[i];
    const float rstd = rsqrt_ieee(tot / (float) N + eps);
    if (on)
    {
        uint16_t o[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = f2h(w[t * 8 + i] * (v[i] * rstd));
        *(uint4*) (y + (size_t) r * N + t * 8) = *(const uint4*) o;
    }
}
