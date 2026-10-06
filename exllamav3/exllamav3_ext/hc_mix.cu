#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include "hc_mix.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#if !defined(USE_ROCM)
#include "graph.cuh"
#endif

/*

Fused mHC HyperConnection mix() kernel

mix(streams (R, H, D) fp32) -> post (R, H), comb (R, H, H), collapsed (R, D):

  flat = rmsnorm_unweighted(streams.flatten)          (row of H * D values)
  mixv = flat @ fn.T                                  (M = 2H + H^2 outputs)
  pre  = sigmoid(mixv[0:H]  * s0 + base[0:H]) + eps
  post = 2 sigmoid(mixv[H:2H] * s1 + base[H:2H])
  comb = sinkhorn(softmax(mixv[2H:] * s2 + base[2H:]))   (iters alternating row/col norms)
  collapsed = sum_h pre[h] * streams[h]

Two launches, no grid-wide sync, deterministic (fixed reduction order, no atomics):
  K1 partials: grid (chunksA, R); each block reduces its column chunk to M + 1 partials.
  K2 finalize: grid (chunksC, R); EVERY block re-reduces the tiny partial matrix and
    derives rmr + pre redundantly (removes the cross-block dependency), then streams its
    chunk of collapsed; the chunk-0 block also runs the sinkhorn on H^2 lanes of warp 0
    (row sums: shfl_xor 1|2, col sums: shfl_xor 4|8 for H = 4) and writes post/comb.

*/

// Partials blocks are small: at R = 1 the grid is the only parallelism, so favor many
// blocks (row_len / (4 * 64) chunks) over wide ones; the M + 1 block reduce also shrinks
#define NUM_THREADS 256
#define NUM_THREADS_A 64

__device__ __forceinline__ float sigmoidf_(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

template <int H, int M_, typename FN_T>
__global__ __launch_bounds__(NUM_THREADS_A)
void hc_mix_partials_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const FN_T* __restrict__ fn,         // (M, H * D) float, or half (opt-in, halves traffic)
    float* __restrict__ partials,        // (R, chunksA, M + 1)
    const int row_len,
    const int chunk_cols                 // multiple of 4 * NUM_THREADS_A
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);

    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int row_len4 = row_len / 4;

    float acc[M + 1];
    #pragma unroll
    for (int k = 0; k <= M; ++k) acc[k] = 0.0f;

    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS_A)
    {
        float4 s = s4[c];
        acc[M] = fmaf(s.x, s.x, acc[M]);
        acc[M] = fmaf(s.y, s.y, acc[M]);
        acc[M] = fmaf(s.z, s.z, acc[M]);
        acc[M] = fmaf(s.w, s.w, acc[M]);
        #pragma unroll
        for (int j = 0; j < M; ++j)
        {
            float4 w;
            if constexpr (std::is_same_v<FN_T, half>)
            {
                // Single vectorized 8-byte load (two half2 loads would double the LDGs)
                int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                float2 lo = __half22float2(*(const half2*) &pk.x);
                float2 hi = __half22float2(*(const half2*) &pk.y);
                w = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            else
                w = ((const float4*) fn)[(size_t) j * row_len4 + c];
            float d = fmaf(s.x, w.x, fmaf(s.y, w.y, fmaf(s.z, w.z, s.w * w.w)));
            acc[j] += d;
        }
    }

    // Block reduce M + 1 lanes' accumulators
    __shared__ float red[NUM_THREADS_A / 32][M + 1];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    #pragma unroll
    for (int k = 0; k <= M; ++k)
    {
        float v = acc[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();

    if (warp == 0)
    {
        float* out = partials + ((size_t) r * gridDim.x + blockIdx.x) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_THREADS_A / 32; ++w)
                v += red[w][k];
            out[k] = v;
        }
    }
}

template <int H, int M_, bool HEAD, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void hc_mix_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ partials,  // (R, chunksA, M + 1)
    const float* __restrict__ base,      // (M)
    const float* __restrict__ scale,     // (3)
    float* __restrict__ post,            // (R, H)
    float* __restrict__ comb,            // (R, H, H)
    void* __restrict__ collapsed,        // (R, D) float or half
    const int D,
    const int chunksA,
    const int chunk_cols_c,              // multiple of 4
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int row_len = H * D;

    // Re-reduce this row's partials (tiny, L2-resident). Every block does this to avoid
    // cross-block dependencies. Warp-split over the chunk axis: the serial loop sits
    // at the head of the kernel's critical path, so with many partials chunks (small
    // NUM_THREADS_A blocks) a single-thread-per-quantity loop is too long
    __shared__ float mix_s[M + 1];
    __shared__ float pre_s[H];
    __shared__ float red_s[NUM_THREADS / 32][M + 1];
    {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane <= M)
        {
            const float* p = partials + (size_t) r * chunksA * (M + 1) + lane;
            float v = 0.0f;
            for (int i = warp; i < chunksA; i += NUM_THREADS / 32)
                v += p[(size_t) i * (M + 1)];
            red_s[warp][lane] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x <= M)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_THREADS / 32; ++w)
            v += red_s[w][threadIdx.x];
        mix_s[threadIdx.x] = v;
    }
    __syncthreads();
    float rmr = rsqrtf(mix_s[M] / (float) row_len + rms_eps);
    if (threadIdx.x < H)
        pre_s[threadIdx.x] = sigmoidf_(fmaf(mix_s[threadIdx.x] * rmr, scale[0], base[threadIdx.x])) + hc_eps;
    __syncthreads();

    // Sinkhorn + post/comb writes: chunk-0 block only, one lane per comb element. Runs in
    // a dedicated warp, concurrent with the other warps' phase C: the ~20-iteration
    // normalization is a serial shfl/div latency chain that only needs the M + 1 reduced
    // scalars, so at small R it sets the kernel's critical path; don't stack phase C
    // work in front of it.
    //
    // Lane l = i * H + j; row sums reduce over j (xor 1..H/2), col sums over i (xor H..).
    const bool sink_warp = !HEAD && blockIdx.x == 0 && threadIdx.x < 32;
    if (sink_warp)
    {
        if (threadIdx.x < H)
            post[(size_t) r * H + threadIdx.x] =
                2.0f * sigmoidf_(fmaf(mix_s[H + threadIdx.x] * rmr, scale[1], base[H + threadIdx.x]));

        if (threadIdx.x < H * H)
        {
            const unsigned mask = (H * H == 32) ? 0xffffffffu : ((1u << (H * H)) - 1u);
            float v = fmaf(mix_s[2 * H + threadIdx.x] * rmr, scale[2], base[2 * H + threadIdx.x]);

            // softmax over rows
            float m = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) m = fmaxf(m, __shfl_xor_sync(mask, m, o));
            v = __expf(v - m);
            float s = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) s += __shfl_xor_sync(mask, s, o);
            v = __fdividef(v, s) + hc_eps;

            // column normalize, then (iters - 1) x (row, column)
            float cs = v;
            #pragma unroll
            for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
            v = __fdividef(v, cs + hc_eps);
            for (int it = 0; it < sinkhorn_iters - 1; ++it)
            {
                float rs = v;
                #pragma unroll
                for (int o = 1; o < H; o <<= 1) rs += __shfl_xor_sync(mask, rs, o);
                v = __fdividef(v, rs + hc_eps);
                cs = v;
                #pragma unroll
                for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
                v = __fdividef(v, cs + hc_eps);
            }
            comb[(size_t) r * H * H + threadIdx.x] = v;
        }
        return;
    }

    // Phase C: collapsed chunk, weighted sum over the H stream rows. In the sinkhorn
    // block the first warp is excluded, so the remaining threads re-cover its lanes
    float pre_r[H];
    #pragma unroll
    for (int h = 0; h < H; ++h) pre_r[h] = pre_s[h];

    const bool shrunk = !HEAD && blockIdx.x == 0;
    const int tid = shrunk ? threadIdx.x - 32 : threadIdx.x;
    const int nth = shrunk ? NUM_THREADS - 32 : NUM_THREADS;
    const int c0 = blockIdx.x * chunk_cols_c;
    const int c1 = min(c0 + chunk_cols_c, D);
    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int D4 = D / 4;
    for (int c = c0 / 4 + tid; c < c1 / 4; c += nth)
    {
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            o.x = fmaf(pre_r[h], s.x, o.x);
            o.y = fmaf(pre_r[h], s.y, o.y);
            o.z = fmaf(pre_r[h], s.z, o.z);
            o.w = fmaf(pre_r[h], s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) collapsed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) collapsed + (size_t) r * D))[c] = o;
    }
}


/*

hc_mix_norm: hc_mix (non-head) with the consumer's RMSNorm fused into the finalize step.
  K2': grid (R), NUM_THREADS + 32 threads. Warps 0..7 re-reduce partials exactly as
  hc_mix_finalize_kernel, then compute the whole collapsed row, round it to half (the
  HALF_OUT store), and apply exl3_dec rms_norm_kernel's arithmetic (mode 0) to it: same
  per-thread element order (i = tid + j * 256), same wave butterfly and serial 8-wave sum.
  Warp 8 runs the sinkhorn / post / comb writes concurrently. Bit-identical to
  hc_mix(collapsed half) -> rms_norm(y half).

*/

#define HCN_MAX 32

__device__ __forceinline__ float hcn_ld_w(const void* w, int wtype, int i)
{
    if (wtype == 1) return __half2float(reinterpret_cast<const half*>(w)[i]);
    if (wtype == 2) return __uint_as_float(((uint32_t) reinterpret_cast<const uint16_t*>(w)[i]) << 16);
    return 1.0f;
}

template <int H, int M_>
__device__ __forceinline__ void hcn_finalize_body
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ partials,  // (R, chunksA, M + 1)
    const float* __restrict__ base,      // (M)
    const float* __restrict__ scale,     // (3)
    float* __restrict__ post,            // (R, H)
    float* __restrict__ comb,            // (R, H, H)
    half* __restrict__ y,                // (R, D) normed half out
    const void* __restrict__ w,          // (D) half / bf16 / null
    const int wtype,
    const int D,
    const int chunksA,
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters,
    const float norm_eps,
    const float constant_bias,
    const float constant_scale
,
    const int r
)
{
    constexpr int M = M_;
    const int row_len = H * D;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const bool sink_warp = warp == NUM_THREADS / 32;

    __shared__ float mix_s[M + 1];
    __shared__ float pre_s[H];
    __shared__ float red_s[NUM_THREADS / 32][M + 1];
    __shared__ float nred[NUM_THREADS / 32];
    if (!sink_warp && lane <= M)
    {
        const float* p = partials + (size_t) r * chunksA * (M + 1) + lane;
        float v = 0.0f;
        for (int i = warp; i < chunksA; i += NUM_THREADS / 32)
            v += p[(size_t) i * (M + 1)];
        red_s[warp][lane] = v;
    }
    __syncthreads();
    if (threadIdx.x <= M)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w_ = 0; w_ < NUM_THREADS / 32; ++w_)
            v += red_s[w_][threadIdx.x];
        mix_s[threadIdx.x] = v;
    }
    __syncthreads();
    float rmr = rsqrtf(mix_s[M] / (float) row_len + rms_eps);
    if (threadIdx.x < H)
        pre_s[threadIdx.x] = sigmoidf_(fmaf(mix_s[threadIdx.x] * rmr, scale[0], base[threadIdx.x])) + hc_eps;
    __syncthreads();

    float v[HCN_MAX];
    float ss = 0.0f;
    if (sink_warp)
    {
        if (lane < H)
            post[(size_t) r * H + lane] =
                2.0f * sigmoidf_(fmaf(mix_s[H + lane] * rmr, scale[1], base[H + lane]));

        if (lane < H * H)
        {
            const unsigned mask = (H * H == 32) ? 0xffffffffu : ((1u << (H * H)) - 1u);
            float c = fmaf(mix_s[2 * H + lane] * rmr, scale[2], base[2 * H + lane]);
            float m = c;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) m = fmaxf(m, __shfl_xor_sync(mask, m, o));
            c = __expf(c - m);
            float s = c;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) s += __shfl_xor_sync(mask, s, o);
            c = __fdividef(c, s) + hc_eps;
            float cs = c;
            #pragma unroll
            for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
            c = __fdividef(c, cs + hc_eps);
            for (int it = 0; it < sinkhorn_iters - 1; ++it)
            {
                float rs = c;
                #pragma unroll
                for (int o = 1; o < H; o <<= 1) rs += __shfl_xor_sync(mask, rs, o);
                c = __fdividef(c, rs + hc_eps);
                cs = c;
                #pragma unroll
                for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
                c = __fdividef(c, cs + hc_eps);
            }
            comb[(size_t) r * H * H + lane] = c;
        }
    }
    else
    {
        float pre_r[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) pre_r[h] = pre_s[h];
        const float* s = streams + (size_t) r * row_len;
        #pragma unroll
        for (int j = 0; j < HCN_MAX; ++j)
        {
            const int i = threadIdx.x + j * NUM_THREADS;
            v[j] = 0.0f;
            if (i < D)
            {
                float o = 0.0f;
                #pragma unroll
                for (int h = 0; h < H; ++h)
                    o = fmaf(pre_r[h], s[(size_t) h * D + i], o);
                // collapsed half store + rms_norm ld_f<half> reload
                float t = fminf(fmaxf(__half2float(__float2half_rn(o)), -65504.0f), 65504.0f);
                v[j] = t;
                ss += t * t;
            }
        }
    }

    // block_sum over the 8 norm waves (rms_norm_kernel order)
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
    __syncthreads();
    if (!sink_warp && lane == 0) nred[warp] = ss;
    __syncthreads();
    if (sink_warp) return;
    float tot = 0.0f;
    for (int w_ = 0; w_ < NUM_THREADS / 32; ++w_) tot += nred[w_];
    const float nscale = rsqrtf(tot / (float) D + norm_eps) * constant_scale;

    half* yr = y + (size_t) r * D;
    #pragma unroll
    for (int j = 0; j < HCN_MAX; ++j)
    {
        const int i = threadIdx.x + j * NUM_THREADS;
        if (i < D)
        {
            float o = v[j] * nscale;
            if (wtype) o *= hcn_ld_w(w, wtype, i) + constant_bias;
            yr[i] = __float2half_rn(o);
        }
    }
}



template <int H, int M_>
__global__ __launch_bounds__(NUM_THREADS + 32)
void hc_mix_finalize_norm_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ partials,  // (R, chunksA, M + 1)
    const float* __restrict__ base,      // (M)
    const float* __restrict__ scale,     // (3)
    float* __restrict__ post,            // (R, H)
    float* __restrict__ comb,            // (R, H, H)
    half* __restrict__ y,                // (R, D) normed half out
    const void* __restrict__ w,          // (D) half / bf16 / null
    const int wtype,
    const int D,
    const int chunksA,
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters,
    const float norm_eps,
    const float constant_bias,
    const float constant_scale
)
{
    hcn_finalize_body<H, M_>(streams, partials, base, scale, post, comb, y, w, wtype, D, chunksA, rms_eps, hc_eps, sinkhorn_iters, norm_eps, constant_bias, constant_scale, blockIdx.x);
}

/*

hc_mix_norm one-pass (r35, EXL3_HC_1PASS): the partials and the finalize_norm in ONE launch.
  grid (chunksA, R), NUM_THREADS + 32 threads. Threads 0..NUM_THREADS_A-1 run
  hc_mix_partials_kernel's exact per-thread loop and block reduce (same order, same partials
  stores); the rest idle through it. Every writer fences, then one atomic ticket per row; the
  last block of row r fences (acquire), runs hcn_finalize_body for r, and resets the ticket.
  Bit-identical to the two-launch form.

*/

template <int H, int M_, typename FN_T>
__global__ __launch_bounds__(NUM_THREADS + 32)
void hc_mix_norm_1pass_kernel
(
    const float* __restrict__ streams,
    const FN_T* __restrict__ fn,
    float* partials,                     // (R, chunksA, M + 1): written here, re-read by the last block
    const int chunk_cols,
    int* __restrict__ tickets,           // (R) zeroed
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float* __restrict__ post,
    float* __restrict__ comb,
    half* __restrict__ y,
    const void* __restrict__ w,
    const int wtype,
    const int D,
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters,
    const float norm_eps,
    const float constant_bias,
    const float constant_scale
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int row_len = H * D;
    const int chunksA = gridDim.x;
    const bool pt = threadIdx.x < NUM_THREADS_A;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);
    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int row_len4 = row_len / 4;

    __shared__ float red[NUM_THREADS_A / 32][M + 1];
    __shared__ int last_s;
    if (pt)
    {
        float acc[M + 1];
        #pragma unroll
        for (int k = 0; k <= M; ++k) acc[k] = 0.0f;
        for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS_A)
        {
            float4 s = s4[c];
            acc[M] = fmaf(s.x, s.x, acc[M]);
            acc[M] = fmaf(s.y, s.y, acc[M]);
            acc[M] = fmaf(s.z, s.z, acc[M]);
            acc[M] = fmaf(s.w, s.w, acc[M]);
            #pragma unroll
            for (int j = 0; j < M; ++j)
            {
                float4 wv;
                if constexpr (std::is_same_v<FN_T, half>)
                {
                    int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                    float2 lo = __half22float2(*(const half2*) &pk.x);
                    float2 hi = __half22float2(*(const half2*) &pk.y);
                    wv = make_float4(lo.x, lo.y, hi.x, hi.y);
                }
                else
                    wv = ((const float4*) fn)[(size_t) j * row_len4 + c];
                float d = fmaf(s.x, wv.x, fmaf(s.y, wv.y, fmaf(s.z, wv.z, s.w * wv.w)));
                acc[j] += d;
            }
        }
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        #pragma unroll
        for (int k = 0; k <= M; ++k)
        {
            float v = acc[k];
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0) red[warp][k] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        const int lane = threadIdx.x;
        float* out = partials + ((size_t) r * chunksA + blockIdx.x) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w_ = 0; w_ < NUM_THREADS_A / 32; ++w_)
                v += red[w_][k];
            out[k] = v;
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        last_s = atomicAdd(&tickets[r], 1) == chunksA - 1;
    __syncthreads();
    if (!last_s) return;
    __threadfence();
    if (threadIdx.x == 0) tickets[r] = 0;
    hcn_finalize_body<H, M_>(streams, partials, base, scale, post, comb, y, w, wtype, D, chunksA,
                             rms_eps, hc_eps, sinkhorn_iters, norm_eps, constant_bias, constant_scale, r);
}


/*

hc_pf_apply_mix_norm_1pass (GLM prefill, EXL3_PF_HC_FUSE): residual apply of site A
fused with the hc_mix_norm_1pass of the NEXT site B. The (R, H, D) stream stack is written
once (by the apply) and read once from HBM (by the finalize); the partials phase re-reads
the just-written row L2-hot.

  x[h] <- post_A[h] yA + sum_g comb_A[g, h] x[g]          (hc_apply, in place)
  mix_B = rmsnorm(x.flatten) @ fn_B.T, then finalize_norm (hc_mix_norm_1pass)

Valid ONLY in the single-chunk regime (grid (1, R)): the launcher requires
hc_mix_num_chunks(R, H * D) == 1 and the hc_apply single-chunk grid, i.e. R >= 257
at D = 4096. Phase A replays hc_apply_kernel<4, Y_T, true>'s exact per-thread loop
(same grid, same column assignment, same fmaf order); phase P replays
hc_mix_norm_1pass_kernel<4, 24, FN_T>'s partials loop verbatim, then the same ticket
protocol and hcn_finalize_body. Bit-identical to apply + 1pass.

Requires comb (mHC site); the GatedResidual no-comb apply stays on the unfused path.

*/

template <int H, int M_, typename Y_T, typename FN_T>
__global__ __launch_bounds__(NUM_THREADS + 32)
void hc_pf_apply_mix_norm_1pass_kernel
(
    float* __restrict__ x,               // (R, H, D), updated in place
    const Y_T* __restrict__ ya,          // (R, D) site-A sublayer output
    const float* __restrict__ post_a,    // (R, H) site A
    const float* __restrict__ comb_a,    // (R, H, H) site A
    const FN_T* __restrict__ fn,         // (M, H * D) site-B mixer matrix
    float* __restrict__ partials,        // (R, 1, M + 1)
    int* __restrict__ tickets,           // (R) zeroed
    const float* __restrict__ base,      // site B
    const float* __restrict__ scale,     // site B
    float* __restrict__ post,            // site B
    float* __restrict__ comb,            // site B
    half* __restrict__ y,                // site B
    const void* __restrict__ w,
    const int wtype,
    const int D,
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters,
    const float norm_eps,
    const float constant_bias,
    const float constant_scale
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int row_len = H * D;
    const int row_len4 = row_len / 4;
    const int D4 = D / 4;

    // ---- Phase A: hc_apply_kernel<4, Y_T, true> verbatim (single chunk: c0 = 0, c1 = D).
    if (threadIdx.x < NUM_THREADS)
    {
        float post_r[H];
        float comb_r[H][H];
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            post_r[h] = __ldg(post_a + (size_t) r * H + h);
            #pragma unroll
            for (int g = 0; g < H; ++g)
                comb_r[h][g] = __ldg(comb_a + ((size_t) r * H + h) * H + g);
        }

        float4* x4 = (float4*) (x + (size_t) r * row_len);
        for (int c = threadIdx.x; c < D4; c += NUM_THREADS)
        {
            float4 xv[H];
            #pragma unroll
            for (int h = 0; h < H; ++h)
                xv[h] = x4[(size_t) h * D4 + c];

            float4 yv;
            if constexpr (std::is_same_v<Y_T, half>)
            {
                half2 y01 = ((const half2*) (ya + (size_t) r * D))[c * 2];
                half2 y23 = ((const half2*) (ya + (size_t) r * D))[c * 2 + 1];
                float2 lo = __half22float2(y01);
                float2 hi = __half22float2(y23);
                yv = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            else
                yv = ((const float4*) (ya + (size_t) r * D))[c];

            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                float4 o;
                o.x = post_r[h] * yv.x;
                o.y = post_r[h] * yv.y;
                o.z = post_r[h] * yv.z;
                o.w = post_r[h] * yv.w;
                #pragma unroll
                for (int g = 0; g < H; ++g)
                {
                    o.x = fmaf(comb_r[g][h], xv[g].x, o.x);
                    o.y = fmaf(comb_r[g][h], xv[g].y, o.y);
                    o.z = fmaf(comb_r[g][h], xv[g].z, o.z);
                    o.w = fmaf(comb_r[g][h], xv[g].w, o.w);
                }
                x4[(size_t) h * D4 + c] = o;
            }
        }
    }
    __threadfence_block();
    __syncthreads();

    // ---- Phase P+T: hc_mix_norm_1pass_kernel<4, 24, FN_T> verbatim (chunksA == 1).
    const float* sro = (const float*) x;
    const float4* s4 = (const float4*) (sro + (size_t) r * row_len);
    const bool pt = threadIdx.x < NUM_THREADS_A;

    __shared__ float red[NUM_THREADS_A / 32][M + 1];
    __shared__ int last_s;
    if (pt)
    {
        float acc[M + 1];
        #pragma unroll
        for (int k = 0; k <= M; ++k) acc[k] = 0.0f;
        for (int c = threadIdx.x; c < row_len4; c += NUM_THREADS_A)
        {
            float4 s = s4[c];
            acc[M] = fmaf(s.x, s.x, acc[M]);
            acc[M] = fmaf(s.y, s.y, acc[M]);
            acc[M] = fmaf(s.z, s.z, acc[M]);
            acc[M] = fmaf(s.w, s.w, acc[M]);
            #pragma unroll
            for (int j = 0; j < M; ++j)
            {
                float4 wv;
                if constexpr (std::is_same_v<FN_T, half>)
                {
                    int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                    float2 lo = __half22float2(*(const half2*) &pk.x);
                    float2 hi = __half22float2(*(const half2*) &pk.y);
                    wv = make_float4(lo.x, lo.y, hi.x, hi.y);
                }
                else
                    wv = ((const float4*) fn)[(size_t) j * row_len4 + c];
                float d = fmaf(s.x, wv.x, fmaf(s.y, wv.y, fmaf(s.z, wv.z, s.w * wv.w)));
                acc[j] += d;
            }
        }
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        #pragma unroll
        for (int k = 0; k <= M; ++k)
        {
            float v = acc[k];
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0) red[warp][k] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        const int lane = threadIdx.x;
        float* out = partials + (size_t) r * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w_ = 0; w_ < NUM_THREADS_A / 32; ++w_)
                v += red[w_][k];
            out[k] = v;
        }
    }
    __threadfence();
    __syncthreads();
    // Single-chunk regime (grid.x == 1): this block is the last of its row.
    if (threadIdx.x == 0)
        last_s = atomicAdd(&tickets[r], 1) == 0;
    __syncthreads();
    if (!last_s) return;
    __threadfence();
    if (threadIdx.x == 0) tickets[r] = 0;
    hcn_finalize_body<H, M_>(sro, partials, base, scale, post, comb, y, w, wtype, D, 1,
                             rms_eps, hc_eps, sinkhorn_iters, norm_eps, constant_bias, constant_scale, r);
}


/*

GatedResidual (Qwen4Exp, the low-rank elementwise cousin of mHC) fused mix, decode form:

  mix(streams (R, H, D) fp32) -> post (R, H), mixed (R, D):
    normed[h] = rmsnorm(streams[h]) * w[h]           (per-STREAM norm, weighted, w incl +1)
    dots      = cat(down, inject) @ normed.flatten   (M = LR + H outputs, LR = low rank ~320)
    t         = silu(dots[:LR] / H)
    post      = 2 sigmoid(dots[LR:] / H)
    g[h, d]   = sigmoid(up[h * D + d, :] @ t)        (per-CHANNEL gate through the low rank)
    mixed[d]  = mean_h g[h, d] * normed[h, d]

  Same two-launch, no-grid-sync, deterministic shape as hc_mix, restructured for the wide low
  rank (LR >> mHC's M = 24, so per-thread accumulator arrays don't fit):
    K1 gr_dots: one block per (fn row | sum-of-squares), computing per-STREAM partial dots
      against the RAW streams -- by linearity the per-stream rms scale applies in K2, and the
      norm WEIGHT is folded into the fn rows at load time.
    K2 gr_finalize: every block re-derives rmr / t redundantly from the K1 output (the mHC
      finalize pattern), then streams its chunk of mixed, evaluating the per-channel up-gate
      inline -- upT is laid out (LR, H * D) so the serial rank loop reads coalesced and stays
      L2-resident at decode R. NOT for large R (the untiled up/down reads defeat the L2);
      the python side runs a plain half-GEMM path for prefill.

  apply_ is hc_apply with no comb: x[h] += post[h] * y.

*/

#define GR_THREADS_A 128

template <int H>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const half* __restrict__ fn,         // (M, H * D) half, norm weight folded in
    float* __restrict__ dots,            // (R, M + 1, H): per-stream dots, row M = sum sq
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;            // fn row, or M for the sum-of-squares row
    const int D4 = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 16-byte fn loads (8 halves) against two 16-byte stream quads
            const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0 = s4[(size_t) h * D4 + 2 * c];
                float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                int4 pk = f8[c];
                half4 w0 = *(half4*) &pk.x;
                half4 w1 = *(half4*) &pk.z;
                a = fmaf(s0.x, LOW_TO_FLOAT(w0.x), a);
                a = fmaf(s0.y, HIGH_TO_FLOAT(w0.x), a);
                a = fmaf(s0.z, LOW_TO_FLOAT(w0.y), a);
                a = fmaf(s0.w, HIGH_TO_FLOAT(w0.y), a);
                a = fmaf(s1.x, LOW_TO_FLOAT(w1.x), a);
                a = fmaf(s1.y, HIGH_TO_FLOAT(w1.x), a);
                a = fmaf(s1.z, LOW_TO_FLOAT(w1.y), a);
                a = fmaf(s1.w, HIGH_TO_FLOAT(w1.y), a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H) from gr_dots
    const half* __restrict__ upt,        // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr (final-mixer form)
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,                        // low rank; M = LR + (post ? H : 0)
    const int chunk_cols,                // multiple of 4
    const float rms_eps
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    // Redundant per-block head derivation (no cross-block dependency): rmr from the sumsq
    // row, then the silu'd low-rank activations into shared memory
    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    // Streamed mixed chunk: one WARP per column quad, the LR-long up-gate dots split across
    // the lanes (lane l covers ranks l, l+32, ...) and shfl-reduced -- at decode R the total
    // column count is small (R * D / 4), so per-thread columns would leave the GPU nearly
    // idle with each thread serializing the rank loop
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = lane; i < LR; i += 32)
        {
            float ti = t_s[i];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                // upx layout (H, D / 4, LR, 4): consecutive lanes (consecutive i) read
                // consecutive 8-byte quads, so the rank loop is fully coalesced
                half4 u = *(const half4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                g[h].x = fmaf(ti, LOW_TO_FLOAT(u.x), g[h].x);
                g[h].y = fmaf(ti, HIGH_TO_FLOAT(u.x), g[h].y);
                g[h].z = fmaf(ti, LOW_TO_FLOAT(u.y), g[h].z);
                g[h].w = fmaf(ti, HIGH_TO_FLOAT(u.y), g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x) * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z) * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}


// Residual update for one sublayer site: x[h, d] <- post[h] * y[d] + sum_h' comb[h', h] *
// x[h', d]. Without comb (GatedResidual): x[h, d] <- post[h] * y[d] + x[h, d]. Pure per-column
// mix of the H stream rows, so it runs in place: each thread loads all H values of its columns
// into registers before writing any back.

template <int H, typename Y_T, bool HAS_COMB>
__global__ __launch_bounds__(NUM_THREADS)
void hc_apply_kernel
(
    float* __restrict__ x,               // (R, H, D), updated in place
    const Y_T* __restrict__ y,           // (R, D) float or half
    const float* __restrict__ post,      // (R, H)
    const float* __restrict__ comb,      // (R, H, H), or null
    const int D,
    const int chunk_cols                 // multiple of 4
)
{
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;

    float post_r[H];
    float comb_r[H][H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        post_r[h] = __ldg(post + (size_t) r * H + h);
        if (HAS_COMB)
        {
            #pragma unroll
            for (int g = 0; g < H; ++g)
                comb_r[h][g] = __ldg(comb + ((size_t) r * H + h) * H + g);
        }
    }

    float4* x4 = (float4*) (x + (size_t) r * H * D);
    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS)
    {
        float4 xv[H];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            xv[h] = x4[(size_t) h * D4 + c];

        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            half2 y01 = ((const half2*) (y + (size_t) r * D))[c * 2];
            half2 y23 = ((const half2*) (y + (size_t) r * D))[c * 2 + 1];
            float2 lo = __half22float2(y01);
            float2 hi = __half22float2(y23);
            yv = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        else
            yv = ((const float4*) (y + (size_t) r * D))[c];

        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 o;
            o.x = post_r[h] * yv.x;
            o.y = post_r[h] * yv.y;
            o.z = post_r[h] * yv.z;
            o.w = post_r[h] * yv.w;
            if (HAS_COMB)
            {
                #pragma unroll
                for (int g = 0; g < H; ++g)
                {
                    o.x = fmaf(comb_r[g][h], xv[g].x, o.x);
                    o.y = fmaf(comb_r[g][h], xv[g].y, o.y);
                    o.z = fmaf(comb_r[g][h], xv[g].z, o.z);
                    o.w = fmaf(comb_r[g][h], xv[g].w, o.w);
                }
            }
            else
            {
                o.x += xv[h].x;
                o.y += xv[h].y;
                o.z += xv[h].z;
                o.w += xv[h].w;
            }
            x4[(size_t) h * D4 + c] = o;
        }
    }
}


#if defined(USE_ROCM)

/*
Row-looped GatedResidual mix for small-CU parts (gfx1151: 20 CUs, 236 GB/s). The reference
gr_dots / gr_finalize launch a (.., R) grid and re-read the 6.3 MiB fn and 6.2 MiB upt once
PER ROW, and gr_dots' one-block-per-fn-row shape (128 threads, ~10 dependent load rounds,
per-h serial chains) ran at ~46 GB/s. Measured 211 us at R=1 / 233 us at R=3 for a 56 us
one-read roofline; at ~17% of decode device time under MTP this was the second-largest
kernel after the grouped MoE.

  gr_dots_rows:     warp per (fn row j, stream h) dot product, lanes stride the D axis with
                    16-byte fn loads, ALL R rows accumulated by the same warp so fn is read
                    once; the sum-of-squares rows ride along as j == M.
  gr_finalize_rows: grid over column chunks only; every block derives rmr / t for all R
                    rows redundantly (tiny), then the warp-per-column-quad rank loop keeps
                    R x H float4 accumulators so upt is read once for all rows.

Numerics: same fp32 accumulate, different summation order -> not bit-identical to the
reference kernels, same class of error (parity checked against _mix_ref in gr_mix_micro.py).
EXL3_HIP_GR_MIX_ROWS=1 enables it (default off, see gr_mix_rows_enabled).
*/

#define GR_ROWS_WARPS 8
#define GR_ROWS_FN_MAX 10        // D <= 10 * 256 = 2560 on the rows path (VGPR budget)
#define GR_ROWS_LR_TILE 5        // ranks per lane hoisted per tile (LR = 320 -> 2 tiles)
#define GR_ROWS_THREADS (GR_ROWS_WARPS * 32)

template <int H, int RMAX>
__global__ __launch_bounds__(GR_ROWS_THREADS)
void gr_dots_rows_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const half* __restrict__ fn,         // (M, H * D) half, norm weight folded in
    float* __restrict__ dots,            // (R, M + 1, H)
    const int R,
    const int M,
    const int D
)
{
    // Block = one stream h x GR_ROWS_WARPS consecutive fn rows j. The block stages
    // streams[0..R, h, :] in LDS once (R x D fp32), so the per-fn-element stream reads
    // that dominated the L2 traffic of the per-row layout (every warp re-read its 10 KiB
    // slice: 13 MiB of L2 reads per call at R=1, 39 MiB at R=3) come from LDS instead.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int h = blockIdx.y;
    const int j = blockIdx.x * GR_ROWS_WARPS + warp;
    const size_t stream_stride = (size_t) H * D;

    extern __shared__ float4 s_stream4[];             // [RMAX][D / 4]
    const int D4 = D / 4;
    for (int idx = threadIdx.x; idx < R * D4; idx += GR_ROWS_THREADS)
    {
        const int r = idx / D4, c = idx % D4;
        s_stream4[r * D4 + c] = ((const float4*) (streams + (size_t) r * stream_stride + (size_t) h * D))[c];
    }
    __syncthreads();
    if (j > M) return;

    float acc[RMAX];
    #pragma unroll
    for (int r = 0; r < RMAX; ++r) acc[r] = 0.0f;

    if (j < M)
    {
        // fn is the DRAM traffic: issue all of this lane's 16-byte fn loads up front
        // (D / 256 per lane, <= GR_ROWS_FN_MAX) so they overlap, then consume from LDS.
        const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
        const int steps = D / 256;                 // D % 256 == 0 checked by the launcher
        int4 pk[GR_ROWS_FN_MAX];
        #pragma unroll
        for (int t = 0; t < GR_ROWS_FN_MAX; ++t)
            if (t < steps) pk[t] = f8[lane + t * 32];
        #pragma unroll
        for (int t = 0; t < GR_ROWS_FN_MAX; ++t)
        {
            if (t >= steps) break;
            const int c = lane + t * 32;               // 8-element chunk index
            const half4 w0 = *(const half4*) &pk[t].x;
            const half4 w1 = *(const half4*) &pk[t].z;
            const float f0 = LOW_TO_FLOAT(w0.x), f1 = HIGH_TO_FLOAT(w0.x);
            const float f2 = LOW_TO_FLOAT(w0.y), f3 = HIGH_TO_FLOAT(w0.y);
            const float f4 = LOW_TO_FLOAT(w1.x), f5 = HIGH_TO_FLOAT(w1.x);
            const float f6 = LOW_TO_FLOAT(w1.y), f7 = HIGH_TO_FLOAT(w1.y);
            #pragma unroll
            for (int r = 0; r < RMAX; ++r)
            {
                if (r < R)
                {
                    const float4 s0 = s_stream4[r * D4 + 2 * c];
                    const float4 s1 = s_stream4[r * D4 + 2 * c + 1];
                    float a = acc[r];
                    a = fmaf(s0.x, f0, a); a = fmaf(s0.y, f1, a);
                    a = fmaf(s0.z, f2, a); a = fmaf(s0.w, f3, a);
                    a = fmaf(s1.x, f4, a); a = fmaf(s1.y, f5, a);
                    a = fmaf(s1.z, f6, a); a = fmaf(s1.w, f7, a);
                    acc[r] = a;
                }
            }
        }
    }
    else
    {
        #pragma unroll
        for (int r = 0; r < RMAX; ++r)
        {
            if (r < R)
            {
                float a = 0.0f;
                for (int c = lane; c < D4; c += 32)
                {
                    const float4 sv = s_stream4[r * D4 + c];
                    a = fmaf(sv.x, sv.x, fmaf(sv.y, sv.y, fmaf(sv.z, sv.z, fmaf(sv.w, sv.w, a))));
                }
                acc[r] = a;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < RMAX; ++r)
    {
        float a = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_xor_sync(0xffffffffu, a, offset);
        acc[r] = a;
    }
    if (lane == 0)
    {
        #pragma unroll
        for (int r = 0; r < RMAX; ++r)
            if (r < R) dots[((size_t) r * (M + 1) + j) * H + h] = acc[r];
    }
}

template <int H, int RMAX, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_rows_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H)
    const half* __restrict__ upt,        // (H, D / 4, LR, 4) half
    const half* __restrict__ w,          // (H * D) half
    float* __restrict__ post,            // (R, H) or nullptr
    void* __restrict__ mixed,            // (R, D) half or float
    const int R,
    const int D,
    const int LR,
    const int chunk_cols,
    const float rms_eps
)
{
    const int M = LR + (post ? H : 0);
    const float inv_h = 1.0f / (float) H;

    __shared__ float rmr_s[RMAX][H];
    extern __shared__ float t_s[];       // [RMAX][LR]
    if (threadIdx.x < RMAX * H)
    {
        const int r = threadIdx.x / H, h = threadIdx.x % H;
        rmr_s[r][h] = r < R
            ? rsqrtf(dots[((size_t) r * (M + 1) + M) * H + h] / (float) D + rms_eps)
            : 0.0f;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < R * LR; idx += NUM_THREADS)
    {
        const int r = idx / LR, i = idx % LR;
        const float* dr = dots + (size_t) r * (M + 1) * H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[r][h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[r * LR + i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < R * H)
    {
        const int r = threadIdx.x / H, hh = threadIdx.x % H;
        const float* dr = dots + (size_t) r * (M + 1) * H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[r][h], dr[(size_t) (LR + hh) * H + h], v);
        post[(size_t) r * H + hh] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[RMAX][H];
        #pragma unroll
        for (int r = 0; r < RMAX; ++r)
            #pragma unroll
            for (int h = 0; h < H; ++h) g[r][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // upt is the traffic here: hoist the H loads of a whole tile of GR_ROWS_LR_TILE ranks
        // before consuming, so each lane keeps H * tile 8-byte loads in flight
        for (int i0 = 0; i0 < LR; i0 += 32 * GR_ROWS_LR_TILE)
        {
            half4 u[GR_ROWS_LR_TILE][H];
            #pragma unroll
            for (int t = 0; t < GR_ROWS_LR_TILE; ++t)
            {
                const int i = i0 + t * 32 + lane;
                #pragma unroll
                for (int h = 0; h < H; ++h)
                    u[t][h] = i < LR
                        ? *(const half4*) (upt + ((((size_t) h * D4 + c) * LR + i) * 4))
                        : half4(__float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f));
            }
            #pragma unroll
            for (int t = 0; t < GR_ROWS_LR_TILE; ++t)
            {
            const int i = i0 + t * 32 + lane;
            if (i >= LR) break;
            #pragma unroll
            for (int r = 0; r < RMAX; ++r)
            {
                if (r < R)
                {
                    const float ti = t_s[r * LR + i];
                    #pragma unroll
                    for (int h = 0; h < H; ++h)
                    {
                        g[r][h].x = fmaf(ti, LOW_TO_FLOAT(u[t][h].x), g[r][h].x);
                        g[r][h].y = fmaf(ti, HIGH_TO_FLOAT(u[t][h].x), g[r][h].y);
                        g[r][h].z = fmaf(ti, LOW_TO_FLOAT(u[t][h].y), g[r][h].z);
                        g[r][h].w = fmaf(ti, HIGH_TO_FLOAT(u[t][h].y), g[r][h].w);
                    }
                }
            }
            }
        }
        #pragma unroll
        for (int r = 0; r < RMAX; ++r)
            #pragma unroll
            for (int h = 0; h < H; ++h)
                for (int offset = 16; offset > 0; offset >>= 1)
                {
                    g[r][h].x += __shfl_xor_sync(0xffffffffu, g[r][h].x, offset);
                    g[r][h].y += __shfl_xor_sync(0xffffffffu, g[r][h].y, offset);
                    g[r][h].z += __shfl_xor_sync(0xffffffffu, g[r][h].z, offset);
                    g[r][h].w += __shfl_xor_sync(0xffffffffu, g[r][h].w, offset);
                }

        // lane r finishes row r (every lane holds the full sums after the xor reduce)
        #pragma unroll
        for (int r = 0; r < RMAX; ++r)
        {
            if (r < R && lane == r)
            {
                const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
                float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                #pragma unroll
                for (int h = 0; h < H; ++h)
                {
                    const float4 sv = s4[(size_t) h * D4 + c];
                    const half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
                    const float coef = rmr_s[r][h] * inv_h;
                    o.x = fmaf(sigmoidf_(g[r][h].x) * coef * LOW_TO_FLOAT(wq.x),  sv.x, o.x);
                    o.y = fmaf(sigmoidf_(g[r][h].y) * coef * HIGH_TO_FLOAT(wq.x), sv.y, o.y);
                    o.z = fmaf(sigmoidf_(g[r][h].z) * coef * LOW_TO_FLOAT(wq.y),  sv.z, o.z);
                    o.w = fmaf(sigmoidf_(g[r][h].w) * coef * HIGH_TO_FLOAT(wq.y), sv.w, o.w);
                }
                if (HALF_OUT)
                {
                    half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
                    out2[c * 2] = __floats2half2_rn(o.x, o.y);
                    out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
                }
                else
                    ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
            }
        }
    }
}

static bool gr_mix_rows_enabled()
{
    static int cached = -1;
    if (cached < 0)
    {
        // Default OFF: measured a null on gfx1151 (in-model gr_dots already ran at ~88 GB/s,
        // the practical single-kernel read rate for a 6 MiB working set on this part; the
        // row-looped version was 42.0 vs 44.0 tok/s end to end). Kept for parts with more CUs.
        const char* e = getenv("EXL3_HIP_GR_MIX_ROWS");
        cached = (e && *e == '1') ? 1 : 0;
    }
    return cached == 1;
}

template <int RMAX>
static void gr_mix_rows_launch
(
    const float* streams, const half* fn, const half* upt, const half* w, float* dots, float* post,
    void* mixed, bool half_out, int R, int M, int D, int LR, float rms_eps, cudaStream_t stream
)
{
    dim3 grid_a((M + 1 + GR_ROWS_WARPS - 1) / GR_ROWS_WARPS, 4);
    const int smem_a = R * D * sizeof(float);
    gr_dots_rows_kernel<4, RMAX><<<grid_a, GR_ROWS_THREADS, smem_a, stream>>>(streams, fn, dots, R, M, D);
    cuda_check(cudaPeekAtLastError());

    // One column quad per warp per pass; spread the quads over as many blocks as the device
    // can hold rather than tying the grid to R
    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, 512));
    int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;
    dim3 grid_c(n_chunks);
    int smem = RMAX * LR * sizeof(float);
    if (half_out)
        gr_finalize_rows_kernel<4, RMAX, true><<<grid_c, NUM_THREADS, smem, stream>>>
            (streams, dots, upt, w, post, mixed, R, D, LR, chunk_cols, rms_eps);
    else
        gr_finalize_rows_kernel<4, RMAX, false><<<grid_c, NUM_THREADS, smem, stream>>>
            (streams, dots, upt, w, post, mixed, R, D, LR, chunk_cols, rms_eps);
    cuda_check(cudaPeekAtLastError());
}

#endif // USE_ROCM

// Shared launch logic. mode: fn rows M = 2H + H^2 (mix) or H (head)

bool hc_mix_supported(int device)
{
#if defined(USE_ROCM)
    constexpr int MAX_CACHED_DEVICES = 128;
    if (device < 0 || device >= MAX_CACHED_DEVICES) return false;
    static std::atomic<int> support_cache[MAX_CACHED_DEVICES];  // 0 unknown, 1 false, 2 true
    int cached = support_cache[device].load(std::memory_order_acquire);
    if (cached) return cached == 2;
    hipDeviceProp_t prop;
    bool supported = hipGetDeviceProperties(&prop, device) == hipSuccess && prop.warpSize == 32;
    support_cache[device].store(supported ? 2 : 1, std::memory_order_release);
    return supported;
#else
    (void) device;
    return true;
#endif
}

static void hc_check_common
(
    const at::Tensor& tensor,
    const at::Device& device,
    const char* op,
    const char* name
)
{
    TORCH_CHECK(tensor.is_cuda(), op, ": ", name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, op, ": all tensors must be on the same device (", name, ")");
    TORCH_CHECK(tensor.is_contiguous(), op, ": ", name, " must be contiguous");
}

static void hc_check_aligned(const at::Tensor& tensor, const char* op, const char* name)
{
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
        op, ": ", name, " must be 16-byte aligned"
    );
}

static void hc_mix_launch
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    float rms_eps,
    float hc_eps,
    int sinkhorn_iters,
    at::Tensor& partials,
    at::Tensor* post,
    at::Tensor* comb,
    at::Tensor& collapsed,
    bool head,
    Graph* graph
)
{
    const char* op = head ? "hc_head" : "hc_mix";
    TORCH_CHECK(streams.is_cuda(), op, ": streams must be a CUDA tensor");
    TORCH_CHECK(streams.dim() == 3, op, ": streams must have shape (R, 4, D)");
    const int R = streams.size(0);
    const int H = streams.size(1);
    const int D = streams.size(2);
    TORCH_CHECK(H == 4 && D > 0 && D % 4 == 0, op, ": streams must have shape (R, 4, D) with D divisible by 4");
    const int row_len = H * D;
    const int M = head ? H : 2 * H + H * H;
    const at::Device device = streams.device();

    hc_check_common(streams, device, op, "streams");
    hc_check_common(fn, device, op, "fn");
    hc_check_common(base, device, op, "base");
    hc_check_common(scale, device, op, "scale");
    hc_check_common(partials, device, op, "partials");
    hc_check_common(collapsed, device, op, "collapsed");
    TORCH_CHECK(streams.scalar_type() == at::kFloat, op, ": streams must have dtype float32");
    TORCH_CHECK(fn.scalar_type() == at::kFloat || fn.scalar_type() == at::kHalf,
                op, ": fn must have dtype float32 or float16");
    TORCH_CHECK(base.scalar_type() == at::kFloat, op, ": base must have dtype float32");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, op, ": scale must have dtype float32");
    TORCH_CHECK(partials.scalar_type() == at::kFloat, op, ": partials must have dtype float32");
    TORCH_CHECK(collapsed.scalar_type() == at::kFloat || collapsed.scalar_type() == at::kHalf,
                op, ": collapsed must have dtype float32 or float16");
    TORCH_CHECK(fn.dim() == 2 && fn.size(0) == M && fn.size(1) == row_len,
                op, ": fn must have shape (", M, ", ", row_len, ")");
    TORCH_CHECK(base.dim() == 1 && base.size(0) == M, op, ": base must have shape (", M, ")");
    const int scale_size = head ? 1 : 3;
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == scale_size,
                op, ": scale must have shape (", scale_size, ")");
    const int n_chunks_a = hc_mix_num_chunks(R, row_len);
    TORCH_CHECK(partials.dim() == 3 && partials.size(0) == R &&
                partials.size(1) >= n_chunks_a && partials.size(2) == M + 1,
                op, ": partials must have shape (R, at least ", n_chunks_a, ", ", M + 1, ")");
    TORCH_CHECK(collapsed.dim() == 2 && collapsed.size(0) == R && collapsed.size(1) == D,
                op, ": collapsed must have shape (R, D)");
    if (head)
    {
        TORCH_CHECK(post == nullptr && comb == nullptr, "hc_head: post and comb must be absent");
    }
    else
    {
        TORCH_CHECK(post != nullptr && comb != nullptr, "hc_mix: post and comb outputs are required");
        hc_check_common(*post, device, op, "post");
        hc_check_common(*comb, device, op, "comb");
        TORCH_CHECK(post->scalar_type() == at::kFloat && post->dim() == 2 &&
                    post->size(0) == R && post->size(1) == H,
                    "hc_mix: post must be float32 with shape (R, 4)");
        TORCH_CHECK(comb->scalar_type() == at::kFloat && comb->dim() == 3 &&
                    comb->size(0) == R && comb->size(1) == H && comb->size(2) == H,
                    "hc_mix: comb must be float32 with shape (R, 4, 4)");
        TORCH_CHECK(sinkhorn_iters >= 1, "hc_mix: sinkhorn_iters must be at least 1");
    }

    hc_check_aligned(streams, op, "streams");
    hc_check_aligned(fn, op, "fn");
    hc_check_aligned(collapsed, op, "collapsed");
    TORCH_CHECK(hc_mix_supported(device.index()), op, ": device must use a 32-lane warp/wavefront");
    if (R == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(device);
#if defined(USE_ROCM)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#else
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
#endif

    const bool fn_half = fn.dtype() == at::kHalf;
    const int target_cols_a = (row_len + n_chunks_a - 1) / n_chunks_a;
    int chunk_cols = ((target_cols_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    int chunks_c = std::min(32, std::max(1, 256 / R));
    const int target_cols_c = (D + chunks_c - 1) / chunks_c;
    int chunk_cols_c = ((target_cols_c + 4 * NUM_THREADS - 1) / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks_c = (D + chunk_cols_c - 1) / chunk_cols_c;

    bool half_out = collapsed.dtype() == at::kHalf;

    dim3 grid_a(n_chunks_a, R);
    dim3 grid_c(n_chunks_c, R);
    #define ARGS_A(FN_T) \
        (const float*) streams.data_ptr(), (const FN_T*) fn.data_ptr(), \
        (float*) partials.data_ptr(), row_len, chunk_cols
    #define ARGS_C(POST, COMB) \
        (const float*) streams.data_ptr(), (const float*) partials.data_ptr(), \
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(), \
        POST, COMB, collapsed.data_ptr(), \
        D, n_chunks_a, chunk_cols_c, rms_eps, hc_eps, sinkhorn_iters
    if (!head)
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 24, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 24, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        float* post_p = (float*) post->data_ptr();
        float* comb_p = (float*) comb->data_ptr();
        if (half_out)
            hc_mix_finalize_kernel<4, 24, false, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
        else
            hc_mix_finalize_kernel<4, 24, false, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
    }
    else
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 4, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 4, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        if (half_out)
            hc_mix_finalize_kernel<4, 4, true, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
        else
            hc_mix_finalize_kernel<4, 4, true, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
    }
    #undef ARGS_A
    #undef ARGS_C
    cuda_check(cudaPeekAtLastError());
}

int hc_mix_num_chunks(int R, int row_len)
{
    TORCH_CHECK(R >= 0 && row_len > 0, "hc_mix_num_chunks: R must be nonnegative and row_len must be positive");
    int chunks_a = std::min(128, std::max(1, 512 / std::max(R, 1)));
    const int target_cols = (row_len + chunks_a - 1) / chunks_a;
    int chunk_cols = ((target_cols + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    return (row_len + chunk_cols - 1) / chunk_cols;
}

void hc_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (2H + H^2, H * D) float
    const at::Tensor& base,              // (2H + H^2) float
    const at::Tensor& scale,             // (3) float
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,                 // (R, chunks, M + 1) float workspace
    at::Tensor post,                     // (R, H) float out
    at::Tensor comb,                     // (R, H, H) float out
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        (int) sinkhorn_iters,
        partials,
        &post,
        &comb,
        collapsed,
        false,
        nullptr
    );
}

// One-pass hc_mix_norm (r35): per-device row tickets, zeroed at allocation (first eager call,
// never under capture); runtime-settable via hc_mix_set_onepass for in-process A/B.
#define HC_MAX_TICKETS 4096
#define HC_DEC_ROWS 64
static int g_hc_onepass = -1;
static bool hc_onepass(int R)
{
    // Prefill rows (R > 64): one-pass behind EXL3_HC_1PASS_PF, default OFF, read per call so
    // one loaded process can A/B both arms. Decode rows keep the r35 default (on).
    if (R > HC_DEC_ROWS)
    {
        const char* p = getenv("EXL3_HC_1PASS_PF");
        return p && *p && *p != '0';
    }
    if (g_hc_onepass < 0)
    {
        const char* e = getenv("EXL3_HC_1PASS");
        g_hc_onepass = (e && *e == '0') ? 0 : 1;
    }
    return g_hc_onepass == 1;
}
static int* hc_tickets(int device)
{
    static int* t[128] = {};
    if (device < 0 || device >= 128) return nullptr;
    if (!t[device])
    {
        cuda_check(cudaMalloc((void**) &t[device], HC_MAX_TICKETS * sizeof(int)));
        cuda_check(cudaMemset(t[device], 0, HC_MAX_TICKETS * sizeof(int)));
        cuda_check(cudaDeviceSynchronize());
    }
    return t[device];
}
void hc_mix_set_onepass(int64_t on) { g_hc_onepass = on ? 1 : 0; }

void hc_mix_norm
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (2H + H^2, H * D) float or half
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,                 // (R, chunks, M + 1) float workspace
    at::Tensor post,                     // (R, H) float out
    at::Tensor comb,                     // (R, H, H) float out
    at::Tensor y,                        // (R, D) half out: rms_norm(half(collapsed))
    const c10::optional<at::Tensor>& w,  // (D) half / bf16 norm weight
    double norm_eps,
    double constant_bias,
    double constant_scale
)
{
    const char* op = "hc_mix_norm";
    TORCH_CHECK(streams.is_cuda() && streams.dim() == 3 && streams.size(1) == 4, op, ": streams must be (R, 4, D)");
    const int R = streams.size(0);
    const int H = 4;
    const int D = streams.size(2);
    const int M = 2 * H + H * H;
    TORCH_CHECK(D > 0 && D % 4 == 0 && D <= NUM_THREADS * HCN_MAX, op, ": unsupported D");
    const at::Device device = streams.device();
    hc_check_common(streams, device, op, "streams");
    hc_check_common(fn, device, op, "fn");
    hc_check_common(base, device, op, "base");
    hc_check_common(scale, device, op, "scale");
    hc_check_common(partials, device, op, "partials");
    hc_check_common(post, device, op, "post");
    hc_check_common(comb, device, op, "comb");
    hc_check_common(y, device, op, "y");
    TORCH_CHECK(streams.scalar_type() == at::kFloat && base.scalar_type() == at::kFloat &&
                scale.scalar_type() == at::kFloat && partials.scalar_type() == at::kFloat &&
                post.scalar_type() == at::kFloat && comb.scalar_type() == at::kFloat, op, ": dtype");
    TORCH_CHECK(fn.scalar_type() == at::kFloat || fn.scalar_type() == at::kHalf, op, ": fn dtype");
    TORCH_CHECK(y.scalar_type() == at::kHalf && y.numel() == (int64_t) R * D, op, ": y must be half (R, D)");
    TORCH_CHECK(fn.dim() == 2 && fn.size(0) == M && fn.size(1) == H * D, op, ": fn shape");
    TORCH_CHECK(base.numel() == M && scale.numel() == 3 && post.numel() == R * H && comb.numel() == R * H * H, op, ": shapes");
    const int n_chunks_a = hc_mix_num_chunks(R, H * D);
    TORCH_CHECK(partials.dim() == 3 && partials.size(0) == R && partials.size(1) >= n_chunks_a &&
                partials.size(2) == M + 1, op, ": partials shape");
    TORCH_CHECK(sinkhorn_iters >= 1, op, ": sinkhorn_iters");
    int wtype = 0;
    if (w.has_value() && w->defined())
    {
        TORCH_CHECK(w->numel() == D && w->is_contiguous() && w->device() == device, op, ": weight shape");
        TORCH_CHECK(w->dtype() == at::kHalf || w->dtype() == at::kBFloat16, op, ": weight must be fp16/bf16");
        wtype = w->dtype() == at::kHalf ? 1 : 2;
    }
    hc_check_aligned(streams, op, "streams");
    hc_check_aligned(fn, op, "fn");
    TORCH_CHECK(hc_mix_supported(device.index()), op, ": device must use a 32-lane warp/wavefront");
    if (R == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int row_len = H * D;
    const int target_cols_a = (row_len + n_chunks_a - 1) / n_chunks_a;
    int chunk_cols = ((target_cols_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    dim3 grid_a(n_chunks_a, R);
    int* tickets = hc_tickets(device.index());
    if (tickets && R <= HC_MAX_TICKETS && hc_onepass(R))
    {
        #define HC1_ARGS(FT) (const float*) streams.data_ptr(), (const FT*) fn.data_ptr(), (float*) partials.data_ptr(), \
            chunk_cols, tickets, (const float*) base.data_ptr(), (const float*) scale.data_ptr(), \
            (float*) post.data_ptr(), (float*) comb.data_ptr(), (half*) y.data_ptr(), \
            wtype ? w->data_ptr() : nullptr, wtype, D, (float) rms_eps, (float) hc_eps, (int) sinkhorn_iters, \
            (float) norm_eps, (float) constant_bias, (float) constant_scale
        if (fn.dtype() == at::kHalf)
            hc_mix_norm_1pass_kernel<4, 24, half><<<grid_a, NUM_THREADS + 32, 0, stream>>>(HC1_ARGS(half));
        else
            hc_mix_norm_1pass_kernel<4, 24, float><<<grid_a, NUM_THREADS + 32, 0, stream>>>(HC1_ARGS(float));
        #undef HC1_ARGS
        cuda_check(cudaPeekAtLastError());
        return;
    }
    if (fn.dtype() == at::kHalf)
        hc_mix_partials_kernel<4, 24, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(
            (const float*) streams.data_ptr(), (const half*) fn.data_ptr(), (float*) partials.data_ptr(), row_len, chunk_cols);
    else
        hc_mix_partials_kernel<4, 24, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(
            (const float*) streams.data_ptr(), (const float*) fn.data_ptr(), (float*) partials.data_ptr(), row_len, chunk_cols);
    cuda_check(cudaPeekAtLastError());
    hc_mix_finalize_norm_kernel<4, 24><<<R, NUM_THREADS + 32, 0, stream>>>(
        (const float*) streams.data_ptr(), (const float*) partials.data_ptr(),
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(),
        (float*) post.data_ptr(), (float*) comb.data_ptr(), (half*) y.data_ptr(),
        wtype ? w->data_ptr() : nullptr, wtype, D, n_chunks_a,
        (float) rms_eps, (float) hc_eps, (int) sinkhorn_iters,
        (float) norm_eps, (float) constant_bias, (float) constant_scale);
    cuda_check(cudaPeekAtLastError());
}

void hc_head
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (H, H * D) float
    const at::Tensor& base,              // (H) float
    const at::Tensor& scale,             // (1) float
    double rms_eps,
    double hc_eps,
    at::Tensor partials,                 // (R, chunks, H + 1) float workspace
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        0,
        partials,
        nullptr,
        nullptr,
        collapsed,
        true,
        nullptr
    );
}

void hc_apply
(
    at::Tensor x,                        // (R, H, D) float, updated IN PLACE
    const at::Tensor& y,                 // (R, D) float or half
    const at::Tensor& post,              // (R, H) float
    const c10::optional<at::Tensor>& comb   // (R, H, H) float, or none (x[h] += post[h] * y)
)
{
    TORCH_CHECK(x.is_cuda(), "hc_apply: x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "hc_apply: x must have shape (R, 4, D)");
    const int R = x.size(0);
    const int H = x.size(1);
    const int D = x.size(2);
    TORCH_CHECK(H == 4 && D > 0 && D % 4 == 0,
                "hc_apply: x must have shape (R, 4, D) with D divisible by 4");
    const at::Device device = x.device();
    hc_check_common(x, device, "hc_apply", "x");
    hc_check_common(y, device, "hc_apply", "y");
    hc_check_common(post, device, "hc_apply", "post");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "hc_apply: x must have dtype float32");
    TORCH_CHECK(y.scalar_type() == at::kFloat || y.scalar_type() == at::kHalf,
                "hc_apply: y must have dtype float32 or float16");
    TORCH_CHECK(post.scalar_type() == at::kFloat, "hc_apply: post must have dtype float32");
    TORCH_CHECK(y.dim() == 2 && y.size(0) == R && y.size(1) == D,
                "hc_apply: y must have shape (R, D)");
    TORCH_CHECK(post.dim() == 2 && post.size(0) == R && post.size(1) == H,
                "hc_apply: post must have shape (R, 4)");
    if (comb)
    {
        hc_check_common(comb.value(), device, "hc_apply", "comb");
        TORCH_CHECK(comb.value().scalar_type() == at::kFloat && comb.value().dim() == 3 &&
                    comb.value().size(0) == R && comb.value().size(1) == H && comb.value().size(2) == H,
                    "hc_apply: comb must be float32 with shape (R, 4, 4)");
    }
    hc_check_aligned(x, "hc_apply", "x");
    hc_check_aligned(y, "hc_apply", "y");
    TORCH_CHECK(hc_mix_supported(device.index()),
                "hc_apply: device must use a 32-lane warp/wavefront");
    if (R == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const float* comb_p = comb ? (const float*) comb.value().data_ptr() : nullptr;

    int chunks_c = std::min(32, std::max(1, 256 / R));
    const int target_cols = (D + chunks_c - 1) / chunks_c;
    int chunk_cols = ((target_cols + 4 * NUM_THREADS - 1)
                      / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;

    dim3 grid(n_chunks, R);
    #define ARGS(Y_T) \
        (float*) x.data_ptr(), (const Y_T*) y.data_ptr(), \
        (const float*) post.data_ptr(), comb_p, D, chunk_cols
    if (y.dtype() == at::kHalf)
    {
        if (comb_p) hc_apply_kernel<4, half, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
        else        hc_apply_kernel<4, half, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
    }
    else
    {
        if (comb_p) hc_apply_kernel<4, float, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
        else        hc_apply_kernel<4, float, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
    }
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}

// Prefill apply+mix_norm fusion (EXL3_PF_HC_FUSE): site-A hc_apply with comb
// followed by the site-B hc_mix_norm_1pass, one launch. Single-chunk regime only
// (grid (1, R)); the caller sizes partials (R, 1, M + 1).
void hc_pf_apply_mix_norm
(
    at::Tensor x,                       // (R, 4, D) float, updated IN PLACE
    const at::Tensor& ya,               // (R, D) float or half: site-A sublayer output
    const at::Tensor& post_a,           // (R, 4) float
    const at::Tensor& comb_a,           // (R, 4, 4) float (required)
    const at::Tensor& fn,               // (24, 4 * D) float or half: site-B mixer
    const at::Tensor& base,             // (24) float
    const at::Tensor& scale,            // (3) float
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,                // (R, 1, 25) float workspace
    at::Tensor post,                    // (R, 4) float out
    at::Tensor comb,                    // (R, 4, 4) float out
    at::Tensor y,                       // (R, D) half out
    const c10::optional<at::Tensor>& w, // (D) half / bf16 norm weight
    double norm_eps,
    double constant_bias,
    double constant_scale
)
{
    const char* op = "hc_pf_apply_mix_norm";
    TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.size(1) == 4, op, ": x must be (R, 4, D)");
    const int R = x.size(0);
    const int H = 4;
    const int D = x.size(2);
    const int M = 2 * H + H * H;
    TORCH_CHECK(H == 4 && D > 0 && D % 4 == 0 && D <= NUM_THREADS * HCN_MAX, op, ": unsupported D");
    const at::Device device = x.device();
    hc_check_common(x, device, op, "x");
    hc_check_common(ya, device, op, "ya");
    hc_check_common(post_a, device, op, "post_a");
    hc_check_common(comb_a, device, op, "comb_a");
    hc_check_common(fn, device, op, "fn");
    hc_check_common(base, device, op, "base");
    hc_check_common(scale, device, op, "scale");
    hc_check_common(partials, device, op, "partials");
    hc_check_common(post, device, op, "post");
    hc_check_common(comb, device, op, "comb");
    hc_check_common(y, device, op, "y");
    TORCH_CHECK(x.scalar_type() == at::kFloat && post_a.scalar_type() == at::kFloat &&
                comb_a.scalar_type() == at::kFloat && base.scalar_type() == at::kFloat &&
                scale.scalar_type() == at::kFloat && partials.scalar_type() == at::kFloat &&
                post.scalar_type() == at::kFloat && comb.scalar_type() == at::kFloat, op, ": dtype");
    TORCH_CHECK(ya.scalar_type() == at::kFloat || ya.scalar_type() == at::kHalf, op, ": ya dtype");
    TORCH_CHECK(fn.scalar_type() == at::kFloat || fn.scalar_type() == at::kHalf, op, ": fn dtype");
    TORCH_CHECK(y.scalar_type() == at::kHalf && y.numel() == (int64_t) R * D, op, ": y must be half (R, D)");
    TORCH_CHECK(ya.dim() == 2 && ya.size(0) == R && ya.size(1) == D, op, ": ya shape");
    TORCH_CHECK(post_a.dim() == 2 && post_a.size(0) == R && post_a.size(1) == H, op, ": post_a shape");
    TORCH_CHECK(comb_a.dim() == 3 && comb_a.size(0) == R && comb_a.size(1) == H &&
                comb_a.size(2) == H, op, ": comb_a shape");
    TORCH_CHECK(fn.dim() == 2 && fn.size(0) == M && fn.size(1) == H * D, op, ": fn shape");
    TORCH_CHECK(base.numel() == M && scale.numel() == 3 && post.numel() == R * H &&
                comb.numel() == R * H * H, op, ": shapes");
    TORCH_CHECK(partials.dim() == 3 && partials.size(0) == R && partials.size(1) == 1 &&
                partials.size(2) == M + 1, op, ": partials must be (R, 1, M + 1)");
    TORCH_CHECK(sinkhorn_iters >= 1, op, ": sinkhorn_iters");
    int wtype = 0;
    if (w.has_value() && w->defined())
    {
        TORCH_CHECK(w->numel() == D && w->is_contiguous() && w->device() == device, op, ": weight shape");
        TORCH_CHECK(w->dtype() == at::kHalf || w->dtype() == at::kBFloat16, op, ": weight must be fp16/bf16");
        wtype = w->dtype() == at::kHalf ? 1 : 2;
    }
    hc_check_aligned(x, op, "x");
    hc_check_aligned(ya, op, "ya");
    hc_check_aligned(fn, op, "fn");
    TORCH_CHECK(hc_mix_supported(device.index()), op, ": device must use a 32-lane warp/wavefront");
    TORCH_CHECK(hc_mix_num_chunks(R, H * D) == 1, op, ": single-chunk regime only (R too small)");
    TORCH_CHECK(R <= HC_MAX_TICKETS, op, ": R exceeds ticket capacity");
    if (R == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(1, R);
    int* tickets = hc_tickets(device.index());
    TORCH_CHECK(tickets, op, ": no ticket buffer");
    #define PF_ARGS(Y_T, FT) (float*) x.data_ptr(), (const Y_T*) ya.data_ptr(), \
        (const float*) post_a.data_ptr(), (const float*) comb_a.data_ptr(), \
        (const FT*) fn.data_ptr(), (float*) partials.data_ptr(), tickets, \
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(), \
        (float*) post.data_ptr(), (float*) comb.data_ptr(), (half*) y.data_ptr(), \
        wtype ? w->data_ptr() : nullptr, wtype, D, (float) rms_eps, (float) hc_eps, \
        (int) sinkhorn_iters, (float) norm_eps, (float) constant_bias, (float) constant_scale
    if (ya.dtype() == at::kHalf)
    {
        if (fn.dtype() == at::kHalf)
            hc_pf_apply_mix_norm_1pass_kernel<4, 24, half, half><<<grid, NUM_THREADS + 32, 0, stream>>>(PF_ARGS(half, half));
        else
            hc_pf_apply_mix_norm_1pass_kernel<4, 24, half, float><<<grid, NUM_THREADS + 32, 0, stream>>>(PF_ARGS(half, float));
    }
    else
    {
        if (fn.dtype() == at::kHalf)
            hc_pf_apply_mix_norm_1pass_kernel<4, 24, float, half><<<grid, NUM_THREADS + 32, 0, stream>>>(PF_ARGS(float, half));
        else
            hc_pf_apply_mix_norm_1pass_kernel<4, 24, float, float><<<grid, NUM_THREADS + 32, 0, stream>>>(PF_ARGS(float, float));
    }
    #undef PF_ARGS
    cuda_check(cudaPeekAtLastError());
}

void gr_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (M, H * D) half: cat(down, inject) * w
    const at::Tensor& upt,               // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const at::Tensor& w,                 // (H * D) half norm weight (incl +1)
    double rms_eps,
    at::Tensor dots,                     // (R, M + 1, H) float workspace
    c10::optional<at::Tensor> post,      // (R, H) float out, or none (final-mixer form)
    at::Tensor mixed                     // (R, D) half or float out
)
{
    TORCH_CHECK(streams.is_cuda(), "gr_mix: streams must be a CUDA tensor");
    TORCH_CHECK(streams.dim() == 3, "gr_mix: streams must have shape (R, 4, D)");
    const int R = streams.size(0);
    const int H = streams.size(1);
    const int D = streams.size(2);
    TORCH_CHECK(H == 4 && D > 0 && D % 8 == 0,
                "gr_mix: streams must have shape (R, 4, D) with D divisible by 8");
    TORCH_CHECK(upt.dim() == 4, "gr_mix: upt must have shape (4, D / 4, LR, 4)");
    const int LR = upt.size(2);
    const int M = LR + (post ? H : 0);
    const at::Device device = streams.device();

    hc_check_common(streams, device, "gr_mix", "streams");
    hc_check_common(fn, device, "gr_mix", "fn");
    hc_check_common(upt, device, "gr_mix", "upt");
    hc_check_common(w, device, "gr_mix", "w");
    hc_check_common(dots, device, "gr_mix", "dots");
    hc_check_common(mixed, device, "gr_mix", "mixed");
    TORCH_CHECK(streams.scalar_type() == at::kFloat, "gr_mix: streams must have dtype float32");
    TORCH_CHECK(fn.scalar_type() == at::kHalf, "gr_mix: fn must have dtype float16");
    TORCH_CHECK(upt.scalar_type() == at::kHalf, "gr_mix: upt must have dtype float16");
    TORCH_CHECK(w.scalar_type() == at::kHalf, "gr_mix: w must have dtype float16");
    TORCH_CHECK(dots.scalar_type() == at::kFloat, "gr_mix: dots must have dtype float32");
    TORCH_CHECK(mixed.scalar_type() == at::kFloat || mixed.scalar_type() == at::kHalf,
                "gr_mix: mixed must have dtype float32 or float16");
    TORCH_CHECK(fn.dim() == 2 && fn.size(0) == M && fn.size(1) == H * D,
                "gr_mix: fn must have shape (LR [+ 4], 4 * D)");
    TORCH_CHECK(upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4,
                "gr_mix: upt must have shape (4, D / 4, LR, 4)");
    TORCH_CHECK(w.dim() == 1 && w.size(0) == H * D, "gr_mix: w must have shape (4 * D)");
    TORCH_CHECK(dots.dim() == 3 && dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H,
                "gr_mix: dots must have shape (R, M + 1, 4)");
    TORCH_CHECK(mixed.dim() == 2 && mixed.size(0) == R && mixed.size(1) == D,
                "gr_mix: mixed must have shape (R, D)");
    if (post)
    {
        hc_check_common(post.value(), device, "gr_mix", "post");
        TORCH_CHECK(post.value().scalar_type() == at::kFloat && post.value().dim() == 2 &&
                    post.value().size(0) == R && post.value().size(1) == H,
                    "gr_mix: post must be float32 with shape (R, 4)");
    }
    hc_check_aligned(streams, "gr_mix", "streams");
    hc_check_aligned(fn, "gr_mix", "fn");
    hc_check_aligned(upt, "gr_mix", "upt");
    hc_check_aligned(w, "gr_mix", "w");
    hc_check_aligned(mixed, "gr_mix", "mixed");
    TORCH_CHECK(hc_mix_supported(device.index()),
                "gr_mix: device must use a 32-lane warp/wavefront");
    if (R == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#if defined(USE_ROCM)
    if (gr_mix_rows_enabled() && R <= 4 && D % 256 == 0 && D <= 256 * GR_ROWS_FN_MAX)
    {
        float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
        const bool half_out = mixed.dtype() == at::kHalf;
        #define GR_ROWS_ARGS \
            (const float*) streams.data_ptr(), (const half*) fn.data_ptr(), (const half*) upt.data_ptr(), \
            (const half*) w.data_ptr(), (float*) dots.data_ptr(), post_p, mixed.data_ptr(), half_out, \
            R, M, D, LR, (float) rms_eps, stream
        if (R <= 1)      gr_mix_rows_launch<1>(GR_ROWS_ARGS);
        else if (R <= 2) gr_mix_rows_launch<2>(GR_ROWS_ARGS);
        else             gr_mix_rows_launch<4>(GR_ROWS_ARGS);
        #undef GR_ROWS_ARGS
        return;
    }
#endif
    dim3 grid_a(M + 1, R);
    gr_dots_kernel<4><<<grid_a, GR_THREADS_A, 0, stream>>>
    (
        (const float*) streams.data_ptr(), (const half*) fn.data_ptr(),
        (float*) dots.data_ptr(), M, D
    );
    cuda_check(cudaPeekAtLastError());

    // Phase C is warp-per-column-quad: chunk at warp granularity (4 * NUM_THREADS / 32
    // columns) so small R still fills the device
    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, 512 / R));
    int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;
    dim3 grid_c(n_chunks, R);
    int smem = LR * sizeof(float);
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    #define ARGS \
        (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
        (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
        post_p, mixed.data_ptr(), D, LR, chunk_cols, (float) rms_eps
    if (mixed.dtype() == at::kHalf)
        gr_finalize_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    else
        gr_finalize_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}

/*
Int8 GatedResidual mix (gfx1151 byte cut). The mixer weights are the only fp16 bulk left in an
EXL3 model: fn (rank+H, H*D) + up (H*D, rank) = 12.5 MiB per site x 97 sites = 1.22 GiB per
trunk forward, 22% of all weight bytes (bytes_by_module.py). Symmetric per-output-row int8
(scale per fn row j / per up output column h*D+d) halves that; both scales factor out of the
dot products, so they are applied once per result, and the kernels are otherwise the
per-row gr_dots / gr_finalize that measured best on this part.
*/

__device__ __forceinline__ void unpack_s8x4(uint32_t v, float& f0, float& f1, float& f2, float& f3)
{
    f0 = (float) ((int) (v << 24) >> 24);
    f1 = (float) ((int) (v << 16) >> 24);
    f2 = (float) ((int) (v << 8) >> 24);
    f3 = (float) ((int) v >> 24);
}

// group scales (gs != 0): fn scales per 128 consecutive k of (h, d), f32 [M][H * D / 128]; up scales per 64 ranks per column,
// f32 [H][D / 4][ceil(LR / 64)][4]. Weight = half(code * scale) as fp32 (bit-exact twin of the fp16 kernels on the dequantised half weights).
#define GS_FN_G 128
#define GS_UP_G 64
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
__device__ __forceinline__ size_t gs_fn_idx(int j, int h, int c8, int D) { return (size_t) j * (4 * D / GS_FN_G) + (size_t) h * (D / GS_FN_G) + (c8 >> 4); }
__device__ __forceinline__ size_t gs_up_idx(int h, int c, int i, int D, int LR) { return (((size_t) h * (D / 4) + c) * ((LR + GS_UP_G - 1) / GS_UP_G) + (i >> 6)) * 4; }

template <int H>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_q8_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const int8_t* __restrict__ fn,       // (M, H * D) int8, norm weight folded in
    const float* __restrict__ fn_scale,  // (M)
    float* __restrict__ dots,            // (R, M + 1, H): per-stream dots, row M = sum sq
    const int M,
    const int D,
    const int gs = 0
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;
    const int D4 = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 8-byte fn loads (8 int8) against two 16-byte stream quads
            const int2* f8 = (const int2*) (fn + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0 = s4[(size_t) h * D4 + 2 * c];
                float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                int2 pk = f8[c];
                float w0, w1, w2, w3, w4, w5, w6, w7;
                if (gs)
                {
                    const float gsc = fn_scale[gs_fn_idx(j, h, c, D)];
                    gs_unpack_s8x4((uint32_t) pk.x, gsc, w0, w1, w2, w3);
                    gs_unpack_s8x4((uint32_t) pk.y, gsc, w4, w5, w6, w7);
                }
                else
                {
                    unpack_s8x4((uint32_t) pk.x, w0, w1, w2, w3);
                    unpack_s8x4((uint32_t) pk.y, w4, w5, w6, w7);
                }
                a = fmaf(s0.x, w0, a); a = fmaf(s0.y, w1, a);
                a = fmaf(s0.z, w2, a); a = fmaf(s0.w, w3, a);
                a = fmaf(s1.x, w4, a); a = fmaf(s1.y, w5, a);
                a = fmaf(s1.z, w6, a); a = fmaf(s1.w, w7, a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 sv = s4[(size_t) h * D4 + c];
                a = fmaf(sv.x, sv.x, fmaf(sv.y, sv.y, fmaf(sv.z, sv.z, fmaf(sv.w, sv.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        if (j < M && !gs) v *= fn_scale[j];
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_q8_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H)
    const int8_t* __restrict__ upt,      // (H, D / 4, LR, 4) int8
    const float* __restrict__ up_scale,  // (H * D) per output column
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,
    const int chunk_cols,
    const float rms_eps,
    const int gs = 0
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 gsc[H];
        for (int i = lane; i < LR; i += 32)
        {
            float ti = t_s[i];
            if (gs && (((i - lane) & 32) == 0))     // the 32-rank step that opens a 64-rank group (warp-uniform, lane < 32)
            {
                #pragma unroll
                for (int h = 0; h < H; ++h) gsc[h] = *(const float4*) (up_scale + gs_up_idx(h, c, i, D, LR));
            }
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                // 4 int8 = one dword per (h, column quad, rank); consecutive lanes consecutive i
                uint32_t u = *(const uint32_t*) (upt + ((((size_t) h * D4 + c) * LR + i) * 4));
                float u0, u1, u2, u3;
                unpack_s8x4(u, u0, u1, u2, u3);
                if (gs) { u0 = gs_dq(u0, gsc[h].x); u1 = gs_dq(u1, gsc[h].y); u2 = gs_dq(u2, gsc[h].z); u3 = gs_dq(u3, gsc[h].w); }
                g[h].x = fmaf(ti, u0, g[h].x);
                g[h].y = fmaf(ti, u1, g[h].y);
                g[h].z = fmaf(ti, u2, g[h].z);
                g[h].w = fmaf(ti, u3, g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            const float4 sc = gs ? make_float4(1.0f, 1.0f, 1.0f, 1.0f) : *(const float4*) (up_scale + (size_t) h * D + 4 * c);
            float4 sv = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x * sc.x) * coef * LOW_TO_FLOAT(wq.x),  sv.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y * sc.y) * coef * HIGH_TO_FLOAT(wq.x), sv.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z * sc.z) * coef * LOW_TO_FLOAT(wq.y),  sv.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w * sc.w) * coef * HIGH_TO_FLOAT(wq.y), sv.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}

#include "hc_mix_q8r.cuh"

// One-pass-over-rows int8 mix: bit-identical to the one-row-per-block kernels, default ON,
// EXL3_GR_MIX_Q8_1PASS=0 restores the old launch (read once at first call).
static bool gr_q8_onepass()
{
    static int v = -1;
    if (v < 0)
    {
        const char* e = getenv("EXL3_GR_MIX_Q8_1PASS");
        v = (e && *e == '0') ? 0 : 1;
    }
    return v == 1;
}

// Largest R that takes the group launch (groups of GRR_MAX_RT rows, one weight pass per group, each group bit-identical to a
// 1..4 row call): EXL3_GR_MIX_Q8_MAXR, default GRR_MAX_RT; 8 makes the 5..8 row verify cost one more group instead of the per-row launches.
static int gr_q8_maxrows()
{
    static int v = -1;
    if (v < 0)
    {
        const char* e = getenv("EXL3_GR_MIX_Q8_MAXR");
        v = e ? std::max(GRR_MAX_RT, std::min(8, atoi(e))) : GRR_MAX_RT;
    }
    return v;
}

void gr_mix_q8
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (M, H * D) int8: cat(down, inject) * w, per-row scaled
    const at::Tensor& fn_scale,          // (M) float
    const at::Tensor& upt,               // (H, D / 4, LR, 4) int8
    const at::Tensor& up_scale,          // (H * D) float, per output column
    const at::Tensor& w,                 // (H * D) half norm weight (incl +1)
    double rms_eps,
    at::Tensor dots,                     // (R, M + 1, H) float workspace
    c10::optional<at::Tensor> post,      // (R, H) float out, or none
    at::Tensor mixed                     // (R, D) half or float out
)
{
    TORCH_CHECK(streams.is_cuda() && streams.dim() == 3, "gr_mix_q8: streams must be a CUDA (R, 4, D) tensor");
    const int R = streams.size(0);
    const int H = streams.size(1);
    const int D = streams.size(2);
    TORCH_CHECK(H == 4 && D > 0 && D % 8 == 0, "gr_mix_q8: streams must have shape (R, 4, D) with D divisible by 8");
    TORCH_CHECK(upt.dim() == 4, "gr_mix_q8: upt must have shape (4, D / 4, LR, 4)");
    const int LR = upt.size(2);
    const int M = LR + (post ? H : 0);
    const at::Device device = streams.device();
    hc_check_common(streams, device, "gr_mix_q8", "streams");
    hc_check_common(fn, device, "gr_mix_q8", "fn");
    hc_check_common(fn_scale, device, "gr_mix_q8", "fn_scale");
    hc_check_common(upt, device, "gr_mix_q8", "upt");
    hc_check_common(up_scale, device, "gr_mix_q8", "up_scale");
    hc_check_common(w, device, "gr_mix_q8", "w");
    hc_check_common(dots, device, "gr_mix_q8", "dots");
    hc_check_common(mixed, device, "gr_mix_q8", "mixed");
    TORCH_CHECK(streams.scalar_type() == at::kFloat, "gr_mix_q8: streams must be float32");
    TORCH_CHECK(fn.scalar_type() == at::kChar && upt.scalar_type() == at::kChar, "gr_mix_q8: fn/upt must be int8");
    TORCH_CHECK(fn_scale.scalar_type() == at::kFloat && up_scale.scalar_type() == at::kFloat, "gr_mix_q8: scales must be float32");
    TORCH_CHECK(w.scalar_type() == at::kHalf, "gr_mix_q8: w must be float16");
    TORCH_CHECK(dots.scalar_type() == at::kFloat, "gr_mix_q8: dots must be float32");
    TORCH_CHECK(mixed.scalar_type() == at::kFloat || mixed.scalar_type() == at::kHalf, "gr_mix_q8: mixed must be float32 or float16");
    TORCH_CHECK(fn.dim() == 2 && fn.size(0) == M && fn.size(1) == H * D, "gr_mix_q8: fn must have shape (LR [+ 4], 4 * D)");
    // group scales when fn_scale has M * H * D / 128 entries (then up_scale has H * D * ceil(LR / 64))
    const int gs = fn_scale.numel() != M ? 1 : 0;
    TORCH_CHECK(!gs || D % GS_FN_G == 0, "gr_mix_q8: group scales need D % 128 == 0");
    TORCH_CHECK(fn_scale.dim() == 1 && fn_scale.size(0) == (gs ? (int64_t) M * H * D / GS_FN_G : (int64_t) M), "gr_mix_q8: fn_scale must have shape (M) or (M * 4 * D / 128)");
    TORCH_CHECK(upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4, "gr_mix_q8: upt must have shape (4, D / 4, LR, 4)");
    TORCH_CHECK(up_scale.dim() == 1 && up_scale.size(0) == (gs ? (int64_t) H * D * ((LR + GS_UP_G - 1) / GS_UP_G) : (int64_t) H * D), "gr_mix_q8: up_scale must have shape (4 * D) or (4 * D * ceil(LR / 64))");
    TORCH_CHECK(w.dim() == 1 && w.size(0) == H * D, "gr_mix_q8: w must have shape (4 * D)");
    TORCH_CHECK(dots.dim() == 3 && dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H, "gr_mix_q8: dots must have shape (R, M + 1, 4)");
    TORCH_CHECK(mixed.dim() == 2 && mixed.size(0) == R && mixed.size(1) == D, "gr_mix_q8: mixed must have shape (R, D)");
    if (post)
    {
        hc_check_common(post.value(), device, "gr_mix_q8", "post");
        TORCH_CHECK(post.value().scalar_type() == at::kFloat && post.value().dim() == 2 &&
                    post.value().size(0) == R && post.value().size(1) == H, "gr_mix_q8: post must be float32 (R, 4)");
    }
    hc_check_aligned(streams, "gr_mix_q8", "streams");
    hc_check_aligned(fn, "gr_mix_q8", "fn");
    hc_check_aligned(upt, "gr_mix_q8", "upt");
    hc_check_aligned(up_scale, "gr_mix_q8", "up_scale");
    hc_check_aligned(w, "gr_mix_q8", "w");
    hc_check_aligned(mixed, "gr_mix_q8", "mixed");
    TORCH_CHECK(hc_mix_supported(device.index()), "gr_mix_q8: device must use a 32-lane warp/wavefront");
    if (R == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (R <= gr_q8_maxrows() && gr_q8_onepass())     // measured domain: decode / MTP verify rows; larger R keeps the old launch
    {
        grr_launch
        (
            (const float*) streams.data_ptr(), (const int8_t*) fn.data_ptr(), (const float*) fn_scale.data_ptr(),
            (const int8_t*) upt.data_ptr(), (const float*) up_scale.data_ptr(), (const half*) w.data_ptr(),
            post ? (float*) post.value().data_ptr() : nullptr, mixed.data_ptr(), (float*) dots.data_ptr(),
            (int) R, (int) M, (int) D, (int) LR, (float) rms_eps, mixed.dtype() == at::kHalf, 512, stream, gs
        );
        cuda_check(cudaPeekAtLastError());
        return;
    }
    dim3 grid_a(M + 1, R);
    gr_dots_q8_kernel<4><<<grid_a, GR_THREADS_A, 0, stream>>>
    (
        (const float*) streams.data_ptr(), (const int8_t*) fn.data_ptr(), (const float*) fn_scale.data_ptr(),
        (float*) dots.data_ptr(), M, D, gs
    );
    cuda_check(cudaPeekAtLastError());

    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, 512 / R));
    int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;
    dim3 grid_c(n_chunks, R);
    int smem = LR * sizeof(float);
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    #define ARGS_Q8 \
        (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
        (const int8_t*) upt.data_ptr(), (const float*) up_scale.data_ptr(), (const half*) w.data_ptr(), \
        post_p, mixed.data_ptr(), D, LR, chunk_cols, (float) rms_eps, gs
    if (mixed.dtype() == at::kHalf)
        gr_finalize_q8_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS_Q8);
    else
        gr_finalize_q8_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS_Q8);
    #undef ARGS_Q8
    cuda_check(cudaPeekAtLastError());
}
