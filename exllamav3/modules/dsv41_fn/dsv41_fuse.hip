// DS4.1 decode fusion kernels (gfx1151, wave32): the glue between the big GEMV kernels of the FFN half of a layer.
// Each kernel replaces a chain of tiny torch launches of the unfused serving path with one launch and the SAME arithmetic
// (same operation order, fp32 intermediates, same rounding points), so the outputs are bit-identical. Built with
// -ffp-contract=off like dsv41_hip.hip. Loader: dsv41_fuse.py.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define WARP 32
#define DIM 5120

static __device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }
static __device__ __forceinline__ uint16_t f2h(float v) { return __half_as_ushort(__float2half_rn(v)); }
static __device__ __forceinline__ float u2f(uint32_t u) { return __uint_as_float(u); }
static __device__ __forceinline__ float wsum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, WARP); return v; }

// ds41_swiglu_h: h = half( silu(min(gate, L)) * clamp(up, -L, L) ) with fp32 math (ops.swiglu then .to(half)); limit <= 0: no clamp.
// NaN passes through the clamps like torch.clamp. silu(x) = x / (1 + exp(-x)) as in ATen's silu kernel.
extern "C" __global__ __launch_bounds__(256)
void ds41_swiglu_h(const uint16_t* __restrict__ gate, const uint16_t* __restrict__ up, uint16_t* __restrict__ out, int n, float limit, int mode)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float g = h2f(gate[i]), u = h2f(up[i]);
    if (limit > 0.0f)
    {
        u = u > limit ? limit : (u < -limit ? -limit : u);
        g = g > limit ? limit : g;
    }
    float s;
    if (mode == 0) s = g / (1.0f + expf(-g));
    else if (mode == 1) s = g * (1.0f / (1.0f + expf(-g)));
    else if (mode == 2) s = g / (1.0f + __expf(-g));
    else if (mode == 3) s = g * (1.0f / (1.0f + __expf(-g)));
    else if (mode == 4) s = g / (1.0f + exp2f(-g * 1.4426950408889634f));
    else s = g * __frcp_rn(1.0f + expf(-g));
    // the product must be a plain fp32 multiply rounded once to fp16: left to the compiler it becomes a mixed-precision mad that
    // rounds denormal fp16 results differently from torch's fp32 multiply + .to(half) (0.17 % of outputs, 1 ulp)
    float p;
    asm("v_mul_f32 %0, %1, %2" : "=v"(p) : "v"(s), "v"(u));
    out[i] = f2h(p);
}

// ds41_moe_tail: out = half( ((o0 + o1) + o2) + float(sh) ). The group outputs of the grouped MoE (fp32, summed in group order
// in place by the unfused path) plus the shared expert, rounded once to the activation dtype.
extern "C" __global__ __launch_bounds__(256)
void ds41_moe_tail(const float* __restrict__ o0, const float* __restrict__ o1, const float* __restrict__ o2,
                   const uint16_t* __restrict__ sh, uint16_t* __restrict__ out, int n)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float a = o0[i];
    if (o1) a = a + o1[i];
    if (o2) a = a + o2[i];
    a = a + h2f(sh[i]);
    out[i] = f2h(a);
}

// ds41_router_scores_d: decode form (R = 1, W fp16 or bf16) of ds41_router_scores: same per-lane accumulation order (bitwise
// identical scores), but all 20 weight / activation loads of a lane are issued before the first FMA chain, so the 384 waves keep
// 40 loads in flight instead of 2.
extern "C" __global__ __launch_bounds__(256)
void ds41_router_scores_d(const uint16_t* __restrict__ x, const uint16_t* __restrict__ W, int wtype, float* __restrict__ scores,
                          int E, float inv_temp)
{
    constexpr int NIT = DIM / 256;
    const int e = blockIdx.x * 8 + threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (e >= E) return;
    uint4 wu[NIT], xu[NIT];
    #pragma unroll
    for (int it = 0; it < NIT; it++)
    {
        const int k = it * 256 + lane * 8;
        wu[it] = *reinterpret_cast<const uint4*>(W + (size_t) e * DIM + k);
        xu[it] = *reinterpret_cast<const uint4*>(x + k);
    }
    float acc = 0.0f;
    #pragma unroll
    for (int it = 0; it < NIT; it++)
    {
        const uint32_t wq[4] = {wu[it].x, wu[it].y, wu[it].z, wu[it].w};
        const uint32_t xq[4] = {xu[it].x, xu[it].y, xu[it].z, xu[it].w};
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            const uint32_t lo = wq[i] & 0xFFFF, hi = wq[i] >> 16;
            const float w0 = wtype == 1 ? h2f(lo) : u2f(lo << 16);
            const float w1 = wtype == 1 ? h2f(hi) : u2f(hi << 16);
            acc += w0 * h2f(xq[i] & 0xFFFF);
            acc += w1 * h2f(xq[i] >> 16);
        }
    }
    const float s = wsum(acc);
    if (lane == 0)
    {
        const float l = s * inv_temp;
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        scores[e] = sqrtf(sp);
    }
}

// ds41_router_topk2: ds41_router_topk (same selection, weights, outputs) that also writes the per-K-group slot of every pick:
// gsel[(g * R + r) * K + t] = lut[g * E + chosen[t]] (-1 = expert not in group g), what MixedKStore.group_selected gathers with
// one torch launch per group.
extern "C" __global__ __launch_bounds__(WARP)
void ds41_router_topk2(const float* __restrict__ scores, const float* __restrict__ bias, int64_t* __restrict__ sel,
                       uint16_t* __restrict__ wts, float* __restrict__ wts_f32, int E, int K, int norm, float scale,
                       const int64_t* __restrict__ lut, int G, int64_t* __restrict__ gsel, int R)
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
            for (int g = 0; g < G; g++) gsel[((size_t) g * R + r) * K + t] = lut[(size_t) g * E + chosen[t]];
        }
    }
}
