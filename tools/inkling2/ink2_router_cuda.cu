// Inkling 2 Mini sigmoid router for the RTX 3090 (sm_86): scores = sigmoid(x R^T), top-8 of 128 on (scores + selection bias), weights = scores[sel] / (sum + 1e-20) * route_scale.
// One block per token (T <= 8), 256 threads: x in shared memory, each warp owns 16 experts (float4 row reads, shuffle reduce), warp 0 does the 8 selection rounds (4 candidates per lane,
// ties -> lowest expert index). Output feeds the experts plan kernel directly (sel int32 [T, 8], wt fp32 [T, 8]); no host sync, graph capturable.
#include <cuda_runtime.h>
#include <stdint.h>
#define E_MAX 128
#define TOPK 8

__device__ unsigned int g_router_fire[2];

// grid (E / 8, T): block = 8 experts of one token (one warp per expert, x in smem), writes sigmoid score + keyed score to a global scratch; the LAST block to arrive for a token (self-resetting
// counter) selects the top-8 (warp 0, 4 candidates per lane, ties -> lowest index) and writes sel / wt.
__global__ __launch_bounds__(256) void router_k(const float* __restrict__ x, const float* __restrict__ R, const float* __restrict__ bias, int D, int E, float wscale,
                                                float* __restrict__ scr, unsigned int* __restrict__ cnt, int* __restrict__ sel, float* __restrict__ wt)
{
    extern __shared__ float xs[];
    __shared__ int last;
    const int t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) atomicAdd(&g_router_fire[0], 1u);
    for (int i = tid; i < D; i += 256) xs[i] = x[(size_t)t * D + i];
    __syncthreads();
    const int e = blockIdx.x * 8 + warp;
    {
        const float4* r4 = (const float4*)(R + (size_t)e * D);
        const float4* x4 = (const float4*)xs;
        float a = 0.f;
        for (int i = lane; i < D / 4; i += 32) {
            const float4 rv = __ldg(r4 + i), xv = x4[i];
            a += rv.x * xv.x + rv.y * xv.y + rv.z * xv.z + rv.w * xv.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) {
            const float s = 1.f / (1.f + expf(-a));
            scr[((size_t)t * 2 + 0) * E_MAX + e] = s;
            scr[((size_t)t * 2 + 1) * E_MAX + e] = s + (bias ? bias[e] : 0.f);
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) last = (atomicAdd(&cnt[t], 1u) == gridDim.x - 1);
    __syncthreads();
    if (!last || warp != 0) return;
    if (lane == 0) cnt[t] = 0;
    __threadfence();
    const float* sc = scr + ((size_t)t * 2 + 0) * E_MAX;
    const float* key = scr + ((size_t)t * 2 + 1) * E_MAX;
    float kv[4];
    int ki[4];
#pragma unroll
    for (int j = 0; j < 4; j++) { ki[j] = lane + 32 * j; kv[j] = ki[j] < E ? __ldcg(key + ki[j]) : -3.0e38f; }
    float wsel[TOPK];
    int isel[TOPK];
    for (int r = 0; r < TOPK; r++) {
        float bv = kv[0]; int bi = ki[0];
#pragma unroll
        for (int j = 1; j < 4; j++) if (kv[j] > bv) { bv = kv[j]; bi = ki[j]; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        isel[r] = bi;
        wsel[r] = __ldcg(sc + bi);
#pragma unroll
        for (int j = 0; j < 4; j++) if (ki[j] == bi) kv[j] = -3.0e38f;
    }
    if (lane == 0) {
        float s = 0.f;
        for (int r = 0; r < TOPK; r++) s += wsel[r];
        const float inv = wscale / (s + 1e-20f);
        for (int r = 0; r < TOPK; r++) { sel[t * TOPK + r] = isel[r]; wt[t * TOPK + r] = wsel[r] * inv; }
    }
}

__device__ float g_router_scr[8 * 2 * E_MAX];
__device__ unsigned int g_router_cnt[8];

extern "C" int ink2_router_launch(const float* x, const float* R, const float* bias, int T, int E, int D, float wscale, int* sel, float* wt, void* stream)
{
    if (T < 1 || T > 8 || E > E_MAX || (E & 7) || (D & 3) || ((uintptr_t)R & 15) || ((uintptr_t)x & 15)) return -2;
    float* scr; unsigned int* cnt;
    cudaGetSymbolAddress((void**)&scr, g_router_scr); cudaGetSymbolAddress((void**)&cnt, g_router_cnt);
    router_k<<<dim3(E / 8, T), 256, D * sizeof(float), (cudaStream_t)stream>>>(x, R, bias, D, E, wscale, scr, cnt, sel, wt);
    return (int)cudaGetLastError();
}
extern "C" int ink2_router_fires(unsigned int* host_out)
{
    cudaDeviceSynchronize();
    return (int)cudaMemcpyFromSymbol(host_out, g_router_fire, 2 * sizeof(unsigned int));
}
