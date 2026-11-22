// Inkling 2 Mini router + LM head kernels for gfx1151 (Strix Halo), T = 1..8 rows (agent routerhead, 20/11 night).
//   ink2_router : sigmoid router (F32 [128, 2304]) + selection bias + top-8 + renormalise x route_scale + the expert UNION PLAN of ink2_experts.hip, ONE launch.
//                 grid 128 / NW blocks, one wave per expert (32 lanes read the F32 row once, float4 loads issued before x is staged), x rows in LDS (chunks of 4),
//                 per-expert sigmoid score to global, last block (atomic counter, reset by itself, graph safe) does top-8 per row (one wave per row) and builds the plan.
//   ink2_head_q8: Q8_0 LM head [N, 2304] as stored (34-byte blocks), T rows in one pass over the weights (decode shared across the rows), softcap, optional logits store,
//                 per-row argmax epilogue (block partials + last block), optional candidate row list (DSpark subset heads). Persistent grid, x staged once per block.
//   ink2_head_bf16: same for a BF16 head (UD-Q8_K_XL).
//   ink2_stream_read: pure streaming read of N bytes (the measured read roof of the same access pattern).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
typedef unsigned int u32;

static constexpr int D = 2304, NE = 128, TOPK = 8;
static constexpr int MAXU = 64;
static constexpr int Q8_BLK = 34;
static constexpr int NPAIR = D / 64;     // 36 pairs of Q8_0 blocks per row

// MUST stay identical to struct Plan of ink2_experts.hip (nu, pad, eid[64], mask[64], w[64][8])
struct Plan { int nu; int pad; int eid[MAXU]; u32 mask[MAXU]; float w[MAXU][8]; };
struct RScratch { u32 counter; u32 pad[63]; float sc[8][NE]; };
static constexpr int HMAXG = 1024;
struct HScratch { u32 counter; u32 pad[63]; float pv[8][HMAXG]; int pi[8][HMAXG]; };

__device__ __forceinline__ float wave_sum_shfl(float v)
{
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
// DPP version: quad xor 1, quad xor 2, row_ror 4, row_ror 8 (all 16 lanes of a row hold the row sum), then one cross-row shuffle
template <int CTRL> __device__ __forceinline__ float dpp(float v)
{
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xf, 0xf, false));
}
__device__ __forceinline__ float wave_sum_dpp(float v)
{
    v += dpp<0xB1>(v);        // quad_perm [1,0,3,2]
    v += dpp<0x4E>(v);        // quad_perm [2,3,0,1]
    v += dpp<0x124>(v);       // row_ror:4
    v += dpp<0x128>(v);       // row_ror:8
    v += __shfl_xor(v, 16, 32);
    return v;
}
template <int RED> __device__ __forceinline__ float wsum(float v) { return RED ? wave_sum_dpp(v) : wave_sum_shfl(v); }

__device__ __forceinline__ float ldv(const float* p) { return *(const volatile float*)p; }
__device__ __forceinline__ int ldvi(const int* p) { return *(const volatile int*)p; }

// =============================================================================================================== router
template <int T, int NW, int RED>
__global__ void __launch_bounds__(NW * 32) ink2_router(const float* __restrict__ x, const float* __restrict__ R, const float* __restrict__ bias,
                                                        int* __restrict__ sel, float* __restrict__ wt, Plan* __restrict__ plan, RScratch* __restrict__ scr, float route_scale)
{
    constexpr int NQ = D / 4;                // 576 float4 per row
    constexpr int NI = NQ / 32;              // 18 float4 per lane
    __shared__ float4 xs[4][NQ];
    __shared__ int amlast;
    __shared__ int ssel[MAXU];
    __shared__ float swt[MAXU];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    int e = blockIdx.x * NW + wave;
    const float4* rr = reinterpret_cast<const float4*>(R + (size_t)e * D);
    float4 rv[NI];
    #pragma unroll
    for (int i = 0; i < NI; i++) rv[i] = rr[lane + 32 * i];
    float acc[T];
    #pragma unroll
    for (int t0 = 0; t0 < T; t0 += 4) {
        int nt = (T - t0) < 4 ? (T - t0) : 4;
        __syncthreads();
        for (int i = tid; i < nt * NQ; i += NW * 32) {
            int tt = i / NQ, q = i - tt * NQ;
            xs[tt][q] = reinterpret_cast<const float4*>(x + (size_t)(t0 + tt) * D)[q];
        }
        __syncthreads();
        #pragma unroll
        for (int tt = 0; tt < 4; tt++) {
            if (t0 + tt < T) {
                float a = 0.0f;
                #pragma unroll
                for (int i = 0; i < NI; i++) {
                    float4 xv = xs[tt][lane + 32 * i];
                    a = fmaf(rv[i].x, xv.x, a); a = fmaf(rv[i].y, xv.y, a); a = fmaf(rv[i].z, xv.z, a); a = fmaf(rv[i].w, xv.w, a);
                }
                acc[(t0 + tt) < T ? (t0 + tt) : 0] = a;
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++) acc[t] = wsum<RED>(acc[t]);
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < T; t++) scr->sc[t][e] = 1.0f / (1.0f + expf(-acc[t]));
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) amlast = (atomicAdd(&scr->counter, 1u) == gridDim.x - 1) ? 1 : 0;
    __syncthreads();
    if (!amlast) return;
    __threadfence();
    // ---- last block: top-8 per row (one wave per row)
    for (int t = wave; t < T; t += NW) {
        float s[4], sb[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) { s[j] = ldv(&scr->sc[t][lane + 32 * j]); sb[j] = s[j] + bias[lane + 32 * j]; }
        float wsum = 0.0f, raws[TOPK];
        int ids[TOPK];
        #pragma unroll
        for (int k = 0; k < TOPK; k++) {
            float bv = sb[0], br = s[0];
            int bi = lane;
            #pragma unroll
            for (int j = 1; j < 4; j++) if (sb[j] > bv) { bv = sb[j]; br = s[j]; bi = lane + 32 * j; }
            #pragma unroll
            for (int o = 16; o >= 1; o >>= 1) {
                float ov = __shfl_xor(bv, o, 32), orr = __shfl_xor(br, o, 32);
                int oi = __shfl_xor(bi, o, 32);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; br = orr; bi = oi; }
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) if (lane + 32 * j == bi) sb[j] = -INFINITY;
            ids[k] = bi; raws[k] = br; wsum += br;
        }
        float inv = route_scale / (wsum + 1e-20f);
        if (lane < TOPK) {
            int idv = ids[0]; float rw = raws[0];
            #pragma unroll
            for (int k = 1; k < TOPK; k++) if (lane == k) { idv = ids[k]; rw = raws[k]; }
            ssel[t * TOPK + lane] = idv;
            swt[t * TOPK + lane] = rw * inv;
            sel[t * TOPK + lane] = idv;
            wt[t * TOPK + lane] = rw * inv;
        }
    }
    __syncthreads();
    // ---- the expert union plan (same maths as ink2_plan of ink2_experts.hip, 64 active threads)
    __shared__ int pfirst[MAXU], prk[MAXU];
    __shared__ int ps[MAXU];
    int i = tid, n = T * TOPK;
    bool act = i < MAXU;
    if (act) {
        plan->mask[i] = 0;
        #pragma unroll
        for (int m = 0; m < 8; m++) plan->w[i][m] = 0.0f;
        ps[i] = i < n ? ssel[i] : -1;
    }
    __syncthreads();
    int d = i;
    if (act && i < n) { for (int j = 0; j < i; j++) if (ps[j] == ps[i]) { d = j; break; } }
    if (act) pfirst[i] = (i < n && d == i) ? 1 : 0;
    __syncthreads();
    int r = 0;
    if (act && i < n) {
        for (int j = 0; j < d; j++) r += pfirst[j];
        prk[i] = r;
        if (d == i) plan->eid[r] = ps[i];
    }
    __syncthreads();
    if (act && i < n) {
        atomicOr(&plan->mask[r], 1u << (i >> 3));
        plan->w[r][i >> 3] = swt[i];
    }
    if (tid == 0) {
        int c = 0;
        for (int j = 0; j < n; j++) c += pfirst[j];
        plan->nu = c;
        scr->counter = 0;
    }
    __threadfence();
}

// =============================================================================================================== head
__device__ __forceinline__ u32 pack_h(float a, float b)
{
    a = fminf(fmaxf(a, -65504.0f), 65504.0f);
    b = fminf(fmaxf(b, -65504.0f), 65504.0f);
    return (u32)__half_as_ushort(__float2half_rn(a)) | ((u32)__half_as_ushort(__float2half_rn(b)) << 16);
}
__device__ __forceinline__ float fd(u32 a, u32 b, float c)
{
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2_t, a), __builtin_bit_cast(h2_t, b), c, false);
}
__device__ __forceinline__ float h16(u32 bits) { return (float)__builtin_bit_cast(_Float16, (unsigned short)bits); }
// 4 int8 -> two fp16 pairs, exact: (b ^ 0x80) | 0x6400 = 1024 + b + 128, minus 1152
__device__ __forceinline__ void dec4(u32 wd, u32& v0, u32& v1)
{
    u32 u = wd ^ 0x80808080u;
    u32 q0 = __builtin_amdgcn_perm(0x64646464u, u, 0x05010400u);
    u32 q1 = __builtin_amdgcn_perm(0x64646464u, u, 0x05030402u);
    const h2_t k = {(_Float16)1152.0f, (_Float16)1152.0f};
    v0 = __builtin_bit_cast(u32, __builtin_bit_cast(h2_t, q0) - k);
    v1 = __builtin_bit_cast(u32, __builtin_bit_cast(h2_t, q1) - k);
}

// one pair of Q8_0 blocks (17 dwords of the row) against T activation rows: xb = xs + p, uint4 index (m * 8 + q) * NPAIR (lane-contiguous: no bank conflicts)
template <int T>
__device__ __forceinline__ void pair_dot(const u32* a, const uint4* __restrict__ xb, float (&acc)[T])
{
    float sc[2] = {h16(a[0] & 0xffff), h16(a[8] >> 16)};
    #pragma unroll
    for (int blk = 0; blk < 2; blk++) {
        float s[T];
        #pragma unroll
        for (int m = 0; m < T; m++) s[m] = 0.0f;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            u32 v0, v1, v2, v3;
            if (blk == 0) {
                dec4(__builtin_amdgcn_alignbit(a[2 * q + 1], a[2 * q], 16), v0, v1);
                dec4(__builtin_amdgcn_alignbit(a[2 * q + 2], a[2 * q + 1], 16), v2, v3);
            } else {
                dec4(a[9 + 2 * q], v0, v1);
                dec4(a[10 + 2 * q], v2, v3);
            }
            #pragma unroll
            for (int m = 0; m < T; m++) {
                uint4 X = xb[(m * 8 + blk * 4 + q) * NPAIR];
                s[m] = fd(X.x, v0, s[m]); s[m] = fd(X.y, v1, s[m]); s[m] = fd(X.z, v2, s[m]); s[m] = fd(X.w, v3, s[m]);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        #pragma unroll
        for (int m = 0; m < T; m++) acc[m] += s[m] * sc[blk];
    }
}

template <int T>
__device__ __forceinline__ void stage_x_q8(const float* __restrict__ x, uint4* __restrict__ xs, int tid, int nthr)
{
    for (int i = tid; i < T * (D / 8); i += nthr) {
        int m = i / (D / 8), r = i - m * (D / 8);          // r = p * 8 + q: 8 floats at 8 r
        const float4* p = reinterpret_cast<const float4*>(x + (size_t)m * D) + 2 * r;
        float4 a = p[0], b = p[1];
        xs[(m * 8 + (r & 7)) * NPAIR + (r >> 3)] = make_uint4(pack_h(a.x, a.y), pack_h(a.z, a.w), pack_h(b.x, b.y), pack_h(b.z, b.w));
    }
}

// shared epilogue state: lane m < T of a wave owns the running best of row m
struct Best { float v; int i; };

template <int T, int NW>
__device__ __forceinline__ void head_finish(Best b, HScratch* __restrict__ hs, int* __restrict__ out_idx, float* __restrict__ out_val, const int* __restrict__ cand)
{
    __shared__ float wv[NW][8];
    __shared__ int wi[NW][8];
    __shared__ int amlast;
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    if (lane < T) { wv[wave][lane] = b.v; wi[wave][lane] = b.i; }
    __syncthreads();
    if (tid < T) {
        float v = wv[0][tid]; int i = wi[0][tid];
        #pragma unroll
        for (int w = 1; w < NW; w++) { float ov = wv[w][tid]; int oi = wi[w][tid]; if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; } }
        hs->pv[tid][blockIdx.x] = v; hs->pi[tid][blockIdx.x] = i;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) amlast = (atomicAdd(&hs->counter, 1u) == gridDim.x - 1) ? 1 : 0;
    __syncthreads();
    if (!amlast) return;
    __threadfence();
    float bv[T]; int bi[T];
    #pragma unroll
    for (int m = 0; m < T; m++) { bv[m] = -INFINITY; bi[m] = 0x7fffffff; }
    for (int g = tid; g < (int)gridDim.x; g += NW * 32) {
        #pragma unroll
        for (int m = 0; m < T; m++) {
            float ov = ldv(&hs->pv[m][g]); int oi = ldvi(&hs->pi[m][g]);
            if (ov > bv[m] || (ov == bv[m] && oi < bi[m])) { bv[m] = ov; bi[m] = oi; }
        }
    }
    #pragma unroll
    for (int m = 0; m < T; m++) {
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            float ov = __shfl_xor(bv[m], o, 32); int oi = __shfl_xor(bi[m], o, 32);
            if (ov > bv[m] || (ov == bv[m] && oi < bi[m])) { bv[m] = ov; bi[m] = oi; }
        }
        if (lane == 0) { wv[wave][m] = bv[m]; wi[wave][m] = bi[m]; }
    }
    __syncthreads();
    if (tid < T) {
        float v = wv[0][tid]; int i = wi[0][tid];
        #pragma unroll
        for (int w = 1; w < NW; w++) { float ov = wv[w][tid]; int oi = wi[w][tid]; if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; } }
        out_idx[tid] = cand ? cand[i] : i;
        out_val[tid] = v;
    }
    if (tid == 0) hs->counter = 0;
    __threadfence();
}

// persistent grid; wave w of block b owns rows (b * NW + w) * RPW + r + k * gridDim.x * NW * RPW
template <int T, int NW, int RPW, int RED>
__global__ void __launch_bounds__(NW * 32) ink2_head_q8(const float* __restrict__ x, const uint8_t* __restrict__ W, int N, const int* __restrict__ cand, float softcap,
                                                         float* __restrict__ logits, HScratch* __restrict__ hs, int* __restrict__ out_idx, float* __restrict__ out_val)
{
    constexpr int RB = D / 32 * Q8_BLK;      // 2448 bytes per row
    __shared__ uint4 xs[T * 8 * NPAIR];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    stage_x_q8<T>(x, xs, tid, NW * 32);
    __syncthreads();
    Best best; best.v = -INFINITY; best.i = 0x7fffffff;
    int stride = gridDim.x * NW * RPW;
    for (int base = (blockIdx.x * NW + wave) * RPW; base < N; base += stride) {
        u32 w[RPW][2][17];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int n = base + r;
            if (n < N) {
                int src = cand ? cand[n] : n;
                const u32* rw = reinterpret_cast<const u32*>(W + (size_t)src * RB);
                #pragma unroll
                for (int k = 0; k < 17; k++) w[r][0][k] = rw[lane * 17 + k];
                if (lane < NPAIR - 32) {
                    #pragma unroll
                    for (int k = 0; k < 17; k++) w[r][1][k] = rw[(lane + 32) * 17 + k];
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int n = base + r;
            if (n < N) {
                float acc[T];
                #pragma unroll
                for (int m = 0; m < T; m++) acc[m] = 0.0f;
                pair_dot<T>(w[r][0], xs + lane, acc);
                if (lane < NPAIR - 32) pair_dot<T>(w[r][1], xs + lane + 32, acc);
                float mine = 0.0f;
                #pragma unroll
                for (int m = 0; m < T; m++) { float s = wsum<RED>(acc[m]); if (lane == m) mine = s; }
                if (lane < T) {
                    float c = softcap > 0.0f ? softcap * tanhf(mine / softcap) : mine;
                    if (logits) logits[(size_t)lane * N + n] = c;
                    if (c > best.v) { best.v = c; best.i = n; }
                }
            }
        }
    }
    head_finish<T, NW>(best, hs, out_idx, out_val, cand);
}

// Q8_0 head, coalesced loads: a wave streams its RPW rows with 16-byte lane-contiguous global loads (153 uint4 per row, prefetched one iteration ahead), parks them in a
// wave-private LDS buffer (612 dwords per row; a pair of blocks = 17 dwords at an odd stride: conflict-free dword reads), and the pair math reads the dwords from LDS.
template <int T, int NW, int RPW, int RED>
__global__ void __launch_bounds__(NW * 32) ink2_head_q8l(const float* __restrict__ x, const uint8_t* __restrict__ W, int N, const int* __restrict__ cand, float softcap,
                                                          float* __restrict__ logits, HScratch* __restrict__ hs, int* __restrict__ out_idx, float* __restrict__ out_val)
{
    constexpr int RB = D / 32 * Q8_BLK;      // 2448 bytes = 153 uint4 per row
    __shared__ uint4 xs[T * 8 * NPAIR];
    __shared__ uint4 rows4[NW * RPW * 153];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    stage_x_q8<T>(x, xs, tid, NW * 32);
    __syncthreads();
    uint4* my4 = rows4 + wave * RPW * 153;
    const u32* my32 = reinterpret_cast<const u32*>(my4);
    Best best; best.v = -INFINITY; best.i = 0x7fffffff;
    int stride = gridDim.x * NW * RPW;
    int base = (blockIdx.x * NW + wave) * RPW;
    uint4 gb[RPW][5];
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int j = 0; j < 5; j++) gb[r][j] = make_uint4(0, 0, 0, 0);
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int n = base + r;
        if (n < N) {
            const uint4* rg = reinterpret_cast<const uint4*>(W + (size_t)(cand ? cand[n] : n) * RB);
            #pragma unroll
            for (int j = 0; j < 5; j++) if (lane + 32 * j < 153) gb[r][j] = rg[lane + 32 * j];
        }
    }
    for (; base < N; base += stride) {
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            if (base + r < N) {
                #pragma unroll
                for (int j = 0; j < 5; j++) if (lane + 32 * j < 153) my4[r * 153 + lane + 32 * j] = gb[r][j];
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int n = base + stride + r;
            if (n < N) {
                const uint4* rg = reinterpret_cast<const uint4*>(W + (size_t)(cand ? cand[n] : n) * RB);
                #pragma unroll
                for (int j = 0; j < 5; j++) if (lane + 32 * j < 153) gb[r][j] = rg[lane + 32 * j];
            }
        }
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int n = base + r;
            if (n < N) {
                float acc[T];
                #pragma unroll
                for (int m = 0; m < T; m++) acc[m] = 0.0f;
                pair_dot<T>(my32 + r * 612 + lane * 17, xs + lane, acc);
                if (lane < NPAIR - 32) pair_dot<T>(my32 + r * 612 + (lane + 32) * 17, xs + lane + 32, acc);
                float mine = 0.0f;
                #pragma unroll
                for (int m = 0; m < T; m++) { float s = wsum<RED>(acc[m]); if (lane == m) mine = s; }
                if (lane < T) {
                    float c = softcap > 0.0f ? softcap * tanhf(mine / softcap) : mine;
                    if (logits) logits[(size_t)lane * N + n] = c;
                    if (c > best.v) { best.v = c; best.i = n; }
                }
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
    }
    head_finish<T, NW>(best, hs, out_idx, out_val, cand);
}

// BF16 head: a lane reads 16 contiguous bytes per load (8 values), row = 288 uint4 = 9 per lane; activations fp16 in LDS (uint4 of 8 halves, same index as the weight uint4:
// lane-contiguous), converted to fp32 in the loop; weights bf16 -> fp32 exactly by a shift; fp32 FMAs.
template <int T, int NW, int RPW, int RED>
__global__ void __launch_bounds__(NW * 32) ink2_head_bf16(const float* __restrict__ x, const uint16_t* __restrict__ W, int N, const int* __restrict__ cand, float softcap,
                                                           float* __restrict__ logits, HScratch* __restrict__ hs, int* __restrict__ out_idx, float* __restrict__ out_val)
{
    constexpr int NQ = D / 8;                // 288 uint4 per row
    constexpr int NI = NQ / 32;              // 9
    __shared__ uint4 xs[T * NQ];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    for (int i = tid; i < T * NQ; i += NW * 32) {
        const float4* p = reinterpret_cast<const float4*>(x) + 2 * i;
        float4 a = p[0], b = p[1];
        xs[i] = make_uint4(pack_h(a.x, a.y), pack_h(a.z, a.w), pack_h(b.x, b.y), pack_h(b.z, b.w));
    }
    __syncthreads();
    Best best; best.v = -INFINITY; best.i = 0x7fffffff;
    int stride = gridDim.x * NW * RPW;
    for (int base = (blockIdx.x * NW + wave) * RPW; base < N; base += stride) {
        uint4 w[RPW][NI];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int n = base + r;
            if (n < N) {
                int src = cand ? cand[n] : n;
                const uint4* rw = reinterpret_cast<const uint4*>(W + (size_t)src * D);
                #pragma unroll
                for (int i = 0; i < NI; i++) w[r][i] = rw[lane + 32 * i];
            }
        }
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int n = base + r;
            if (n < N) {
                float acc[T];
                #pragma unroll
                for (int m = 0; m < T; m++) acc[m] = 0.0f;
                #pragma unroll
                for (int i = 0; i < NI; i++) {
                    float wf[8];
                    wf[0] = __builtin_bit_cast(float, w[r][i].x << 16); wf[1] = __builtin_bit_cast(float, w[r][i].x & 0xffff0000u);
                    wf[2] = __builtin_bit_cast(float, w[r][i].y << 16); wf[3] = __builtin_bit_cast(float, w[r][i].y & 0xffff0000u);
                    wf[4] = __builtin_bit_cast(float, w[r][i].z << 16); wf[5] = __builtin_bit_cast(float, w[r][i].z & 0xffff0000u);
                    wf[6] = __builtin_bit_cast(float, w[r][i].w << 16); wf[7] = __builtin_bit_cast(float, w[r][i].w & 0xffff0000u);
                    #pragma unroll
                    for (int m = 0; m < T; m++) {
                        uint4 X = xs[m * NQ + lane + 32 * i];
                        float s = acc[m];
                        s = fmaf(wf[0], h16(X.x & 0xffff), s); s = fmaf(wf[1], h16(X.x >> 16), s);
                        s = fmaf(wf[2], h16(X.y & 0xffff), s); s = fmaf(wf[3], h16(X.y >> 16), s);
                        s = fmaf(wf[4], h16(X.z & 0xffff), s); s = fmaf(wf[5], h16(X.z >> 16), s);
                        s = fmaf(wf[6], h16(X.w & 0xffff), s); s = fmaf(wf[7], h16(X.w >> 16), s);
                        acc[m] = s;
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
                float mine = 0.0f;
                #pragma unroll
                for (int m = 0; m < T; m++) { float s = wsum<RED>(acc[m]); if (lane == m) mine = s; }
                if (lane < T) {
                    float c = softcap > 0.0f ? softcap * tanhf(mine / softcap) : mine;
                    if (logits) logits[(size_t)lane * N + n] = c;
                    if (c > best.v) { best.v = c; best.i = n; }
                }
            }
        }
    }
    head_finish<T, NW>(best, hs, out_idx, out_val, cand);
}

// pure streaming read (the roof of this access pattern): each block reads a contiguous slab with 16-byte loads, xor-sums into sink
__global__ void __launch_bounds__(256) ink2_stream_read(const uint4* __restrict__ p, size_t n16, uint32_t* __restrict__ sink)
{
    uint32_t a = 0;
    size_t stride = (size_t)gridDim.x * 256;
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n16; i += stride) { uint4 v = p[i]; a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}

#define DISPATCH_T(FN, ...) switch (T) { case 1: return FN<1>(__VA_ARGS__); case 2: return FN<2>(__VA_ARGS__); case 3: return FN<3>(__VA_ARGS__); case 4: return FN<4>(__VA_ARGS__); \
    case 5: return FN<5>(__VA_ARGS__); case 6: return FN<6>(__VA_ARGS__); case 7: return FN<7>(__VA_ARGS__); case 8: return FN<8>(__VA_ARGS__); default: return 2; }

template <int T> static int run_router(int cfg, const float* x, const float* R, const float* bias, int* sel, float* wt, Plan* plan, RScratch* scr, float rs, hipStream_t st)
{
    if (cfg == 0) ink2_router<T, 4, 0><<<NE / 4, 128, 0, st>>>(x, R, bias, sel, wt, plan, scr, rs);
    else if (cfg == 1) ink2_router<T, 8, 0><<<NE / 8, 256, 0, st>>>(x, R, bias, sel, wt, plan, scr, rs);
    else if (cfg == 2) ink2_router<T, 2, 0><<<NE / 2, 64, 0, st>>>(x, R, bias, sel, wt, plan, scr, rs);
    else if (cfg == 3) ink2_router<T, 8, 1><<<NE / 8, 256, 0, st>>>(x, R, bias, sel, wt, plan, scr, rs);
    else if (cfg == 4) ink2_router<T, 4, 1><<<NE / 4, 128, 0, st>>>(x, R, bias, sel, wt, plan, scr, rs);
    else return 1;
    return 0;
}

// head configs: (NW, RPW, grid, RED); 0..7 direct dword loads, 8..11 coalesced loads through LDS (skipped = 1 when the LDS would exceed 64 KB)
template <int T, int NW, int RPW> static constexpr bool lds_ok() { return T * 8 * NPAIR * 16 + NW * RPW * 153 * 16 <= 65536; }
template <int T, int NW, int RPW, int RED> static int launch_q8l(int grid, const float* x, const uint8_t* W, int N, const int* cand, float sc, float* lg, HScratch* hs, int* oi, float* ov, hipStream_t st)
{
    if constexpr (lds_ok<T, NW, RPW>()) { ink2_head_q8l<T, NW, RPW, RED><<<grid, NW * 32, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); return 0; }
    else return 1;
}
template <int T> static int run_head_q8(int cfg, const float* x, const uint8_t* W, int N, const int* cand, float sc, float* lg, HScratch* hs, int* oi, float* ov, hipStream_t st)
{
    switch (cfg) {
    case 0: ink2_head_q8<T, 8, 1, 0><<<240, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 1: ink2_head_q8<T, 8, 2, 0><<<240, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 2: ink2_head_q8<T, 4, 1, 0><<<480, 128, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 3: ink2_head_q8<T, 4, 2, 0><<<480, 128, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 4: ink2_head_q8<T, 8, 1, 1><<<240, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 5: ink2_head_q8<T, 8, 2, 1><<<240, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 6: ink2_head_q8<T, 8, 1, 0><<<480, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 7: ink2_head_q8<T, 8, 1, 0><<<120, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 8: return launch_q8l<T, 8, 1, 0>(240, x, W, N, cand, sc, lg, hs, oi, ov, st);
    case 9: return launch_q8l<T, 8, 2, 0>(240, x, W, N, cand, sc, lg, hs, oi, ov, st);
    case 10: return launch_q8l<T, 4, 1, 0>(480, x, W, N, cand, sc, lg, hs, oi, ov, st);
    case 11: return launch_q8l<T, 8, 1, 0>(480, x, W, N, cand, sc, lg, hs, oi, ov, st);
    case 12: return launch_q8l<T, 8, 1, 1>(240, x, W, N, cand, sc, lg, hs, oi, ov, st);
    default: return 1;
    }
    return 0;
}
template <int T> static int run_head_bf16(int cfg, const float* x, const uint16_t* W, int N, const int* cand, float sc, float* lg, HScratch* hs, int* oi, float* ov, hipStream_t st)
{
    switch (cfg) {
    case 0: ink2_head_bf16<T, 8, 1, 0><<<240, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 1: ink2_head_bf16<T, 8, 2, 0><<<240, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 2: ink2_head_bf16<T, 4, 1, 0><<<480, 128, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 3: ink2_head_bf16<T, 8, 1, 0><<<480, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    case 4: ink2_head_bf16<T, 8, 1, 1><<<240, 256, 0, st>>>(x, W, N, cand, sc, lg, hs, oi, ov); break;
    default: return 1;
    }
    return 0;
}

extern "C" {
int ink2_router_scratch_bytes() { return (int)sizeof(RScratch); }
int ink2_plan_bytes() { return (int)sizeof(Plan); }
int ink2_head_scratch_bytes() { return (int)sizeof(HScratch); }
int ink2_router_launch(int T, int cfg, const float* x, const float* R, const float* bias, int* sel, float* wt, void* plan, void* scr, float route_scale, void* stream)
{
    DISPATCH_T(run_router, cfg, x, R, bias, sel, wt, (Plan*)plan, (RScratch*)scr, route_scale, (hipStream_t)stream)
}
int ink2_head_q8_launch(int T, int cfg, const float* x, const uint8_t* W, int N, const int* cand, float softcap, float* logits, void* scr, int* out_idx, float* out_val, void* stream)
{
    DISPATCH_T(run_head_q8, cfg, x, W, N, cand, softcap, logits, (HScratch*)scr, out_idx, out_val, (hipStream_t)stream)
}
int ink2_head_bf16_launch(int T, int cfg, const float* x, const uint16_t* W, int N, const int* cand, float softcap, float* logits, void* scr, int* out_idx, float* out_val, void* stream)
{
    DISPATCH_T(run_head_bf16, cfg, x, W, N, cand, softcap, logits, (HScratch*)scr, out_idx, out_val, (hipStream_t)stream)
}
int ink2_stream_read_launch(const void* p, long bytes, int grid, void* sink, void* stream)
{
    ink2_stream_read<<<grid, 256, 0, (hipStream_t)stream>>>((const uint4*)p, (size_t)bytes / 16, (uint32_t*)sink);
    return 0;
}
}
