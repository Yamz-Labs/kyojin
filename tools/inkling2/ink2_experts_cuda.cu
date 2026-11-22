// Inkling 2 Mini grouped routed-experts kernels for sm_86 (RTX 3090), T = 1..8 rows, one resident copy of the GGUF bytes. CUDA port of ink2_experts.hip.
//   plan   : device-side union of the router picks (no host sync, graph capturable)          [same as the HIP kernel]
//   gate_up: per union entry and per 16-row tile (8 gate rows j0..j0+7 = A rows 0..7, the matching 8 up rows = A rows 8..15) one warp runs
//            mma.sync.m16n8k16 (fp16 in, fp32 accumulate) over K = 2304; the tokens are the n = 8 columns (T padded to 8 with zero columns).
//            c0 / c1 = gate and c2 / c3 = up of the SAME hidden unit and tokens (2t, 2t+1) sit in one thread: softcapped SwiGLU in registers -> h fp32 [entry, row, FF]
//   down   : per 16 output rows a block of NW warps; warp w loops over the union entries u = w, w + NW, ...; per entry one mma chain over K = 1152, the fp32 result is
//            scaled by router weight x per-expert scale and accumulated in registers; the warps are summed in shared memory in fixed warp order (deterministic, no atomics)
// NVFP4 = GGUF type 40 AS STORED (36-byte blocks: 4 UE4M3 scale bytes + 4 x 8 data bytes; byte i of a 16-group: low nibble = value i, high nibble = value i + 8).
// Lane (g = lane >> 2, t = lane & 3) owns the whole 16-group t of each 64-block (two data dwords + the scale dword, 12 bytes per row per block); the k slots of the mma are a
// permutation of the 16 values of the group (applied identically to A and B), so the nibbles are decoded in place:
//   dec(w) = E2M1 * 2^-14 as fp16 bits ((n & 7) << 9) | ((n & 8) << 12); the UE4M3 scale byte s -> fp16 bits (s & 0x7f) << 7 = scale * 2^-8, times 2^15 (exact) = scale * 2^7;
//   A = dec x scale = E2M1 * scale * 2^-7, EXACT in fp16 (<= 5 significant bits, range 2^-17 .. 21): the scale is folded into A, K is chained in one accumulator, final x 2^7.
// Activations (B) are fp16 pairs (saturating clamp +-65504), products and sums are fp32 (same precision class as the HIP kernel).
// C ABI, Plan struct and the h scratch layout [(entry * T + row) * FF + j] are identical to the HIP kernel (ink2_experts.py loads either .so).
#ifndef INK_PF
#define INK_PF ""                                   // -DINK_PF='".L2::128B"' = L2 prefetch size hint of the cp.async copies
#endif
#include <map>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

typedef unsigned int u32;

static constexpr int D = 2304, FF = 1152;
static constexpr int MAXU = 64;
static constexpr int NV_BLK = 36;                       // bytes per 64 values
static constexpr float NV_FIN = 128.0f;                 // 2^7

struct Plan { int nu; int pad; int eid[MAXU]; u32 mask[MAXU]; float w[MAXU][8]; };

__device__ unsigned int g_fires[4];                     // path counters: [0] gate_up blocks 0, [1] down blocks 0

__device__ __forceinline__ void plan_body(const int* __restrict__ sel, const float* __restrict__ wt, int T, Plan* __restrict__ p)
{
    __shared__ int s[MAXU], first[MAXU];
    int i = threadIdx.x, n = T * 8;
    if (i < MAXU) {
        p->mask[i] = 0;
        for (int m = 0; m < 8; m++) p->w[i][m] = 0.0f;
        s[i] = i < n ? sel[i] : -1;
    }
    __syncthreads();
    int d = i;
    if (i < n) { for (int j = 0; j < i; j++) if (s[j] == s[i]) { d = j; break; } }
    if (i < MAXU) first[i] = (i < n && d == i) ? 1 : 0;
    __syncthreads();
    int r = 0;
    if (i < n) {
        for (int j = 0; j < d; j++) r += first[j];
        if (d == i) p->eid[r] = s[i];
    }
    __syncthreads();
    if (i < n) {
        atomicOr(&p->mask[r], 1u << (i >> 3));
        p->w[r][i >> 3] = wt[i];
    }
    if (i == 0) { int c = 0; for (int j = 0; j < n; j++) c += first[j]; p->nu = c; }
}
__global__ void ink2_plan(const int* __restrict__ sel, const float* __restrict__ wt, int T, Plan* __restrict__ p) { plan_body(sel, wt, T, p); }

__device__ __forceinline__ u32 dec_pair(u32 t) { return ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12); }
__device__ __forceinline__ u32 hmul2(u32 a, u32 b) { u32 r; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
// UE4M3 byte (bit 7 ignored) -> fp16x2 pair of scale * 2^7
__device__ __forceinline__ u32 scale2(u32 sb)
{
    u32 s = (sb & 0x7fu) << 7;
    return hmul2(s | (s << 16), 0x78007800u);           // x 2^15
}
__device__ __forceinline__ u32 pack_h(float a, float b)
{
    a = fminf(fmaxf(a, -65504.0f), 65504.0f);
    b = fminf(fmaxf(b, -65504.0f), 65504.0f);
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<u32*>(&h);
}
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// B pairs of the 16-group of lane (g, t): token row xt (fp32, NULL = padding column -> zeros), 64-block b. Order matches dec(w0), dec(w0 >> 8), dec(w0 >> 4), dec(w0 >> 12), dec(w1), ...
__device__ __forceinline__ void load_b(const float* __restrict__ xt, int b, int t, u32 xb[8])
{
    if (xt == nullptr) {
        #pragma unroll
        for (int i = 0; i < 8; i++) xb[i] = 0u;
        return;
    }
    const float4* p = reinterpret_cast<const float4*>(xt + 64 * b + 16 * t);
    float4 a = p[0], bq = p[1], c = p[2], d = p[3];
    xb[0] = pack_h(a.x, a.z);   xb[1] = pack_h(a.y, a.w);                    // (e0 e2) (e1 e3)
    xb[2] = pack_h(c.x, c.z);   xb[3] = pack_h(c.y, c.w);                    // (e8 e10) (e9 e11)
    xb[4] = pack_h(bq.x, bq.z); xb[5] = pack_h(bq.y, bq.w);                  // (e4 e6) (e5 e7)
    xb[6] = pack_h(d.x, d.z);   xb[7] = pack_h(d.y, d.w);                    // (e12 e14) (e13 e15)
}

// one 16 x 8 output tile over NB 64-blocks. rA / rB = stored rows g and g + 8 (u32*), xt = token row of column g (or NULL), c = fp32 accumulators (c0 c1: row g, c2 c3: row g + 8)
template <int NB, int CB, int ABL = 0>     // ABL (ablation, numerics invalid): 0 full, 1 loads only, 2 no B load, 3 no decode, 4 no mma
__device__ __forceinline__ void tile_dot(const u32* __restrict__ rA, const u32* __restrict__ rB, const float* __restrict__ xt, int t, float c[4])
{
    static_assert(NB % CB == 0, "chunk");
    constexpr int NCH = NB / CB;
    u32 cur[CB][2][3], nxt[CB][2][3];
    #pragma unroll
    for (int i = 0; i < CB; i++) {
        const u32* pa = rA + 9 * i; const u32* pb = rB + 9 * i;
        cur[i][0][0] = __ldg(pa); cur[i][0][1] = __ldg(pa + 1 + 2 * t); cur[i][0][2] = __ldg(pa + 2 + 2 * t);
        cur[i][1][0] = __ldg(pb); cur[i][1][1] = __ldg(pb + 1 + 2 * t); cur[i][1][2] = __ldg(pb + 2 + 2 * t);
    }
    float ca[4] = {0.f, 0.f, 0.f, 0.f}, cb[4] = {0.f, 0.f, 0.f, 0.f};
    u32 sink = 0;
    #pragma unroll
    for (int ch = 0; ch < NCH; ch++) {
        if (ch + 1 < NCH) {
            #pragma unroll
            for (int i = 0; i < CB; i++) {
                const u32* pa = rA + 9 * ((ch + 1) * CB + i); const u32* pb = rB + 9 * ((ch + 1) * CB + i);
                nxt[i][0][0] = __ldg(pa); nxt[i][0][1] = __ldg(pa + 1 + 2 * t); nxt[i][0][2] = __ldg(pa + 2 + 2 * t);
                nxt[i][1][0] = __ldg(pb); nxt[i][1][1] = __ldg(pb + 1 + 2 * t); nxt[i][1][2] = __ldg(pb + 2 + 2 * t);
            }
        }
        #pragma unroll
        for (int i = 0; i < CB; i++) {
            if (ABL == 1) {
                sink ^= cur[i][0][0] ^ cur[i][0][1] ^ cur[i][0][2] ^ cur[i][1][0] ^ cur[i][1][1] ^ cur[i][1][2];
                continue;
            }
            u32 xb[8];
            if (ABL == 2) {
                #pragma unroll
                for (int q = 0; q < 8; q++) xb[q] = 0x3c003c00u + (u32)(t + q);
            } else load_b(xt, ch * CB + i, t, xb);
            u32 pa[8], pb[8];
            u32 sa = scale2(cur[i][0][0] >> (8 * t)), sb = scale2(cur[i][1][0] >> (8 * t));
            u32 wa0 = cur[i][0][1], wa1 = cur[i][0][2], wb0 = cur[i][1][1], wb1 = cur[i][1][2];
            if (ABL == 3) {
                pa[0] = wa0; pb[0] = wb0; pa[1] = wa0 >> 8; pb[1] = wb0 >> 8; pa[2] = wa0 >> 4; pb[2] = wb0 >> 4; pa[3] = wa0 >> 12; pb[3] = wb0 >> 12;
                pa[4] = wa1; pb[4] = wb1; pa[5] = wa1 >> 8; pb[5] = wb1 >> 8; pa[6] = wa1 >> 4; pb[6] = wb1 >> 4; pa[7] = wa1 >> 12; pb[7] = wb1 >> 12;
                pa[0] ^= sa; pb[0] ^= sb;
            } else {
            pa[0] = hmul2(dec_pair(wa0), sa);       pb[0] = hmul2(dec_pair(wb0), sb);
            pa[1] = hmul2(dec_pair(wa0 >> 8), sa);  pb[1] = hmul2(dec_pair(wb0 >> 8), sb);
            pa[2] = hmul2(dec_pair(wa0 >> 4), sa);  pb[2] = hmul2(dec_pair(wb0 >> 4), sb);
            pa[3] = hmul2(dec_pair(wa0 >> 12), sa); pb[3] = hmul2(dec_pair(wb0 >> 12), sb);
            pa[4] = hmul2(dec_pair(wa1), sa);       pb[4] = hmul2(dec_pair(wb1), sb);
            pa[5] = hmul2(dec_pair(wa1 >> 8), sa);  pb[5] = hmul2(dec_pair(wb1 >> 8), sb);
            pa[6] = hmul2(dec_pair(wa1 >> 4), sa);  pb[6] = hmul2(dec_pair(wb1 >> 4), sb);
            pa[7] = hmul2(dec_pair(wa1 >> 12), sa); pb[7] = hmul2(dec_pair(wb1 >> 12), sb);
            }
            // step s: a0 = P(row g), a1 = P(row g + 8), a2 = Q(row g), a3 = Q(row g + 8); b0 = xP, b1 = xQ  (P = reg 2s, Q = reg 2s + 1)
            if (ABL == 4) {
                #pragma unroll
                for (int q = 0; q < 8; q++) sink ^= pa[q] ^ pb[q] ^ xb[q];
            } else {
            mma16816(ca, pa[0], pb[0], pa[1], pb[1], xb[0], xb[1]);
            mma16816(cb, pa[2], pb[2], pa[3], pb[3], xb[2], xb[3]);
            mma16816(ca, pa[4], pb[4], pa[5], pb[5], xb[4], xb[5]);
            mma16816(cb, pa[6], pb[6], pa[7], pb[7], xb[6], xb[7]);
            }
        }
        if (ch + 1 < NCH) {
            #pragma unroll
            for (int i = 0; i < CB; i++)
                #pragma unroll
                for (int r = 0; r < 2; r++)
                    #pragma unroll
                    for (int k = 0; k < 3; k++) cur[i][r][k] = nxt[i][r][k];
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) c[i] = ca[i] + cb[i] + (ABL ? (float)(sink & 0xffu) : 0.0f);
}


// ------------------------------------------------------------------------------------------------------------------ Q8_0 (as stored: 34-byte blocks = fp16 d + 32 int8)
// The int8 x fp16-scale product is not exact in fp16: the scale is applied in fp32 after the mma of each 32-block (fresh accumulator, 2 k-steps), acc += c * d(row).
// A lane (g, t) owns bytes 8t .. 8t + 7 of every 32-block of rows g and g + 8 (k slots permuted identically in A and B). A block PAIR is 68 bytes = 17 aligned dwords d[]:
// block 0 data = bytes 2 .. 33 (funnel-shifted by 16 bits), block 1 data = bytes 36 .. 67 (dword aligned), scales d[0] & 0xffff and d[8] >> 16.
// int8 -> fp16 exactly: (b ^ 0x80) | 0x6400 = 1024 + b + 128, minus 1152 (byte_perm + sub.f16x2).
__device__ __forceinline__ u32 hsub2(u32 a, u32 b) { u32 r; asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ void dec_q8(u32 w, u32& P, u32& Q)
{
    u32 u = w ^ 0x80808080u;
    P = hsub2(__byte_perm(u, 0x64646464u, 0x5140), 0x64806480u);   // (b0, b1)
    Q = hsub2(__byte_perm(u, 0x64646464u, 0x5342), 0x64806480u);   // (b2, b3)
}
__device__ __forceinline__ float h16f(u32 bits) { return __half2float(__ushort_as_half((unsigned short)bits)); }

template <int NB, int CB>      // NB = block pairs per row
__device__ __forceinline__ void tile_dot_q8(const u32* __restrict__ rA, const u32* __restrict__ rB, const float* __restrict__ xt, int t, float c[4])
{
    static_assert(NB % CB == 0, "chunk");
    constexpr int NCH = NB / CB;
    u32 cur[CB][2][7], nxt[CB][2][7];
    #pragma unroll
    for (int i = 0; i < CB; i++) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            const u32* p = (r ? rB : rA) + 17 * i;
            cur[i][r][0] = __ldg(p + 2 * t); cur[i][r][1] = __ldg(p + 2 * t + 1); cur[i][r][2] = __ldg(p + 2 * t + 2);
            cur[i][r][3] = __ldg(p + 9 + 2 * t); cur[i][r][4] = __ldg(p + 10 + 2 * t); cur[i][r][5] = __ldg(p); cur[i][r][6] = __ldg(p + 8);
        }
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int ch = 0; ch < NCH; ch++) {
        if (ch + 1 < NCH) {
            #pragma unroll
            for (int i = 0; i < CB; i++) {
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    const u32* p = (r ? rB : rA) + 17 * ((ch + 1) * CB + i);
                    nxt[i][r][0] = __ldg(p + 2 * t); nxt[i][r][1] = __ldg(p + 2 * t + 1); nxt[i][r][2] = __ldg(p + 2 * t + 2);
                    nxt[i][r][3] = __ldg(p + 9 + 2 * t); nxt[i][r][4] = __ldg(p + 10 + 2 * t); nxt[i][r][5] = __ldg(p); nxt[i][r][6] = __ldg(p + 8);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < CB; i++) {
            int pr = ch * CB + i;
            u32 wa[2][2], wb[2][2];                     // [block][dword] of rows A / B
            wa[0][0] = __funnelshift_r(cur[i][0][0], cur[i][0][1], 16); wa[0][1] = __funnelshift_r(cur[i][0][1], cur[i][0][2], 16);
            wa[1][0] = cur[i][0][3]; wa[1][1] = cur[i][0][4];
            wb[0][0] = __funnelshift_r(cur[i][1][0], cur[i][1][1], 16); wb[0][1] = __funnelshift_r(cur[i][1][1], cur[i][1][2], 16);
            wb[1][0] = cur[i][1][3]; wb[1][1] = cur[i][1][4];
            float dA[2] = {h16f(cur[i][0][5] & 0xffffu), h16f(cur[i][0][6] >> 16)};
            float dB[2] = {h16f(cur[i][1][5] & 0xffffu), h16f(cur[i][1][6] >> 16)};
            #pragma unroll
            for (int blk = 0; blk < 2; blk++) {
                const float4* xp = reinterpret_cast<const float4*>(xt ? xt + 64 * pr + 32 * blk + 8 * t : nullptr);
                u32 xb[4] = {0u, 0u, 0u, 0u};
                if (xt) {
                    float4 a = xp[0], b = xp[1];
                    xb[0] = pack_h(a.x, a.y); xb[1] = pack_h(a.z, a.w); xb[2] = pack_h(b.x, b.y); xb[3] = pack_h(b.z, b.w);
                }
                float cc[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int s = 0; s < 2; s++) {
                    u32 PA, QA, PB, QB;
                    dec_q8(wa[blk][s], PA, QA);
                    dec_q8(wb[blk][s], PB, QB);
                    mma16816(cc, PA, PB, QA, QB, xb[2 * s], xb[2 * s + 1]);
                }
                acc[0] += cc[0] * dA[blk]; acc[1] += cc[1] * dA[blk]; acc[2] += cc[2] * dB[blk]; acc[3] += cc[3] * dB[blk];
            }
        }
        if (ch + 1 < NCH) {
            #pragma unroll
            for (int i = 0; i < CB; i++)
                #pragma unroll
                for (int r = 0; r < 2; r++)
                    #pragma unroll
                    for (int k = 0; k < 7; k++) cur[i][r][k] = nxt[i][r][k];
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) c[i] = acc[i];
}

// ------------------------------------------------------------------------------------------------------------------ gate_up + situ
// grid (FF / 8 / NW, 8 * T): blockIdx.y = union entry. Warp = one tile of 8 hidden units (gate rows j0..j0+7, up rows j0 + FF ..).
template <int NW, int CB, bool Q8, int ABL = 0>
__global__ void __launch_bounds__(NW * 32) ink2_gu(int T, const float* __restrict__ x, const Plan* __restrict__ plan, const uint8_t* __restrict__ gu,
                                                    const float* __restrict__ s_gu, float* __restrict__ hs, float beta, float lb)
{
    constexpr int RB = Q8 ? D / 32 * 34 : D / 64 * NV_BLK;       // 2448 / 1296 bytes per row
    int u = blockIdx.y;
    if (u >= plan->nu) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_fires[0], 1u);
    int e = plan->eid[u];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int j0 = (blockIdx.x * NW + wave) * 8;
    const uint8_t* eb = gu + (size_t)e * (2 * FF) * RB;
    const u32* rg = reinterpret_cast<const u32*>(eb + (size_t)(j0 + g) * RB);
    const u32* ru = reinterpret_cast<const u32*>(eb + (size_t)(j0 + g + FF) * RB);
    float c[4];
    if (Q8) tile_dot_q8<D / 64, CB>(rg, ru, g < T ? x + (size_t)g * D : nullptr, t, c);
    else tile_dot<D / 64, CB, ABL>(rg, ru, g < T ? x + (size_t)g * D : nullptr, t, c);
    float scale = (Q8 ? 1.0f : NV_FIN) * (s_gu ? s_gu[e] : 1.0f);
    int j = j0 + g;
    #pragma unroll
    for (int q = 0; q < 2; q++) {
        int m = 2 * t + q;
        if (m < T) {
            float gg = c[q] * scale, uu = c[2 + q] * scale;
            float act = beta * tanhf(gg / beta) * (1.0f / (1.0f + expf(-gg)));
            float upv = lb > 0.0f ? lb * tanhf(uu / lb) : uu;
            hs[((size_t)u * T + m) * FF + j] = act * upv;
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------ down
// grid (D / 16): block = NW warps on the same 16 output rows; warp w takes union entries u = w, w + NW, ...; shared-memory sum in fixed warp order.
template <int NW, int CB, bool Q8>
__global__ void __launch_bounds__(NW * 32) ink2_dn(int T, const Plan* __restrict__ plan, const float* __restrict__ hs, const uint8_t* __restrict__ dn,
                                                    const float* __restrict__ s_dn, float* __restrict__ out)
{
    constexpr int RB = Q8 ? FF / 32 * 34 : FF / 64 * NV_BLK;      // 1224 / 648 bytes per row
    __shared__ float red[NW][4][32];
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_fires[1], 1u);
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int n0 = blockIdx.x * 16;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int nu = plan->nu;
    for (int u = wave; u < nu; u += NW) {
        int e = plan->eid[u];
        const uint8_t* eb = dn + ((size_t)e * D + n0) * RB;
        const u32* rA = reinterpret_cast<const u32*>(eb + (size_t)g * RB);
        const u32* rB = reinterpret_cast<const u32*>(eb + (size_t)(g + 8) * RB);
        float c[4];
        if (Q8) tile_dot_q8<FF / 64, CB>(rA, rB, g < T ? hs + ((size_t)u * T + g) * FF : nullptr, t, c);
        else tile_dot<FF / 64, CB>(rA, rB, g < T ? hs + ((size_t)u * T + g) * FF : nullptr, t, c);
        float sd = s_dn ? s_dn[e] : 1.0f;
        float cw0 = plan->w[u][2 * t] * sd, cw1 = plan->w[u][2 * t + 1] * sd;      // 0 for tokens that did not pick this entry
        acc[0] += c[0] * cw0; acc[1] += c[1] * cw1; acc[2] += c[2] * cw0; acc[3] += c[3] * cw1;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) red[wave][i][lane] = acc[i];
    __syncthreads();
    if (wave == 0) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int w = 0; w < NW; w++)
            #pragma unroll
            for (int i = 0; i < 4; i++) v[i] += red[w][i][lane];
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            int m = 2 * t + q;
            if (m < T) { out[(size_t)m * D + n0 + g] = v[q] * (Q8 ? 1.0f : NV_FIN); out[(size_t)m * D + n0 + g + 8] = v[2 + q] * (Q8 ? 1.0f : NV_FIN); }
        }
    }
}


// ------------------------------------------------------------------------------------------------------------------ v2 kernels: cp.async slab pipeline through shared memory (NVFP4)
// The weight tile of a warp (16 rows x K) is streamed with 16-byte (gate_up: 144 B = 4 blocks per row and slab) / 8-byte (down: 72 B = 2 blocks per row and slab) cp.async copies into a per-warp
// ring of S stages (no block barrier, __syncwarp only); lanes read their fragments from shared memory. The activations B are pre-converted once per call by ink2_xprep
// (gate_up: fp16 pairs in the exact load_b order, g_xh) so that the B operand is two 16-byte loads per block and no ALU conversion.
__device__ __forceinline__ void cp_async16(void* sm, const void* gl)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(sm);
    asm volatile("cp.async.cg.shared.global" INK_PF " [%0], [%1], 16;\n" :: "r"(s), "l"(gl));
}
__device__ __forceinline__ void cp_async8(void* sm, const void* gl)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(sm);
    asm volatile("cp.async.ca.shared.global" INK_PF " [%0], [%1], 8;\n" :: "r"(s), "l"(gl));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __align__(16) u32 g_xh[8 * 36 * 4 * 8];                    // [token][64-block][t][8] fp16 pairs (B fragments of x), tokens >= T are zero

__device__ __forceinline__ void xprep_nv(int T, const float* __restrict__ x, int i)      // (token, block, t) = i
{
    int m = i / 144, r = i - m * 144, b = r >> 2, t = r & 3;
    u32 xb[8];
    load_b(m < T ? x + (size_t)m * D : nullptr, b, t, xb);
    uint4* o = reinterpret_cast<uint4*>(g_xh + (size_t)i * 8);
    o[0] = make_uint4(xb[0], xb[1], xb[2], xb[3]);
    o[1] = make_uint4(xb[4], xb[5], xb[6], xb[7]);
}
__device__ __forceinline__ void xprep_q8(int T, const float* __restrict__ x, int i)      // (token, block of 32, t) = i
{
    int m = i / 288, r = i - m * 288, b = r >> 2, t = r & 3;
    u32 xb[4] = {0u, 0u, 0u, 0u};
    if (m < T) {
        const float4* xp = reinterpret_cast<const float4*>(x + (size_t)m * D + 32 * b + 8 * t);
        float4 a = xp[0], c = xp[1];
        xb[0] = pack_h(a.x, a.y); xb[1] = pack_h(a.z, a.w); xb[2] = pack_h(c.x, c.y); xb[3] = pack_h(c.z, c.w);
    }
    reinterpret_cast<uint4*>(g_xh)[i] = make_uint4(xb[0], xb[1], xb[2], xb[3]);
}
// plan + activation pre-conversion in one launch: block 0 = plan, blocks 1.. = x -> fp16 B fragments (fmt 1 NVFP4 order, 2 Q8_0 order, 0 none)
__global__ void __launch_bounds__(128) ink2_plan_x(const int* __restrict__ sel, const float* __restrict__ wt, int T, Plan* __restrict__ p, const float* __restrict__ x, int fmt)
{
    if (blockIdx.x == 0) { plan_body(sel, wt, T, p); return; }
    int i = (blockIdx.x - 1) * 128 + threadIdx.x;
    if (fmt == 1 && i < 1152) xprep_nv(T, x, i);
    else if (fmt == 2 && i < 2304) xprep_q8(T, x, i);
}

__device__ __align__(16) u32 g_hh[64 * 8 * 72 * 8];      // [union entry][token][18 blocks x 4 t][8] fp16 pairs (B fragments of h for the down kernel), tokens >= T zero

__global__ void __launch_bounds__(128) ink2_hprep(int T, const Plan* __restrict__ plan, const float* __restrict__ hs)
{
    int u = blockIdx.x, m = blockIdx.y, i = threadIdx.x;
    if (u >= plan->nu || i >= 72) return;
    u32 xb[8];
    load_b(m < T ? hs + ((size_t)u * T + m) * FF : nullptr, i >> 2, i & 3, xb);
    uint4* o = reinterpret_cast<uint4*>(g_hh + ((size_t)(u * 8 + m) * 72 + i) * 8);
    o[0] = make_uint4(xb[0], xb[1], xb[2], xb[3]);
    o[1] = make_uint4(xb[4], xb[5], xb[6], xb[7]);
}

// decode + 4 mma of one 64-block of a 16 x 8 tile: sA / sB scale dwords, wa0 wa1 / wb0 wb1 data dwords of rows g and g + 8, xb the 8 B pairs
__device__ __forceinline__ void blk_mma(u32 sA, u32 wa0, u32 wa1, u32 sB, u32 wb0, u32 wb1, const u32 xb[8], int t, float* ca, float* cb)
{
    u32 pa[8], pb[8];
    u32 sa = scale2(sA >> (8 * t)), sb = scale2(sB >> (8 * t));
    pa[0] = hmul2(dec_pair(wa0), sa);       pb[0] = hmul2(dec_pair(wb0), sb);
    pa[1] = hmul2(dec_pair(wa0 >> 8), sa);  pb[1] = hmul2(dec_pair(wb0 >> 8), sb);
    pa[2] = hmul2(dec_pair(wa0 >> 4), sa);  pb[2] = hmul2(dec_pair(wb0 >> 4), sb);
    pa[3] = hmul2(dec_pair(wa0 >> 12), sa); pb[3] = hmul2(dec_pair(wb0 >> 12), sb);
    pa[4] = hmul2(dec_pair(wa1), sa);       pb[4] = hmul2(dec_pair(wb1), sb);
    pa[5] = hmul2(dec_pair(wa1 >> 8), sa);  pb[5] = hmul2(dec_pair(wb1 >> 8), sb);
    pa[6] = hmul2(dec_pair(wa1 >> 4), sa);  pb[6] = hmul2(dec_pair(wb1 >> 4), sb);
    pa[7] = hmul2(dec_pair(wa1 >> 12), sa); pb[7] = hmul2(dec_pair(wb1 >> 12), sb);
    mma16816(ca, pa[0], pb[0], pa[1], pb[1], xb[0], xb[1]);
    mma16816(cb, pa[2], pb[2], pa[3], pb[3], xb[2], xb[3]);
    mma16816(ca, pa[4], pb[4], pa[5], pb[5], xb[4], xb[5]);
    mma16816(cb, pa[6], pb[6], pa[7], pb[7], xb[6], xb[7]);
}

// gate_up v2: same grid / tile / epilogue as ink2_gu. smem ring per warp: S stages x 16 rows x 144 B (rows 0..7 gate, 8..15 up), 36 dwords per smem row.
template <int NW, int S, int ABL = 0, int SBG = 4>        // SBG = 64-blocks per slab (4: 144 B row segments, 12: 432 B).  ABL (timing probes, numerics invalid): bit 0 = no cp.async after the prologue, bit 1 = no decode / mma
__global__ void __launch_bounds__(NW * 32) ink2_gu2(int T, const Plan* __restrict__ plan, const uint8_t* __restrict__ gu, const float* __restrict__ s_gu, float* __restrict__ hs, float beta, float lb)
{
    extern __shared__ __align__(16) u32 smem[];
    constexpr int RB = D / 64 * NV_BLK, NSL = D / 64 / SBG;       // 1296 bytes per row
    constexpr int CHK = SBG * NV_BLK / 16, NCP = 16 * CHK, NIT = (NCP + 31) / 32, SDW = SBG * 9, STG = 16 * SDW;   // 16-byte chunks per row and slab, copies per stage, lane iterations, dwords per smem row / stage
    int u = blockIdx.y;
    if (u >= plan->nu) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_fires[0], 1u);
    int e = plan->eid[u];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int j0 = (blockIdx.x * NW + wave) * 8;
    const uint8_t* eb = gu + (size_t)e * (2 * FF) * RB;
    u32* ring = smem + wave * S * STG;
    auto issue = [&](int sl, int st) {
        if (sl < NSL) {
            #pragma unroll
            for (int i = 0; i < NIT; i++) {
                int q = lane + 32 * i;
                if (NCP % 32 == 0 || q < NCP) {
                    int r = q / CHK, cc = q - r * CHK;
                    int row = r < 8 ? j0 + r : j0 + (r - 8) + FF;
                    cp_async16(reinterpret_cast<uint8_t*>(ring + st * STG) + r * (SDW * 4) + 16 * cc, eb + (size_t)row * RB + sl * (SBG * NV_BLK) + 16 * cc);
                }
            }
        }
        cp_commit();
    };
    #pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, s);
    float ca[4] = {0.f, 0.f, 0.f, 0.f}, cb[4] = {0.f, 0.f, 0.f, 0.f};
    const uint4* xp = reinterpret_cast<const uint4*>(g_xh + (size_t)g * (36 * 4 * 8) + t * 8);
    #pragma unroll 1
    for (int sl = 0; sl < NSL; sl++) {
        cp_wait<S - 2>();
        __syncwarp();
        if (!(ABL & 1)) issue(sl + S - 1, (sl + S - 1) % S); else cp_commit();
        const u32* st = ring + (sl % S) * STG;
        const u32* ra = st + g * SDW; const u32* rb = ra + 8 * SDW;
        #pragma unroll
        for (int b = 0; b < SBG; b++) {
            u32 xb[8];
            uint4 x0 = xp[(sl * SBG + b) * 8], x1 = xp[(sl * SBG + b) * 8 + 1];
            xb[0] = x0.x; xb[1] = x0.y; xb[2] = x0.z; xb[3] = x0.w; xb[4] = x1.x; xb[5] = x1.y; xb[6] = x1.z; xb[7] = x1.w;
            if (ABL & 2) { ca[0] += __uint_as_float(ra[9 * b + 1 + 2 * t] ^ rb[9 * b + 2 + 2 * t] ^ xb[0] ^ xb[7] ^ ra[9 * b] ^ rb[9 * b]); continue; }
            blk_mma(ra[9 * b], ra[9 * b + 1 + 2 * t], ra[9 * b + 2 + 2 * t], rb[9 * b], rb[9 * b + 1 + 2 * t], rb[9 * b + 2 + 2 * t], xb, t, ca, cb);
        }
    }
    float c[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) c[i] = ca[i] + cb[i];
    float scale = NV_FIN * (s_gu ? s_gu[e] : 1.0f);
    int j = j0 + g;
    #pragma unroll
    for (int q = 0; q < 2; q++) {
        int m = 2 * t + q;
        if (m < T) {
            float gg = c[q] * scale, uu = c[2 + q] * scale;
            float act = beta * tanhf(gg / beta) * (1.0f / (1.0f + expf(-gg)));
            float upv = lb > 0.0f ? lb * tanhf(uu / lb) : uu;
            float hv = act * upv;
            hs[((size_t)u * T + m) * FF + j] = hv;
            int b = j >> 6, tt = (j >> 4) & 3, pp = j & 15;       // pair-ordered fp16 copy for the down kernel (load_b order)
            int slot = (pp & 1) + 2 * ((pp >> 3) & 1) + 4 * ((pp >> 2) & 1);
            reinterpret_cast<unsigned short*>(g_hh)[((size_t)(u * 8 + m) * 576 + (b * 4 + tt) * 8 + slot) * 2 + ((pp >> 1) & 1)] = __half_as_ushort(__float2half_rn(fminf(fmaxf(hv, -65504.0f), 65504.0f)));
        }
    }
}

// down v2: same grid / semantics as ink2_dn. Per warp a flat sequence f = (entry iteration, slab of SBD blocks) through the ring (rows are 648 B = 8-byte aligned only: 8-byte copies, 36 * SBD B per row and slab).
// The B operand is read from g_hh (written by the v2 gate_up epilogue, or by ink2_hprep_launch when h came from elsewhere).
template <int NW, int S, int SBD, int ABL = 0, bool W16 = false>   // W16: 16-byte cp.async over the 16-byte-aligned window of each 36 * SBD byte row segment (rows are only 8-byte aligned)
__global__ void __launch_bounds__(NW * 32) ink2_dn2(int T, const Plan* __restrict__ plan, const uint8_t* __restrict__ dn, const float* __restrict__ s_dn, float* __restrict__ out)
{
    extern __shared__ __align__(16) u32 smem[];
    constexpr int RB = FF / 64 * NV_BLK, NSL = FF / 64 / SBD;     // 648 bytes per row
    constexpr int CH = W16 ? (SBD * NV_BLK + 8 + 15) / 16 : SBD * NV_BLK / 8, SD = W16 ? CH * 4 : SBD * 9;   // copies per row and slab, dwords per smem row
    constexpr int NCP = 16 * CH, NIT = (NCP + 31) / 32, STG = 16 * SD;   // copies per stage, per lane, dwords per stage
    __shared__ float red[NW][4][32];
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_fires[1], 1u);
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int n0 = blockIdx.x * 16;
    int nu = plan->nu;
    int nit = nu > wave ? (nu - wave + NW - 1) / NW : 0;
    int total = nit * NSL;
    u32* ring = smem + wave * S * STG;
    auto issue = [&](int f, int st) {
        if (f < total) {
            int it = f / NSL, sl = f - it * NSL;
            const uint8_t* eb = dn + ((size_t)plan->eid[wave + it * NW] * D + n0) * RB + sl * (SBD * NV_BLK);
            #pragma unroll
            for (int i = 0; i < NIT; i++) {
                int q = lane + 32 * i;
                if (NCP % 32 == 0 || q < NCP) {
                    int r = q / CH, cc = q - r * CH;
                    if (W16) { int sh = (sl * (SBD * NV_BLK) + r * RB) & 15; cp_async16(reinterpret_cast<uint8_t*>(ring + st * STG) + r * (SD * 4) + 16 * cc, eb + (size_t)r * RB - sh + 16 * cc); }
                    else cp_async8(reinterpret_cast<uint8_t*>(ring + st * STG) + r * (SD * 4) + 8 * cc, eb + (size_t)r * RB + 8 * cc);
                }
            }
        }
        cp_commit();
    };
    #pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, s);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float ca[4] = {0.f, 0.f, 0.f, 0.f}, cb[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 1
    for (int f = 0; f < total; f++) {
        cp_wait<S - 2>();
        __syncwarp();
        if (!(ABL & 1)) issue(f + S - 1, (f + S - 1) % S); else cp_commit();
        int it = f / NSL, sl = f - it * NSL, uu = wave + it * NW;
        const u32* st = ring + (f % S) * STG;
        const u32* ra = st + g * SD + (W16 ? ((sl * (SBD * NV_BLK) + g * RB) & 15) >> 2 : 0); const u32* rb = ra + 8 * SD;
        const uint4* xp = reinterpret_cast<const uint4*>(g_hh + (size_t)(uu * 8 + g) * 576 + t * 8);
        #pragma unroll
        for (int b = 0; b < SBD; b++) {
            u32 xb[8];
            uint4 x0 = xp[(sl * SBD + b) * 8], x1 = xp[(sl * SBD + b) * 8 + 1];
            xb[0] = x0.x; xb[1] = x0.y; xb[2] = x0.z; xb[3] = x0.w; xb[4] = x1.x; xb[5] = x1.y; xb[6] = x1.z; xb[7] = x1.w;
            if (ABL & 2) { ca[0] += __uint_as_float(ra[9 * b + 1 + 2 * t] ^ rb[9 * b + 2 + 2 * t] ^ xb[0] ^ xb[7] ^ ra[9 * b] ^ rb[9 * b]); continue; }
            blk_mma(ra[9 * b], ra[9 * b + 1 + 2 * t], ra[9 * b + 2 + 2 * t], rb[9 * b], rb[9 * b + 1 + 2 * t], rb[9 * b + 2 + 2 * t], xb, t, ca, cb);
        }
        if (sl == NSL - 1) {
            int e = plan->eid[uu];
            float sd = s_dn ? s_dn[e] : 1.0f;
            float cw0 = plan->w[uu][2 * t] * sd, cw1 = plan->w[uu][2 * t + 1] * sd;
            acc[0] += (ca[0] + cb[0]) * cw0; acc[1] += (ca[1] + cb[1]) * cw1; acc[2] += (ca[2] + cb[2]) * cw0; acc[3] += (ca[3] + cb[3]) * cw1;
            #pragma unroll
            for (int i = 0; i < 4; i++) { ca[i] = 0.f; cb[i] = 0.f; }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) red[wave][i][lane] = acc[i];
    __syncthreads();
    if (wave == 0) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int w = 0; w < NW; w++)
            #pragma unroll
            for (int i = 0; i < 4; i++) v[i] += red[w][i][lane];
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            int m = 2 * t + q;
            if (m < T) { out[(size_t)m * D + n0 + g] = v[q] * NV_FIN; out[(size_t)m * D + n0 + g + 8] = v[2 + q] * NV_FIN; }
        }
    }
}

template <typename K> static void set_smem(K k, int bytes)
{
    static std::map<void*, int> done;
    if (bytes > 32 * 1024 && done[(void*)k] < bytes) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); done[(void*)k] = bytes; }
}
template <int NW, int S, int ABL = 0, int SBG = 4> static void launch_gu2(int T, const float* x, const void* plan, const uint8_t* gu, const float* s_gu, float* hs, float beta, float lb, cudaStream_t st)
{
    int bytes = NW * S * 576 * SBG;
    set_smem(ink2_gu2<NW, S, ABL, SBG>, bytes);
    ink2_gu2<NW, S, ABL, SBG><<<dim3(FF / 8 / NW, 8 * T), NW * 32, bytes, st>>>(T, (const Plan*)plan, gu, s_gu, hs, beta, lb);
}
template <int NW, int S, int SBD = 2, int ABL = 0, bool W16 = false> static int launch_dn2(int T, const void* plan, const float* hs, const uint8_t* dn, const float* s_dn, float* out, cudaStream_t st)
{
    int bytes = NW * S * (W16 ? 16 * ((SBD * NV_BLK + 8 + 15) / 16) * 16 : 576 * SBD);
    if (bytes + NW * 512 > 99 * 1024) return 4;
    set_smem(ink2_dn2<NW, S, SBD, ABL, W16>, bytes);
    ink2_dn2<NW, S, SBD, ABL, W16><<<D / 16, NW * 32, bytes, st>>>(T, (const Plan*)plan, dn, s_dn, out);
    return 0;
}



// down v3 (NVFP4): block-cooperative. One block = 16 output rows; the stage is the WHOLE contiguous 16 x 648 B tile of one union entry (10368 B, 16-byte aligned: fully coalesced 16-byte copies, one DRAM-friendly
// stream per block); the NW warps split K (18 blocks of 64 / NW per warp) and the entries run sequentially through a ring of S stages (one __syncthreads per entry). Per-warp accumulators
// (router weight x scale applied per entry) are summed over the warps in shared memory in fixed order (deterministic).
template <int NW, int S>
__global__ void __launch_bounds__(NW * 32) ink2_dn3(int T, const Plan* __restrict__ plan, const uint8_t* __restrict__ dn, const float* __restrict__ s_dn, float* __restrict__ out)
{
    extern __shared__ __align__(16) u32 smem[];
    constexpr int RB = FF / 64 * NV_BLK, TB = 16 * RB, NCH = TB / 16, BPW = FF / 64 / NW, TD = TB / 4;   // 648, 10368 bytes, 648 copies, blocks per warp, dwords per stage
    static_assert((FF / 64) % NW == 0, "K split");
    __shared__ float red[NW][4][32];
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_fires[1], 1u);
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int n0 = blockIdx.x * 16;
    int nu = plan->nu;
    auto issue = [&](int u, int st) {
        if (u < nu) {
            const uint8_t* src = dn + ((size_t)plan->eid[u] * D + n0) * RB;
            #pragma unroll
            for (int c = tid; c < NCH; c += NW * 32) cp_async16(reinterpret_cast<uint8_t*>(smem + st * TD) + 16 * c, src + 16 * c);
        }
        cp_commit();
    };
    #pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, s);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 1
    for (int u = 0; u < nu; u++) {
        cp_wait<S - 2>();
        __syncthreads();
        issue(u + S - 1, (u + S - 1) % S);
        const u32* ra = smem + (u % S) * TD + g * (RB / 4); const u32* rb = ra + 8 * (RB / 4);
        const uint4* xp = reinterpret_cast<const uint4*>(g_hh + (size_t)(u * 8 + g) * 576 + t * 8);
        float ca[4] = {0.f, 0.f, 0.f, 0.f}, cb[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int bb = 0; bb < BPW; bb++) {
            int b = wave * BPW + bb;
            u32 xb[8];
            uint4 x0 = xp[b * 8], x1 = xp[b * 8 + 1];
            xb[0] = x0.x; xb[1] = x0.y; xb[2] = x0.z; xb[3] = x0.w; xb[4] = x1.x; xb[5] = x1.y; xb[6] = x1.z; xb[7] = x1.w;
            blk_mma(ra[9 * b], ra[9 * b + 1 + 2 * t], ra[9 * b + 2 + 2 * t], rb[9 * b], rb[9 * b + 1 + 2 * t], rb[9 * b + 2 + 2 * t], xb, t, ca, cb);
        }
        int e = plan->eid[u];
        float sd = s_dn ? s_dn[e] : 1.0f;
        float cw0 = plan->w[u][2 * t] * sd, cw1 = plan->w[u][2 * t + 1] * sd;
        acc[0] += (ca[0] + cb[0]) * cw0; acc[1] += (ca[1] + cb[1]) * cw1; acc[2] += (ca[2] + cb[2]) * cw0; acc[3] += (ca[3] + cb[3]) * cw1;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) red[wave][i][lane] = acc[i];
    __syncthreads();
    if (wave == 0) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int w = 0; w < NW; w++)
            #pragma unroll
            for (int i = 0; i < 4; i++) v[i] += red[w][i][lane];
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            int m = 2 * t + q;
            if (m < T) { out[(size_t)m * D + n0 + g] = v[q] * NV_FIN; out[(size_t)m * D + n0 + g + 8] = v[2 + q] * NV_FIN; }
        }
    }
}
template <int NW, int S> static int launch_dn3(int T, const void* plan, const uint8_t* dn, const float* s_dn, float* out, cudaStream_t st)
{
    int bytes = S * 10368;
    if (bytes + NW * 512 > 99 * 1024) return 4;
    set_smem(ink2_dn3<NW, S>, bytes);
    ink2_dn3<NW, S><<<D / 16, NW * 32, bytes, st>>>(T, (const Plan*)plan, dn, s_dn, out);
    return 0;
}


// gate_up v3 (NVFP4): block-cooperative. One block = one 16 x 8 tile (8 gate rows j0.. + the 8 matching up rows, each half a CONTIGUOUS 10368 B region), the whole tile (20736 B) is copied with
// coalesced 16-byte cp.async, the NW warps split K (36 blocks of 64 / NW each) and are summed in shared memory in fixed warp order. S tile buffers per block (S = 1: no intra-block prefetch, the
// overlap comes from the co-resident blocks).
template <int NW, int S>
__global__ void __launch_bounds__(NW * 32) ink2_gu3(int T, const Plan* __restrict__ plan, const uint8_t* __restrict__ gu, const float* __restrict__ s_gu, float* __restrict__ hs, float beta, float lb)
{
    extern __shared__ __align__(16) u32 smem[];
    constexpr int RB = D / 64 * NV_BLK, HB = 8 * RB, BPW = D / 64 / NW, NCH = 2 * HB / 16;     // 1296, 10368 B per half tile, blocks per warp, 16-byte copies
    static_assert((D / 64) % NW == 0, "K split");
    __shared__ float red[NW][4][32];
    int u = blockIdx.y;
    if (u >= plan->nu) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_fires[0], 1u);
    int e = plan->eid[u];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int j0 = blockIdx.x * 8;
    const uint8_t* eb = gu + (size_t)e * (2 * FF) * RB;
    const uint8_t* sg = eb + (size_t)j0 * RB;
    const uint8_t* su = eb + (size_t)(j0 + FF) * RB;
    #pragma unroll
    for (int c = tid; c < NCH; c += NW * 32) cp_async16(reinterpret_cast<uint8_t*>(smem) + 16 * c, (c < NCH / 2 ? sg + 16 * c : su + 16 * (c - NCH / 2)));
    cp_commit();
    const uint4* xp = reinterpret_cast<const uint4*>(g_xh + (size_t)g * (36 * 4 * 8) + t * 8);
    cp_wait<0>();
    __syncthreads();
    const u32* ra = smem + g * (RB / 4); const u32* rb = ra + 8 * (RB / 4);
    float ca[4] = {0.f, 0.f, 0.f, 0.f}, cb[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int bb = 0; bb < BPW; bb++) {
        int b = wave * BPW + bb;
        u32 xb[8];
        uint4 x0 = xp[b * 8], x1 = xp[b * 8 + 1];
        xb[0] = x0.x; xb[1] = x0.y; xb[2] = x0.z; xb[3] = x0.w; xb[4] = x1.x; xb[5] = x1.y; xb[6] = x1.z; xb[7] = x1.w;
        blk_mma(ra[9 * b], ra[9 * b + 1 + 2 * t], ra[9 * b + 2 + 2 * t], rb[9 * b], rb[9 * b + 1 + 2 * t], rb[9 * b + 2 + 2 * t], xb, t, ca, cb);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) red[wave][i][lane] = ca[i] + cb[i];
    __syncthreads();
    if (wave == 0) {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int w = 0; w < NW; w++)
            #pragma unroll
            for (int i = 0; i < 4; i++) c[i] += red[w][i][lane];
        float scale = NV_FIN * (s_gu ? s_gu[e] : 1.0f);
        int j = j0 + g;
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            int m = 2 * t + q;
            if (m < T) {
                float gg = c[q] * scale, uu = c[2 + q] * scale;
                float act = beta * tanhf(gg / beta) * (1.0f / (1.0f + expf(-gg)));
                float upv = lb > 0.0f ? lb * tanhf(uu / lb) : uu;
                float hv = act * upv;
                hs[((size_t)u * T + m) * FF + j] = hv;
                int b = j >> 6, tt = (j >> 4) & 3, pp = j & 15;
                int slot = (pp & 1) + 2 * ((pp >> 3) & 1) + 4 * ((pp >> 2) & 1);
                reinterpret_cast<unsigned short*>(g_hh)[((size_t)(u * 8 + m) * 576 + (b * 4 + tt) * 8 + slot) * 2 + ((pp >> 1) & 1)] = __half_as_ushort(__float2half_rn(fminf(fmaxf(hv, -65504.0f), 65504.0f)));
            }
        }
    }
}
template <int NW, int S> static void launch_gu3(int T, const float* x, const void* plan, const uint8_t* gu, const float* s_gu, float* hs, float beta, float lb, cudaStream_t st)
{
    int bytes = 20736;
    set_smem(ink2_gu3<NW, S>, bytes);
    ink2_gu3<NW, S><<<dim3(FF / 8, 8 * T), NW * 32, bytes, st>>>(T, (const Plan*)plan, gu, s_gu, hs, beta, lb);
}

// ---- Q8_0 v2 (34-byte blocks, block PAIR = 68 B = 17 dwords; gate_up slab = 4 pairs = 272 B per row (16-byte copies), down slab = 2 pairs = 136 B (8-byte copies))
// B operand pre-converted by ink2_xprep_q8 / ink2_hprep_q8: per token [block of 32][t][4 u32] = the 4 fp16 pairs of values 8t .. 8t + 7 (natural order, as tile_dot_q8)
__global__ void __launch_bounds__(160) ink2_hprep_q8(int T, const Plan* __restrict__ plan, const float* __restrict__ hs)
{
    int u = blockIdx.x, m = blockIdx.y, i = threadIdx.x;  // i = block of 32 (36) x t (4)
    if (u >= plan->nu || i >= 144) return;
    u32 xb[4] = {0u, 0u, 0u, 0u};
    if (m < T) {
        const float4* xp = reinterpret_cast<const float4*>(hs + ((size_t)u * T + m) * FF + 32 * (i >> 2) + 8 * (i & 3));
        float4 a = xp[0], c = xp[1];
        xb[0] = pack_h(a.x, a.y); xb[1] = pack_h(a.z, a.w); xb[2] = pack_h(c.x, c.y); xb[3] = pack_h(c.z, c.w);
    }
    reinterpret_cast<uint4*>(g_hh)[(size_t)(u * 8 + m) * 144 + i] = make_uint4(xb[0], xb[1], xb[2], xb[3]);
}

// one block pair of a 16 x 8 tile: pa / pb = smem dwords of the pair of rows g / g + 8; xv0 / xv1 = B pairs of the two 32-blocks
__device__ __forceinline__ void pair_q8(const u32* pa, const u32* pb, uint4 xv0, uint4 xv1, int t, float acc[4])
{
    u32 wa[2][2], wb[2][2];
    wa[0][0] = __funnelshift_r(pa[2 * t], pa[2 * t + 1], 16); wa[0][1] = __funnelshift_r(pa[2 * t + 1], pa[2 * t + 2], 16);
    wa[1][0] = pa[9 + 2 * t]; wa[1][1] = pa[10 + 2 * t];
    wb[0][0] = __funnelshift_r(pb[2 * t], pb[2 * t + 1], 16); wb[0][1] = __funnelshift_r(pb[2 * t + 1], pb[2 * t + 2], 16);
    wb[1][0] = pb[9 + 2 * t]; wb[1][1] = pb[10 + 2 * t];
    float dA[2] = {h16f(pa[0] & 0xffffu), h16f(pa[8] >> 16)};
    float dB[2] = {h16f(pb[0] & 0xffffu), h16f(pb[8] >> 16)};
    #pragma unroll
    for (int blk = 0; blk < 2; blk++) {
        uint4 xv = blk ? xv1 : xv0;
        u32 xb[4] = {xv.x, xv.y, xv.z, xv.w};
        float cc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            u32 PA, QA, PB, QB;
            dec_q8(wa[blk][s], PA, QA);
            dec_q8(wb[blk][s], PB, QB);
            mma16816(cc, PA, PB, QA, QB, xb[2 * s], xb[2 * s + 1]);
        }
        acc[0] += cc[0] * dA[blk]; acc[1] += cc[1] * dA[blk]; acc[2] += cc[2] * dB[blk]; acc[3] += cc[3] * dB[blk];
    }
}

template <int NW, int S>
__global__ void __launch_bounds__(NW * 32) ink2_gu2q(int T, const Plan* __restrict__ plan, const uint8_t* __restrict__ gu, const float* __restrict__ s_gu, float* __restrict__ hs, float beta, float lb)
{
    extern __shared__ __align__(16) u32 smem[];
    constexpr int RB = D / 32 * 34, NSL = D / 64 / 4;             // 2448 bytes per row, 9 slabs of 4 pairs (272 B)
    int u = blockIdx.y;
    if (u >= plan->nu) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_fires[0], 1u);
    int e = plan->eid[u];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int j0 = (blockIdx.x * NW + wave) * 8;
    const uint8_t* eb = gu + (size_t)e * (2 * FF) * RB;
    u32* ring = smem + wave * S * 1088;
    uint32_t goff[9], soff[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        int q = lane + 32 * i, r = q / 17, cc = q - 17 * r;
        int row = r < 8 ? j0 + r : j0 + (r - 8) + FF;
        goff[i] = (uint32_t)row * RB + 16 * cc;
        soff[i] = r * 272 + 16 * cc;
    }
    auto issue = [&](int sl, int st) {
        if (sl < NSL) {
            #pragma unroll
            for (int i = 0; i < 9; i++)
                if (lane + 32 * i < 272) cp_async16(reinterpret_cast<uint8_t*>(ring + st * 1088) + soff[i], eb + goff[i] + sl * 272);
        }
        cp_commit();
    };
    #pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, s);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const uint4* xq = reinterpret_cast<const uint4*>(g_xh) + g * 288 + t;
    #pragma unroll 1
    for (int sl = 0; sl < NSL; sl++) {
        cp_wait<S - 2>();
        __syncwarp();
        issue(sl + S - 1, (sl + S - 1) % S);
        const u32* st = ring + (sl % S) * 1088;
        const u32* ra = st + g * 68; const u32* rb = ra + 544;
        #pragma unroll
        for (int pr = 0; pr < 4; pr++) {
            int bb = (sl * 4 + pr) * 2;
            pair_q8(ra + 17 * pr, rb + 17 * pr, xq[4 * bb], xq[4 * (bb + 1)], t, acc);
        }
    }
    float scale = (s_gu ? s_gu[e] : 1.0f);
    int j = j0 + g;
    #pragma unroll
    for (int q = 0; q < 2; q++) {
        int m = 2 * t + q;
        if (m < T) {
            float gg = acc[q] * scale, uu = acc[2 + q] * scale;
            float act = beta * tanhf(gg / beta) * (1.0f / (1.0f + expf(-gg)));
            float upv = lb > 0.0f ? lb * tanhf(uu / lb) : uu;
            float hv = act * upv;
            hs[((size_t)u * T + m) * FF + j] = hv;
            reinterpret_cast<unsigned short*>(g_hh)[(size_t)(u * 8 + m) * 1152 + j] = __half_as_ushort(__float2half_rn(fminf(fmaxf(hv, -65504.0f), 65504.0f)));
        }
    }
}

template <int NW, int S>
__global__ void __launch_bounds__(NW * 32) ink2_dn2q(int T, const Plan* __restrict__ plan, const uint8_t* __restrict__ dn, const float* __restrict__ s_dn, float* __restrict__ out)
{
    extern __shared__ __align__(16) u32 smem[];
    constexpr int RB = FF / 32 * 34, NSL = FF / 64 / 2;           // 1224 bytes per row, 9 slabs of 2 pairs (136 B)
    __shared__ float red[NW][4][32];
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_fires[1], 1u);
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int n0 = blockIdx.x * 16;
    int nu = plan->nu;
    int nit = nu > wave ? (nu - wave + NW - 1) / NW : 0;
    int total = nit * NSL;
    u32* ring = smem + wave * S * 544;
    uint32_t goff[9], soff[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        int q = lane + 32 * i, r = q / 17, cc = q - 17 * r;
        goff[i] = (uint32_t)r * RB + 8 * cc;
        soff[i] = r * 136 + 8 * cc;
    }
    auto issue = [&](int f, int st) {
        if (f < total) {
            int it = f / NSL, sl = f - it * NSL;
            const uint8_t* eb = dn + ((size_t)plan->eid[wave + it * NW] * D + n0) * RB + sl * 136;
            #pragma unroll
            for (int i = 0; i < 9; i++)
                if (lane + 32 * i < 272) cp_async8(reinterpret_cast<uint8_t*>(ring + st * 544) + soff[i], eb + goff[i]);
        }
        cp_commit();
    };
    #pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, s);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 1
    for (int f = 0; f < total; f++) {
        cp_wait<S - 2>();
        __syncwarp();
        issue(f + S - 1, (f + S - 1) % S);
        int it = f / NSL, sl = f - it * NSL, uu = wave + it * NW;
        const u32* st = ring + (f % S) * 544;
        const u32* ra = st + g * 34; const u32* rb = ra + 272;
        const uint4* xq = reinterpret_cast<const uint4*>(g_hh) + (size_t)(uu * 8 + g) * 144 + t;
        #pragma unroll
        for (int pr = 0; pr < 2; pr++) {
            int bb = (sl * 2 + pr) * 2;
            pair_q8(ra + 17 * pr, rb + 17 * pr, xq[4 * bb], xq[4 * (bb + 1)], t, c);
        }
        if (sl == NSL - 1) {
            int e = plan->eid[uu];
            float sd = s_dn ? s_dn[e] : 1.0f;
            float cw0 = plan->w[uu][2 * t] * sd, cw1 = plan->w[uu][2 * t + 1] * sd;
            acc[0] += c[0] * cw0; acc[1] += c[1] * cw1; acc[2] += c[2] * cw0; acc[3] += c[3] * cw1;
            #pragma unroll
            for (int i = 0; i < 4; i++) c[i] = 0.f;
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) red[wave][i][lane] = acc[i];
    __syncthreads();
    if (wave == 0) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int w = 0; w < NW; w++)
            #pragma unroll
            for (int i = 0; i < 4; i++) v[i] += red[w][i][lane];
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            int m = 2 * t + q;
            if (m < T) { out[(size_t)m * D + n0 + g] = v[q]; out[(size_t)m * D + n0 + g + 8] = v[2 + q]; }
        }
    }
}
template <int NW, int S> static void launch_gu2q(int T, const float* x, const void* plan, const uint8_t* gu, const float* s_gu, float* hs, float beta, float lb, cudaStream_t st)
{
    int bytes = NW * S * 4352;
    set_smem(ink2_gu2q<NW, S>, bytes);
    ink2_gu2q<NW, S><<<dim3(FF / 8 / NW, 8 * T), NW * 32, bytes, st>>>(T, (const Plan*)plan, gu, s_gu, hs, beta, lb);
}
template <int NW, int S> static int launch_dn2q(int T, const void* plan, const float* hs, const uint8_t* dn, const float* s_dn, float* out, cudaStream_t st)
{
    int bytes = NW * S * 2176;
    if (bytes + NW * 512 > 99 * 1024) return 4;
    set_smem(ink2_dn2q<NW, S>, bytes);
    ink2_dn2q<NW, S><<<D / 16, NW * 32, bytes, st>>>(T, (const Plan*)plan, dn, s_dn, out);
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------ host entry points
static int lerr() { return cudaGetLastError() == cudaSuccess ? 0 : 3; }

extern "C" {
int ink2_plan_launch(const int* sel, const float* wt, int T, void* plan, void* stream)
{
    ink2_plan<<<1, 64, 0, (cudaStream_t)stream>>>(sel, wt, T, (Plan*)plan);
    return lerr();
}
// plan + x pre-conversion (fmt 1 NVFP4 / 2 Q8_0) for the v2 gate_up kernels (cfg >= 16)
int ink2_plan_x_launch(const int* sel, const float* wt, int T, void* plan, const float* x, int fmt, void* stream)
{
    ink2_plan_x<<<1 + (fmt == 2 ? 18 : 9), 128, 0, (cudaStream_t)stream>>>(sel, wt, T, (Plan*)plan, x, fmt);
    return lerr();
}
// fp32 h scratch -> fp16 B fragments of the down kernels (needed only when h was not written by a v2 gate_up kernel)
int ink2_hprep_launch(int T, int fmt, const void* plan, const float* hs, void* stream)
{
    if (fmt == 2) ink2_hprep_q8<<<dim3(8 * T, 8), 160, 0, (cudaStream_t)stream>>>(T, (const Plan*)plan, hs);
    else ink2_hprep<<<dim3(8 * T, 8), 128, 0, (cudaStream_t)stream>>>(T, (const Plan*)plan, hs);
    return lerr();
}
}
template <bool Q8> static int run_gu(int T, int cfg, const float* x, const void* plan, const uint8_t* gu, const float* s_gu, float* hs, float beta, float lb, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    if (T < 1 || T > 8) return 2;
    if (cfg == 100) cfg = Q8 ? 23 : (T <= 4 ? 66 : 64);      // auto: best measured gate_up config per format / T (RTX 3090)
    if (cfg == 0) ink2_gu<8, (Q8 ? 3 : 6), Q8><<<dim3(FF / 8 / 8, 8 * T), 256, 0, st>>>(T, x, (const Plan*)plan, gu, s_gu, hs, beta, lb);
    else if (cfg == 1) ink2_gu<4, (Q8 ? 3 : 6), Q8><<<dim3(FF / 8 / 4, 8 * T), 128, 0, st>>>(T, x, (const Plan*)plan, gu, s_gu, hs, beta, lb);
    else if (cfg == 2) ink2_gu<8, (Q8 ? 2 : 4), Q8><<<dim3(FF / 8 / 8, 8 * T), 256, 0, st>>>(T, x, (const Plan*)plan, gu, s_gu, hs, beta, lb);
    else if (!Q8 && cfg >= 8 && cfg <= 11) {       // ablations of the cfg 2 geometry (NW 8 x CB 4), numerics invalid
        dim3 gr(FF / 8 / 8, 8 * T);
        if (cfg == 8) ink2_gu<8, 4, false, 1><<<gr, 256, 0, st>>>(T, x, (const Plan*)plan, gu, s_gu, hs, beta, lb);
        if (cfg == 9) ink2_gu<8, 4, false, 2><<<gr, 256, 0, st>>>(T, x, (const Plan*)plan, gu, s_gu, hs, beta, lb);
        if (cfg == 10) ink2_gu<8, 4, false, 3><<<gr, 256, 0, st>>>(T, x, (const Plan*)plan, gu, s_gu, hs, beta, lb);
        if (cfg == 11) ink2_gu<8, 4, false, 4><<<gr, 256, 0, st>>>(T, x, (const Plan*)plan, gu, s_gu, hs, beta, lb);
    }
    else if (Q8 && cfg >= 16) {                   // v2 Q8_0
        if (cfg == 16) launch_gu2q<4, 3>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else if (cfg == 17) launch_gu2q<4, 2>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else if (cfg == 18) launch_gu2q<8, 2>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else if (cfg == 19) launch_gu2q<2, 2>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else if (cfg == 20) launch_gu2q<2, 3>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else if (cfg == 21) launch_gu2q<3, 2>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else if (cfg == 22) launch_gu2q<6, 2>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else if (cfg == 23) launch_gu2q<1, 2>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else if (cfg == 24) launch_gu2q<4, 4>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else if (cfg == 25) launch_gu2q<6, 3>(T, x, plan, gu, s_gu, hs, beta, lb, st);
        else return 1;
    }
    else if (!Q8 && cfg >= 16) {                  // v2: cp.async slab pipeline
        cudaStream_t s2 = st;
        if (cfg == 16) launch_gu2<4, 3>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 17) launch_gu2<4, 4>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 18) launch_gu2<6, 3>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 19) launch_gu2<8, 2>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 20) launch_gu2<8, 3>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 21) launch_gu2<2, 4>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 22) launch_gu2<4, 2>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 23) launch_gu2<3, 4>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 24) launch_gu2<2, 2>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 25) launch_gu2<3, 2>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 26) launch_gu2<6, 2>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 27) launch_gu2<1, 2>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 28) launch_gu2<2, 3>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 29) launch_gu2<4, 5>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 66) launch_gu3<18, 1>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 60) launch_gu3<6, 1>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 61) launch_gu3<4, 1>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 62) launch_gu3<9, 1>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 63) launch_gu3<3, 1>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 64) launch_gu3<12, 1>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 65) launch_gu3<2, 1>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 50) launch_gu2<1, 2, 0, 12>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 51) launch_gu2<2, 2, 0, 12>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 52) launch_gu2<1, 3, 0, 12>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 53) launch_gu2<2, 3, 0, 12>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 54) launch_gu2<1, 2, 0, 36>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 55) launch_gu2<4, 2, 0, 12>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 40) launch_gu2<4, 2, 1>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 41) launch_gu2<4, 2, 2>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 42) launch_gu2<4, 2, 3>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 43) launch_gu2<8, 3, 1>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else if (cfg == 44) launch_gu2<8, 3, 2>(T, x, plan, gu, s_gu, hs, beta, lb, s2);
        else return 1;
    }
    else return 1;
    return lerr();
}
template <bool Q8> static int run_dn(int T, int cfg, const void* plan, const float* hs, const uint8_t* dn, const float* s_dn, float* out, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    int rc = 0;
    if (T < 1 || T > 8) return 2;
    if (cfg == 100) cfg = Q8 ? 23 : (T <= 4 ? 69 : 63);      // auto: best measured down config per format / T (RTX 3090)
    if (cfg == 0) ink2_dn<8, (Q8 ? 3 : 6), Q8><<<D / 16, 256, 0, st>>>(T, (const Plan*)plan, hs, dn, s_dn, out);
    else if (cfg == 1) ink2_dn<4, (Q8 ? 3 : 6), Q8><<<D / 16, 128, 0, st>>>(T, (const Plan*)plan, hs, dn, s_dn, out);
    else if (cfg == 2) ink2_dn<8, (Q8 ? 2 : 3), Q8><<<D / 16, 256, 0, st>>>(T, (const Plan*)plan, hs, dn, s_dn, out);
    else if (Q8 && cfg >= 16) {
        if (cfg == 16) rc = launch_dn2q<4, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 17) rc = launch_dn2q<8, 4>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 18) rc = launch_dn2q<6, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 19) rc = launch_dn2q<8, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 20) rc = launch_dn2q<8, 8>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 21) rc = launch_dn2q<4, 8>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 22) rc = launch_dn2q<8, 5>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 23) rc = launch_dn2q<4, 4>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 24) rc = launch_dn2q<6, 4>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 25) rc = launch_dn2q<12, 4>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 26) rc = launch_dn2q<16, 3>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 27) rc = launch_dn2q<2, 8>(T, plan, hs, dn, s_dn, out, st);
        else return 1;
    }
    else if (!Q8 && cfg >= 16) {
        if (cfg == 16) rc = launch_dn2<4, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 17) rc = launch_dn2<8, 4>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 18) rc = launch_dn2<6, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 19) rc = launch_dn2<8, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 20) rc = launch_dn2<8, 8>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 21) rc = launch_dn2<4, 8>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 22) rc = launch_dn2<8, 5>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 23) rc = launch_dn2<4, 4>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 24) rc = launch_dn2<6, 4>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 25) rc = launch_dn2<12, 4>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 26) rc = launch_dn2<16, 3>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 27) rc = launch_dn2<2, 8>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 28) rc = launch_dn2<4, 3, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 29) rc = launch_dn2<6, 3, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 30) rc = launch_dn2<8, 2, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 31) rc = launch_dn2<4, 4, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 32) rc = launch_dn2<8, 3, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 33) rc = launch_dn2<12, 2, 6>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 69) rc = launch_dn3<18, 3>(T, plan, dn, s_dn, out, st);
        else if (cfg == 70) rc = launch_dn3<9, 2>(T, plan, dn, s_dn, out, st);
        else if (cfg == 71) rc = launch_dn3<18, 4>(T, plan, dn, s_dn, out, st);
        else if (cfg == 72) rc = launch_dn3<9, 5>(T, plan, dn, s_dn, out, st);
        else if (cfg == 73) rc = launch_dn3<18, 2>(T, plan, dn, s_dn, out, st);
        else if (cfg == 60) rc = launch_dn3<6, 3>(T, plan, dn, s_dn, out, st);
        else if (cfg == 61) rc = launch_dn3<6, 4>(T, plan, dn, s_dn, out, st);
        else if (cfg == 62) rc = launch_dn3<3, 3>(T, plan, dn, s_dn, out, st);
        else if (cfg == 63) rc = launch_dn3<9, 3>(T, plan, dn, s_dn, out, st);
        else if (cfg == 64) rc = launch_dn3<2, 3>(T, plan, dn, s_dn, out, st);
        else if (cfg == 65) rc = launch_dn3<6, 2>(T, plan, dn, s_dn, out, st);
        else if (cfg == 66) rc = launch_dn3<9, 4>(T, plan, dn, s_dn, out, st);
        else if (cfg == 67) rc = launch_dn3<3, 4>(T, plan, dn, s_dn, out, st);
        else if (cfg == 68) rc = launch_dn3<6, 6>(T, plan, dn, s_dn, out, st);
        else if (cfg == 50) rc = launch_dn2<8, 4, 2, 0, true>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 51) rc = launch_dn2<4, 4, 2, 0, true>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 52) rc = launch_dn2<8, 3, 6, 0, true>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 53) rc = launch_dn2<4, 3, 6, 0, true>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 54) rc = launch_dn2<6, 4, 2, 0, true>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 55) rc = launch_dn2<8, 6, 2, 0, true>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 56) rc = launch_dn2<12, 3, 2, 0, true>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 57) rc = launch_dn2<6, 3, 6, 0, true>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 58) rc = launch_dn2<8, 4, 2, 2, true>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 40) rc = launch_dn2<8, 4, 2, 1>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 41) rc = launch_dn2<8, 4, 2, 2>(T, plan, hs, dn, s_dn, out, st);
        else if (cfg == 42) rc = launch_dn2<8, 4, 2, 3>(T, plan, hs, dn, s_dn, out, st);
        else return 1;
    }
    else return 1;
    return rc ? rc : lerr();
}
extern "C" {
int ink2_gu_launch(int T, int cfg, const float* x, const void* plan, const uint8_t* gu, const float* s_gu, float* hs, float beta, float lb, void* stream)
{ return run_gu<false>(T, cfg, x, plan, gu, s_gu, hs, beta, lb, stream); }
int ink2_dn_launch(int T, int cfg, const void* plan, const float* hs, const uint8_t* dn, const float* s_dn, float* out, void* stream)
{ return run_dn<false>(T, cfg, plan, hs, dn, s_dn, out, stream); }
int ink2_gu_q8_launch(int T, int cfg, const float* x, const void* plan, const uint8_t* gu, const float* s_gu, float* hs, float beta, float lb, void* stream)
{ return run_gu<true>(T, cfg, x, plan, gu, s_gu, hs, beta, lb, stream); }
int ink2_dn_q8_launch(int T, int cfg, const void* plan, const float* hs, const uint8_t* dn, const float* s_dn, float* out, void* stream)
{ return run_dn<true>(T, cfg, plan, hs, dn, s_dn, out, stream); }
int ink2_plan_bytes() { return (int)sizeof(Plan); }
}

// ------------------------------------------------------------------------------------------------------------------ path counters
__global__ void ink2_fires_copy(unsigned int* dst) { if (threadIdx.x < 4) dst[threadIdx.x] = g_fires[threadIdx.x]; }
extern "C" int ink2_fires_launch(unsigned int* dst, void* stream)
{
    ink2_fires_copy<<<1, 32, 0, (cudaStream_t)stream>>>(dst);
    return lerr();
}

// ------------------------------------------------------------------------------------------------------------------ read-bandwidth probe
// reads the gate_up + down bytes of the listed experts (each expert once, cold when the list spans > L2) with plain 16-byte loads, no math
__global__ void __launch_bounds__(256) ink2_readbw_exp(const uint4* __restrict__ gu, const uint4* __restrict__ dn, size_t gu_q, size_t dn_q, const int* __restrict__ ids, float* __restrict__ sink)
{
    size_t e = (size_t)ids[blockIdx.y];
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    const uint4* a = gu + e * gu_q;
    const uint4* b = dn + e * dn_q;
    uint4 acc = make_uint4(0, 0, 0, 0);
    #pragma unroll 4
    for (size_t i = tid; i < gu_q; i += stride) { uint4 v = __ldg(a + i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    #pragma unroll 4
    for (size_t i = tid; i < dn_q; i += stride) { uint4 v = __ldg(b + i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if ((acc.x | acc.y | acc.z | acc.w) == 0x12345678u) sink[0] = 1.0f;
}
extern "C" int ink2_readbw_launch(const void* gu, const void* dn, long gu_bytes, long dn_bytes, const int* ids, int n, float* sink, void* stream)
{
    ink2_readbw_exp<<<dim3(256, n), 256, 0, (cudaStream_t)stream>>>((const uint4*)gu, (const uint4*)dn, gu_bytes / 16, dn_bytes / 16, ids, sink);
    return lerr();
}
