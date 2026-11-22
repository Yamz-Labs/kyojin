// Inkling 2 Mini grouped routed-experts kernels for gfx1151 (Strix Halo), T = 1..8 rows, one resident copy of the GGUF bytes.
//   plan   : device-side union of the router picks (no host sync, graph capturable)
//   gate_up: per union entry, row j (gate) and row j + FF (up) of the stored [E, 2D, rowbytes] tensor, softcapped SwiGLU fused -> h [entry, row, FF]
//   down   : per output row, loop over the union entries, router weight x per-expert scale accumulated in registers, one lane reduce, plain store
// NVFP4 = GGUF type 40 AS STORED (36-byte blocks: 4 UE4M3 scale bytes + 4 x 8 data bytes, low nibble = value j, high nibble = value j + 8 of a 16-group).
// Decode: nibble -> fp16 bits ((n & 7) << 9) | ((n & 8) << 12) = E2M1 * 2^-14, UE4M3 byte -> fp16 bits (s & 0x7f) << 7 = scale * 2^-8; one v_dot2_f32_f16 per
// value pair against the fp16 activations in LDS; the final result is multiplied by 2^22.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
typedef unsigned int u32;

static constexpr int D = 2304, FF = 1152;
static constexpr int MAXU = 64;
static constexpr int NV_BLK = 36;                       // bytes per 64 values
static constexpr float NV_FIN = 4194304.0f;             // 2^22
static constexpr int Q8_BLK = 34;                       // bytes per 32 values

struct Plan { int nu; int pad; int eid[MAXU]; u32 mask[MAXU]; float w[MAXU][8]; };

__global__ void ink2_plan(const int* __restrict__ sel, const float* __restrict__ wt, int T, Plan* __restrict__ p)
{
    __shared__ int s[MAXU], first[MAXU], rk[MAXU];
    int i = threadIdx.x, n = T * 8;
    p->mask[i] = 0;
    for (int m = 0; m < 8; m++) p->w[i][m] = 0.0f;
    s[i] = i < n ? sel[i] : -1;
    __syncthreads();
    int d = i;
    if (i < n) { for (int j = 0; j < i; j++) if (s[j] == s[i]) { d = j; break; } }
    first[i] = (i < n && d == i) ? 1 : 0;
    __syncthreads();
    int r = 0;
    if (i < n) {
        for (int j = 0; j < d; j++) r += first[j];
        rk[i] = r;
        if (d == i) p->eid[r] = s[i];
    }
    __syncthreads();
    if (i < n) {
        atomicOr(&p->mask[r], 1u << (i >> 3));
        p->w[r][i >> 3] = wt[i];
    }
    if (i == 0) { int c = 0; for (int j = 0; j < n; j++) c += first[j]; p->nu = c; }
}

__device__ __forceinline__ u32 dec_pair(u32 t) { return ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12); }
__device__ __forceinline__ float fd(u32 a, u32 b, float c)
{
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2_t, a), __builtin_bit_cast(h2_t, b), c, false);
}
__device__ __forceinline__ float sc_of(u32 s) { return (float)__builtin_bit_cast(_Float16, (unsigned short)((s & 0x7f) << 7)); }

// one 16-group: 8 stored bytes (two dwords) against 16 activations laid out as pairs [lo02 lo13 hi02 hi13] per dword
__device__ __forceinline__ float dot_group(u32 w0, u32 w1, uint4 xa, uint4 xb)
{
    float p = fd(xa.x, dec_pair(w0), 0.0f);
    p = fd(xa.y, dec_pair(w0 >> 8), p);
    p = fd(xa.z, dec_pair(w0 >> 4), p);
    p = fd(xa.w, dec_pair(w0 >> 12), p);
    p = fd(xb.x, dec_pair(w1), p);
    p = fd(xb.y, dec_pair(w1 >> 8), p);
    p = fd(xb.z, dec_pair(w1 >> 4), p);
    p = fd(xb.w, dec_pair(w1 >> 12), p);
    return p;
}

__device__ __forceinline__ u32 pack_h(float a, float b)
{
    a = fminf(fmaxf(a, -65504.0f), 65504.0f);
    b = fminf(fmaxf(b, -65504.0f), 65504.0f);
    return (u32)__half_as_ushort(__float2half_rn(a)) | ((u32)__half_as_ushort(__float2half_rn(b)) << 16);
}

// activations of one group g (16 floats) -> the two uint4 of the LDS layout
__device__ __forceinline__ void stage_group(const float* __restrict__ xr, int g, uint4& A, uint4& B)
{
    const float4* p = reinterpret_cast<const float4*>(xr) + 4 * g;
    float4 a = p[0], b = p[1], c = p[2], d = p[3];
    float e[16] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
    A = make_uint4(pack_h(e[0], e[2]), pack_h(e[1], e[3]), pack_h(e[8], e[10]), pack_h(e[9], e[11]));
    B = make_uint4(pack_h(e[4], e[6]), pack_h(e[5], e[7]), pack_h(e[12], e[14]), pack_h(e[13], e[15]));
}

__device__ __forceinline__ float wave_sum(float v)
{
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

// ------------------------------------------------------------------------------------------------------------------ gate_up + situ
// grid (FF / (NW * UPW), 8 * T): blockIdx.y = union entry. A wave owns UPW consecutive hidden units; unit j reads rows j (gate) and j + FF (up).
template <int T, int NW, int UPW>
__global__ void __launch_bounds__(NW * 32) ink2_gu_nvfp4(const float* __restrict__ x, const Plan* __restrict__ plan, const uint8_t* __restrict__ gu,
                                                          const float* __restrict__ s_gu, float* __restrict__ hs, float beta, float lb)
{
    constexpr int G = D / 16;                 // 144
    constexpr int RB = D / 64 * NV_BLK;       // 1296 bytes per row
    int u = blockIdx.y;
    if (u >= plan->nu) return;
    int e = plan->eid[u];
    u32 mask = plan->mask[u];
    __shared__ uint4 xa[T][G], xb[T][G];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    #pragma unroll
    for (int m = 0; m < T; m++) {
        if (!((mask >> m) & 1)) continue;
        for (int g = tid; g < G; g += NW * 32) stage_group(x + (size_t)m * D, g, xa[m][g], xb[m][g]);
    }
    __syncthreads();
    float scale = NV_FIN * (s_gu ? s_gu[e] : 1.0f);
    const uint8_t* eb = gu + (size_t)e * (2 * FF) * RB;
    #pragma unroll
    for (int ui = 0; ui < UPW; ui++) {
        int j = (blockIdx.x * NW + wave) * UPW + ui;
        const u32* rg = reinterpret_cast<const u32*>(eb + (size_t)j * RB);
        const u32* ru = reinterpret_cast<const u32*>(eb + (size_t)(j + FF) * RB);
        constexpr int NIT = (G + 31) / 32;    // 5
        u32 g0[NIT], g1[NIT], u0[NIT], u1[NIT], sg[NIT], su[NIT];
        #pragma unroll
        for (int it = 0; it < NIT; it++) {
            int g = lane + 32 * it;
            if (g < G) {
                int blk = g >> 2, s4 = g & 3;
                const u32* a = rg + blk * 9, * b = ru + blk * 9;
                g0[it] = a[1 + 2 * s4]; g1[it] = a[2 + 2 * s4]; sg[it] = (a[0] >> (8 * s4)) & 0xff;
                u0[it] = b[1 + 2 * s4]; u1[it] = b[2 + 2 * s4]; su[it] = (b[0] >> (8 * s4)) & 0xff;
            }
        }
        float ag[T], au[T];
        #pragma unroll
        for (int m = 0; m < T; m++) { ag[m] = 0.0f; au[m] = 0.0f; }
        #pragma unroll
        for (int it = 0; it < NIT; it++) {
            int g = lane + 32 * it;
            if (g < G) {
                float fg = sc_of(sg[it]), fu = sc_of(su[it]);
                #pragma unroll
                for (int m = 0; m < T; m++) {
                    if ((mask >> m) & 1) {
                        uint4 A = xa[m][g], B = xb[m][g];
                        ag[m] += dot_group(g0[it], g1[it], A, B) * fg;
                        au[m] += dot_group(u0[it], u1[it], A, B) * fu;
                    }
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < T; m++) {
            if ((mask >> m) & 1) {
                float gg = wave_sum(ag[m]) * scale, uu = wave_sum(au[m]) * scale;
                if (lane == 0) {
                    float act = beta * tanhf(gg / beta) * (1.0f / (1.0f + expf(-gg)));
                    float upv = lb > 0.0f ? lb * tanhf(uu / lb) : uu;
                    hs[((size_t)u * T + m) * FF + j] = act * upv;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------ down
// grid (D / (NW * NPW)): a wave owns NPW output rows and loops over the union entries; h of the entry is staged in LDS (fp16 pairs).
template <int T, int NW, int NPW>
__global__ void __launch_bounds__(NW * 32) ink2_dn_nvfp4(const Plan* __restrict__ plan, const float* __restrict__ hs, const uint8_t* __restrict__ dn,
                                                          const float* __restrict__ s_dn, float* __restrict__ out)
{
    constexpr int G = FF / 16;                // 72
    constexpr int RB = FF / 64 * NV_BLK;      // 648 bytes per row
    constexpr int NIT = (G + 31) / 32;        // 3
    __shared__ uint4 xa[T][G], xb[T][G];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    int n0 = (blockIdx.x * NW + wave) * NPW;
    float acc[NPW][T];
    #pragma unroll
    for (int r = 0; r < NPW; r++)
        #pragma unroll
        for (int m = 0; m < T; m++) acc[r][m] = 0.0f;
    int nu = plan->nu;
    for (int u = 0; u < nu; u++) {
        int e = plan->eid[u];
        u32 mask = plan->mask[u];
        float sd = s_dn ? s_dn[e] : 1.0f;
        u32 w0[NPW][NIT], w1[NPW][NIT], sb[NPW][NIT];
        #pragma unroll
        for (int r = 0; r < NPW; r++) {
            const u32* rw = reinterpret_cast<const u32*>(dn + ((size_t)e * D + n0 + r) * RB);
            #pragma unroll
            for (int it = 0; it < NIT; it++) {
                int g = lane + 32 * it;
                if (g < G) {
                    int blk = g >> 2, s4 = g & 3;
                    const u32* a = rw + blk * 9;
                    w0[r][it] = a[1 + 2 * s4]; w1[r][it] = a[2 + 2 * s4]; sb[r][it] = (a[0] >> (8 * s4)) & 0xff;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < T; m++) {
            if (!((mask >> m) & 1)) continue;
            for (int g = tid; g < G; g += NW * 32) stage_group(hs + ((size_t)u * T + m) * FF, g, xa[m][g], xb[m][g]);
        }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < T; m++) {
            if (!((mask >> m) & 1)) continue;
            float cw = plan->w[u][m] * sd;
            #pragma unroll
            for (int r = 0; r < NPW; r++) {
                float a = 0.0f;
                #pragma unroll
                for (int it = 0; it < NIT; it++) {
                    int g = lane + 32 * it;
                    if (g < G) a += dot_group(w0[r][it], w1[r][it], xa[m][g], xb[m][g]) * sc_of(sb[r][it]);
                }
                acc[r][m] += a * cw;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < NPW; r++)
        #pragma unroll
        for (int m = 0; m < T; m++) {
            float v = wave_sum(acc[r][m]) * NV_FIN;
            if (lane == 0) out[(size_t)m * D + n0 + r] = v;
        }
}

// ------------------------------------------------------------------------------------------------------------------ host entry points
template <int T> static int run_gu(int cfg, dim3 gy, const float* x, const Plan* plan, const uint8_t* gu, const float* sg, float* hs, float beta, float lb, hipStream_t st)
{
    if (cfg == 0) { gy.x = FF / (8 * 1); ink2_gu_nvfp4<T, 8, 1><<<gy, 256, 0, st>>>(x, plan, gu, sg, hs, beta, lb); }
    else if (cfg == 1) { gy.x = FF / (8 * 2); ink2_gu_nvfp4<T, 8, 2><<<gy, 256, 0, st>>>(x, plan, gu, sg, hs, beta, lb); }
    else if (cfg == 2) { gy.x = FF / (4 * 4); ink2_gu_nvfp4<T, 4, 4><<<gy, 128, 0, st>>>(x, plan, gu, sg, hs, beta, lb); }
    else return 1;
    return 0;
}
template <int T> static int run_dn(int cfg, const Plan* plan, const float* hs, const uint8_t* dn, const float* sd, float* out, hipStream_t st)
{
    if (cfg == 0) ink2_dn_nvfp4<T, 8, 1><<<D / 8, 256, 0, st>>>(plan, hs, dn, sd, out);
    else if (cfg == 1) ink2_dn_nvfp4<T, 8, 2><<<D / 16, 256, 0, st>>>(plan, hs, dn, sd, out);
    else if (cfg == 2) ink2_dn_nvfp4<T, 4, 2><<<D / 8, 128, 0, st>>>(plan, hs, dn, sd, out);
    else return 1;
    return 0;
}

#define DISPATCH_T(FN, ...) switch (T) { case 1: return FN<1>(__VA_ARGS__); case 2: return FN<2>(__VA_ARGS__); case 3: return FN<3>(__VA_ARGS__); case 4: return FN<4>(__VA_ARGS__); \
    case 5: return FN<5>(__VA_ARGS__); case 6: return FN<6>(__VA_ARGS__); case 7: return FN<7>(__VA_ARGS__); case 8: return FN<8>(__VA_ARGS__); default: return 2; }

extern "C" {
int ink2_plan_launch(const int* sel, const float* wt, int T, void* plan, void* stream)
{
    ink2_plan<<<1, 64, 0, (hipStream_t)stream>>>(sel, wt, T, (Plan*)plan);
    return 0;
}
int ink2_gu_launch(int T, int cfg, const float* x, const void* plan, const uint8_t* gu, const float* s_gu, float* hs, float beta, float lb, void* stream)
{
    DISPATCH_T(run_gu, cfg, dim3(1, 8 * T), x, (const Plan*)plan, gu, s_gu, hs, beta, lb, (hipStream_t)stream)
}
int ink2_dn_launch(int T, int cfg, const void* plan, const float* hs, const uint8_t* dn, const float* s_dn, float* out, void* stream)
{
    DISPATCH_T(run_dn, cfg, (const Plan*)plan, hs, dn, s_dn, out, (hipStream_t)stream)
}
int ink2_plan_bytes() { return (int)sizeof(Plan); }
}

// ================================================================================================================== Q8_0 (as stored: 34-byte blocks = fp16 d + 32 int8)
// A lane owns one PAIR of blocks (68 bytes = 17 aligned dwords): block 0 data is shifted by 16 bits (funnel shift), block 1 data is aligned.
// int8 -> fp16 exactly: (b ^ 0x80) | 0x6400 = 1024 + b + 128, minus 1152 (v_perm + v_pk_add_f16), then one v_dot2_f32_f16 per value pair.
__device__ __forceinline__ float dot_dw(u32 wd, u32 xlo, u32 xhi, float p)
{
    u32 u = wd ^ 0x80808080u;
    u32 q0 = __builtin_amdgcn_perm(0x64646464u, u, 0x05010400u);
    u32 q1 = __builtin_amdgcn_perm(0x64646464u, u, 0x05030402u);
    const h2_t k = {(_Float16)1152.0f, (_Float16)1152.0f};
    h2_t v0 = __builtin_bit_cast(h2_t, q0) - k, v1 = __builtin_bit_cast(h2_t, q1) - k;
    p = fd(xlo, __builtin_bit_cast(u32, v0), p);
    return fd(xhi, __builtin_bit_cast(u32, v1), p);
}
__device__ __forceinline__ float dot_blk(const u32* d, uint4 X0, uint4 X1, uint4 X2, uint4 X3)
{
    float p = dot_dw(d[0], X0.x, X0.y, 0.0f);
    p = dot_dw(d[1], X0.z, X0.w, p);
    p = dot_dw(d[2], X1.x, X1.y, p);
    p = dot_dw(d[3], X1.z, X1.w, p);
    p = dot_dw(d[4], X2.x, X2.y, p);
    p = dot_dw(d[5], X2.z, X2.w, p);
    p = dot_dw(d[6], X3.x, X3.y, p);
    return dot_dw(d[7], X3.z, X3.w, p);
}
__device__ __forceinline__ float h16(u32 bits) { return (float)__builtin_bit_cast(_Float16, (unsigned short)bits); }
// one pair of blocks (17 dwords) against the 8 uint4 of fp16 activations
__device__ __forceinline__ float dot_pair_q8(const u32* a, const uint4* xq)
{
    u32 d0[8], d1[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) { d0[k] = __builtin_amdgcn_alignbit(a[k + 1], a[k], 16); d1[k] = a[9 + k]; }
    float s0 = dot_blk(d0, xq[0], xq[1], xq[2], xq[3]), s1 = dot_blk(d1, xq[4], xq[5], xq[6], xq[7]);
    return s0 * h16(a[0] & 0xffff) + s1 * h16(a[8] >> 16);
}
__device__ __forceinline__ void stage_q8(const float* __restrict__ xr, int q, uint4& A)
{
    const float4* p = reinterpret_cast<const float4*>(xr) + 2 * q;
    float4 a = p[0], b = p[1];
    A = make_uint4(pack_h(a.x, a.y), pack_h(a.z, a.w), pack_h(b.x, b.y), pack_h(b.z, b.w));
}

template <int T, int NW, int UPW>
__global__ void __launch_bounds__(NW * 32) ink2_gu_q8(const float* __restrict__ x, const Plan* __restrict__ plan, const uint8_t* __restrict__ gu,
                                                       const float* __restrict__ s_gu, float* __restrict__ hs, float beta, float lb)
{
    constexpr int NP = D / 64;               // 36 pairs per row
    constexpr int Q4 = D / 8;                // 288 uint4 per activation row
    constexpr int RB = D / 32 * Q8_BLK;      // 2448
    int u = blockIdx.y;
    if (u >= plan->nu) return;
    int e = plan->eid[u];
    u32 mask = plan->mask[u];
    __shared__ uint4 xs[T][Q4];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    #pragma unroll
    for (int m = 0; m < T; m++) {
        if (!((mask >> m) & 1)) continue;
        for (int q = tid; q < Q4; q += NW * 32) stage_q8(x + (size_t)m * D, q, xs[m][q]);
    }
    __syncthreads();
    float scale = s_gu ? s_gu[e] : 1.0f;
    const uint8_t* eb = gu + (size_t)e * (2 * FF) * RB;
    #pragma unroll
    for (int ui = 0; ui < UPW; ui++) {
        int j = (blockIdx.x * NW + wave) * UPW + ui;
        const u32* rg = reinterpret_cast<const u32*>(eb + (size_t)j * RB);
        const u32* ru = reinterpret_cast<const u32*>(eb + (size_t)(j + FF) * RB);
        constexpr int NIT = (NP + 31) / 32;  // 2
        u32 wg[NIT][17], wu[NIT][17];
        #pragma unroll
        for (int it = 0; it < NIT; it++) {
            int p = lane + 32 * it;
            if (p < NP) {
                #pragma unroll
                for (int k = 0; k < 17; k++) { wg[it][k] = rg[p * 17 + k]; wu[it][k] = ru[p * 17 + k]; }
            }
        }
        float ag[T], au[T];
        #pragma unroll
        for (int m = 0; m < T; m++) { ag[m] = 0.0f; au[m] = 0.0f; }
        #pragma unroll
        for (int it = 0; it < NIT; it++) {
            int p = lane + 32 * it;
            if (p < NP) {
                #pragma unroll
                for (int m = 0; m < T; m++) {
                    if ((mask >> m) & 1) {
                        ag[m] += dot_pair_q8(wg[it], &xs[m][8 * p]);
                        au[m] += dot_pair_q8(wu[it], &xs[m][8 * p]);
                    }
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < T; m++) {
            if ((mask >> m) & 1) {
                float gg = wave_sum(ag[m]) * scale, uu = wave_sum(au[m]) * scale;
                if (lane == 0) {
                    float act = beta * tanhf(gg / beta) * (1.0f / (1.0f + expf(-gg)));
                    float upv = lb > 0.0f ? lb * tanhf(uu / lb) : uu;
                    hs[((size_t)u * T + m) * FF + j] = act * upv;
                }
            }
        }
    }
}

template <int T, int NW, int NPW>
__global__ void __launch_bounds__(NW * 32) ink2_dn_q8(const Plan* __restrict__ plan, const float* __restrict__ hs, const uint8_t* __restrict__ dn,
                                                       const float* __restrict__ s_dn, float* __restrict__ out)
{
    constexpr int NP = FF / 64;              // 18 pairs per row
    constexpr int Q4 = FF / 8;               // 144
    constexpr int RB = FF / 32 * Q8_BLK;     // 1224
    __shared__ uint4 xs[T][Q4];
    int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    int n0 = (blockIdx.x * NW + wave) * NPW;
    float acc[NPW][T];
    #pragma unroll
    for (int r = 0; r < NPW; r++)
        #pragma unroll
        for (int m = 0; m < T; m++) acc[r][m] = 0.0f;
    int nu = plan->nu;
    for (int u = 0; u < nu; u++) {
        int e = plan->eid[u];
        u32 mask = plan->mask[u];
        float sd = s_dn ? s_dn[e] : 1.0f;
        u32 w[NPW][17];
        #pragma unroll
        for (int r = 0; r < NPW; r++) {
            const u32* rw = reinterpret_cast<const u32*>(dn + ((size_t)e * D + n0 + r) * RB);
            if (lane < NP) {
                #pragma unroll
                for (int k = 0; k < 17; k++) w[r][k] = rw[lane * 17 + k];
            }
        }
        #pragma unroll
        for (int m = 0; m < T; m++) {
            if (!((mask >> m) & 1)) continue;
            for (int q = tid; q < Q4; q += NW * 32) stage_q8(hs + ((size_t)u * T + m) * FF, q, xs[m][q]);
        }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < T; m++) {
            if (!((mask >> m) & 1)) continue;
            float cw = plan->w[u][m] * sd;
            #pragma unroll
            for (int r = 0; r < NPW; r++)
                if (lane < NP) acc[r][m] += dot_pair_q8(w[r], &xs[m][8 * lane]) * cw;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < NPW; r++)
        #pragma unroll
        for (int m = 0; m < T; m++) {
            float v = wave_sum(acc[r][m]);
            if (lane == 0) out[(size_t)m * D + n0 + r] = v;
        }
}

template <int T> static int run_gu_q8(int cfg, dim3 gy, const float* x, const Plan* plan, const uint8_t* gu, const float* sg, float* hs, float beta, float lb, hipStream_t st)
{
    if (cfg == 0) { gy.x = FF / (8 * 1); ink2_gu_q8<T, 8, 1><<<gy, 256, 0, st>>>(x, plan, gu, sg, hs, beta, lb); }
    else if (cfg == 1) { gy.x = FF / (8 * 2); ink2_gu_q8<T, 8, 2><<<gy, 256, 0, st>>>(x, plan, gu, sg, hs, beta, lb); }
    else if (cfg == 2) { gy.x = FF / (4 * 4); ink2_gu_q8<T, 4, 4><<<gy, 128, 0, st>>>(x, plan, gu, sg, hs, beta, lb); }
    else return 1;
    return 0;
}
template <int T> static int run_dn_q8(int cfg, const Plan* plan, const float* hs, const uint8_t* dn, const float* sd, float* out, hipStream_t st)
{
    if (cfg == 0) ink2_dn_q8<T, 8, 1><<<D / 8, 256, 0, st>>>(plan, hs, dn, sd, out);
    else if (cfg == 1) ink2_dn_q8<T, 8, 2><<<D / 16, 256, 0, st>>>(plan, hs, dn, sd, out);
    else if (cfg == 2) ink2_dn_q8<T, 4, 2><<<D / 8, 128, 0, st>>>(plan, hs, dn, sd, out);
    else return 1;
    return 0;
}
extern "C" {
int ink2_gu_q8_launch(int T, int cfg, const float* x, const void* plan, const uint8_t* gu, const float* s_gu, float* hs, float beta, float lb, void* stream)
{
    DISPATCH_T(run_gu_q8, cfg, dim3(1, 8 * T), x, (const Plan*)plan, gu, s_gu, hs, beta, lb, (hipStream_t)stream)
}
int ink2_dn_q8_launch(int T, int cfg, const void* plan, const float* hs, const uint8_t* dn, const float* s_dn, float* out, void* stream)
{
    DISPATCH_T(run_dn_q8, cfg, (const Plan*)plan, hs, dn, s_dn, out, (hipStream_t)stream)
}
}

// ------------------------------------------------------------------------------------------------------------------ read-bandwidth probe
// reads the gate_up + down bytes of the listed experts (each expert once, cold when the list spans > 32 MB of MALL) with plain 16-byte loads, no math
__global__ void __launch_bounds__(256) ink2_readbw_exp(const uint4* __restrict__ gu, const uint4* __restrict__ dn, size_t gu_q, size_t dn_q, const int* __restrict__ ids, float* __restrict__ sink)
{
    size_t e = (size_t)ids[blockIdx.y];
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    const uint4* a = gu + e * gu_q;
    const uint4* b = dn + e * dn_q;
    uint4 acc = make_uint4(0, 0, 0, 0);
    #pragma unroll 4
    for (size_t i = tid; i < gu_q; i += stride) { uint4 v = a[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    #pragma unroll 4
    for (size_t i = tid; i < dn_q; i += stride) { uint4 v = b[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if ((acc.x | acc.y | acc.z | acc.w) == 0x12345678u) sink[0] = 1.0f;
}
extern "C" int ink2_readbw_launch(const void* gu, const void* dn, long gu_bytes, long dn_bytes, const int* ids, int n, float* sink, void* stream)
{
    ink2_readbw_exp<<<dim3(256, n), 256, 0, (hipStream_t)stream>>>((const uint4*)gu, (const uint4*)dn, gu_bytes / 16, dn_bytes / 16, ids, sink);
    return 0;
}
