// Inkling 2 Mini attention-projection kernel for gfx1151 (Strix Halo): ONE launch for every projection that shares an input, T = 1..8 rows,
// weights read ONCE for all T rows. Three storage formats behind one segment table:
//   F_Q8  Q8_0 split at load: int8 data [N][K] (natural order) + fp16 scales [N][K/32]  (same bytes as GGUF type 8: 34 B per 32 values)
//   F_NV  NVFP4 quantised at load: nibbles [N][K/2] in the "unit" order below + UE4M3 scales [N][K/16] (one per 16 values) + one fp32 per segment (gs)
//         (0.5625 B per value: 36 B per 64, the same density as GGUF type 40)
//   F_BF  BF16 as stored [N][K]
// A segment = (weights, rows N, K, input index xi, fp32 output [T][ldo], gs). A block works inside ONE segment (stages that segment's input in LDS as fp16),
// so q / gate / rel / sink (input xq), k / k_conv_proj (xk), v / v_conv_proj (xv) are one launch; o_proj is another (K = 3072).
// Per wave: RW = 32 / G rows; G lanes per row, lane j owns units j, j + G, ... (NU units, all loads issued before any math).
//   unit: Q8 16 values (one uint4 of int8 + one fp16 scale shared by two units), NV 32 values (one uint4 of nibbles + two scale bytes), BF 8 values (one uint4).
// NV nibble order inside a unit (32 values, 4 dwords d): value e = 8 d + 2 o + m sits in dword d at bit 4 o + 16 m (o = 0..3, m = 0..1); the decode of
// one dword is dec_pair(w >> 4 o) = fp16x2 (value 8d+2o, value 8d+2o+1) * 2^-14 against the NATURAL-order fp16 pair of the activations (one v_dot2_f32_f16).
// UE4M3 scale byte s -> fp16 bits (s & 0x7f) << 7 = scale * 2^-8: out = gs * 2^22 * sum_group(scale_fp16 * dot).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef STAGE_FIRST
#define STAGE_FIRST 0         // 1: stage x + barrier BEFORE the weight loads are issued (the barrier then does not wait for them)
#endif
#ifndef WPB
#define WPB 4                 // waves per block (compile-time; the micro job builds 4 and 8)
#endif

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
typedef unsigned int u32;

enum { F_Q8 = 0, F_NV = 1, F_BF = 2 };
static constexpr int MAXSEG = 12;

struct Seg {
    const uint8_t* w; const uint8_t* sc; float* out;
    int rows, K, ldo, xi, fmt, blk0, nblk;
    float gs;
};
struct Params {
    Seg s[MAXSEG];
    const float* x[3];
    int nseg, T, total_blocks, pad;
};

__device__ unsigned int g_fires = 0;
extern __shared__ __attribute__((aligned(16))) unsigned char lds[];

__device__ __forceinline__ float fd(u32 a, u32 b, float c)
{
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2_t, a), __builtin_bit_cast(h2_t, b), c, false);
}
#ifndef NVLUT
#define NVLUT 0               // 0: E2M1 as fp16 SUBNORMAL-based bit trick (value * 2^-14, scale * 2^-8, final x 2^22); 1: exact E2M1 through a v_perm byte LUT (no fp16 denormals anywhere)
#endif
#if NVLUT
__device__ __forceinline__ u32 dec_pair(u32 t)
{
    // fp16 high bytes of 0 .5 1 1.5 2 3 4 6 = 00 38 3C 3E 40 42 44 46 (low bytes 0): selector byte 0x0c = constant 0
    const u32 sel = ((t & 0x00070007u) << 8) | 0x000c000cu;
    return __builtin_amdgcn_perm(0x46444240u, 0x3E3C3800u, sel) | ((t & 0x00080008u) << 12);
}
__device__ __forceinline__ float sc_of(u32 s)
{
    const u32 s7 = s & 0x7f, e = s7 >> 3, m = s7 & 7;
    const float f1 = __uint_as_float(((e + 120u) << 23) | (m << 20));          // (1 + m/8) 2^(e-7)
    const float f0 = (float)m * (1.0f / 512.0f);                                // m 2^-9
    return e == 0 ? f0 : f1;
}
static constexpr float NV_FIN = 1.0f;
#else
__device__ __forceinline__ u32 dec_pair(u32 t) { return ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12); }
__device__ __forceinline__ float sc_of(u32 s) { return (float)__builtin_bit_cast(_Float16, (unsigned short)((s & 0x7f) << 7)); }
static constexpr float NV_FIN = 4194304.0f;     // 2^22
#endif
__device__ __forceinline__ u32 pack_h(float a, float b)
{
    a = fminf(fmaxf(a, -65504.0f), 65504.0f);
    b = fminf(fmaxf(b, -65504.0f), 65504.0f);
    return (u32)__half_as_ushort(__float2half_rn(a)) | ((u32)__half_as_ushort(__float2half_rn(b)) << 16);
}

template <int FMT, int K>
__device__ __forceinline__ void body(const Params& P, const Seg& S, int bi)
{
    constexpr int G = FMT == F_Q8 ? 16 : (FMT == F_NV ? 8 : 32);          // lanes per row
    constexpr int UE = FMT == F_Q8 ? 16 : (FMT == F_NV ? 32 : 8);         // values per unit
    constexpr int UPR = K / UE;
    constexpr int NU = UPR / G;
    constexpr int RW = 32 / G;
    static_assert(UPR % G == 0, "units per row must divide by the lane group");
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int grp = lane / G, j = lane % G;
    const int row = (bi * WPB + wave) * RW + grp;
    const bool ok = row < S.rows;
    const int rl = ok ? row : 0;
    const int T = P.T;

    constexpr int PU = UE / 8;                                            // 16-byte x pieces (8 values) per unit
    const float4* xg = reinterpret_cast<const float4*>(P.x[S.xi]);
    // x layout in LDS (piece-major, conflict-free for consecutive units): byte ((t * PU + m) * UPR + u) * 16 = 8 fp16 values of unit u, piece m
    auto stage = [&]() {
        for (int p = tid; p < T * UPR * PU; p += WPB * 32) {
            const int m = p % PU, r = p / PU, u = r % UPR, t = r / UPR;
            const float4* src = xg + ((size_t)t * K + (size_t)u * UE + 8 * m) / 4;
            const float4 v0 = src[0], v1 = src[1];
            *reinterpret_cast<uint4*>(lds + ((size_t)(t * PU + m) * UPR + u) * 16) = make_uint4(pack_h(v0.x, v0.y), pack_h(v0.z, v0.w), pack_h(v1.x, v1.y), pack_h(v1.z, v1.w));
        }
    };
#if STAGE_FIRST
    stage();
    __syncthreads();
#endif
    uint4 wq[NU];
    unsigned sv[NU];
    if constexpr (FMT == F_Q8) {
        const uint8_t* wr = S.w + (size_t)rl * K;
        const unsigned short* sr = reinterpret_cast<const unsigned short*>(S.sc) + (size_t)rl * (K / 32);
        #pragma unroll
        for (int i = 0; i < NU; i++) { const int u = j + G * i; wq[i] = *reinterpret_cast<const uint4*>(wr + 16 * u); sv[i] = sr[u >> 1]; }
    } else if constexpr (FMT == F_NV) {
        const uint8_t* wr = S.w + (size_t)rl * (K / 2);
        const unsigned short* sr = reinterpret_cast<const unsigned short*>(S.sc + (size_t)rl * (K / 16));
        #pragma unroll
        for (int i = 0; i < NU; i++) { const int u = j + G * i; wq[i] = *reinterpret_cast<const uint4*>(wr + 16 * u); sv[i] = sr[u]; }
    } else {
        const uint8_t* wr = S.w + (size_t)rl * (K * 2);
        #pragma unroll
        for (int i = 0; i < NU; i++) { const int u = j + G * i; wq[i] = *reinterpret_cast<const uint4*>(wr + 16 * u); sv[i] = 0; }
    }
#if !STAGE_FIRST
    stage();
    __syncthreads();
#endif

    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.0f;

    #pragma unroll
    for (int i = 0; i < NU; i++) {
        const int u = j + G * i;
        if constexpr (FMT == F_Q8) {
            const u32 wd[4] = {wq[i].x, wq[i].y, wq[i].z, wq[i].w};
            u32 w[8];
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                const u32 v = wd[d] ^ 0x80808080u;
                const u32 a = __builtin_amdgcn_perm(0x64646464u, v, 0x04010400u);
                const u32 b = __builtin_amdgcn_perm(0x64646464u, v, 0x04030402u);
                const h2_t k1152 = {(_Float16)1152.0f, (_Float16)1152.0f};
                w[2 * d] = __builtin_bit_cast(u32, __builtin_bit_cast(h2_t, a) - k1152);
                w[2 * d + 1] = __builtin_bit_cast(u32, __builtin_bit_cast(h2_t, b) - k1152);
            }
            const float dsc = (float)__builtin_bit_cast(_Float16, (unsigned short)sv[i]);
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                if (t < T) {
                    const uint4 xa = *reinterpret_cast<const uint4*>(lds + ((size_t)(t * PU + 0) * UPR + u) * 16);
                    const uint4 xb = *reinterpret_cast<const uint4*>(lds + ((size_t)(t * PU + 1) * UPR + u) * 16);
                    float p = fd(w[0], xa.x, 0.0f);
                    p = fd(w[1], xa.y, p); p = fd(w[2], xa.z, p); p = fd(w[3], xa.w, p);
                    p = fd(w[4], xb.x, p); p = fd(w[5], xb.y, p); p = fd(w[6], xb.z, p); p = fd(w[7], xb.w, p);
                    acc[t] = fmaf(dsc, p, acc[t]);
                }
            }
        } else if constexpr (FMT == F_NV) {
            const u32 wd[4] = {wq[i].x, wq[i].y, wq[i].z, wq[i].w};
            const float s0 = sc_of(sv[i] & 0xff), s1 = sc_of(sv[i] >> 8);
            u32 wv[16];                                                       // decoded fp16 pairs, shared by all T rows
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                wv[4 * d + 0] = dec_pair(wd[d]); wv[4 * d + 1] = dec_pair(wd[d] >> 4); wv[4 * d + 2] = dec_pair(wd[d] >> 8); wv[4 * d + 3] = dec_pair(wd[d] >> 12);
            }
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                if (t < T) {
                    float pa = 0.0f, pb = 0.0f;
                    #pragma unroll
                    for (int m = 0; m < 4; m++) {
                        const uint4 xm = *reinterpret_cast<const uint4*>(lds + ((size_t)(t * PU + m) * UPR + u) * 16);
                        float p = fd(xm.x, wv[4 * m + 0], 0.0f);
                        p = fd(xm.y, wv[4 * m + 1], p);
                        p = fd(xm.z, wv[4 * m + 2], p);
                        p = fd(xm.w, wv[4 * m + 3], p);
                        if (m < 2) pa += p; else pb += p;
                    }
                    acc[t] = fmaf(s0, pa, fmaf(s1, pb, acc[t]));
                }
            }
        } else {
            const u32 wd[4] = {wq[i].x, wq[i].y, wq[i].z, wq[i].w};
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                if (t < T) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(lds + ((size_t)t * UPR + u) * 16);
                    const u32 xd[4] = {xv.x, xv.y, xv.z, xv.w};
                    float a = acc[t];
                    #pragma unroll
                    for (int d = 0; d < 4; d++) {
                        const h2_t h = __builtin_bit_cast(h2_t, xd[d]);
                        a = fmaf(__uint_as_float(wd[d] << 16), (float)h[0], a);
                        a = fmaf(__uint_as_float(wd[d] & 0xffff0000u), (float)h[1], a);
                    }
                    acc[t] = a;
                }
            }
        }
    }

    const float fin = S.gs * (FMT == F_NV ? NV_FIN : 1.0f);
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        if (t < T) {
            float v = acc[t];
            #pragma unroll
            for (int o = G / 2; o >= 1; o >>= 1) v += __shfl_xor(v, o, 32);
            if (j == 0 && ok) S.out[(size_t)t * S.ldo + row] = v * fin;
        }
    }
}

__global__ void __launch_bounds__(WPB * 32) ink2_ap(const Params P)
{
    const int b = blockIdx.x;
    int s = 0;
    #pragma unroll
    for (int i = 1; i < MAXSEG; i++) if (i < P.nseg && b >= P.s[i].blk0) s = i;
    const Seg& S = P.s[s];
    if (b == 0 && threadIdx.x == 0) atomicAdd(&g_fires, 1u);
    const int bi = b - S.blk0;
    const bool big = S.K == 3072;
    switch (S.fmt) {
        case F_Q8: if (big) body<F_Q8, 3072>(P, S, bi); else body<F_Q8, 2304>(P, S, bi); break;
        case F_NV: if (big) body<F_NV, 3072>(P, S, bi); else body<F_NV, 2304>(P, S, bi); break;
        default:   if (big) body<F_BF, 3072>(P, S, bi); else body<F_BF, 2304>(P, S, bi); break;
    }
}

// pure-read probe over the very same bytes (data + scales of every segment): the measured floor of this access pattern. 8 independent 16-byte loads per thread per step.
__device__ __forceinline__ void rd8(const uint4* __restrict__ a, size_t nvec, size_t tid, size_t stride, uint4& acc)
{
    for (size_t i = tid; i < nvec; i += stride * 8) {
        uint4 v[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) { const size_t idx = i + (size_t)k * stride; v[k] = idx < nvec ? a[idx] : make_uint4(0, 0, 0, 0); }
        #pragma unroll
        for (int k = 0; k < 8; k++) { acc.x ^= v[k].x; acc.y ^= v[k].y; acc.z ^= v[k].z; acc.w ^= v[k].w; }
    }
}
__global__ void __launch_bounds__(256) ink2_ap_read(const Params P, float* __restrict__ sink)
{
    uint4 acc = make_uint4(0, 0, 0, 0);
    const size_t tid = (size_t)blockIdx.x * 256 + threadIdx.x, stride = (size_t)gridDim.x * 256;
    for (int s = 0; s < P.nseg; s++) {
        const Seg& S = P.s[s];
        const size_t nw = S.fmt == F_Q8 ? (size_t)S.rows * S.K : (S.fmt == F_NV ? (size_t)S.rows * S.K / 2 : (size_t)S.rows * S.K * 2);
        const size_t ns = S.fmt == F_BF ? 0 : (size_t)S.rows * S.K / 16;
        rd8(reinterpret_cast<const uint4*>(S.w), nw / 16, tid, stride, acc);
        if (ns) rd8(reinterpret_cast<const uint4*>(S.sc), ns / 16, tid, stride, acc);
    }
    if ((acc.x | acc.y | acc.z | acc.w) == 0x12345678u) sink[0] = 1.0f;
}

__global__ void __launch_bounds__(256) ink2_ap_readraw(const uint4* __restrict__ a, size_t nvec, float* __restrict__ sink)
{
    uint4 acc = make_uint4(0, 0, 0, 0);
    rd8(a, nvec, (size_t)blockIdx.x * 256 + threadIdx.x, (size_t)gridDim.x * 256, acc);
    if ((acc.x | acc.y | acc.z | acc.w) == 0x12345678u) sink[0] = 1.0f;
}

extern "C" {
int ink2_ap_sizeof_params() { return (int)sizeof(Params); }
int ink2_ap_launch(const Params* P, int smem_bytes, void* stream)
{
    ink2_ap<<<dim3(P->total_blocks), dim3(WPB * 32), smem_bytes, (hipStream_t)stream>>>(*P);
    return (int)hipGetLastError();
}
int ink2_ap_read_launch(const Params* P, float* sink, int blocks, void* stream)
{
    ink2_ap_read<<<dim3(blocks), dim3(256), 0, (hipStream_t)stream>>>(*P, sink);
    return (int)hipGetLastError();
}
int ink2_ap_readraw_launch(const void* p, long nbytes, float* sink, int blocks, void* stream)
{
    ink2_ap_readraw<<<dim3(blocks), dim3(256), 0, (hipStream_t)stream>>>((const uint4*)p, (size_t)(nbytes / 16), sink);
    return (int)hipGetLastError();
}
int ink2_ap_wpb() { return WPB; }
int ink2_ap_nvlut() { return NVLUT; }
int ink2_ap_stagefirst() { return STAGE_FIRST; }
unsigned int ink2_ap_fires(int reset)
{
    unsigned int v = 0;
    hipDeviceSynchronize();
    hipMemcpyFromSymbol(&v, HIP_SYMBOL(g_fires), sizeof(v));
    if (reset) { unsigned int z = 0; hipMemcpyToSymbol(HIP_SYMBOL(g_fires), &z, sizeof(z)); }
    return v;
}
}
