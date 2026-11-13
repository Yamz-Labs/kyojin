// q8_1-activation decode kernels for K-quant + IQ4_XS weights
// port of the llama.cpp mmvq vmmq arms (ggml/src/ggml-cuda/vecdotq.cuh).
// Activations are quantized ONCE per row to q8_1 blocks by the shared
// fq_quant_q8_1_rows (fq_gemv_q8.cu) into the caller-cached global scratch,
// then row-per-block dots (fq_dot_q4_k_q8 / fq_dot_q5_k_q8 / fq_dot_q6_k_q8 /
// fq_dot_iq4_xs_q8 in fq_gemv.cuh) run INTEGER dp4a dots.
//
// Same portability contract as fq_gemv_q8.cu (no PTX, no warp shuffles,
// shared-memory reduction, fq_dp4a + fq_vsub4 builtins with scalar
// fallbacks off-CUDA).
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
#include "fq_kq_item.cuh"
#include "fq_api.h"

namespace {

// Row-per-block KQ q8 dot: block bidx computes output row n of x-row m
// against the packed K-quant matrix. Threads take sub-chunks round-robin;
// partials reduce sequentially (portable, same grid as the IQ2/NVFP4/Q8 row
// kernels in fq_gemv_q8.cu). FMT selects the dot (0 = Q4_K, 1 = Q5_K,
// 2 = Q6_K, 3 = IQ4_XS).
template <typename OUT, int FMT>
__global__ void fq_gemv_kq_kernel(const uint8_t* __restrict__ aqb,
                                  const uint8_t* __restrict__ packed,
                                  OUT* __restrict__ out,
                                  int M, int N, int K, int64_t packed_stride,
                                  int64_t qb_stride)
{
    int bidx = blockIdx.x;
    int m = bidx / N;
    int n = bidx % N;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    extern __shared__ float red[];

    float acc = 0.0f;
    if (n < N)
    {
        const uint8_t* wr = packed + (int64_t)n * packed_stride;
        const uint8_t* aq = aqb + (int64_t)m * qb_stride;
        if constexpr (FMT == 0)
            acc = fq_dot_q4_k_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 1)
            acc = fq_dot_q5_k_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 2)
            acc = fq_dot_q6_k_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 3)
            acc = fq_dot_iq4_xs_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 4)
            acc = fq_dot_iq3_xxs_q8(wr, aq, K, tid, nt);
        else if constexpr (FMT == 5)
            acc = fq_dot_iq2_s_q8(wr, aq, K, tid, nt);
        else
            acc = fq_dot_iq3_s_q8(wr, aq, K, tid, nt);
    }
    red[tid] = acc;
    __syncthreads();
    if (n < N && tid == 0)
    {
        float s = 0.0f;
        for (int l = 0; l < nt; l++)
            s = __fadd_rn(s, red[l]);
        if constexpr (std::is_same<OUT, __half>::value)
            out[(int64_t)m * N + n] = __float2half_rn(s);
        else
            out[(int64_t)m * N + n] = s;
    }
}

// ---------------------------------------------------------------------------
// Q4_K / Q6_K decode GEMV v2 ("rows" kernel): same float result bit for bit as
// fq_gemv_kq_kernel above, but built for memory-level parallelism.
//
// What the v1 kernel above does wrong on a GPU with a deep memory pipeline (RTX 3090: Q4_K ran at ~36 % of
// the DRAM roof, llama.cpp mmvq at ~70 %):
//  - its weight loads are memcpy() from a uint8_t*, which the compiler must treat as byte-aligned: LDG.U8
//    chains instead of one 16-byte load per thread;
//  - one 2.3 KB row per 256-thread block, and thread 0 then adds the 256 partials one after the other
//    (about 1 us of dependent adds), so a block spends half of its life with no load in flight.
//
// v2: an "item" is the work one thread does with ONE aligned wide load pair (Q4_K: two 16-byte loads = 4 of
// the v1 chunks; Q6_K: 4 chunks from u16 pairs, the block is 210 bytes so only 2-byte aligned, like ggml
// get_int_b2). Items of ROWS output rows go to shared memory as per-chunk float values; then ROWS threads
// (one per row) add them in EXACTLY the order v1 does (per-lane partial c = l, l + nt, ..., then lane by lane),
// so every output is bit-identical to v1 (the integer dp4a parts are exact; only the float order matters and
// it is kept). The serial add chain is now shared by ROWS rows per block and overlaps with the loads of the
// other resident blocks.
//
// Dot structure after ggml-cuda mmvq vec_dot_q4_K_q8_1 / vec_dot_q6_K_q8_1 (llama.cpp ggml/src/ggml-cuda/
// vecdotq.cuh, MIT, Copyright (c) 2023-2024 The ggml authors). Portable like the rest of fq_*: no PTX, no
// shuffles (HIP builds the same source; fq_dp4a is sudot4 there).
// ---------------------------------------------------------------------------

// fq_h2f_lo, fq_ld_u16x2, fq_kq_item_q4_k, fq_kq_item_q6_k: fq_kq_item.cuh (shared with fq_moe_kq.cu)

// One block = ROWS consecutive output rows of x-row m = blockIdx.x / nbn, 64 threads per row. Thread j of a row owns
// the items j, j + 64, j + 128, ... (one 256-chunk window of v1 per step); those items are chunks whose v1 lane
// (c mod nt, nt = 256 when K >= 4096 / 2048) is fixed, so thread j keeps four v1 lane partials in registers, adding
// the windows in increasing order exactly as v1's per-thread loop does. The nt lane partials go to shared memory
// (row stride 260 floats: the ROWS chain threads hit different banks), then thread r < ROWS adds row r lane by lane
// in v1's order. Lanes >= nsub (K small, nt padded to a multiple of 32) are written as 0.0f like v1's idle threads.
template <typename OUT, int FMT, int ROWS>
__global__ void __launch_bounds__(64 * ROWS) fq_gemv_kq_rows_kernel(const uint8_t* __restrict__ aqb,
                                                                    const uint8_t* __restrict__ packed,
                                                                    OUT* __restrict__ out,
                                                                    int N, int K, int64_t packed_stride, int64_t qb_stride,
                                                                    int nbn, int nt)
{
    constexpr int PS = 260;
    __shared__ __align__(16) float part[ROWS * PS];
    constexpr int IPB = (FMT == 0) ? 4 : 8;                    // items per 256-block
    const int tid = threadIdx.x;
    const int r = tid >> 6, j = tid & 63;
    const int m = blockIdx.x / nbn;
    const int n0 = (blockIdx.x - m * nbn) * ROWS;
    const int n = n0 + r;
    const int nitems = (K / 256) * IPB;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (n < N)
    {
        const uint8_t* aq = aqb + (int64_t)m * qb_stride;
        const uint8_t* wr = packed + (int64_t)n * packed_stride;
        for (int it = j; it < nitems; it += 64)
        {
            float t[4];
            if constexpr (FMT == 0)
                fq_kq_item_q4_k(wr, aq, it, t);
            else
                fq_kq_item_q6_k(wr, aq, it, t);
            #pragma unroll
            for (int q = 0; q < 4; q++)
                acc[q] = __fadd_rn(acc[q], t[q]);
        }
    }
    #pragma unroll
    for (int q = 0; q < 4; q++)
    {
        const int lane = (FMT == 0) ? (4 * j + q) : (32 * (j >> 3) + (j & 7) + 8 * q);
        if (lane < nt)
            part[r * PS + lane] = acc[q];
    }
    __syncthreads();
    if (tid < ROWS && n0 + tid < N)
    {
        const float4* pv = reinterpret_cast<const float4*>(part + tid * PS);
        float s = 0.0f;
        #pragma unroll 8
        for (int l = 0; l < nt / 4; l++)
        {
            const float4 v = pv[l];
            s = __fadd_rn(s, v.x);
            s = __fadd_rn(s, v.y);
            s = __fadd_rn(s, v.z);
            s = __fadd_rn(s, v.w);
        }
        if constexpr (std::is_same<OUT, __half>::value)
            out[(int64_t)m * N + n0 + tid] = __float2half_rn(s);
        else
            out[(int64_t)m * N + n0 + tid] = s;
    }
}


// ---------------------------------------------------------------------------
// Q4_K / Q6_K multi-row decode GEMV (R = 2..16 x-rows: multi-user decode, verify). Every weight block is loaded and
// unpacked ONCE per output row and dotted with all R q8_1 activation rows (the one-row kernels re-read the weights per
// x-row, the Triton rows kernels run at Triton speed; ggml mmvq does the same with ncols_dst <= 8). The weight half of
// fq_kq_item_q4_k / _q6_k is split from its activation half; the activation half repeats the item's float expressions in
// the same order, so output row m is bit-identical to the one-row kernel's output for x-row m: the batch composition
// never changes a result (tests/test_kq_gemv_mrows.py).
//
// Activations: the q8_1 blocks (36 bytes, 4-byte aligned) are read by 32 lanes with a 72-byte stride, and R rows multiply the loads.
// fq_kq_act_pack re-lays the R rows once per call (a tiny kernel, one thread per item) into "records" of 64 items (one window of the
// one-row kernel's thread loop) in structure-of-arrays form, so that lane j of a 64-thread row reads consecutive words (measured on the
// RTX 3090, 8B gate/up shape, 8 x-rows: 240 -> 154 us with this and the padded partial buffer):
//   Q4_K record (1088 words): qs[w * 64 + j], w = 0..15 = the 16 words of the item's two q8_1 blocks (block 2g words 1..8, block
//                             2g+1 words 1..8); dd[j] = d(block 2g) | d(block 2g+1) << 16 (the low halves of their word 0).
//   Q6_K record (576 words):  qs[s * 64 + j], s = 0..7 = word 1 + p of q8_1 block 8b + s; dd[s * 8 + (j >> 3)] = word 0 of that block.
// Dot structure after ggml-cuda mmvq vec_dot_q4_K_q8_1 / vec_dot_q6_K_q8_1 (llama.cpp ggml/src/ggml-cuda/vecdotq.cuh, MIT, Copyright
// (c) 2023-2024 The ggml authors), as the one-row items it mirrors. All integers are copied unchanged. The dot product's int -> float conversions use the exact magic-number form (every
// integer here is below 2^22 in magnitude), which runs on the integer/FP pipes instead of the quarter-rate conversion unit.
// ---------------------------------------------------------------------------

constexpr int FQ_KQ_REC_Q4 = 1088;
constexpr int FQ_KQ_REC_Q6 = 576;

template <int FMT>
__global__ void fq_kq_act_pack_kernel(const uint8_t* __restrict__ aqb, uint32_t* __restrict__ rec, int M, int nwin, int nitems,
                                      int64_t qb_stride)
{
    constexpr int RECW = (FMT == 0) ? FQ_KQ_REC_Q4 : FQ_KQ_REC_Q6;
    constexpr int IPB = (FMT == 0) ? 4 : 8;
    const int64_t tot = (int64_t)M * nwin * 64;
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= tot)
        return;
    const int j = (int)(gid & 63);
    const int64_t mw = gid >> 6;                       // m * nwin + win
    const int m = (int)(mw / nwin), win = (int)(mw - (int64_t)m * nwin);
    const int it = win * 64 + j;
    if (it >= nitems)
        return;
    uint32_t* r = rec + mw * RECW;
    const uint8_t* aq = aqb + (int64_t)m * qb_stride;
    const int b = it / IPB;
    if constexpr (FMT == 0)
    {
        const int g = it & 3;
        const uint32_t* a0 = reinterpret_cast<const uint32_t*>(aq + (size_t)(b * 8 + 2 * g) * FQ_Q8_1_BYTES);
        const uint32_t* a1 = a0 + FQ_Q8_1_BYTES / 4;
        #pragma unroll
        for (int w = 0; w < 8; w++)
        {
            r[w * 64 + j] = a0[1 + w];
            r[(8 + w) * 64 + j] = a1[1 + w];
        }
        r[1024 + j] = (a0[0] & 0xFFFFu) | (a1[0] << 16);
    }
    else
    {
        const int p = it & 7;
        #pragma unroll
        for (int s = 0; s < 8; s++)
        {
            const uint32_t* a = reinterpret_cast<const uint32_t*>(aq + (size_t)(b * 8 + s) * FQ_Q8_1_BYTES);
            r[s * 64 + j] = a[1 + p];
            if (p == 0)
                r[512 + s * 8 + (j >> 3)] = a[0];
        }
    }
}

__device__ __forceinline__ float fq_i2f_exact(int x)       // (float)x for |x| < 2^22, without the conversion unit
{
    return __int_as_float(0x4B400000 + x) - 12582912.0f;
}

struct FqKqW4 { uint32_t v0[4][2], v1[4][2]; float dm0, dm1; int sc[2], mn[2]; };

__device__ __forceinline__ void fq_kq_wld_q4_k(const uint8_t* __restrict__ wr, int it, FqKqW4& w)
{
    const int b = it >> 2, g = it & 3;
    const uint8_t* blk = wr + (size_t)b * FQ_Q4_K_BYTES;
    const uint32_t hd = *reinterpret_cast<const uint32_t*>(blk);
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(blk + 4);
    const uint32_t sw0 = sp[0], sw1 = sp[1], sw2 = sp[2];
    const uint4 va = *reinterpret_cast<const uint4*>(blk + 16 + 32 * g);
    const uint4 vb = *reinterpret_cast<const uint4*>(blk + 16 + 32 * g + 16);
    w.dm0 = fq_h2f_lo(hd);
    w.dm1 = fq_h2f_lo(hd >> 16);
    const int sh = 16 * (g & 1);
    const uint32_t s0 = (sw0 >> sh) & 0xFFFFu, s2 = (sw1 >> sh) & 0xFFFFu, s4 = (sw2 >> sh) & 0xFFFFu;
    const uint32_t hi = (g >= 2) ? 0xFFFFFFFFu : 0u;
    const uint16_t aux0 = (uint16_t)(((s0 & 0x3F3Fu) & ~hi) | ((((s4 >> 0) & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2)) & hi));
    const uint16_t aux1 = (uint16_t)(((s2 & 0x3F3Fu) & ~hi) | ((((s4 >> 4) & 0x0F0Fu) | ((s2 & 0xC0C0u) >> 2)) & hi));
    w.sc[0] = (int)(aux0 & 0xFFu); w.sc[1] = (int)((aux0 >> 8) & 0xFFu);
    w.mn[0] = (int)(aux1 & 0xFFu); w.mn[1] = (int)((aux1 >> 8) & 0xFFu);
    const uint32_t w0[4] = {va.x, va.y, va.z, va.w};
    const uint32_t w1[4] = {vb.x, vb.y, vb.z, vb.w};
    #pragma unroll
    for (int qo = 0; qo < 4; qo++)
    {
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            w.v0[qo][i] = (w0[qo] >> (4 * i)) & 0x0F0F0F0Fu;
            w.v1[qo][i] = (w1[qo] >> (4 * i)) & 0x0F0F0F0Fu;
        }
    }
}

// rec = record of (x-row, window) from fq_kq_act_pack, j = lane within the window (it & 63). WT weight rows share the activation
// loads. Same float expressions as fq_kq_item_q4_k.
template <int WT>
__device__ __forceinline__ void fq_kq_wdot_q4_k(const FqKqW4 (&w)[WT], const uint32_t* __restrict__ rec, int j, float (&dst)[WT][4])
{
    const uint32_t ddw = rec[1024 + j];
    const float d8[2] = {fq_h2f_lo(ddw), fq_h2f_lo(ddw >> 16)};
    float sumf_d[WT][4], sumf_m[WT][4];
    #pragma unroll
    for (int x = 0; x < WT; x++)
    {
        #pragma unroll
        for (int qo = 0; qo < 4; qo++)
        {
            sumf_d[x][qo] = 0.0f;
            sumf_m[x][qo] = 0.0f;
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        #pragma unroll
        for (int qo = 0; qo < 4; qo++)
        {
            const int32_t u0 = (int32_t)rec[(8 * i + qo) * 64 + j], u1 = (int32_t)rec[(8 * i + 4 + qo) * 64 + j];
            #pragma unroll
            for (int x = 0; x < WT; x++)
            {
                int dot1 = fq_dp4a((int32_t)w[x].v0[qo][i], u0, 0);
                dot1 = fq_dp4a((int32_t)w[x].v1[qo][i], u1, dot1);
                int dot2 = fq_dp4a(0x01010101, u0, 0);
                dot2 = fq_dp4a(0x01010101, u1, dot2);
                sumf_d[x][qo] = __fadd_rn(sumf_d[x][qo], __fmul_rn(d8[i], fq_i2f_exact(dot1 * w[x].sc[i])));
                sumf_m[x][qo] = __fadd_rn(sumf_m[x][qo], __fmul_rn(d8[i], fq_i2f_exact(dot2 * w[x].mn[i])));
            }
        }
    }
    #pragma unroll
    for (int x = 0; x < WT; x++)
    {
        #pragma unroll
        for (int qo = 0; qo < 4; qo++)
            dst[x][qo] = __fadd_rn(__fmul_rn(w[x].dm0, sumf_d[x][qo]), -__fmul_rn(w[x].dm1, sumf_m[x][qo]));
    }
}

struct FqKqW6 { uint32_t vi[4][2]; int sc[4][2]; float d; };

__device__ __forceinline__ void fq_kq_wld_q6_k(const uint8_t* __restrict__ wr, int it, FqKqW6& w)
{
    const int b = it >> 3, p = it & 7;
    const uint8_t* blk = wr + (size_t)b * FQ_Q6_K_BYTES;
    uint32_t vl[4];
    #pragma unroll
    for (int q = 0; q < 4; q++)
        vl[q] = fq_ld_u16x2(blk + 4 * p + 32 * q);
    const uint32_t vhw[2] = {fq_ld_u16x2(blk + 128 + 4 * p), fq_ld_u16x2(blk + 128 + 32 + 4 * p)};
    const int8_t* scl = reinterpret_cast<const int8_t*>(blk + 192);
    w.d = fq_h2f_lo((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 208));
    #pragma unroll
    for (int q = 0; q < 4; q++)
    {
        const int sc_off = 8 * (q >> 1) + (p >> 2) + 2 * (q & 1);
        const uint32_t vh = vhw[q >> 1] >> (2 * (q & 1));
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            w.sc[q][i] = (int)scl[sc_off + 4 * i];
            const uint32_t vil = (vl[q] >> (4 * i)) & 0x0F0F0F0Fu;
            const uint32_t vih = ((vh >> (4 * i)) << 4) & 0x30303030u;
            w.vi[q][i] = fq_vsub4(vil | vih, 0x20202020u);
        }
    }
}

template <int WT>
__device__ __forceinline__ void fq_kq_wdot_q6_k(const FqKqW6 (&w)[WT], const uint32_t* __restrict__ rec, int j, float (&dst)[WT][4])
{
    float sumf[WT][4];
    #pragma unroll
    for (int x = 0; x < WT; x++)
    {
        #pragma unroll
        for (int q = 0; q < 4; q++)
            sumf[x][q] = 0.0f;
    }
    #pragma unroll
    for (int q = 0; q < 4; q++)
    {
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            const int s = 4 * (q >> 1) + (q & 1) + 2 * i;
            const float d8 = fq_h2f_lo(rec[512 + s * 8 + (j >> 3)]);
            const int32_t u = (int32_t)rec[s * 64 + j];
            #pragma unroll
            for (int x = 0; x < WT; x++)
                sumf[x][q] = __fadd_rn(sumf[x][q], __fmul_rn(d8, fq_i2f_exact(fq_dp4a((int32_t)w[x].vi[q][i], u, 0) * w[x].sc[q][i])));
        }
    }
    #pragma unroll
    for (int x = 0; x < WT; x++)
    {
        #pragma unroll
        for (int q = 0; q < 4; q++)
            dst[x][q] = __fmul_rn(w[x].d, sumf[x][q]);
    }
}

// One block = WROWS x WT consecutive output rows x RT consecutive x-rows (x-row chunk blockIdx.y); a group of 64 threads handles WT
// output rows. Thread j owns items j, j + 64, ... exactly like the one-row kernel, but keeps WT x RT x 4 lane partials; x-rows >= M
// are skipped. The lane partials of every (output row, x-row) pair go to shared memory and one thread per pair adds them lane by
// lane in the one-row kernel's order.
template <typename OUT, int FMT, int WROWS, int RT, int WT>
__global__ void __launch_bounds__(64 * WROWS) fq_gemv_kq_mrows_kernel(const uint32_t* __restrict__ rec,
                                                                     const uint8_t* __restrict__ packed,
                                                                     OUT* __restrict__ out,
                                                                     int M, int N, int K, int64_t packed_stride,
                                                                     int nwin, int nt, int ny)
{
    constexpr int PS = 312;                              // 256 lanes + 8 floats of padding per 32 lanes (conflict-free partial stores)
    constexpr int RECW = (FMT == 0) ? FQ_KQ_REC_Q4 : FQ_KQ_REC_Q6;
    constexpr int NR = WROWS * WT;                       // output rows per block
    __shared__ __align__(16) float part[NR * RT * PS];
    constexpr int IPB = (FMT == 0) ? 4 : 8;
    using WTy = typename std::conditional<FMT == 0, FqKqW4, FqKqW6>::type;
    const int tid = threadIdx.x;
    const int r = tid >> 6, j = tid & 63;
    // x-row chunks of one weight tile are neighbouring blocks (they run together, the weights come from L2 for the later chunks)
    const int bid = blockIdx.x;
    const int m0 = (bid % ny) * RT;
    const int n0 = (bid / ny) * NR;
    const int nitems = (K / 256) * IPB;
    float acc[WT][RT][4];
    #pragma unroll
    for (int x = 0; x < WT; x++)
    {
        #pragma unroll
        for (int rr = 0; rr < RT; rr++)
        {
            #pragma unroll
            for (int q = 0; q < 4; q++)
                acc[x][rr][q] = 0.0f;
        }
    }
    if (n0 + r * WT < N)
    {
        const uint8_t* wrow[WT];
        #pragma unroll
        for (int x = 0; x < WT; x++)
            wrow[x] = packed + (int64_t)min(n0 + r * WT + x, N - 1) * packed_stride;        // rows past N are computed, never stored
        const uint32_t* rec0 = rec + (int64_t)m0 * nwin * RECW;
        for (int it = j; it < nitems; it += 64)
        {
            const int win = it >> 6;
            WTy w[WT];
            #pragma unroll
            for (int x = 0; x < WT; x++)
            {
                if constexpr (FMT == 0)
                    fq_kq_wld_q4_k(wrow[x], it, w[x]);
                else
                    fq_kq_wld_q6_k(wrow[x], it, w[x]);
            }
            #pragma unroll
            for (int rr = 0; rr < RT; rr++)
            {
                if (m0 + rr < M)
                {
                    float t[WT][4];
                    const uint32_t* rc = rec0 + ((int64_t)rr * nwin + win) * RECW;
                    if constexpr (FMT == 0)
                        fq_kq_wdot_q4_k<WT>(w, rc, j, t);
                    else
                        fq_kq_wdot_q6_k<WT>(w, rc, j, t);
                    #pragma unroll
                    for (int x = 0; x < WT; x++)
                    {
                        #pragma unroll
                        for (int q = 0; q < 4; q++)
                            acc[x][rr][q] = __fadd_rn(acc[x][rr][q], t[x][q]);
                    }
                }
            }
        }
    }
    if constexpr (FMT == 0)
    {
        // lanes 4j .. 4j+3 are contiguous: one 16-byte store per (output row, x-row)
        if (4 * j < nt)
        {
            #pragma unroll
            for (int x = 0; x < WT; x++)
            {
                #pragma unroll
                for (int rr = 0; rr < RT; rr++)
                    *reinterpret_cast<float4*>(part + (((r * WT + x) * RT) + rr) * PS + 4 * j + 8 * (j >> 3)) =
                        make_float4(acc[x][rr][0], acc[x][rr][1], acc[x][rr][2], acc[x][rr][3]);
            }
        }
    }
    else
    {
        #pragma unroll
        for (int q = 0; q < 4; q++)
        {
            const int lane = 32 * (j >> 3) + (j & 7) + 8 * q;
            if (lane < nt)
            {
                #pragma unroll
                for (int x = 0; x < WT; x++)
                {
                    #pragma unroll
                    for (int rr = 0; rr < RT; rr++)
                        part[(((r * WT + x) * RT) + rr) * PS + lane + 8 * (lane >> 5)] = acc[x][rr][q];
                }
            }
        }
    }
    __syncthreads();
    if (tid < NR * RT)
    {
        const int cr = tid / RT, rr = tid % RT;
        if (n0 + cr < N && m0 + rr < M)
        {
            const float4* pv = reinterpret_cast<const float4*>(part + (cr * RT + rr) * PS);
            float s = 0.0f;
            #pragma unroll 8
            for (int l = 0; l < nt / 4; l++)
            {
                const float4 v = pv[l + 2 * (l >> 3)];
                s = __fadd_rn(s, v.x);
                s = __fadd_rn(s, v.y);
                s = __fadd_rn(s, v.z);
                s = __fadd_rn(s, v.w);
            }
            if constexpr (std::is_same<OUT, __half>::value)
                out[(int64_t)(m0 + rr) * N + n0 + cr] = __float2half_rn(s);
            else
                out[(int64_t)(m0 + rr) * N + n0 + cr] = s;
        }
    }
}

}  // namespace

// BOLT_FQ_KQ_ROWS: output rows per block of the v2 Q4_K / Q6_K kernel (1, 2, 4). 0 = the v1 kernel (A/B); unset = the per-format
// default measured on the RTX 3090 (Q4_K 4 rows, Q6_K 1 row: its item has ~2x the load instructions, 64-thread blocks keep more
// of them resident: down 4096x12288 Q6_K 85 % of the DRAM roof at 1 row vs 63 % at 4). HIP default 1 row (Strix Halo, Qwen3-8B
// Q4_K_M served N1 v1 36.9 -> 1 row 39.1 / 4 rows 39.3 tok/s, same output; 35B-A3B unchanged: 53.3 / 53.3 / 53.1).
static int fq_kq_rows_env(int fmt)
{
    static const char* e = std::getenv("BOLT_FQ_KQ_ROWS");
    static const int forced = e ? std::atoi(e) : -1;
    if (forced >= 0)
        return forced;
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
    return 1;
#else
    return fmt == 0 ? 4 : 1;
#endif
}

template <typename OUT, int FMT>
static void fq_gemv_kq_rows_dispatch(int rows, int M, const uint8_t* qb, const uint8_t* packed, OUT* out, int N, int K,
                                     int64_t pstride, int64_t qb_stride, int nt, cudaStream_t stream)
{
    auto run = [&](auto rows_c)
    {
        constexpr int R = decltype(rows_c)::value;
        const int nbn = (N + R - 1) / R;
        fq_gemv_kq_rows_kernel<OUT, FMT, R><<<(unsigned)((int64_t)M * nbn), 64 * R, 0, stream>>>(
            qb, packed, out, N, K, pstride, qb_stride, nbn, nt);
    };
    if (rows >= 4) run(std::integral_constant<int, 4>());
    else if (rows >= 2) run(std::integral_constant<int, 2>());
    else run(std::integral_constant<int, 1>());
}

// BOLT_FQ_KQ_MROWS: x-rows 2..16 of the Q4_K / Q6_K decode GEMV run the multi-row kernel (weights read once for all rows). 0 = the
// per-row launch of the one-row kernels (A/B). Default on (CUDA and HIP).
// BOLT_FQ_KQ_MR_ROWS / BOLT_FQ_KQ_MR_RT force the output rows per block (1, 2, 4) / x-rows per pass (2, 4, 8, 16) for tuning.
static int fq_kq_mrows_env(const char* name, int dflt)
{
    const char* e = std::getenv(name);
    return e ? std::atoi(e) : dflt;
}

static bool fq_kq_mrows_on()
{
    static const int on = fq_kq_mrows_env("BOLT_FQ_KQ_MROWS", 1);
    return on != 0;
}

template <typename OUT, int FMT, int WROWS, int RT, int WT>
static void fq_gemv_kq_mrows_go(int M, const uint32_t* rec, const uint8_t* packed, OUT* out, int N, int K, int64_t pstride,
                                int nwin, int nt, cudaStream_t stream)
{
    const unsigned ny = (unsigned)((M + RT - 1) / RT);
    dim3 grid(((unsigned)((N + WROWS * WT - 1) / (WROWS * WT))) * ny);          // blockIdx.x = (weight tile, x-row chunk), chunk fastest
    fq_gemv_kq_mrows_kernel<OUT, FMT, WROWS, RT, WT><<<grid, 64 * WROWS, 0, stream>>>(rec, packed, out, M, N, K, pstride, nwin, nt, (int)ny);
}

// Launch shape: 2 x 2 output rows per block, 4 x-rows per pass (2 for R <= 2); measured on the RTX 3090 (tools/kq_mbench.py sweep).
template <typename OUT, int FMT>
static void fq_gemv_kq_mrows_dispatch(int M, const uint32_t* rec, const uint8_t* packed, OUT* out, int N, int K, int64_t pstride,
                                      int nwin, int nt, cudaStream_t stream)
{
    int rt = fq_kq_mrows_env("BOLT_FQ_KQ_MR_RT", 0);
    int wr = fq_kq_mrows_env("BOLT_FQ_KQ_MR_ROWS", 0);
    int wt = fq_kq_mrows_env("BOLT_FQ_KQ_MR_WT", 0);
    if (rt <= 0 && wr <= 0 && wt <= 0)
    {
        wr = 2; wt = 2; rt = M <= 2 ? 2 : 4;         // best or within 5 % of the best on every 8B shape for R >= 3 (R = 2 within 10 %)
    }
    if (rt <= 0) rt = M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : 16;
    if (wr <= 0) wr = 1;
    if (wt <= 0) wt = 1;
    auto go = [&](auto wc, auto rc, auto tc)
    {
        fq_gemv_kq_mrows_go<OUT, FMT, decltype(wc)::value, decltype(rc)::value, decltype(tc)::value>(M, rec, packed, out, N, K, pstride, nwin, nt, stream);
    };
    using I1 = std::integral_constant<int, 1>;
    using I2 = std::integral_constant<int, 2>;
    using I4 = std::integral_constant<int, 4>;
    using I8 = std::integral_constant<int, 8>;
    using I16 = std::integral_constant<int, 16>;
    if (wt >= 2)
    {
        if (rt <= 2) { if (wr >= 2) go(I2(), I2(), I2()); else go(I1(), I2(), I2()); }
        else { if (wr >= 2) go(I2(), I4(), I2()); else go(I1(), I4(), I2()); }
    }
    else if (wr >= 4)
    {
        if (rt <= 2) go(I4(), I2(), I1()); else go(I4(), I4(), I1());
    }
    else if (wr >= 2)
    {
        if (rt <= 2) go(I2(), I2(), I1()); else if (rt <= 4) go(I2(), I4(), I1()); else go(I2(), I8(), I1());
    }
    else
    {
        if (rt <= 2) go(I1(), I2(), I1()); else if (rt <= 4) go(I1(), I4(), I1()); else if (rt <= 8) go(I1(), I8(), I1()); else go(I1(), I16(), I1());
    }
}

// Shared row-per-block q8 launch sizing: threads take sub-chunks
// round-robin (same as fq_q8_nt in fq_gemv_q8.cu, generalized).
static inline int fq_kq_nt(int K, int sub)
{
    int nsub = K / sub;
    return nsub >= 256 ? 256 : ((nsub + 31) / 32) * 32;
}

template <int FMT, int SUB>
static void fq_gemv_kq_launch(torch::Tensor qb, torch::Tensor packed, torch::Tensor out,
                              int64_t block_bytes)
{
    TORCH_CHECK(qb.is_cuda() && packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(qb.is_contiguous() && packed.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(out.dtype() == torch::kHalf || out.dtype() == torch::kFloat,
                "out must be half or float");
    TORCH_CHECK_DIM(qb, 2);
    TORCH_CHECK_DIM(packed, 2);
    TORCH_CHECK_DIM(out, 2);
    int M = (int)qb.size(0);
    int N = (int)packed.size(0);
    TORCH_CHECK((int)out.size(0) == M, "out rows must match qb rows");
    TORCH_CHECK((int)out.size(1) == N, "out cols must match N");
    int K = (int)qb.size(1) / FQ_Q8_1_BYTES * FQ_Q8_1_GROUP;
    TORCH_CHECK(K % SUB == 0, "K must be a multiple of the dot sub-chunk");
    TORCH_CHECK((int)packed.size(1) == (K / FQ_Q4_K_GROUP) * block_bytes, "packed width must match K");
    int64_t pstride = (int64_t)(K / FQ_Q4_K_GROUP) * block_bytes;
    int64_t qb_stride = (int64_t)(K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
    int nt = fq_kq_nt(K, SUB);
    size_t shmem = (size_t)nt * sizeof(float);
    at::cuda::CUDAGuard guard(qb.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if constexpr (FMT == 0 || FMT == 2)
    {
        if (M >= 2 && M <= 16 && fq_kq_mrows_on() && K % 256 == 0 && (uintptr_t)packed.data_ptr() % ((FMT == 0) ? 16 : 2) == 0 &&
            (uintptr_t)qb.data_ptr() % 4 == 0 && pstride % ((FMT == 0) ? 16 : 2) == 0 && qb_stride % 4 == 0)
        {
            // multi-row kernel (per x-row bit-identical to the one-row kernels): weights read once for all M rows
            constexpr int IPB = (FMT == 0) ? 4 : 8;
            constexpr int RECW = (FMT == 0) ? FQ_KQ_REC_Q4 : FQ_KQ_REC_Q6;
            const int nitems = (K / 256) * IPB;
            const int nwin = (nitems + 63) / 64;
            torch::Tensor rect = torch::empty({(int64_t)M * nwin * RECW}, torch::TensorOptions().dtype(torch::kInt32).device(qb.device()));
            uint32_t* rec = reinterpret_cast<uint32_t*>(rect.data_ptr<int32_t>());
            fq_kq_act_pack_kernel<FMT><<<(unsigned)(((int64_t)M * nwin * 64 + 127) / 128), 128, 0, stream>>>(
                qb.data_ptr<uint8_t>(), rec, M, nwin, nitems, qb_stride);
            auto go = [&](auto* op)
            {
                fq_gemv_kq_mrows_dispatch<std::decay_t<decltype(*op)>, FMT>(
                    M, rec, packed.data_ptr<uint8_t>(), op, N, K, pstride, nwin, nt, stream);
            };
            if (out.dtype() == torch::kHalf)
                go(reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
            else
                go(out.data_ptr<float>());
            return;
        }
        // v2 rows kernel (bit-identical to the v1 kernel below): 16-byte aligned Q4_K rows, 2-byte aligned Q6_K rows
        const int rows = fq_kq_rows_env(FMT);
        const uintptr_t align = (FMT == 0) ? 16 : 2;
        if (rows > 0 && K % 256 == 0 && (uintptr_t)packed.data_ptr() % align == 0 && (uintptr_t)qb.data_ptr() % 4 == 0 &&
            pstride % align == 0)
        {
            auto go = [&](auto* op)
            {
                fq_gemv_kq_rows_dispatch<std::decay_t<decltype(*op)>, FMT>(
                    rows, M, qb.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(), op, N, K, pstride, qb_stride, nt, stream);
            };
            if (out.dtype() == torch::kHalf)
                go(reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
            else
                go(out.data_ptr<float>());
            return;
        }
    }
    auto launch = [&](auto* op)
    {
        fq_gemv_kq_kernel<std::decay_t<decltype(*op)>, FMT> <<<M * N, nt, shmem, stream>>>(
            qb.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(),
            op, M, N, K, pstride, qb_stride);
    };
    if (out.dtype() == torch::kHalf)
        launch(reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
    else
        launch(out.data_ptr<float>());
}

void fq_gemv_q4_k_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<0, 16>(qb, packed, out, FQ_Q4_K_BYTES);
}

void fq_gemv_q5_k_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<1, 16>(qb, packed, out, FQ_Q5_K_BYTES);
}

void fq_gemv_q6_k_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<2, 8>(qb, packed, out, FQ_Q6_K_BYTES);
}

void fq_gemv_iq4_xs_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<3, 32>(qb, packed, out, FQ_IQ4_XS_BYTES);
}

void fq_gemv_iq3_xxs_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<4, 32>(qb, packed, out, FQ_IQ3_XXS_BYTES);
}

void fq_gemv_iq2_s_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<5, 32>(qb, packed, out, FQ_IQ2_S_BYTES);
}

void fq_gemv_iq3_s_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    fq_gemv_kq_launch<6, 32>(qb, packed, out, FQ_IQ3_S_BYTES);
}
