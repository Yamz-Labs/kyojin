// Few-row MMQ (2..24 activation rows) for the dense GGUF Q8_0 / Q4_K / Q5_K / Q6_K linears: int8 MMA (m16n8k32 / m16n8k16) on raw
// GGUF weight rows streamed through a per-warp cp.async ring, activations quantized to q8_1 in the MMA fragment layout.
//
// Ported from llama.cpp ggml/src/ggml-cuda/mmq-few.cu and quantize.cu (quantize_mmq_few_q8_1), PR #31533 "CUDA: few-row MMQ for
// BS 2-24" (https://github.com/ggml-org/llama.cpp, merge commit 0fb8f15770, release v1.0.0).
// MIT License, Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.md at the repository root.
// Changes: only the Q8_0 / Q4_K / Q5_K / Q6_K decoders; one channel and sample (no broadcast strides); half or float output; the
// tile / mma / cp.async helpers of mma.cuh / cp-async.cuh inlined as PTX (Ampere+ only); no PDL, no DGX Spark L2 prefetch;
// the src1 input is half (converted to float before the same quantization arithmetic). The decoder and kernel bodies are unchanged.
// NVIDIA only (sm_80+): this file is not built on ROCm (foreign_quant is excluded there, see build_config.py).
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "ggml_mmvq/ggml_vecdotq_k46.cuh"

namespace {

using namespace ggml_iq;

enum { MF_Q8_0 = 0, MF_Q4_K = 1, MF_Q5_K = 2, MF_Q6_K = 3 };

struct block_q8_0_mf { half d; int8_t qs[32]; };
static_assert(sizeof(block_q8_0_mf) == 34, "ggml block layout");

constexpr int MMQ_FEW_MAX_NCG   = 3;  // max. number of column groups
constexpr int MMQ_FEW_NROWS     = 16; // src0 rows of one MMA tile
constexpr int MMQ_FEW_NWARPS    = 4;
constexpr int MMQ_FEW_NCOLS     = 8;  // src1 columns per column group
constexpr int MF_WARP           = 32;
constexpr int MMQ_FEW_Q8_1_INT4 = 4*MF_WARP + 4*4; // int4 per 256 values and column group
constexpr int MF_QUANT_BLOCK    = 128;             // CUDA_QUANTIZE_BLOCK_SIZE_MMQ

// src0 bytes per 256 values
constexpr __host__ __device__ int mmq_few_unit_size(const int type) {
    return type == MF_Q8_0 ? 8*34 : type == MF_Q4_K ? 144 : type == MF_Q5_K ? 176 : type == MF_Q6_K ? 210 : 0;
}

// Q4_K and Q5_K also need the src1 sums for their mins (DS4 layout: half2 (d, sum)), the others take float d (D4).
constexpr __host__ __device__ bool mmq_few_ds4(const int type) {
    return type == MF_Q4_K || type == MF_Q5_K;
}

// Units of 256 values per ring stage: 1 for Q6_K, 3 column groups (fewer free registers) and 32-row tiles (2 blocks per SM), else 2.
constexpr __host__ __device__ int mmq_few_nunits_stage(const int type, const int ncg, const int nmt) {
    return type == MF_Q6_K || ncg > 2 || nmt > 1 ? 1 : 2;
}

// A third ring stage only makes Q6_K faster, the other types get slower on large matrices.
constexpr __host__ __device__ int mmq_few_nstages(const int type) {
    return type == MF_Q6_K ? 3 : 2;
}

// The src0 data of a ring stage does not fill whole 16 byte chunks, it is copied from the preceding 16 byte boundary.
constexpr __host__ __device__ bool mmq_few_unaligned(const int type, const int nunits_stage) {
    return (nunits_stage*mmq_few_unit_size(type)) % 16 != 0;
}

// for unaligned data the max. over the 8 possible start offsets
constexpr __host__ __device__ int mmq_few_nchunks(const int type, const int nunits_stage) {
    return mmq_few_unaligned(type, nunits_stage) ? (14 + nunits_stage*mmq_few_unit_size(type) + 15)/16 : nunits_stage*mmq_few_unit_size(type)/16;
}

// Unaligned rows are padded to an odd number of 16 byte chunks against bank conflicts, for aligned rows the padding costs more than it saves.
constexpr __host__ __device__ int mmq_few_sram_stride(const int type, const int nunits_stage) {
    return mmq_few_unaligned(type, nunits_stage) && mmq_few_nchunks(type, nunits_stage) % 2 == 0
        ? 16*(mmq_few_nchunks(type, nunits_stage) + 1) : 16*mmq_few_nchunks(type, nunits_stage);
}

constexpr __host__ __device__ int mmq_few_nbytes_shared(const int type, const int ncg, const int nmt) {
    return MMQ_FEW_NWARPS*mmq_few_nstages(type)*nmt*MMQ_FEW_NROWS*mmq_few_sram_stride(type, mmq_few_nunits_stage(type, ncg, nmt));
}

// With Q6_K and Q8_0 two CUDA blocks with 32-row tiles would not fit into an SM.
constexpr __host__ __device__ bool mmq_few_tall_supported(const int type) {
    return type == MF_Q4_K || type == MF_Q5_K;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define MF_AVAILABLE
#endif

#ifdef MF_AVAILABLE

// --- mma.cuh / cp-async.cuh pieces (NVIDIA Ampere+) ---

// tile<16, 8, int> C: x[0], x[1] row g = lane/4, columns 2t, 2t+1 (t = lane%4); x[2], x[3] row g+8
struct mf_c_tile { int x[4] = {0, 0, 0, 0}; };

__device__ __forceinline__ int mf_get_i(const int l) { return ((l/2) % 2)*8 + threadIdx.x/4; }
__device__ __forceinline__ int mf_get_j(const int l) { return 2*(threadIdx.x % 4) + l % 2; }

__device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }
template <int n> __device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" : : "n"(n)); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); }

template <int width>
__device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

template <int width>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

// mmq-load-tiles.cuh
__device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

// --- mmq-few.cu ---

// The src1 fragments of a lane for one unit of 256 values, see the quantize kernel below.
struct mmq_few_y_frag {
    int4 qs[4];
    int4 ds[4];
};

__device__ __forceinline__ void mmq_few_load_y(mmq_few_y_frag & yf, const int4 * y) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.qs[j] = y[j*MF_WARP + threadIdx.x];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.ds[j] = y[4*MF_WARP + 4*(threadIdx.x % 4) + j];
    }
}

__device__ __forceinline__ int2 mmq_few_get_b(const mmq_few_y_frag & yf, const int s) {
    const int4 q = yf.qs[s/2];
    return s % 2 == 0 ? make_int2(q.x, q.y) : make_int2(q.z, q.w);
}

template <bool ds4>
__device__ __forceinline__ float2 mmq_few_get_d(const mmq_few_y_frag & yf, const int s) {
    if constexpr (!ds4) {
        const float * d = (const float *) &yf.ds[s/2];
        return make_float2(d[2*(s % 2) + 0], d[2*(s % 2) + 1]);
    } else {
        const half2 * ds = (const half2 *) &yf.ds[s/2];
        return make_float2(__low2float(ds[2*(s % 2) + 0]), __low2float(ds[2*(s % 2) + 1]));
    }
}

__device__ __forceinline__ float2 mmq_few_get_sum(const mmq_few_y_frag & yf, const int s) {
    const half2 * ds = (const half2 *) &yf.ds[s/2];
    return make_float2(__high2float(ds[2*(s % 2) + 0]), __high2float(ds[2*(s % 2) + 1]));
}

// C.x[0], C.x[1]: row g, columns 2t and 2t+1. C.x[2], C.x[3]: row g+8.
__device__ __forceinline__ mf_c_tile mmq_few_mma(const int (&lo)[2], const int (&hi)[2], const int2 b) {
    mf_c_tile C;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(C.x[0]), "+r"(C.x[1]), "+r"(C.x[2]), "+r"(C.x[3])
        : "r"(lo[0]), "r"(lo[1]), "r"(hi[0]), "r"(hi[1]), "r"(b.x), "r"(b.y));
    return C;
}

__device__ __forceinline__ mf_c_tile mmq_few_mma_k16(const int (&a)[2], const int b) {
    mf_c_tile C;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(C.x[0]), "+r"(C.x[1]), "+r"(C.x[2]), "+r"(C.x[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
    return C;
}

// sum_d += d * (C * ls) of sub-block s for all column groups, ls are the src0 row scales
template <int type, int ncg, typename scale_t>
__device__ __forceinline__ void mmq_few_mma_add(
        const int (&lo)[2], const int (&hi)[2], const scale_t (&ls)[2], const mmq_few_y_frag (&yf)[ncg], const int s, float (&sum_d)[ncg][4]) {
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
        const mf_c_tile C = mmq_few_mma(lo, hi, mmq_few_get_b(yf[c], s));

        const float2 d = mmq_few_get_d<mmq_few_ds4(type)>(yf[c], s);
        sum_d[c][0] += d.x * float(C.x[0]*ls[0]);
        sum_d[c][1] += d.y * float(C.x[1]*ls[0]);
        sum_d[c][2] += d.x * float(C.x[2]*ls[1]);
        sum_d[c][3] += d.y * float(C.x[3]*ls[1]);
    }
}

// Same with one k=16 MMA per half of sub-block s, the int8 scales of the halves are bytes 0 and 1 of sc[r].
template <int type, int ncg>
__device__ __forceinline__ void mmq_few_mma_k16_add(
        const int (&lo)[2], const int (&hi)[2], const int (&sc)[2], const mmq_few_y_frag (&yf)[ncg], const int s, float (&sum_d)[ncg][4]) {
    const int sc_lo[2] = {(int8_t) sc[0], (int8_t) sc[1]};
    const int sc_hi[2] = {(int8_t) (sc[0] >> 8), (int8_t) (sc[1] >> 8)};
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
        const int2 b = mmq_few_get_b(yf[c], s);
        const mf_c_tile C_lo = mmq_few_mma_k16(lo, b.x);
        const mf_c_tile C_hi = mmq_few_mma_k16(hi, b.y);

        const float2 d = mmq_few_get_d<mmq_few_ds4(type)>(yf[c], s);
        sum_d[c][0] += d.x * float(C_lo.x[0]*sc_lo[0] + C_hi.x[0]*sc_hi[0]);
        sum_d[c][1] += d.y * float(C_lo.x[1]*sc_lo[0] + C_hi.x[1]*sc_hi[0]);
        sum_d[c][2] += d.x * float(C_lo.x[2]*sc_lo[1] + C_hi.x[2]*sc_hi[1]);
        sum_d[c][3] += d.y * float(C_lo.x[3]*sc_lo[1] + C_hi.x[3]*sc_hi[1]);
    }
}

template <int ncg>
__device__ __forceinline__ void mmq_few_add_scaled(float (&sum)[ncg][4], const float (&d)[2], const float (&sum_d)[ncg][4]) {
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[c][l] += d[l/2]*sum_d[c][l];
        }
    }
}

// vec_dot: one unit of 256 values of the 16 rows of a warp, x[0] and x[1] are the rows g and g+8 of lane (g, t).
// The lane decodes the values [4t, 4t + 4) and [16 + 4t, 16 + 4t + 4) of each sub-block, its A fragments.

template <int type, int ncg>
__device__ __forceinline__ void mmq_few_vec_dot_q45_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    using block_t = std::conditional_t<type == MF_Q4_K, block_q4_K, block_q5_K>;
    const int t = threadIdx.x % 4;

    int   qs[2][4][2];
    [[maybe_unused]] int qh[2][2]; // Q5_K: bit s holds the high bit of sub-block s
    int   sc[2][2];
    int   mn[2][2];
    float d[2];
    float dmin[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const block_t * bx = (const block_t *) x[r];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            qs[r][j][0] = get_int_b4(bx->qs + 32*j,      t);
            qs[r][j][1] = get_int_b4(bx->qs + 32*j + 16, t);
        }
        if constexpr (type == MF_Q5_K) {
            qh[r][0] = get_int_b4(bx->qh,      t);
            qh[r][1] = get_int_b4(bx->qh + 16, t);
        }
        const int4 hdr = *(const int4 *) bx; // dm and the 12 scale bytes
        const int scales[3] = {hdr.y, hdr.z, hdr.w};
        sc[r][0] = unpack_scales_q45_K(scales, 0);
        sc[r][1] = unpack_scales_q45_K(scales, 1);
        mn[r][0] = unpack_scales_q45_K(scales, 2);
        mn[r][1] = unpack_scales_q45_K(scales, 3);
        const float2 dm = __half22float2(bx->dm);
        d[r]    = dm.x;
        dmin[r] = dm.y;
    }

    float sum_d[ncg][4] = {{0.0f}};
    float sum_m[ncg][4] = {{0.0f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int j     = s/2;
        const int shift = 4*(s % 2);
        int lo[2];
        int hi[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            lo[r] = (qs[r][j][0] >> shift) & 0x0F0F0F0F;
            hi[r] = (qs[r][j][1] >> shift) & 0x0F0F0F0F;
            if constexpr (type == MF_Q5_K) {
                lo[r] |= ((qh[r][0] >> s) & 0x01010101) << 4;
                hi[r] |= ((qh[r][1] >> s) & 0x01010101) << 4;
            }
        }
        const int   ls[2] = {(sc[0][s/4] >> (8*(s % 4))) & 0xFF, (sc[1][s/4] >> (8*(s % 4))) & 0xFF};
        const float m0    = (mn[0][s/4] >> (8*(s % 4))) & 0xFF;
        const float m1    = (mn[1][s/4] >> (8*(s % 4))) & 0xFF;
        mmq_few_mma_add<type>(lo, hi, ls, yf, s, sum_d);
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
            const float2 sm = mmq_few_get_sum(yf[c], s);
            sum_m[c][0] += m0 * sm.x;
            sum_m[c][1] += m0 * sm.y;
            sum_m[c][2] += m1 * sm.x;
            sum_m[c][3] += m1 * sm.y;
        }
    }
#pragma unroll
    for (int c = 0; c < ncg; ++c) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sum[c][l] += d[l/2]*sum_d[c][l] - dmin[l/2]*sum_m[c][l];
        }
    }
}

// Q6_K has a scale per 16 values: one k=16 MMA per half of a sub-block.
template <int ncg>
__device__ __forceinline__ void mmq_few_vec_dot_q6_K(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

    float d[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        d[r] = __half2float(((const block_q6_K *) x[r])->d);
    }

    float sum_d[ncg][4] = {{0.0f}};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int ql[2][2][2]; // [row][ql offset 0 or 32][lo or hi]
        int qh[2][2];
        int sc[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_q6_K * bx = (const block_q6_K *) x[r];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                ql[r][k][0] = get_int_b2(bx->ql + 64*n + 32*k,      t);
                ql[r][k][1] = get_int_b2(bx->ql + 64*n + 32*k + 16, t);
            }
            qh[r][0] = get_int_b2(bx->qh + 32*n,      t);
            qh[r][1] = get_int_b2(bx->qh + 32*n + 16, t);
            sc[r][0] = get_int_b2(bx->scales + 8*n, 0);
            sc[r][1] = get_int_b2(bx->scales + 8*n, 1);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s = 4*n + j;
            int lo[2];
            int hi[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                lo[r] = __vsub4(((ql[r][j % 2][0] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][0] >> (2*j)) & 0x03030303) << 4), 0x20202020);
                hi[r] = __vsub4(((ql[r][j % 2][1] >> (4*(j/2))) & 0x0F0F0F0F) | (((qh[r][1] >> (2*j)) & 0x03030303) << 4), 0x20202020);
            }
            const int sc_s[2] = {sc[0][j/2] >> (16*(j % 2)), sc[1][j/2] >> (16*(j % 2))}; // scales 2s and 2s+1
            mmq_few_mma_k16_add<MF_Q6_K>(lo, hi, sc_s, yf, s, sum_d);
        }
    }
    mmq_few_add_scaled(sum, d, sum_d);
}

template <int ncg>
__device__ __forceinline__ void mmq_few_vec_dot_q8_0(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    const int t = threadIdx.x % 4;

#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int   lo[2];
        int   hi[2];
        float d[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const block_q8_0_mf * bx = (const block_q8_0_mf *) x[r] + s;
            lo[r] = get_int_b2(bx->qs,      t);
            hi[r] = get_int_b2(bx->qs + 16, t);
            d[r]  = __half2float(bx->d);
        }
        mmq_few_mma_add<MF_Q8_0>(lo, hi, d, yf, s, sum);
    }
}

template <int type, int ncg>
__device__ __forceinline__ void mmq_few_vec_dot(const char * const (&x)[2], const mmq_few_y_frag (&yf)[ncg], float (&sum)[ncg][4]) {
    if constexpr (type == MF_Q8_0) {
        mmq_few_vec_dot_q8_0<ncg>(x, yf, sum);
    } else if constexpr (type == MF_Q4_K || type == MF_Q5_K) {
        mmq_few_vec_dot_q45_K<type, ncg>(x, yf, sum);
    } else {
        mmq_few_vec_dot_q6_K<ncg>(x, yf, sum);
    }
}

#endif // MF_AVAILABLE

template <typename OUT> __device__ __forceinline__ void mf_store(OUT * p, float v);
template <> __device__ __forceinline__ void mf_store<float>(float * p, float v) { *p = v; }
template <> __device__ __forceinline__ void mf_store<__half>(__half * p, float v) { *p = __float2half_rn(v); }

template <int type, int ncg, int nmt, typename OUT>
__global__ void __launch_bounds__(MMQ_FEW_NWARPS*MF_WARP, 1) mul_mat_q_few(
        const char * x, const int4 * y, OUT * dst, const int nunits, const int nrows_x, const int ncols_dst, const int64_t stride_row_x,
        const int stride_col_dst) {
#ifdef MF_AVAILABLE
    constexpr int nwarps    = MMQ_FEW_NWARPS;
    constexpr int warp_size = MF_WARP;

    constexpr int  unit_size    = mmq_few_unit_size(type);
    constexpr int  nrows        = nmt*MMQ_FEW_NROWS;
    constexpr int  nunits_stage = mmq_few_nunits_stage(type, ncg, nmt);
    constexpr int  nstages      = mmq_few_nstages(type);
    constexpr bool unaligned    = mmq_few_unaligned(type, nunits_stage);
    constexpr int  nchunks      = mmq_few_nchunks(type, nunits_stage);
    constexpr int  sram_stride  = mmq_few_sram_stride(type, nunits_stage);
    constexpr int  nbytes_stage = nrows*sram_stride;
    constexpr int  ncopies      = nrows*nchunks;
    static_assert(unit_size > 0, "unsupported type");

    extern __shared__ char data_mmq_few[];
    char * ring = data_mmq_few + threadIdx.y*nstages*nbytes_stage;

    const int row0 = blockIdx.x*nrows;

    // No bounds check for the units, the gate makes nunits a multiple of nunits_stage.
    // Unaligned data (Q6_K) reads up to 15 bytes past the last row; the caching allocator pads allocations to 512 bytes.
    auto load_stage = [&](const int stage, const int iu0) {
        const unsigned int dst_stage = (unsigned int) __cvta_generic_to_shared(ring + stage*nbytes_stage);
#pragma unroll
        for (int c0 = 0; c0 < ncopies; c0 += warp_size) {
            const int c  = c0 + threadIdx.x;
            const int i  = c / nchunks;
            const int ic = c % nchunks;
            if (ncopies % warp_size == 0 || c < ncopies) {
                const int row = min(row0 + i, nrows_x - 1);
                const char * src = x + int64_t(row)*stride_row_x + int64_t(iu0)*unit_size;
                if constexpr (unaligned) {
                    const int off = (uintptr_t) src % 16;
                    if (16*ic < off + nunits_stage*unit_size) {
                        cp_async_cg_16(dst_stage + i*sram_stride + 16*ic, src - off + 16*ic);
                    }
                } else {
                    cp_async_cg_16(dst_stage + i*sram_stride + 16*ic, src + 16*ic);
                }
            }
        }
    };

    // offsets of the rows of this lane from their 16 byte boundary
    int off_row[2*nmt] = {0};
    if constexpr (unaligned) {
#pragma unroll
        for (int r = 0; r < 2*nmt; ++r) {
            const int row = min(row0 + int(threadIdx.x)/4 + 8*r, nrows_x - 1);
            off_row[r] = (uintptr_t) (x + int64_t(row)*stride_row_x) % 16;
        }
    }

    const int iu_step = nunits_stage*nwarps;

#pragma unroll
    for (int stage = 0; stage < nstages - 1; ++stage) {
        const int iu0 = nunits_stage*threadIdx.y + stage*iu_step;
        if (iu0 < nunits) {
            load_stage(stage, iu0);
        }
        cp_async_commit_group();
    }

    auto load_y = [&](mmq_few_y_frag (&yf)[ncg], const int iu) {
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
            mmq_few_load_y(yf[c], y + (int64_t(iu)*ncg + c)*MMQ_FEW_Q8_1_INT4);
        }
    };

    // The next src1 fragments are loaded right after the MMAs that use the current ones, an earlier load is slower.
    mmq_few_y_frag yf[nunits_stage][ncg];
    if (nunits_stage*threadIdx.y < nunits) {
#pragma unroll
        for (int k = 0; k < nunits_stage; ++k) {
            load_y(yf[k], nunits_stage*threadIdx.y + k);
        }
    }

    float sum[nmt][ncg][4] = {{{0.0f}}};

    for (int iter = 0, iu0 = nunits_stage*threadIdx.y; iu0 < nunits; ++iter, iu0 += iu_step) {
        const int iu_load = iu0 + (nstages - 1)*iu_step;
        if (iu_load < nunits) {
            load_stage((iter + nstages - 1) % nstages, iu_load);
        }
        cp_async_commit_group();

        const bool has_next = iu0 + iu_step < nunits;

        cp_async_wait_group<nstages - 1>();
        __syncwarp();

        const char * xs = ring + (iter % nstages)*nbytes_stage;
        const int off_stage = unaligned ? (iu0*unit_size) % 16 : 0;
#pragma unroll
        for (int k = 0; k < nunits_stage; ++k) {
#pragma unroll
            for (int m = 0; m < nmt; ++m) {
                const char * const xr[2] = {
                    xs + (MMQ_FEW_NROWS*m + threadIdx.x/4 + 0)*sram_stride + (off_row[2*m + 0] + off_stage) % 16 + k*unit_size,
                    xs + (MMQ_FEW_NROWS*m + threadIdx.x/4 + 8)*sram_stride + (off_row[2*m + 1] + off_stage) % 16 + k*unit_size,
                };
                mmq_few_vec_dot<type, ncg>(xr, yf[k], sum[m]);
            }
            if (has_next) {
                load_y(yf[k], iu0 + iu_step + k);
            }
        }

        __syncwarp(); // the stage is overwritten in the next iteration
    }
    cp_async_wait_all();

    // reduce the sums over the warps as one flat array
    constexpr int nsum = nmt*ncg*4;
    float * sum_flat = &sum[0][0][0];

    float * buf_sum = (float *) data_mmq_few;
    __syncthreads();
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < nsum; ++i) {
            buf_sum[((threadIdx.y - 1)*nsum + i)*warp_size + threadIdx.x] = sum_flat[i];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int w = 0; w < nwarps - 1; ++w) {
#pragma unroll
        for (int i = 0; i < nsum; ++i) {
            sum_flat[i] += buf_sum[(w*nsum + i)*warp_size + threadIdx.x];
        }
    }

#pragma unroll
    for (int m = 0; m < nmt; ++m) {
#pragma unroll
        for (int c = 0; c < ncg; ++c) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + MMQ_FEW_NROWS*m + mf_get_i(l);
                const int j = c*MMQ_FEW_NCOLS + mf_get_j(l);
                if (i < nrows_x && j < ncols_dst) {
                    mf_store(dst + int64_t(j)*stride_col_dst + i, sum[m][c][l]);
                }
            }
        }
    }
#else
    __trap();
#endif // MF_AVAILABLE
}

// quantize.cu quantize_mmq_few_q8_1: each thread quantizes 4 values of one column. Per 256 values and group of MMQ_FEW_NCOLS columns:
//   qs: int4[4][32], [j][lane] holds the B fragments of sub-blocks 2j and 2j+1 of lane (g, t): the values [4t, 4t + 4) and [16 + 4t, 16 + 4t + 4) of column g.
//   ds: int4[4][4], [t][j] holds d (D4, float) or (d, sum) (DS4, half2) of the columns 2t and 2t+1 for sub-block 2j, then 2j+1.
// The column groups of a unit are adjacent, columns beyond ne1 are zero.
template <bool ds4>
__global__ void quantize_mmq_few_q8_1(const __half * x, int4 * vy, const int64_t ne00, const int64_t s01, const int ne1) {
#ifdef MF_AVAILABLE
    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne00) {
        return;
    }

    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if ((int) blockIdx.x < ne1) {
        const half2 * xp = (const half2 *) (x + blockIdx.x*s01 + i0);
        const float2 a = __half22float2(xp[0]);
        const float2 b = __half22float2(xp[1]);
        xi = make_float4(a.x, a.y, b.x, b.y);
    }

    // the 32 values of a sub-block are held by 8 threads
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    amax = warp_reduce_max<8>(amax);

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    // as in quantize_mmq_q8_1: the sum of the quantized values
    [[maybe_unused]] float sum = 0.0f;
    if constexpr (ds4) {
        sum = d*warp_reduce_sum<8>(float(q.x + q.y + q.z + q.w));
    }

    const int s = i0 % QK_K / 32;             // sub-block
    const int k = i0 % 32;                    // first value in sub-block
    const int g = blockIdx.x % MMQ_FEW_NCOLS; // column in group
    int4 * yg = vy + ((i0/QK_K)*(gridDim.x/MMQ_FEW_NCOLS) + blockIdx.x/MMQ_FEW_NCOLS)*MMQ_FEW_Q8_1_INT4;
    ((int *) (yg + (s/2)*MF_WARP + 4*g + (k % 16)/4))[2*(s % 2) + k/16] = *(const int *) &q;
    if (k == 0) {
        int4 * ds = yg + 4*MF_WARP + 4*(g/2) + s/2;
        if constexpr (!ds4) {
            ((float *) ds)[2*(s % 2) + g % 2] = d;
        } else {
            ((half2 *) ds)[2*(s % 2) + g % 2] = __floats2half2_rn(d, sum);
        }
    }
#else
    __trap();
#endif // MF_AVAILABLE
}

template <int type, int ncg, int nmt, typename OUT>
void launch_mul_mat_q_few(const uint8_t * x, const int4 * y, OUT * dst, int nunits, int N, int R, int64_t row_bytes, cudaStream_t stream) {
    constexpr int nbytes_shared = mmq_few_nbytes_shared(type, ncg, nmt);
    static bool attr_set = false;  // per instantiation (one GPU type per process)
    if (!attr_set) {
        cudaFuncSetAttribute(mul_mat_q_few<type, ncg, nmt, OUT>, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
        attr_set = true;
    }
    const dim3 block_nums((N + nmt*MMQ_FEW_NROWS - 1) / (nmt*MMQ_FEW_NROWS), 1, 1);
    const dim3 block_dims(MF_WARP, MMQ_FEW_NWARPS, 1);
    mul_mat_q_few<type, ncg, nmt, OUT><<<block_nums, block_dims, nbytes_shared, stream>>>(
        (const char *) x, y, dst, nunits, N, R, row_bytes, N);
}

// With 2 column groups, 32-row tiles halve the src1 loads per src0 byte. With 3 column groups they spill registers.
int mmq_few_nmt(const int type, const int ncg, const int64_t nrows_x, const int nsm) {
    return mmq_few_tall_supported(type) && ncg == 2 && nrows_x >= int64_t(nsm)*2*MMQ_FEW_NROWS ? 2 : 1;
}

template <int type, typename OUT>
void launch_switch_ncg(const uint8_t * x, const int4 * y, OUT * dst, int nunits, int N, int R, int64_t row_bytes, int ncg, int nmt,
                       cudaStream_t stream) {
    switch (ncg) {
        case 1: launch_mul_mat_q_few<type, 1, 1>(x, y, dst, nunits, N, R, row_bytes, stream); break;
        case 2:
            if constexpr (mmq_few_tall_supported(type)) {
                if (nmt == 2) { launch_mul_mat_q_few<type, 2, 2>(x, y, dst, nunits, N, R, row_bytes, stream); break; }
            }
            launch_mul_mat_q_few<type, 2, 1>(x, y, dst, nunits, N, R, row_bytes, stream);
            break;
        default: launch_mul_mat_q_few<type, 3, 1>(x, y, dst, nunits, N, R, row_bytes, stream); break;
    }
}

}  // namespace

// out[R, N] = x[R, K] @ W^T for raw GGUF rows packed [N, K/256 * unit bytes] (fmt 0 Q8_0, 1 Q4_K, 2 Q5_K, 3 Q6_K), 1 <= R <= 24.
// scratch: uint8 CUDA tensor of at least ceil(R/8) * K * 9 bytes (fq_q8_scratch(dev, 8*ceil(R/8), K) has exactly that size).
// Returns false (nothing launched) when the case is not served: GPU older than sm_80, K not a whole number of ring stages, short
// rows (K < 2048: the warps idle), misaligned weights.
bool fq_mmq_few(torch::Tensor x, torch::Tensor scratch, torch::Tensor packed, torch::Tensor out, int fmt)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous() && x.dim() == 2, "x must be contiguous CUDA half [R, K]");
    TORCH_CHECK(fmt >= 0 && fmt <= 3, "fmt must be 0..3");
    const int R = (int) x.size(0), K = (int) x.size(1), N = (int) packed.size(0);
    TORCH_CHECK(R >= 1 && R <= MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS, "R must be 1..24");
    TORCH_CHECK(out.is_contiguous() && out.size(0) == R && out.size(1) == N, "out must be contiguous [R, N]");
    TORCH_CHECK(out.dtype() == torch::kHalf || out.dtype() == torch::kFloat, "out must be half or float");
    const int ncg = (R + MMQ_FEW_NCOLS - 1) / MMQ_FEW_NCOLS;
    const int nunits = K / QK_K;
    const int ubytes = mmq_few_unit_size(fmt);
    TORCH_CHECK(packed.is_contiguous() && packed.numel() == (int64_t) N * nunits * ubytes, "packed must be [N, K/256 * unit bytes]");
    TORCH_CHECK(scratch.is_cuda() && scratch.numel() * scratch.element_size() >= (int64_t) ncg * nunits * MMQ_FEW_Q8_1_INT4 * 16, "scratch too small");

    at::cuda::CUDAGuard guard(x.device());
    const cudaDeviceProp * prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major < 8 || K % QK_K != 0 || K < 8*QK_K)
        return false;
    const int nmt = mmq_few_nmt(fmt, ncg, N, prop->multiProcessorCount);
    if (K % (mmq_few_nunits_stage(fmt, ncg, nmt)*QK_K) != 0)
        return false;
    const int x_alignment = mmq_few_unaligned(fmt, mmq_few_nunits_stage(fmt, ncg, nmt)) ? 2 : 16;
    if ((uintptr_t) packed.data_ptr() % x_alignment != 0 || (uintptr_t) scratch.data_ptr() % 16 != 0 || (uintptr_t) x.data_ptr() % 4 != 0)
        return false;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int4 * y = (int4 *) scratch.data_ptr();
    const dim3 qblocks(ncg*MMQ_FEW_NCOLS, (K + 4*MF_QUANT_BLOCK - 1) / (4*MF_QUANT_BLOCK), 1);
    if (mmq_few_ds4(fmt))
        quantize_mmq_few_q8_1<true><<<qblocks, MF_QUANT_BLOCK, 0, stream>>>((const __half *) x.data_ptr(), y, K, K, R);
    else
        quantize_mmq_few_q8_1<false><<<qblocks, MF_QUANT_BLOCK, 0, stream>>>((const __half *) x.data_ptr(), y, K, K, R);

    const uint8_t * w = (const uint8_t *) packed.data_ptr();
    const int64_t row_bytes = (int64_t) nunits * ubytes;
    auto run = [&](auto * dst) {
        using OUT = std::remove_pointer_t<decltype(dst)>;
        switch (fmt) {
            case MF_Q8_0: launch_switch_ncg<MF_Q8_0, OUT>(w, y, dst, nunits, N, R, row_bytes, ncg, nmt, stream); break;
            case MF_Q4_K: launch_switch_ncg<MF_Q4_K, OUT>(w, y, dst, nunits, N, R, row_bytes, ncg, nmt, stream); break;
            case MF_Q5_K: launch_switch_ncg<MF_Q5_K, OUT>(w, y, dst, nunits, N, R, row_bytes, ncg, nmt, stream); break;
            default:      launch_switch_ncg<MF_Q6_K, OUT>(w, y, dst, nunits, N, R, row_bytes, ncg, nmt, stream); break;
        }
    };
    if (out.dtype() == torch::kHalf)
        run((__half *) out.data_ptr());
    else
        run(out.data_ptr<float>());
    return true;
}
