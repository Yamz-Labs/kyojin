// IQ vec-dot functions (activations as q8_1 blocks, integer dp4a dots) ported from llama.cpp
// ggml/src/ggml-cuda/vecdotq.cuh (https://github.com/ggml-org/llama.cpp, commit 7182ef7) and the block
// layouts of ggml/src/ggml-common.h.
// MIT License, Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.md at the repository root.
// Changes: ggml_cuda_dp4a -> ggml_dp4a, the HIP helpers of ggml/src/ggml-cuda/vendors/hip.h (__vsub4, __vcmpne4, ggml_hip_perm) and
// the HIP branch of get_int_from_table_16 (llama.cpp commit ded522f) under USE_ROCM, no MUSA branches, everything in namespace
// ggml_iq, MMQ constants dropped. The dot-product bodies are unchanged.
#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include "ggml_iq_tables.cuh"

#if defined(USE_ROCM)
// HIP has no __vsub4 / __vcmpne4: llama.cpp ggml/src/ggml-cuda/vendors/hip.h (MIT)
static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    const unsigned int a_large = a | 0x80808080;
    const unsigned int b_small = b & 0x7f7f7f7f;
    const unsigned int result_low_7bits = a_large - b_small;
    const unsigned int should_flip_high_1bit = (a ^ ~b) & 0x80808080;
    return result_low_7bits ^ should_flip_high_1bit;
}
static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const unsigned int x = a ^ b;
    const unsigned int ne_low_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    const unsigned int ne_high_1bit = x & 0x80808080;
    const unsigned int ne_any_bit = ne_low_7bits | ne_high_1bit;
    return (ne_any_bit >> 7) * 0xff;
}
#endif

namespace ggml_iq {

// signed int8 x int8 4-way dot (llama.cpp ggml_cuda_dp4a): CUDA __dp4a; RDNA3 / RDNA3.5 / RDNA4 v_dot4_i32_iu8; else scalar
static __device__ __forceinline__ int ggml_dp4a(const int a, const int b, int c) {
#if !defined(USE_ROCM)
    return __dp4a(a, b, c);
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
      defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    const int8_t * a8 = reinterpret_cast<const int8_t *>(&a);
    const int8_t * b8 = reinterpret_cast<const int8_t *>(&b);
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

#define QK_K 256
#define QK4_NL 32
#define IQ3S_N_SCALE (QK_K / 64)
#define VDR_Q4_0_Q8_1_MMVQ 2

struct block_q8_1 {
    half2 ds;          // ds.x = d (scale), ds.y = d * sum(qs)
    int8_t qs[32];
};
struct block_iq2_xxs { half d; uint16_t qs[QK_K / 8]; };
struct block_iq2_xs  { half d; uint16_t qs[QK_K / 8]; uint8_t scales[QK_K / 32]; };
struct block_iq2_s   { half d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t scales[QK_K / 32]; };
struct block_iq3_xxs { half d; uint8_t qs[3 * QK_K / 8]; };
struct block_iq3_s   { half d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t signs[QK_K / 8]; uint8_t scales[IQ3S_N_SCALE]; };
struct block_iq4_nl  { half d; uint8_t qs[QK4_NL / 2]; };
struct block_iq4_xs  { half d; uint16_t scales_h; uint8_t scales_l[QK_K / 64]; uint8_t qs[QK_K / 2]; };
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 && sizeof(block_iq3_xxs) == 98 &&
              sizeof(block_iq3_s) == 110 && sizeof(block_iq4_nl) == 18 && sizeof(block_iq4_xs) == 136 && sizeof(block_q8_1) == 36,
              "ggml block layout");

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// q4 contains 8 indices with 4 bit each; returns the table bytes at the even / odd indices (llama.cpp get_int_from_table_16)
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
#if defined(USE_ROCM)
    const uint32_t * values = (const uint32_t *) table;
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);
    const uint32_t v_even_low  = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);
    const uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    const uint32_t mask_odd  = 0x03020100 | ((q_odd & 0x08080808) >> 1);
    return make_int2(__builtin_amdgcn_perm(v_even_high, v_even_low, mask_even), __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd));
#else
    const uint32_t * table32 = (const uint32_t *) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#endif
}

static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    // v is a 7 bit int, with the 8th sign being encodable as popcnt
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}

#define VDR_IQ2_XXS_Q8_1_MMVQ 2

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_XS_Q8_1_MMVQ 2

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_S_Q8_1_MMVQ 2

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_XXS_Q8_1_MMVQ 2

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_dp4a(grid_l, u0, sumi);
        sumi = ggml_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_S_Q8_1_MMVQ 2

// TODO: don't use lookup table for signs
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_dp4a(grid_l, u0, sumi);
        sumi = ggml_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ4_NL_Q8_1_MMVQ 2

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        sumi = ggml_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

#define VDR_IQ4_XS_Q8_1_MMVQ 4

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);

        sumi = ggml_dp4a(v.x, u0, sumi);
        sumi = ggml_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}

}  // namespace ggml_iq
