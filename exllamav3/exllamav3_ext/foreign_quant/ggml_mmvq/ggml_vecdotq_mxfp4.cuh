// MXFP4 x q8_1 dot product for the fq_mmvq family: port of llama.cpp ggml/src/ggml-cuda/vecdotq.cuh (get_int_b1 :7-15,
// get_int_from_table_16 :34 (shared, see ggml_vecdotq_iq.cuh), vec_dot_mxfp4_q8_1 :310-329, VDR_MXFP4_Q8_1_MMVQ :307),
// block_mxfp4 ggml-common.h:214-219, kvalues_mxfp4 (= kvalues_fp4, e2m1 values doubled) ggml-common.h:1124-1129.
// MIT License, Copyright (c) 2023-2026 The ggml authors (llama.cpp commit 7182ef7). See THIRD_PARTY_NOTICES.md.
// Change: ggml_cuda_dp4a -> ggml_dp4a; ggml_cuda_e8m0_to_fp32 written with integer bits (exact same value for every e8m0 byte
// incl. 0 -> 2^-127 denormal), so the file also compiles on HIP.
#pragma once

#include "ggml_vecdotq_iq.cuh"

namespace ggml_iq {

#define QK_MXFP4 32
#define QI_MXFP4 4
#define VDR_MXFP4_Q8_1_MMVQ 2

struct block_mxfp4 { uint8_t e; uint8_t qs[QK_MXFP4 / 2]; };   // 17 bytes: [E8M0 scale][elements j (low nibble) and j + 16 (high nibble)]
static_assert(sizeof(block_mxfp4) == 17, "ggml block layout");

alignas(16) static __device__ const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;
    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;
    return x32;
}

static __device__ __forceinline__ float mxfp4_e8m0_to_fp32(uint8_t x) {   // 2^(x-127); x == 0 -> 2^-127
    const uint32_t bits = x == 0 ? 0x00400000u : (uint32_t) x << 23;
    return __uint_as_float(bits);
}

static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

        sumi = ggml_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = mxfp4_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * sumi;
}

}  // namespace ggml_iq
