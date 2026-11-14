// Q4_K and Q6_K vec-dots (q8_1 activations, integer dp4a dots) ported from llama.cpp ggml/src/ggml-cuda/vecdotq.cuh
// (https://github.com/ggml-org/llama.cpp, commit 61dbf49) and block_q4_K / block_q6_K of ggml/src/ggml-common.h.
// MIT License, Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.md at the repository root.
// Changes: ggml_cuda_dp4a -> ggml_dp4a, namespace ggml_iq, plain half2 / half block fields. The dot-product bodies are unchanged.
#pragma once

#include "ggml_vecdotq_k.cuh"

namespace ggml_iq {

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))

struct block_q4_K {
    half2 dm;                  // d, dmin
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "ggml block layout");

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    half    d;
};
static_assert(sizeof(block_q6_K) == 210, "ggml block layout");

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMVQ 2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_dp4a(v1i, u[2*i+1], ggml_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_dp4a(0x01010101, u[2*i+1], ggml_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        int sumi = 0;
#pragma unroll
        for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
            const int vil = (vl[j] >> (4*i)) & 0x0F0F0F0F;
            const int vih = ((vh[j] >> (4*i)) << 4) & 0x30303030;

            // (vil | vih) - 32 for each byte, the - 32 is done by the second dp4a
            sumi = ggml_dp4a(vil | vih,  u[VDR_Q6_K_Q8_1_MMVQ*i + j], sumi); // SIMD dot product
            sumi = ggml_dp4a(0xE0E0E0E0, u[VDR_Q6_K_Q8_1_MMVQ*i + j], sumi);
        }

        sumf += d8[i] * (sumi * sc);
    }

    return d*sumf;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    // branchless so nvcc can hoist this out of the ncols_dst loop
    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    int vl[VDR_Q6_K_Q8_1_MMVQ];
    int vh[VDR_Q6_K_Q8_1_MMVQ];
#pragma unroll
    for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
        vl[j] = get_int_b2(bq6_K->ql, iqs + j);
        vh[j] = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4) + j) >> vh_shift;
    }

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[VDR_Q6_K_Q8_1_MMVQ*QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
            u[VDR_Q6_K_Q8_1_MMVQ*i + j] = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1 + j);
        }
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, __half2float(bq6_K->d), d8);
}

}  // namespace ggml_iq
