// Legacy GGUF block vec-dots Q4_0, Q4_1, Q5_0, Q5_1 (q8_1 activations, integer dp4a dots), MMVQ variants (VDR = 2), ported from llama.cpp
// ggml/src/ggml-cuda/vecdotq.cuh (vec_dot_q{4_0,4_1,5_0,5_1}_q8_1 and their _impl) and the block structs of ggml/src/ggml-common.h
// (https://github.com/ggml-org/llama.cpp). MIT License, Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.md.
// Changes: ggml_cuda_dp4a -> ggml_dp4a, namespace ggml_iq, plain half / half2 block fields, the FAST_FP16_AVAILABLE half2 multiply
// replaced by its float branch (identical to ggml's non-fp16 path). The dot-product bodies are unchanged.
#pragma once

#include "ggml_vecdotq_iq.cuh"

namespace ggml_iq {

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
#define QI8_1 8

struct block_q4_0 { half d; uint8_t qs[QK4_0 / 2]; };
struct block_q4_1 { half2 dm; uint8_t qs[QK4_1 / 2]; };
struct block_q5_0 { half d; uint8_t qh[4]; uint8_t qs[QK5_0 / 2]; };
struct block_q5_1 { half2 dm; uint32_t qh; uint8_t qs[QK5_1 / 2]; };
static_assert(sizeof(block_q4_0) == 18 && sizeof(block_q4_1) == 20 && sizeof(block_q5_0) == 22 && sizeof(block_q5_1) == 24, "ggml block layout");

#ifndef VDR_Q4_0_Q8_1_MMVQ
#define VDR_Q4_0_Q8_1_MMVQ 2
#endif
#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMVQ 2

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = ggml_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 ds8f = __half22float2(ds8);
    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = ggml_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

// the 5th bit of the 8 quants held in vl[i] (low nibbles: bits 0..3 of vh, high nibbles: bits 16..19 of vh)
#define GGML_LEGACY_Q5_SUMI(vl_i, vh_i, u0, u1, sumi)                                                                           \
    {                                                                                                                           \
        int vi0 = ((vl_i) >>  0) & 0x0F0F0F0F;                                                                                  \
        vi0    |= ((vh_i) <<  4) & 0x00000010;                                                                                  \
        vi0    |= ((vh_i) << 11) & 0x00001000;                                                                                  \
        vi0    |= ((vh_i) << 18) & 0x00100000;                                                                                  \
        vi0    |= ((vh_i) << 25) & 0x10000000;                                                                                  \
        sumi = ggml_dp4a(vi0, (u0), sumi);                                                                                      \
        int vi1 = ((vl_i) >>  4) & 0x0F0F0F0F;                                                                                  \
        vi1    |= ((vh_i) >> 12) & 0x00000010;                                                                                  \
        vi1    |= ((vh_i) >>  5) & 0x00001000;                                                                                  \
        vi1    |= ((vh_i) <<  2) & 0x00100000;                                                                                  \
        vi1    |= ((vh_i) <<  9) & 0x10000000;                                                                                  \
        sumi = ggml_dp4a(vi1, (u1), sumi);                                                                                      \
    }

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) GGML_LEGACY_Q5_SUMI(vl[i], vh[i], u[2*i+0], u[2*i+1], sumi)
    const float2 ds8f = __half22float2(ds8);
    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) GGML_LEGACY_Q5_SUMI(vl[i], vh[i], u[2*i+0], u[2*i+1], sumi)
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi * d5d8 + m5s8 / (QI5_1 / vdr);
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;
    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b2(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
    }
    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, __half2float(bq4_0->d), bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq + kbx;
    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b4(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
    }
    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq + kbx;
    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b2(bq5_0->qs, iqs + i);
        vh[i]    = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
    }
    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, __half2float(bq5_0->d), bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq + kbx;
    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b4(bq5_1->qs, iqs + i);
        vh[i]    = get_int_b4(&bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
    }
    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

}  // namespace ggml_iq
