// Q4_K / Q6_K v2 "items" (one thread, one aligned wide load pair = 4 consecutive v1 sub-chunk values), shared by the dense
// rows GEMV (fq_gemv_kq.cu) and the grouped MoE gate+up (fq_moe_kq.cu). Include after fq_gemv.cuh, inside no namespace.
#pragma once
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ float fq_h2f_lo(uint32_t w)
{
    uint16_t lo = (uint16_t)(w & 0xFFFFu);
    __half h;
    memcpy(&h, &lo, 2);
    return __half2float(h);
}

__device__ __forceinline__ uint32_t fq_ld_u16x2(const uint8_t* __restrict__ p)   // p is 2-byte aligned
{
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t)q[0] | ((uint32_t)q[1] << 16);
}

// Q4_K item: 256-block b = it >> 2, group g = it & 3 (qs bytes 32g .. 32g+31: low nibbles = sub-block 2g,
// high nibbles = sub-block 2g+1). dst[qo] = v1 chunk value c = 16 b + 4 g + qo, qo = 0..3.
__device__ __forceinline__ void fq_kq_item_q4_k(const uint8_t* __restrict__ wr, const uint8_t* __restrict__ aq,
                                                int it, float* __restrict__ dst)
{
    const int b = it >> 2, g = it & 3;
    const uint8_t* blk = wr + (size_t)b * FQ_Q4_K_BYTES;                 // 144 B: 16-byte aligned rows
    const uint32_t hd = *reinterpret_cast<const uint32_t*>(blk);          // d | dmin
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(blk + 4);
    const uint32_t sw0 = sp[0], sw1 = sp[1], sw2 = sp[2];
    const uint4 va = *reinterpret_cast<const uint4*>(blk + 16 + 32 * g);
    const uint4 vb = *reinterpret_cast<const uint4*>(blk + 16 + 32 * g + 16);
    const int32_t* a0 = reinterpret_cast<const int32_t*>(aq + (size_t)(b * 8 + 2 * g) * FQ_Q8_1_BYTES);
    const int32_t* a1 = a0 + FQ_Q8_1_BYTES / 4;
    const float dm0 = fq_h2f_lo(hd);
    const float dm1 = fq_h2f_lo(hd >> 16);
    const float d8[2] = {fq_h2f_lo((uint32_t)a0[0]), fq_h2f_lo((uint32_t)a1[0])};
    // scales / mins of sub-blocks 2g, 2g+1 (verbatim fq_q4k_scales on the three u16 words it reads)
    const int sh = 16 * (g & 1);
    const uint32_t s0 = (sw0 >> sh) & 0xFFFFu, s2 = (sw1 >> sh) & 0xFFFFu, s4 = (sw2 >> sh) & 0xFFFFu;
    const uint32_t hi = (g >= 2) ? 0xFFFFFFFFu : 0u;
    const uint16_t aux0 = (uint16_t)(((s0 & 0x3F3Fu) & ~hi) | ((((s4 >> 0) & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2)) & hi));
    const uint16_t aux1 = (uint16_t)(((s2 & 0x3F3Fu) & ~hi) | ((((s4 >> 4) & 0x0F0Fu) | ((s2 & 0xC0C0u) >> 2)) & hi));
    const int sc[2] = {(int)(aux0 & 0xFFu), (int)((aux0 >> 8) & 0xFFu)};
    const int mn[2] = {(int)(aux1 & 0xFFu), (int)((aux1 >> 8) & 0xFFu)};
    const uint32_t w0[4] = {va.x, va.y, va.z, va.w};
    const uint32_t w1[4] = {vb.x, vb.y, vb.z, vb.w};
    #pragma unroll
    for (int qo = 0; qo < 4; qo++)
    {
        float sumf_d = 0.0f, sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            const int32_t* a = i ? a1 : a0;
            const uint32_t v0i = (w0[qo] >> (4 * i)) & 0x0F0F0F0Fu;
            const uint32_t v1i = (w1[qo] >> (4 * i)) & 0x0F0F0F0Fu;
            const int32_t u0 = a[1 + qo], u1 = a[5 + qo];
            int dot1 = fq_dp4a((int32_t)v0i, u0, 0);
            dot1 = fq_dp4a((int32_t)v1i, u1, dot1);
            int dot2 = fq_dp4a(0x01010101, u0, 0);
            dot2 = fq_dp4a(0x01010101, u1, dot2);
            sumf_d = __fadd_rn(sumf_d, __fmul_rn(d8[i], (float)(dot1 * sc[i])));
            sumf_m = __fadd_rn(sumf_m, __fmul_rn(d8[i], (float)(dot2 * mn[i])));
        }
        dst[qo] = __fadd_rn(__fmul_rn(dm0, sumf_d), -__fmul_rn(dm1, sumf_m));
    }
}

// Q6_K item: 256-block b = it >> 3, lane p = it & 7; the four v1 chunks iqs = p + 8 q (q = 0..3) of that block,
// c = 32 b + p + 8 q. dst[q] = value of chunk q.
__device__ __forceinline__ void fq_kq_item_q6_k(const uint8_t* __restrict__ wr, const uint8_t* __restrict__ aq,
                                                int it, float* __restrict__ dst)
{
    const int b = it >> 3, p = it & 7;
    const uint8_t* blk = wr + (size_t)b * FQ_Q6_K_BYTES;                 // 210 B: 2-byte aligned only
    uint32_t vl[4];
    #pragma unroll
    for (int q = 0; q < 4; q++)
        vl[q] = fq_ld_u16x2(blk + 4 * p + 32 * q);
    const uint32_t vhw[2] = {fq_ld_u16x2(blk + 128 + 4 * p), fq_ld_u16x2(blk + 128 + 32 + 4 * p)};
    const int8_t* scl = reinterpret_cast<const int8_t*>(blk + 192);
    const float d = fq_h2f_lo((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 208));
    #pragma unroll
    for (int q = 0; q < 4; q++)
    {
        const int sc_off = 8 * (q >> 1) + (p >> 2) + 2 * (q & 1);
        const int bq8_off = 4 * (q >> 1) + (q & 1);
        const uint32_t vh = vhw[q >> 1] >> (2 * (q & 1));
        float sumf = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            const int scc = (int)scl[sc_off + 4 * i];
            const uint32_t vil = (vl[q] >> (4 * i)) & 0x0F0F0F0Fu;
            const uint32_t vih = ((vh >> (4 * i)) << 4) & 0x30303030u;
            const uint32_t vi = fq_vsub4(vil | vih, 0x20202020u);
            const int32_t* a = reinterpret_cast<const int32_t*>(aq + (size_t)(b * 8 + bq8_off + 2 * i) * FQ_Q8_1_BYTES);
            const float d8 = fq_h2f_lo((uint32_t)a[0]);
            sumf = __fadd_rn(sumf, __fmul_rn(d8, (float)(fq_dp4a((int32_t)vi, a[1 + p], 0) * scc)));
        }
        dst[q] = __fmul_rn(d, sumf);
    }
}

// Q5_K item (same grid as the Q4_K item): 256-block b = it >> 2, group g = it & 3; dst[qo] = fq_dot_q5_k_q8's chunk value
// c = 16 b + 4 g + qo (bo = 2 g). qs words come from two 16-byte loads at 48 + 32 g, the high-bit words from the two 16-byte
// loads at 16 and 32 (shared by all g, shifted by bo as in the v1 dot). 176-byte blocks: rows 16-byte aligned.
__device__ __forceinline__ void fq_kq_item_q5_k(const uint8_t* __restrict__ wr, const uint8_t* __restrict__ aq,
                                                int it, float* __restrict__ dst)
{
    const int b = it >> 2, g = it & 3, bo = 2 * g;
    const uint8_t* blk = wr + (size_t)b * FQ_Q5_K_BYTES;
    const uint32_t hd = *reinterpret_cast<const uint32_t*>(blk);
    const float dm0 = fq_h2f_lo(hd);
    const float dm1 = fq_h2f_lo(hd >> 16);
    uint8_t sc[2], m[2];
    fq_q4k_scales(blk + 4, g, sc, m);
    const uint4 la = *reinterpret_cast<const uint4*>(blk + 48 + 32 * g);
    const uint4 lb = *reinterpret_cast<const uint4*>(blk + 48 + 32 * g + 16);
    const uint4 ha = *reinterpret_cast<const uint4*>(blk + 16);
    const uint4 hb = *reinterpret_cast<const uint4*>(blk + 32);
    const uint32_t l0[4] = {la.x, la.y, la.z, la.w}, l1[4] = {lb.x, lb.y, lb.z, lb.w};
    const uint32_t h0[4] = {ha.x, ha.y, ha.z, ha.w}, h1[4] = {hb.x, hb.y, hb.z, hb.w};
    const int32_t* a0 = reinterpret_cast<const int32_t*>(aq + (size_t)(b * 8 + bo) * FQ_Q8_1_BYTES);
    const int32_t* a1 = a0 + FQ_Q8_1_BYTES / 4;
    const float d8[2] = {fq_h2f_lo((uint32_t)a0[0]), fq_h2f_lo((uint32_t)a1[0])};
    #pragma unroll
    for (int qo = 0; qo < 4; qo++)
    {
        const uint32_t vh0 = h0[qo] >> bo, vh1 = h1[qo] >> bo;
        float sumf_d = 0.0f, sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            const int32_t* a = i ? a1 : a0;
            const uint32_t v0i = ((l0[qo] >> (4 * i)) & 0x0F0F0F0Fu) | (((vh0 >> i) << 4) & 0x10101010u);
            const uint32_t v1i = ((l1[qo] >> (4 * i)) & 0x0F0F0F0Fu) | (((vh1 >> i) << 4) & 0x10101010u);
            const int32_t u0 = a[1 + qo], u1 = a[5 + qo];
            int dot1 = fq_dp4a((int32_t)v0i, u0, 0);
            dot1 = fq_dp4a((int32_t)v1i, u1, dot1);
            int dot2 = fq_dp4a(0x01010101, u0, 0);
            dot2 = fq_dp4a(0x01010101, u1, dot2);
            sumf_d = __fadd_rn(sumf_d, __fmul_rn(d8[i], (float)(dot1 * sc[i])));
            sumf_m = __fadd_rn(sumf_m, __fmul_rn(d8[i], (float)(dot2 * m[i])));
        }
        dst[qo] = __fadd_rn(__fmul_rn(dm0, sumf_d), -__fmul_rn(dm1, sumf_m));
    }
}

}  // namespace
