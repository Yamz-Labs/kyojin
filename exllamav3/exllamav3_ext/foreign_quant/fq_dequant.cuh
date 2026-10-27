// Foreign-quant on-the-fly dequant (device-inline, shared by the dequant and
// fused dequant-GEMM kernels). Portable CUDA/HIP: no inline PTX, no warp
// shuffles or wave-size assumptions (every thread decodes its own values from
// global/constant memory with plain scalar ops), only <cuda_fp16.h> half
// intrinsics (supported by HIP) plus __fmul_rn/__fadd_rn.
//
// Bit-exactness contract: fq_*_block() must reproduce
// exllamav3/foreign_quant/*.py in float32 op-for-op (same order, single
// roundings), so a final __float2half_rn matches numpy .astype(fp16).
// __fmul_rn/__fadd_rn are explicit IEEE round-to-nearest ops the compiler may
// not fuse or reorder, which keeps this true under --use_fast_math. (HIP
// provides the same intrinsics.)
#pragma once

#include <cstdlib>
#include <cuda_fp16.h>
#include "fq_tables.cuh"

#define FQ_Q8_0_GROUP 32
#define FQ_Q8_0_BYTES 34
#define FQ_IQ2_XXS_GROUP 256
#define FQ_IQ2_XXS_BYTES 66
#define FQ_NVFP4_GROUP 16

// Streaming dequant v2 (cooperative-by-construction, no shared/sync):
// 2D grids over (block, row) so no thread does index division; headers
// load direct as u16 pairs (ggml get_int_b2 pattern); IQ2 signs unpack
// inline (ggml unpack_ksigns, no divergent const LUT); NVFP4 reads a
// 3 KB global table pair (E2M1 float2 + E4M3 float, built once on device
// by fq_dq_tables_build) instead of branchy fq_e4m3 + divergent LDC.
// Same per-element float32 op order as the v1 kernels below, hence
// bitwise-identical output. Default on; BOLT_FQ_DQV2=0 restores v1.
inline bool fq_use_dqv2()
{
    const char* e = std::getenv("BOLT_FQ_DQV2");
    return !e || e[0] != '0';
}

// One 34-byte Q8_0 block -> 32 float32 values. Matches q8_0.py dequant_q8_0:
// y[j] = (float)(int8_t)qs[j] * (float)(fp16)d, single fp32 rounding.
__forceinline__ __device__ void fq_q8_0_block(const uint8_t* blk, float* out32)
{
    __half dh;
    memcpy(&dh, blk, 2);
    float d = __half2float(dh);
    for (int j = 0; j < 32; j++)
        out32[j] = __fmul_rn((float)((int8_t)blk[2 + j]), d);
}

// One 66-byte IQ2_XXS block -> 256 float32 values. Port of llama.cpp
// dequantize_row_iq2_xxs in the same op order as iq2_xxs.py _dequant_block:
//   db = d * (0.5 + (a1 >> 28)) * 0.25        (fp32, left-assoc)
//   out = (db * grid_mag) * sign              (fp32, left-assoc)
__forceinline__ __device__ void fq_iq2_xxs_block(const uint8_t* blk, float* out256)
{
    __half dh;
    memcpy(&dh, blk, 2);
    float d = __half2float(dh);
    int pos = 0;
    for (int ib = 0; ib < 8; ib++)
    {
        uint32_t q0 = (uint32_t)blk[2 + 8 * ib] | ((uint32_t)blk[2 + 8 * ib + 1] << 8);
        uint32_t q1 = (uint32_t)blk[2 + 8 * ib + 2] | ((uint32_t)blk[2 + 8 * ib + 3] << 8);
        uint32_t q2 = (uint32_t)blk[2 + 8 * ib + 4] | ((uint32_t)blk[2 + 8 * ib + 5] << 8);
        uint32_t q3 = (uint32_t)blk[2 + 8 * ib + 6] | ((uint32_t)blk[2 + 8 * ib + 7] << 8);
        uint32_t a0 = q0 | (q1 << 16);
        uint32_t a1 = q2 | (q3 << 16);
        float db = __fmul_rn(__fmul_rn(d, 0.5f + (float)(a1 >> 28)), 0.25f);
        for (int elle = 0; elle < 4; elle++)
        {
            uint64_t grid = FQ_IQ2XXS_GRID[(a0 >> (8 * elle)) & 0xFF];
            uint8_t signs = FQ_KSIGNS_IQ2XS[(a1 >> (7 * elle)) & 127];
            for (int j = 0; j < 8; j++)
            {
                float mag = (float)((grid >> (8 * j)) & 0xFF);
                float sgn = (signs & FQ_KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
                out256[pos++] = __fmul_rn(__fmul_rn(db, mag), sgn);
            }
        }
    }
}

// OCP fp8 E4M3 byte -> float32. Matches nvfp4.py e4m3_to_float32 bit-exactly:
// normal (incl. e==15, m!=7): scalbn(1 + m/8, e-7); subnormal: m * 2^-9;
// zero -> signed zero; 0x7F/0xFF -> NaN. All intermediate values are exact
// (power-of-two scalings, 1 + m/8 representable), so plain ops suffice.
__forceinline__ __device__ float fq_e4m3(uint8_t b)
{
    int e = (b >> 3) & 0xF;
    int m = b & 0x7;
    if (e == 15 && m == 7) return nanf("");
    float v;
    if (e == 0)
        v = (m == 0) ? 0.0f : (float)m * 0.001953125f;   // m * 2^-9, exact
    else
        v = scalbnf(1.0f + (float)m * 0.125f, e - 7);     // exact
    return (b & 0x80) ? -v : v;
}

// One NVFP4 group (8 packed E2M1 bytes + 1 E4M3 scale + fp32 global) ->
// 16 float32 values. Matches nvfp4.py dequant_nvfp4: (e2m1 * gscale) * global,
// left-assoc fp32. Low nibble = even index (ModelOpt packing).
__forceinline__ __device__ void fq_nvfp4_group(const uint8_t* packed8, float gscale, float global, float* out16)
{
    for (int i = 0; i < 8; i++)
    {
        float lo = FQ_E2M1_LUT[packed8[i] & 0xF];
        float hi = FQ_E2M1_LUT[(packed8[i] >> 4) & 0xF];
        out16[2 * i]     = __fmul_rn(__fmul_rn(lo, gscale), global);
        out16[2 * i + 1] = __fmul_rn(__fmul_rn(hi, gscale), global);
    }
}
