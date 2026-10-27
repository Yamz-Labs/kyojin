// DeepSeek-V4.1 number formats shared by the HIP kernels (dsv41_hip.hip) and a CPU build (tests/dsv41/num_cpu.cpp).
// Every function is plain C++ on float/uint32 with explicit rounding, so the host build is bit-identical to the device
// build and is compared bit for bit against the torch reference (exllamav3/dsv41/quant.py) in tests/dsv41/test_dsv41_num.py.
// Compile with -ffp-contract=off (no fused multiply-add anywhere in these formulas).
#pragma once
#include <stdint.h>
#include <math.h>

#if defined(__HIPCC__)
#define DS_HD static __device__ __host__ __forceinline__
#else
#define DS_HD static inline
#endif

DS_HD uint32_t ds_f2u(float v) { union { float f; uint32_t u; } c; c.f = v; return c.u; }
DS_HD float ds_u2f(uint32_t u) { union { float f; uint32_t u; } c; c.u = u; return c.f; }

// 1/448 and 1/6 as fp32 (what torch's F32_INV_448 / F32_INV_6 tensors hold)
DS_HD float ds_inv448() { return (float) (1.0 / 448.0); }
DS_HD float ds_inv6() { return (float) (1.0 / 6.0); }

// 2 ** ceil(log2(t)) of a positive fp32 from its bit pattern (quant._round_scale_pow2)
DS_HD float ds_round_pow2(float t)
{
    uint32_t b = ds_f2u(t);
    int32_t exp = (int32_t) ((b >> 23) & 0xFF);
    int32_t man = (int32_t) (b & 0x7FFFFF);
    int32_t e = exp - 127 + (man != 0 ? 1 : 0);
    return ds_u2f((uint32_t) (e + 127) << 23);
}

// Round-to-nearest-even onto the e4m3fn grid, returned as the dequantised float. |v| must be <= 448 (callers clamp first).
// subnormals: step 2^-9 below 2^-6; normals: 3 mantissa bits, step 2^(floor(log2|v|) - 3).
DS_HD float ds_e4m3_round(float v)
{
    float a = fabsf(v);
    float q;
    if (a < 0.015625f)                               // 2^-6: e4m3 subnormal range
        q = rintf(a * 512.0f) * (1.0f / 512.0f);
    else
    {
        int32_t e = (int32_t) ((ds_f2u(a) >> 23) & 0xFF) - 127;
        float step = ds_u2f((uint32_t) (e - 3 + 127) << 23);
        q = rintf(a / step) * step;
    }
    return copysignf(q, v);
}

// e2m1 rounding, ties to the even code (quant.fp4_round); v already clamped to +-6
DS_HD float ds_fp4_round(float v)
{
    float a = fabsf(v), o = 0.0f;
    if (a > 0.25f) o = 0.5f;
    if (a >= 0.75f) o = 1.0f;
    if (a > 1.25f) o = 1.5f;
    if (a >= 1.75f) o = 2.0f;
    if (a > 2.5f) o = 3.0f;
    if (a >= 3.5f) o = 4.0f;
    if (a > 5.0f) o = 6.0f;
    return copysignf(o, v);
}

// scale of one fp8 group (act_quant_fp8_): ue8m0 of amax / 448, amax >= 1e-4
DS_HD float ds_fp8_scale(float amax) { return ds_round_pow2(fmaxf(amax, 1e-4f) * ds_inv448()); }
// one value of an fp8 group: dequantised e4m3(clamp(x / s, +-448)) * s
DS_HD float ds_fp8_q(float x, float s) { return ds_e4m3_round(fminf(fmaxf(x / s, -448.0f), 448.0f)) * s; }

// scale of one fp4 group with a ue8m0 scale (indexer q/k, group 32): amax >= 6 * 2^-126
DS_HD float ds_fp4_scale_ue8m0(float amax) { return ds_round_pow2(fmaxf(amax, 6.0f * 1.1754943508222875e-38f) * ds_inv6()); }
// scale of one fp4 group with an e4m3 scale (compressed KV, group 16): e4m3(amax / 6), amax >= 6 * 2^-9
DS_HD float ds_fp4_scale_e4m3(float amax) { return ds_e4m3_round(fminf(fmaxf(amax, 6.0f * 0.001953125f) / 6.0f, 448.0f)); }
DS_HD float ds_fp4_q(float x, float s) { return ds_fp4_round(fminf(fmaxf(x / s, -6.0f), 6.0f)) * s; }

// e2m1 magnitude of a nibble (sign in bit 3), lut {0,.5,1,1.5,2,3,4,6}
DS_HD float ds_e2m1(uint32_t nib)
{
    uint32_t m = nib & 7u;
    float v = m < 4 ? 0.5f * (float) m : (m == 4 ? 2.0f : (m == 5 ? 3.0f : (m == 6 ? 4.0f : 6.0f)));
    return (nib & 8u) ? -v : v;
}
// e8m0 byte -> 2^(b-127) (255 = NaN in the spec, never present in the table; 0 -> 2^-127 which is a fp32 subnormal)
DS_HD float ds_e8m0(uint32_t b) { return b == 0 ? ds_u2f(0x00400000u) : ds_u2f(b << 23); }

// Engram hash helpers: int64 wrap-around multiply, floor modulo with a positive modulus (torch.remainder)
DS_HD int64_t ds_mul_wrap(int64_t a, int64_t b) { return (int64_t) ((uint64_t) a * (uint64_t) b); }
DS_HD int64_t ds_floor_mod(int64_t x, int64_t p) { int64_t r = x % p; return r < 0 ? r + p : r; }

// Engram n-gram tokens at absolute position `pos` (NgramHasher.__call__): cache[p] holds the compressed id of position p
// (-1 = DEAD / masked). toks[sh] = id at pos - sh, or `pad` once any earlier slot (pos < sh, or DEAD) blocked the chain.
DS_HD void ds_engram_toks(const int64_t* cache, int64_t pos, int ng, int64_t pad, int64_t* toks)
{
    bool blocked = false;
    for (int sh = 0; sh < ng; sh++)
    {
        int64_t p = pos - sh;
        int64_t src = cache[p < 0 ? 0 : p];
        blocked = blocked || (pos < sh) || (src == -1);
        toks[sh] = blocked ? pad : src;
    }
}
// One hash column of one position and layer: column c = (i - 1) * nh + h for n-gram size i = 2..ng, head h.
DS_HD int64_t ds_engram_hash_col(const int64_t* toks, int nh, int c, const int64_t* mult, const int64_t* primes,
                                 const int64_t* offsets)
{
    int ni = c / nh + 1;
    int64_t rolling = ds_mul_wrap(toks[0], mult[0]);
    for (int i = 1; i <= ni; i++) rolling ^= ds_mul_wrap(toks[i], mult[i]);
    return ds_floor_mod(rolling, primes[c]) + offsets[c];
}
// All (ng - 1) * nh columns of one position and layer (reference: rolling xor of id * multiplier, % prime + offset).
DS_HD void ds_engram_hash(const int64_t* toks, int ng, int nh, const int64_t* mult, const int64_t* primes,
                          const int64_t* offsets, int64_t* out)
{
    for (int c = 0; c < (ng - 1) * nh; c++) out[c] = ds_engram_hash_col(toks, nh, c, mult, primes, offsets);
}
