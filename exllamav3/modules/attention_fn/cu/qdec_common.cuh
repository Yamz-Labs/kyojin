// shared helpers of the qdec kernels (typedefs, cp.async, fp16x2 math, mma, bit-plane dequant); prepended by qdec_cuda.py
typedef unsigned int u32;
typedef unsigned short u16;

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred)
{
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_u32(dst)), "l"(src), "r"(sz));
}
__device__ __forceinline__ u32 hsub2(u32 a, u32 b) { u32 d; asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 hmul2(u32 a, u32 b) { u32 d; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 hfma2(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }

__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 movtrans(u32 a) { u32 d; asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a)); return d; }

__device__ __forceinline__ u32 pack_h2(float lo, float hi)
{
    u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d;
}
__device__ __forceinline__ float h2f(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }

// inverse of the grid step as fp16 bits (2^-(B-1)), duplicated in both halves
template<int B> __device__ __forceinline__ u32 inv2() { return (u32)((15 - (B - 1)) << 10) * 0x10001u; }
template<int B> __device__ __forceinline__ u32 c2() { return (u32)(0x6400 | (1 << (B - 1))) * 0x10001u; }

// half2(scale_a, scale_b) -> half2 of grid-step scales
template<int B> __device__ __forceinline__ u32 scale2(u16 a, u16 b) { return hmul2((u32)a | ((u32)b << 16), inv2<B>()); }

// two adjacent raw values (positions p, p+1, p even) of a token's group -> packed in the two 16-bit halves
template<int B> __device__ __forceinline__ u32 raw_pair(const u32* base, int p)
{
    u32 raw = 0; int pb = 0; bool first = true;
    if (B & 8) { u32 v = (base[pb + (p >> 2)] >> ((p & 3) << 3)) & 0xFFFFu; raw = (v & 0xFFu) | ((v & 0xFF00u) << 8); pb += 8; first = false; }
    if (B & 4) { u32 v = (base[pb + (p >> 3)] >> ((p & 7) << 2)) & 0xFFu; u32 x = (v & 0xFu) | ((v & 0xF0u) << 12); raw = first ? x : ((raw << 4) | x); pb += 4; first = false; }
    if (B & 2) { u32 v = (base[pb + (p >> 4)] >> ((p & 15) << 1)) & 0xFu; u32 x = (v & 3u) | ((v & 12u) << 14); raw = first ? x : ((raw << 2) | x); pb += 2; first = false; }
    if (B & 1) { u32 v = (base[pb + (p >> 5)] >> (p & 31)) & 3u; u32 x = (v & 1u) | ((v & 2u) << 15); raw = first ? x : ((raw << 1) | x); }
    return raw;
}
// one raw value at position p
template<int B> __device__ __forceinline__ u32 raw_one(const u32* base, int p)
{
    u32 raw = 0; int pb = 0; bool first = true;
    if (B & 8) { raw = (base[pb + (p >> 2)] >> ((p & 3) << 3)) & 0xFFu; pb += 8; first = false; }
    if (B & 4) { u32 x = (base[pb + (p >> 3)] >> ((p & 7) << 2)) & 0xFu; raw = first ? x : ((raw << 4) | x); pb += 4; first = false; }
    if (B & 2) { u32 x = (base[pb + (p >> 4)] >> ((p & 15) << 1)) & 3u; raw = first ? x : ((raw << 2) | x); pb += 2; first = false; }
    if (B & 1) { u32 x = (base[pb + (p >> 5)] >> (p & 31)) & 1u; raw = first ? x : ((raw << 1) | x); }
    return raw;
}
// packed raw halves + half2 grid-step scale -> half2 values (raw - (2^(B-1) - 0.5)) * scale, one rounding
template<int B> __device__ __forceinline__ u32 dequant2(u32 raw2, u32 s2)
{
    u32 x = hsub2(raw2 | 0x64006400u, c2<B>());
    return hfma2(x, s2, hmul2(s2, 0x38003800u));   // 0x3800 = 0.5
}

__device__ __forceinline__ void cp4(void* dst, const void* src, bool pred)
{
    int sz = pred ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(smem_u32(dst)), "l"(src), "r"(sz));
}
