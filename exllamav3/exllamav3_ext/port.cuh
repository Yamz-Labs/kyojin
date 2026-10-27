#pragma once

// One-source portability layer for the Kyojin kernels (HIP / gfx11.5 and CUDA / sm_86).
// Every macro expands to the exact construct the ROCm build used before, so the ROCm build is unchanged;
// the CUDA side gets the nearest equivalent. Add a macro here instead of a new #if in a kernel.

#define PORT_PRAGMA(x) _Pragma(#x)
// nvcc does not macro-expand the count of "#pragma unroll N"; going through _Pragma expands it first (same pragma on HIP).
#define PORT_UNROLL(n) PORT_PRAGMA(unroll n)

// Inline-asm register class of a 32-bit float: VGPR on AMDGPU, .f32 register on NVPTX.
#if defined(USE_ROCM)
#define PORT_ASM_F32 "v"
#else
#define PORT_ASM_F32 "f"
#endif

// ------------------------------------------------------------------------------------------------
// exl3_dec family (quant/exl3_dec.cu, RDNA wave32 code; one CUDA warp = one RDNA wave).
// ROCm: the gfx11.5 builtins the kernels were written with. CUDA: nearest sm_86 equivalent; same
// values everywhere except PORT_FDOT2_F32 (fp32 rounding of the pair sum may differ from v_dot2_f32_f16).
#if defined(USE_ROCM)
#define PORT_UBFE16(w, s) __builtin_amdgcn_ubfe(w, s, 16)
#define PORT_ALIGNBIT(hi, lo, s) __builtin_amdgcn_alignbit(hi, lo, s)
#define PORT_SAD_U8(a, acc) __builtin_amdgcn_sad_u8(a, 0u, acc)
#define PORT_SAD_HI_U8(a, acc) __builtin_amdgcn_sad_hi_u8(a, 0u, acc)
#define PORT_FDOT2_F32(a, b, acc) __builtin_amdgcn_fdot2(a, b, acc, false)
#define PORT_HFMA2(a, b, c) __builtin_elementwise_fma(a, b, c)
#define PORT_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#define PORT_LD_STREAM(p) __builtin_nontemporal_load(p)
#define PORT_LD_COHERENT(p) __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define PORT_SHFL_XOR(v, m) __shfl_xor(v, m)
// wave-uniform u32 into an SGPR (frees the VGPRs a uniform value would hold)
#define PORT_READFIRSTLANE(u) __builtin_amdgcn_readfirstlane(u)
#define PORT_TRAP() __builtin_trap()
#define PORT_F16_BITS_TO_F32(u) ((float) __builtin_bit_cast(_Float16, (unsigned short) (u)))
// u32 -> packed fp16 pair of the kernel's own f16x2 type
#define PORT_AS_F16X2(u) __builtin_bit_cast(f16x2, u)
#else
// (w >> s) & 0xffff
#define PORT_UBFE16(w, s) (((uint32_t) (w) >> (s)) & 0xffffu)
// low 32 bits of ({hi, lo} >> s), s in 0..31: shf.r.wrap
#define PORT_ALIGNBIT(hi, lo, s) __funnelshift_r(lo, hi, s)
// sum of the 4 bytes of a + acc / (that sum << 16) + acc: dp4a against 0x01010101
#define PORT_SAD_U8(a, acc) __dp4a((uint32_t) (a), 0x01010101u, (uint32_t) (acc))
#define PORT_SAD_HI_U8(a, acc) ((__dp4a((uint32_t) (a), 0x01010101u, 0u) << 16) + (uint32_t) (acc))
// acc + a.x * b.x + a.y * b.y in fp32 (fp16 x fp16 products are exact in fp32)
__device__ __forceinline__ float port_fdot2_f32(__half2 a, __half2 b, float acc)
{
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    return fmaf(fa.y, fb.y, fmaf(fa.x, fb.x, acc));
}
#define PORT_FDOT2_F32(a, b, acc) port_fdot2_f32(a, b, acc)
#define PORT_HFMA2(a, b, c) __hfma2(a, b, c)
#define PORT_SCHED_BARRIER() ((void) 0)
#define PORT_LD_STREAM(p) __ldcs(p)
// L2 load, not served from a stale L1 line (reads of other blocks' partials after __threadfence)
#define PORT_LD_COHERENT(p) __ldcg(p)
#define PORT_SHFL_XOR(v, m) __shfl_xor_sync(0xffffffffu, v, m)
#define PORT_READFIRSTLANE(u) (u)
#define PORT_TRAP() __trap()
#define PORT_F16_BITS_TO_F32(u) __half2float(__ushort_as_half((unsigned short) (u)))
// __half2 is not trivially copyable, so no __builtin_bit_cast
__device__ __forceinline__ __half2 port_as_half2(uint32_t u)
{
    __half2 h;
    memcpy(&h, &u, sizeof(h));
    return h;
}
#define PORT_AS_F16X2(u) port_as_half2(u)
#endif
