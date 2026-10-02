#pragma once

// Small-m GEMV path for the EXL3 GEMM, QTIP-style structure (see Cornell-RelaxML/qtip,
// qtip-kernels/src/inference.cu) on the unmodified EXL3 format:
//
// - warps split k and never synchronize during the main loop: no block-wide pipeline barriers;
//   B streams straight to registers with ld.global.cs (evict-first; B is single-use) behind a
//   register prefetch ring
// - the two-word bit windows of the trellis stream are resolved in-warp: with SMEM_STAGE = false
//   via lane shuffles (the extraction helpers in exl3_dq.cuh read exactly two words per lane, at
//   lane-computable indices), with SMEM_STAGE = true by staging the tile words through
//   warp-private shared memory and calling the standard dq_dispatch
// - one m16n8k16 MMA pair per 16x16 weight tile with fp16 accumulation, folded to fp32 on a fixed
//   cadence; per-block cross-warp reduction over the k splits through shared memory
//
// Same launch signature as exl3_gemm_kernel so kernel args and graph parameter patching are
// interchangeable. CUDA uses a cooperative launch with one grid.sync after the input Hadamard
// stage and one before the output stage. HIP launches the Hadamard stages separately and runs only
// the main GEMV loop here. 2, 3 and 4 bpw.
//
// CFG 0 ("narrow", 512 threads, 2 n-tiles/warp, 16 k-splits) wins at attention-projection sizes;
// CFG 1 ("wide", 256 threads, 4 n-tiles/warp, 8 k-splits) wins at large-n FFN sizes. CFG 2
// ("prefill", 128 threads, 4 n-tiles/warp, 4 k-splits) serves the 16-row grouped-MoE body.
// MMODE 0 is the m == 1 fast path, MMODE 1 covers 2 <= m <= 8, and HIP MMODE 2 covers
// 9 <= m <= 16 with row-guarded fragment loads.

#if !defined(USE_ROCM) && !defined(__HIPCC__)
#include <cooperative_groups.h>
#endif
#include "hadamard_inner.cuh"   // also pulls in ../compat.cuh -> compat_rocm.cuh (FragB, shims) on ROCm
#if defined(USE_ROCM) || defined(__HIPCC__)
// gfx12 (RDNA4) tensor-core path: v_wmma_f32_16x16x16_f16 via the on-device-oracle-verified
// primitive. Must come after hadamard_inner.cuh so compat_rocm.cuh's FragB is already in
// scope (hip_mma.cuh defers to it when present).
#include "../hip/hip_mma.cuh"
#else
#include "../ptx.cuh"
#endif
#include "exl3_dq.cuh"
#include "exl3_kernel_map.cuh"

#define EXL3_GEMV_MAX_M 8

namespace exl3_gemv_ns {

#if !defined(USE_ROCM) && !defined(__HIPCC__)

// mma.m16n8k16 with the A operand supplied as two FragB halves, fp16 accumulate
__device__ __forceinline__ void mma_ab_h(const FragB& a01, const FragB& a23, const FragB& b, FragC_h& c)
{
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
           "r"(bb[0]), "r"(bb[1])
    );
}

#endif  // !USE_ROCM && !__HIPCC__

// mul1 codebook pair decode via dp4a byte sum (bit-identical to the vabsdiff4 form)
__device__ __forceinline__ half2 decode_pair_cb2_dp4a_(uint32_t x0, uint32_t x1)
{
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    half_uint16 h0((uint16_t) sum0);
    half_uint16 h1((uint16_t) sum1);
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

// gfx1201 fast form: V_SAD_U8(p, 0, 0x6400) == byte_sum(p) + 0x6400 == __dp4a(p,
// 0x01010101, 0x6400) bit-exactly (the multiplier is 1 and the addend rides the
// instruction's accumulator operand). gfx1201 lacks V_DOT4, but V_SAD_U8 is present
// (MC encoding 0xd6 0x22 on gfx1201). This replaces ~5 VALU ops per state with ~3
// (mul + sad + pair-pack) — the decode chain was instruction-issue-bound.
// gfx11.5 (gfx1150/1151/1152, Strix Point / Strix Halo) also has V_SAD_U8, and additionally the
// native V_DOT4_U32_U8 that gfx12 lacks. Either is one VALU op versus six for the compat_rocm.cuh
// dp4a polyfill (and, bfe, lshr, bfe, add3, add3) that the generic branch below expands to.
// EXL3_MUL1_DECODE_DOT4=1 at build time selects the dot4 form on gfx11.5 for A/B.
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define EXL3_HAVE_SAD_DECODE 1
#if !defined(EXL3_CB_HAVE_SAD)
__device__ __forceinline__ uint32_t exl3_sad_u8_(uint32_t p, uint32_t addend)
{
    uint32_t u;
    asm("v_sad_u8 %0, %1, %2, %3" : "=v"(u) : "v"(p), "n"(0), "n"(addend));
    return u;
}
#endif

__device__ __forceinline__ half2 decode_pair_cb2_sad_(uint32_t x0, uint32_t x1)
{
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    // sum_i = u_i + 1024 (<= 2029, no 16-bit carry): pack the two states' values
    // into one dword, then the same hfma2 as the dp4a form.
#if (defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)) && defined(EXL3_MUL1_DECODE_DOT4)
    const uint32_t sum1 = __builtin_amdgcn_udot4(x1, 0x01010101u, 0x6400u, false);
    const uint32_t sum0 = __builtin_amdgcn_udot4(x0, 0x01010101u, 0x6400u, false);
#else
    const uint32_t sum1 = exl3_sad_u8_(x1, 0x6400u);
    const uint32_t sum0 = exl3_sad_u8_(x0, 0x6400u);
#endif
    const uint32_t packed = (sum1 << 16) + sum0;
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    half_uint16 h0((uint16_t) packed);
    half_uint16 h1((uint16_t)(packed >> 16));
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}
#endif

template <int cb>
__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    if constexpr (cb == 2)
    {
#if defined(EXL3_HAVE_SAD_DECODE)
        f0[0] = decode_pair_cb2_sad_(w0, w1);
        f0[1] = decode_pair_cb2_sad_(w2, w3);
        f1[0] = decode_pair_cb2_sad_(w4, w5);
        f1[1] = decode_pair_cb2_sad_(w6, w7);
#else
        f0[0] = decode_pair_cb2_dp4a_(w0, w1);
        f0[1] = decode_pair_cb2_dp4a_(w2, w3);
        f1[0] = decode_pair_cb2_dp4a_(w4, w5);
        f1[1] = decode_pair_cb2_dp4a_(w6, w7);
#endif
    }
    else
    {
        f0[0] = decode_3inst_2<cb>(w0, w1);
        f0[1] = decode_3inst_2<cb>(w2, w3);
        f1[0] = decode_3inst_2<cb>(w4, w5);
        f1[1] = decode_3inst_2<cb>(w6, w7);
    }
}

// Window extraction from two already-loaded words, same order as dq8_aligned_4bits
template <int cb>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB& f0, FragB& f1)
{
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8_aligned_2bits: the two words and the funnel shift are lane-dependent
template <int cb>
__device__ __forceinline__ void dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8<3, cb, 4> with the per-lane funnel alignment precomputed
template <int cb>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2);
    w6 = w7 >> 3;
    w5 = w6 >> 3;
    w4 = w5 >> 3;
    w3 = fshift(b, a, s2 + 12);
    w2 = w3 >> 3;
    w1 = w2 >> 3;
    w0 = w1 >> 3;
    decode8<cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

}  // namespace exl3_gemv_ns

template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE, bool HALF = false>
#if defined(USE_ROCM) || defined(__HIPCC__)
__device__ __forceinline__
void exl3_gemv_kernel_body(EXL3_GEMM_ARGS)
#else
__global__ __launch_bounds__(CFG == 0 ? 512 : 256)
void exl3_gemv_kernel(EXL3_GEMM_ARGS)
#endif
{
#if defined(USE_ROCM) || defined(__HIPCC__)
    static_assert(HALF ? (bits >= 1 && bits <= 3 && cb == 2) :
                  (bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6),
                  "HIP GEMV supports integer 2..6 bpw and half-integer 1.5..3.5 bpw with MUL1");
#else
    static_assert(!HALF && (bits == 2 || bits == 3 || bits == 4),
                  "CUDA GEMV supports integer 2, 3 and 4 bpw only");
#endif
    static_assert(CFG >= 0 && CFG <= 2, "unsupported GEMV configuration");
    constexpr int WK   = CFG == 0 ? 16 : CFG == 1 ? 8 : 4;  // k-split (warps per block)
    constexpr int WNT  = CFG == 0 ? 2 : 4;                    // adjacent n-tiles per warp
#if defined(EXL3_HIP_PF_DEPTH) && (defined(USE_ROCM) || defined(__HIPCC__))
    // gfx1151 experiment: the CFG 1/2 ring depth of 2 leaves each lane with only two B rows
    // in flight; on a 20-CU part with ~2 us DRAM latency the grouped-MoE body (CFG 2) is
    // latency-bound. Build with EXL3_HIP_DEFINES="EXL3_HIP_PF_DEPTH=4" to widen it.
    constexpr int PF   = CFG == 0 ? 4 : EXL3_HIP_PF_DEPTH;
#else
    constexpr int PF   = CFG == 0 ? 4 : 2;                    // prefetch ring depth
#endif
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;

#if defined(USE_ROCM) || defined(__HIPCC__)
    // HIP: the WMMA fp32 C fragment holds rows 0..7 in lanes 0..15 and rows 8..15
    // in lanes 16..31. MMODE 2 reduces those halves in separate passes.
    constexpr int ROWS = MMODE == 0 ? 1 : MMODE == 2 ? 16 : EXL3_GEMV_MAX_M;
    constexpr int RED_ROWS = MMODE == 2 ? 8 : ROWS;
#else
    constexpr int FOLD = CFG == 0 ? 4 : 2;      // fp16->fp32 fold cadence (divides PF)
    constexpr int ROWS = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    constexpr int RED_ROWS = ROWS;
#endif

    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;
    constexpr bool TWO_PER_LOAD = HALF ? bits == 1 : bits == 2;
    constexpr int LSTRIDE = TWO_PER_LOAD ? 2 * TWORDS : (TWORDS < 32 ? TWORDS : 32);
    constexpr int LOADS = (WNT * TWORDS + LSTRIDE - 1) / LSTRIDE;
    static_assert(!TWO_PER_LOAD || WNT % 2 == 0,
                  "two tiles per warp load requires an even tile count");
    static_assert(LOADS * LSTRIDE >= WNT * TWORDS &&
                  LOADS * LSTRIDE < WNT * TWORDS + LSTRIDE,
                  "lane-wide loads must cover staged tiles with at most one padded tail");

#if !defined(USE_ROCM) && !defined(__HIPCC__)
    auto grid = cooperative_groups::this_grid();

    // Input scales and Hadamard transform, same as exl3_gemm_kernel
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }
#endif

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;   // uint32 per k-slice row
    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // A fragment row indices for this lane
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const size_t a_row1 = (size_t) (r0 + 8) * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;
    const bool r1_ok = MMODE == 2 && r0 + 8 < size_m;

    // Per-lane extraction constants (see dq8_aligned_2bits / dq8<3, cb, 4> in exl3_dq.cuh)
    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    // LDS banks: 32 x 1 dword, bank = dword_addr % 32. float is one dword, so the
    // innermost extent is the stride between consecutive rows -- and the per-warp
    // stride for the cross-warp reduce below is RED_ROWS*COLS. With COLS a multiple
    // of 32 that product is too, so every warp's slice shares one bank:
    //
    //   for (j = 0; j < WK; ++j) sum += sh_red[j][r][c];   // fixed r, c
    //
    //   COLS=64 (CFG2): 512-dword stride -> 1 bank for WK accesses  -> WK-way
    //   COLS=65       : 520-dword stride -> WK distinct banks       -> clean
    //
    // Padding by one dword costs +128 B at CFG=2 (+512 B at CFG=0).
    // The store path is 2-way regardless (16 distinct c over 32 lanes) -- inherent
    // to the fragment layout, not addressed here.
#if defined(EXL3_HIP_RED_PAD) && defined(USE_ROCM)
    #define EXL3_RED_COLS (COLS + 1)
#else
    #define EXL3_RED_COLS COLS
#endif
    __shared__ float sh_red[WK][RED_ROWS][EXL3_RED_COLS];
#if defined(USE_ROCM) || defined(__HIPCC__)
    // HIP: staged extraction always on (gfx12 WMMA operands must not come from __shfl_sync;
    // see hip_mma.cuh). SMEM_STAGE is ignored and the staging buffer is unconditional.
    __shared__ uint32_t sh_stage[WK][LOADS * LSTRIDE];
#else
    [[maybe_unused]] __shared__ uint32_t sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];
#endif

    for (int group = blockIdx.x; group < num_groups; group += gridDim.x)
    {
        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        // Prefetch ring (indices must be compile-time or pf lands in local memory)
        auto ld_b = [&] (int i, int l) -> uint32_t
        {
#if defined(USE_ROCM) || defined(__HIPCC__)
            // K5's narrow configuration has a 16-word padded tail. Guard every staged
            // word so the last output group cannot read into the next slice or past B.
            const int stage_word = l * LSTRIDE + lane;
            return lane < LSTRIDE && stage_word < WNT * TWORDS
                ? *(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
#else
            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
#endif
        };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

#if defined(EXL3_HIP_A_RING)
        // A fragment prefetch ring, same depth as B. Without it every k-slice issues its A loads
        // and immediately consumes them, and the compiler's s_waitcnt vmcnt(0) for that consume
        // exposes a full global-memory round trip per slice (measured: waves stalled ~half their
        // lifetime, 36% VALU issue, 144 of 236 GB/s on gfx1151 at m=3). A is tiny; latency is not.
        auto ld_a = [&] (int i, FragB& a01, FragB& a23)
        {
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = r1_ok ? A2[a_row1 + a_col] : hzero;
            a23[1] = r1_ok ? A2[a_row1 + a_col + 4] : hzero;
        };
        FragB pa01[PF], pa23[PF];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn) ld_a(d, pa01[d], pa23[d]);
#endif

#if defined(USE_ROCM) || defined(__HIPCC__)
        // fp32 accumulator: gfx12 WMMA accumulates in fp32 natively, one 16x16 tile per
        // call, no cadence fold needed
        FragC8 acc[WNT] = {};
#else
        FragC_h ch[WNT][2] = {};
        float2 acc0[WNT][2] = {};
#endif

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

#if defined(EXL3_HIP_A_RING)
            // A fragment for this slice from the prefetch ring (loaded PF slices ago)
            FragB a01 = pa01[d], a23 = pa23[d];
#endif

#if !defined(EXL3_HIP_PF_AFTER_STAGE)
            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
#if defined(EXL3_HIP_A_RING)
                ld_a(i + PF, pa01[d], pa23[d]);
#endif
            }
#endif

#if defined(USE_ROCM) || defined(__HIPCC__)
            // Staged extraction: write valid tile words to warp-private smem, then decode.
            // The padded K5 tail is never read or written.
            __syncwarp();
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
            {
                const int stage_word = l * LSTRIDE + lane;
                if (lane < LSTRIDE && stage_word < WNT * TWORDS)
                    sh_stage[warp][stage_word] = bw[l];
            }
            __syncwarp();
#if defined(EXL3_HIP_PF_AFTER_STAGE)
            // Next B row into the ring only now: nothing below this point needs it this
            // iteration, so the wait the compiler emits for the stage store above cannot
            // include it, and the load overlaps the extraction + WMMA of this slice.
            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
#if defined(EXL3_HIP_A_RING)
                ld_a(i + PF, pa01[d], pa23[d]);
#endif
            }
#endif
#else
            if constexpr (SMEM_STAGE)
            {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (bits != 3 || lane < 24)
                        sh_stage[warp][l * LSTRIDE + lane] = bw[l];
                __syncwarp();
            }
#endif

#if !defined(EXL3_HIP_A_RING)
            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = r1_ok ? A2[a_row1 + a_col] : hzero;
            a23[1] = r1_ok ? A2[a_row1 + a_col + 4] : hzero;
#endif

#if defined(EXL3_HIP_WMMA_GFX12)
            // Every adjacent N tile uses the same A rows for this K slice. Assemble the full
            // operand before the B tiles reuse the warp-private staging buffer.
            HipFp16x8 a_frag = assemble_a_frag_gfx12(a01, a23);
#elif defined(EXL3_HIP_WMMA_GFX115)
            HipFp16x16 a_frag = assemble_a_frag_gfx115(a01, a23);
#endif

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
#if defined(USE_ROCM) || defined(__HIPCC__)
                const uint32_t* tp = &sh_stage[warp][t * TWORDS];
                if constexpr (HALF || bits == 5 || bits == 6)
                    dq_dispatch<bits, cb, HALF>(tp, lane * 8, f0, f1);
                else if constexpr (bits == 4)
                    exl3_gemv_ns::dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
                else if constexpr (bits == 2)
                    exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                else
                    exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);

                // One WMMA covers the full 16x16 tile: f0 = cols 0..7 (b_hi), f1 = cols 8..15
                // (b_lo); operands are reassembled through the LDS staging inside hip_mma.cuh,
                // never via lane shuffles
#if defined(EXL3_HIP_WMMA_GFX12)
                mma_ab_h_hip_gfx12_preassembled_a(a_frag, f0, f1, acc[t]);
#elif defined(EXL3_HIP_WMMA_GFX115)
                mma_ab_h_hip_gfx115_preassembled_a(a_frag, f0, f1, acc[t]);
#else
                mma_ab_h_hip(a01, a23, f0, f1, acc[t]);
#endif
#else
                if constexpr (SMEM_STAGE)
                {
                    const uint32_t* tp = &sh_stage[warp][t * TWORDS];
                    if constexpr (bits == 4)
                        exl3_gemv_ns::dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
                    else if constexpr (bits == 2)
                        exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                    else
                        exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    // Two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
#endif
            }

#if !defined(USE_ROCM) && !defined(__HIPCC__)
            if ((d + 1) % FOLD == 0 || i + 1 == myn)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
#endif
        }
        }

#if defined(USE_ROCM) || defined(__HIPCC__)
        // Cross-warp reduction over the k splits, HIP C-fragment layout (oracle-verified):
        //   C(lane, reg) = C[row = (lane>>4)*8 + reg][col = lane & 15]
        if constexpr (MMODE == 2)
        {
            // MMODE 2 reuses eight shared rows for the two C-fragment half-warps, avoiding
            // the 16-row reduction buffer that limits gfx12 occupancy to one block per WGP.
#if defined(EXL3_HIP_WMMA_GFX115)
            // gfx11.5 C rows interleave (row = 2*reg + half), so the low eight rows
            // live in regs 0..3 of BOTH half-warps rather than in lanes 0..15.
            {
                const int h = lane >> 4;
                const int c = lane & 15;
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r)
                        sh_red[warp][2 * r + h][t * 16 + c] = acc[t][r];
            }
#else
            if (lane < 16)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < 8; ++r)
                        sh_red[warp][r][t * 16 + lane] = acc[t][r];
            }
#endif
            __syncthreads();

            const int rows_out = min(size_m, 8);
            for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
            {
                const int r = idx / COLS;
                const int c = idx % COLS;
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < WK; ++j)
                    sum += sh_red[j][r][c];
                const int col = group * COLS + c;
                if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
                else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
            }
            __syncthreads();

            if (size_m > 8)
            {
#if defined(EXL3_HIP_WMMA_GFX115)
                // upper eight rows: regs 4..7, both half-warps (row = 2*(r-4) + half + 8)
                {
                    const int h = lane >> 4;
                    const int c = lane & 15;
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t)
                        #pragma unroll
                        for (int r = 4; r < 8; ++r)
                            sh_red[warp][2 * (r - 4) + h][t * 16 + c] = acc[t][r];
                }
#else
                if (lane >= 16)
                {
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t)
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            sh_red[warp][r][t * 16 + (lane & 15)] = acc[t][r];
                }
#endif
                __syncthreads();

                const int upper_rows_out = min(size_m - 8, 8);
                for (int idx = threadIdx.x; idx < COLS * upper_rows_out; idx += THREADS)
                {
                    const int r = idx / COLS;
                    const int c = idx % COLS;
                    float sum = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < WK; ++j)
                        sum += sh_red[j][r][c];
                    const int col = group * COLS + c;
                    if constexpr (c_fp32) ((float*) C)[(size_t) (r + 8) * size_n + col] = sum;
                    else                  ((half*)  C)[(size_t) (r + 8) * size_n + col] = __float2half_rn(sum);
                }
                __syncthreads();
            }
        }
        else
        {
            // m <= 8 keeps all valid rows in the first half-warp: lanes 0..15 each hold one
            // column of the tile across all 8 row registers.
#if defined(EXL3_HIP_WMMA_GFX115)
            // gfx11.5: row = 2*reg + (lane>>4). For m <= 8 the rows are spread over
            // regs 0..3 of both half-warps, not regs 0..7 of the first one.
            {
                const int h = lane >> 4;
                const int c = lane & 15;
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r)
                    {
                        const int row = 2 * r + h;
                        if (row < min(ROWS, 8))
                            sh_red[warp][row][t * 16 + c] = acc[t][r];
                    }
            }
#else
            if (lane < 16)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int r = 0; r < min(ROWS, 8); ++r)
                        sh_red[warp][r][t * 16 + lane] = acc[t][r];
            }
#endif
            __syncthreads();

            const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
            for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
            {
                const int r = idx / COLS;
                const int c = idx % COLS;
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < WK; ++j)
                    sum += sh_red[j][r][c];
                const int col = group * COLS + c;
                if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
                else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
            }
            __syncthreads();
        }
#else
        // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
        // tile*16 + frag*8 + 2*(l%4) (+1)
        {
            const int c0 = 2 * (lane & 3);
            const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
            const int sr0 = MMODE == 0 ? 0 : r0;
            if (store0)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        const int col = t * 16 + f * 8 + c0;
                        sh_red[warp][sr0][col + 0] = acc0[t][f].x;
                        sh_red[warp][sr0][col + 1] = acc0[t][f].y;
                    }
            }
        }
#endif
#if !defined(USE_ROCM) && !defined(__HIPCC__)
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
        {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
            else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
        __syncthreads();
#endif
    }

#if !defined(USE_ROCM) && !defined(__HIPCC__)
    // Output scales and Hadamard transform, same semantics as the inner GEMM epilogue
    {
        grid.sync();

        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner<false, true>
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
            else
                had_hf_r_128_inner<false, true>
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
    }
#endif
}

#if defined(USE_ROCM) || defined(__HIPCC__)
template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE, bool HALF = false>
__global__ __launch_bounds__(CFG == 0 ? 512 : CFG == 1 ? 256 : 128)
void exl3_gemv_kernel(EXL3_GEMM_ARGS)
{
    exl3_gemv_kernel_body<bits, c_fp32, cb, MMODE, CFG, SMEM_STAGE, HALF>
    (A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh);
}
#endif

#undef EXL3_RED_COLS
