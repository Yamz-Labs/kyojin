// Batch-1 EXL3 decode kernels for RDNA3.5 (gfx1151). See exl3_dec.cuh.
//
// Layout facts used here (ref/frac_reconstruct.py, validated bit-exact against the port's HIP
// reconstruct kernels):
//   - trellis[kt][nt][WORDS] u32: tile (kt, nt) covers W rows 16kt.., columns 16nt.. (y = x @ W)
//   - the tile bit stream is MSB-first inside each u32; ring position p has its 16-bit window at
//     stream bits [S(p) - 16, S(p)) mod L, S(p) = (p >> 4) * BPB + sum_{j <= p & 15} D(j),
//     D(j) = KA (+1 for odd j when K = KA + 0.5)
//   - ring position p = 8t + 2jj + h maps to tile row r0 + h + 8 * (jj & 1) and tile column
//     t / 4 + 8 * (jj >> 1), r0 = 2 * (t % 4): each decoded pair is two consecutive rows of one
//     column, i.e. one v_dot2_f32_f16 against a pair of consecutive input values
//   - mul1 codebook: v = fp16(0x6400 + bytesum(w * 0x83DCD12D)) * fp16(0x1eee) + fp16(0xc931)
//   - W = diag(suh) H128 W_hat H128 diag(svh), H128 the normalised Sylvester Hadamard per 128 block

#if defined(USE_ROCM)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Tensor.h>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/empty.h>
#include "exl3_dec.cuh"
#include "../util.cuh"

// QWG_ABLATE=1 compiles runtime ablation arms into the dense gemv (env EXL3_QWG_DABL; timing only, garbage results by design):
// bit 1 = no trellis decode (loads kept alive by an xor), bit 2 = no DRAM stream (every k-tile read is k-row 0..ktw-1), bit 4 = no epilogue
#if !defined(QWG_ABLATE)
#define QWG_ABLATE 0
#endif

namespace exl3dec {

typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

constexpr int WPB = 8;              // waves per block
constexpr int THREADS = WPB * 32;
constexpr int STRIP = 512;          // output columns per block: 32 lanes x 16
constexpr float HAD_SCALE = 0.088388347648f;
constexpr int RED_STRIDE = 17;      // floats per lane row in the LDS reduction (16 + pad)
constexpr int KTW_MAX = 32;         // k-tiles per wave
constexpr int KTB_MAX = WPB * KTW_MAX;
// EXL3_DEC_RED2 1: two-stage block reduction through half the LDS (waves 4-7 store, waves 0-3
// add in place), so more blocks fit per WGP
#if !defined(EXL3_DEC_RED2)
#define EXL3_DEC_RED2 1
#endif
constexpr int RED_WAVES = EXL3_DEC_RED2 ? WPB / 2 : WPB;
#if !defined(EXL3_DEC_LDS_PAD)
#define EXL3_DEC_LDS_PAD 0
#endif
constexpr int SMEM_FLOATS = ((KTB_MAX * 16 / 2 > RED_WAVES * 32 * RED_STRIDE) ? KTB_MAX * 16 / 2 : RED_WAVES * 32 * RED_STRIDE) + EXL3_DEC_LDS_PAD;

template <int KB2>
struct Fmt
{
    static constexpr int KA = KB2 / 2;
    static constexpr int HALF = KB2 & 1;
    static constexpr int BPB = 16 * KA + 8 * HALF;
    static constexpr int L = 16 * BPB;
    static constexpr int WORDS = L / 32;
    static constexpr int VEC = WORDS / 4;
    static constexpr int send(int p)
    {
        return (p >> 4) * BPB + ((p & 15) + 1) * KA + (HALF ? ((p & 15) + 1) / 2 : 0);
    }
    static constexpr int wstart(int p) { return ((send(p) - 16) % L + L) % L; }
};

template <int I, int N, typename F>
__device__ __forceinline__ void static_for(F&& f)
{
    if constexpr (I < N)
    {
        f(std::integral_constant<int, I>{});
        static_for<I + 1, N>(f);
    }
}

// Raw 16-bit window of ring position P in the low half of the result; the upper half is garbage
// unless MASK (the product variants that read only 16 bits do not need the mask).
template <int KB2, int P, bool MASK>
__device__ __forceinline__ uint32_t window(const uint32_t* w)
{
    constexpr int s = Fmt<KB2>::wstart(P);
    constexpr int i = s >> 5;
    constexpr int o = s & 31;
    constexpr int W = Fmt<KB2>::WORDS;
    if constexpr (o <= 16)
        return __builtin_amdgcn_ubfe(w[i], 16 - o, 16);
    else if constexpr (MASK)
        return __builtin_amdgcn_alignbit(w[i], w[(i + 1) % W], 48 - o) & 0xffffu;
    else
        return __builtin_amdgcn_alignbit(w[i], w[(i + 1) % W], 48 - o);
}

// Product variants for p = (x & 0xffff) * 0x83DCD12D mod 2^32:
//   0: v_mul_lo_u32 on the masked window
//   1: two v_mul_u32_u24 + shift-add on the masked window
//   2: v_pk_mul_lo_u16 (x.lo * 0x83DC into the high half) + v_mad_u32_u16 (x.lo * 0xD12D + that);
//      both read only the low 16 bits of x, so the window needs no mask
#if !defined(EXL3_DEC_PROD)
#define EXL3_DEC_PROD 0
#endif
constexpr bool PROD_MASK = EXL3_DEC_PROD != 2;
#if !defined(EXL3_DEC_SB)
#define EXL3_DEC_SB 4
#endif

// Decode constants, per codebook (CB 2 = mul1, CB 1 = mcg). EXL3_DEC_SCONST 1: materialized once
// into SGPRs through opaque s_mov so every use is an 8-byte VOP3 with an SGPR operand instead of
// carrying a 32-bit literal (the fully unrolled tile decode is instruction-cache bound on gfx1151,
// bytes matter). The mcg codebook's decode yields the final fp16 weight directly, so its affine
// fold (kinv2/kbias2, applied in pair_dot/block_reduce) is the identity.
#if !defined(EXL3_DEC_SCONST)
#define EXL3_DEC_SCONST 1
#endif
struct DecConst
{
    uint32_t mul, sadacc, kinv2, kbias2;
};

template <int CB = 2>
__device__ __forceinline__ DecConst dec_const()
{
    DecConst c;
#if EXL3_DEC_SCONST
    if constexpr (CB >= 2)
    {
        asm volatile("s_mov_b32 %0, 0x83dcd12d" : "=s"(c.mul));
        asm volatile("s_mov_b32 %0, 0x64006400" : "=s"(c.sadacc));
        asm volatile("s_mov_b32 %0, 0x1eee1eee" : "=s"(c.kinv2));
        asm volatile("s_mov_b32 %0, 0xc931c931" : "=s"(c.kbias2));
    }
    else
    {
        asm volatile("s_mov_b32 %0, 0xcbac1fed" : "=s"(c.mul));
        asm volatile("s_mov_b32 %0, 0x3c003c00" : "=s"(c.kinv2));   // fp16 1.0 pair
        asm volatile("s_mov_b32 %0, 0x00000000" : "=s"(c.kbias2));  // fp16 0.0 pair
        c.sadacc = 0;
    }
#else
    if constexpr (CB >= 2)
    {
        c.mul = 0x83DCD12Du; c.sadacc = 0x64006400u; c.kinv2 = 0x1eee1eeeu; c.kbias2 = 0xc931c931u;
    }
    else
    {
        c.mul = 0xCBAC1FEDu; c.sadacc = 0u; c.kinv2 = 0x3c003c00u; c.kbias2 = 0u;
    }
#endif
    return c;
}

__device__ __forceinline__ uint32_t mul1_prod(uint32_t x, const DecConst& c)
{
#if EXL3_DEC_PROD == 0
    return x * c.mul;
#elif EXL3_DEC_PROD == 1
    return __umul24(x, 0xDCD12Du) + (__umul24(x, 0x83u) << 24);
#else
    uint32_t hi, p;
    asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(hi) : "v"(x), "s"(0x83DC0000u));
    asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(p) : "v"(x), "s"(0xD12Du), "v"(hi));
    return p;
#endif
}

// Two windows -> packed fp16 pair {0x6400 + bytesum(w0 * C), 0x6400 + bytesum(w1 * C)}, C = 0x83DCD12D.
// One volatile asm block: keeps the products next to their consumers (as separate non-volatile
// statements the scheduler hoisted hundreds of products ahead and spilled), two independent
// chains interleaved. Only the low 16 bits of w0/w1 are read.
__device__ __forceinline__ uint32_t mul1_pair(uint32_t w0, uint32_t w1, const DecConst& c)
{
#if EXL3_DEC_PROD == 2
    uint32_t h0, h1, r;
    asm volatile(
        "v_pk_mul_lo_u16 %1, %3, %5 op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %2, %4, %5 op_sel_hi:[0,1]\n\t"
        "v_mad_u32_u16 %1, %3, %6, %1\n\t"
        "v_mad_u32_u16 %2, %4, %6, %2\n\t"
        "v_sad_u8 %1, %1, 0, %7\n\t"
        "v_sad_hi_u8 %0, %2, 0, %1"
        : "=v"(r), "=&v"(h0), "=&v"(h1)
        : "v"(w0), "v"(w1), "s"(0x83DC0000u), "s"(0xD12Du), "s"(0x64006400u));
    return r;
#else
    const uint32_t p0 = mul1_prod(w0, c);
    const uint32_t p1 = mul1_prod(w1, c);
    const uint32_t s0 = __builtin_amdgcn_sad_u8(p0, 0u, c.sadacc);
    return __builtin_amdgcn_sad_hi_u8(p1, 0u, s0);
#endif
}

// MCG codebook (cb 1): the same 3-instruction construction as codebook.cuh decode_mcg_product_2 /
// decode_3inst<1>. One 16-bit state -> p = state * 0xCBAC1FED, then the lop3 (0x3b603b60 ^
// (p & 0x8fff8fff)) whose two fp16 halves sum to the weight. Unlike mul1 there is no byte-sum and
// no affine map: the fp16 sum IS the value, so the pair comes out directly (identity fold).
__device__ __forceinline__ uint32_t mcg_fold(uint32_t w, const DecConst& c)
{
    return 0x3b603b60u ^ ((w * c.mul) & 0x8fff8fffu);
}

__device__ __forceinline__ uint32_t mcg_pair(uint32_t w0, uint32_t w1, const DecConst& c)
{
    const uint32_t x0 = mcg_fold(w0, c);
    const uint32_t x1 = mcg_fold(w1, c);
    const half h0 = __hadd(__builtin_bit_cast(half, (unsigned short) (x0 & 0xffffu)),
                           __builtin_bit_cast(half, (unsigned short) (x0 >> 16)));
    const half h1 = __hadd(__builtin_bit_cast(half, (unsigned short) (x1 & 0xffffu)),
                           __builtin_bit_cast(half, (unsigned short) (x1 >> 16)));
    return (uint32_t) __builtin_bit_cast(unsigned short, h0) |
           ((uint32_t) __builtin_bit_cast(unsigned short, h1) << 16);
}

template <int CB>
__device__ __forceinline__ uint32_t pair_decode(uint32_t w0, uint32_t w1, const DecConst& c)
{
    if constexpr (CB >= 2) return mul1_pair(w0, w1, c);
    else return mcg_pair(w0, w1, c);
}

// EXL3_DEC_EXACT 1: each codebook value is rounded to fp16 exactly as the reference decode does
// (v = fma_f16(h, KINV, KBIAS), one v_pk_fma_f16 per pair) and dotted directly. 0: the affine map
// is folded into the block sums (one op per pair cheaper, but the unrounded values differ from
// the real weights by 2.1e-4 rms relative).
#if !defined(EXL3_DEC_EXACT)
#define EXL3_DEC_EXACT 1
#endif
// Codebook template ids: 1 = mcg, 2 = mul1, 3 = mul1 with the affine map folded into the block sums
// regardless of EXL3_DEC_EXACT (runtime knob EXL3_DEC_MOE_FOLD, routed MoE only, r37: one
// v_pk_fma_f16 per pair less: GLM MoE 309.7 -> 293.1 us/call, decode 49.74 -> 48.90 ms/tok). Default 1 since r39:
// not bitwise (rel-L2 6.2e-4 per call) and flips greedy ids, but the decode teacher-forced NLL is
// at the noise floor (-0.015 +- 0.068 %, 8 x 1024 tok @4K). EXL3_DEC_MOE_FOLD=0 restores the exact path.
template <int CB> constexpr bool dec_exact() { return CB == 3 ? false : (bool) EXL3_DEC_EXACT; }

template <int KB2, int M, int CB = 2>
__device__ __forceinline__ void pair_dot(const uint32_t* w, const f16x2* xp, float* acc, const DecConst& c)
{
    // mul1: lo = 0x6400 + bytesum(p0), hi = 0x6400 + bytesum(p1): two fp16 values in [1024, 2044],
    // the affine map to the real weights is folded into the block sums (see block_reduce).
    // mcg: the folded pair already is the fp16 weight pair (identity affine map).
    // PROD_MASK only matters for the mul1 product variants that read the low 16 bits; mcg
    // multiplies the full 32-bit window, so it always needs the masked form.
    constexpr bool MASK = (CB >= 2) ? PROD_MASK : true;
    const uint32_t h2 = pair_decode<CB>(window<KB2, 2 * M, MASK>(w), window<KB2, 2 * M + 1, MASK>(w), c);
    f16x2 h = __builtin_bit_cast(f16x2, h2);
    if constexpr (dec_exact<CB>())
        h = __builtin_elementwise_fma(h, __builtin_bit_cast(f16x2, c.kinv2), __builtin_bit_cast(f16x2, c.kbias2));
    constexpr int t = M >> 2;
    constexpr int jj = M & 3;
    constexpr int q = (t & 3) + ((jj & 1) ? 4 : 0);
    constexpr int col = (t >> 2) + ((jj & 2) ? 8 : 0);
    acc[col] = __builtin_amdgcn_fdot2(h, xp[q], acc[col], false);
    // Scheduling fence every SB pairs: left free, the scheduler hoists the window/product work of a
    // whole tile ahead of the dot products and spills (hundreds of VGPRs)
    if constexpr ((M % EXL3_DEC_SB) == EXL3_DEC_SB - 1) __builtin_amdgcn_sched_barrier(0);
    if constexpr (M + 1 < 128) pair_dot<KB2, M + 1, CB>(w, xp, acc, c);
}


// Decode one tile (all 256 weights) and accumulate x . h into acc[16] (one per tile column), with
// h = fp16(1024 + bytesum) the codebook value before its affine map: the map
// v = h * KINV + KBIAS is applied once per block on the sums (see block_reduce), so the exact
// (unrounded) codebook value is used instead of its fp16 rounding.
template <int KB2, int CB = 2>
__device__ __forceinline__ void tile_dot(const uint32_t* w, const f16x2* xp, float* acc, const DecConst& c)
{
#if defined(EXL3_DEC_NODECODE)
    // Bandwidth probe (wrong results): consume every word, no decode
    uint32_t t = 0;
    #pragma unroll
    for (int i = 0; i < Fmt<KB2>::WORDS; ++i) t ^= w[i];
    acc[0] += (float) (t & 0xff) * (float) xp[0][0];
#else
    pair_dot<KB2, 0, CB>(w, xp, acc, c);
#endif
}

// (EXL3_GEMV_R_DEC1): R-row twin of pair_dot/tile_dot. Each weight pair is decoded ONCE
// and applied to every row (tile_dot per row re-decodes the whole tile per row, which made each
// extra verify row cost ~7 % of the dense gemv time in pure ALU). Per row, the fdot2 operands, the
// accumulator and the order of accumulation are exactly tile_dot's, so the output is bit-identical.
// Rows >= nrows (RM padding) are computed on whatever LDS holds and never stored.
template <int KB2, int M, int CB, int RM>
__device__ __forceinline__ void pair_dot_r(const uint32_t* w, const f16x2 (*xp)[8], float (*acc)[16], const DecConst& c)
{
    constexpr bool MASK = (CB >= 2) ? PROD_MASK : true;
    const uint32_t h2 = pair_decode<CB>(window<KB2, 2 * M, MASK>(w), window<KB2, 2 * M + 1, MASK>(w), c);
    f16x2 h = __builtin_bit_cast(f16x2, h2);
    if constexpr (dec_exact<CB>())
        h = __builtin_elementwise_fma(h, __builtin_bit_cast(f16x2, c.kinv2), __builtin_bit_cast(f16x2, c.kbias2));
    constexpr int t = M >> 2;
    constexpr int jj = M & 3;
    constexpr int q = (t & 3) + ((jj & 1) ? 4 : 0);
    constexpr int col = (t >> 2) + ((jj & 2) ? 8 : 0);
    #pragma unroll
    for (int j = 0; j < RM; ++j)
        acc[j][col] = __builtin_amdgcn_fdot2(h, xp[j][q], acc[j][col], false);
    if constexpr ((M % EXL3_DEC_SB) == EXL3_DEC_SB - 1) __builtin_amdgcn_sched_barrier(0);
    if constexpr (M + 1 < 128) pair_dot_r<KB2, M + 1, CB, RM>(w, xp, acc, c);
}

// fp16 codebook constants (codebook.cuh decode_3inst<2>)
__device__ __forceinline__ float cb_kinv() { return (float) __builtin_bit_cast(_Float16, (unsigned short) 0x1eee); }
__device__ __forceinline__ float cb_kbias() { return (float) __builtin_bit_cast(_Float16, (unsigned short) 0xc931); }

// The mcg decode is its own affine map (identity), so EXL3_DEC_EXACT 0 (affine folded into the
// block sums) and 1 (folded per value) coincide there.
template <int CB> __device__ __forceinline__ float cb_kinv_of() { return CB >= 2 ? cb_kinv() : 1.0f; }
template <int CB> __device__ __forceinline__ float cb_kbias_of() { return CB >= 2 ? cb_kbias() : 0.0f; }

template <int KB2>
__device__ __forceinline__ void load_tile(const u32x4* p, u32x4* r)
{
#if defined(EXL3_DEC_NODECODE_CONTIG)
    // Bandwidth probe: the wave's 32 tiles read as VEC fully contiguous 512-byte rows
    const int ln = threadIdx.x & 31;
    const u32x4* base = p - ln * Fmt<KB2>::VEC;
    #pragma unroll
    for (int v = 0; v < Fmt<KB2>::VEC; ++v) r[v] = __builtin_nontemporal_load(base + v * 32 + ln);
#else
    #pragma unroll
    for (int v = 0; v < Fmt<KB2>::VEC; ++v) r[v] = __builtin_nontemporal_load(p + v);
#endif
}

// Tile-register ring depth: RING tiles in flight per lane (RING - 1 tiles of decode hide each load).
// Measured on gfx1151 (test_dec.py --bench): the unrolled tile decode is ~10 KB of code, and two
// copies in the loop (ring 2) thrash the instruction cache: ring 1 is 10 % faster on the MoE
// (K2/K2.5) and on K4, ring 3-4 are 25-40 % slower. K5/K6 tiles keep ring 2 (head: 1.5 % better).
#if !defined(EXL3_DEC_RING_SMALL)
#define EXL3_DEC_RING_SMALL 1
#endif
#if !defined(EXL3_DEC_RING_BIG)
#define EXL3_DEC_RING_BIG 2
#endif
// EXL3_DEC_PF 1: ring-1 lanes prefetch one tile ahead (see lane_gemv). Bit-identical.
#if !defined(EXL3_DEC_PF)
#define EXL3_DEC_PF 0
#endif
template <int KB2> constexpr int ring_of() { return Fmt<KB2>::WORDS <= 32 ? EXL3_DEC_RING_SMALL : EXL3_DEC_RING_BIG; }

// Accumulate x[kt0 .. kt0 + n) . W[.., nt] for one lane (n % RING == 0), tile words streamed
// through a RING-deep register ring. xs: LDS input (fp16) for this wave's k range, 16 per k-tile.
template <int KB2, int CB = 2, int RING_ = 0, bool NODEC = false>
__device__ __forceinline__ void lane_gemv(const u32x4* tiles, size_t tstride, const half* xs, int n, float* acc)
{
    constexpr int VEC = Fmt<KB2>::VEC;
    constexpr int RING = RING_ ? RING_ : ring_of<KB2>();
    const DecConst dc = dec_const<CB>();
#if EXL3_DEC_PF
    // Ring-1 with one tile of prefetch and a single copy of the unrolled decode: tile i+1's load is
    // issued before tile i's decode (ring 2 overlaps the same way but duplicates the decode code).
    if constexpr (RING == 1)
    {
        u32x4 nxt[VEC], cur[VEC];
        if (n > 0) load_tile<KB2>(tiles, nxt);
        for (int i = 0; i < n; ++i)
        {
            #pragma unroll
            for (int v = 0; v < VEC; ++v) cur[v] = nxt[v];
            if (i + 1 < n) load_tile<KB2>(tiles + (i + 1) * tstride, nxt);
            f16x2 xp[8];
            const f16x2* xq = reinterpret_cast<const f16x2*>(xs + i * 16);
            #pragma unroll
            for (int q = 0; q < 8; ++q) xp[q] = xq[q];
            tile_dot<KB2, CB>(reinterpret_cast<const uint32_t*>(cur), xp, acc, dc);
            __builtin_amdgcn_sched_barrier(0);
        }
        return;
    }
#endif
    u32x4 buf[RING][VEC];
    #pragma unroll
    for (int r = 0; r < RING; ++r) if (r < n) load_tile<KB2>(tiles + r * tstride, buf[r]);
    for (int i0 = 0; i0 < n; i0 += RING)
    {
        #pragma unroll
        for (int r = 0; r < RING; ++r)
        {
            const int i = i0 + r;
            if (i >= n) break;
            f16x2 xp[8];
            const f16x2* xq = reinterpret_cast<const f16x2*>(xs + i * 16);
            #pragma unroll
            for (int q = 0; q < 8; ++q) xp[q] = xq[q];
            if constexpr (NODEC) { uint32_t xx = 0; for (int v = 0; v < VEC; ++v) xx ^= buf[r][v].x ^ buf[r][v].y ^ buf[r][v].z ^ buf[r][v].w; acc[0] += __uint_as_float(xx & 0x3fffffff); }
            else tile_dot<KB2, CB>(reinterpret_cast<const uint32_t*>(buf[r]), xp, acc, dc);
            if (i + RING < n) load_tile<KB2>(tiles + (i + RING) * tstride, buf[r]);
            __builtin_amdgcn_sched_barrier(0);   // keep the scheduler from interleaving tiles (VGPR blowup)
        }
    }
}

// R-row variant: each tile is loaded once (identical RING pipeline to lane_gemv), then
// applied to `nrows` independent input rows in the exact same per-row tile_dot/pair_dot arithmetic
// lane_gemv would run for a batch-1 call on that row -- so row j of acc is bit-identical to what a
// batch-1 lane_gemv(..., xs_rows[j], n, acc_row_j) call would produce. xp is scoped inside the row
// loop (not hoisted per-row), so only one row's 8 xp registers are live at a time; acc[MOE_R_MAX][16]
// (only the first `nrows` rows used) is the only cost that scales with rows.
// RM: compile-time row bound. The row loop used to run to the runtime nrows, which
// indexes acc[j] dynamically and pushes the whole accumulator array to scratch memory; unrolled to
// RM with a uniform `j < nrows` guard, acc stays in VGPRs. Arithmetic per row is unchanged.
// RING_ (default ring_of<KB2>) only sets how many tiles are in flight; the per-tile arithmetic and
// its order are the same for any ring depth, so it never changes the result.
template <int KB2, int CB, int RM, int RING_ = 0>
__device__ __forceinline__ void lane_gemv_r(const u32x4* tiles, size_t tstride, const half* const* xs_rows, int nrows, int n, float acc[][16])
{
    constexpr int VEC = Fmt<KB2>::VEC;
    constexpr int RING = RING_ > 0 ? RING_ : ring_of<KB2>();
    u32x4 buf[RING][VEC];
    const DecConst dc = dec_const<CB>();
    #pragma unroll
    for (int r = 0; r < RING; ++r) if (r < n) load_tile<KB2>(tiles + r * tstride, buf[r]);
    for (int i0 = 0; i0 < n; i0 += RING)
    {
        #pragma unroll
        for (int r = 0; r < RING; ++r)
        {
            const int i = i0 + r;
            if (i >= n) break;
            #pragma unroll
            for (int j = 0; j < RM; ++j)
            {
                if (j >= nrows) break;
                f16x2 xp[8];
                const f16x2* xq = reinterpret_cast<const f16x2*>(xs_rows[j] + i * 16);
                #pragma unroll
                for (int q = 0; q < 8; ++q) xp[q] = xq[q];
                tile_dot<KB2, CB>(reinterpret_cast<const uint32_t*>(buf[r]), xp, acc[j], dc);
            }
            if (i + RING < n) load_tile<KB2>(tiles + (i + RING) * tstride, buf[r]);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
}

// lane_gemv_r with the decode shared across rows (pair_dot_r). Same ring, same loads.
template <int KB2, int CB, int RM, int RING_ = 0>
__device__ __forceinline__ void lane_gemv_r1(const u32x4* tiles, size_t tstride, const half* const* xs_rows, int n, float acc[][16])
{
    constexpr int VEC = Fmt<KB2>::VEC;
    constexpr int RING = RING_ > 0 ? RING_ : ring_of<KB2>();
    u32x4 buf[RING][VEC];
    const DecConst dc = dec_const<CB>();
    #pragma unroll
    for (int r = 0; r < RING; ++r) if (r < n) load_tile<KB2>(tiles + r * tstride, buf[r]);
    for (int i0 = 0; i0 < n; i0 += RING)
    {
        #pragma unroll
        for (int r = 0; r < RING; ++r)
        {
            const int i = i0 + r;
            if (i >= n) break;
            f16x2 xp[RM][8];
            #pragma unroll
            for (int j = 0; j < RM; ++j)
            {
                const f16x2* xq = reinterpret_cast<const f16x2*>(xs_rows[j] + i * 16);
                #pragma unroll
                for (int q = 0; q < 8; ++q) xp[j][q] = xq[q];
            }
            pair_dot_r<KB2, 0, CB, RM>(reinterpret_cast<const uint32_t*>(buf[r]), xp, acc, dc);
            if (i + RING < n) load_tile<KB2>(tiles + (i + RING) * tstride, buf[r]);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
}

// In-place normalised... (unscaled) 128-point Walsh-Hadamard over a wave: lane holds elements
// 4 * lane + i, i = 0..3.
__device__ __forceinline__ void fwht128(float* v, int lane)
{
    const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2; v[2] = a0 - a2; v[1] = a1 + a3; v[3] = a1 - a3;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1)
    {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const float o = __shfl_xor(v[i], m);
            v[i] = (lane & m) ? (o - v[i]) : (v[i] + o);
        }
    }
}

__device__ __forceinline__ half sat_half(float f)
{
    return __float2half_rn(fminf(fmaxf(f, -65504.0f), 65504.0f));
}

__device__ __forceinline__ float wave_sum(float v)
{
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) v += __shfl_xor(v, m);
    return v;
}

// Prologue: xs[0 .. n) = fp16(H128(x[k0 ..] * suh[k0 ..]) * scale), n a multiple of 128, one
// 128-chunk per wave per step; suh == nullptr: plain copy (input already rotated). xsum[c] = sum
// of the stored fp16 values of chunk c.
// x_gstride > 0: input chunk g (inputs 128g .. 128g + 127) sits at x + g * x_gstride (e.g. the
// first 128 of every 192-wide head), otherwise x is contiguous.
__device__ __forceinline__ void prologue_x(const half* x, const half* suh, int k0, int n, half* xs, float* xsum, int wave, int lane, int x_gstride = 0)
{
    for (int c = wave; c < n / 128; c += WPB)
    {
        const int k = k0 + c * 128 + lane * 4;
        const int kx = x_gstride ? (k0 / 128 + c) * x_gstride + lane * 4 : k;
        const uint2 xr = *reinterpret_cast<const uint2*>(x + kx);
        const half* xh = reinterpret_cast<const half*>(&xr);
        half o[4];
#if defined(EXL3_DEC_DEBUG)
        if (false)
#else
        if (suh)
#endif
        {
            const uint2 sr = *reinterpret_cast<const uint2*>(suh + k);
            const half* sh = reinterpret_cast<const half*>(&sr);
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = __half2float(xh[i]) * __half2float(sh[i]);
            fwht128(v, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = sat_half(v[i] * HAD_SCALE);
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = xh[i];
        }
        *reinterpret_cast<uint2*>(xs + c * 128 + lane * 4) = *reinterpret_cast<uint2*>(o);
        float t = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) t += __half2float(o[i]);
        t = wave_sum(t);
        if (lane == 0) xsum[c] = t;
    }
}

// Block reduction over the WPB waves' k slices. Result: z[512] floats in LDS (column within strip).
template <int CB = 2>
__device__ __forceinline__ void block_reduce(const float* acc, float* red, float* z, float xsum, int wave, int lane, int tid)
{
    __syncthreads();   // red aliases the input staging
#if EXL3_DEC_RED2
    float* r = red + (wave & (RED_WAVES - 1)) * (32 * RED_STRIDE) + lane * RED_STRIDE;
    if (wave >= RED_WAVES)
    {
        #pragma unroll
        for (int c = 0; c < 16; ++c) r[c] = acc[c];
    }
    __syncthreads();
    if (wave < RED_WAVES)
    {
        #pragma unroll
        for (int c = 0; c < 16; ++c) r[c] += acc[c];
    }
#else
    float* r = red + wave * (32 * RED_STRIDE) + lane * RED_STRIDE;
    #pragma unroll
    for (int c = 0; c < 16; ++c) r[c] = acc[c];
#endif
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < STRIP / THREADS; ++j)
    {
        const int col = tid + j * THREADS;
        const int idx = (col >> 4) * RED_STRIDE + (col & 15);
        float s = 0.0f;
        #pragma unroll
        for (int w = 0; w < RED_WAVES; ++w) s += red[w * (32 * RED_STRIDE) + idx];
        if constexpr (dec_exact<CB>()) z[col] = s;
        else z[col] = s * cb_kinv_of<CB>() + xsum * cb_kbias_of<CB>();
    }
    __syncthreads();
}

// Cross-block k reduction: store this block's z to scratch, count arrivals; returns true in the
// last arriving block (for this counter), which then holds nothing yet (caller sums partials).
__device__ __forceinline__ bool arrive(const float* z, float* part, int ncols, int* counter, int target, int tid)
{
    for (int col = tid; col < ncols; col += THREADS) part[col] = z[col];
    __threadfence();
    __syncthreads();
    __shared__ int s_last;
    if (tid == 0)
    {
        const int old = atomicAdd(counter, 1);
        s_last = (old == target - 1);
        if (s_last) *counter = 0;   // no other block touches it again in this launch
    }
    __syncthreads();
    if (!s_last) return false;
    __threadfence();
    return true;
}

// Ragged k split: the k extent is G groups of 8 tiles (128 inputs); block kb of kbs takes groups
// [kb * G / kbs, (kb + 1) * G / kbs), i.e. ktw = group count tiles per wave (1 .. KTW_MAX).
__device__ __forceinline__ void k_split(int G, int kb, int kbs, int& k0, int& ktw)
{
    const int g0 = kb * G / kbs;
    const int g1 = (kb + 1) * G / kbs;
    k0 = g0 * 128;
    ktw = g1 - g0;
}

__device__ __forceinline__ float ld_coherent(const float* p)
{
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// ------------------------------------------------------------------------------------------------
// Dense GEMV, up to 4 matrices sharing x (and K bits), one launch.

struct Job
{
    const u32x4* trellis;
    const half* suh;
    const half* svh;
    void* out;
    int N;          // output columns
    int out_fp32;
    int strips;
    int block_base; // first block of this job
    int part_base;  // float offset into scratch
    int ctr_base;   // int offset into counters
};

struct Jobs
{
    Job j[4];
    int count;
    int K;
    int kbs;        // k blocks
    int x_gstride;  // strided input (see prologue_x), 0: contiguous
#if QWG_ABLATE
    int abl;
#endif
};

template <int KB2, int CB = 2>
__global__ __launch_bounds__(THREADS)
void gemv_kernel(const half* __restrict__ x, Jobs jobs, float* scratch, int* counters)
{
    constexpr int VEC = Fmt<KB2>::VEC;
    __shared__ __attribute__((aligned(16))) float smem[SMEM_FLOATS];
    half* xs = reinterpret_cast<half*>(smem);   // prologue + main loop
    float* red = smem;                          // block reduction (after a barrier)
    __shared__ float s_xsum[KTB_MAX * 16 / 128];
    __shared__ float z[STRIP];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    int ji = 0;
    #pragma unroll
    for (int i = 1; i < 4; ++i) if (i < jobs.count && (int) blockIdx.x >= jobs.j[i].block_base) ji = i;
    const Job& job = jobs.j[ji];
    const int local = blockIdx.x - job.block_base;
    const int strip = local / jobs.kbs;
    const int kb = local % jobs.kbs;
    const int NT = job.N / 16;
    int k0, ktw;
    k_split(jobs.K / 128, kb, jobs.kbs, k0, ktw);

    // Input rotation for this block's ktw * 128 input values
    prologue_x(x, job.suh, k0, ktw * 128, xs, s_xsum, wave, lane, jobs.x_gstride);
    __syncthreads();
    float xsum = 0.0f;
    for (int c = 0; c < ktw; ++c) xsum += s_xsum[c];

    float acc[16];
    #pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] = 0.0f;
    const int nt = strip * 32 + lane;
    if (nt < NT)
    {
#if QWG_ABLATE
        const int kt0 = (jobs.abl & 2) ? 0 : k0 / 16 + wave * ktw;
        const u32x4* tiles = job.trellis + ((size_t) kt0 * NT + nt) * VEC;
        if (jobs.abl & 1) lane_gemv<KB2, CB, 0, true>(tiles, (size_t) NT * VEC, xs + wave * ktw * 16, ktw, acc);
        else              lane_gemv<KB2, CB>(tiles, (size_t) NT * VEC, xs + wave * ktw * 16, ktw, acc);
#else
        const int kt0 = k0 / 16 + wave * ktw;
        const u32x4* tiles = job.trellis + ((size_t) kt0 * NT + nt) * VEC;
        lane_gemv<KB2, CB>(tiles, (size_t) NT * VEC, xs + wave * ktw * 16, ktw, acc);
#endif
    }
#if QWG_ABLATE
    if (jobs.abl & 4) { if (acc[0] == 1.2345f) *reinterpret_cast<float*>(job.out) = acc[1]; return; }
#endif
    block_reduce<CB>(acc, red, z, xsum, wave, lane, tid);

    const int col0 = strip * STRIP;
    const int ncols = min(STRIP, job.N - col0);
    if (jobs.kbs > 1)
    {
        float* part = scratch + job.part_base + (size_t) kb * job.N + col0;
        if (!arrive(z, part, ncols, counters + job.ctr_base + strip, jobs.kbs, tid)) return;
        for (int col = tid; col < ncols; col += THREADS)
        {
            float s = 0.0f;
            for (int b = 0; b < jobs.kbs; ++b) s += ld_coherent(scratch + job.part_base + (size_t) b * job.N + col0 + col);
            z[col] = s;
        }
        __syncthreads();
    }

    // Output rotation and store
    if (wave < 4 && wave * 128 < ncols)
    {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = z[wave * 128 + lane * 4 + i];
        const int col = col0 + wave * 128 + lane * 4;
#if !defined(EXL3_DEC_DEBUG)
        fwht128(v, lane);
        const uint2 sr = *reinterpret_cast<const uint2*>(job.svh + col);
        const half* sh = reinterpret_cast<const half*>(&sr);
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= HAD_SCALE * __half2float(sh[i]);
#endif
        if (job.out_fp32)
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(job.out) + col) = make_float4(v[0], v[1], v[2], v[3]);
        else
        {
            half o[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = sat_half(v[i]);
            *reinterpret_cast<uint2*>(reinterpret_cast<half*>(job.out) + col) = *reinterpret_cast<uint2*>(o);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Routed MoE, 1 token. Kernel A: gate and up GEMVs of every selected expert, epilogue fuses the
// output rotations, silu(g) * u and the down projection's input rotation. Kernel B: down GEMVs,
// epilogue fuses the output rotation, routing weight and the sum over experts.
//
// Clamped SwiGLU (GLM-5.3-Flash `swiglu_limit`, act_limit > 0): silu(gate) is clamped to +limit
// and the up operand to +/-limit, exactly as activation_kernels.cuh's ACT_SILU branch does for
// the reference ext.silu_mul path. act_limit == 0 takes the unclamped expression, bit-identical
// to the pre-act_limit kernels.
__device__ __forceinline__ half swiglu_act(float gf, half uh, float act_limit)
{
    const half s = sat_half(gf / (1.0f + __expf(-gf)));
    if (act_limit == 0.0f) return __hmul(s, uh);
    const half lim = __float2half_rn(act_limit);
    const half u = __hmin(__hmax(uh, __float2half_rn(-act_limit)), lim);
    return __hmul(__hmin(s, lim), u);
}

struct MoeArgs
{
    const int64_t* sel;
    const half* wts;
    const int64_t* gt; const int64_t* gs; const int64_t* gv;
    const int64_t* ut; const int64_t* us; const int64_t* uv;
    const int64_t* dt; const int64_t* ds; const int64_t* dv;
    half* act;          // [topk, I]
    float* out;         // [H]
    float* scratch;
    int* counters;
    int H, I, topk, experts;
    int accumulate;     // out += routed sum (out holds the residual) instead of out = sum
    float act_limit;    // swiglu_limit (0 = unclamped)
    // Shared expert folded in as an extra slot (decB8, EXL3_DEC_SHARED_FOLD; kernels with KB2S > 0
    // only): gate / up / down trellis, suh, svh. Weight 1, its own K and codebook. grid.y slot 0
    // is the shared expert (heaviest blocks dispatch first); its act/scratch/counter index is topk.
    const void* sht[3];
    const half* shs[3];
    const half* shv[3];
};

constexpr int CTR_A = 0;        // counters [topk * strips_I]
constexpr int CTR_B = 1024;     // counters [strips_H]
constexpr int CTR_GEMV = 2048;  // dense GEMV counters

template <int KB2, int CB = 2, int KB2S = 0, int CBS = 2, int RS = 0>
__global__ __launch_bounds__(THREADS)
void moe_gu_kernel(const half* __restrict__ x, MoeArgs a, int kbs)
{
    constexpr int VEC = Fmt<KB2>::VEC;
    __shared__ __attribute__((aligned(16))) float smem[SMEM_FLOATS];
    half* xs = reinterpret_cast<half*>(smem);   // prologue + main loop
    float* red = smem;                          // block reduction (after a barrier)
    __shared__ float s_xsum[KTB_MAX * 16 / 128];
    __shared__ float z[2 * STRIP];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int strip = blockIdx.x / kbs;
    const int kb = blockIdx.x % kbs;
    const bool sh = KB2S > 0 && blockIdx.y == 0;
    const int slot = KB2S > 0 ? (sh ? a.topk : (int) blockIdx.y - 1) : (int) blockIdx.y;
    const int proj = blockIdx.z;
    const int strips = a.I / STRIP;
    const int64_t e = sh ? 0 : a.sel[slot];
    const bool valid = sh || (e >= 0 && e < a.experts);
    const int NT = a.I / 16;

    const u32x4* trellis = sh ? reinterpret_cast<const u32x4*>(a.sht[proj]) :
        valid ? reinterpret_cast<const u32x4*>((proj ? a.ut : a.gt)[e]) : nullptr;
    const half* suh = sh ? a.shs[proj] :
        valid ? reinterpret_cast<const half*>((proj ? a.us : a.gs)[e]) : nullptr;

    int k0, ktw;
    k_split(a.H / 128, kb, kbs, k0, ktw);
    prologue_x(x, suh, k0, ktw * 128, xs, s_xsum, wave, lane);
    __syncthreads();
    float xsum = 0.0f;
    for (int c = 0; c < ktw; ++c) xsum += s_xsum[c];

    float acc[16];
    #pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] = 0.0f;
    if constexpr (KB2S > 0)
    {
        if (sh)
        {
            constexpr int VS = Fmt<KB2S>::VEC;
            const int nt = strip * 32 + lane;
            const int kt0 = k0 / 16 + wave * ktw;
            const u32x4* tiles = trellis + ((size_t) kt0 * NT + nt) * VS;
            lane_gemv<KB2S, CBS, RS>(tiles, (size_t) NT * VS, xs + wave * ktw * 16, ktw, acc);
            block_reduce<CBS>(acc, red, z, xsum, wave, lane, tid);
        }
    }
    if (!sh)
    {
        if (valid)
        {
            const int nt = strip * 32 + lane;
            const int kt0 = k0 / 16 + wave * ktw;
            const u32x4* tiles = trellis + ((size_t) kt0 * NT + nt) * VEC;
            lane_gemv<KB2, CB>(tiles, (size_t) NT * VEC, xs + wave * ktw * 16, ktw, acc);
        }
        block_reduce<CB>(acc, red, z, xsum, wave, lane, tid);
    }

    const int col0 = strip * STRIP;
    float* part_base = a.scratch;
    float* part = part_base + ((size_t) (slot * 2 + proj) * kbs + kb) * a.I + col0;
    if (!arrive(z, part, STRIP, a.counters + CTR_A + slot * strips + strip, 2 * kbs, tid)) return;
    for (int col = tid; col < 2 * STRIP; col += THREADS)
    {
        const int p = col / STRIP;
        const int c = col % STRIP;
        float s = 0.0f;
        for (int b = 0; b < kbs; ++b) s += ld_coherent(part_base + ((size_t) (slot * 2 + p) * kbs + b) * a.I + col0 + c);
        z[col] = s;
    }
    __syncthreads();

    if (wave < 4)
    {
        const int col = col0 + wave * 128 + lane * 4;
        float g[4], u[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            g[i] = z[wave * 128 + lane * 4 + i];
            u[i] = z[STRIP + wave * 128 + lane * 4 + i];
        }
        fwht128(g, lane);
        fwht128(u, lane);
        float v[4];
        if (valid)
        {
            const half* gsv = (sh ? a.shv[0] : reinterpret_cast<const half*>(a.gv[e])) + col;
            const half* usv = (sh ? a.shv[1] : reinterpret_cast<const half*>(a.uv[e])) + col;
            const half* dsu = (sh ? a.shs[2] : reinterpret_cast<const half*>(a.ds[e])) + col;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const half gh = sat_half(g[i] * HAD_SCALE * __half2float(gsv[i]));
                const half uh = sat_half(u[i] * HAD_SCALE * __half2float(usv[i]));
                const float gf = __half2float(gh);
                const half ah = swiglu_act(gf, uh, a.act_limit);
                v[i] = __half2float(ah) * __half2float(dsu[i]);
            }
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = 0.0f;
        }
        fwht128(v, lane);
        half o[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = sat_half(v[i] * HAD_SCALE);
        *reinterpret_cast<uint2*>(a.act + (size_t) slot * a.I + col) = *reinterpret_cast<uint2*>(o);
    }
}

template <int KB2, int CB = 2, int KB2S = 0, int CBS = 2, int RS = 0>
__global__ __launch_bounds__(THREADS)
void moe_down_kernel(MoeArgs a, int kbs)
{
    constexpr int VEC = Fmt<KB2>::VEC;
    __shared__ __attribute__((aligned(16))) float smem[SMEM_FLOATS];
    half* xs = reinterpret_cast<half*>(smem);   // prologue + main loop
    float* red = smem;                          // block reduction (after a barrier)
    __shared__ float s_xsum[KTB_MAX * 16 / 128];
    __shared__ float z[STRIP];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int strip = blockIdx.x / kbs;
    const int kb = blockIdx.x % kbs;
    const bool sh = KB2S > 0 && blockIdx.y == 0;
    const int slot = KB2S > 0 ? (sh ? a.topk : (int) blockIdx.y - 1) : (int) blockIdx.y;
    const int64_t e = sh ? 0 : a.sel[slot];
    const bool valid = sh || (e >= 0 && e < a.experts);
    const int NT = a.H / 16;

    int k0, ktw;
    k_split(a.I / 128, kb, kbs, k0, ktw);
    prologue_x(a.act + (size_t) slot * a.I, nullptr, k0, ktw * 128, xs, s_xsum, wave, lane);
    __syncthreads();
    float xsum = 0.0f;
    for (int c = 0; c < ktw; ++c) xsum += s_xsum[c];

    float acc[16];
    #pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] = 0.0f;
    if constexpr (KB2S > 0)
    {
        if (sh)
        {
            constexpr int VS = Fmt<KB2S>::VEC;
            const u32x4* trellis = reinterpret_cast<const u32x4*>(a.sht[2]);
            const int nt = strip * 32 + lane;
            const int kt0 = k0 / 16 + wave * ktw;
            const u32x4* tiles = trellis + ((size_t) kt0 * NT + nt) * VS;
            lane_gemv<KB2S, CBS, RS>(tiles, (size_t) NT * VS, xs + wave * ktw * 16, ktw, acc);
            block_reduce<CBS>(acc, red, z, xsum, wave, lane, tid);
        }
    }
    if (!sh)
    {
        if (valid)
        {
            const u32x4* trellis = reinterpret_cast<const u32x4*>(a.dt[e]);
            const int nt = strip * 32 + lane;
            const int kt0 = k0 / 16 + wave * ktw;
            const u32x4* tiles = trellis + ((size_t) kt0 * NT + nt) * VEC;
            lane_gemv<KB2, CB>(tiles, (size_t) NT * VEC, xs + wave * ktw * 16, ktw, acc);
        }
        block_reduce<CB>(acc, red, z, xsum, wave, lane, tid);
    }

    const int col0 = strip * STRIP;
    float* part_base = a.scratch;
    float* part = part_base + ((size_t) slot * kbs + kb) * a.H + col0;
    if (!arrive(z, part, STRIP, a.counters + CTR_B + strip, (a.topk + (KB2S > 0 ? 1 : 0)) * kbs, tid)) return;

    if (wave < 4)
    {
        const int col = col0 + wave * 128 + lane * 4;
        float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        // Folded shared expert: routed sum, then the shared expert, then the residual (the
        // unfolded order: routed out, += shared, residual add after the MLP)
        if (KB2S == 0 && a.accumulate)
        {
            const float4 r = *reinterpret_cast<const float4*>(a.out + col);
            sum[0] = r.x; sum[1] = r.y; sum[2] = r.z; sum[3] = r.w;
        }
        for (int s = 0; s < a.topk; ++s)
        {
            const int64_t es = a.sel[s];
            if (es < 0 || es >= a.experts) continue;
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float t = 0.0f;
                for (int b = 0; b < kbs; ++b) t += ld_coherent(part_base + ((size_t) s * kbs + b) * a.H + col + i);
                v[i] = t;
            }
            fwht128(v, lane);
            const half* sv = reinterpret_cast<const half*>(a.dv[es]) + col;
            const float w = __half2float(a.wts[s]);
            #pragma unroll
            for (int i = 0; i < 4; ++i) sum[i] += v[i] * HAD_SCALE * __half2float(sv[i]) * w;
        }
        if constexpr (KB2S > 0)
        {
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float t = 0.0f;
                for (int b = 0; b < kbs; ++b) t += ld_coherent(part_base + ((size_t) a.topk * kbs + b) * a.H + col + i);
                v[i] = t;
            }
            fwht128(v, lane);
            const half* sv = a.shv[2] + col;
            #pragma unroll
            for (int i = 0; i < 4; ++i) sum[i] += v[i] * HAD_SCALE * __half2float(sv[i]);
            if (a.accumulate)
            {
                const float4 r = *reinterpret_cast<const float4*>(a.out + col);
                sum[0] += r.x; sum[1] += r.y; sum[2] += r.z; sum[3] += r.w;
            }
        }
        *reinterpret_cast<float4*>(a.out + col) = make_float4(sum[0], sum[1], sum[2], sum[3]);
    }
}


// ------------------------------------------------------------------------------------------------
// R-row union MoE: verify-window MoE, R<=8 rows sharing one launch. Each *unique*
// expert across the R rows' top-k selections is decoded exactly once (grid.y = unique-expert
// index u, not row) and its lane_gemv accumulation is run once per row assigned to it, in the
// exact per-lane arithmetic order (tile_dot/pair_dot, same acc[16] sequence) a batch-1
// exl3_dec_moe call would use for that row -- so weight bytes read scale with the union size, not
// R * topk, while each row's result is bit-identical to a fresh batch-1 call. The down-projection
// combine (topk-weighted sum, sequential in the row's own k order) cannot happen inside the
// per-unique-expert block (its topk siblings may be handled by other blocks finishing at
// different times), so it is split into a 3rd, trivial kernel: moe_down_kernel_r writes each
// (row, k) assignment's rotated-and-scaled partial to down_part[row, k, :], and
// moe_combine_kernel_r sums, per row, over k = 0 .. topk-1 in that exact order (same
// left-to-right ((v * HAD_SCALE) * sv) * w grouping as the batch-1 epilogue), reproducing its
// rounding.
//
// Host builds the union/assignment (CPU-side, R * topk <= 64 entries, negligible next to the
// GEMVs) and packs it into `assign[U * R]`: assign[u * R + j] = row * topk + k for the j-th row
// assigned to unique expert u, or -1. A row selects each expert at most once, so R slots per
// unique expert are always enough.

struct MoeArgsR
{
    const int64_t* usel;     // [U] unique expert ids
    const int* assign;       // [U * R], row*topk+k or -1
    const int64_t* sel;      // [R * topk] original per-row selected experts (for down_svh lookup)
    const half* wts;         // [R * topk]
    const int64_t* gt; const int64_t* gs; const int64_t* gv;
    const int64_t* ut; const int64_t* us; const int64_t* uv;
    const int64_t* dt; const int64_t* ds; const int64_t* dv;
    const half* x;            // [R, H]
    half* act;                // [R * topk, I]
    float* down_part;         // [R * topk, H]
    float* out;                // [R, H]
    float* scratch;
    int* counters;
    int H, I, topk, experts, R, U;
    int accumulate;
    float act_limit;    // swiglu_limit (0 = unclamped), see swiglu_act
};

constexpr int MOE_R_MAX = 8;
constexpr int MOE_R_KTW_MAX = 16;   // max ktw for the R-row MoE kernels (gu:8, down:16 typical)
constexpr int CTR_RA = 0;                       // [U-independent, sized R*topk*strips_I at call time]
// CTR_RB placed after CTR_RA's worst case (R*topk*strips_I, topk<=10, I/STRIP<=32 -> <=2560);
// host computes the actual split and checks against counters.numel().

template <int KB2, int CB, int RM>
__global__ __launch_bounds__(THREADS)
void moe_gu_kernel_r(MoeArgsR a, int kbs)
{
    constexpr int VEC = Fmt<KB2>::VEC;
    // Size staging to max(row payload, block-reduction scratch). The old fixed RM=8 payload
    // cost 37.9 KB/block even for R=1; the reduction still needs SMEM_FLOATS.
    constexpr int R_SMEM_FLOATS = (RM * MOE_R_KTW_MAX * 128 / 2 > SMEM_FLOATS) ?
        RM * MOE_R_KTW_MAX * 128 / 2 : SMEM_FLOATS;
    __shared__ __attribute__((aligned(16))) float smem[R_SMEM_FLOATS];
    half* xs = reinterpret_cast<half*>(smem);
    float* red = smem;
    __shared__ float s_xsum[RM][MOE_R_KTW_MAX];
    __shared__ float z[2 * STRIP];
    __shared__ int s_assign[RM];
    __shared__ int s_n;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int strip = blockIdx.x / kbs;
    const int kb = blockIdx.x % kbs;
    const int u = blockIdx.y;
    const int proj = blockIdx.z;
    const int strips = a.I / STRIP;
    const int64_t e = a.usel[u];
    const bool valid = e >= 0 && e < a.experts;
    // Device-built union table (exl3_dec_moe_union_dev) launches grid.y = max possible U and pads
    // usel with -1: those blocks have no assigned row and leave at once (block-uniform exit).
    if (!valid) return;
    const int NT = a.I / 16;

    if (tid == 0)
    {
        int n = 0;
        for (int j = 0; j < a.R; ++j) { const int v = a.assign[u * MOE_R_MAX + j]; if (v >= 0) s_assign[n++] = v; }
        s_n = n;
    }
    __syncthreads();
    const int n = s_n;

    const u32x4* trellis = valid ? reinterpret_cast<const u32x4*>((proj ? a.ut : a.gt)[e]) : nullptr;
    const half* suh = valid ? reinterpret_cast<const half*>((proj ? a.us : a.gs)[e]) : nullptr;

    int k0, ktw;
    k_split(a.H / 128, kb, kbs, k0, ktw);
    for (int j = 0; j < n; ++j)
    {
        const int row = s_assign[j] / a.topk;
        prologue_x(a.x + (size_t) row * a.H, suh, k0, ktw * 128, xs + j * (MOE_R_KTW_MAX * 128), s_xsum[j], wave, lane);
    }
    __syncthreads();

    // Rows are processed in groups of GM <= 4: a [8][16] fp32 accumulator block does not fit the 256
    // VGPRs next to the tile ring and spills to scratch, and the RM=8 kernels then lose arrivals /
    // corrupt state at random on some bit-widths. A group re-reads the expert's tiles, but
    // only experts picked by more than GM rows of the window have a 2nd group (the arithmetic of
    // every row is unchanged, so the result is bit-identical to the RM=8 grouping).
    constexpr int GM = RM > 4 ? 4 : RM;
    const int col0 = strip * STRIP;
    for (int g0 = 0; g0 < n; g0 += GM)
    {
    const int ng = (RM <= 4) ? n : min(n - g0, GM);   // RM <= 4: one group = the original single pass
    float acc[GM][16];
    #pragma unroll
    for (int j = 0; j < GM; ++j)
        #pragma unroll
        for (int c = 0; c < 16; ++c) acc[j][c] = 0.0f;

    if (valid)
    {
        const int nt = strip * 32 + lane;
        const int kt0 = k0 / 16 + wave * ktw;
        const u32x4* tiles = trellis + ((size_t) kt0 * NT + nt) * VEC;
        const half* xs_rows[GM];
        #pragma unroll
        for (int j = 0; j < GM; ++j) xs_rows[j] = xs + (g0 + j) * (MOE_R_KTW_MAX * 128) + wave * ktw * 16;
        lane_gemv_r<KB2, CB, GM>(tiles, (size_t) NT * VEC, xs_rows, ng, ktw, acc);
    }

    #pragma unroll
    for (int jg = 0; jg < GM; ++jg)
    {
        if (jg >= ng) break;
        const int j = g0 + jg;
        float xsum = 0.0f;
        for (int c = 0; c < ktw; ++c) xsum += s_xsum[j][c];
        block_reduce<CB>(acc[jg], red, z, xsum, wave, lane, tid);
        const int slot = s_assign[j];
        float* part_base = a.scratch;
        float* part = part_base + ((size_t) (slot * 2 + proj) * kbs + kb) * a.I + col0;
        const bool last = arrive(z, part, STRIP, a.counters + CTR_RA + slot * strips + strip, 2 * kbs, tid);
        __syncthreads();
        if (!last) continue;
        for (int col = tid; col < 2 * STRIP; col += THREADS)
        {
            const int p = col / STRIP;
            const int c = col % STRIP;
            float s = 0.0f;
            for (int b = 0; b < kbs; ++b) s += ld_coherent(part_base + ((size_t) (slot * 2 + p) * kbs + b) * a.I + col0 + c);
            z[col] = s;
        }
        __syncthreads();
        if (wave < 4)
        {
            const int col = col0 + wave * 128 + lane * 4;
            float g[4], uu[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { g[i] = z[wave * 128 + lane * 4 + i]; uu[i] = z[STRIP + wave * 128 + lane * 4 + i]; }
            fwht128(g, lane);
            fwht128(uu, lane);
            float v[4];
            const half* gsv = reinterpret_cast<const half*>(a.gv[e]) + col;
            const half* usv = reinterpret_cast<const half*>(a.uv[e]) + col;
            const half* dsu = reinterpret_cast<const half*>(a.ds[e]) + col;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const half gh = sat_half(g[i] * HAD_SCALE * __half2float(gsv[i]));
                const half uh = sat_half(uu[i] * HAD_SCALE * __half2float(usv[i]));
                const float gf = __half2float(gh);
                const half ah = swiglu_act(gf, uh, a.act_limit);
                v[i] = __half2float(ah) * __half2float(dsu[i]);
            }
            fwht128(v, lane);
            half o[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = sat_half(v[i] * HAD_SCALE);
            *reinterpret_cast<uint2*>(a.act + (size_t) slot * a.I + col) = *reinterpret_cast<uint2*>(o);
        }
        __syncthreads();
    }
    if constexpr (RM <= 4) break;
    }
}

template <int KB2, int CB, int RM>
__global__ __launch_bounds__(THREADS)
void moe_down_kernel_r(MoeArgsR a, int kbs, int ctr_b_off)
{
    constexpr int VEC = Fmt<KB2>::VEC;
    // Match the compile-time row bound while retaining reduction scratch.
    constexpr int R_SMEM_FLOATS = (RM * MOE_R_KTW_MAX * 128 / 2 > SMEM_FLOATS) ?
        RM * MOE_R_KTW_MAX * 128 / 2 : SMEM_FLOATS;
    __shared__ __attribute__((aligned(16))) float smem[R_SMEM_FLOATS];
    half* xs = reinterpret_cast<half*>(smem);
    float* red = smem;
    __shared__ float s_xsum[RM][MOE_R_KTW_MAX];
    __shared__ float z[STRIP];
    __shared__ int s_assign[RM];
    __shared__ int s_n;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int strip = blockIdx.x / kbs;
    const int kb = blockIdx.x % kbs;
    const int u = blockIdx.y;
    const int64_t e = a.usel[u];
    const bool valid = e >= 0 && e < a.experts;
    // Device-built union table (exl3_dec_moe_union_dev) launches grid.y = max possible U and pads
    // usel with -1: those blocks have no assigned row and leave at once (block-uniform exit).
    if (!valid) return;
    const int NT = a.H / 16;

    if (tid == 0)
    {
        int n = 0;
        for (int j = 0; j < a.R; ++j) { const int v = a.assign[u * MOE_R_MAX + j]; if (v >= 0) s_assign[n++] = v; }
        s_n = n;
    }
    __syncthreads();
    const int n = s_n;

    int k0, ktw;
    k_split(a.I / 128, kb, kbs, k0, ktw);
    for (int j = 0; j < n; ++j)
        prologue_x(a.act + (size_t) s_assign[j] * a.I, nullptr, k0, ktw * 128, xs + j * (MOE_R_KTW_MAX * 128), s_xsum[j], wave, lane);
    __syncthreads();

    // Row groups of GM <= 4, see moe_gu_kernel_r.
    constexpr int GM = RM > 4 ? 4 : RM;
    const int col0 = strip * STRIP;
    for (int g0 = 0; g0 < n; g0 += GM)
    {
    const int ng = (RM <= 4) ? n : min(n - g0, GM);   // RM <= 4: one group = the original single pass
    float acc[GM][16];
    #pragma unroll
    for (int j = 0; j < GM; ++j)
        #pragma unroll
        for (int c = 0; c < 16; ++c) acc[j][c] = 0.0f;

    if (valid)
    {
        const u32x4* trellis = reinterpret_cast<const u32x4*>(a.dt[e]);
        const int nt = strip * 32 + lane;
        const int kt0 = k0 / 16 + wave * ktw;
        const u32x4* tiles = trellis + ((size_t) kt0 * NT + nt) * VEC;
        const half* xs_rows[GM];
        #pragma unroll
        for (int j = 0; j < GM; ++j) xs_rows[j] = xs + (g0 + j) * (MOE_R_KTW_MAX * 128) + wave * ktw * 16;
        lane_gemv_r<KB2, CB, GM>(tiles, (size_t) NT * VEC, xs_rows, ng, ktw, acc);
    }

    #pragma unroll
    for (int jg = 0; jg < GM; ++jg)
    {
        if (jg >= ng) break;
        const int j = g0 + jg;
        float xsum = 0.0f;
        for (int c = 0; c < ktw; ++c) xsum += s_xsum[j][c];
        block_reduce<CB>(acc[jg], red, z, xsum, wave, lane, tid);
        const int slot = s_assign[j];
        float* part_base = a.scratch;
        float* part = part_base + (size_t) slot * kbs * a.H + (size_t) kb * a.H + col0;
        const bool last = arrive(z, part, STRIP, a.counters + ctr_b_off + slot * (a.H / STRIP) + strip, kbs, tid);
        __syncthreads();
        if (!last) continue;
        if (wave < 4)
        {
            const int col = col0 + wave * 128 + lane * 4;
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float t = 0.0f;
                for (int b = 0; b < kbs; ++b) t += ld_coherent(part_base + (size_t) slot * kbs * a.H + (size_t) b * a.H + col + i);
                v[i] = t;
            }
            fwht128(v, lane);
            const half* sv = reinterpret_cast<const half*>(a.dv[e]) + col;
            float o[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = v[i] * HAD_SCALE * __half2float(sv[i]);
            *reinterpret_cast<float4*>(a.down_part + (size_t) slot * a.H + col) = make_float4(o[0], o[1], o[2], o[3]);
        }
        __syncthreads();
    }
    if constexpr (RM <= 4) break;
    }
}

// Device-side unique-expert / assignment table for the R-row union MoE: no host sync.
// Generic in E, topk (<= 64) and R (<= MOE_R_MAX). Slot i = row * topk + k. A slot "leads" when no
// earlier slot selected the same expert; leader i gets u = number of leaders before it, and
// assign[u * MOE_R_MAX + c] lists every slot with that expert in slot order (same as the host build).
// usel/assign entries for u >= U are padded with -1 so the fixed-size grid can skip them.
constexpr int MOE_UNION_MAX_SLOTS = MOE_R_MAX * 64;

__global__ void moe_union_build_kernel(const int64_t* __restrict__ sel, int nslots, int experts, int maxU,
                                       int64_t* __restrict__ usel, int* __restrict__ assign)
{
    __shared__ int64_t s_sel[MOE_UNION_MAX_SLOTS];
    __shared__ int s_lead[MOE_UNION_MAX_SLOTS];
    const int t = threadIdx.x;
    for (int i = t; i < nslots; i += blockDim.x) s_sel[i] = sel[i];
    __syncthreads();
    for (int i = t; i < nslots; i += blockDim.x)
    {
        const int64_t e = s_sel[i];
        int lead = (e >= 0 && e < experts) ? 1 : 0;
        for (int j = 0; j < i && lead; ++j) if (s_sel[j] == e) lead = 0;
        s_lead[i] = lead;
    }
    __syncthreads();
    int U = 0;
    for (int j = 0; j < nslots; ++j) U += s_lead[j];
    for (int i = t; i < nslots; i += blockDim.x)
    {
        if (!s_lead[i]) continue;
        int u = 0;
        for (int j = 0; j < i; ++j) u += s_lead[j];
        const int64_t e = s_sel[i];
        usel[u] = e;
        int c = 0;
        for (int j = i; j < nslots && c < MOE_R_MAX; ++j) if (s_sel[j] == e) assign[u * MOE_R_MAX + c++] = j;
        for (; c < MOE_R_MAX; ++c) assign[u * MOE_R_MAX + c] = -1;
    }
    for (int u = U + t; u < maxU; u += blockDim.x)
    {
        usel[u] = -1;
        for (int c = 0; c < MOE_R_MAX; ++c) assign[u * MOE_R_MAX + c] = -1;
    }
}

// One block per row, sequential sum over k = 0 .. topk-1 (same order as the batch-1 epilogue).
// grid (R, colsplit): blockIdx.y owns columns [y * H / gridDim.y, (y + 1) * H / gridDim.y). Each
// output element is still one thread's sequential k-sum, so any colsplit is bitwise equal.
__global__ void moe_combine_kernel_r(MoeArgsR a)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int cw = a.H / gridDim.y;
    const int c0 = blockIdx.y * cw;
    const int c1 = blockIdx.y + 1 == gridDim.y ? a.H : c0 + cw;
    for (int col = c0 + tid; col < c1; col += blockDim.x)
    {
        float sum = a.accumulate ? a.out[(size_t) row * a.H + col] : 0.0f;
        for (int k = 0; k < a.topk; ++k)
        {
            const int64_t es = a.sel[(size_t) row * a.topk + k];
            if (es < 0 || es >= a.experts) continue;
            const float w = __half2float(a.wts[(size_t) row * a.topk + k]);
            sum += a.down_part[((size_t) row * a.topk + k) * a.H + col] * w;
        }
        a.out[(size_t) row * a.H + col] = sum;
    }
}

// ------------------------------------------------------------------------------------------------
// Sigmoid top-k router (noaux_tc, one group) for 1 token: logits = fp16(x @ G) (G fp16 [H, E],
// k-major), score = sigmoid(logit), choice = score + bias, top-k by choice, weights = score / sum
// * scale. Matches routing_dots' torch fallback (fp16 logits). Blocks split H; the last block to
// arrive reduces and selects.

constexpr int ROUTER_KCHUNK = 64;
constexpr int CTR_ROUTER = 4000;          // one arrival counter per row: CTR_ROUTER + row
constexpr int ROUTER_NORM_MAX_ROWS = 8;

// Fused pre-norm (NORM): the block also computes the MLP pre-norm of r + xa (rms_norm_res_in
// semantics, the same arithmetic as rms_norm_kernel mode 2), writes its 64 normalized fp16 inputs to
// y and uses them for its logits; the last block applies the residual update r += xa (every block
// reads the old r first).
struct RouterNorm
{
    const void* xa;     // sublayer output, fp16 (xa_half) or fp32
    int xa_half;
    float* r;           // fp32 residual
    const void* w;      // norm weight: wtype 0 none, 1 fp16, 2 bf16
    int wtype;
    float eps, cbias, cscale;
    half* y;            // [H] fp16 normalized output
};

__device__ __forceinline__ float rn_xa(const RouterNorm& n, int i)
{
    if (n.xa_half) return fminf(fmaxf(__half2float(reinterpret_cast<const half*>(n.xa)[i]), -65504.0f), 65504.0f);
    return reinterpret_cast<const float*>(n.xa)[i];
}

__device__ __forceinline__ float rn_w(const RouterNorm& n, int i)
{
    if (n.wtype == 1) return __half2float(reinterpret_cast<const half*>(n.w)[i]) + n.cbias;
    if (n.wtype == 2) return __uint_as_float(((uint32_t) reinterpret_cast<const uint16_t*>(n.w)[i]) << 16) + n.cbias;
    return 1.0f;
}

template <bool NORM>
__global__ __launch_bounds__(256)
void router_kernel(const half* __restrict__ x, RouterNorm rn, const half* __restrict__ G, const float* bias,
                   int64_t* sel, half* wts, float* scratch, int* counters, int H, int E, int topk,
                   float scale)
{
    const int tid = threadIdx.x;
    const int kb = blockIdx.x;
    const int nkb = gridDim.x;
    // Multi-row launch (gridDim.y = NR verify rows, exl3_dec_router_norm with NR > 1): block row `row`
    // runs exactly the single-row code on row `row`'s slices of xa / r / y / sel / wts, its own
    // scratch slab and its own arrival counter, so every row is bitwise a single-row launch.
    // gridDim.y == 1 (all single-row callers): row 0, no offset, nothing changes.
    if (const int row = blockIdx.y)
    {
        x += (size_t) row * H;
        rn.xa = reinterpret_cast<const char*>(rn.xa) + (size_t) row * H * (rn.xa_half ? 2 : 4);
        rn.r += (size_t) row * H;
        rn.y += (size_t) row * H;
        sel += (size_t) row * topk;
        wts += (size_t) row * topk;
        scratch += (size_t) row * nkb * E;
        counters += row;
    }
    __shared__ float xs[ROUTER_KCHUNK];
    __shared__ float red[2048];
    const int k0 = kb * ROUTER_KCHUNK;
    // 16-byte loads: E / 8 threads cover one row of G, R = 256 / (E / 8) rows in flight; the first
    // 8 rows per thread are fetched before the (optional) norm so both latencies overlap
    const int gpr = E / 8;
    const int R = 256 / gpr;
    const int kr = tid / gpr;
    const int e8 = (tid % gpr) * 8;
    uint4 gv[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u)
    {
        const int k = kr + u * R;
        if (kr < R && k < ROUTER_KCHUNK) gv[u] = *reinterpret_cast<const uint4*>(G + (size_t) (k0 + k) * E + e8);
    }
    if constexpr (NORM)
    {
        float ss = 0.0f;
        #pragma unroll 8
        for (int i = tid; i < H; i += 256)
        {
            const float t = rn.r[i] + rn_xa(rn, i);
            ss += t * t;
        }
        ss = wave_sum(ss);
        __shared__ float s_red[8];
        if ((tid & 31) == 0) s_red[tid >> 5] = ss;
        __syncthreads();
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        const float sc = rsqrtf(tot / (float) H + rn.eps) * rn.cscale;
        if (tid < ROUTER_KCHUNK)
        {
            const int k = k0 + tid;
            const float t = rn.r[k] + rn_xa(rn, k);
            float o = t * sc;
            if (rn.wtype) o *= rn_w(rn, k);
            const half yh = __float2half_rn(o);
            rn.y[k] = yh;
            xs[tid] = __half2float(yh);
        }
    }
    else
    {
        if (tid < ROUTER_KCHUNK) xs[tid] = __half2float(x[k0 + tid]);
    }
    __syncthreads();
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    if (kr < R)
    {
        #pragma unroll
        for (int u = 0; u < 8; ++u)
        {
            const int k = kr + u * R;
            if (k < ROUTER_KCHUNK)
            {
                const half* gh = reinterpret_cast<const half*>(&gv[u]);
                const float xv = xs[k];
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] = fmaf(xv, __half2float(gh[i]), acc[i]);
            }
        }
        for (int k = kr + 8 * R; k < ROUTER_KCHUNK; k += R)
        {
            const uint4 g1 = *reinterpret_cast<const uint4*>(G + (size_t) (k0 + k) * E + e8);
            const half* gh = reinterpret_cast<const half*>(&g1);
            const float xv = xs[k];
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = fmaf(xv, __half2float(gh[i]), acc[i]);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) red[kr * E + e8 + i] = acc[i];
    }
    __syncthreads();
    for (int e = tid; e < E; e += 256)
    {
        float t = 0.0f;
        for (int r = 0; r < R; ++r) t += red[r * E + e];
        scratch[(size_t) kb * E + e] = t;
    }
    __threadfence();
    __syncthreads();
    __shared__ int s_last;
    if (tid == 0)
    {
        const int old = atomicAdd(counters + CTR_ROUTER, 1);
        s_last = (old == nkb - 1);
        if (s_last) counters[CTR_ROUTER] = 0;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if constexpr (NORM)
    {
        for (int i = tid; i < H; i += 256) rn.r[i] = rn.r[i] + rn_xa(rn, i);
    }
#if defined(EXL3_DEC_ROUTER_NOTAIL)
    if (tid == 0) { for (int j = 0; j < topk; ++j) { sel[j] = j; wts[j] = __float2half_rn(0.125f); } }
    return;
#endif

    __shared__ float score[1024];
    float ch[4];   // this thread's choices, e = tid + 256 * i (E <= 1024)
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int e = tid + 256 * i;
        ch[i] = -INFINITY;
        if (e < E)
        {
            float l = 0.0f;
            for (int b = 0; b < nkb; ++b) l += ld_coherent(scratch + (size_t) b * E + e);
            l = __half2float(__float2half_rn(l));
            const float sc = 1.0f / (1.0f + __expf(-l));
            score[e] = sc;
            ch[i] = sc + (bias ? bias[e] : 0.0f);
        }
    }

    // top-k: per round one wave argmax (shuffles), one barrier, every thread combines the 8
    // wave winners itself (double-buffered slots), the owner retires the winner in registers
    __shared__ float s_val[2][8];
    __shared__ int s_idx[2][8];
    __shared__ int s_sel[64];
    const int lane = tid & 31;
    const int wave = tid >> 5;
    for (int j = 0; j < topk; ++j)
    {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int e = tid + 256 * i;
            if (e < E && (ch[i] > bv || (ch[i] == bv && e < bi))) { bv = ch[i]; bi = e; }
        }
        #pragma unroll
        for (int m = 16; m >= 1; m >>= 1)
        {
            const float ov = __shfl_xor(bv, m);
            const int oi = __shfl_xor(bi, m);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const int pb = j & 1;
        if (lane == 0) { s_val[pb][wave] = bv; s_idx[pb][wave] = bi; }
        __syncthreads();
        float v = s_val[pb][0]; int w0 = s_idx[pb][0];
        #pragma unroll
        for (int w = 1; w < 8; ++w)
            if (s_val[pb][w] > v || (s_val[pb][w] == v && s_idx[pb][w] < w0)) { v = s_val[pb][w]; w0 = s_idx[pb][w]; }
        if (tid == 0) s_sel[j] = w0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) if (tid + 256 * i == w0) ch[i] = -INFINITY;
    }
    __syncthreads();
    if (tid == 0)
    {
        float sum = 0.0f;
        for (int j = 0; j < topk; ++j) sum += score[s_sel[j]];
        const float f = scale / (sum + 1e-20f);
        for (int j = 0; j < topk; ++j)
        {
            sel[j] = s_sel[j];
            wts[j] = __float2half_rn(score[s_sel[j]] * f);
        }
    }
}

// Multi-row router (EXL3_DEC_ROUTER_ROWS=2): NR verify rows in ONE launch. Every block streams its
// 64-row slice of G once and keeps NR accumulator sets; the last block to arrive reduces and selects
// each row in turn. Per row, the fma order, the cross-block sum order and the top-k are exactly
// router_kernel<false>'s, so sel/wts are bitwise equal to NR single-row launches.
template <int NR>
__global__ __launch_bounds__(256)
void router_rows_kernel(const half* __restrict__ x, const half* __restrict__ G, const float* bias,
                        int64_t* sel, half* wts, float* scratch, int* counters, int H, int E, int topk,
                        float scale)
{
    const int tid = threadIdx.x;
    const int kb = blockIdx.x;
    const int nkb = gridDim.x;
    __shared__ float xs[NR][ROUTER_KCHUNK];
    __shared__ float red[2048];
    const int k0 = kb * ROUTER_KCHUNK;
    const int gpr = E / 8;
    const int R = 256 / gpr;
    const int kr = tid / gpr;
    const int e8 = (tid % gpr) * 8;
    uint4 gv[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u)
    {
        const int k = kr + u * R;
        if (kr < R && k < ROUTER_KCHUNK) gv[u] = *reinterpret_cast<const uint4*>(G + (size_t) (k0 + k) * E + e8);
    }
    for (int i = tid; i < NR * ROUTER_KCHUNK; i += 256)
    {
        const int r = i / ROUTER_KCHUNK, k = i % ROUTER_KCHUNK;
        xs[r][k] = __half2float(x[(size_t) r * H + k0 + k]);
    }
    __syncthreads();
    float acc[NR][8];
    #pragma unroll
    for (int r = 0; r < NR; ++r)
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][i] = 0.0f;
    if (kr < R)
    {
        #pragma unroll
        for (int u = 0; u < 8; ++u)
        {
            const int k = kr + u * R;
            if (k < ROUTER_KCHUNK)
            {
                const half* gh = reinterpret_cast<const half*>(&gv[u]);
                #pragma unroll
                for (int r = 0; r < NR; ++r)
                {
                    const float xv = xs[r][k];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[r][i] = fmaf(xv, __half2float(gh[i]), acc[r][i]);
                }
            }
        }
        for (int k = kr + 8 * R; k < ROUTER_KCHUNK; k += R)
        {
            const uint4 g1 = *reinterpret_cast<const uint4*>(G + (size_t) (k0 + k) * E + e8);
            const half* gh = reinterpret_cast<const half*>(&g1);
            #pragma unroll
            for (int r = 0; r < NR; ++r)
            {
                const float xv = xs[r][k];
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][i] = fmaf(xv, __half2float(gh[i]), acc[r][i]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < NR; ++r)
    {
        if (kr < R)
        {
            #pragma unroll
            for (int i = 0; i < 8; ++i) red[kr * E + e8 + i] = acc[r][i];
        }
        __syncthreads();
        for (int e = tid; e < E; e += 256)
        {
            float t = 0.0f;
            for (int q = 0; q < R; ++q) t += red[q * E + e];
            scratch[((size_t) r * nkb + kb) * E + e] = t;
        }
        __syncthreads();
    }
    __threadfence();
    __shared__ int s_last;
    if (tid == 0)
    {
        const int old = atomicAdd(counters + CTR_ROUTER, 1);
        s_last = (old == nkb - 1);
        if (s_last) counters[CTR_ROUTER] = 0;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();

    // Tail: every thread reduces the cross-block partials of all rows (same b order per element),
    // then 64-thread groups (2 waves) run the rows' top-k concurrently, group g = row g. The pick
    // (max choice, lowest index on ties) and the weight sum order equal router_kernel's.
    __shared__ float score[4][320];
    __shared__ float chs[4][320];
    for (int i = tid; i < NR * E; i += 256)
    {
        const int r = i / E, e = i - r * E;
        const float* sc_r = scratch + (size_t) r * nkb * E;
        float l = 0.0f;
        for (int b = 0; b < nkb; ++b) l += ld_coherent(sc_r + (size_t) b * E + e);
        l = __half2float(__float2half_rn(l));
        const float sc = 1.0f / (1.0f + __expf(-l));
        score[r][e] = sc;
        chs[r][e] = sc + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();
    __shared__ float s_val[2][8];
    __shared__ int s_idx[2][8];
    __shared__ int s_sel[4][64];
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int g = tid >> 6;
    const int gl = tid & 63;
    const bool act = g < NR;
    float ch[5];
    #pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        const int e = gl + 64 * i;
        ch[i] = (act && e < E) ? chs[g][e] : -INFINITY;
    }
    for (int j = 0; j < topk; ++j)
    {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        #pragma unroll
        for (int i = 0; i < 5; ++i)
        {
            const int e = gl + 64 * i;
            if (e < E && (ch[i] > bv || (ch[i] == bv && e < bi))) { bv = ch[i]; bi = e; }
        }
        #pragma unroll
        for (int m = 16; m >= 1; m >>= 1)
        {
            const float ov = __shfl_xor(bv, m);
            const int oi = __shfl_xor(bi, m);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const int pb = j & 1;
        if (lane == 0) { s_val[pb][wave] = bv; s_idx[pb][wave] = bi; }
        __syncthreads();
        float v = s_val[pb][2 * g]; int w0 = s_idx[pb][2 * g];
        {
            const float v1 = s_val[pb][2 * g + 1]; const int w1 = s_idx[pb][2 * g + 1];
            if (v1 > v || (v1 == v && w1 < w0)) { v = v1; w0 = w1; }
        }
        if (act && gl == 0) s_sel[g][j] = w0;
        #pragma unroll
        for (int i = 0; i < 5; ++i) if (gl + 64 * i == w0) ch[i] = -INFINITY;
    }
    __syncthreads();
    if (act && gl == 0)
    {
        float sum = 0.0f;
        for (int j = 0; j < topk; ++j) sum += score[g][s_sel[g][j]];
        const float f = scale / (sum + 1e-20f);
        for (int j = 0; j < topk; ++j)
        {
            sel[g * topk + j] = s_sel[g][j];
            wts[g * topk + j] = __float2half_rn(score[g][s_sel[g][j]] * f);
        }
    }
}


// ------------------------------------------------------------------------------------------------
// RMSNorm for decode-sized inputs (one block per row), same semantics as ext_fallbacks.rms_norm
// (w_groups == 1, no span_heads) and rms_norm_res_in. Replaces a 5-6 kernel torch chain.

template <typename T> __device__ __forceinline__ float ld_f(const T* p, int i);
template <> __device__ __forceinline__ float ld_f<half>(const half* p, int i) { return fminf(fmaxf(__half2float(p[i]), -65504.0f), 65504.0f); }
template <> __device__ __forceinline__ float ld_f<float>(const float* p, int i) { return p[i]; }
template <typename T> __device__ __forceinline__ void st_f(T* p, int i, float v);
template <> __device__ __forceinline__ void st_f<half>(half* p, int i, float v) { p[i] = __float2half_rn(v); }
template <> __device__ __forceinline__ void st_f<float>(float* p, int i, float v) { p[i] = v; }

// Value as re-read after a store of type T (no memory round trip)
template <typename T> __device__ __forceinline__ float rt_f(float v);
template <> __device__ __forceinline__ float rt_f<half>(float v) { return fminf(fmaxf(__half2float(__float2half_rn(v)), -65504.0f), 65504.0f); }
template <> __device__ __forceinline__ float rt_f<float>(float v) { return v; }

__device__ __forceinline__ float ld_w(const void* w, int wtype, int i)
{
    if (wtype == 1) return __half2float(reinterpret_cast<const half*>(w)[i]);
    if (wtype == 2) return __uint_as_float(((uint32_t) reinterpret_cast<const uint16_t*>(w)[i]) << 16);
    return 1.0f;
}

__device__ __forceinline__ float block_sum(float v, float* red)
{
    v = wave_sum(v);
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[wave] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < (int) (blockDim.x >> 5); ++w) t += red[w];
    return t;
}

constexpr int NORM_MAX_PER_THREAD = 32;

// mode 0: y = norm(x) (* w); 1: y += norm(x); 2 (res_in): r += x, y = norm(r)
template <typename TX, typename TY, typename TR>
__global__ __launch_bounds__(256)
void rms_norm_kernel(const TX* __restrict__ x, const void* w, int wtype, TY* y, TR* r, int dim,
                     float eps, float constant_bias, float constant_scale, int mode)
{
    __shared__ float red[8];
    const size_t row = blockIdx.x;
    x += row * dim;
    y += row * dim;
    if (r) r += row * dim;
    float v[NORM_MAX_PER_THREAD];
    float wv[NORM_MAX_PER_THREAD];
    float ss = 0.0f;
    // weights and (mode 1) the old output are fetched up front, in the same memory round trip as x
    #pragma unroll
    for (int j = 0; j < NORM_MAX_PER_THREAD; ++j)
    {
        const int i = threadIdx.x + j * 256;
        wv[j] = 0.0f;
        if (i < dim)
        {
            wv[j] = wtype ? ld_w(w, wtype, i) + constant_bias : 1.0f;
        }
    }
    float yo[NORM_MAX_PER_THREAD];
    #pragma unroll
    for (int j = 0; j < NORM_MAX_PER_THREAD; ++j)
    {
        const int i = threadIdx.x + j * 256;
        yo[j] = (mode == 1 && i < dim) ? ld_f<TY>(y, i) : 0.0f;
    }
    #pragma unroll
    for (int j = 0; j < NORM_MAX_PER_THREAD; ++j)
    {
        const int i = threadIdx.x + j * 256;
        if (i < dim)
        {
            float t = ld_f<TX>(x, i);
            if (mode == 2)
            {
                t += ld_f<TR>(r, i);
                st_f<TR>(r, i, t);
                t = rt_f<TR>(t);      // the fallback re-reads r after its (possibly fp16) store
            }
            v[j] = t;
            ss += t * t;
        }
    }
    const float scale = rsqrtf(block_sum(ss, red) / (float) dim + eps) * constant_scale;
    #pragma unroll
    for (int j = 0; j < NORM_MAX_PER_THREAD; ++j)
    {
        const int i = threadIdx.x + j * 256;
        if (i < dim)
        {
            float o = v[j] * scale;
            if (wtype) o *= wv[j];
            if (mode == 1) o += yo[j];
            st_f<TY>(y, i, o);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Host side

// k-tiles per wave (ktw). Measured on gfx1151 at MiMo shapes (tools/decode/sweep.py): the best
// ktw is the largest one that still leaves >= min_blocks (32) blocks (o_proj 16, q 16, mlp 32, head
// 32); when no ktw reaches that (small k/v and expert GEMVs), 8 wins over both 4 and 16.
// Blocks per strip along k (ragged split, see k_split): forced (env) wins, else the ktw pick.
inline int kbs_of(int kt, int ktw, int forced)
{
    const int G = kt / 8;
    const int lo = (G + KTW_MAX - 1) / KTW_MAX;
    if (forced > 0) return std::min(std::max(forced, lo), G);
    return kt / (WPB * ktw);
}

inline bool ktw_valid(int kt, int ring, int ktw)
{
    return ktw >= ring && ktw % ring == 0 && ktw <= KTW_MAX && kt % (WPB * ktw) == 0;
}

inline int pick_ktw(int kt, int strips_total, int ring, int min_blocks, int forced)
{
    if (forced > 0)
    {
        TORCH_CHECK(ktw_valid(kt, ring, forced), "exl3_dec: bad ktw ", forced);
        return forced;
    }
    for (int ktw = KTW_MAX; ktw >= 4; ktw >>= 1)
        if (ktw_valid(kt, ring, ktw) && strips_total * (kt / (WPB * ktw)) >= min_blocks) return ktw;
    for (int ktw : {8, 4, 16, 2, 32})
        if (ktw_valid(kt, ring, ktw)) return ktw;
    TORCH_CHECK(false, "exl3_dec: k extent ", kt * 16, " not supported");
    return 0;
}

inline int ring_of_kb2(int kb2)
{
    const int words = 4 * kb2;   // u32 words per tile
    return words <= 32 ? EXL3_DEC_RING_SMALL : EXL3_DEC_RING_BIG;
}

static int env_int(const char* name, int dflt)
{
    const char* e = getenv(name);
    return e ? atoi(e) : dflt;
}

inline int kb2_of(double K)
{
    const int kb2 = (int) (K * 2.0 + 0.5);
    TORCH_CHECK(kb2 >= 4 && kb2 <= 12 && (double) kb2 == K * 2.0, "exl3_dec: unsupported K ", K);
    return kb2;
}

#define EXL3_DEC_DISPATCH_KB2(kb2, ...)                                             \
    switch (kb2)                                                                     \
    {                                                                                \
        case 4:  { constexpr int KB2 = 4;  __VA_ARGS__; } break;                     \
        case 5:  { constexpr int KB2 = 5;  __VA_ARGS__; } break;                     \
        case 6:  { constexpr int KB2 = 6;  __VA_ARGS__; } break;                     \
        case 8:  { constexpr int KB2 = 8;  __VA_ARGS__; } break;                     \
        case 10: { constexpr int KB2 = 10; __VA_ARGS__; } break;                     \
        case 12: { constexpr int KB2 = 12; __VA_ARGS__; } break;                     \
        default: TORCH_CHECK(false, "exl3_dec: unsupported K*2 ", kb2);              \
    }

// Codebook dispatch (1 = mcg, 2 = mul1, 3 = mul1 folded), same shape as EXL3_DEC_DISPATCH_KB2.
#define EXL3_DEC_DISPATCH_CB(cb, ...)                                               \
    switch (cb)                                                                      \
    {                                                                                \
        case 1:  { constexpr int CB = 1; __VA_ARGS__; } break;                       \
        case 2:  { constexpr int CB = 2; __VA_ARGS__; } break;                       \
        case 3:  { constexpr int CB = 3; __VA_ARGS__; } break;                       \
        default: TORCH_CHECK(false, "exl3_dec: unsupported codebook ", cb);          \
    }

// Codebook + K dispatch for the dense GEMV launch. Same reason as the launch_moe_* helpers
// below: the comma in gemv_kernel<KB2, CB> would split a nested macro's arguments.
template <int CB>
inline void launch_gemv_cb(int kb2, int blocks, const half* xp, const Jobs& jobs, float* sp,
                           int* cp, cudaStream_t stream)
{
    switch (kb2)
    {
        case 4:  gemv_kernel<4, CB><<<blocks, THREADS, 0, stream>>>(xp, jobs, sp, cp); break;
        case 5:  gemv_kernel<5, CB><<<blocks, THREADS, 0, stream>>>(xp, jobs, sp, cp); break;
        case 6:  gemv_kernel<6, CB><<<blocks, THREADS, 0, stream>>>(xp, jobs, sp, cp); break;
        case 8:  gemv_kernel<8, CB><<<blocks, THREADS, 0, stream>>>(xp, jobs, sp, cp); break;
        case 10: gemv_kernel<10, CB><<<blocks, THREADS, 0, stream>>>(xp, jobs, sp, cp); break;
        case 12: gemv_kernel<12, CB><<<blocks, THREADS, 0, stream>>>(xp, jobs, sp, cp); break;
        default: TORCH_CHECK(false, "exl3_dec_gemv: unsupported K*2 ", kb2);
    }
}

inline void launch_gemv(int kb2, int blocks, const half* xp, const Jobs& jobs, float* sp,
                        int* cp, cudaStream_t stream, bool mcg)
{
    if (mcg) launch_gemv_cb<1>(kb2, blocks, xp, jobs, sp, cp, stream);
    else     launch_gemv_cb<2>(kb2, blocks, xp, jobs, sp, cp, stream);
}

// Codebook + K dispatch for the routed-MoE launches. Not EXL3_DEC_DISPATCH_CB(EXL3_DEC_DISPATCH_
// KB2(...)): the comma in the kernel's own <KB2, CB> template argument list would split the outer
// macro's arguments, so each launch gets a small template helper instead.
template <int CB>
inline void launch_moe_gu(int kb2, const half* xp, const MoeArgs& a, int kbs_h, cudaStream_t stream)
{
    dim3 grid((a.I / STRIP) * kbs_h, a.topk, 2);
    switch (kb2)
    {
        case 4:  moe_gu_kernel<4, CB><<<grid, THREADS, 0, stream>>>(xp, a, kbs_h); break;
        case 5:  moe_gu_kernel<5, CB><<<grid, THREADS, 0, stream>>>(xp, a, kbs_h); break;
        case 6:  moe_gu_kernel<6, CB><<<grid, THREADS, 0, stream>>>(xp, a, kbs_h); break;
        case 8:  moe_gu_kernel<8, CB><<<grid, THREADS, 0, stream>>>(xp, a, kbs_h); break;
        case 10: moe_gu_kernel<10, CB><<<grid, THREADS, 0, stream>>>(xp, a, kbs_h); break;
        case 12: moe_gu_kernel<12, CB><<<grid, THREADS, 0, stream>>>(xp, a, kbs_h); break;
        default: TORCH_CHECK(false, "exl3_dec_moe: unsupported K*2 ", kb2);
    }
}

template <int CB>
inline void launch_moe_down(int kb2, const MoeArgs& a, int kbs_i, cudaStream_t stream)
{
    dim3 grid((a.H / STRIP) * kbs_i, a.topk, 1);
    switch (kb2)
    {
        case 4:  moe_down_kernel<4, CB><<<grid, THREADS, 0, stream>>>(a, kbs_i); break;
        case 5:  moe_down_kernel<5, CB><<<grid, THREADS, 0, stream>>>(a, kbs_i); break;
        case 6:  moe_down_kernel<6, CB><<<grid, THREADS, 0, stream>>>(a, kbs_i); break;
        case 8:  moe_down_kernel<8, CB><<<grid, THREADS, 0, stream>>>(a, kbs_i); break;
        case 10: moe_down_kernel<10, CB><<<grid, THREADS, 0, stream>>>(a, kbs_i); break;
        case 12: moe_down_kernel<12, CB><<<grid, THREADS, 0, stream>>>(a, kbs_i); break;
        default: TORCH_CHECK(false, "exl3_dec_moe: unsupported K*2 ", kb2);
    }
}

// Folded shared expert (decB8): grid.y = topk + 1, slot 0 = shared. Instantiated for the routed
// K2 / K2.5 experts with a K4 shared expert (GLM-5.3 td205), the pairs that exist today; the
// host falls back to the unfolded path (Python eligibility) for anything else.
template <int CB, int CBS, int RS>
inline void launch_moe_gu_sh(int kb2, int kb2s, const half* xp, const MoeArgs& a, int kbs_h, cudaStream_t stream)
{
    dim3 grid((a.I / STRIP) * kbs_h, a.topk + 1, 2);
    TORCH_CHECK(kb2s == 8, "exl3_dec_moe_shared: unsupported shared K*2 ", kb2s);
    switch (kb2)
    {
        case 4:  moe_gu_kernel<4, CB, 8, CBS, RS><<<grid, THREADS, 0, stream>>>(xp, a, kbs_h); break;
        case 5:  moe_gu_kernel<5, CB, 8, CBS, RS><<<grid, THREADS, 0, stream>>>(xp, a, kbs_h); break;
        default: TORCH_CHECK(false, "exl3_dec_moe_shared: unsupported K*2 ", kb2);
    }
}

template <int CB, int CBS, int RS>
inline void launch_moe_down_sh(int kb2, int kb2s, const MoeArgs& a, int kbs_i, cudaStream_t stream)
{
    dim3 grid((a.H / STRIP) * kbs_i, a.topk + 1, 1);
    TORCH_CHECK(kb2s == 8, "exl3_dec_moe_shared: unsupported shared K*2 ", kb2s);
    switch (kb2)
    {
        case 4:  moe_down_kernel<4, CB, 8, CBS, RS><<<grid, THREADS, 0, stream>>>(a, kbs_i); break;
        case 5:  moe_down_kernel<5, CB, 8, CBS, RS><<<grid, THREADS, 0, stream>>>(a, kbs_i); break;
        default: TORCH_CHECK(false, "exl3_dec_moe_shared: unsupported K*2 ", kb2);
    }
}

template <int CB, int CBS>
inline void launch_moe_sh(int kb2_gu, int kb2_d, int kb2s_gu, int kb2s_d, const half* xp, const MoeArgs& a,
                          int kbs_h, int kbs_i, int rs, cudaStream_t stream)
{
    if (rs == 1)
    {
        launch_moe_gu_sh<CB, CBS, 1>(kb2_gu, kb2s_gu, xp, a, kbs_h, stream);
        launch_moe_down_sh<CB, CBS, 1>(kb2_d, kb2s_d, a, kbs_i, stream);
    }
    else
    {
        launch_moe_gu_sh<CB, CBS, 0>(kb2_gu, kb2s_gu, xp, a, kbs_h, stream);
        launch_moe_down_sh<CB, CBS, 0>(kb2_d, kb2s_d, a, kbs_i, stream);
    }
}

template <int CB, int RM>
inline void launch_moe_gu_r(int kb2, const MoeArgsR& a, int kbs_h, cudaStream_t stream)
{
    dim3 grid((a.I / STRIP) * kbs_h, a.U, 2);
    switch (kb2)
    {
        case 4:  moe_gu_kernel_r<4, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_h); break;
        case 5:  moe_gu_kernel_r<5, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_h); break;
        case 6:  moe_gu_kernel_r<6, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_h); break;
        case 8:  moe_gu_kernel_r<8, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_h); break;
        case 10: moe_gu_kernel_r<10, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_h); break;
        case 12: moe_gu_kernel_r<12, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_h); break;
        default: TORCH_CHECK(false, "exl3_dec_moe_union: unsupported K*2 ", kb2);
    }
}

template <int CB, int RM>
inline void launch_moe_down_r(int kb2, const MoeArgsR& a, int kbs_i, int ctr_b_off, cudaStream_t stream)
{
    dim3 grid((a.H / STRIP) * kbs_i, a.U, 1);
    switch (kb2)
    {
        case 4:  moe_down_kernel_r<4, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_i, ctr_b_off); break;
        case 5:  moe_down_kernel_r<5, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_i, ctr_b_off); break;
        case 6:  moe_down_kernel_r<6, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_i, ctr_b_off); break;
        case 8:  moe_down_kernel_r<8, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_i, ctr_b_off); break;
        case 10: moe_down_kernel_r<10, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_i, ctr_b_off); break;
        case 12: moe_down_kernel_r<12, CB, RM><<<grid, THREADS, 0, stream>>>(a, kbs_i, ctr_b_off); break;
        default: TORCH_CHECK(false, "exl3_dec_moe_union: unsupported K*2 ", kb2);
    }
}

#define EXL3_DEC_DISPATCH_RM(rows, ...)                                              \
    if ((rows) <= 1)      { constexpr int RM = 1; __VA_ARGS__; }                     \
    else if ((rows) <= 2) { constexpr int RM = 2; __VA_ARGS__; }                     \
    else if ((rows) <= 4) { constexpr int RM = 4; __VA_ARGS__; }                     \
    else                  { constexpr int RM = 8; __VA_ARGS__; }

} // namespace exl3dec

using namespace exl3dec;

static void dec_gemv_impl
(
    const at::Tensor& x,
    const std::vector<at::Tensor>& trellis,
    const std::vector<at::Tensor>& suh,
    const std::vector<at::Tensor>& svh,
    const std::vector<at::Tensor>& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    const std::vector<double>& K,
    int64_t Kdim_strided,
    int64_t x_gstride,
    bool mcg
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int n = trellis.size();
    TORCH_CHECK(n >= 1 && n <= 4, "exl3_dec_gemv: 1..4 matrices");
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous(), "exl3_dec_gemv: x must be contiguous fp16");
    const int Kdim = x_gstride ? (int) Kdim_strided : (int) x.size(-1);
    TORCH_CHECK(x_gstride ? (x_gstride >= 128 && x.numel() >= (int64_t) (Kdim / 128 - 1) * x_gstride + 128)
                          : x.numel() == x.size(-1), "exl3_dec_gemv: x must be fp16 [1, K] (or cover the strided groups)");
    TORCH_CHECK(Kdim % 512 == 0, "exl3_dec_gemv: K must be a multiple of 512");
    const int kb2 = kb2_of(K[0]);
    int strips_total = 0;
    for (int i = 0; i < n; ++i) strips_total += (out[i].size(-1) + STRIP - 1) / STRIP;
    const int min_blocks = env_int("EXL3_DEC_MIN_BLOCKS", 32);
    const int forced = env_int("EXL3_DEC_KTW", 0);
    const int ktw = pick_ktw(Kdim / 16, strips_total, ring_of_kb2(kb2), min_blocks, forced);
    Jobs jobs;
    jobs.count = n;
    jobs.K = Kdim;
    jobs.kbs = kbs_of(Kdim / 16, ktw, env_int("EXL3_DEC_KBS", 0));
    jobs.x_gstride = (int) x_gstride;
#if QWG_ABLATE
    jobs.abl = env_int("EXL3_QWG_DABL", 0);
#endif
    int blocks = 0, part = 0, ctr = CTR_GEMV;
    for (int i = 0; i < n; ++i)
    {
        TORCH_CHECK(kb2_of(K[i]) == kb2, "exl3_dec_gemv: all matrices must share K");
        const int N = out[i].size(-1);
        TORCH_CHECK(N % 128 == 0 && out[i].numel() == N && out[i].is_contiguous(), "exl3_dec_gemv: out must be [1, N], N % 128 == 0");
        TORCH_CHECK(trellis[i].size(0) == Kdim / 16 && trellis[i].size(1) == N / 16 && trellis[i].size(2) == kb2 * 8,
                    "exl3_dec_gemv: trellis shape mismatch");
        Job& j = jobs.j[i];
        j.trellis = reinterpret_cast<const u32x4*>(trellis[i].data_ptr());
        j.suh = reinterpret_cast<const half*>(suh[i].data_ptr());
        j.svh = reinterpret_cast<const half*>(svh[i].data_ptr());
        j.out = out[i].data_ptr();
        j.N = N;
        j.out_fp32 = out[i].dtype() == at::kFloat;
        j.strips = (N + STRIP - 1) / STRIP;
        j.block_base = blocks;
        j.part_base = part;
        j.ctr_base = ctr;
        blocks += j.strips * jobs.kbs;
        part += jobs.kbs * N;
        ctr += j.strips;
    }
    TORCH_CHECK(scratch.numel() >= part, "exl3_dec_gemv: scratch too small");
    TORCH_CHECK(counters.numel() >= ctr, "exl3_dec_gemv: counters too small");
    const half* xp = reinterpret_cast<const half*>(x.data_ptr());
    float* sp = scratch.data_ptr<float>();
    int* cp = counters.data_ptr<int>();
    launch_gemv(kb2, blocks, xp, jobs, sp, cp, stream, mcg);
    cuda_check(cudaPeekAtLastError());
}

void exl3_dec_gemv_multi
(
    const at::Tensor& x,
    const std::vector<at::Tensor>& trellis,
    const std::vector<at::Tensor>& suh,
    const std::vector<at::Tensor>& svh,
    const std::vector<at::Tensor>& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    const std::vector<double>& K,
    bool mcg
)
{
    dec_gemv_impl(x, trellis, suh, svh, out, scratch, counters, K, 0, 0, mcg);
}

void exl3_dec_gemv
(
    const at::Tensor& x,
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    at::Tensor& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    double K,
    bool mcg
)
{
    dec_gemv_impl(x, {trellis}, {suh}, {svh}, {out}, scratch, counters, {K}, 0, 0, mcg);
}

void exl3_dec_gemv_strided
(
    const at::Tensor& x,
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    at::Tensor& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    double K,
    int64_t in_features,
    int64_t x_gstride,
    bool mcg
)
{
    dec_gemv_impl(x, {trellis}, {suh}, {svh}, {out}, scratch, counters, {K}, in_features, x_gstride, mcg);
}

// ------------------------------------------------------------------------------------------------
// R-row dense GEMV: each weight tile is decoded once (identical RING pipeline to
// batch-1's gemv_kernel) and applied to up to `rpb` rows via lane_gemv_r, in the exact per-row
// arithmetic order (tile_dot/pair_dot/block_reduce/arrive) a batch-1 exl3_dec_gemv call would
// use for that row -- so row j's output is bit-identical to a fresh batch-1 call, PROVIDED ktw
// and kbs are the ones batch-1 would pick for the same (K, N, matrix-group): pick_ktw/kbs_of are
// called with the identical arguments dec_gemv_impl uses, not a separate R-row formula (same
// lesson as exl3_dec_moe_union's pick_ktw alignment).
//
// LDS holds `rpb` rows' rotated input at once, `rpb * ktw <= MOE_R_MAX * MOE_R_KTW_MAX` (the same
// 32KB budget the MoE R-row kernels use) -- for shapes where batch-1's own ktw choice already
// fits MOE_R_KTW_MAX (most dense projections here), rpb = R, one block covers every row. For a
// shape wide enough that pick_ktw picks a larger ktw (e.g. lm_head), rpb shrinks below R and the
// host launches grid.y > 1 row-chunks instead -- ktw/kbs stay exactly what batch-1 would pick
// either way, only the row-parallelism per block changes, so bit-exactness doesn't depend on rpb.
struct JobR
{
    const u32x4* trellis;
    const half* suh;
    const half* svh;
    void* out;          // [R, N], row-major, row stride N
    int N;
    int out_fp32;
    int strips;
    int block_base;
    int part_base;      // scratch float offset for row 0 (row stride: kbs * N)
    int ctr_base;        // counters int offset for row 0 (row stride: strips)
};

struct JobsR
{
    JobR j[4];
    int count;
    int K;
    int kbs;
    int R;
    int rpb;             // rows per block (chunk size); grid.y = ceil(R / rpb)
    int swap;            // 1: grid = (row chunks, blocks), so the chunks of one weight block run side by side and share its reads
    int x_row_pitch;      // half-elements between consecutive rows of x (K for contiguous x)
};

constexpr int GEMV_R_ROW_BUDGET = MOE_R_MAX * MOE_R_KTW_MAX;   // 128: max rpb * ktw
// Tile ring depth for gemv_kernel_r. 0 (default) = batch-1's ring at RM 1, else 1: the 2-deep ring of
// the > 4 bpw formats (80 VGPRs of tile words) plus RM x 16 accumulators spilled 125 VGPRs at
// KB2 10 RM 2 (GLM lm_head, R 2 ran 5.0 ms vs 3.6 ms for two batch-1 launches). Result-neutral.
#if !defined(EXL3_GEMV_R_RING)
#define EXL3_GEMV_R_RING 0
#endif
template <int KB2, int RM> constexpr int gemv_r_ring() { return EXL3_GEMV_R_RING > 0 ? EXL3_GEMV_R_RING : RM == 1 ? ring_of<KB2>() : 1; }

template <int KB2, int CB, int RM, int MODE = 0>
__global__ __launch_bounds__(THREADS)
void gemv_kernel_r(const half* __restrict__ x, JobsR jobs, float* scratch, int* counters)
{
    constexpr int VEC = Fmt<KB2>::VEC;
    __shared__ __attribute__((aligned(16))) float smem[MOE_R_MAX * MOE_R_KTW_MAX * 128 / 2];
    half* xs = reinterpret_cast<half*>(smem);
    float* red = smem;
    __shared__ float s_xsum[MOE_R_MAX][KTW_MAX];
    __shared__ float z[STRIP];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    const int bid = jobs.swap ? (int) blockIdx.y : (int) blockIdx.x;
    const int chunk = jobs.swap ? (int) blockIdx.x : (int) blockIdx.y;
    int ji = 0;
    #pragma unroll
    for (int i = 1; i < 4; ++i) if (i < jobs.count && bid >= jobs.j[i].block_base) ji = i;
    const JobR& job = jobs.j[ji];
    const int local = bid - job.block_base;
    const int strip = local / jobs.kbs;
    const int kb = local % jobs.kbs;
    const int NT = job.N / 16;

    const int row0 = chunk * jobs.rpb;
    const int nrows = min(jobs.rpb, jobs.R - row0);

    int k0, ktw;
    k_split(jobs.K / 128, kb, jobs.kbs, k0, ktw);

    for (int j = 0; j < nrows; ++j)
    {
        const int row = row0 + j;
        prologue_x(x + (size_t) row * jobs.x_row_pitch, job.suh, k0, ktw * 128,
                   xs + j * (ktw * 128), s_xsum[j], wave, lane);
    }
    __syncthreads();

    float acc[RM][16];
    #pragma unroll
    for (int j = 0; j < RM; ++j)
        #pragma unroll
        for (int c = 0; c < 16; ++c) acc[j][c] = 0.0f;

    const int nt = strip * 32 + lane;
    if (nt < NT)
    {
        const int kt0 = k0 / 16 + wave * ktw;
        const u32x4* tiles = job.trellis + ((size_t) kt0 * NT + nt) * VEC;
        const half* xs_rows[RM];
        #pragma unroll
        for (int j = 0; j < RM; ++j) xs_rows[j] = xs + (MODE ? min(j, nrows - 1) : j) * (ktw * 128) + wave * ktw * 16;
        if constexpr (MODE == 1 && RM > 1)
            lane_gemv_r1<KB2, CB, RM, gemv_r_ring<KB2, RM>()>(tiles, (size_t) NT * VEC, xs_rows, ktw, acc);
        else
            lane_gemv_r<KB2, CB, RM, gemv_r_ring<KB2, RM>()>(tiles, (size_t) NT * VEC, xs_rows, nrows, ktw, acc);
    }

    const int col0 = strip * STRIP;
    const int ncols = min(STRIP, job.N - col0);
    #pragma unroll
    for (int j = 0; j < RM; ++j)
    {
        if (j >= nrows) break;
        const int row = row0 + j;
        float xsum = 0.0f;
        for (int c = 0; c < ktw; ++c) xsum += s_xsum[j][c];
        block_reduce<CB>(acc[j], red, z, xsum, wave, lane, tid);

        bool last = true;
        if (jobs.kbs > 1)
        {
            float* part = scratch + job.part_base + (size_t) row * jobs.kbs * job.N + (size_t) kb * job.N + col0;
            last = arrive(z, part, ncols, counters + job.ctr_base + row * job.strips + strip, jobs.kbs, tid);
            __syncthreads();
            if (!last) continue;
            for (int col = tid; col < ncols; col += THREADS)
            {
                float s = 0.0f;
                for (int b = 0; b < jobs.kbs; ++b)
                    s += ld_coherent(scratch + job.part_base + (size_t) row * jobs.kbs * job.N + (size_t) b * job.N + col0 + col);
                z[col] = s;
            }
            __syncthreads();
        }

        if (wave < 4 && wave * 128 < ncols)
        {
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = z[wave * 128 + lane * 4 + i];
            const int col = col0 + wave * 128 + lane * 4;
#if !defined(EXL3_DEC_DEBUG)
            fwht128(v, lane);
            const uint2 sr = *reinterpret_cast<const uint2*>(job.svh + col);
            const half* sh = reinterpret_cast<const half*>(&sr);
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= HAD_SCALE * __half2float(sh[i]);
#endif
            if (job.out_fp32)
                *reinterpret_cast<float4*>(reinterpret_cast<float*>(job.out) + (size_t) row * job.N + col) =
                    make_float4(v[0], v[1], v[2], v[3]);
            else
            {
                half o[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) o[i] = sat_half(v[i]);
                *reinterpret_cast<uint2*>(reinterpret_cast<half*>(job.out) + (size_t) row * job.N + col) =
                    *reinterpret_cast<uint2*>(o);
            }
        }
        if (jobs.kbs > 1) __syncthreads();
    }
}

static void dec_gemv_r_impl
(
    const at::Tensor& x,
    const std::vector<at::Tensor>& trellis,
    const std::vector<at::Tensor>& suh,
    const std::vector<at::Tensor>& svh,
    const std::vector<at::Tensor>& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    const std::vector<double>& K,
    bool mcg
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int n = trellis.size();
    TORCH_CHECK(n >= 1 && n <= 4, "exl3_dec_gemv_r: 1..4 matrices");
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous() && x.dim() == 2, "exl3_dec_gemv_r: x must be [R, K] fp16");
    const int R = (int) x.size(0);
    const int Kdim = (int) x.size(1);
    TORCH_CHECK(R >= 1 && R <= MOE_R_MAX, "exl3_dec_gemv_r: R must be 1..MOE_R_MAX");
    TORCH_CHECK(Kdim % 512 == 0, "exl3_dec_gemv_r: K must be a multiple of 512");
    const int kb2 = kb2_of(K[0]);
    int strips_total = 0;
    for (int i = 0; i < n; ++i) strips_total += ((int) out[i].size(-1) + STRIP - 1) / STRIP;
    // Identical to dec_gemv_impl's own ktw/kbs choice (same pick_ktw/kbs_of call, same args) --
    // this is what makes the block-reduce partition, and so the rounding, match batch-1 exactly.
    const int min_blocks = env_int("EXL3_DEC_MIN_BLOCKS", 32);
    const int forced = env_int("EXL3_DEC_KTW", 0);
    const int ktw = pick_ktw(Kdim / 16, strips_total, ring_of_kb2(kb2), min_blocks, forced);
    const int kbs = kbs_of(Kdim / 16, ktw, env_int("EXL3_DEC_KBS", 0));
    // EXL3_GEMV_R_RPB caps rows per block (grid.y = ceil(R / rpb) row chunks, each re-reading the
    // weights). Null result (glm-mtp step 4): RM 4 spills a little at KB2 >= 8 but a cap of 1 or 2
    // always lost (lm_head R 4: 1.81 ms default vs 3.53 ms cap 2; R 3 verify 80.4 vs 94.5 ms).
    int rpb = std::max(1, std::min(R, GEMV_R_ROW_BUDGET / ktw));
    const int rpb_cap = env_int("EXL3_GEMV_R_RPB", 0);
    if (rpb_cap > 0) rpb = std::min(rpb, rpb_cap);
    // (MiMo, R 5..8 verify, bitwise independent of rpb): the 8-row kernel variant wins at R 5-6
    // but loses at R 7-8 (131 vs 123 ms per verify forward, spills); from R 7 on run two 4-row chunks (grid.y = 2).
    else if (R >= 7 && rpb > 4 && env_int("EXL3_GEMV_R_RPB78", 4) > 0) rpb = env_int("EXL3_GEMV_R_RPB78", 4);

    // EXL3_GEMV_R_RM6=1 (needs EXL3_GEMV_R_DEC1=1): 5 or 6 rows run in ONE launch of the 6-row instantiation (one decode pass,
    // no VGPR spill at K*2 = 8 / 10) instead of a 4-row chunk plus a 1..2 row chunk. Same per-row arithmetic (rows are independent
    // in lane_gemv_r1 / pair_dot_r), rows 1..4 keep their own instantiations untouched.
    const bool rm6 = R >= 5 && R <= 6 && (kb2 == 8 || kb2 == 10) && env_int("EXL3_GEMV_R_RM6", 0) != 0 &&
                     env_int("EXL3_GEMV_R_DEC1", 0) == 1 && GEMV_R_ROW_BUDGET / ktw >= R;
    if (rm6) rpb = R;

    JobsR jobs;
    jobs.count = n;
    jobs.K = Kdim;
    jobs.kbs = kbs;
    jobs.R = R;
    jobs.rpb = rpb;
    jobs.swap = env_int("EXL3_GEMV_R_CHUNK_FAST", 0) != 0 && R > 4 && R > rpb;   // 5..8 rows only: the <= 4 row launches keep their grid
    jobs.x_row_pitch = Kdim;
    int blocks = 0, part = 0, ctr = 0;
    for (int i = 0; i < n; ++i)
    {
        TORCH_CHECK(kb2_of(K[i]) == kb2, "exl3_dec_gemv_r: all matrices must share K");
        const int N = (int) out[i].size(-1);
        TORCH_CHECK(N % 128 == 0 && out[i].size(0) == R && out[i].is_contiguous(),
                    "exl3_dec_gemv_r: out must be [R, N], N % 128 == 0");
        TORCH_CHECK(trellis[i].size(0) == Kdim / 16 && trellis[i].size(1) == N / 16 && trellis[i].size(2) == kb2 * 8,
                    "exl3_dec_gemv_r: trellis shape mismatch");
        JobR& j = jobs.j[i];
        j.trellis = reinterpret_cast<const u32x4*>(trellis[i].data_ptr());
        j.suh = reinterpret_cast<const half*>(suh[i].data_ptr());
        j.svh = reinterpret_cast<const half*>(svh[i].data_ptr());
        j.out = out[i].data_ptr();
        j.N = N;
        j.out_fp32 = out[i].dtype() == at::kFloat;
        j.strips = (N + STRIP - 1) / STRIP;
        j.block_base = blocks;
        j.part_base = part;
        j.ctr_base = ctr;
        blocks += j.strips * kbs;
        part += R * kbs * N;
        ctr += R * j.strips;
    }
    TORCH_CHECK(scratch.numel() >= part, "exl3_dec_gemv_r: scratch too small");
    TORCH_CHECK(counters.numel() >= ctr, "exl3_dec_gemv_r: counters too small");
    const half* xp = reinterpret_cast<const half*>(x.data_ptr());
    float* sp = scratch.data_ptr<float>();
    int* cp = counters.data_ptr<int>();
    const int nch = (R + rpb - 1) / rpb;
    TORCH_CHECK(!jobs.swap || blocks <= 65535, "exl3_dec_gemv_r: too many blocks for the swapped grid");
    const dim3 grid = jobs.swap ? dim3(nch, blocks) : dim3(blocks, nch);
    if (rm6)
    {
        EXL3_DEC_DISPATCH_CB(mcg ? 1 : 2,
        {
            if (kb2 == 8) { auto kfn = gemv_kernel_r<8, CB, 6, 1>; kfn<<<grid, THREADS, 0, stream>>>(xp, jobs, sp, cp); }
            else          { auto kfn = gemv_kernel_r<10, CB, 6, 1>; kfn<<<grid, THREADS, 0, stream>>>(xp, jobs, sp, cp); }
        });
        cuda_check(cudaPeekAtLastError());
        return;
    }
    EXL3_DEC_DISPATCH_CB(mcg ? 1 : 2, EXL3_DEC_DISPATCH_RM(rpb, EXL3_DEC_DISPATCH_KB2(kb2,
        {
            // EXL3_GEMV_R_DEC1=1 decodes each tile once for all rows (bit-identical)
            if (RM > 1 && env_int("EXL3_GEMV_R_DEC1", 0) == 1)
            { auto kfn = gemv_kernel_r<KB2, CB, RM, 1>; kfn<<<grid, THREADS, 0, stream>>>(xp, jobs, sp, cp); }
            else
            { auto kfn = gemv_kernel_r<KB2, CB, RM>; kfn<<<grid, THREADS, 0, stream>>>(xp, jobs, sp, cp); }
        })));
    cuda_check(cudaPeekAtLastError());
}

void exl3_dec_gemv_r_multi
(
    const at::Tensor& x,
    const std::vector<at::Tensor>& trellis,
    const std::vector<at::Tensor>& suh,
    const std::vector<at::Tensor>& svh,
    const std::vector<at::Tensor>& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    const std::vector<double>& K,
    bool mcg
)
{
    dec_gemv_r_impl(x, trellis, suh, svh, out, scratch, counters, K, mcg);
}

void exl3_dec_gemv_r
(
    const at::Tensor& x,
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    at::Tensor& out,
    at::Tensor& scratch,
    at::Tensor& counters,
    double K,
    bool mcg
)
{
    dec_gemv_r_impl(x, {trellis}, {suh}, {svh}, {out}, scratch, counters, {K}, mcg);
}

void exl3_dec_moe
(
    const at::Tensor& x,
    at::Tensor& out,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    at::Tensor& act,
    at::Tensor& scratch,
    at::Tensor& counters,
    int64_t intermediate,
    double K_gu,
    double K_down,
    bool accumulate,
    double act_limit,
    bool mcg
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous() && x.numel() == x.size(-1), "exl3_dec_moe: x must be fp16 [1, H]");
    TORCH_CHECK(out.dtype() == at::kFloat && out.is_contiguous() && out.numel() == x.numel(), "exl3_dec_moe: out must be fp32 [1, H]");
    TORCH_CHECK(selected.dtype() == at::kLong && weights.dtype() == at::kHalf &&
                selected.numel() == weights.numel() && selected.is_contiguous() && weights.is_contiguous(),
                "exl3_dec_moe: selected int64 / weights fp16 [1, topk]");
    const int H = x.numel();
    const int I = intermediate;
    const int topk = selected.numel();
    TORCH_CHECK(H % STRIP == 0 && I % STRIP == 0, "exl3_dec_moe: H and I must be multiples of 512");
    const int kb2_gu = kb2_of(K_gu);
    const int kb2_d = kb2_of(K_down);
    const int min_blocks_a = env_int("EXL3_DEC_MOE_MIN_BLOCKS_A", 32);
    const int min_blocks_b = env_int("EXL3_DEC_MOE_MIN_BLOCKS_B", 32);
    const int forced_a = env_int("EXL3_DEC_MOE_KTW_A", 0);
    const int forced_b = env_int("EXL3_DEC_MOE_KTW_B", 0);
    // MoE sweep winners: gate/up 8 (K2.5 loses 12 % at 16/32, K2 is flat), down 16
    const int ring_a = ring_of_kb2(kb2_gu), ring_b = ring_of_kb2(kb2_d);
    const int ktw_a = forced_a > 0 || !ktw_valid(H / 16, ring_a, 8) ?
        pick_ktw(H / 16, 2 * topk * (I / STRIP), ring_a, min_blocks_a, forced_a) : 8;
    const int ktw_b = forced_b > 0 || !ktw_valid(I / 16, ring_b, 16) ?
        pick_ktw(I / 16, topk * (H / STRIP), ring_b, min_blocks_b, forced_b) : 16;
    TORCH_CHECK(act.dtype() == at::kHalf && act.numel() >= (int64_t) topk * I, "exl3_dec_moe: act too small");
    // gate/up: 6 blocks per strip along k (ragged 5-6 tiles per wave) measured 2-4 % faster than 4
    const int kbs_a_forced = env_int("EXL3_DEC_MOE_KBS_A", 0);
    const int kbs_h = kbs_of(H / 16, ktw_a, kbs_a_forced > 0 ? kbs_a_forced : (H / 128 >= 6 ? 6 : 0));
    const int kbs_i = kbs_of(I / 16, ktw_b, env_int("EXL3_DEC_MOE_KBS_B", 0));
    const int64_t need = std::max((int64_t) topk * 2 * kbs_h * I, (int64_t) topk * kbs_i * H);
    TORCH_CHECK(scratch.numel() >= need, "exl3_dec_moe: scratch too small");
    TORCH_CHECK(counters.numel() >= CTR_GEMV && topk * (I / STRIP) <= CTR_B - CTR_A && H / STRIP <= CTR_GEMV - CTR_B,
                "exl3_dec_moe: counters too small");

    MoeArgs a;
    a.sel = selected.data_ptr<int64_t>();
    a.wts = reinterpret_cast<const half*>(weights.data_ptr());
    a.gt = gate_trellis.data_ptr<int64_t>(); a.gs = gate_suh.data_ptr<int64_t>(); a.gv = gate_svh.data_ptr<int64_t>();
    a.ut = up_trellis.data_ptr<int64_t>(); a.us = up_suh.data_ptr<int64_t>(); a.uv = up_svh.data_ptr<int64_t>();
    a.dt = down_trellis.data_ptr<int64_t>(); a.ds = down_suh.data_ptr<int64_t>(); a.dv = down_svh.data_ptr<int64_t>();
    a.act = reinterpret_cast<half*>(act.data_ptr());
    a.out = out.data_ptr<float>();
    a.scratch = scratch.data_ptr<float>();
    a.counters = counters.data_ptr<int>();
    a.H = H; a.I = I; a.topk = topk; a.experts = gate_trellis.numel();
    a.accumulate = accumulate ? 1 : 0;
    a.act_limit = (float) act_limit;

    const half* xp = reinterpret_cast<const half*>(x.data_ptr());
    const int cb = mcg ? 1 : (env_int("EXL3_DEC_MOE_FOLD", 1) ? 3 : 2);
    EXL3_DEC_DISPATCH_CB(cb, launch_moe_gu<CB>(kb2_gu, xp, a, kbs_h, stream));
    EXL3_DEC_DISPATCH_CB(cb, launch_moe_down<CB>(kb2_d, a, kbs_i, stream));
    cuda_check(cudaPeekAtLastError());
}


// Batch-1 routed MoE with the shared expert folded in as an extra slot (decB8,
// EXL3_DEC_SHARED_FOLD). Same two launches as exl3_dec_moe, grid.y = topk + 1; the shared slot
// has weight 1 and its own K and codebook. out = routed sum + shared (+ residual if accumulate).
// act: fp16 [topk + 1, I]. Split knobs are separate from the unfolded path's so the tuning under
// fold never moves the default: EXL3_DEC_SHF_KBS_A/B, EXL3_DEC_SHF_KTW_A/B, EXL3_DEC_SHF_RING1.
void exl3_dec_moe_shared
(
    const at::Tensor& x,
    at::Tensor& out,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    const at::Tensor& sh_gate_trellis,
    const at::Tensor& sh_gate_suh,
    const at::Tensor& sh_gate_svh,
    const at::Tensor& sh_up_trellis,
    const at::Tensor& sh_up_suh,
    const at::Tensor& sh_up_svh,
    const at::Tensor& sh_down_trellis,
    const at::Tensor& sh_down_suh,
    const at::Tensor& sh_down_svh,
    at::Tensor& act,
    at::Tensor& scratch,
    at::Tensor& counters,
    int64_t intermediate,
    double K_gu,
    double K_down,
    double K_sh_gu,
    double K_sh_down,
    bool accumulate,
    double act_limit,
    bool mcg
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous() && x.numel() == x.size(-1), "exl3_dec_moe_shared: x must be fp16 [1, H]");
    TORCH_CHECK(out.dtype() == at::kFloat && out.is_contiguous() && out.numel() == x.numel(), "exl3_dec_moe_shared: out must be fp32 [1, H]");
    TORCH_CHECK(selected.dtype() == at::kLong && weights.dtype() == at::kHalf &&
                selected.numel() == weights.numel() && selected.is_contiguous() && weights.is_contiguous(),
                "exl3_dec_moe_shared: selected int64 / weights fp16 [1, topk]");
    const int H = x.numel();
    const int I = intermediate;
    const int topk = selected.numel();
    const int slots = topk + 1;
    TORCH_CHECK(H % STRIP == 0 && I % STRIP == 0, "exl3_dec_moe_shared: H and I must be multiples of 512");
    TORCH_CHECK(sh_gate_trellis.size(0) == H / 16 && sh_gate_trellis.size(1) == I / 16 &&
                sh_up_trellis.size(0) == H / 16 && sh_up_trellis.size(1) == I / 16 &&
                sh_down_trellis.size(0) == I / 16 && sh_down_trellis.size(1) == H / 16,
                "exl3_dec_moe_shared: shared expert shape must match the routed experts");
    const int kb2_gu = kb2_of(K_gu);
    const int kb2_d = kb2_of(K_down);
    const int kb2s_gu = kb2_of(K_sh_gu);
    const int kb2s_d = kb2_of(K_sh_down);
    const int min_blocks_a = env_int("EXL3_DEC_MOE_MIN_BLOCKS_A", 32);
    const int min_blocks_b = env_int("EXL3_DEC_MOE_MIN_BLOCKS_B", 32);
    const int forced_a = env_int("EXL3_DEC_SHF_KTW_A", 0);
    const int forced_b = env_int("EXL3_DEC_SHF_KTW_B", 0);
    const int ring_a = ring_of_kb2(kb2_gu), ring_b = ring_of_kb2(kb2_d);
    const int ktw_a = forced_a > 0 || !ktw_valid(H / 16, ring_a, 8) ?
        pick_ktw(H / 16, 2 * slots * (I / STRIP), ring_a, min_blocks_a, forced_a) : 8;
    const int ktw_b = forced_b > 0 || !ktw_valid(I / 16, ring_b, 16) ?
        pick_ktw(I / 16, slots * (H / STRIP), ring_b, min_blocks_b, forced_b) : 16;
    TORCH_CHECK(act.dtype() == at::kHalf && act.numel() >= (int64_t) slots * I, "exl3_dec_moe_shared: act too small");
    const int kbs_a_forced = env_int("EXL3_DEC_SHF_KBS_A", 0);
    const int kbs_h = kbs_of(H / 16, ktw_a, kbs_a_forced > 0 ? kbs_a_forced : (H / 128 >= 6 ? 6 : 0));
    const int kbs_i = kbs_of(I / 16, ktw_b, env_int("EXL3_DEC_SHF_KBS_B", 0));
    const int64_t need = std::max((int64_t) slots * 2 * kbs_h * I, (int64_t) slots * kbs_i * H);
    TORCH_CHECK(scratch.numel() >= need, "exl3_dec_moe_shared: scratch too small");
    TORCH_CHECK(counters.numel() >= CTR_GEMV && slots * (I / STRIP) <= CTR_B - CTR_A && H / STRIP <= CTR_GEMV - CTR_B,
                "exl3_dec_moe_shared: counters too small");

    MoeArgs a;
    a.sel = selected.data_ptr<int64_t>();
    a.wts = reinterpret_cast<const half*>(weights.data_ptr());
    a.gt = gate_trellis.data_ptr<int64_t>(); a.gs = gate_suh.data_ptr<int64_t>(); a.gv = gate_svh.data_ptr<int64_t>();
    a.ut = up_trellis.data_ptr<int64_t>(); a.us = up_suh.data_ptr<int64_t>(); a.uv = up_svh.data_ptr<int64_t>();
    a.dt = down_trellis.data_ptr<int64_t>(); a.ds = down_suh.data_ptr<int64_t>(); a.dv = down_svh.data_ptr<int64_t>();
    a.act = reinterpret_cast<half*>(act.data_ptr());
    a.out = out.data_ptr<float>();
    a.scratch = scratch.data_ptr<float>();
    a.counters = counters.data_ptr<int>();
    a.H = H; a.I = I; a.topk = topk; a.experts = gate_trellis.numel();
    a.accumulate = accumulate ? 1 : 0;
    a.act_limit = (float) act_limit;
    a.sht[0] = sh_gate_trellis.data_ptr(); a.sht[1] = sh_up_trellis.data_ptr(); a.sht[2] = sh_down_trellis.data_ptr();
    a.shs[0] = reinterpret_cast<const half*>(sh_gate_suh.data_ptr());
    a.shs[1] = reinterpret_cast<const half*>(sh_up_suh.data_ptr());
    a.shs[2] = reinterpret_cast<const half*>(sh_down_suh.data_ptr());
    a.shv[0] = reinterpret_cast<const half*>(sh_gate_svh.data_ptr());
    a.shv[1] = reinterpret_cast<const half*>(sh_up_svh.data_ptr());
    a.shv[2] = reinterpret_cast<const half*>(sh_down_svh.data_ptr());

    const half* xp = reinterpret_cast<const half*>(x.data_ptr());
    const int rs = env_int("EXL3_DEC_SHF_RING1", 0) ? 1 : 0;
    // Routed codebook as exl3_dec_moe; the shared expert keeps the dense GEMV's exact decode
    // (mul1 -> CB 2, never the MOE_FOLD variant), so only its split differs from the unfolded path.
    if (mcg)
        launch_moe_sh<1, 1>(kb2_gu, kb2_d, kb2s_gu, kb2s_d, xp, a, kbs_h, kbs_i, rs, stream);
    else if (env_int("EXL3_DEC_MOE_FOLD", 1))
        launch_moe_sh<3, 2>(kb2_gu, kb2_d, kb2s_gu, kb2s_d, xp, a, kbs_h, kbs_i, rs, stream);
    else
        launch_moe_sh<2, 2>(kb2_gu, kb2_d, kb2s_gu, kb2s_d, xp, a, kbs_h, kbs_i, rs, stream);
    cuda_check(cudaPeekAtLastError());
}

// R-row union MoE. See the comment above MoeArgsR. Builds the unique-expert /
// assignment table on the host (R * topk <= 80 entries, negligible next to the GEMVs) and runs
// gu + down + combine. x: [R, H] fp16, out: [R, H] fp32, selected/weights: [R, topk].
void exl3_dec_moe_union
(
    const at::Tensor& x,
    at::Tensor& out,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    at::Tensor& act,
    at::Tensor& down_part,
    at::Tensor& scratch,
    at::Tensor& counters,
    int64_t intermediate,
    double K_gu,
    double K_down,
    bool accumulate,
    double act_limit,
    bool mcg,
    bool device_table
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous() && x.dim() == 2, "exl3_dec_moe_union: x must be fp16 [R, H]");
    TORCH_CHECK(out.dtype() == at::kFloat && out.is_contiguous() && out.sizes() == x.sizes(), "exl3_dec_moe_union: out must be fp32 [R, H]");
    TORCH_CHECK(selected.dtype() == at::kLong && weights.dtype() == at::kHalf &&
                selected.sizes() == weights.sizes() && selected.dim() == 2 && selected.is_contiguous() && weights.is_contiguous(),
                "exl3_dec_moe_union: selected/weights int64/fp16 [R, topk]");
    const int R = x.size(0);
    const int H = x.size(1);
    const int I = intermediate;
    const int topk = selected.size(1);
    TORCH_CHECK(R >= 1 && R <= MOE_R_MAX, "exl3_dec_moe_union: R must be 1..8");
    TORCH_CHECK(H % STRIP == 0 && I % STRIP == 0, "exl3_dec_moe_union: H and I must be multiples of 512");

    const int64_t experts = gate_trellis.numel();
    at::Tensor uexp_t, assign_t;
    int U;
    if (!device_table)
    {
        // Host-side unique-expert / assignment table.
        auto sel_cpu = selected.to(at::kCPU).contiguous();
        const int64_t* selp = sel_cpu.data_ptr<int64_t>();
        std::vector<int64_t> uexp;
        std::vector<int> assign;
        {
            std::unordered_map<int64_t, int> idx;
            for (int r = 0; r < R; ++r)
            {
                for (int k = 0; k < topk; ++k)
                {
                    const int64_t e = selp[(size_t) r * topk + k];
                    if (e < 0 || e >= experts) continue;
                    auto it = idx.find(e);
                    int u;
                    if (it == idx.end()) { u = (int) uexp.size(); uexp.push_back(e); idx[e] = u; assign.resize(assign.size() + MOE_R_MAX, -1); }
                    else u = it->second;
                    for (int j = 0; j < MOE_R_MAX; ++j)
                        if (assign[(size_t) u * MOE_R_MAX + j] < 0) { assign[(size_t) u * MOE_R_MAX + j] = r * topk + k; break; }
                }
            }
        }
        U = (int) uexp.size();
        auto opts_l = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
        auto opts_i = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
        uexp_t = U > 0 ? at::from_blob(uexp.data(), { U }, opts_l).clone().to(x.device()) : at::empty({ 0 }, opts_l.device(x.device()));
        assign_t = U > 0 ? at::from_blob(assign.data(), { U * MOE_R_MAX }, opts_i).clone().to(x.device()) : at::empty({ 0 }, opts_i.device(x.device()));
    }
    else
    {
        // Table built on device, grid.y = max possible U, padded slots exit at once
        const int64_t experts_d = experts;
        TORCH_CHECK(topk <= 64, "exl3_dec_moe_union: device table needs topk <= 64");
        U = (int) std::min<int64_t>((int64_t) R * topk, experts_d);
        // hostidle2: these two tables are pure scratch (moe_union_build_kernel overwrites
        // every element it reads back) and their size depends only on (device, R, topk,
        // experts), so a decode round used to pay two at::empty per MoE layer -- 86 allocator
        // calls per round on the served GLM-5.3 shape (42 layers x 2). Keep one buffer per
        // distinct geometry. Same memory, same kernel, same values: the build kernel writes
        // the whole table before any reader sees it, exactly as with a fresh allocation.
        static std::mutex tbl_mu;
        // intentionally leaked: static at::Tensor destruction would run after the CUDA
        // context is gone at interpreter exit
        static auto* tbls = new std::map<std::tuple<int, int, int, int64_t>, std::pair<at::Tensor, at::Tensor>>();
        const auto tkey = std::make_tuple((int) x.device().index(), R, topk, experts_d);
        std::lock_guard<std::mutex> tbl_guard(tbl_mu);
        auto it = tbls->find(tkey);
        if (it == tbls->end())
        {
            auto ins = tbls->emplace(tkey, std::make_pair(
                at::empty({ U }, selected.options()),
                at::empty({ (int64_t) U * MOE_R_MAX }, selected.options().dtype(at::kInt))));
            it = ins.first;
        }
        uexp_t = it->second.first;
        assign_t = it->second.second;
        moe_union_build_kernel<<<1, 256, 0, stream>>>(selected.data_ptr<int64_t>(), R * topk, (int) experts_d, U,
                                                      uexp_t.data_ptr<int64_t>(), assign_t.data_ptr<int>());
    }

    const int kb2_gu = kb2_of(K_gu);
    const int kb2_d = kb2_of(K_down);
    // Same knobs, same formulas as exl3_dec_moe (env_int names shared on purpose): the R-row
    // kernel must pick the identical kbs (k-split count -> cross-block partial-sum order) as a
    // batch-1 call would for the same shapes, or the two aren't comparable at all. The only
    // deliberate difference from exl3_dec_moe is topk -> max(U, 1) inside pick_ktw's
    // strips_total, since U (not topk) is this kernel's actual block count along that axis.
    const int min_blocks_a = env_int("EXL3_DEC_MOE_MIN_BLOCKS_A", 32);
    const int min_blocks_b = env_int("EXL3_DEC_MOE_MIN_BLOCKS_B", 32);
    const int forced_a = env_int("EXL3_DEC_MOE_KTW_A", 0);
    const int forced_b = env_int("EXL3_DEC_MOE_KTW_B", 0);
    const int ring_a = ring_of_kb2(kb2_gu), ring_b = ring_of_kb2(kb2_d);
    // topk, not U: the invariant is "same kbs as a batch-1 call for these shapes" (kbs sets the
    // cross-block partial-sum order via arrive()), and batch-1's own pick_ktw uses topk here too.
    // U only ever changes grid.y (how many unique-expert blocks launch), never this choice.
    const int ktw_a = forced_a > 0 || !ktw_valid(H / 16, ring_a, 8) ?
        pick_ktw(H / 16, 2 * topk * (I / STRIP), ring_a, min_blocks_a, forced_a) : 8;
    const int ktw_b = forced_b > 0 || !ktw_valid(I / 16, ring_b, 16) ?
        pick_ktw(I / 16, topk * (H / STRIP), ring_b, min_blocks_b, forced_b) : 16;
    const int kbs_a_forced = env_int("EXL3_DEC_MOE_KBS_A", 0);
    // EXL3_MOEDEC1_KBS_A (default 0 = legacy behavior): union-only k-split override.
    // kbs=1 is invalid for the R-row kernels (k_split ktw=32 overflows the
    // MOE_R_KTW_MAX=16 staging) and is rejected; measured R3U21: 2/3 beat 6 by ~3 %.
    const int moedec1_kbs_a = env_int("EXL3_MOEDEC1_KBS_A", 0);
    TORCH_CHECK(moedec1_kbs_a == 0 || (moedec1_kbs_a >= 2 && moedec1_kbs_a <= 32),
                "exl3_dec_moe_union: EXL3_MOEDEC1_KBS_A must be 0 or 2..32");
    const int kbs_h = kbs_of(H / 16, ktw_a, moedec1_kbs_a > 0 ? moedec1_kbs_a :
                             (kbs_a_forced > 0 ? kbs_a_forced : (H / 128 >= 6 ? 6 : 0)));
    const int kbs_i = kbs_of(I / 16, ktw_b, env_int("EXL3_DEC_MOE_KBS_B", 0));
    TORCH_CHECK(ktw_a <= MOE_R_KTW_MAX && ktw_b <= MOE_R_KTW_MAX, "exl3_dec_moe_union: ktw exceeds MOE_R_KTW_MAX, widen it");

    TORCH_CHECK(act.dtype() == at::kHalf && act.numel() >= (int64_t) R * topk * I, "exl3_dec_moe_union: act too small");
    TORCH_CHECK(down_part.dtype() == at::kFloat && down_part.numel() >= (int64_t) R * topk * H, "exl3_dec_moe_union: down_part too small");
    const int64_t need = std::max((int64_t) R * topk * 2 * kbs_h * I, (int64_t) R * topk * kbs_i * H);
    TORCH_CHECK(scratch.numel() >= need, "exl3_dec_moe_union: scratch too small");
    const int ctr_a_size = R * topk * (I / STRIP);
    const int ctr_b_off = ctr_a_size;
    const int ctr_b_size = R * topk * (H / STRIP);
    TORCH_CHECK(counters.numel() >= ctr_a_size + ctr_b_size, "exl3_dec_moe_union: counters too small");

    MoeArgsR a;
    a.usel = U > 0 ? uexp_t.data_ptr<int64_t>() : nullptr;
    a.assign = U > 0 ? assign_t.data_ptr<int>() : nullptr;
    a.sel = selected.data_ptr<int64_t>();
    a.wts = reinterpret_cast<const half*>(weights.data_ptr());
    a.gt = gate_trellis.data_ptr<int64_t>(); a.gs = gate_suh.data_ptr<int64_t>(); a.gv = gate_svh.data_ptr<int64_t>();
    a.ut = up_trellis.data_ptr<int64_t>(); a.us = up_suh.data_ptr<int64_t>(); a.uv = up_svh.data_ptr<int64_t>();
    a.dt = down_trellis.data_ptr<int64_t>(); a.ds = down_suh.data_ptr<int64_t>(); a.dv = down_svh.data_ptr<int64_t>();
    a.x = reinterpret_cast<const half*>(x.data_ptr());
    a.act = reinterpret_cast<half*>(act.data_ptr());
    a.down_part = down_part.data_ptr<float>();
    a.out = out.data_ptr<float>();
    a.scratch = scratch.data_ptr<float>();
    a.counters = counters.data_ptr<int>();
    a.H = H; a.I = I; a.topk = topk; a.experts = experts; a.R = R; a.U = U;
    a.accumulate = accumulate ? 1 : 0;
    a.act_limit = (float) act_limit;

    if (U > 0)
    {
        const int cb = mcg ? 1 : (env_int("EXL3_DEC_MOE_FOLD", 1) ? 3 : 2);
        // EXL3_MOEDEC1_RM3=1 (default 0): R=3 union uses RM=3 instead of RM=4 (tighter LDS
        // footprint / fewer wasted registers when every expert serves at most 3 rows).
        if (env_int("EXL3_MOEDEC1_RM3", 0) && R == 3)
        {
            EXL3_DEC_DISPATCH_CB(cb, launch_moe_gu_r<CB, 3>(kb2_gu, a, kbs_h, stream));
            EXL3_DEC_DISPATCH_CB(cb, launch_moe_down_r<CB, 3>(kb2_d, a, kbs_i, ctr_b_off, stream));
        }
        else
        {
            EXL3_DEC_DISPATCH_RM(R, EXL3_DEC_DISPATCH_CB(cb, launch_moe_gu_r<CB, RM>(kb2_gu, a, kbs_h, stream)));
            EXL3_DEC_DISPATCH_RM(R, EXL3_DEC_DISPATCH_CB(cb, launch_moe_down_r<CB, RM>(kb2_d, a, kbs_i, ctr_b_off, stream)));
        }
    }
    else if (!accumulate)
        out.zero_();
    if (U > 0 || accumulate)
    {
        // EXL3_MOE_COMBINE_WIDE=1 (default 0): H / 512 column blocks per row instead of one block
        // per row (R = 2 ran 2 blocks for 4096 columns, latency-bound at ~18 us)
        const int cs = env_int("EXL3_MOE_COMBINE_WIDE", 0) ? max(1, H / 512) : 1;
        moe_combine_kernel_r<<<dim3(R, cs), 256, 0, stream>>>(a);
    }
    cuda_check(cudaPeekAtLastError());
}


void exl3_dec_router
(
    const at::Tensor& x,
    const at::Tensor& gate,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& selected,
    at::Tensor& weights,
    at::Tensor& scratch,
    at::Tensor& counters,
    double scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int H = x.numel();
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous() && x.numel() == x.size(-1), "exl3_dec_router: x fp16 [1, H]");
    TORCH_CHECK(gate.dtype() == at::kHalf && gate.is_contiguous() && gate.dim() == 2 && gate.size(0) == H, "exl3_dec_router: gate fp16 [H, E]");
    const int E = gate.size(1);
    const int topk = selected.numel();
    TORCH_CHECK(H % ROUTER_KCHUNK == 0 && E <= 1024 && E % 8 == 0 && topk <= 64 && topk <= E, "exl3_dec_router: unsupported shape");
    TORCH_CHECK(selected.dtype() == at::kLong && weights.dtype() == at::kHalf && weights.numel() == topk, "exl3_dec_router: outputs");
    TORCH_CHECK(scratch.numel() >= (int64_t) (H / ROUTER_KCHUNK) * E && counters.numel() > CTR_ROUTER, "exl3_dec_router: workspace");
    const float* bp = nullptr;
    if (bias.has_value() && bias->defined())
    {
        TORCH_CHECK(bias->dtype() == at::kFloat && bias->numel() == E && bias->is_contiguous(), "exl3_dec_router: bias fp32 [E]");
        bp = bias->data_ptr<float>();
    }
    RouterNorm rn = {};
    router_kernel<false><<<H / ROUTER_KCHUNK, 256, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()), rn, reinterpret_cast<const half*>(gate.data_ptr()), bp,
        selected.data_ptr<int64_t>(), reinterpret_cast<half*>(weights.data_ptr()),
        scratch.data_ptr<float>(), counters.data_ptr<int>(), H, E, topk, (float) scale);
    cuda_check(cudaPeekAtLastError());
}

// NR = 2..4 verify rows, one launch (router_rows_kernel); bitwise = NR exl3_dec_router calls.
void exl3_dec_router_rows
(
    const at::Tensor& x,
    const at::Tensor& gate,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& selected,
    at::Tensor& weights,
    at::Tensor& scratch,
    at::Tensor& counters,
    double scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == at::kHalf && x.is_contiguous() && x.dim() == 2, "exl3_dec_router_rows: x fp16 [NR, H]");
    const int NR = x.size(0);
    const int H = x.size(1);
    TORCH_CHECK(gate.dtype() == at::kHalf && gate.is_contiguous() && gate.dim() == 2 && gate.size(0) == H, "exl3_dec_router_rows: gate fp16 [H, E]");
    const int E = gate.size(1);
    TORCH_CHECK(selected.dim() == 2 && selected.size(0) == NR && selected.is_contiguous(), "exl3_dec_router_rows: selected [NR, topk]");
    const int topk = selected.size(1);
    TORCH_CHECK(NR >= 2 && NR <= 4, "exl3_dec_router_rows: NR in 2..4");
    TORCH_CHECK(H % ROUTER_KCHUNK == 0 && E <= 320 && E % 8 == 0 && topk <= 64 && topk <= E, "exl3_dec_router_rows: unsupported shape");
    TORCH_CHECK(selected.dtype() == at::kLong && weights.dtype() == at::kHalf && weights.is_contiguous() && weights.numel() == NR * topk, "exl3_dec_router_rows: outputs");
    TORCH_CHECK(scratch.numel() >= (int64_t) NR * (H / ROUTER_KCHUNK) * E && counters.numel() > CTR_ROUTER, "exl3_dec_router_rows: workspace");
    const float* bp = nullptr;
    if (bias.has_value() && bias->defined())
    {
        TORCH_CHECK(bias->dtype() == at::kFloat && bias->numel() == E && bias->is_contiguous(), "exl3_dec_router_rows: bias fp32 [E]");
        bp = bias->data_ptr<float>();
    }
    #define ROUTER_ROWS_ARGS reinterpret_cast<const half*>(x.data_ptr()), reinterpret_cast<const half*>(gate.data_ptr()), bp, \
        selected.data_ptr<int64_t>(), reinterpret_cast<half*>(weights.data_ptr()), \
        scratch.data_ptr<float>(), counters.data_ptr<int>(), H, E, topk, (float) scale
    if (NR == 2) { router_rows_kernel<2><<<H / ROUTER_KCHUNK, 256, 0, stream>>>(ROUTER_ROWS_ARGS); }
    else if (NR == 3) { router_rows_kernel<3><<<H / ROUTER_KCHUNK, 256, 0, stream>>>(ROUTER_ROWS_ARGS); }
    else { router_rows_kernel<4><<<H / ROUTER_KCHUNK, 256, 0, stream>>>(ROUTER_ROWS_ARGS); }
    #undef ROUTER_ROWS_ARGS
    cuda_check(cudaPeekAtLastError());
}

void exl3_dec_router_norm
(
    const at::Tensor& xa,
    at::Tensor& r,
    const c10::optional<at::Tensor>& w,
    double eps,
    double constant_bias,
    double constant_scale,
    at::Tensor& y,
    const at::Tensor& gate,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& selected,
    at::Tensor& weights,
    at::Tensor& scratch,
    at::Tensor& counters,
    double scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(xa.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    // NR rows (1..ROUTER_NORM_MAX_ROWS) in one launch: xa / r / y are [NR, H], selected / weights [NR, topk]
    TORCH_CHECK(gate.dtype() == at::kHalf && gate.is_contiguous() && gate.dim() == 2, "exl3_dec_router_norm: gate fp16 [H, E]");
    const int H = gate.size(0);
    const int E = gate.size(1);
    TORCH_CHECK(H > 0 && xa.numel() % H == 0, "exl3_dec_router_norm: xa size");
    const int NR = xa.numel() / H;
    TORCH_CHECK(NR >= 1 && NR <= ROUTER_NORM_MAX_ROWS, "exl3_dec_router_norm: rows 1..8");
    TORCH_CHECK((xa.dtype() == at::kHalf || xa.dtype() == at::kFloat) && xa.is_contiguous() && xa.numel() == (int64_t) NR * H,
                "exl3_dec_router_norm: xa fp16/fp32 [NR, H]");
    TORCH_CHECK(r.dtype() == at::kFloat && r.is_contiguous() && r.numel() == (int64_t) NR * H, "exl3_dec_router_norm: r fp32 [NR, H]");
    TORCH_CHECK(y.dtype() == at::kHalf && y.is_contiguous() && y.numel() == (int64_t) NR * H, "exl3_dec_router_norm: y fp16 [NR, H]");
    TORCH_CHECK(selected.numel() % NR == 0, "exl3_dec_router_norm: selected size");
    const int topk = selected.numel() / NR;
    TORCH_CHECK(H % ROUTER_KCHUNK == 0 && E <= 1024 && E % 8 == 0 && topk <= 64 && topk <= E, "exl3_dec_router_norm: unsupported shape");
    TORCH_CHECK(selected.dtype() == at::kLong && selected.is_contiguous() && weights.dtype() == at::kHalf && weights.is_contiguous() &&
                weights.numel() == (int64_t) NR * topk, "exl3_dec_router_norm: outputs");
    TORCH_CHECK(scratch.numel() >= (int64_t) NR * (H / ROUTER_KCHUNK) * E && counters.numel() >= CTR_ROUTER + NR, "exl3_dec_router_norm: workspace");
    const float* bp = nullptr;
    if (bias.has_value() && bias->defined())
    {
        TORCH_CHECK(bias->dtype() == at::kFloat && bias->numel() == E && bias->is_contiguous(), "exl3_dec_router_norm: bias fp32 [E]");
        bp = bias->data_ptr<float>();
    }
    RouterNorm rn;
    rn.xa = xa.data_ptr();
    rn.xa_half = xa.dtype() == at::kHalf;
    rn.r = r.data_ptr<float>();
    rn.wtype = 0;
    rn.w = nullptr;
    if (w.has_value() && w->defined())
    {
        TORCH_CHECK(w->numel() == H && w->is_contiguous() && (w->dtype() == at::kHalf || w->dtype() == at::kBFloat16),
                    "exl3_dec_router_norm: weight fp16/bf16 [H]");
        rn.w = w->data_ptr();
        rn.wtype = w->dtype() == at::kHalf ? 1 : 2;
    }
    rn.eps = (float) eps; rn.cbias = (float) constant_bias; rn.cscale = (float) constant_scale;
    rn.y = reinterpret_cast<half*>(y.data_ptr());
    router_kernel<true><<<dim3(H / ROUTER_KCHUNK, NR), 256, 0, stream>>>(
        nullptr, rn, reinterpret_cast<const half*>(gate.data_ptr()), bp,
        selected.data_ptr<int64_t>(), reinterpret_cast<half*>(weights.data_ptr()),
        scratch.data_ptr<float>(), counters.data_ptr<int>(), H, E, topk, (float) scale);
    cuda_check(cudaPeekAtLastError());
}


static int norm_wtype(const c10::optional<at::Tensor>& w)
{
    if (!w.has_value() || !w->defined()) return 0;
    if (w->dtype() == at::kHalf) return 1;
    TORCH_CHECK(w->dtype() == at::kBFloat16, "exl3_dec_rms_norm: weight must be fp16/bf16");
    return 2;
}

// mode: 0 plain, 1 add into y, 2 residual-in (r += x; y = norm(r))
void exl3_dec_rms_norm
(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& w,
    at::Tensor& y,
    const c10::optional<at::Tensor>& r,
    double eps,
    double constant_bias,
    double constant_scale,
    int64_t mode
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int dim = x.size(-1);
    const int rows = x.numel() / dim;
    TORCH_CHECK(dim <= 256 * NORM_MAX_PER_THREAD && x.is_contiguous() && y.is_contiguous() && y.numel() == x.numel(),
                "exl3_dec_rms_norm: unsupported shape");
    const int wtype = norm_wtype(w);
    if (wtype) TORCH_CHECK(w->numel() == dim && w->is_contiguous(), "exl3_dec_rms_norm: weight shape");
    const void* wp = wtype ? w->data_ptr() : nullptr;
    const bool fx = x.dtype() == at::kFloat, fy = y.dtype() == at::kFloat;
    TORCH_CHECK((fx || x.dtype() == at::kHalf) && (fy || y.dtype() == at::kHalf), "exl3_dec_rms_norm: fp16/fp32 only");
    const float e = eps, cb = constant_bias, cs = constant_scale;
    if (mode == 2)
    {
        TORCH_CHECK(r.has_value() && r->is_contiguous() && r->numel() == x.numel(), "exl3_dec_rms_norm: residual");
        const bool fr = r->dtype() == at::kFloat;
        TORCH_CHECK(fr || r->dtype() == at::kHalf, "exl3_dec_rms_norm: residual dtype");
        #define RN_LAUNCH(TX, TY, TR) rms_norm_kernel<TX, TY, TR><<<rows, 256, 0, stream>>>( \
            reinterpret_cast<const TX*>(x.data_ptr()), wp, wtype, reinterpret_cast<TY*>(y.data_ptr()), \
            reinterpret_cast<TR*>(r->data_ptr()), dim, e, cb, cs, 2)
        if (fx && fr && !fy) RN_LAUNCH(float, half, float);
        else if (!fx && fr && !fy) RN_LAUNCH(half, half, float);
        else if (fx && !fr && !fy) RN_LAUNCH(float, half, half);
        else if (!fx && !fr && !fy) RN_LAUNCH(half, half, half);
        else TORCH_CHECK(false, "exl3_dec_rms_norm: res_in output must be fp16");
        #undef RN_LAUNCH
    }
    else
    {
        #define RN_LAUNCH(TX, TY) rms_norm_kernel<TX, TY, float><<<rows, 256, 0, stream>>>( \
            reinterpret_cast<const TX*>(x.data_ptr()), wp, wtype, reinterpret_cast<TY*>(y.data_ptr()), \
            nullptr, dim, e, cb, cs, (int) mode)
        if (fx && fy) RN_LAUNCH(float, float);
        else if (fx && !fy) RN_LAUNCH(float, half);
        else if (!fx && fy) RN_LAUNCH(half, float);
        else RN_LAUNCH(half, half);
        #undef RN_LAUNCH
    }
    cuda_check(cudaPeekAtLastError());
}

#endif
