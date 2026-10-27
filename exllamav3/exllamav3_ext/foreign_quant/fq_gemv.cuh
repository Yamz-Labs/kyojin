// Shared device-inline helpers for the foreign-quant decode kernels
// (fq_gemv.cu standalone GEMV, fq_moe.cu grouped MoE): x staging, lane dot
// products per format, shared-memory codebooks. See fq_gemv.cu for the
// design rationale (cooperative decode GEMV) and portability contract
// (no PTX, no warp shuffles, __fmul_rn/__fadd_rn only).
#pragma once

#include <cstdlib>
#include <cuda_fp16.h>
#include "fq_dequant.cuh"

namespace {

// Same-binary A/B switch (read per launch, so one process can interleave
// arms by flipping the env var): BOLT_FQ_STAB=0 restores the per-pair NVFP4
// gs multiply. Default on.
inline bool fq_use_stab()
{
    const char* e = std::getenv("BOLT_FQ_STAB");
    return !e || e[0] != '0';
}

// q8_1-activation decode path (Bolt phase 2: port of the llama.cpp mmvq
// design). Default ON (gates passed: exact-arithmetic, KLD rule,
// e2e coherence): BOLT_FQ_Q8ACT=0 restores the fp32 IQ2_XXS dots,
// BOLT_FQ_Q8NV=0 the fp32 NVFP4 dots, BOLT_FQ_Q8Q8=0 the fp32 Q8_0 dots.
// Separate flags so one binary A/Bs each format independently.
inline bool fq_use_q8act()
{
    const char* e = std::getenv("BOLT_FQ_Q8ACT");
    return !e || e[0] != '0';
}

inline bool fq_use_q8nv()
{
    const char* e = std::getenv("BOLT_FQ_Q8NV");
    return !e || e[0] != '0';
}

inline bool fq_use_q8q8()
{
    const char* e = std::getenv("BOLT_FQ_Q8Q8");
    return !e || e[0] != '0';
}

// v2 IQ2_XXS q8 dot (mirrors ggml vec_dot_iq2_xxs_q8_1 exactly): grid
// from __constant__ (like llama's iq2xxs_grid: 4 const u64 loads per
// 32-elem chunk, fully cached 2 KB table), signs unpacked inline,
// headers via direct u16 loads (2-aligned like ggml get_int_b2, no
// staging). Drops the v1 header staging + signs32 table build + 2
// barriers and the 18 bank-conflicted shared loads per chunk (SASS:
// v1 executes ~1.76x llama's per-row instructions). Default on;
// BOLT_FQ_Q8V2=0 restores the v1 dot for same-binary A/B.
inline bool fq_use_q8v2()
{
    const char* e = std::getenv("BOLT_FQ_Q8V2");
    return !e || e[0] != '0';
}

// Experiment knob: BOLT_FQ_Q8NT2=1 halves the row-per-block
// IQ2 thread count (2 chunks per thread: better latency hiding + less
// const-cache pressure at large nt). Same-binary A/B with full nt.
// (Grouped MoE keeps its fixed 256-thread tiles; only the row-per-block
// IQ2 launches use this.)
inline int fq_q8_nt_iq2(int K)
{
    int nchunks = K / 32;
    int nt = nchunks >= 256 ? 256 : ((nchunks + 31) / 32) * 32;
    const char* e = std::getenv("BOLT_FQ_Q8NT2");
    if (e && e[0] == '1')
        nt = nt > 32 ? nt / 2 : nt;
    return nt;
}

// Portable 4-way int8 dot-product-accumulate. CUDA: __dp4a (all our targets
// are SM >= 61). HIP/other: widening scalar fallback (same values, no PTX,
// no asm; mirrors llama.cpp ggml_cuda_dp4a's fallback arm).
__device__ __forceinline__ int fq_dp4a(int a, int b, int c)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    return c + (int)a8[0] * (int)b8[0] + (int)a8[1] * (int)b8[1] +
               (int)a8[2] * (int)b8[2] + (int)a8[3] * (int)b8[3];
#endif
}

// q8_1 block geometry (ggml block_q8_1: half d, half s, 32 int8).
#define FQ_Q8_1_GROUP 32
#define FQ_Q8_1_BYTES 36

// 8 outputs per block, 32 lanes per output, 256 threads per block.
#define FQ_GEMV_OUT_PER_BLOCK 8
#define FQ_GEMV_LANES 32
#define FQ_GEMV_THREADS (FQ_GEMV_OUT_PER_BLOCK * FQ_GEMV_LANES)

// Stage one x row (half or float in) to shared half. Vectorized float4 path
// when K is a multiple of 8, scalar tail otherwise.
template <typename IN>
__device__ void fq_stage_x(const IN* __restrict__ xr, __half* __restrict__ xs, int K)
{
    int tid = threadIdx.x;
    if (K % 8 == 0)
    {
        int nv = K / 8;
        if constexpr (std::is_same<IN, __half>::value)
        {
            const float4* v = reinterpret_cast<const float4*>(xr);
            float4* d = reinterpret_cast<float4*>(xs);
            for (int i = tid; i < nv; i += blockDim.x)
                d[i] = v[i];
        }
        else
        {
            const float* f = reinterpret_cast<const float*>(xr);
            for (int i = tid; i < nv; i += blockDim.x)
            {
                __half h[8];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    h[j] = __float2half_rn(f[i * 8 + j]);
                reinterpret_cast<float4*>(xs)[i] =
                    reinterpret_cast<const float4*>(h)[0];
            }
        }
    }
    else
    {
        for (int i = tid; i < K; i += blockDim.x)
        {
            if constexpr (std::is_same<IN, __half>::value)
                xs[i] = xr[i];
            else
                xs[i] = __float2half_rn(xr[i]);
        }
    }
}

// Small codebooks staged in shared memory per block (loaded once,
// cooperatively). They MUST NOT stay in __constant__ memory: every lane of a
// group reads a different entry (nibbles / grid indices diverge), and the
// constant cache serializes divergent addresses, while shared memory serves
// them at full speed (measured: NVFP4 mamba_in 474 -> ~70us).
struct fq_tables_sh
{
    // e2m1 replicated 4x (same value at nib<<2|c for c in 0..3): a warp's
    // 32 divergent nibble reads spread over 64 banks instead of piling on
    // 16 (measured NVFP4 at 211 GB/s vs Q8 485 GB/s on the same shape).
    float e2m1[64];
    uint32_t e4m3[256];
    uint64_t grid[256];
    uint8_t signs[128];
    uint8_t kmask[8];
};

__device__ void fq_tables_load(fq_tables_sh* t, int need_e2m1_e4m3, int need_iq2)
{
    int tid = threadIdx.x;
    if (need_e2m1_e4m3)
    {
        for (int i = tid; i < 64; i += blockDim.x)
            t->e2m1[i] = FQ_E2M1_LUT[i >> 2];   // index (nib<<2)|rep -> LUT[nib]
        for (int i = tid; i < 256; i += blockDim.x)
            t->e4m3[i] = FQ_E4M3_BITS[i];
    }
    if (need_iq2)
    {
        for (int i = tid; i < 256; i += blockDim.x)
            t->grid[i] = FQ_IQ2XXS_GRID[i];
        for (int i = tid; i < 128; i += blockDim.x)
            t->signs[i] = FQ_KSIGNS_IQ2XS[i];
        if (tid < 8)
            t->kmask[tid] = FQ_KMASK_IQ2XS[tid];
    }
}

// Q8_0 lane dot: blocks of 32, lane l takes elements l (mod 32). No table
// lookups (weight = int8 * scale), so nothing to stage. Four independent
// scalar accumulators over strided blocks (b = c mod 4): one chain stalls
// on the FMA + cvt latency every element; four chains keep independent
// FMAs in flight. Scalars, not acc[b&3]: dynamic array indexing spills to
// local memory. Values identical, summation order differs (fp16 tol).
__device__ float fq_dot_q8_0(const uint8_t* __restrict__ wr, const __half* __restrict__ xs,
                             int K, int lane)
{
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int nb = K / 32;
    for (int b = 0; b < nb; b++)
    {
        const uint8_t* blk = wr + b * 34;
        __half dh;
        memcpy(&dh, blk, 2);
        float d = __half2float(dh);
        float w = __fmul_rn((float)((int8_t)blk[2 + lane]), d);
        float p = __fmul_rn(__half2float(xs[b * 32 + lane]), w);
        int c = b & 3;
        if (c == 0) acc0 = __fadd_rn(acc0, p);
        else if (c == 1) acc1 = __fadd_rn(acc1, p);
        else if (c == 2) acc2 = __fadd_rn(acc2, p);
        else acc3 = __fadd_rn(acc3, p);
    }
    return __fadd_rn(__fadd_rn(acc0, acc1), __fadd_rn(acc2, acc3));
}

// Cooperatively stage a block's IQ2_XXS headers to shared: 8 groups x
// nb*16 u32 (the 64 q bytes of every 256-wide block for each of the 8
// output rows this block computes; scales excluded). Called in the kernel
// prologue before the consuming __syncthreads, so it adds no sync of its
// own. Rows past N are clamped to row 0 (callers' dots are n-guarded, so
// the clamp data is never consumed and no OOB read is possible).
// (The u32 source is generally misaligned, so plain LDG.U32 would fault;
// the staging loop moves ~nb u32 per thread and is off the hot path.)
__device__ void fq_stage_iq2_hdrs(const uint8_t* __restrict__ packed, int64_t stride,
                                   uint32_t* __restrict__ hdrs, int nb, int n0, int N)
{
    int tid = threadIdx.x;
    int per = nb * 16;
    for (int i = tid; i < 8 * per; i += FQ_GEMV_THREADS)
    {
        int g = i / per;
        int j = i % per;
        int n = (n0 + g < N) ? n0 + g : 0;
        uint32_t v;
        memcpy(&v, packed + (int64_t)n * stride + (j >> 4) * 66 + 2 + (j & 15) * 4, 4);
        hdrs[i] = v;
    }
}

// Single-row variant for the row-per-block q8 kernels (stages one row's
// headers: nb*16 u32). Same misalignment-safe memcpy loads as above.
__device__ void fq_stage_iq2_hdrs_row(const uint8_t* __restrict__ packed, int64_t stride,
                                      uint32_t* __restrict__ hdrs, int nb, int n, int N)
{
    int tid = threadIdx.x;
    int per = nb * 16;
    int nn = (n < N) ? n : 0;
    for (int j = tid; j < per; j += blockDim.x)
    {
        uint32_t v;
        memcpy(&v, packed + (int64_t)nn * stride + (j >> 4) * 66 + 2 + (j & 15) * 4, 4);
        hdrs[j] = v;
    }
}

// IQ2_XXS lane dot, v4: block headers (a0/a1/db source bytes) come from
// the cooperatively staged shared hdrs instead of per-lane global byte
// loads. SASS showed the old dot at ~220 instructions per 8-element
// block step, ~64 of them LDG.E.U8 + ~40 address/shift/combine ops, ALL
// redundantly executed by all 32 lanes for the same 8 bytes (the q chunk
// is lane-invariant by construction of the ib-outer layout). Staging
// loads each header byte once per block (coalesced, 1 u32 per thread at
// K=4096) and every lane reads a0/a1 as uniform shared broadcasts.
// Hoisted: elle/jj, kmask[jj], d*0.25; grid magnitude is a shared byte
// index (v3). Values identical, summation order differs (fp16 tol).
// (Step-10 negative: hoisting the uniform db = d4*(0.5+(a1>>28)) into a
// prologue table measured +9-17% slower -- the extra shared load sits on
// the dot's critical path at these occupancies. Reverted.)
__device__ float fq_dot_iq2_xxs(const uint8_t* __restrict__ wr, const __half* __restrict__ xs,
                                 const fq_tables_sh* __restrict__ tab,
                                 const uint32_t* __restrict__ hdrs, int K, int lane)
{
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    int elle = lane >> 3, jj = lane & 7;
    uint8_t mask = tab->kmask[jj];
    const uint8_t* grid8 = reinterpret_cast<const uint8_t*>(tab->grid);
    int nb = K / 256;
    for (int b = 0; b < nb; b++)
    {
        const uint8_t* blk = wr + b * 66;
        __half dh;
        memcpy(&dh, blk, 2);
        float d4 = __fmul_rn(__half2float(dh), 0.25f);
        const __half* xsb = xs + b * 256;
        const uint32_t* hb = hdrs + b * 16;
        for (int ib = 0; ib < 8; ib++)
        {
            uint32_t a0 = hb[ib * 2], a1 = hb[ib * 2 + 1];
            float db = __fmul_rn(d4, 0.5f + (float)(a1 >> 28));
            // The magnitude is the raw grid byte (idx, jj): one shared
            // byte load instead of u64 load + shift + mask.
            float mag = (float)grid8[(((a0 >> (8 * elle)) & 0xFF) << 3) + jj];
            uint8_t signs = tab->signs[(a1 >> (7 * elle)) & 127];
            float w = __fmul_rn(__fmul_rn(db, mag), (signs & mask) ? -1.0f : 1.0f);
            float p = __fmul_rn(__half2float(xsb[(ib << 5) + lane]), w);
            if (ib == 0) acc0 = __fadd_rn(acc0, p);
            else if (ib == 1) acc1 = __fadd_rn(acc1, p);
            else if (ib == 2) acc2 = __fadd_rn(acc2, p);
            else if (ib == 3) acc3 = __fadd_rn(acc3, p);
            else if (ib == 4) acc4 = __fadd_rn(acc4, p);
            else if (ib == 5) acc5 = __fadd_rn(acc5, p);
            else if (ib == 6) acc6 = __fadd_rn(acc6, p);
            else acc7 = __fadd_rn(acc7, p);
        }
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), __fadd_rn(acc2, acc3)),
                     __fadd_rn(__fadd_rn(acc4, acc5), __fadd_rn(acc6, acc7)));
}

// NVFP4 lane dot, v4: 64-k steps over CONSECUTIVE k (step s covers
// [64s, 64s+64); lane l takes the pair (64s+2l, 64s+2l+1)) instead of the
// v2/v3 strided layout. Consecutive pairs make the warp's x/pr reads fully
// coalesced (a strided consecutive-per-lane layout put every lane in one
// shared bank: 32-way conflicts, 2.4x slower), each pr byte serve its
// natural even/odd pair, x be read as half2, and each 16-wide scale group
// share one broadcast e4m3 read. The E2M1 codebook stays in shared (a
// register lut[] indexed by a runtime nibble would spill to local memory)
// with the v2 4x replica bank spread. Eight scalar accumulators over steps
// (s uniform across the warp, so the select is straight-line); values
// identical to v3, summation order differs (fp16 tol).
// NVFP4 E2M1 pair table: 256 float2 entries built cooperatively straight
// from the __constant__ codebook (no dependency on the shared e2m1, so no
// extra __syncthreads needed before the existing one; the 2 constant reads
// per thread sit in the prologue, amortized over the whole dot).
// Entry [byte] = (e2m1[lo nibble], e2m1[hi nibble]) unscaled; the per-step
// scale gs multiplies after. One shared load + zero index ALU per pair
// instead of two replicated-e2m1 loads + nibble extract/index math.
// No 4x replica (the full byte index spreads naturally); +2 KB dynamic
// shared for FMT==2 only, occupancy-neutral on Bolt shapes.
__device__ void fq_e2m1p_build(float2* __restrict__ dst)
{
    int tid = threadIdx.x;
    for (int b = tid; b < 256; b += FQ_GEMV_THREADS)
    {
        float lo = FQ_E2M1_LUT[b & 0xF];
        float hi = FQ_E2M1_LUT[(b >> 4) & 0xF];
        dst[b] = make_float2(lo, hi);
    }
}

// NVFP4 scaled-e4m3 table: stab[i] = e4m3_float[i] * global, 256 entries
// built cooperatively straight from the __constant__ codebook (no
// dependency on shared state, so no extra __syncthreads; 256 threads do
// one entry each, amortized over the whole dot). The per-launch-uniform
// `global` multiply moves from every pair (one FMUL per 2 elements) into
// the prologue; the dot then pays one shared broadcast load per pair
// instead of (sr byte load + e4m3 load + FMUL). Same FMUL with the same
// operands, so gs (and the output) is bit-identical.
__device__ void fq_e2m1s_build(float* __restrict__ dst, float global)
{
    int tid = threadIdx.x;
    if (tid < 256)
        dst[tid] = __fmul_rn(__uint_as_float(FQ_E4M3_BITS[tid]), global);
}

// One 64-k step of the v4 NVFP4 dot: lane l's pair
// (elements 64s+2l, 64s+2l+1) -> partial dot. Split out so the main loop
// can unroll 8 straight-line steps (one per accumulator, no select).
// STAB selects the gs source at compile time: true = prologue
// scaled-e4m3 table (BOLT_FQ_STAB=1), false = e4m3 load + global FMUL.
// Bit-identical either way; the template keeps each arm clean.
template <bool PAIR, bool STAB>
__device__ float fq_nvfp4_pair(const uint8_t* __restrict__ pr, const uint8_t* __restrict__ sr,
                               const half2* __restrict__ x2,
                               const fq_tables_sh* __restrict__ tab,
                               const float2* __restrict__ e2m1p,
                               const float* __restrict__ stab,
                               int s, int lane, float global)
{
    // Lane l's pair = byte/pr index 32s+l (consecutive across the warp).
    int bi = (s << 5) + lane;
    uint8_t pbyte = pr[bi];
    float2 xv = __half22float2(x2[bi]);
    // 4 unique scale bytes per step across the warp; shared broadcast.
    uint8_t sc = sr[(s << 2) + (lane >> 3)];
    float gs;
    if constexpr (STAB)
        gs = stab[sc];
    else
        gs = __fmul_rn(__uint_as_float(tab->e4m3[sc]), global);
    float w0, w1;
    if constexpr (PAIR)
    {
        float2 w = e2m1p[pbyte];
        w0 = __fmul_rn(w.x, gs);
        w1 = __fmul_rn(w.y, gs);
    }
    else
    {
        int rep = lane & 3;
        w0 = __fmul_rn(tab->e2m1[((pbyte & 0xF) << 2) | rep], gs);
        w1 = __fmul_rn(tab->e2m1[(((pbyte >> 4) & 0xF) << 2) | rep], gs);
    }
    return __fadd_rn(__fmul_rn(xv.x, w0), __fmul_rn(xv.y, w1));
}

template <bool PAIR, bool STAB>
__device__ float fq_dot_nvfp4(const uint8_t* __restrict__ pr, const uint8_t* __restrict__ sr,
                              const __half* __restrict__ xs,
                              const fq_tables_sh* __restrict__ tab,
                              const float2* __restrict__ e2m1p,
                              const float* __restrict__ stab,
                              int K, int lane, float global)
{
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    int S = K >> 6;               // 64-k steps (K a multiple of 64)
    const half2* __restrict__ x2 = reinterpret_cast<const half2*>(xs);
    // Full chunks of 8 steps: straight-line, one accumulator each.
    int S8 = S & ~7;
    for (int s0 = 0; s0 < S8; s0 += 8)
    {
        acc0 = __fadd_rn(acc0, fq_nvfp4_pair<PAIR, STAB>(pr, sr, x2, tab, e2m1p, stab, s0 + 0, lane, global));
        acc1 = __fadd_rn(acc1, fq_nvfp4_pair<PAIR, STAB>(pr, sr, x2, tab, e2m1p, stab, s0 + 1, lane, global));
        acc2 = __fadd_rn(acc2, fq_nvfp4_pair<PAIR, STAB>(pr, sr, x2, tab, e2m1p, stab, s0 + 2, lane, global));
        acc3 = __fadd_rn(acc3, fq_nvfp4_pair<PAIR, STAB>(pr, sr, x2, tab, e2m1p, stab, s0 + 3, lane, global));
        acc4 = __fadd_rn(acc4, fq_nvfp4_pair<PAIR, STAB>(pr, sr, x2, tab, e2m1p, stab, s0 + 4, lane, global));
        acc5 = __fadd_rn(acc5, fq_nvfp4_pair<PAIR, STAB>(pr, sr, x2, tab, e2m1p, stab, s0 + 5, lane, global));
        acc6 = __fadd_rn(acc6, fq_nvfp4_pair<PAIR, STAB>(pr, sr, x2, tab, e2m1p, stab, s0 + 6, lane, global));
        acc7 = __fadd_rn(acc7, fq_nvfp4_pair<PAIR, STAB>(pr, sr, x2, tab, e2m1p, stab, s0 + 7, lane, global));
    }
    // Remainder steps (S < 8 shapes, e.g. the K=256 E4M3 sweep).
    for (int s = S8; s < S; s++)
    {
        float p = fq_nvfp4_pair<PAIR, STAB>(pr, sr, x2, tab, e2m1p, stab, s, lane, global);
        int a = s & 7;
        if (a == 0) acc0 = __fadd_rn(acc0, p);
        else if (a == 1) acc1 = __fadd_rn(acc1, p);
        else if (a == 2) acc2 = __fadd_rn(acc2, p);
        else if (a == 3) acc3 = __fadd_rn(acc3, p);
        else if (a == 4) acc4 = __fadd_rn(acc4, p);
        else if (a == 5) acc5 = __fadd_rn(acc5, p);
        else if (a == 6) acc6 = __fadd_rn(acc6, p);
        else acc7 = __fadd_rn(acc7, p);
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), __fadd_rn(acc2, acc3)),
                     __fadd_rn(__fadd_rn(acc4, acc5), __fadd_rn(acc6, acc7)));
}

// ---- q8_1-activation decode path (Bolt phase 2) ----
//
// Port of the llama.cpp mmvq design: activations are quantized once per
// (row) to q8_1 blocks (int8 + fp16 scale, same arithmetic as ggml's
// quantize_q8_1: d = amax/127, q = round-half-away(xi/d)), then each
// weight dot is an INTEGER dp4a accumulate with ONE float scale per
// 32-elem chunk (same op order as ggml vec_dot_*_q8_1). ~10-20x fewer
// instructions per element than the per-element fp32 dequant dots above.
//
// Thread mapping (row-per-block kernels below): thread tid takes 32-elem
// chunks c = tid, tid+NT, ... (NT = blockDim.x). All 4-element groups
// inside a chunk stay within one 32-block, so int32 activation loads
// never span blocks and weight grid/sign bytes are consecutive.

// One 32-block of a row: amax, scale, round-half-away quantize, store.
// Matches ggml quantize_q8_1 (d = amax/127, q = amax==0 ? 0 : round(xi/d),
// ds = (d, sum)). NOTE: kept for documentation; the live quant kernel
// (fq_gemv_q8.cu) inlines its own per-thread loops. If you revive this,
// store dh/sh with single u16 stores, NOT memcpy (the memcpy byte-split
// store pattern loses the low byte with this toolchain, see phase-2 log).
template <typename IN>
__device__ void fq_quant_q8_1_block(const IN* __restrict__ xr, uint8_t* __restrict__ qb,
                                    int K)
{
    int tid = threadIdx.x;
    int nb32 = K / FQ_Q8_1_GROUP;
    for (int b = tid; b < nb32; b += blockDim.x)
    {
        float amax = 0.0f, sum = 0.0f;
        for (int i = 0; i < 32; i++)
        {
            float v;
            if constexpr (std::is_same<IN, __half>::value)
                v = __half2float(xr[b * 32 + i]);
            else
                v = xr[b * 32 + i];
            float a = fabsf(v);
            amax = fmaxf(amax, a);
            sum += v;
        }
        float d = amax / 127.0f;
        uint8_t* blk = qb + (size_t)b * FQ_Q8_1_BYTES;
        __half dh = __float2half_rn(d);
        __half sh = __float2half_rn(sum);
        memcpy(blk, &dh, 2);
        memcpy(blk + 2, &sh, 2);
        #pragma unroll
        for (int i = 0; i < 32; i++)
        {
            float v;
            if constexpr (std::is_same<IN, __half>::value)
                v = __half2float(xr[b * 32 + i]);
            else
                v = xr[b * 32 + i];
            // roundf = round-half-away, like ggml's quantize_q8_1.
            int q = (amax == 0.0f) ? 0 : (int)roundf(v / d);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            blk[4 + i] = (uint8_t)(int8_t)q;
        }
    }
}

// Expanded IQ2_XXS sign table for the q8 dot: signs32[i][h] holds the 4
// sign bytes (0x00/0xFF) for sign-index i and jj-half h (jj = 4h..4h+3),
// built cooperatively in the prologue from the staged signs table
// (kmask[jj] = 1<<jj single-bit selects). 256 entries; the q8 kernels run
// 32..256 threads, so loop (a bare `tid < 256` guard leaves entries
// unwritten at small NT).
__device__ void fq_signs32_build(const fq_tables_sh* __restrict__ tab,
                                 uint32_t* __restrict__ s32)
{
    int tid = threadIdx.x;
    for (int e = tid; e < 256; e += blockDim.x)
    {
        int i = e >> 1, h = e & 1;
        uint32_t S = tab->signs[i];
        uint32_t v = 0;
        #pragma unroll
        for (int m = 0; m < 4; m++)
        {
            uint32_t bit = (S >> (4 * h + m)) & 1u;
            // Mask to one byte: (0u-bit) is 0x00000000/0xFFFFFFFF.
            v |= ((0u - bit) & 0xFFu) << (8 * m);
        }
        s32[e] = v;
    }
}

// ksigns unpack for the v2 IQ2 q8 dot: verbatim ggml unpack_ksigns
// (vecdotq.cuh): v is the 7-bit sign index, the 8th sign rides the
// parity bit. Broadcast over u32 so the 0x08040201/0x80402010 masks
// select one sign bit per byte. Portable scalar ALU.
__device__ __forceinline__ uint32_t fq_unpack_ksigns(uint8_t v)
{
    uint32_t p = (uint32_t)__popc((unsigned int)v) & 1u;
    uint32_t s = (uint32_t)v ^ (p << 7);
    return s * 0x01010101u;
}

// Per-byte != 0 expand (0xFF where set, 0x00 where clear). CUDA: the
// __vcmpne4 video-SIMD builtin (1 insn; a compiler builtin like
// __dp4a/__byte_perm, not inline PTX). Other toolchains: widening
// scalar fallback (same values; mirrors fq_dp4a's fallback arm).
__device__ __forceinline__ uint32_t fq_vcmpne4(uint32_t a)
{
#if defined(__CUDA_ARCH__)
    return __vcmpne4(a, 0u);
#else
    uint32_t v = 0;
    #pragma unroll
    for (int m = 0; m < 4; m++)
        v |= ((0u - ((a >> (8 * m)) & 1u)) & 0xFFu) << (8 * m);
    return v;
#endif
}

// IQ2_XXS q8_1 lane dot v2: per-thread partial sum over its 32-elem
// chunks. Same integer arithmetic and op order as ggml
// vec_dot_iq2_xxs_q8_1 (4 grid-uint2 groups per chunk, 2 dp4a each,
// integer ls scale, one float scale per chunk), so KLD-vs-reference
// equals llama's by construction. Grid comes straight from the
// __constant__ table (no staging); headers load direct as u16 pairs
// (2-aligned, ggml get_int_b2 pattern); signs unpack inline (no
// signs32 table). The sign apply keeps the proven branchless
// ((g^s)+(s&0x01010101)) formula: grid bytes are never 0 so no
// cross-byte borrow, identical to llama's xor+__vsub4.
__device__ float fq_dot_iq2_xxs_q8v2(const uint8_t* __restrict__ wr,
                                     const uint8_t* __restrict__ aqb,
                                     int K, int tid, int nthreads)
{
    int nchunks = K / 32;
    float acc = 0.0f;
    const uint2* grid2 = reinterpret_cast<const uint2*>(FQ_IQ2XXS_GRID);
    for (int c = tid; c < nchunks; c += nthreads)
    {
        int b = c >> 3;          // 256-block
        int sub = c & 7;         // 32-chunk within the block
        const uint8_t* blk = wr + (size_t)b * FQ_IQ2_XXS_BYTES;
        __half dwh;
        memcpy(&dwh, blk, 2);
        float dw = __half2float(dwh);
        // Header pair (a0, a1): 4 u16 at blk+2+sub*8 (even = 2-aligned:
        // row base 16B-aligned + b*66 even + 2 + sub*8 even).
        const uint16_t* h16 = reinterpret_cast<const uint16_t*>(blk + 2 + sub * 8);
        uint16_t h0 = h16[0], h1 = h16[1], h2 = h16[2], h3 = h16[3];
        uint32_t a0 = (uint32_t)h0 | ((uint32_t)h1 << 16);
        uint32_t a1 = (uint32_t)h2 | ((uint32_t)h3 << 16);
        int ls = (int)((a1 >> 27) | 1u);   // (scale*2+1), integer path
        __half dah;
        memcpy(&dah, aqb + (size_t)c * FQ_Q8_1_BYTES, 2);
        float da = __half2float(dah);
        // Vectorized u32 activation loads (qb stride/offsets multiples of 4).
        const int32_t* aq32 = reinterpret_cast<const int32_t*>(aqb + (size_t)c * FQ_Q8_1_BYTES + 4);
        const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(&a0);
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            uint2 gp = grid2[aux8[j]];
            uint32_t signs = fq_unpack_ksigns((uint8_t)(a1 >> (7 * j)));
            uint32_t sv0 = fq_vcmpne4(signs & 0x08040201u);
            uint32_t sv1 = fq_vcmpne4(signs & 0x80402010u);
            uint32_t w0 = (gp.x ^ sv0) + (sv0 & 0x01010101u);
            uint32_t w1 = (gp.y ^ sv1) + (sv1 & 0x01010101u);
            sumi = fq_dp4a((int32_t)w0, aq32[2 * j], sumi);
            sumi = fq_dp4a((int32_t)w1, aq32[2 * j + 1], sumi);
        }
        int scaled = sumi * ls / 8;
        acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(dw, da), (float)scaled));
    }
    return acc;
}

// IQ2_XXS q8_1 lane dot: per-thread partial sum over its 32-elem chunks.
// Same integer arithmetic as ggml vec_dot_iq2_xxs_q8_1: 8 dp4a per chunk,
// integer ls scale ((a1>>27)|1, applied as sumi*ls/8 in int), one float
// scale (block d * activation d) per chunk. grid8 = staged grid bytes
// (u32-aligned pairs: chunk sub-group j2 reads u32 [g*2+(j2&1)]),
// signs32 = expanded signs (entry [(sidx<<1)+(j2&1)], bytes 0x00/0xFF),
// aqb = global q8_1 blocks of the row, hdrs = staged row headers
// (16 u32 per 256-block).
__device__ float fq_dot_iq2_xxs_q8(const uint8_t* __restrict__ wr,
                                   const uint32_t* __restrict__ hdrs,
                                   const uint8_t* __restrict__ grid8,
                                   const uint32_t* __restrict__ signs32,
                                   const uint8_t* __restrict__ aqb,
                                   int K, int tid, int nthreads)
{
    int nchunks = K / 32;
    float acc = 0.0f;
    const uint32_t* grid32 = reinterpret_cast<const uint32_t*>(grid8);
    for (int c = tid; c < nchunks; c += nthreads)
    {
        int b = c >> 3;          // 256-block
        int sub = c & 7;         // 32-chunk within the block
        const uint8_t* blk = wr + (size_t)b * FQ_IQ2_XXS_BYTES;
        __half dwh;
        memcpy(&dwh, blk, 2);
        float dw = __half2float(dwh);
        uint32_t a0 = hdrs[(size_t)b * 16 + sub * 2];
        uint32_t a1 = hdrs[(size_t)b * 16 + sub * 2 + 1];
        int ls = (int)((a1 >> 27) | 1u);   // (scale*2+1), integer path
        __half dah;
        memcpy(&dah, aqb + (size_t)c * FQ_Q8_1_BYTES, 2);
        float da = __half2float(dah);
        // Vectorized u32 activation loads (qb stride/offsets are multiples
        // of 4; the old memcpy-bytes compiled to LDG.E.U8).
        const int32_t* aq32 = reinterpret_cast<const int32_t*>(aqb + (size_t)c * FQ_Q8_1_BYTES + 4);
        int sumi = 0;
        #pragma unroll
        for (int j2 = 0; j2 < 8; j2++)
        {
            // 4 consecutive elements: grid bytes [g*8+4h .. +3] (one u32),
            // signs s32 for (sidx, h), activations 4 consecutive int8.
            int elle = j2 >> 1;
            int h = j2 & 1;
            uint32_t g = (a0 >> (8 * elle)) & 0xFFu;
            int32_t gv;
            memcpy(&gv, &grid32[g * 2 + h], 4);
            uint32_t sidx = (a1 >> (7 * elle)) & 127u;
            uint32_t sv = signs32[(sidx << 1) + h];
            // Apply signs without cross-byte borrow: grid bytes are never
            // 0 (iq2xxs magnitudes are 8/25/43), so (g^0xFF)+1 = -g mod
            // 256 carries at most into... nothing: 255-g <= 247, +1 <= 248.
            // s=0x00 keeps g (+0). Portable, no video intrinsics.
            uint32_t w4 = ((uint32_t)gv ^ sv) + (sv & 0x01010101u);
            sumi = fq_dp4a((int32_t)w4, aq32[j2], sumi);
        }
        int scaled = sumi * ls / 8;
        acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(dw, da), (float)scaled));
    }
    return acc;
}

// Doubled-E2M1 int8 table for the NVFP4 q8 dot, replicated 8x in shared
// (128 B): lookup j of a thread uses replica (j & 7), so consecutive
// lookups spread over 8 banks (same bank-spread pattern as the proven
// float e2m1 4x replica). Built cooperatively straight from the
// __constant__ table (no dependency on shared state, no extra sync).
// (Fallback path only: CUDA uses the __byte_perm expansion below.)
__device__ void fq_e2m1db_build(int8_t* __restrict__ dst)
{
    int tid = threadIdx.x;
    for (int e = tid; e < 128; e += blockDim.x)
        dst[e] = FQ_E2M1_DBL[e >> 3];
}

// Expand 8 consecutive E2M1 nibbles (one u32 of packed bytes, little-endian:
// byte j holds nibbles 2j (lo) and 2j+1 (hi)) to 8 consecutive doubled
// int8 as int2 (x = elems 0-3, y = elems 4-7). Same construction as
// llama.cpp's get_int_from_table_16 (vecdotq.cuh) MINUS its final
// even/odd shuffle: the intermediate tmp[i] already holds 4 consecutive
// values (nibbles 4i..4i+3), which is exactly our (consecutive) activation
// order, so the 0x6420/0x7531 reshuffle is skipped. 6 __byte_perm per 8
// nibbles. __byte_perm is a CUDA compiler builtin (no inline PTX); the
// table is __constant__ and read UNIFORMLY (all lanes read the same 4
// u32s, then permute in-register), so no constant-cache serialization
// (contrast the step-4 lesson on DIVERGENT constant reads). Non-CUDA
// toolchains take the shared-table scalar fallback (fq_dot_nvfp4_q8_fb
// below), which is the same verified arithmetic.
__device__ __forceinline__ int2 fq_e2m1_expand_8(uint32_t q4)
{
#if defined(__CUDA_ARCH__)
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(FQ_E2M1_DBL);
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t lo = __byte_perm(table32[0], table32[1], q4);
    uint32_t hi = __byte_perm(table32[2], table32[3], q4);
    uint32_t t0 = __byte_perm(lo, hi, sel);
    uint32_t q4h = q4 >> 16;
    lo = __byte_perm(table32[0], table32[1], q4h);
    hi = __byte_perm(table32[2], table32[3], q4h);
    uint32_t t1 = __byte_perm(lo, hi, sel >> 16);
    return make_int2((int)t0, (int)t1);
#else
    (void)q4;
    return make_int2(0, 0);
#endif
}

// NVFP4 q8_1 lane dot: per-thread partial sum over its 32-elem chunks.
// Same integer factorization as ggml vec_dot_nvfp4_q8_1: per 16-elem
// group, 4 dp4a over (doubled-e2m1 int8 x q8_1 int8), one float scale
// (ue4m3 = raw e4m3 x 0.5, exact dyadic halving, x activation d) per
// group. The row-uniform `global` multiplies once at the end (one
// rounding instead of one per group in the fp32 path). Layout note:
// unlike ggml's interleaved d[4]+qs[32] 64-blocks, ours is split
// (packed [N, K/2] E2M1 bytes, scales [N, K/16] E4M3 bytes): chunk c
// owns pr bytes [c*16, c*16+16) and scale bytes [c*2, c*2+2), and the
// activation q8_1 block c owns the 32 int8. ModelOpt packing: low
// nibble = even index, like fq_nvfp4_group.
// Weight expansion: CUDA packs each 8-nibble u32 via fq_e2m1_expand_8
// (6 __byte_perm); other toolchains use the shared-table scalar
// fallback (same arithmetic, verified by the exact gate).
__device__ float fq_dot_nvfp4_q8(const uint8_t* __restrict__ pr,
                                const uint8_t* __restrict__ sr,
                                const uint8_t* __restrict__ aqb,
                                const int8_t* __restrict__ e2m1db,
                                int K, float global, int tid, int nthreads)
{
    int nchunks = K / 32;
    float acc = 0.0f;
    for (int c = tid; c < nchunks; c += nthreads)
    {
        __half dah;
        memcpy(&dah, aqb + (size_t)c * FQ_Q8_1_BYTES, 2);
        float da = __half2float(dah);
        const int8_t* aq = reinterpret_cast<const int8_t*>(aqb + (size_t)c * FQ_Q8_1_BYTES + 4);
        const uint8_t* prc = pr + (size_t)c * 16;
        for (int g = 0; g < 2; g++)
        {
            // ue4m3 scale x 0.5 (matches ggml_cuda_ue4m3_to_fp32's /2;
            // exact: FQ_E4M3_BITS holds exact dyadics) x activation d.
            float d = __fmul_rn(__fmul_rn(__uint_as_float(FQ_E4M3_BITS[sr[c * 2 + g]]),
                                          0.5f), da);
            int32_t w0, w1, w2, w3;
#if defined(__CUDA_ARCH__)
            // Vectorized u32 weight loads (one LDG.E.U32 each, not 4 byte
            // loads: SASS showed the memcpy-bytes compile to LDG.E.U8).
            // Safe: pr rows have stride K/2 (multiple of 4, K%32 checked)
            // and chunk offsets c*16+g*8 (multiples of 8); torch bases are
            // 16B+ aligned. (Llama's get_int_b4 makes the same assumption.)
            // (pr/sr arrive row-offset by the caller.)
            const uint8_t* prn = pr + (size_t)c * 16 + g * 8;
            uint32_t q4a = reinterpret_cast<const uint32_t*>(prn)[0];
            uint32_t q4b = reinterpret_cast<const uint32_t*>(prn)[1];
            int2 e01 = fq_e2m1_expand_8(q4a);
            int2 e23 = fq_e2m1_expand_8(q4b);
            w0 = e01.x;
            w1 = e01.y;
            w2 = e23.x;
            w3 = e23.y;
            // Vectorized u32 activation loads: qb rows stride nb32*36
            // (multiple of 4: 36 = 4*9) and offsets c*36+4+g*16 (multiple
            // of 4), so these are 4-aligned too.
            const int32_t* aq32 = reinterpret_cast<const int32_t*>(aq + g * 16);
            int32_t u0 = aq32[0];
            int32_t u1 = aq32[1];
            int32_t u2 = aq32[2];
            int32_t u3 = aq32[3];
            int sumi = 0;
            sumi = fq_dp4a(w0, u0, sumi);
            sumi = fq_dp4a(w1, u1, sumi);
            sumi = fq_dp4a(w2, u2, sumi);
            sumi = fq_dp4a(w3, u3, sumi);
#else
            // Scalar fallback: 16 nibble lookups in the bank-spread
            // table ((j & 7) replica spreads lookups over 8 banks).
            uint32_t b[16];
            #pragma unroll
            for (int j = 0; j < 16; j++)
            {
                uint8_t pbyte = prc[g * 8 + (j >> 1)];
                uint8_t nib = (j & 1) ? (pbyte >> 4) & 0xF : pbyte & 0xF;
                b[j] = (uint32_t)(uint8_t)e2m1db[((uint32_t)nib << 3) | (uint32_t)(j & 7)];
            }
            w0 = (int32_t)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
            w1 = (int32_t)(b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24));
            w2 = (int32_t)(b[8] | (b[9] << 8) | (b[10] << 16) | (b[11] << 24));
            w3 = (int32_t)(b[12] | (b[13] << 8) | (b[14] << 16) | (b[15] << 24));
            int32_t u0, u1, u2, u3;
            memcpy(&u0, aq + g * 16, 4);
            memcpy(&u1, aq + g * 16 + 4, 4);
            memcpy(&u2, aq + g * 16 + 8, 4);
            memcpy(&u3, aq + g * 16 + 12, 4);
            int sumi = 0;
            sumi = fq_dp4a(w0, u0, sumi);
            sumi = fq_dp4a(w1, u1, sumi);
            sumi = fq_dp4a(w2, u2, sumi);
            sumi = fq_dp4a(w3, u3, sumi);
#endif
            acc = __fadd_rn(acc, __fmul_rn(d, (float)sumi));
        }
    }
    return __fmul_rn(acc, global);
}

// Q8_0 q8_1 lane dot: per-thread partial sum over its 32-elem chunks.
// Same integer arithmetic as ggml vec_dot_q8_0_q8_1 (VDR_MMVQ arm): 8
// dp4a over (weight int8 x activation int8), one float scale (weight d
// x activation d) per 32-block. No tables at all (weights are int8).
__device__ float fq_dot_q8_0_q8(const uint8_t* __restrict__ wr,
                                const uint8_t* __restrict__ aqb,
                                int K, int tid, int nthreads)
{
    int nchunks = K / 32;
    float acc = 0.0f;
    for (int c = tid; c < nchunks; c += nthreads)
    {
        const uint8_t* blk = wr + (size_t)c * FQ_Q8_0_BYTES;
        __half dwh;
        memcpy(&dwh, blk, 2);
        float dw = __half2float(dwh);
        __half dah;
        memcpy(&dah, aqb + (size_t)c * FQ_Q8_1_BYTES, 2);
        float da = __half2float(dah);
        const int8_t* aq = reinterpret_cast<const int8_t*>(aqb + (size_t)c * FQ_Q8_1_BYTES + 4);
        // Vectorized loads (SASS showed memcpy-bytes compile to LDG.E.U8):
        // weights as u16 pairs (Q8 blocks stride 34B: 2-aligned but not
        // 4-aligned -- same reason llama's Q8 path uses get_int_b2),
        // activations as u32 (qb stride/offsets are multiples of 4).
        const uint16_t* w16 = reinterpret_cast<const uint16_t*>(blk + 2);
        const int32_t* aq32 = reinterpret_cast<const int32_t*>(aq);
        int sumi = 0;
        #pragma unroll
        for (int j2 = 0; j2 < 8; j2++)
        {
            int32_t w4 = (int32_t)(w16[j2 * 2] | ((uint32_t)w16[j2 * 2 + 1] << 16));
            sumi = fq_dp4a(w4, aq32[j2], sumi);
        }
        acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(dw, da), (float)sumi));
    }
    return acc;
}

// K-quant + IQ4_XS q8_1-activation lane dots (port of the
// llama.cpp mmvq vmmq arms in ggml/src/ggml-cuda/vecdotq.cuh, MIT ggml-org).
// Same portable contract as the dots above (no PTX, no shuffles; __dp4a via
// fq_dp4a with HIP fallback, __vsub4/__vcmpne4 builtins with scalar
// fallback). Threads take sub-chunks round-robin (16 elems for Q4_K/Q5_K,
// 8 for Q6_K, 32 for IQ4_XS -- one vec_dot call each, same integer op
// order as ggml, so KLD-vs-reference equals llama's by construction).
// Weight words load via 4B memcpy (compiles to a single LDG.U32 when
// aligned) and scales via 2B memcpy; both are safe on any alignment.

// Q4_K geometry (QK_K = 256): dm half2 at +0, scales[12] at +4, qs[128]
// at +16; 144 B per 256-block.
#define FQ_Q4_K_GROUP 256
#define FQ_Q4_K_BYTES 144

// Q5_K geometry: dm half2 at +0, scales[12] at +4, qh[32] at +16,
// qs[128] at +48; 176 B per 256-block.
#define FQ_Q5_K_GROUP 256
#define FQ_Q5_K_BYTES 176

// Q6_K geometry: ql[128] at +0, qh[64] at +128, scales[16] int8 at +192,
// d half at +208; 210 B per 256-block.
#define FQ_Q6_K_GROUP 256
#define FQ_Q6_K_BYTES 210

// IQ4_XS geometry: d half at +0, scales_h u16 at +2, scales_l[4] at +4,
// qs[128] at +8; 136 B per 256-block.
#define FQ_IQ4_XS_GROUP 256
#define FQ_IQ4_XS_BYTES 136

// IQ4_XS nonlinear codebook (verbatim ggml kvalues_iq4nl, 16 int8).
// Plain __device__ (= global, L1-cached LDG), NOT __constant__: every lane
// reads a different nibble entry (divergent), and the constant cache
// serializes divergent addresses (measured; same reason the IQ2 grid
// is __device__). Internal linkage (const) like FQ_IQ2XXS_GRID, so sharing
// this header across TUs is safe.
__device__ const int8_t FQ_KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// Per-byte subtract mod 256. CUDA: the __vsub4 video-SIMD builtin (1 insn;
// a compiler builtin like __vcmpne4, not inline PTX). Other toolchains:
// scalar fallback (same values; mirrors fq_vcmpne4's fallback arm).
__device__ __forceinline__ uint32_t fq_vsub4(uint32_t a, uint32_t b)
{
#if defined(__CUDA_ARCH__)
    return __vsub4(a, b);
#else
    uint32_t v = 0;
    #pragma unroll
    for (int m = 0; m < 4; m++)
        v |= (((((a >> (8 * m)) & 0xFFu) - ((b >> (8 * m)) & 0xFFu)) & 0xFFu) << (8 * m));
    return v;
#endif
}

// Q4_K branchless scale/min decode (verbatim ggml get_scale_min_k4): 8
// (sc, m) pairs per 256-block from the 12 scale bytes; j = sub-group.
__device__ __forceinline__ void fq_q4k_scales(const uint8_t* __restrict__ sc12,
                                             int j, uint8_t* sc, uint8_t* m)
{
    uint16_t s0, s2, s4;
    memcpy(&s0, sc12 + (j & 1) * 2, 2);
    memcpy(&s2, sc12 + ((j & 1) + 2) * 2, 2);
    memcpy(&s4, sc12 + ((j & 1) + 4) * 2, 2);
    uint32_t hi = (j >= 2) ? 0xFFFFFFFFu : 0u;
    uint16_t aux0 = (uint16_t)(((s0 & 0x3F3Fu) & ~hi) |
                               ((((s4 >> 0) & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2)) & hi));
    uint16_t aux1 = (uint16_t)(((s2 & 0x3F3Fu) & ~hi) |
                               ((((s4 >> 4) & 0x0F0Fu) | ((s2 & 0xC0C0u) >> 2)) & hi));
    sc[0] = (uint8_t)(aux0 & 0xFFu);
    sc[1] = (uint8_t)((aux0 >> 8) & 0xFFu);
    m[0] = (uint8_t)(aux1 & 0xFFu);
    m[1] = (uint8_t)((aux1 >> 8) & 0xFFu);
}

// Q4_K q8_1 lane dot: per-thread partial over 16-elem sub-chunks
// (t = c & 15 within each 256-block b = c >> 4). Same integer arithmetic
// and op order as ggml vec_dot_q4_K_q8_1 (vmmq arm).
__device__ float fq_dot_q4_k_q8(const uint8_t* __restrict__ wr,
                               const uint8_t* __restrict__ aqb,
                               int K, int tid, int nthreads)
{
    int nsub = K / 16;
    float acc = 0.0f;
    for (int c = tid; c < nsub; c += nthreads)
    {
        int b = c >> 4;
        int t = c & 15;
        int bo = 2 * (t >> 2);
        int qo = t & 3;
        const uint8_t* blk = wr + (size_t)b * FQ_Q4_K_BYTES;
        __half dmh[2];
        memcpy(dmh, blk, 4);
        float dm0 = __half2float(dmh[0]);
        float dm1 = __half2float(dmh[1]);
        uint8_t sc[2], m[2];
        fq_q4k_scales(blk + 4, bo >> 1, sc, m);
        uint32_t v0, v1;
        memcpy(&v0, blk + 16 + 16 * bo + 4 * qo, 4);
        memcpy(&v1, blk + 16 + 16 * bo + 4 * qo + 16, 4);
        float sumf_d = 0.0f, sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            uint32_t v0i = (v0 >> (4 * i)) & 0x0F0F0F0Fu;
            uint32_t v1i = (v1 >> (4 * i)) & 0x0F0F0F0Fu;
            int ab = b * 8 + bo + i;
            const uint8_t* aqb_blk = aqb + (size_t)ab * FQ_Q8_1_BYTES;
            __half dah;
            memcpy(&dah, aqb_blk, 2);
            float d8 = __half2float(dah);
            int32_t u0, u1;
            memcpy(&u0, aqb_blk + 4 + 4 * qo, 4);
            memcpy(&u1, aqb_blk + 4 + 16 + 4 * qo, 4);
            int dot1 = fq_dp4a((int32_t)v0i, u0, 0);
            dot1 = fq_dp4a((int32_t)v1i, u1, dot1);
            int32_t su0 = u0, su1 = u1;
            int dot2 = fq_dp4a(0x01010101, su0, 0);
            dot2 = fq_dp4a(0x01010101, su1, dot2);
            sumf_d = __fadd_rn(sumf_d, __fmul_rn(d8, (float)(dot1 * sc[i])));
            sumf_m = __fadd_rn(sumf_m, __fmul_rn(d8, (float)(dot2 * m[i])));
        }
        acc = __fadd_rn(acc, __fadd_rn(__fmul_rn(dm0, sumf_d), -__fmul_rn(dm1, sumf_m)));
    }
    return acc;
}

// Q5_K q8_1 lane dot: same grid as Q4_K plus the high-bit plane
// (verbatim ggml vec_dot_q5_K_q8_1 vmmq arm).
__device__ float fq_dot_q5_k_q8(const uint8_t* __restrict__ wr,
                               const uint8_t* __restrict__ aqb,
                               int K, int tid, int nthreads)
{
    int nsub = K / 16;
    float acc = 0.0f;
    for (int c = tid; c < nsub; c += nthreads)
    {
        int b = c >> 4;
        int t = c & 15;
        int bo = 2 * (t >> 2);
        int qo = t & 3;
        const uint8_t* blk = wr + (size_t)b * FQ_Q5_K_BYTES;
        __half dmh[2];
        memcpy(dmh, blk, 4);
        float dm0 = __half2float(dmh[0]);
        float dm1 = __half2float(dmh[1]);
        uint8_t sc[2], m[2];
        fq_q4k_scales(blk + 4, bo >> 1, sc, m);
        uint32_t vl0, vl1, vh0, vh1;
        memcpy(&vl0, blk + 48 + 16 * bo + 4 * qo, 4);
        memcpy(&vl1, blk + 48 + 16 * bo + 4 * qo + 16, 4);
        memcpy(&vh0, blk + 16 + 4 * qo, 4);
        memcpy(&vh1, blk + 16 + 4 * qo + 16, 4);
        vh0 >>= bo;
        vh1 >>= bo;
        float sumf_d = 0.0f, sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            uint32_t vl0i = (vl0 >> (4 * i)) & 0x0F0F0F0Fu;
            uint32_t vl1i = (vl1 >> (4 * i)) & 0x0F0F0F0Fu;
            uint32_t vh0i = ((vh0 >> i) << 4) & 0x10101010u;
            uint32_t vh1i = ((vh1 >> i) << 4) & 0x10101010u;
            uint32_t v0i = vl0i | vh0i;
            uint32_t v1i = vl1i | vh1i;
            int ab = b * 8 + bo + i;
            const uint8_t* aqb_blk = aqb + (size_t)ab * FQ_Q8_1_BYTES;
            __half dah;
            memcpy(&dah, aqb_blk, 2);
            float d8 = __half2float(dah);
            int32_t u0, u1;
            memcpy(&u0, aqb_blk + 4 + 4 * qo, 4);
            memcpy(&u1, aqb_blk + 4 + 16 + 4 * qo, 4);
            int dot1 = fq_dp4a((int32_t)v0i, u0, 0);
            dot1 = fq_dp4a((int32_t)v1i, u1, dot1);
            int dot2 = fq_dp4a(0x01010101, u0, 0);
            dot2 = fq_dp4a(0x01010101, u1, dot2);
            sumf_d = __fadd_rn(sumf_d, __fmul_rn(d8, (float)(dot1 * sc[i])));
            sumf_m = __fadd_rn(sumf_m, __fmul_rn(d8, (float)(dot2 * m[i])));
        }
        acc = __fadd_rn(acc, __fadd_rn(__fmul_rn(dm0, sumf_d), -__fmul_rn(dm1, sumf_m)));
    }
    return acc;
}

// Q6_K q8_1 lane dot: per-thread partial over 8-elem sub-chunks
// (iqs = c & 31 within each 256-block b = c >> 5). Verbatim ggml
// vec_dot_q6_K_q8_1 (mmvq arm).
__device__ float fq_dot_q6_k_q8(const uint8_t* __restrict__ wr,
                               const uint8_t* __restrict__ aqb,
                               int K, int tid, int nthreads)
{
    int nsub = K / 8;
    float acc = 0.0f;
    for (int c = tid; c < nsub; c += nthreads)
    {
        int b = c >> 5;
        int iqs = c & 31;
        const uint8_t* blk = wr + (size_t)b * FQ_Q6_K_BYTES;
        int bq8_off = 4 * (iqs / 16) + (iqs % 16) / 8;
        int sc_off = 8 * (iqs / 16) + (iqs % 16) / 4;
        int vh_shift = 2 * ((iqs % 16) / 8);
        uint32_t vl, vh;
        memcpy(&vl, blk + 4 * iqs, 4);
        memcpy(&vh, blk + 128 + 4 * (8 * (iqs / 16) + iqs % 8), 4);
        vh >>= vh_shift;
        const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192 + sc_off);
        __half dh;
        memcpy(&dh, blk + 208, 2);
        float d = __half2float(dh);
        float sumf = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++)
        {
            int scc = (int)scales[4 * i];
            uint32_t vil = (vl >> (4 * i)) & 0x0F0F0F0Fu;
            uint32_t vih = ((vh >> (4 * i)) << 4) & 0x30303030u;
            uint32_t vi = fq_vsub4(vil | vih, 0x20202020u);
            int ab = b * 8 + bq8_off + 2 * i;
            const uint8_t* aqb_blk = aqb + (size_t)ab * FQ_Q8_1_BYTES;
            __half dah;
            memcpy(&dah, aqb_blk, 2);
            float d8 = __half2float(dah);
            int32_t u;
            memcpy(&u, aqb_blk + 4 + 4 * (iqs % 8), 4);
            sumf = __fadd_rn(sumf, __fmul_rn(d8, (float)(fq_dp4a((int32_t)vi, u, 0) * scc)));
        }
        acc = __fadd_rn(acc, __fmul_rn(d, sumf));
    }
    return acc;
}

// IQ4_XS codebook as two register-held u64 (bytes = verbatim kvalues_iq4nl
// as u8: P0 = bytes 0..7, P1 = bytes 8..15). Register shifts replace the 32
// divergent L1 loads per chunk below; the extracted bytes (hence the dp4a
// chain and every downstream value) are bitwise unchanged.
#define FQ_IQ4XS_LUT0 0xF6EADDCFBFAD9881ULL
#define FQ_IQ4XS_LUT1 0x7159453526190D01ULL
__device__ __forceinline__ uint32_t fq_iq4xs_lut(uint32_t idx)
{
    uint64_t p = (idx < 8) ? FQ_IQ4XS_LUT0 : FQ_IQ4XS_LUT1;
    return (uint32_t)((p >> (8 * (idx & 7))) & 0xFFu);
}

// IQ4_XS q8_1 lane dot: per-thread partial over 32-elem chunks
// (iqs = 4 * (c & 7) within each 256-block b = c >> 3). Verbatim ggml
// vec_dot_iq4_xs_q8_1; the codebook comes from the register LUT above.
__device__ float fq_dot_iq4_xs_q8(const uint8_t* __restrict__ wr,
                                 const uint8_t* __restrict__ aqb,
                                 int K, int tid, int nthreads)
{
    int nchunks = K / 32;
    float acc = 0.0f;
    for (int c = tid; c < nchunks; c += nthreads)
    {
        int b = c >> 3;
        int sub = c & 7;
        int iqs = 4 * sub;
        const uint8_t* blk = wr + (size_t)b * FQ_IQ4_XS_BYTES;
        __half dwh;
        memcpy(&dwh, blk, 2);
        float dw = __half2float(dwh);
        uint16_t sh;
        memcpy(&sh, blk + 2, 2);
        uint8_t sl = *(blk + 4 + sub / 2);
        int ls = (int)(((sl >> (4 * (sub % 2))) & 0xF) | ((((uint32_t)sh >> (2 * sub)) & 3u) << 4));
        const uint8_t* qs = blk + 8;
        const uint8_t* aqbb = aqb + (size_t)(b * 8 + sub) * FQ_Q8_1_BYTES;
        __half dah;
        memcpy(&dah, aqbb, 2);
        float da = __half2float(dah);
        const uint8_t* aq = aqbb + 4;
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            uint32_t aux;
            memcpy(&aux, qs + 4 * (iqs + j), 4);
            uint32_t vx = 0, vy = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++)
            {
                // get_int_from_table_16 even/odd de-interleave: x takes the
                // even nibbles (low of each byte), y the odd nibbles (high).
                vx |= fq_iq4xs_lut((aux >> (8 * k)) & 0xFu) << (8 * k);
                vy |= fq_iq4xs_lut((aux >> (8 * k + 4)) & 0xFu) << (8 * k);
            }
            int32_t u0, u1;
            memcpy(&u0, aq + 4 * j, 4);
            memcpy(&u1, aq + 4 * (j + 4), 4);
            sumi = fq_dp4a((int32_t)vx, u0, sumi);
            sumi = fq_dp4a((int32_t)vy, u1, sumi);
        }
        sumi *= ls - 32;
        acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(dw, da), (float)sumi));
    }
    return acc;
}

}  // namespace
