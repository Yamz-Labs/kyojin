// SPDX-License-Identifier: MIT
// Kyojin: routed (MoE) F16-WMMA prefill expert GEMMs for HIP gfx1151 (rocm/routed_f16). Adapted from Gufo
// (github.com/gufo-org/gufo, commit 07e1191, MIT, Copyright (c) 2026 gufo contributors),
// src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp: RoutedPairF16Kernel (Q4_K / Q5_K gate+up with SwiGLU),
// RoutedDownF16Kernel (Q5_1 / Q8_0 down), RoutedPadBoundsKernel, RoutedScatterKernel and the small device helpers they
// use (Wmma, SigmoidF, CodesToHalves, SwapHalves, GatherBit, SpreadHighBits, the GGUF block layouts). See
// THIRD_PARTY_NOTICES.md.
// Changes: weights through per-expert device pointer tables (Kyojin keeps each expert in its own allocation) instead of
// one contiguous [E, M, K] tensor; only the 128-token-tile kernels are kept; the host side is a torch op
// (fq_routed_f16_moe) that takes the tile list built by the caller. The kernel bodies are otherwise unchanged and depend
// on the gfx1151 (RDNA3.5, wave32) WMMA lane layout.
// fq_routed_f16_combine is the routing-weighted top-k reduction of the down rows, after Gufo's MoeEpilogueVec4Kernel<__half> (same
// kernels.hip.cpp; the shared-expert term is left to the caller): one pass in registers instead of torch's F32 broadcast multiply + sum.
// fq_routed_f16_combine_shared adds that shared-expert term (and the shared gate's dot product) to the same pass, as Gufo's MoeEpilogueVec4Kernel does.
#if defined(USE_ROCM)

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

namespace kyo_routed_f16 {
// gfx1151 only (WMMA lane layout, permlanex16); other device targets get no kernel and the Python side never routes there.
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
namespace {
constexpr unsigned kThreads = 256;
enum class WeightType : std::uint32_t { kQ5_1 = 7, kQ8_0 = 8, kQ4_K = 12, kQ5_K = 13 };
__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");
// Routed expert GEMMs: assignment rows are compacted by expert with every
// bucket padded to a 16-row tile (`pad_bounds`), so a token tile never
// straddles experts; `rows_out` maps a compact row to its (token, slot)
// output row or -1 for padding.
constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

// Routed F16 WMMA expert GEMM. The int8 kernel above pays a float epilogue
// and an activation-sum correction every K block because the per-32 scales
// of both operands sit outside the integer dot product; here the weights are
// dequantized to F16 right after the LDS read (each wave decodes only its own
// sixteen rows) and the activations are F16 rows, so the matrix core
// accumulates the whole K extent in F32 with no per-block work. The codes
// stay packed in LDS (4 bits for Q4_K, 4 + 1 for Q5_1), which keeps a
// two-K-block stage at 11-12 KB and five blocks resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or F16 with the SwiGLU applied when `out_half`
// is given: the up projection then writes the down projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K   ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1 ? (k / 32) * sizeof(Q5_1Block)
                                      : (k / 32) * sizeof(Q8_0Block);
}

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Lane i receives lane i ^ 16's value (XorLane(v, 16)) without tying the
/// result to v: every lane is active, so the instruction's old operand is
/// never read and needs no copy of v.
__device__ __forceinline__ std::uint32_t SwapHalves(std::uint32_t v) {
  std::uint32_t old;
  asm("" : "=v"(old));
  return static_cast<std::uint32_t>(
      __builtin_amdgcn_permlanex16(static_cast<int>(old), static_cast<int>(v),
                                   0x76543210, 0xfedcba98, false, false));
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

// Wide routed down projection (Q5_1 / Q8_0 weights, F16 rows in
// and out) over 128-token tiles: RoutedF16GEMMKernel's arithmetic, laid out
// like the paired gate/up waves. Each wave owns sixteen rows and keeps their
// codes in registers; activations are the only LDS stage.
//
// A gfx1151 WMMA reads the A operand of row r from one lane only: lane r for
// even rows, lane r + 16 for odd ones (the other copy is ignored). So each
// lane decodes one K16 half of its row's K32 block, the half the WMMA
// reading it needs first (lo in a reading lane, hi in its partner), and one
// v_permlanex16 per dword hands every reading lane its other half. Every
// weight is decoded once per wave instead of twice, with no selects.
//
// Every output keeps RoutedF16GEMMKernel's values exactly: the same F16
// weights (code + magic, then one packed FMA with the same scale and bias),
// the same F16 activations, the same chain of two K16 WMMAs per K32 block in
// K order from zero, and one round-to-nearest F16 store. Rows therefore do
// not depend on the tile height or on the other rows in the pass.
//
// grid (ceil(m / 128), tiles): `tiles[y]` is expert | (128-token tile << 16).
template<WeightType kType>
__launch_bounds__(256) __global__
    void RoutedDownF16Kernel(const void* const* __restrict__ w_ptrs,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ rows_in,
                             const std::int32_t* __restrict__ rows_out,
                             __half* __restrict__ out, std::size_t m,
                             std::size_t k) {
  constexpr int BN = 128;
  constexpr int BK = 2;  // K32 blocks per stage
  constexpr int kTokTiles = BN / 16;
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr int kBlockBytes = kQ5 ? 24 : 34;
  // Activation stage: [kb][16-element quarter][token][16 B], as in
  // RoutedF16GEMMKernel, one padding slot per quarter plane.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // Epilogue: half the tile's F16 results at a time, rows padded to keep
  // 16-byte alignment and spread the 2-byte writes across banks.
  constexpr int kOutStride = 128 + 8;
  constexpr int kEpiTokens = 64;
  constexpr int kEpiBytes = kEpiTokens * kOutStride * 2;
  constexpr int kLdsBytes = kActBytes > kEpiBytes ? kActBytes : kEpiBytes;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_act = reinterpret_cast<uint4*>(lds);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = (k / 32) * kBlockBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int r_block = static_cast<int>(blockIdx.x) * 128;
  // The K16 half this lane decodes: lo where the WMMA reads A, hi in the
  // partner lane.
  const int f_c = half_id ^ (sub_lane & 1);
  const int row = r_block + (wave_id * 16) + sub_lane;
  const bool row_live = row < m_i;
  const std::uint8_t* w_row =
      static_cast<const std::uint8_t*>(w_ptrs[expert]) +
      static_cast<std::size_t>(row_live ? row : m_i - 1) * row_bytes;

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; four per thread.
  constexpr int kActFetch = BN * BK * 4 / 256;
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src =
        c_row < bucket_rows ? rows_in[bucket_begin + c_row] : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    a_slot[i] = (sub * kActStride) + t;
  }

  // Raw weights (scale word and 16 code bytes per K32 block) in two sets:
  // a stage computes from one while the next stage loads into the other.
  // Activations pass through registers into LDS before each stage.
  uint4 w_codes[2][BK];
  std::uint32_t w_head[2][BK];  ///< Q8_0: d; Q5_1: d | m
  std::uint32_t w_high[2][BK];  ///< Q5_1: the block's high-bit word
  // Padding tokens keep whatever their slots held: their columns are never
  // stored, and WMMA columns do not mix.
  uint4 a_data[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i)
    a_data[i] = make_uint4(0u, 0u, 0u, 0u);
  const auto fetch_stage = [&]<int S>(int kb0) {
#pragma unroll
    for (int b = 0; b < BK; ++b) {
      const std::uint8_t* blk = w_row + ((kb0 + b) * kBlockBytes);
      if constexpr (kQ5) {
        const uint2 head = *reinterpret_cast<const uint2*>(blk);
        w_head[S][b] = head.x;
        w_high[S][b] = head.y;
        w_codes[S][b] = *reinterpret_cast<const uint4*>(blk + 8);
      } else {
        w_head[S][b] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&w_codes[S][b], blk + 2 + (16 * f_c), 16);
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_src[i] != nullptr)
        a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32));
    }
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j)
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

  constexpr float kMagic = kQ5 ? -1024.0F : -1152.0F;
  const __half2 magic = __floats2half2_rn(kMagic, kMagic);
  const auto compute_stage = [&]<int S>() {
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      // This lane's K16 half of the row's block, as RoutedF16GEMMKernel
      // forms it: (scale, bias) = (d, 0) for Q8_0, (d, m) for Q5_1.
      const std::uint32_t scale_bias =
          kQ5 ? w_head[S][kb] : (row_live ? w_head[S][kb] : 0U);
      const __half2 sb = __builtin_bit_cast(__half2, scale_bias);
      const __half2 scale2 = __low2half2(sb);
      const __half2 bias2 = __high2half2(sb);
      const uint4 c = w_codes[S][kb];
      const std::uint32_t words[4] = {c.x, c.y, c.z, c.w};
      __half2 h[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        std::uint32_t nib;
        if constexpr (kQ5) {
          nib = ((words[i] >> (4U * f_c)) & 0x0F0F0F0FU) |
                SpreadHighBits((w_high[S][kb] >> (16 * f_c + 4 * i)) & 0xFU);
        } else {
          nib = words[i] ^ 0x80808080U;
        }
        CodesToHalves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
      }
      using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
      v8u own;
      __builtin_memcpy(&own, &h[0], 32);
      v8u peer;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        peer[i] = SwapHalves(own[i]);
      const v16h a_lo = __builtin_bit_cast(v16h, own);
      const v16h a_hi = __builtin_bit_cast(v16h, peer);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        // One token tile's LDS fragments live at a time (a compiler
        // barrier only): hoisting all eight spills registers.
        asm volatile("" ::: "memory");
        if (j >= live_tok_tiles)
          continue;
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
          b[q] = frag[q * kActStride];
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
        acc[j] = Wmma(a_lo, b_lo, acc[j]);
        acc[j] = Wmma(a_hi, b_hi, acc[j]);
      }
    }
  };

  const auto run_stage = [&]<int S>(int kb0) {
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      s_act[a_slot[i]] = a_data[i];
    __syncthreads();
    if (kb0 + BK < num_kb)
      fetch_stage.template operator()<1 - S>(kb0 + BK);
    compute_stage.template operator()<S>();
    __syncthreads();
  };
  fetch_stage.template operator()<0>(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += 2 * BK) {
    run_stage.template operator()<0>(kb0);
    if (kb0 + BK < num_kb)
      run_stage.template operator()<1>(kb0 + BK);
  }

  // Lane (token n, rows 2i + half) holds acc[j][i]: stage F16 values per
  // half tile, then store each token's 128 rows as 16-byte chunks.
  __half* scratch = reinterpret_cast<__half*>(lds);
#pragma unroll
  for (int part = 0; part < BN / kEpiTokens; ++part) {
    if (part * (kEpiTokens / 16) >= live_tok_tiles)
      break;
    if (part > 0)
      __syncthreads();
#pragma unroll
    for (int jj = 0; jj < kEpiTokens / 16; ++jj) {
      const int j = part * (kEpiTokens / 16) + jj;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        scratch[((jj * 16) + sub_lane) * kOutStride + (wave_id * 16) + (2 * i) +
                half_id] = __float2half(acc[j][i]);
    }
    __syncthreads();
#pragma unroll
    for (int unit = 0; unit < kEpiTokens * 16 / 256; ++unit) {
      const int flat = (unit * 256) + tid;
      const int tr = flat / 16;
      const int chunk = flat % 16;
      const int t = t_local + (part * kEpiTokens) + tr;
      const int r = r_block + (chunk * 8);
      if (t >= bucket_rows || r >= m_i)
        continue;
      const std::int32_t dst = rows_out[bucket_begin + t];
      if (dst < 0)
        continue;
      const __half* src = scratch + (tr * kOutStride) + (chunk * 8);
      __half* o = out + (static_cast<std::size_t>(dst) * m) + r;
      if (m % 8 == 0) {
        *reinterpret_cast<uint4*>(o) = *reinterpret_cast<const uint4*>(src);
      } else {
        for (int e = 0; e < 8 && r + e < m_i; ++e)
          o[e] = src[e];
      }
    }
  }
}

// Paired K-quant gate/up projection with SwiGLU over 128-token tiles (F16
// rows in and out). Each wave owns eight gate rows and their matching up
// rows as one 16-row WMMA tile, with the codes in registers; activations
// are the only LDS stage. A stage is one 32-byte nibble group: K32 blocks
// 2g (low nibbles) and 2g + 1 (high), whose 16-byte halves each hold one
// K16 fragment of both blocks.
//
// As in RoutedDownF16Kernel, each lane decodes the K16 fragment its WMMA
// reads first (lo in the lane the hardware reads, hi in its partner) and
// one v_permlanex16 per dword supplies the other. A superblock's codes and
// header are loaded raw one stage ahead and decoded when their stage is
// committed, so no stage waits on the loads it has just issued.
//
// Every output keeps the previous paired route's arithmetic: the same F16
// weights (code * (d * sc) - dmin * mn, rounded once to F16 pairs), the same
// chain of two K16 WMMAs per K32 block from zero in K order, and the same
// SwiGLU rounding, so a row matches the separate gate and up projections at
// every tile height.
//
// grid (ceil(m / 64), tiles): `tiles[y]` is expert | (128-token tile << 16).
template<WeightType kType>
__launch_bounds__(256) __global__
    void RoutedPairF16Kernel(const void* const* __restrict__ w_gate,
                             const void* const* __restrict__ w_up,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ rows_in,
                             const std::int32_t* __restrict__ rows_out,
                             __half* __restrict__ out, std::size_t m,
                             std::size_t k) {
  constexpr int BN = 128;
  constexpr int BK = 2;  // K32 blocks per stage: one nibble group
  constexpr int kTokTiles = BN / 16;
  constexpr int kRows = 64;  // output rows (gate/up pairs) per block
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr std::size_t kBlockBytes = kQ5K ? kQ5KBlockBytes : sizeof(Q4KBlock);
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  constexpr int kPairStride = kRows + 2;
  constexpr int kEpilogueBytes = BN * kPairStride * 2;
  constexpr int kLdsBytes =
      kActBytes > kEpilogueBytes ? kActBytes : kEpilogueBytes;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_act = reinterpret_cast<uint4*>(lds);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_sb = static_cast<int>(k / 256);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = (k / 256) * kBlockBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;
  // Tile rows 0-7 are gate rows, 8-15 the matching up rows.
  const int row = r_block + (wave_id * 8) + (sub_lane % 8);
  const bool row_live = row < m_i;
  const int f_c = half_id ^ (sub_lane & 1);
  const std::uint8_t* w_row =
      static_cast<const std::uint8_t*>(sub_lane < 8 ? w_gate[expert]
                                                    : w_up[expert]) +
      static_cast<std::size_t>(row_live ? row : m_i - 1) * row_bytes;

  constexpr int kActFetch = BN * BK * 4 / 256;
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src =
        c_row < bucket_rows ? rows_in[bucket_begin + c_row] : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    a_slot[i] = (sub * kActStride) + t;
  }

  // A superblock's raw header, high bits and this lane's code chunks,
  // loaded with its first stage's fetch and decoded at that stage's commit.
  uint4 raw_header;
  uint4 raw_codes[4];  ///< chunk f_c of groups 0-3
  uint4 raw_qh[2];     ///< Q5_K: the 32 high-bit bytes
  // The committed superblock: header, nibble-packed codes per group.
  uint4 header = make_uint4(0u, 0u, 0u, 0u);
  uint4 codes[4];
  uint4 qh[2];
  uint4 a_data[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i)
    a_data[i] = make_uint4(0u, 0u, 0u, 0u);

  const auto fetch_stage = [&]<int G>(int sb) {
    if constexpr (G == 0) {
      const std::uint8_t* blk = w_row + (sb * kBlockBytes);
      const auto* chunks = reinterpret_cast<const uint4*>(blk);
      constexpr int kCodeChunk = kQ5K ? 3 : 1;
      raw_header = chunks[0];
      if constexpr (kQ5K) {
        raw_qh[0] = chunks[1];
        raw_qh[1] = chunks[2];
      }
#pragma unroll
      for (int g = 0; g < 4; ++g)
        raw_codes[g] = chunks[kCodeChunk + (g * 2) + f_c];
    }
    const int kb0 = (sb * 8) + (G * 2);
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_src[i] != nullptr)
        a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32));
    }
  };

  // The committed stage's (scale, bias) for K32 block kb of the group, and
  // Q5_K's high-bit word.
  std::uint32_t own_scale = 0;  ///< block 2G + f_c
  std::uint32_t own_high = 0;   ///< Q5_K: block 2G + f_c
  const auto commit_stage = [&]<int G>() {
    if constexpr (G == 0) {
      header = raw_header;
#pragma unroll
      for (int g = 0; g < 4; ++g)
        codes[g] = raw_codes[g];
      if constexpr (kQ5K) {
        qh[0] = raw_qh[0];
        qh[1] = raw_qh[1];
      }
    }
    const int sb32 = (2 * G) + f_c;
    // The first four scale/min bytes live in y/z; the others' low nibbles
    // share w, their high two bits reuse those y/z bytes.
    const unsigned shift = 8U * (static_cast<unsigned>(sb32) & 3U);
    const std::uint32_t scale_byte = header.y >> shift;
    const std::uint32_t min_byte = header.z >> shift;
    std::uint32_t sc;
    std::uint32_t mn;
    if (sb32 < 4) {
      sc = scale_byte & 0x3FU;
      mn = min_byte & 0x3FU;
    } else {
      const std::uint32_t packed = header.w >> shift;
      sc = (packed & 0x0FU) | ((scale_byte & 0xC0U) >> 2U);
      mn = ((packed >> 4U) & 0x0FU) | ((min_byte & 0xC0U) >> 2U);
    }
    const __half2 dm = __builtin_bit_cast(__half2, header.x);
    const float scale =
        row_live ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
    const float offset =
        row_live ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
    own_scale =
        __builtin_bit_cast(std::uint32_t, __floats2half2_rn(scale, -offset));
    if constexpr (kQ5K) {
      // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
      const std::uint32_t q[8] = {qh[0].x, qh[0].y, qh[0].z, qh[0].w,
                                  qh[1].x, qh[1].y, qh[1].z, qh[1].w};
      std::uint32_t high = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        high |= GatherBit(q[i], sb32) << (4 * i);
      own_high = high;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      s_act[a_slot[i]] = a_data[i];
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j)
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

  const __half2 magic = __floats2half2_rn(-1024.0F, -1024.0F);
  const auto compute_stage = [&]<int G>() {
    const uint4 c = codes[G];
    const std::uint32_t words[4] = {c.x, c.y, c.z, c.w};
    const std::uint32_t peer_scale = SwapHalves(own_scale);
    const std::uint32_t peer_high = kQ5K ? SwapHalves(own_high) : 0U;
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      const __half2 sb =
          __builtin_bit_cast(__half2, f_c == kb ? own_scale : peer_scale);
      const __half2 scale2 = __low2half2(sb);
      const __half2 bias2 = __high2half2(sb);
      const std::uint32_t high =
          kQ5K ? ((f_c == kb ? own_high : peer_high) >> (16U * f_c)) : 0U;
      __half2 h[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        std::uint32_t nib = (words[i] >> (4U * kb)) & 0x0F0F0F0FU;
        if constexpr (kQ5K)
          nib |= SpreadHighBits((high >> (4 * i)) & 0xFU);
        CodesToHalves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
      }
      using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
      v8u own;
      __builtin_memcpy(&own, &h[0], 32);
      v8u peer;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        peer[i] = SwapHalves(own[i]);
      const v16h a_lo = __builtin_bit_cast(v16h, own);
      const v16h a_hi = __builtin_bit_cast(v16h, peer);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        // One token tile's LDS fragments live at a time (a compiler
        // barrier only): hoisting all eight spills registers.
        asm volatile("" ::: "memory");
        if (j >= live_tok_tiles)
          continue;
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
          b[q] = frag[q * kActStride];
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
        acc[j] = Wmma(a_lo, b_lo, acc[j]);
        acc[j] = Wmma(a_hi, b_hi, acc[j]);
      }
    }
  };

  const auto run_stage = [&]<int G>(int sb) {
    commit_stage.template operator()<G>();
    __syncthreads();
    if constexpr (G < 3)
      fetch_stage.template operator()<G + 1>(sb);
    else if (sb + 1 < num_sb)
      fetch_stage.template operator()<0>(sb + 1);
    compute_stage.template operator()<G>();
    __syncthreads();
  };
  fetch_stage.template operator()<0>(0);
  for (int sb = 0; sb < num_sb; ++sb) {
    run_stage.template operator()<0>(sb);
    run_stage.template operator()<1>(sb);
    run_stage.template operator()<2>(sb);
    run_stage.template operator()<3>(sb);
  }

  // Lane (token n, rows 2l + half) holds gate rows in acc[j][l] and the
  // matching up rows in acc[j][l + 4]. SwiGLU in registers, then all F16
  // results are staged once so stores cover the full 64-column output row.
  __half* scratch = reinterpret_cast<__half*>(lds);
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    if (j >= live_tok_tiles)
      continue;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      float product = acc[j][l + 4] * acc[j][l];
      asm volatile("" : "+v"(product));
      float value = product * SigmoidF(acc[j][l]);
      asm volatile("" : "+v"(value));
      scratch[(j * 16 + sub_lane) * kPairStride + wave_id * 8 + 2 * l +
              half_id] = __float2half(value);
    }
  }
  __syncthreads();
#pragma unroll
  for (int unit = 0; unit < BN * kRows / (2 * 256); ++unit) {
    const int flat = (unit * 256 + tid) * 2;
    const int t = t_local + flat / kRows;
    const int r = r_block + flat % kRows;
    if (t < bucket_rows && r < m_i) {
      const std::int32_t dst = rows_out[bucket_begin + t];
      if (dst >= 0) {
        const auto offset = static_cast<std::size_t>(dst) * m + r;
        const int src = (flat / kRows) * kPairStride + flat % kRows;
        if (m % 2 == 0 && r + 1 < m_i) {
          *reinterpret_cast<__half2*>(out + offset) =
              *reinterpret_cast<const __half2*>(scratch + src);
        } else {
          out[offset] = scratch[src];
          if (r + 1 < m_i)
            out[offset + 1] = scratch[src + 1];
        }
      }
    }
  }
}

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::int32_t* __restrict__ rows_token,
                                      std::int32_t* __restrict__ rows_slot,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
  __syncthreads();
  // Scatter overwrites every occupied row. Only the padding can retain a
  // previous batch's assignments; clear it while the bucket bounds are hot.
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    for (std::uint32_t row = counts[e];
         row < static_cast<std::uint32_t>(padded[e]); ++row) {
      const auto at = pad_bounds[e] + row;
      rows_token[at] = -1;
      rows_slot[at] = -1;
    }
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}

// out[t, i] (F32) = sum over s of weights[t, s] * float(rows[t, s, i]), s in order (Gufo MoeEpilogueVec4Kernel<__half>, without the shared
// expert). Four adjacent lanes per thread, grid (T, ceil(K / (4 * kThreads))).
__global__ void RoutedCombineKernel(const __half* __restrict__ rows, const float* __restrict__ weights, float* __restrict__ out,
                                    std::uint32_t top_k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) {
    return;
  }
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  const __half* src = rows + static_cast<std::size_t>(t) * top_k * dim + i;
  for (std::uint32_t s = 0; s < top_k; ++s) {
    const float w = weights[static_cast<std::size_t>(t) * top_k + s];
    const uint2 raw = *reinterpret_cast<const uint2*>(src + static_cast<std::size_t>(s) * dim);
    const __half2 lo = *reinterpret_cast<const __half2*>(&raw.x);
    const __half2 hi = *reinterpret_cast<const __half2*>(&raw.y);
    a0 = __fadd_rn(a0, __fmul_rn(w, __low2float(lo)));
    a1 = __fadd_rn(a1, __fmul_rn(w, __high2float(lo)));
    a2 = __fadd_rn(a2, __fmul_rn(w, __low2float(hi)));
    a3 = __fadd_rn(a3, __fmul_rn(w, __high2float(hi)));
  }
  *reinterpret_cast<float4*>(out + static_cast<std::size_t>(t) * dim + i) = float4{a0, a1, a2, a3};
}

// RoutedCombineKernel with the gated shared expert fused in (Gufo MoeEpilogueVec4Kernel<__half> with its shared-expert term, the gate taken as in
// Kyojin's torch path): out[t, :] = sum_s weights[t, s] * rows[t, s, :] + shared[t, :] * sigmoid(round_f16(x[t, :] . w)). One block per token:
// the gate dot product (F32 accumulation, rounded to F16 like the linear's output the torch path widens), then the top-k sum in registers, then the
// shared term, so the F32 widen of the shared output, the gate GEMM (N = 1), the sigmoid / multiply / add passes and one F32 round trip of
// the output are gone.
__global__ void RoutedCombineSharedKernel(const __half* __restrict__ rows, const float* __restrict__ weights,
                                          const __half* __restrict__ shared, const __half* __restrict__ x,
                                          const __half* __restrict__ w, float* __restrict__ out, std::uint32_t top_k,
                                          std::uint32_t dim, std::size_t x_pitch) {
  __shared__ float part[kThreads / 32];
  __shared__ float s_gate;
  const std::size_t t = blockIdx.x;
  // gate: x[t, :] . w, four adjacent lanes per thread
  float dot = 0.0f;
  const __half* xr = x + t * x_pitch;
  for (std::uint32_t i = threadIdx.x * 4; i < dim; i += kThreads * 4) {
    const uint2 xr_raw = *reinterpret_cast<const uint2*>(xr + i);
    const uint2 w_raw = *reinterpret_cast<const uint2*>(w + i);
    const __half2 x0 = *reinterpret_cast<const __half2*>(&xr_raw.x), x1 = *reinterpret_cast<const __half2*>(&xr_raw.y);
    const __half2 w0 = *reinterpret_cast<const __half2*>(&w_raw.x), w1 = *reinterpret_cast<const __half2*>(&w_raw.y);
    dot = __fmaf_rn(__low2float(x0), __low2float(w0), dot);
    dot = __fmaf_rn(__high2float(x0), __high2float(w0), dot);
    dot = __fmaf_rn(__low2float(x1), __low2float(w1), dot);
    dot = __fmaf_rn(__high2float(x1), __high2float(w1), dot);
  }
  for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor(dot, o);
  if ((threadIdx.x & 31u) == 0) part[threadIdx.x >> 5] = dot;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (unsigned q = 0; q < kThreads / 32; ++q) sum += part[q];
    const float gate = __half2float(__float2half_rn(sum));
    s_gate = 1.0f / (1.0f + expf(-gate));
  }
  __syncthreads();
  const float g = s_gate;
  for (std::uint32_t i = threadIdx.x * 4; i < dim; i += kThreads * 4) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const __half* src = rows + t * top_k * dim + i;
    for (std::uint32_t s = 0; s < top_k; ++s) {
      const float wt = weights[t * top_k + s];
      const uint2 raw = *reinterpret_cast<const uint2*>(src + static_cast<std::size_t>(s) * dim);
      const __half2 lo = *reinterpret_cast<const __half2*>(&raw.x);
      const __half2 hi = *reinterpret_cast<const __half2*>(&raw.y);
      a0 = __fadd_rn(a0, __fmul_rn(wt, __low2float(lo)));
      a1 = __fadd_rn(a1, __fmul_rn(wt, __high2float(lo)));
      a2 = __fadd_rn(a2, __fmul_rn(wt, __low2float(hi)));
      a3 = __fadd_rn(a3, __fmul_rn(wt, __high2float(hi)));
    }
    const uint2 sraw = *reinterpret_cast<const uint2*>(shared + t * dim + i);
    const __half2 slo = *reinterpret_cast<const __half2*>(&sraw.x);
    const __half2 shi = *reinterpret_cast<const __half2*>(&sraw.y);
    a0 = __fadd_rn(a0, __fmul_rn(__low2float(slo), g));
    a1 = __fadd_rn(a1, __fmul_rn(__high2float(slo), g));
    a2 = __fadd_rn(a2, __fmul_rn(__low2float(shi), g));
    a3 = __fadd_rn(a3, __fmul_rn(__high2float(shi), g));
    *reinterpret_cast<float4*>(out + t * dim + i) = float4{a0, a1, a2, a3};
  }
}
}  // namespace
#endif
}  // namespace kyo_routed_f16
// out[slot, :K] (F16, slot = t * top_k + s) = down[ids[t, s]] @ (silu(gate[ids[t, s]] @ x[t]) * (up[ids[t, s]] @ x[t])),
// F16 operands, F32 WMMA accumulation. x: half [T, K]; ids: int32 [T, top_k] (ids outside [0, E) are skipped); counts:
// int32 [E] = rows per expert; tiles: int32 [n_tiles], one per 128-row tile of an expert's bucket, expert | (tile << 16);
// gate_ptrs / up_ptrs / down_ptrs: int64 [E] device pointers to each expert's raw GGUF rows (gate/up: I rows of K / 256
// Q4_K or Q5_K blocks; down: K rows of I / 32 Q5_1 or Q8_0 blocks). Rows of a bucket are padded to 16.
void fq_routed_f16_moe(torch::Tensor x, torch::Tensor ids, torch::Tensor counts, torch::Tensor tiles,
                       torch::Tensor gate_ptrs, torch::Tensor up_ptrs, torch::Tensor down_ptrs, int64_t gate_fmt,
                       int64_t down_fmt, int64_t I, torch::Tensor out)
{
    using namespace kyo_routed_f16;
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.dtype() == torch::kHalf, "x must be half [T, K]");
    TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && ids.is_contiguous() && ids.dtype() == torch::kInt &&
                ids.size(0) == x.size(0), "ids must be int32 [T, top_k]");
    TORCH_CHECK(counts.is_cuda() && counts.dtype() == torch::kInt && counts.is_contiguous(), "counts must be int32 [E]");
    TORCH_CHECK(tiles.is_cuda() && tiles.dtype() == torch::kInt && tiles.is_contiguous() && tiles.numel() > 0,
                "tiles must be int32 [n_tiles]");
    for (auto & p : {gate_ptrs, up_ptrs, down_ptrs})
        TORCH_CHECK(p.is_cuda() && p.dtype() == torch::kLong && p.is_contiguous() && p.numel() == counts.numel(),
                    "weight pointer tables must be int64 [E]");
    const int64_t T = x.size(0), K = x.size(1), top_k = ids.size(1), E = counts.numel(), slots = T * top_k;
    TORCH_CHECK(E <= 1024 && E < 65536 && slots < ((int64_t) 1 << 31) / std::max<int64_t>(K, I), "shape too large");
    TORCH_CHECK(gate_fmt == 12 || gate_fmt == 13, "gate/up must be Q4_K or Q5_K");
    TORCH_CHECK(down_fmt == 7 || down_fmt == 8, "down must be Q5_1 or Q8_0");
    TORCH_CHECK(K % 256 == 0 && I % 64 == 0, "need K % 256 == 0 and I % 64 == 0");
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kHalf && out.is_contiguous() && out.dim() == 2 &&
                out.size(0) == slots && out.size(1) == K, "out must be half [T * top_k, K]");
    if (T == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto i32 = x.options().dtype(torch::kInt);
    const int64_t rows = slots + E * 15;    // 16-row padded buckets
    torch::Tensor pad_bounds = torch::empty({E + 1}, i32), cursors = torch::empty({E}, i32);
    torch::Tensor rows_token = torch::empty({rows}, i32), rows_slot = torch::empty({rows}, i32);
    torch::Tensor act = torch::empty({slots, I}, x.options());
    hipLaunchKernelGGL(RoutedPadBoundsKernel, dim3(1), dim3(1024), 0, stream, (const std::uint32_t *) counts.data_ptr(),
                       (std::int32_t *) pad_bounds.data_ptr(), (std::int32_t *) cursors.data_ptr(),
                       (std::int32_t *) rows_token.data_ptr(), (std::int32_t *) rows_slot.data_ptr(), (std::uint32_t) E);
    hipLaunchKernelGGL(RoutedScatterKernel, dim3((slots + 255) / 256), dim3(kThreads), 0, stream,
                       (const std::int32_t *) ids.data_ptr(), (const std::int32_t *) pad_bounds.data_ptr(),
                       (std::int32_t *) cursors.data_ptr(), (std::int32_t *) rows_token.data_ptr(),
                       (std::int32_t *) rows_slot.data_ptr(), (std::uint32_t) slots, (std::uint32_t) top_k);
    const unsigned n_tiles = (unsigned) tiles.numel();
    const auto gk = gate_fmt == 12 ? RoutedPairF16Kernel<WeightType::kQ4_K> : RoutedPairF16Kernel<WeightType::kQ5_K>;
    hipLaunchKernelGGL(gk, dim3((I + 63) / 64, n_tiles), dim3(kThreads), 0, stream,
                       (const void * const *) gate_ptrs.data_ptr(), (const void * const *) up_ptrs.data_ptr(),
                       (const __half *) x.data_ptr(), (const std::int32_t *) tiles.data_ptr(),
                       (const std::int32_t *) pad_bounds.data_ptr(), (const std::int32_t *) rows_token.data_ptr(),
                       (const std::int32_t *) rows_slot.data_ptr(), (__half *) act.data_ptr(), (std::size_t) I,
                       (std::size_t) K);
    const auto dk = down_fmt == 7 ? RoutedDownF16Kernel<WeightType::kQ5_1> : RoutedDownF16Kernel<WeightType::kQ8_0>;
    hipLaunchKernelGGL(dk, dim3((K + 127) / 128, n_tiles), dim3(kThreads), 0, stream,
                       (const void * const *) down_ptrs.data_ptr(), (const __half *) act.data_ptr(),
                       (const std::int32_t *) tiles.data_ptr(), (const std::int32_t *) pad_bounds.data_ptr(),
                       (const std::int32_t *) rows_slot.data_ptr(), (const std::int32_t *) rows_slot.data_ptr(),
                       (__half *) out.data_ptr(), (std::size_t) K, (std::size_t) I);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "fq_routed_f16_moe: launch failed");
}

// out[t, :] (F32) = sum_s weights[t, s] * rows[t * top_k + s, :]; rows: half [T * top_k, K] (the fq_routed_f16_moe output), weights: float [T, top_k].
void fq_routed_f16_combine(torch::Tensor rows, torch::Tensor weights, torch::Tensor out)
{
    using namespace kyo_routed_f16;
    const at::cuda::OptionalCUDAGuard device_guard(rows.device());
    TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 && rows.is_contiguous() && rows.dtype() == torch::kHalf, "rows must be half [T * top_k, K]");
    TORCH_CHECK(weights.is_cuda() && weights.dim() == 2 && weights.is_contiguous() && weights.dtype() == torch::kFloat, "weights must be float [T, top_k]");
    const int64_t T = weights.size(0), top_k = weights.size(1), K = rows.size(1);
    TORCH_CHECK(rows.size(0) == T * top_k && K % 4 == 0, "rows must be [T * top_k, K] with K % 4 == 0");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dtype() == torch::kFloat && out.dim() == 2 && out.size(0) == T && out.size(1) == K,
                "out must be float [T, K]");
    if (T == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(RoutedCombineKernel, dim3((unsigned) T, (unsigned) ((K / 4 + kThreads - 1) / kThreads)), dim3(kThreads), 0, stream,
                       (const __half *) rows.data_ptr(), (const float *) weights.data_ptr(), (float *) out.data_ptr(), (std::uint32_t) top_k,
                       (std::uint32_t) K);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "fq_routed_f16_combine: launch failed");
}


// out[t, :] (F32) = sum_s weights[t, s] * rows[t * top_k + s, :] + shared[t, :] * sigmoid(f16(x[t, :] . gate_w)): fq_routed_f16_combine with the gated shared expert
// fused (see RoutedCombineSharedKernel). rows half [T * top_k, K], weights float [T, top_k], shared half [T, K] (contiguous), x half [T, K] with
// unit column stride and any row pitch, gate_w half, K elements.
void fq_routed_f16_combine_shared(torch::Tensor rows, torch::Tensor weights, torch::Tensor shared, torch::Tensor x, torch::Tensor gate_w,
                                  torch::Tensor out)
{
    using namespace kyo_routed_f16;
    const at::cuda::OptionalCUDAGuard device_guard(rows.device());
    TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 && rows.is_contiguous() && rows.dtype() == torch::kHalf, "rows must be half [T * top_k, K]");
    TORCH_CHECK(weights.is_cuda() && weights.dim() == 2 && weights.is_contiguous() && weights.dtype() == torch::kFloat, "weights must be float [T, top_k]");
    const int64_t T = weights.size(0), top_k = weights.size(1), K = rows.size(1);
    TORCH_CHECK(rows.size(0) == T * top_k && K % 4 == 0, "rows must be [T * top_k, K] with K % 4 == 0");
    TORCH_CHECK(shared.is_cuda() && shared.is_contiguous() && shared.dtype() == torch::kHalf && shared.dim() == 2 && shared.size(0) == T && shared.size(1) == K,
                "shared must be half [T, K]");
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.dtype() == torch::kHalf && x.size(0) == T && x.size(1) == K && x.stride(1) == 1 &&
                x.stride(0) % 4 == 0 && ((std::uintptr_t) x.data_ptr() % 8) == 0, "x must be half [T, K], unit column stride, 8-byte aligned rows");
    TORCH_CHECK(gate_w.is_cuda() && gate_w.is_contiguous() && gate_w.dtype() == torch::kHalf && gate_w.numel() == K, "gate_w must be half with K elements");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dtype() == torch::kFloat && out.dim() == 2 && out.size(0) == T && out.size(1) == K,
                "out must be float [T, K]");
    if (T == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(RoutedCombineSharedKernel, dim3((unsigned) T), dim3(kThreads), 0, stream,
                       (const __half *) rows.data_ptr(), (const float *) weights.data_ptr(), (const __half *) shared.data_ptr(),
                       (const __half *) x.data_ptr(), (const __half *) gate_w.data_ptr(), (float *) out.data_ptr(), (std::uint32_t) top_k,
                       (std::uint32_t) K, (std::size_t) x.stride(0));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "fq_routed_f16_combine_shared: launch failed");
}

#endif
