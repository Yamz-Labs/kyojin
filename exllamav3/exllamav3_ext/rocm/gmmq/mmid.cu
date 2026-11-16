// SPDX-License-Identifier: MIT
// Adapted from Gufo (github.com/gufo-org/gufo, 07e1191, MIT, Copyright (c) 2026 Gufo contributors), path
// src/models/qwen38_flash_next/kernels/rocm/mmq/mmid.hip.cpp (renamed .cu), itself adapted from llama.cpp 5c0e946 ggml/src/ggml-cuda
// (github.com/ggml-org/llama.cpp, MIT, Copyright (c) 2023-2026 The ggml authors). See THIRD_PARTY_NOTICES.md.
// Changes: scan scratch supplied by the caller instead of a global buffer grown with gufo::hip::AllocateDevice;
// qfn_mmq_build_ids_maps C entry dropped.
#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "common.hpp"
#include "mmid.hpp"

// qwen38 local: parallel, deterministic id-map builder, replacing llama.cpp's
// mm_ids_helper. That helper ran one warp per expert over every assignment
// row (512 warps x 20,480 rows at a 2,048-token chunk), so its serial token
// loop cost 0.6-1.0 ms per launch. This one keeps its output contract (rows
// of one expert in ascending assignment order, so ids_src1/ids_dst/
// expert_bounds are bit-identical) with three parallel passes:
//   1. per block of kIdsRowsPerBlock assignment rows: each lane's rank among
//      the same expert inside its wave (a 32-step shuffle loop), wave counts
//      per expert in LDS, then per-expert prefix over the block's waves; the
//      in-block rank goes to scratch and the block's per-expert counts to
//      global memory.
//   2. one thread per expert: prefix over the blocks' counts (block bases)
//      and the expert bounds.
//   3. scatter: position = bound[e] + block_base[b][e] + in-block rank.
// Like llama.cpp's helper, a negative id is counted below every expert (it
// shifts all bounds by one and leaves its slot unwritten) and an id at or
// past n_experts is dropped; block_counts carries the negatives in column
// n_experts.
constexpr int kIdsRowsPerBlock = 256;
constexpr int kIdsWaves = kIdsRowsPerBlock / 32;

static __global__ void mm_ids_rank_kernel(
        const int32_t * __restrict__ ids, int32_t * __restrict__ local_rank,
        int32_t * __restrict__ block_counts, const int n_rows,
        const int n_experts, const int n_expert_used, const int si1) {
    extern __shared__ int32_t wave_counts[];  // [n_experts][kIdsWaves]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    for (int i = tid; i < (n_experts + 1) * kIdsWaves; i += blockDim.x) {
        wave_counts[i] = 0;
    }
    __syncthreads();
    const int row = blockIdx.x * kIdsRowsPerBlock + tid;
    int expert = -2;  // rows past the end are dropped
    if (row < n_rows) {
        const int it = row / n_expert_used;
        const int iex = row - it * n_expert_used;
        expert = ids[it * si1 + iex];
        if (expert >= n_experts) {
            expert = -2;  // dropped
        } else if (expert < 0) {
            expert = -1;  // counted below expert 0, never written
        }
    }
    // Rank among the same expert at lower lanes of this wave.
    int rank = 0;
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        const int other = __shfl(expert, l, 32);
        rank += (l < lane && other == expert) ? 1 : 0;
    }
    if (expert >= 0) {
        atomicAdd(&wave_counts[expert * kIdsWaves + wave], 1);
    } else if (expert == -1) {
        atomicAdd(&wave_counts[n_experts * kIdsWaves + wave], 1);
    }
    __syncthreads();
    // Exclusive prefix over the waves of each expert; the total is the
    // block's count.
    for (int e = tid; e <= n_experts; e += blockDim.x) {
        int32_t * counts = wave_counts + e * kIdsWaves;
        int running = 0;
#pragma unroll
        for (int w = 0; w < kIdsWaves; ++w) {
            const int c = counts[w];
            counts[w] = running;
            running += c;
        }
        block_counts[blockIdx.x * (n_experts + 1) + e] = running;
    }
    __syncthreads();
    if (row < n_rows) {
        local_rank[row] = expert >= 0
            ? wave_counts[expert * kIdsWaves + wave] + rank : -1;
    }
}

static __global__ void mm_ids_bounds_kernel(
        int32_t * __restrict__ block_counts, int32_t * __restrict__ expert_bounds,
        const int n_blocks, const int n_experts) {
    // One block: per-expert totals, then a block-wide exclusive scan over
    // experts for the bounds. Block counts become block bases in place.
    extern __shared__ int32_t totals[];  // [n_experts + 1]
    for (int e = threadIdx.x; e <= n_experts; e += blockDim.x) {
        int running = 0;
        for (int b = 0; b < n_blocks; ++b) {
            const int c = block_counts[b * (n_experts + 1) + e];
            block_counts[b * (n_experts + 1) + e] = running;
            running += c;
        }
        totals[e] = running;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int running = totals[n_experts];  // negative ids sit below expert 0
        for (int e = 0; e < n_experts; ++e) {
            expert_bounds[e] = running;
            running += totals[e];
        }
        expert_bounds[n_experts] = running;
    }
}

static __global__ void mm_ids_scatter_kernel(
        const int32_t * __restrict__ ids, const int32_t * __restrict__ local_rank,
        const int32_t * __restrict__ block_counts,
        const int32_t * __restrict__ expert_bounds,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int n_rows, const int n_experts, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1) {
    const int row = blockIdx.x * kIdsRowsPerBlock + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const int rank = local_rank[row];
    if (rank < 0) {
        return;
    }
    const int it = row / n_expert_used;
    const int iex = row - it * n_expert_used;
    const int expert = ids[it * si1 + iex];
    const int pos = expert_bounds[expert] +
                    block_counts[blockIdx.x * (n_experts + 1) + expert] + rank;
    ids_src1[pos] = it * sis1 + iex % nchannels_y;
    ids_dst[pos] = it * n_expert_used + iex;
}

// kyojin: the scan scratch (gmmq_ids_scratch_elems int32) is supplied by the caller (a torch allocation on the
// launch stream) instead of a process-global buffer grown with gufo::hip::AllocateDevice.
size_t gmmq_ids_scratch_elems(const int n_experts, const int n_tokens, const int n_expert_used) {
    const int n_rows = n_tokens * n_expert_used;
    const int n_blocks = (n_rows + kIdsRowsPerBlock - 1) / kIdsRowsPerBlock;
    return (size_t)n_rows + (size_t)n_blocks * (n_experts + 1);
}

void ggml_hip_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        int32_t * __restrict__ scratch, hipStream_t stream) {
    const int n_rows = n_tokens * n_expert_used;
    const int n_blocks = (n_rows + kIdsRowsPerBlock - 1) / kIdsRowsPerBlock;
    const size_t lds = (size_t)(n_experts + 1) * kIdsWaves * sizeof(int32_t);
    GGML_ASSERT(n_rows > 0 && lds <= 48 * 1024 && "routed id maps: shape");
    int32_t * local_rank = scratch;
    int32_t * block_counts = scratch + n_rows;
    mm_ids_rank_kernel<<<n_blocks, kIdsRowsPerBlock, lds, stream>>>(
        ids, local_rank, block_counts, n_rows, n_experts, n_expert_used, si1);
    mm_ids_bounds_kernel<<<1, 1024, (size_t)(n_experts + 1) * sizeof(int32_t), stream>>>(
        block_counts, expert_bounds, n_blocks, n_experts);
    mm_ids_scatter_kernel<<<n_blocks, kIdsRowsPerBlock, 0, stream>>>(
        ids, local_rank, block_counts, expert_bounds, ids_src1, ids_dst, n_rows,
        n_experts, n_expert_used, nchannels_y, si1, sis1);
}

}  // namespace qfn_mmq
