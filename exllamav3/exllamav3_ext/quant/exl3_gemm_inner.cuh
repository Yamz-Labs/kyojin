#pragma once

#include "../ptx.cuh"

// Constants
#define EXL3_GEMM_BASE_THREADS 256
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6

#include "exl3_dq.cuh"

// On GA10x, HMMA with fp32 accumulation runs at half rate and dominates the m=1 (decode-bound) case.
// Accumulate MMA results in fp16 instead and fold into the fp32 accumulators once per k-slice: ~14%
// faster at bsz 1 on RTX 3090 together with the codebook.cuh IMUL change (see benchmarks/exl3_m1_bench).
// Max observed error vs fp32 accumulation is ~1% of output RMS at k=4096, well below quantization noise.
// Only enabled for sm_86 for now; unvalidated on other archs where fp32-acc HMMA is also half rate.
// EXL3_GEMM_NO_H_ACC (per translation unit): keep fp32 accumulation, e.g. for the fused 3INST MoE instances
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860) && !defined(EXL3_GEMM_NO_H_ACC)
    #define EXL3_GEMM_H_ACC 1
#else
    #define EXL3_GEMM_H_ACC 0
#endif

// ldmatrix of two / four 8x8 b16 matrices straight into B-operand fragments (swapped-operand tile, rows in the n8 dimension)
#if !defined(USE_ROCM)
__device__ __forceinline__ void ldsm4_xb(FragB& x0, FragB& x1, const void* smem_ptr)
{
    uint32_t* a = reinterpret_cast<uint32_t*>(&x0);
    uint32_t* b = reinterpret_cast<uint32_t*>(&x1);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile
    (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(b[0]), "=r"(b[1]) : "r"(smem)
    );
}

__device__ __forceinline__ void ldsm2_xb(FragB& x, const void* smem_ptr)
{
    uint32_t* a = reinterpret_cast<uint32_t*>(&x);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile
    (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(a[0]), "=r"(a[1]) : "r"(smem)
    );
}
#else
// Never reached: the swapped tile has no ROCm instance
__device__ __forceinline__ void ldsm4_xb(FragB&, FragB&, const void*) {}
__device__ __forceinline__ void ldsm2_xb(FragB&, const void*) {}
#endif

// TILESIZE_M == 24 is the swapped-operand tile for 17..24 rows (CUDA): the decoded weight fragments are the A operand
// (m16 = 16 output columns) and the activation rows the B operand (n8), so 24 rows cost three MMAs per 16 columns and k16
// where the 32-row tile pays four. Same warp-private k16 pipeline as the 16-row tile; each output element still
// accumulates the same products in k order (the C fragment is the transpose of the row-major tile).
// TILESIZE_M == 16 is the dense / decode shape: one m16 row fragment, A fragments double-buffered
// across the fragment stages (codegen unchanged by the multi-row support below). TILESIZE_M > 16
// is the fused-MoE prefill shape (32 / 64 rows per tile): TILEBLOCKS_M row fragments share each
// dequantized B fragment instead of re-running the whole B pipeline per 16 rows, with A
// single-buffered so the row fragments fit alongside the prefetched B fragments.
template<EXL3_GEMM_T_ARGS, bool shmem_out_had>
inline __device__
void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* post_scale,
    int size_n_stride = 0     // full width of B and C when computing a column slice (0: = size_n)
)
{
    constexpr bool SWAP = TILESIZE_M == 24;
    constexpr int SW_RB = 3;                                // row blocks of 8 in the swapped tile
    const int TILEBLOCKS_M = TILESIZE_M / 16;
    if (size_n_stride == 0) size_n_stride = size_n;
    // Column blocks of the full-width B row: slices index B relative to their own column offset,
    // but a k-tile row still spans the whole matrix
    const int blocks_n_full = size_n_stride / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    // const int FRAGS_M = TILEBLOCKS_M;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    // Warp-private pipeline (CUDA, 16x16 tiles). Each warp only ever decodes its own FRAGS_N_PER_WARP / 2 B blocks,
    // contiguous in smem and in global, so it copies exactly those (lane-indexed) plus its own copy of the A tile, and waits for
    // its stage with cp.async.wait_group + __syncwarp: no block-wide barrier in the main loop. Same data, same smem slots for B,
    // same arithmetic and order (bitwise). The column-end reduce keeps its barriers
#if !defined(USE_ROCM) && !defined(EXL3_GEMM_NO_PW)
    constexpr bool PW = (TILESIZE_M == 16 || SWAP) && TILESIZE_K == 16;
#else
    constexpr bool PW = false;
#endif
    constexpr int PW_WARPS = PW ? EXL3_GEMM_BASE_THREADS / 32 : 1;
#if defined(USE_ROCM)
    static_assert(!SWAP, "The swapped-operand tile is CUDA only");
#endif
    static_assert(!SWAP || (PW && TILESIZE_K == 16), "The swapped-operand tile needs the warp-private k16 pipeline");
    constexpr int PW_B_INT4 = FRAGS_N_PER_WARP / 2 * 256 / 16 * bits / 8;        // B int4 chunks per warp per stage

    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K * PW_WARPS;              // in halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;   // in uint16s
    const int sh_c_size = MAX  // in floats
    (
        4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP * TILEBLOCKS_M,
        shmem_out_had ? TILESIZE_N * TILESIZE_M : 0
    );

    // XOR-swizzle constants for bank-conflict-free A fragment loads
    // col_swizzled = col ^ ((row >> SHIFT) & MASK)
    const int A_COLS = TILESIZE_K / 8;                                            // int4 columns per row
    const int A_SWIZZLE_MASK = A_COLS - 1;
    const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    // Sanity checks
    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M >= 16 && (TILESIZE_M % 16 == 0 || SWAP), "Invalid kernel params");
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "Invalid kernel params (insufficient shared memory for shape)"
    );

    // Shared memory
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

    // Thread index
    int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    // Dimensions
    //int tiles_m = CEIL_DIVIDE(size_m, TILESIZE_M);
    int tiles_k = size_k / TILESIZE_K;
    int tiles_n = size_n / TILESIZE_N;
    //int blocks_m = 1;
    //int blocks_k = tiles_k * TILEBLOCKS_K;
    int blocks_n = tiles_n * TILEBLOCKS_N;

    // Start and end index of current slice, must span at least one tile
    int num_slices = gridDim.x;
    int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_m = [&] (int slice_i) { return 0; }; //blockIdx.y; };
    auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles_k); };

    // Batch dimension
    // int slice_m = index_m(slice_beg);
    // int max_m = MIN(size_m - slice_m * TILESIZE_M, TILESIZE_M);
    const int slice_m = 0;

    // Pipe 0, global A, B tile and shared A, B tile
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    // PW: the 32 lanes of each warp copy the whole 16x16 A tile (32 int4) into the warp's own A slot
    const int a_ld_stride = PW ? 32 : EXL3_GEMM_BASE_THREADS;
    const int load_a_iters = CEIL_DIVIDE(sh0_a_stride_m / 8, a_ld_stride);
    const int a_t = PW ? lane_id : t;
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        int k = (i * a_ld_stride + a_t) % (gl_a_stride_k / 8);
        int m = (i * a_ld_stride + a_t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK)) + (PW ? warp_id * (TILESIZE_M * TILESIZE_K / 8) : 0);
        pred_a_gl[i] = m < size_m;
    }

    int gl_b_stride_k = blocks_n_full * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    // PW: chunk c of the stage row is copied by warp c / PW_B_INT4 (the warp that decodes it), lane (c % PW_B_INT4) % 32
    const int load_b_iters = PW ? CEIL_DIVIDE(PW_B_INT4, 32) : CEIL_DIVIDE(sh0_b_stride_k / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    int load_b_c[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        int c = PW ? warp_id * PW_B_INT4 + i * 32 + lane_id : i * EXL3_GEMM_BASE_THREADS + t;
        int n = c % (gl_b_stride_n / 8);
        int k = c / (gl_b_stride_n / 8);
        load_b_c[i] = c;
        load_b_gl[i] = k * (blocks_n_full * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = PW ? i * 32 + lane_id < PW_B_INT4 : c < sh0_b_stride_k / 8;
    }

    // Per-thread copy addresses are loop-carried (advanced by a step, or by a precomputed wrap delta at the
    // end of a k column) so ptxas keeps them in registers instead of rematerializing the row / column / swizzle split
    // from t on every iteration. Same addresses as gl_a_ptr / gl_b_ptr / sh0_*_ptr + the per-thread offsets above
    const int4* a_src[load_a_iters];
    int4* a_dst[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        a_src[i] = (const int4*) gl_a_ptr + load_a_gl[i];
        a_dst[i] = (int4*) sh0_a_ptr + load_a_sh[i];
    }
    const int4* b_src[load_b_iters];
    int4* b_dst[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        b_src[i] = (const int4*) gl_b_ptr + load_b_gl[i];
        b_dst[i] = (int4*) sh0_b_ptr + load_b_c[i];
    }
    const int a_step = gl_a_stride_k / 8;                                         // int4 units
    const int a_wrap = -(tiles_k - 1) * (gl_a_stride_k / 8);                      // back to k = 0 (slice_m fixed)
    const int b_step = gl_b_stride_k / 8;
    const int b_wrap = (gl_b_stride_n - (tiles_k - 1) * gl_b_stride_k) / 8;       // k = 0 of the next column
    const int sh_a_back = -sh_a_stage_size / 8, sh_a_wrap = (SH_STAGES - 1) * sh_a_stage_size / 8;
    const int sh_b_back = -sh_b_stage_size / 8, sh_b_wrap = (SH_STAGES - 1) * sh_b_stage_size / 8;

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        // stage = slice0_iters % SH_STAGES steps down by one and wraps from 0 to SH_STAGES - 1
        const bool sw = (unsigned) slice0_iters % SH_STAGES == SH_STAGES - 1;
        const bool kw = slice0_k >= tiles_k;
        if (kw) { slice0_k = 0; slice0_n++; }

        #pragma unroll
        for (int i = 0; i < load_a_iters; ++i)
        {
            a_src[i] += kw ? a_wrap : a_step;
            a_dst[i] += sw ? sh_a_wrap : sh_a_back;
        }
        #pragma unroll
        for (int i = 0; i < load_b_iters; ++i)
        {
            b_src[i] += kw ? b_wrap : b_step;
            b_dst[i] += sw ? sh_b_wrap : sh_b_back;
        }
    };

    // Pipe 1, shared A, B tile and registers
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;

    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    // Fragment read addresses, loop-carried like the copy addresses
    constexpr int A_PTRS = SWAP ? 2 : TILEBLOCKS_M;
    const int4* a_frag[A_PTRS];
    if constexpr (SWAP)
    {
        // x4: rows 0..15 as (rows 0-7 k lo, rows 0-7 k hi, rows 8-15 k lo, rows 8-15 k hi); x2: rows 16..23 (k lo, k hi)
        int base_c = (lane_id / 8) % 2;
        int R0 = (lane_id % 8) + 8 * (lane_id / 16);
        int R1 = 16 + (lane_id % 8);
        a_frag[0] = (const int4*) sh1_a_ptr + R0 * A_COLS + (base_c ^ ((R0 >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK)) + warp_id * (TILESIZE_M * TILESIZE_K / 8);
        a_frag[1] = (const int4*) sh1_a_ptr + R1 * A_COLS + (base_c ^ ((R1 >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK)) + warp_id * (TILESIZE_M * TILESIZE_K / 8);
    }
    else
    {
        int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
        int base_c = lane_id / 16 + sub_k * 2;
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            int R = r + m * 16;
            int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
            a_frag[m] = (const int4*) sh1_a_ptr + R * A_COLS + c_swizzled + (PW ? warp_id * (TILESIZE_M * TILESIZE_K / 8) : 0);
        }
    }
    const uint32_t* b_frag = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + warp_id * FRAGS_N_PER_WARP / 2) * 256 / 16 * bits);
    const int shf_b_back = -sh_b_stage_size / 2, shf_b_wrap = (SH_STAGES - 1) * sh_b_stage_size / 2;   // uint32 units

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        const bool sw = (unsigned) slice1_iters % SH_STAGES == SH_STAGES - 1;
        #pragma unroll
        for (int m = 0; m < A_PTRS; ++m)
            a_frag[m] += sw ? sh_a_wrap : sh_a_back;
        b_frag += sw ? shf_b_wrap : shf_b_back;

        if (slice1_k >= tiles_k)
        {
            slice1_k = 0;
            slice1_n++;
        }
    };

    // Pipe 2
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TILESIZE_M * size_n_stride;

    half* gl_c_ptr_16 = ((half*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

    // TILEBLOCKS_M == 1 (dense / decode): A fragments double-buffered across the fragment stages,
    // as before. TILEBLOCKS_M > 1 (fused MoE prefill tiles): one A fragment per 16-row block,
    // single-buffered, so the row fragments fit alongside the prefetched B fragments
    register FragA frag_a[TILEBLOCKS_M == 1 ? FRAG_STAGES : TILEBLOCKS_M];
    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    register FragC frag_c[TILEBLOCKS_M][FRAGS_N_PER_WARP];
    #if EXL3_GEMM_H_ACC
        register FragC_h frag_c_h[TILEBLOCKS_M][FRAGS_N_PER_WARP];
    #endif
    // Swapped tile: activation fragments per fragment stage, accumulators per (row block, 16 output columns)
    constexpr int SW_PAIRS = SWAP ? FRAGS_N_PER_WARP / 2 : 1;
    register FragB frag_x[SWAP ? FRAG_STAGES : 1][SW_RB];
    register FragC frag_cs[SWAP ? SW_RB : 1][SW_PAIRS];
    #if EXL3_GEMM_H_ACC
        register FragC_h frag_cs_h[SWAP ? SW_RB : 1][SW_PAIRS];
    #endif

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (c_fp32)
                gl_c_ptr_32 += gl_c_stride_n;
            else
                gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    // Schedule load of the next A, B tiles to shared memory and advance the pipeline
    auto async_load_gl = [&] ()
    {
        if (TILESIZE_K > 16 && sub_k)   // sub_k is 0 for 16-wide k tiles (256 threads)
        {
            cp_async_fence();
            return;
        }

        if (slice0_iters)
        {
            // Copy tile from row-major A matrix (XOR-swizzled for bank-conflict-free ldmatrix)
            #pragma unroll
            for (int i = 0; i < load_a_iters; ++i)
            {
                if (pred_a_gl[i]) cp_async(a_dst[i], a_src[i]);
            }

            // Copy tile of 256-element blocks from quantized B matrix
            #pragma unroll
            for (int i = 0; i < load_b_iters; ++i)
            {
                if (pred_b_gl[i]) cp_async(b_dst[i], b_src[i]);
            }
            advance0();
        }

        // Sync and advance
        cp_async_fence();
    };

    // Load fragments
    // Ref. for fragment layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        // A fragments (XOR-swizzled shared memory layout)
        if constexpr (SWAP)
        {
            ldsm4_xb(frag_x[buf][0], frag_x[buf][1], (int4*) a_frag[0]);
            ldsm2_xb(frag_x[buf][2], (int4*) a_frag[1]);
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                ldsm4(frag_a[TILEBLOCKS_M == 1 ? buf : m], (int4*) a_frag[m]);
        }

        // B fragments
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            const uint32_t* shb = b_frag + (n2 / 2) * ((256 / 16 * bits / 2));

            dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }

        // Redundant for SH_STAGES >= 3 (wait_stage() orders these reads before the next cp.async overwrite).
        // CUDA sm_86, 16-wide k tiles: +4..12 % K3 / ~+1 % K4 per call, bitwise identical;
        // kept for other tiles (-1 % without it) and on ROCm (unmeasured)
#if defined(USE_ROCM)
        __syncthreads();
#else
        if constexpr (TILESIZE_K != 16 || SH_STAGES < 3) __syncthreads();
#endif
        advance1();
    };

    // Clear C fragments
    auto clear_frag_c = [&] ()
    {
        if constexpr (SWAP)
        {
            #pragma unroll
            for (int rb = 0; rb < SW_RB; ++rb)
                #pragma unroll
                for (int j = 0; j < SW_PAIRS; ++j)
                {
                    frag_cs[rb][j] = {};
                    #if EXL3_GEMM_H_ACC
                        frag_cs_h[rb][j] = {};
                    #endif
                }
        }
        else if constexpr (TILEBLOCKS_M == 1)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c[0][n] = {};
            #if EXL3_GEMM_H_ACC
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    frag_c_h[0][n] = {};
            #endif
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    frag_c[m][n] = {};
            #if EXL3_GEMM_H_ACC
                #pragma unroll
                for (int m = 0; m < TILEBLOCKS_M; ++m)
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        frag_c_h[m][n] = {};
            #endif
        }
    };

    // Threadblock reduction
    auto threadblock_reduce = [&] ()
    {
        auto store = [&] (int i)
        {
            if (sub_k == i)
            {
                if constexpr (TILEBLOCKS_M == 1)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[0][n][j];
                    }
                }
                else
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[m][n][j];
                        }
                }
            }
            __syncthreads();
        };

        auto add = [&] (int i)
        {
            if (sub_k == i)
            {
                if constexpr (TILEBLOCKS_M == 1)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) frag_c[0][n][j] += *sh_red++;
                    }
                }
                else
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) frag_c[m][n][j] += *sh_red++;
                        }
                }
            }
        };

        auto store_small = [&] (int i)
        {
            if constexpr (TILEBLOCKS_M == 1)
            {
                if (sub_k == i && lane_id / 4 < size_m)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        *sh_red++ = frag_c[0][n][0];
                        *sh_red++ = frag_c[0][n][1];
                    }
                }
                __syncthreads();
            }
            else
            {
                if (sub_k == i)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                    {
                        bool row_ok = (m * 16 + lane_id / 4) < size_m;
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        {
                            if (row_ok) { *sh_red = frag_c[m][n][0]; *(sh_red + 1) = frag_c[m][n][1]; }
                            sh_red += 2;
                        }
                    }
                }
                __syncthreads();
            }
        };

        auto add_small = [&] (int i)
        {
            if constexpr (TILEBLOCKS_M == 1)
            {
                if (sub_k == i && lane_id / 4 < size_m)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        frag_c[0][n][0] += *sh_red++;
                        frag_c[0][n][1] += *sh_red++;
                    }
                }
            }
            else
            {
                if (sub_k == i)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                    {
                        bool row_ok = (m * 16 + lane_id / 4) < size_m;
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        {
                            if (row_ok) { frag_c[m][n][0] += *sh_red; frag_c[m][n][1] += *(sh_red + 1); }
                            sh_red += 2;
                        }
                    }
                }
            }
        };

        if (size_m <= 8)
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store_small(1);
                add_small(0);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store_small(1);
                add_small(0);
                store_small(2);
                add_small(0);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store_small(3);
                add_small(2);
                store_small(1);
                add_small(0);
                store_small(2);
                add_small(0);
            }
        }
        else
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store(1);
                add(0);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store(1);
                add(0);
                store(2);
                add(0);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store(3);
                add(2);
                store(1);
                add(0);
                store(2);
                add(0);
            }
        }
    };

    // Swapped tile: visit every accumulator element as (row, column in the tile, value). The C fragment of the
    // transposed product holds rows 2t, 2t+1 of the row block at output columns g and g + 8 of each 16-column pair
    auto sw_visit = [&] (auto&& f)
    {
        const int g = lane_id / 4, tq = lane_id % 4;
        #pragma unroll
        for (int rb = 0; rb < SW_RB; ++rb)
            #pragma unroll
            for (int j = 0; j < SW_PAIRS; ++j)
            {
                const int col = warp_id * FRAGS_N_PER_WARP * 8 + 16 * j + g;
                const int row = 8 * rb + 2 * tq;
                f(row,     col,     frag_cs[rb][j][0]);
                f(row + 1, col,     frag_cs[rb][j][1]);
                f(row,     col + 8, frag_cs[rb][j][2]);
                f(row + 1, col + 8, frag_cs[rb][j][3]);
            }
    };

    // Pre-hadamard: Write final output tile to shmem
    auto write_sum_tile_sh = [&]()
    {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        if constexpr (SWAP)
        {
            sw_visit([&] (int row, int col, float& v) { if (row < size_m) ((float*) sh_c)[row * TILESIZE_N + col] = v; });
        }
        else if constexpr (TILEBLOCKS_M == 1)
        {
            const int r0 = lane_id / 4;
            const int r1 = r0 + 8;
            if (r0 < size_m)
            {
                const int c = (lane_id % 4) * 2;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    float* c_ptr = ((float*) sh_c) + r0 * TILESIZE_N + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[0][n][0];
                    *c_ptr++ = frag_c[0][n][1];
                }
            }
            if (r1 < size_m)
            {
                const int c = (lane_id % 4) * 2;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    float* c_ptr = ((float*) sh_c) + r1 * TILESIZE_N + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[0][n][2];
                    *c_ptr++ = frag_c[0][n][3];
                }
            }
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                const int r0 = m * 16 + lane_id / 4;
                const int r1 = r0 + 8;
                if (r0 < size_m)
                {
                    const int c = (lane_id % 4) * 2;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        float* c_ptr = ((float*) sh_c) + r0 * TILESIZE_N + (n0 + n) * 8 + c;
                        *c_ptr++ = frag_c[m][n][0];
                        *c_ptr++ = frag_c[m][n][1];
                    }
                }
                if (r1 < size_m)
                {
                    const int c = (lane_id % 4) * 2;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        float* c_ptr = ((float*) sh_c) + r1 * TILESIZE_N + (n0 + n) * 8 + c;
                        *c_ptr++ = frag_c[m][n][2];
                        *c_ptr++ = frag_c[m][n][3];
                    }
                }
            }
        }
    };

    // Copy output tile to global with hadamard transform and out scale
    auto output_had_sh_gl = [&]()
    {
        int sh_warp = warp_id;
        constexpr int active_warps = EXL3_GEMM_BASE_THREADS / 32;
        for (;; sh_warp += active_warps)
        {
            int col = sh_warp % (TILESIZE_N / 128);
            int row = sh_warp / (TILESIZE_N / 128);
            if (row >= size_m) break;

            const float* had_in = sh_c + row * TILESIZE_N + col * 128;
            const half* post_scale_c = post_scale + slice2_n * gl_c_stride_n + col * 128;

            if constexpr (c_fp32)
            {
                float* had_out = gl_c_ptr_32 + row * size_n_stride + col * 128;
                had_ff_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
            else
            {
                half* had_out = gl_c_ptr_16 + row * size_n_stride + col * 128;
                had_fh_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
        }
    };

    auto read_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        if constexpr (SWAP)
        {
            sw_visit([&] (int row, int col, float& v)
            {
                if (row >= size_m) return;
                if constexpr (c_fp32) v += gl_c_ptr_32[row * size_n_stride + col];
                else v += __half2float(gl_c_ptr_16[row * size_n_stride + col]);
            });
        }
        else if constexpr (TILEBLOCKS_M == 1)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                int r0 = lane_id / 4;
                int r1 = r0 + 8;
                int c = (lane_id % 4) * 2;
                if (r0 < size_m)
                {
                    if constexpr (c_fp32)
                    {
                        float* c_ptr = gl_c_ptr_32 + r0 * size_n_stride + (n0 + n) * 8 + c;
                        frag_c[0][n][0] += *c_ptr++;
                        frag_c[0][n][1] += *c_ptr++;
                    }
                    else
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n_stride + (n0 + n) * 8 + c);
                        float2 interm = __half22float2(*c_ptr);
                        frag_c[0][n][0] += interm.x;
                        frag_c[0][n][1] += interm.y;
                    }
                }
                if (r1 < size_m)
                {
                    if constexpr (c_fp32)
                    {
                        float* c_ptr = gl_c_ptr_32 + r1 * size_n_stride + (n0 + n) * 8 + c;
                        frag_c[0][n][2] += *c_ptr++;
                        frag_c[0][n][3] += *c_ptr++;
                    }
                    else
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n_stride + (n0 + n) * 8 + c);
                        float2 interm = __half22float2(*c_ptr);
                        frag_c[0][n][2] += interm.x;
                        frag_c[0][n][3] += interm.y;
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    int r0 = m * 16 + lane_id / 4;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < size_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n_stride + (n0 + n) * 8 + c;
                            frag_c[m][n][0] += *c_ptr++;
                            frag_c[m][n][1] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n_stride + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][0] += interm.x;
                            frag_c[m][n][1] += interm.y;
                        }
                    }
                    if (r1 < size_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n_stride + (n0 + n) * 8 + c;
                            frag_c[m][n][2] += *c_ptr++;
                            frag_c[m][n][3] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n_stride + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][2] += interm.x;
                            frag_c[m][n][3] += interm.y;
                        }
                    }
                }
        }
    };

    auto write_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        if constexpr (SWAP)
        {
            sw_visit([&] (int row, int col, float& v)
            {
                if (row >= size_m) return;
                if constexpr (c_fp32) gl_c_ptr_32[row * size_n_stride + col] = v;
                else gl_c_ptr_16[row * size_n_stride + col] = __float2half_rn(v);
            });
        }
        else if constexpr (TILEBLOCKS_M == 1)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                int r0 = lane_id / 4;
                int r1 = r0 + 8;
                int c = (lane_id % 4) * 2;
                if (r0 < size_m)
                {
                    if constexpr (c_fp32)
                    {
                        float* c_ptr = gl_c_ptr_32 + r0 * size_n_stride + (n0 + n) * 8 + c;
                        *c_ptr++ = frag_c[0][n][0];
                        *c_ptr++ = frag_c[0][n][1];
                    }
                    else
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n_stride + (n0 + n) * 8 + c);
                        half2 sum = __floats2half2_rn(frag_c[0][n][0], frag_c[0][n][1]);
                        *c_ptr = sum;
                    }
                }
                if (r1 < size_m)
                {
                    if constexpr (c_fp32)
                    {
                        float* c_ptr = gl_c_ptr_32 + r1 * size_n_stride + (n0 + n) * 8 + c;
                        *c_ptr++ = frag_c[0][n][2];
                        *c_ptr++ = frag_c[0][n][3];
                    }
                    else
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n_stride + (n0 + n) * 8 + c);
                        half2 sum = __floats2half2_rn(frag_c[0][n][2], frag_c[0][n][3]);
                        *c_ptr = sum;
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    int r0 = m * 16 + lane_id / 4;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < size_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n_stride + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][0];
                            *c_ptr++ = frag_c[m][n][1];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n_stride + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                            *c_ptr = sum;
                        }
                    }
                    if (r1 < size_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n_stride + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][2];
                            *c_ptr++ = frag_c[m][n][3];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n_stride + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                            *c_ptr = sum;
                        }
                    }
                }
        }
    };

    // Output reduction
    auto reduce = [&] ()
    {
        #if EXL3_GEMM_H_ACC
            // Fold the fp16 MMA accumulators into the fp32 accumulators once per k-slice
            if constexpr (SWAP)
            {
                #pragma unroll
                for (int rb = 0; rb < SW_RB; ++rb)
                    #pragma unroll
                    for (int j = 0; j < SW_PAIRS; ++j)
                    {
                        float2 f0 = __half22float2(frag_cs_h[rb][j][0]);
                        float2 f1 = __half22float2(frag_cs_h[rb][j][1]);
                        frag_cs[rb][j][0] += f0.x; frag_cs[rb][j][1] += f0.y;
                        frag_cs[rb][j][2] += f1.x; frag_cs[rb][j][3] += f1.y;
                    }
            }
            else if constexpr (TILEBLOCKS_M == 1)
            {
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    float2 f0 = __half22float2(frag_c_h[0][n][0]);
                    float2 f1 = __half22float2(frag_c_h[0][n][1]);
                    frag_c[0][n][0] += f0.x; frag_c[0][n][1] += f0.y;
                    frag_c[0][n][2] += f1.x; frag_c[0][n][3] += f1.y;
                }
            }
            else
            {
                #pragma unroll
                for (int m = 0; m < TILEBLOCKS_M; ++m)
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        float2 f0 = __half22float2(frag_c_h[m][n][0]);
                        float2 f1 = __half22float2(frag_c_h[m][n][1]);
                        frag_c[m][n][0] += f0.x; frag_c[m][n][1] += f0.y;
                        frag_c[m][n][2] += f1.x; frag_c[m][n][3] += f1.y;
                    }
            }
        #endif

        // First reduce all partial sums along k for the current slice
        threadblock_reduce();

        // Process (partial) slices within column in reverse order so the threadblock doing the bottom slice is
        // free to proceed to the next column right away
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks_n + slice2_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        // Second and subsequent threadblocks in column read back the intermediate sum from global memory
        if (!sub_k && !first)
        {
            read_sum_gl();
        }

        // All but last threadblock in column write the intermediate result to global memory
        if (!sub_k && !last)
        {
            write_sum_gl();
        }

        // Last block writes in row-major format
        if (!sub_k && last)
        {
            if constexpr (shmem_out_had)
                write_sum_tile_sh();
            else
                write_sum_gl();
        }

        if constexpr (shmem_out_had)
        {
            if (last) __syncthreads();
            if (!sub_k && last)
                output_had_sh_gl();
        }

        barrier_release(lock, lock_d, last);

        clear_frag_c();
    };

    // Wait until there are at most SH_STAGES - 2 async copies pending, i.e. at least one stage has finished loading
    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
#if defined(USE_ROCM)
        __syncthreads();
#else
        if constexpr (PW) __syncwarp();
        else __syncthreads();
#endif
    };

    // Perform tensor core matmul on current tile
    auto matmul = [&] (int buf)
    {
        if constexpr (SWAP)
        {
            #pragma unroll
            for (int j = 0; j < SW_PAIRS; ++j)
            {
                // The two decoded n8 fragments of a 16-column block are the rows 0-7 / 8-15 halves of the A operand
                FragA w;
                w.elems[0] = frag_b[buf][2 * j].elems[0];
                w.elems[1] = frag_b[buf][2 * j + 1].elems[0];
                w.elems[2] = frag_b[buf][2 * j].elems[1];
                w.elems[3] = frag_b[buf][2 * j + 1].elems[1];
                #pragma unroll
                for (int rb = 0; rb < SW_RB; ++rb)
                {
                    #if EXL3_GEMM_H_ACC
                        ptx_mma_m16n8k16(w, frag_x[buf][rb], frag_cs_h[rb][j]);
                    #else
                        ptx_mma_m16n8k16(w, frag_x[buf][rb], frag_cs[rb][j]);
                    #endif
                }
            }
        }
        else if constexpr (TILEBLOCKS_M == 1)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                #if EXL3_GEMM_H_ACC
                    ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c_h[0][n]);
                #else
                    ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c[0][n]);
                #endif
            }
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #if EXL3_GEMM_H_ACC
                        ptx_mma_m16n8k16(frag_a[m], frag_b[buf][n], frag_c_h[m][n]);
                    #else
                        ptx_mma_m16n8k16(frag_a[m], frag_b[buf][n], frag_c[m][n]);
                    #endif
                }
        }
    };

    // Start global to shared pipeline
    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    // Start shared to register pipeline.
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

    // Main loop. Fragments are double buffered to allow more interleaving. This is especially important to hide the
    // dequantization overhead, but we need two different iterations of the main loop to avoid confusing the compiler
    // and making it (sometimes) place the fragment arrays in local memory

    #define FSTAGE_OLD(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \

    #define FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \
        load_frags(_load); \

    if constexpr (FRAG_STAGES == 1)
    {
        while (true)
        {
            FSTAGE_OLD(0, 0);
        }
    }

    if constexpr (FRAG_STAGES == 2)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(0, 1);
        }
    }

    if constexpr (FRAG_STAGES == 3)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(0, 2);
        }
    }

    if constexpr (FRAG_STAGES == 4)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(0, 3);
        }
    }

    if constexpr (FRAG_STAGES == 5)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(4, 3);
            FSTAGE(0, 4);
        }
    }
}
