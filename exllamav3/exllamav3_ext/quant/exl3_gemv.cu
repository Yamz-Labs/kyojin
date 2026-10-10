#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include "exl3_gemv.cuh"
#include "hadamard.cuh"
#if defined(USE_ROCM)
#include "exl3_gemv_int8.cuh"   // fused int8-activation GEMV (mul1 tensors, m <= 2), opt-in via EXL3_INT8_GEMV
#endif

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#if !defined(USE_ROCM)
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#if defined(USE_ROCM)
#include "exl3_moe_valu.cuh"   // VALU grouped-MoE decode matvec, opt-in EXL3_MOE_VALU=1 (K2 and K4 experts)
#endif
#include "exl3_devctx.cuh"
#include <cstdio>
#include <cstring>
#include <map>
#include <cstdlib>
#include <mutex>
#include <algorithm>

/*
QTIP-style small-m GEMV path, kernel in exl3_gemv_kernel.cuh. Dispatched from exl3_gemm via
exl3_gemv_try_launch when the shape heuristic applies, or forced through the exl3_gemv entry
point. Kernel arguments and graph parameter offsets are identical to exl3_gemm_kernel.

Env: EXL3_GEMV = 0 disables the path, 1/unset = heuristic (default), 2 = use wherever the hard
constraints allow (testing). EXL3_GEMV_DEBUG=1 prints each newly queried kernel occupancy.

The CUDA heuristic envelope (measured on RTX 3090 at 4 bpw and m <= 8) favors the narrow
config at attention-projection sizes (n <= 4096) and the wide config at large-n/small-k FFN
sizes. HIP gfx12 additionally supports m <= 16 through its 16-row WMMA mode.
*/

#if defined(USE_ROCM)
constexpr int EXL3_GEMV_ROCM_MAX_M = 16;


namespace {

constexpr int MOE_HIDDEN = 2560;
constexpr int MOE_TOP_K = 10;
constexpr int MOE_MAX_ROWS = 16;
constexpr int MOE_PREFILL_MAX_ROWS = 2048;
constexpr int MOE_PREFILL_ROWS_PER_CHUNK = 16;
constexpr int MOE_PREFILL_MAX_EXPERT_ROWS = MOE_PREFILL_MAX_ROWS * MOE_TOP_K;
constexpr int MOE_PREFILL_CHUNKS_PER_EXPERT =
    MOE_PREFILL_MAX_EXPERT_ROWS / MOE_PREFILL_ROWS_PER_CHUNK;
constexpr int MOE_THREADS = 512;

// Runtime k-split selection for the grouped-MoE decode GEMV. Default 0 keeps the
// original 512-thread/WK=16 shape; 1 and 2 narrow it for small-CU parts.
static inline int moe_decode_cfg()
{
    static const int cfg = [] {
        const char* e = getenv("EXL3_MOE_CFG");
        int v = e ? atoi(e) : 0;
        return (v < 0 || v > 2) ? 0 : v;
    }();
    return cfg;
}

static inline int moe_decode_threads(int cfg)
{
    return cfg == 0 ? 512 : cfg == 1 ? 256 : 128;
}

// Same knob for the grouped-MoE PREFILL GEMV (MMODE 2, 16-row expert chunks), which the MTP
// verify batches take via EXL3_HIP_PREFILL_MIN_ROWS=2. It was hardcoded to CFG 2 (4 warps,
// 64 columns per block) and never swept on gfx1151. EXL3_MOE_PREFILL_CFG=0/1/2, default 2.
static inline int moe_prefill_cfg()
{
    static const int cfg = [] {
        const char* e = getenv("EXL3_MOE_PREFILL_CFG");
        int v = e ? atoi(e) : 2;
        return (v < 0 || v > 2) ? 2 : v;
    }();
    return cfg;
}


constexpr float HAD_SCALE = 0.088388347648f;

template <bool PRE_SCALE, bool FP32>
__global__ __launch_bounds__(32)
void moe_had_rows_kernel
(
    const void* input,
    void* output,
    const int64_t* selected,
    const int64_t* scale_tables_0,
    const int64_t* scale_tables_1,
    int assignments,
    int width,
    int experts,
    bool token_input,
    int top_k,
    const int* token_row = nullptr
)
{
    const int matrix = blockIdx.x;
    const int slot = matrix % assignments;
    const int projection = matrix / assignments;
    const int64_t expert = selected[slot];
    const int64_t* scale_table = projection == 0 ? scale_tables_0 : scale_tables_1;
    const size_t output_offset = (size_t) matrix * width + blockIdx.y * 128;
    const bool valid_expert = expert >= 0 && expert < experts;
    const int64_t scale_ptr = valid_expert ? scale_table[expert] : 0;
    if (!scale_ptr)
    {
        if constexpr (FP32)
        {
            float* output_ptr = reinterpret_cast<float*>(output) + output_offset;
            for (int idx = threadIdx.x; idx < 128; idx += blockDim.x) output_ptr[idx] = 0.0f;
        }
        else
        {
            half* output_ptr = reinterpret_cast<half*>(output) + output_offset;
            for (int idx = threadIdx.x; idx < 128; idx += blockDim.x) output_ptr[idx] = __float2half_rn(0.0f);
        }
        return;
    }

    const half* scale = reinterpret_cast<const half*>(scale_ptr);
    // Token-major input (the layer input, one row per token) vs assignment-major: the routing
    // slot -> token mapping needs the runtime top-k, not the old compile-time 10.
    const size_t input_row = token_input ? (token_row ? token_row[slot] : slot / top_k) : matrix;
    const size_t input_offset = input_row * width + blockIdx.y * 128;

    if constexpr (FP32)
        had_ff_r_128_inner<PRE_SCALE, !PRE_SCALE>
        (
            reinterpret_cast<const float*>(input) + input_offset,
            reinterpret_cast<float*>(output) + output_offset,
            scale,
            HAD_SCALE
        );
    else
        had_hf_r_128_inner<PRE_SCALE, !PRE_SCALE>
        (
            reinterpret_cast<const half*>(input) + input_offset,
            reinterpret_cast<half*>(output) + output_offset,
            scale,
            HAD_SCALE
        );
}

__global__ void moe_silu_mul_kernel(const half* gate, const half* up, half* output, int count)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < count; idx += blockDim.x * gridDim.x)
    {
        const float g = __half2float(gate[idx]);
        const half activated = __float2half_rn(g / (1.0f + expf(-g)));
        output[idx] = __hmul(activated, up[idx]);
    }
}

// MOE_CFG selects the k-split width WK (warps per block): 0 -> 16, 1 -> 8, 2 -> 4.
// gfx1151 has only 20 CUs, where the widest split adds LDS pressure and cross-warp
// reduction work without adding useful parallelism, so it is tunable at runtime.
template <int K, bool HALF, bool FP32, bool TWO_PROJECTIONS, int MOE_CFG = 0>
__global__ __launch_bounds__(MOE_CFG == 0 ? 512 : MOE_CFG == 1 ? 256 : 128)
void moe_grouped_gemv_k3_kernel
(
    const half* A,
    const int64_t* selected,
    const int64_t* B_table_0,
    const int64_t* B_table_1,
    void* C,
    int size_k,
    int size_n,
    int experts,
    int assignments
)
{
    const int slot = blockIdx.y;
    const int projection = TWO_PROJECTIONS ? blockIdx.z : 0;
    const int64_t expert = selected[slot];
    const int64_t* B_table = projection == 0 ? B_table_0 : B_table_1;
    const size_t matrix = (size_t) projection * assignments + slot;
    void* C_row = FP32
        ? static_cast<void*>(reinterpret_cast<float*>(C) + matrix * size_n)
        : static_cast<void*>(reinterpret_cast<half*>(C) + matrix * size_n);
    const bool valid_expert = expert >= 0 && expert < experts;
    const int64_t B_ptr = valid_expert ? B_table[expert] : 0;
    if (!B_ptr)
    {
        const int col = blockIdx.x * 32 + threadIdx.x;
        if (threadIdx.x < 32 && col < size_n)
        {
            if constexpr (FP32) reinterpret_cast<float*>(C_row)[col] = 0.0f;
            else reinterpret_cast<half*>(C_row)[col] = __float2half_rn(0.0f);
        }
        return;
    }

    const uint16_t* B = reinterpret_cast<const uint16_t*>(B_ptr);
    const half* A_row = A + matrix * size_k;
    // K is the trellis bitrate of this layer's experts (uniform across gate/up/down); the
    // grouped path used to be K=3-only, which is why it kept the _k3 name.
    exl3_gemv_kernel_body<K, FP32, 2, 0, MOE_CFG, true, HALF>
    (A_row, B, C_row, 1, size_k, size_n, nullptr, nullptr, nullptr, nullptr);
}

// Expert-deduplicated variant for verify batches. The assignments are sorted by expert
// (moe_dedup_metadata_kernel), so every expert that several tokens picked is a contiguous
// run of rows; one block column decodes that expert's trellis ONCE and applies it to all of
// the run's rows (MMODE 1, <= 8 rows). blockIdx.y is the run index; runs past num_groups
// exit. Each output row is the same WMMA row accumulation as the single-row kernel.
template <int K, bool HALF, bool FP32, bool TWO_PROJECTIONS, int MOE_CFG = 0>
__global__ __launch_bounds__(MOE_CFG == 0 ? 512 : MOE_CFG == 1 ? 256 : 128)
void moe_dedup_gemv_kernel
(
    const half* A,
    const int64_t* sorted_expert,
    const int* group_start,
    const int* num_groups,
    const int64_t* B_table_0,
    const int64_t* B_table_1,
    void* C,
    int size_k,
    int size_n,
    int experts,
    int assignments
)
{
    const int g = blockIdx.y;
    if (g >= *num_groups) return;
    const int p0 = group_start[g];
    const int rows = group_start[g + 1] - p0;
    const int projection = TWO_PROJECTIONS ? blockIdx.z : 0;
    const int64_t expert = sorted_expert[p0];
    const int64_t* B_table = projection == 0 ? B_table_0 : B_table_1;
    const size_t matrix = (size_t) projection * assignments + p0;
    void* C_rows = FP32
        ? static_cast<void*>(reinterpret_cast<float*>(C) + matrix * size_n)
        : static_cast<void*>(reinterpret_cast<half*>(C) + matrix * size_n);
    const bool valid_expert = expert >= 0 && expert < experts;
    const int64_t B_ptr = valid_expert ? B_table[expert] : 0;
    if (!B_ptr)
    {
        const int col = blockIdx.x * 32 + threadIdx.x;
        if (threadIdx.x < 32 && col < size_n)
            for (int r = 0; r < rows; ++r)
            {
                if constexpr (FP32) reinterpret_cast<float*>(C_rows)[(size_t) r * size_n + col] = 0.0f;
                else reinterpret_cast<half*>(C_rows)[(size_t) r * size_n + col] = __float2half_rn(0.0f);
            }
        return;
    }
    const uint16_t* B = reinterpret_cast<const uint16_t*>(B_ptr);
    const half* A_rows = A + matrix * size_k;
    exl3_gemv_kernel_body<K, FP32, 2, 1, MOE_CFG, true, HALF>
    (A_rows, B, C_rows, rows, size_k, size_n, nullptr, nullptr, nullptr, nullptr);
}

// Sort the (token, slot) assignments by expert, stably (ties keep assignment order), in one
// block (assignments <= 128). Outputs, all indexed by sorted position p unless noted:
//   sorted_expert[p], token_row[p] (the source token), pos_of[a] (assignment a -> p),
//   group_start[0..G] (runs of equal expert, G+1 entries), num_groups[0] = G.
__global__ void moe_dedup_metadata_kernel
(
    const int64_t* selected,
    int64_t* sorted_expert,
    int* token_row,
    int* pos_of,
    int* group_start,
    int* num_groups,
    int assignments,
    int top_k
)
{
    __shared__ int64_t s_exp[128];
    __shared__ int s_head[128];
    const int a = threadIdx.x;
    if (a < assignments)
    {
        const int64_t e = selected[a];
        int p = 0;
        for (int b = 0; b < assignments; ++b)
        {
            const int64_t eb = selected[b];
            p += (eb < e) || (eb == e && b < a);
        }
        s_exp[p] = e;
        sorted_expert[p] = e;
        token_row[p] = a / top_k;
        pos_of[a] = p;
    }
    __syncthreads();
    if (a < assignments) s_head[a] = (a == 0 || s_exp[a - 1] != s_exp[a]) ? 1 : 0;
    __syncthreads();
    if (a == 0)
    {
        int g = 0;
        for (int q = 0; q < assignments; ++q)
            if (s_head[q]) group_start[g++] = q;
        group_start[g] = assignments;
        num_groups[0] = g;
    }
}

// TOP_K is the routing width of this layer (Qwen3.8: 10, MiMo-V2: 8). It stays a template
// parameter so the expert-sorted accumulation below keeps its fully unrolled form.
template <int TOP_K>
__global__ void moe_weighted_reduce_kernel
(
    const float* rows,
    const int64_t* selected,
    const half* weights,
    float* output,
    int width,
    const int* pos_of
)
{
    const int row = blockIdx.y;
    // pos_of (dedup path) maps assignment row*TOP_K+slot to its expert-sorted row in `rows`
    const int* pos = pos_of ? pos_of + row * TOP_K : nullptr;
    if (!pos) rows += (size_t) row * TOP_K * width;
    selected += row * TOP_K;
    weights += row * TOP_K;
    output += (size_t) row * width;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x;
         col < width; col += blockDim.x * gridDim.x)
    {
        // Match the established fallback's expert-sorted accumulation order independently
        // for each token. Ties retain routing-slot order, so duplicates remain distinct.
        unsigned used = 0;
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < TOP_K; ++rank)
        {
            int best_slot = -1;
            int64_t best_expert = INT64_MAX;
            #pragma unroll
            for (int slot = 0; slot < TOP_K; ++slot)
            {
                const bool available = !(used & (1u << slot));
                const int64_t expert = selected[slot];
                if (available && (expert < best_expert ||
                    (expert == best_expert && (best_slot < 0 || slot < best_slot))))
                {
                    best_expert = expert;
                    best_slot = slot;
                }
            }
            used |= 1u << best_slot;
            const float weighted = __fmul_rn(
                rows[(size_t) (pos ? pos[best_slot] : best_slot) * width + col], __half2float(weights[best_slot]));
            sum = __fadd_rn(sum, weighted);
        }
        output[col] = sum;
    }
}

__global__ void moe_prefill_metadata_kernel
(
    const int64_t* expert_count,
    const int64_t* order,
    int64_t* expert_offsets,
    int64_t* inverse_order,
    int* expert_chunks,
    int* num_chunks_out,
    int experts,
    int assignments
)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < assignments; idx += blockDim.x * gridDim.x)
    {
        const int64_t original_slot = order[idx];
        if (original_slot >= 0 && original_slot < assignments)
            inverse_order[original_slot] = idx;
    }

    // Block 0 builds expert_offsets (exclusive scan of counts) and the compacted chunk list
    // with a block-wide scan. The previous single-thread loop over 512 experts took ~31 us
    // per call on gfx1151 = ~1.5 ms per MTP round across 48 layers.
    if (blockIdx.x != 0) return;
    __shared__ int64_t warp_off[32];
    __shared__ int warp_chk[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    int64_t run_off = 0;
    int run_chk = 0;
    if (threadIdx.x == 0) expert_offsets[0] = 0;
    for (int base = 0; base < experts; base += blockDim.x)
    {
        const int e = base + threadIdx.x;
        const int64_t count = e < experts ? expert_count[e] : 0;
        const int nch = (e < experts && count > 0 && count <= MOE_PREFILL_MAX_EXPERT_ROWS)
            ? (int) ((count + MOE_PREFILL_ROWS_PER_CHUNK - 1) / MOE_PREFILL_ROWS_PER_CHUNK) : 0;
        // inclusive warp scans
        int64_t so = count; int sc = nch;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            const int64_t to = __shfl_up_sync(0xffffffffu, so, o);
            const int tc = __shfl_up_sync(0xffffffffu, sc, o);
            if (lane >= o) { so += to; sc += tc; }
        }
        if (lane == 31) { warp_off[warp] = so; warp_chk[warp] = sc; }
        __syncthreads();
        int64_t wbase_o = run_off; int wbase_c = run_chk;
        for (int w = 0; w < warp; ++w) { wbase_o += warp_off[w]; wbase_c += warp_chk[w]; }
        const int64_t excl_o = wbase_o + so - count;
        const int excl_c = wbase_c + sc - nch;
        if (e < experts)
        {
            expert_offsets[e + 1] = excl_o + count;
            for (int c = 0; c < nch; ++c)
                expert_chunks[excl_c + c] = e * MOE_PREFILL_CHUNKS_PER_EXPERT + c;
        }
        __syncthreads();
        for (int w = 0; w < nwarps; ++w) { run_off += warp_off[w]; run_chk += warp_chk[w]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) *num_chunks_out = run_chk;
}


template <bool PRE_SCALE, bool FP32>
__global__ __launch_bounds__(32)
void moe_prefill_had_rows_kernel
(
    const void* input,
    void* output,
    const int64_t* selected,
    const int64_t* order,
    const int64_t* expert_count,
    const int64_t* scale_tables_0,
    const int64_t* scale_tables_1,
    int assignments,
    int width,
    int experts,
    bool token_input
)
{
    const int matrix = blockIdx.x;
    const int sorted_slot = matrix % assignments;
    const int projection = matrix / assignments;
    const int64_t original_slot = order[sorted_slot];
    const bool valid_order = original_slot >= 0 && original_slot < assignments;
    const int64_t expert = valid_order ? selected[original_slot] : -1;
    const int64_t* scale_table = projection == 0 ? scale_tables_0 : scale_tables_1;
    const size_t output_offset = (size_t) matrix * width + blockIdx.y * 128;
    const bool valid_expert = expert >= 0 && expert < experts &&
        expert_count[expert] <= MOE_PREFILL_MAX_EXPERT_ROWS;
    const int64_t scale_ptr = valid_expert ? scale_table[expert] : 0;
    if (!scale_ptr)
    {
        if constexpr (FP32)
        {
            float* output_ptr = reinterpret_cast<float*>(output) + output_offset;
            for (int idx = threadIdx.x; idx < 128; idx += blockDim.x) output_ptr[idx] = 0.0f;
        }
        else
        {
            half* output_ptr = reinterpret_cast<half*>(output) + output_offset;
            for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
                output_ptr[idx] = __float2half_rn(0.0f);
        }
        return;
    }

    const half* scale = reinterpret_cast<const half*>(scale_ptr);
    const size_t input_row = token_input ? original_slot / MOE_TOP_K : matrix;
    const size_t input_offset = input_row * width + blockIdx.y * 128;
    if constexpr (FP32)
        had_ff_r_128_inner<PRE_SCALE, !PRE_SCALE>
        (
            reinterpret_cast<const float*>(input) + input_offset,
            reinterpret_cast<float*>(output) + output_offset,
            scale,
            HAD_SCALE
        );
    else
        had_hf_r_128_inner<PRE_SCALE, !PRE_SCALE>
        (
            reinterpret_cast<const half*>(input) + input_offset,
            reinterpret_cast<half*>(output) + output_offset,
            scale,
            HAD_SCALE
        );
}

template <bool FP32, bool TWO_PROJECTIONS, int CFG>
__global__ __launch_bounds__(CFG == 0 ? 512 : CFG == 1 ? 256 : 128)
void moe_prefill_grouped_gemv_k3_kernel
(
    const half* A,
    const int64_t* expert_offsets,
    const int* expert_chunks,
    const int* num_chunks,
    const int64_t* B_table_0,
    const int64_t* B_table_1,
    void* C,
    int size_k,
    int size_n,
    int experts,
    int assignments
)
{
    if (blockIdx.y >= *num_chunks) return;
    const int expert_chunk = expert_chunks[blockIdx.y];
    const int expert = expert_chunk / MOE_PREFILL_CHUNKS_PER_EXPERT;
    const int chunk = expert_chunk % MOE_PREFILL_CHUNKS_PER_EXPERT;

    const int64_t start = expert_offsets[expert];
    const int64_t end = expert_offsets[expert + 1];
    const int64_t row0 = start + chunk * MOE_PREFILL_ROWS_PER_CHUNK;
    const int rows = min((int64_t) MOE_PREFILL_ROWS_PER_CHUNK, end - row0);
    if (rows <= 0) return;

    const int projection = TWO_PROJECTIONS ? blockIdx.z : 0;
    const int64_t* B_table = projection == 0 ? B_table_0 : B_table_1;
    const int64_t B_ptr = B_table[expert];
    const size_t matrix = (size_t) projection * assignments + row0;
    void* C_rows = FP32
        ? static_cast<void*>(reinterpret_cast<float*>(C) + matrix * size_n)
        : static_cast<void*>(reinterpret_cast<half*>(C) + matrix * size_n);
    if (!B_ptr)
    {
        constexpr int cols = CFG == 0 ? 32 : 64;
        for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x)
        {
            const int row = idx / cols;
            const int col = blockIdx.x * cols + idx % cols;
            if (col < size_n)
            {
                if constexpr (FP32)
                    reinterpret_cast<float*>(C_rows)[(size_t) row * size_n + col] = 0.0f;
                else
                    reinterpret_cast<half*>(C_rows)[(size_t) row * size_n + col] = __float2half_rn(0.0f);
            }
        }
        return;
    }

    exl3_gemv_kernel_body<3, FP32, 2, 2, CFG, true>
    (
        A + matrix * size_k,
        reinterpret_cast<const uint16_t*>(B_ptr),
        C_rows,
        rows,
        size_k,
        size_n,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );
}

__global__ void moe_prefill_weighted_reduce_kernel
(
    const float* sorted_rows,
    const int64_t* selected,
    const half* weights,
    const int64_t* inverse_order,
    const int64_t* expert_count,
    float* output,
    int rows,
    int experts,
    int width
)
{
    const int row = blockIdx.y;
    if (row >= rows) return;
    selected += row * MOE_TOP_K;
    weights += row * MOE_TOP_K;
    output += (size_t) row * width;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x;
         col < width; col += blockDim.x * gridDim.x)
    {
        unsigned used = 0;
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < MOE_TOP_K; ++rank)
        {
            int best_slot = -1;
            int64_t best_expert = INT64_MAX;
            #pragma unroll
            for (int slot = 0; slot < MOE_TOP_K; ++slot)
            {
                const bool available = !(used & (1u << slot));
                const int64_t expert = selected[slot];
                if (available && (expert < best_expert ||
                    (expert == best_expert && (best_slot < 0 || slot < best_slot))))
                {
                    best_expert = expert;
                    best_slot = slot;
                }
            }
            used |= 1u << best_slot;
            if (best_expert >= 0 && best_expert < experts &&
                expert_count[best_expert] <= MOE_PREFILL_MAX_EXPERT_ROWS)
            {
                const int64_t original_slot = (int64_t) row * MOE_TOP_K + best_slot;
                const int64_t sorted_slot = inverse_order[original_slot];
                if (sorted_slot >= 0 && sorted_slot < (int64_t) rows * MOE_TOP_K)
                {
                    const float weighted = __fmul_rn(
                        sorted_rows[(size_t) sorted_slot * width + col],
                        __half2float(weights[best_slot]));
                    sum = __fadd_rn(sum, weighted);
                }
            }
        }
        output[col] = sum;
    }
}

void check_ptr_table
(
    const at::Tensor& table,
    int64_t experts,
    const at::Device& device,
    const char* name
)
{
    TORCH_CHECK_DTYPE(table, kLong);
    TORCH_CHECK(table.device() == device && table.is_contiguous() && table.dim() == 1 &&
                table.numel() == experts, name, " must be a contiguous same-device pointer table");
}

void check_moe_alignment(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "exl3_moe_gfx12_k3: ", name, " must be 16-byte aligned");
}

} // namespace

// Launch helpers for the grouped decode path. K (the experts' trellis bitrate) is a template
// parameter of the grouped GEMV, so the entry point dispatches on it; MOE_CFG stays the runtime
// knob it always was. The Qwen3.8 path keeps its exact instantiations (K=3, top_k=10, MOE_CFG
// from the environment), so its code is unchanged.
// Dedup metadata (sorted expert runs), device-resident; nullptr selects the per-assignment path
struct MoeDedup
{
    const int64_t* sorted_expert;
    const int* token_row;
    const int* pos_of;
    const int* group_start;
    const int* num_groups;
};

template <int K, bool HALF, bool FP32, bool TWO, int CFG>
static void launch_moe_grouped_one
(
    dim3 grid, const half* A, const int64_t* selected, const MoeDedup* dd,
    const int64_t* t0, const int64_t* t1, void* C,
    int size_k, int size_n, int experts, int assignments, hipStream_t stream
)
{
    if (dd)
        moe_dedup_gemv_kernel<K, HALF, FP32, TWO, CFG><<<grid, moe_decode_threads(CFG), 0, stream>>>
        (A, dd->sorted_expert, dd->group_start, dd->num_groups, t0, t1, C, size_k, size_n, experts, assignments);
    else
        moe_grouped_gemv_k3_kernel<K, HALF, FP32, TWO, CFG><<<grid, moe_decode_threads(CFG), 0, stream>>>
        (A, selected, t0, t1, C, size_k, size_n, experts, assignments);
}

// K (the experts' trellis bitrate, integer or half-integer) is a template parameter of the
// grouped GEMV, so the entry point dispatches on it; MOE_CFG stays the runtime knob it always was.
// EXL3_MOE_VALU=1: VALU decode matvec (exl3_moe_valu.cuh) for K2/K4 experts instead of the WMMA body. Default off.
static int g_moe_valu_override = -1;   // test hook (exl3_moe_valu_override): -1 = env, 0/1 = forced
static inline bool moe_valu_enabled()
{
    if (g_moe_valu_override >= 0) return g_moe_valu_override != 0;
    static const bool on = [] { const char* e = getenv("EXL3_MOE_VALU"); return e && atoi(e) != 0; }();
    return on;
}
void exl3_moe_valu_override(int64_t v) { g_moe_valu_override = (int) v; }

template <int K, bool FP32, bool TWO>
static void launch_moe_valu
(
    const half* A, const int64_t* selected, const MoeDedup* dd,
    const int64_t* t0, const int64_t* t1, void* C,
    int size_k, int size_n, int experts, int assignments, hipStream_t stream
)
{
    constexpr int W = EXL3_MOE_VALU_W;
    const dim3 grid(mv::grid_x(size_n / 16, W), assignments, TWO ? 2 : 1);
    const bool deep = mv::chunk(size_k / 16) >= 5;   // register ring of 5 slices, else 3
    const int64_t* sel = dd ? dd->sorted_expert : selected;
    const int* gs = dd ? dd->group_start : nullptr;
    const int* ng = dd ? dd->num_groups : nullptr;
#define MV_LAUNCH(DD, VU) \
    moe_valu_kernel<K, FP32, DD, TWO, VU><<<grid, W * 32, 0, stream>>>(A, sel, gs, ng, t0, t1, C, size_k, size_n, experts, assignments)
    if (dd) { if (deep) MV_LAUNCH(true, 5); else MV_LAUNCH(true, 3); }
    else    { if (deep) MV_LAUNCH(false, 5); else MV_LAUNCH(false, 3); }
#undef MV_LAUNCH
}

template <int K, bool HALF, bool FP32, bool TWO>
static void launch_moe_grouped
(
    const half* A, const int64_t* selected, const MoeDedup* dd,
    const int64_t* t0, const int64_t* t1, void* C,
    int size_k, int size_n, int experts, int assignments, hipStream_t stream
)
{
    if constexpr (!HALF && (K == 2 || K == 4))
    {
        if (moe_valu_enabled() && size_k % 16 == 0 && size_n % 32 == 0)
        {
            launch_moe_valu<K, FP32, TWO>(A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream);
            return;
        }
    }
    dim3 grid(size_n / 32, assignments, TWO ? 2 : 1);
    switch (moe_decode_cfg())
    {
    case 1:
        launch_moe_grouped_one<K, HALF, FP32, TWO, 1>(grid, A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream);
        break;
    case 2:
        launch_moe_grouped_one<K, HALF, FP32, TWO, 2>(grid, A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream);
        break;
    default:
        launch_moe_grouped_one<K, HALF, FP32, TWO, 0>(grid, A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream);
        break;
    }
}

template <bool FP32, bool TWO>
static void launch_moe_grouped_k
(
    int k2, const half* A, const int64_t* selected, const MoeDedup* dd,
    const int64_t* t0, const int64_t* t1, void* C,
    int size_k, int size_n, int experts, int assignments, hipStream_t stream
)
{
    // k2 = 2 * K: 4 -> K2, 5 -> K2.5, 6 -> K3, 8 -> K4, 10 -> K5, 12 -> K6
    switch (k2)
    {
    case 4: launch_moe_grouped<2, false, FP32, TWO>(A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream); break;
    case 5: launch_moe_grouped<2, true,  FP32, TWO>(A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream); break;
    case 6: launch_moe_grouped<3, false, FP32, TWO>(A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream); break;
    case 8: launch_moe_grouped<4, false, FP32, TWO>(A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream); break;
    case 10: launch_moe_grouped<5, false, FP32, TWO>(A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream); break;
    case 12: launch_moe_grouped<6, false, FP32, TWO>(A, selected, dd, t0, t1, C, size_k, size_n, experts, assignments, stream); break;
    default: TORCH_CHECK(false, "exl3_moe_gfx12_k3: unsupported expert bitrate (K*2 = ", k2, ")");
    }
}

template <int TOP_K>
static void launch_moe_weighted_reduce
(
    const float* rows,
    const int64_t* selected,
    const half* weights,
    float* output,
    int width,
    int num_rows,
    const int* pos_of,
    hipStream_t stream
)
{
    moe_weighted_reduce_kernel<TOP_K><<<dim3(CEIL_DIVIDE(width, 256), num_rows), 256, 0, stream>>>
    (rows, selected, weights, output, width, pos_of);
}

void exl3_moe_gfx12_k3
(
    const at::Tensor& A,
    at::Tensor& output,
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
    at::Tensor& gu_had,
    at::Tensor& gu_out,
    at::Tensor& down_had,
    at::Tensor& down_out,
    double K,
    bool dedup
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int device;
    cuda_check(hipGetDevice(&device));
    // Despite the name, this path only launches exl3_gemv_kernel_body (ported to
    // every WMMA family) plus elementwise helpers with no arch intrinsics, so any
    // device with a WMMA GEMV can run it.
    TORCH_CHECK(exl3_gemv_wmma_family(device) != 0,
                "exl3_moe_gfx12_k3 requires a WMMA GEMV arch (gfx1200/1201 or gfx1150/1151/1152)");
    const int k2 = (int) (2.0 * K + 0.25);
    TORCH_CHECK(k2 == 4 || k2 == 5 || k2 == 6 || k2 == 8 || k2 == 10 || k2 == 12,
                "exl3_moe_gfx12_k3 expert bitrate K must be 2, 2.5, 3, 4, 5 or 6");

    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kHalf &&
                A.dim() == 2 && A.size(1) % 128 == 0 &&
                A.size(0) >= 1 && A.size(0) <= MOE_MAX_ROWS,
                "exl3_moe_gfx12_k3 requires contiguous fp16[R, hidden], R=1..16, hidden % 128 == 0");
    const int rows = A.size(0);
    const int hidden = (int) A.size(1);
    const int top_k = (int) selected.size(1);
    TORCH_CHECK(top_k == 8 || top_k == 10,
                "exl3_moe_gfx12_k3 top-k must be 8 (MiMo-V2) or 10 (Qwen3.8)");
    const int assignments = rows * top_k;
    TORCH_CHECK(output.device() == A.device() && output.is_contiguous() && output.dtype() == at::kFloat &&
                output.dim() == 2 && output.size(0) == rows && output.size(1) == hidden,
                "exl3_moe_gfx12_k3 output must be contiguous fp32[R, hidden]");
    TORCH_CHECK(selected.device() == A.device() && selected.is_contiguous() && selected.dtype() == at::kLong &&
                selected.dim() == 2 && selected.size(0) == rows,
                "exl3_moe_gfx12_k3 selected must be contiguous device int64[R, top_k]");
    TORCH_CHECK(weights.device() == A.device() && weights.is_contiguous() && weights.dtype() == at::kHalf &&
                weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == top_k,
                "exl3_moe_gfx12_k3 weights must be contiguous device fp16[R, top_k]");

    const int64_t experts = gate_trellis.numel();
    TORCH_CHECK(experts > 0, "exl3_moe_gfx12_k3 requires nonempty expert tables");
    check_ptr_table(gate_trellis, experts, A.device(), "gate_trellis");
    check_ptr_table(gate_suh, experts, A.device(), "gate_suh");
    check_ptr_table(gate_svh, experts, A.device(), "gate_svh");
    check_ptr_table(up_trellis, experts, A.device(), "up_trellis");
    check_ptr_table(up_suh, experts, A.device(), "up_suh");
    check_ptr_table(up_svh, experts, A.device(), "up_svh");
    check_ptr_table(down_trellis, experts, A.device(), "down_trellis");
    check_ptr_table(down_suh, experts, A.device(), "down_suh");
    check_ptr_table(down_svh, experts, A.device(), "down_svh");

    TORCH_CHECK(down_had.numel() % assignments == 0,
                "down_had has the wrong shape");
    const int intermediate = down_had.numel() / assignments;
    TORCH_CHECK(intermediate % 128 == 0 && intermediate >= 128,
                "exl3_moe_gfx12_k3 intermediate width must be a multiple of 128");
    TORCH_CHECK(gu_had.device() == A.device() && gu_had.is_contiguous() && gu_had.dtype() == at::kHalf &&
                gu_had.numel() == 2 * assignments * hidden,
                "gu_had must be contiguous fp16[2 * R * top_k, hidden]");
    TORCH_CHECK(gu_out.device() == A.device() && gu_out.is_contiguous() && gu_out.dtype() == at::kHalf &&
                gu_out.numel() == 2 * assignments * intermediate,
                "gu_out has the wrong shape");
    TORCH_CHECK(down_had.device() == A.device() && down_had.is_contiguous() && down_had.dtype() == at::kHalf &&
                down_had.numel() == assignments * intermediate,
                "down_had has the wrong shape");
    TORCH_CHECK(down_out.device() == A.device() && down_out.is_contiguous() && down_out.dtype() == at::kFloat &&
                down_out.numel() == assignments * hidden,
                "down_out must be contiguous fp32[R * top_k, hidden]");

    check_moe_alignment(A, "A");
    check_moe_alignment(output, "output");
    check_moe_alignment(gu_had, "gu_had");
    check_moe_alignment(gu_out, "gu_out");
    check_moe_alignment(down_had, "down_had");
    check_moe_alignment(down_out, "down_out");

    // MultiLinear constructs these same-device tables once and keeps their source tensors
    // alive. Entries remain an internal raw-pointer trust boundary: checking every pointer on
    // the host here would synchronize and copy all tables on every decode call.
    const int64_t* selected_ptr = reinterpret_cast<const int64_t*>(selected.data_ptr());
    const half* weights_ptr = reinterpret_cast<const half*>(weights.data_ptr());
    const int64_t* gt = reinterpret_cast<const int64_t*>(gate_trellis.data_ptr());
    const int64_t* gsuh = reinterpret_cast<const int64_t*>(gate_suh.data_ptr());
    const int64_t* gsvh = reinterpret_cast<const int64_t*>(gate_svh.data_ptr());
    const int64_t* ut = reinterpret_cast<const int64_t*>(up_trellis.data_ptr());
    const int64_t* usuh = reinterpret_cast<const int64_t*>(up_suh.data_ptr());
    const int64_t* usvh = reinterpret_cast<const int64_t*>(up_svh.data_ptr());
    const int64_t* dt = reinterpret_cast<const int64_t*>(down_trellis.data_ptr());
    const int64_t* dsuh = reinterpret_cast<const int64_t*>(down_suh.data_ptr());
    const int64_t* dsvh = reinterpret_cast<const int64_t*>(down_svh.data_ptr());

    // Expert dedup (verify batches): sort assignments by expert once, then every kernel below
    // works in sorted-slot space and the final reduce maps back. Groups hold <= rows tokens,
    // and the dedup GEMV handles <= 8 rows per expert (MMODE 1).
    MoeDedup dd_s {};
    const MoeDedup* dd = nullptr;
    at::Tensor meta_l, meta_i;
    const int64_t* sel_k = selected_ptr;   // expert per slot as the kernels below see it
    const int* token_row = nullptr;
    if (dedup && rows <= 8 && assignments <= 128)
    {
        meta_l = at::empty({assignments}, A.options().dtype(at::kLong));
        meta_i = at::empty({3 * assignments + 2}, A.options().dtype(at::kInt));
        int* mi = meta_i.data_ptr<int>();
        dd_s.sorted_expert = meta_l.data_ptr<int64_t>();
        dd_s.token_row = mi;
        dd_s.pos_of = mi + assignments;
        dd_s.group_start = mi + 2 * assignments;
        dd_s.num_groups = mi + 3 * assignments + 1;
        moe_dedup_metadata_kernel<<<1, 128, 0, stream>>>
        (selected_ptr, meta_l.data_ptr<int64_t>(), mi, mi + assignments, mi + 2 * assignments,
         mi + 3 * assignments + 1, assignments, top_k);
        dd = &dd_s;
        sel_k = dd_s.sorted_expert;
        token_row = dd_s.token_row;
    }

    dim3 had_gu_grid(2 * assignments, hidden / 128);
    moe_had_rows_kernel<true, false><<<had_gu_grid, 32, 0, stream>>>
    (A.data_ptr(), gu_had.data_ptr(), sel_k, gsuh, usuh,
     assignments, hidden, experts, true, top_k, token_row);

    launch_moe_grouped_k<false, true>(k2, reinterpret_cast<const half*>(gu_had.data_ptr()), sel_k, dd, gt, ut,
                                      gu_out.data_ptr(), hidden, intermediate, experts, assignments, stream);

    moe_had_rows_kernel<false, false><<<dim3(2 * assignments, intermediate / 128), 32, 0, stream>>>
    (gu_out.data_ptr(), gu_out.data_ptr(), sel_k, gsvh, usvh,
     assignments, intermediate, experts, false, top_k);

    const int activation_count = assignments * intermediate;
    const half* gu_ptr = reinterpret_cast<const half*>(gu_out.data_ptr());
    moe_silu_mul_kernel<<<CEIL_DIVIDE(activation_count, 256), 256, 0, stream>>>
    (gu_ptr, gu_ptr + activation_count, reinterpret_cast<half*>(down_had.data_ptr()), activation_count);

    moe_had_rows_kernel<true, false><<<dim3(assignments, intermediate / 128), 32, 0, stream>>>
    (down_had.data_ptr(), down_had.data_ptr(), sel_k, dsuh, dsuh,
     assignments, intermediate, experts, false, top_k);

    launch_moe_grouped_k<true, false>(k2, reinterpret_cast<const half*>(down_had.data_ptr()), sel_k, dd, dt, dt,
                                      down_out.data_ptr(), intermediate, hidden, experts, assignments, stream);

    moe_had_rows_kernel<false, true><<<dim3(assignments, hidden / 128), 32, 0, stream>>>
    (down_out.data_ptr(), down_out.data_ptr(), sel_k, dsvh, dsvh,
     assignments, hidden, experts, false, top_k);

    const int* pos_of = dd ? dd_s.pos_of : nullptr;
    switch (top_k)
    {
    case 8:
        launch_moe_weighted_reduce<8>(reinterpret_cast<const float*>(down_out.data_ptr()), selected_ptr,
                                      weights_ptr, reinterpret_cast<float*>(output.data_ptr()),
                                      hidden, rows, pos_of, stream);
        break;
    default:
        launch_moe_weighted_reduce<10>(reinterpret_cast<const float*>(down_out.data_ptr()), selected_ptr,
                                       weights_ptr, reinterpret_cast<float*>(output.data_ptr()),
                                       hidden, rows, pos_of, stream);
        break;
    }
}

void exl3_moe_gfx12_k3_prefill
(
    const at::Tensor& A,
    at::Tensor& output,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& order,
    const at::Tensor& expert_count,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    at::Tensor& gu_had,
    at::Tensor& gu_out,
    at::Tensor& down_out,
    at::Tensor& expert_offsets,
    at::Tensor& inverse_order,
    at::Tensor& expert_chunks,
    at::Tensor& chunk_count
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int device;
    cuda_check(hipGetDevice(&device));
    TORCH_CHECK(exl3_gemv_wmma_family(device) != 0,
                "exl3_moe_gfx12_k3_prefill requires a WMMA GEMV arch (gfx1200/1201 or gfx1150/1151/1152)");

    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kHalf &&
                A.dim() == 2 && A.size(1) == MOE_HIDDEN &&
                A.size(0) >= 2 && A.size(0) <= MOE_PREFILL_MAX_ROWS,
                "exl3_moe_gfx12_k3_prefill requires contiguous fp16[R, 2560], R=2..2048");
    const int rows = A.size(0);
    const int assignments = rows * MOE_TOP_K;
    TORCH_CHECK(output.device() == A.device() && output.is_contiguous() &&
                output.dtype() == at::kFloat && output.dim() == 2 &&
                output.size(0) == rows && output.size(1) == MOE_HIDDEN,
                "prefill output must be contiguous fp32[R, 2560]");
    TORCH_CHECK(selected.device() == A.device() && selected.is_contiguous() &&
                selected.dtype() == at::kLong && selected.dim() == 2 &&
                selected.size(0) == rows && selected.size(1) == MOE_TOP_K,
                "prefill selected must be contiguous device int64[R, 10]");
    TORCH_CHECK(weights.device() == A.device() && weights.is_contiguous() &&
                weights.dtype() == at::kHalf && weights.sizes() == selected.sizes(),
                "prefill weights must be contiguous device fp16[R, 10]");
    TORCH_CHECK(order.device() == A.device() && order.is_contiguous() &&
                order.dtype() == at::kLong && order.dim() == 1 && order.numel() == assignments,
                "prefill order must be contiguous device int64[R * 10]");

    const int64_t experts = gate_trellis.numel();
    TORCH_CHECK(experts > 0, "prefill requires nonempty expert tables");
    TORCH_CHECK(expert_count.device() == A.device() && expert_count.is_contiguous() &&
                expert_count.dtype() == at::kLong && expert_count.dim() == 1 &&
                expert_count.numel() == experts + 1,
                "prefill expert_count must be contiguous device int64[E + 1]");
    check_ptr_table(gate_trellis, experts, A.device(), "gate_trellis");
    check_ptr_table(gate_suh, experts, A.device(), "gate_suh");
    check_ptr_table(gate_svh, experts, A.device(), "gate_svh");
    check_ptr_table(up_trellis, experts, A.device(), "up_trellis");
    check_ptr_table(up_suh, experts, A.device(), "up_suh");
    check_ptr_table(up_svh, experts, A.device(), "up_svh");
    check_ptr_table(down_trellis, experts, A.device(), "down_trellis");
    check_ptr_table(down_suh, experts, A.device(), "down_suh");
    check_ptr_table(down_svh, experts, A.device(), "down_svh");

    TORCH_CHECK(gu_out.numel() % (2 * assignments) == 0, "prefill gu_out has the wrong shape");
    const int intermediate = gu_out.numel() / (2 * assignments);
    TORCH_CHECK(intermediate == 640 || intermediate == 768,
                "prefill intermediate width must be 640 or 768");
    TORCH_CHECK(gu_had.device() == A.device() && gu_had.is_contiguous() &&
                gu_had.dtype() == at::kHalf &&
                gu_had.numel() == 2 * assignments * MOE_HIDDEN,
                "prefill gu_had must be contiguous fp16[2 * R * 10, 2560]");
    TORCH_CHECK(gu_out.device() == A.device() && gu_out.is_contiguous() &&
                gu_out.dtype() == at::kHalf &&
                gu_out.numel() == 2 * assignments * intermediate,
                "prefill gu_out has the wrong shape");
    TORCH_CHECK(down_out.device() == A.device() && down_out.is_contiguous() &&
                down_out.dtype() == at::kFloat &&
                down_out.numel() == assignments * MOE_HIDDEN,
                "prefill down_out must be contiguous fp32[R * 10, 2560]");
    TORCH_CHECK(expert_offsets.device() == A.device() && expert_offsets.is_contiguous() &&
                expert_offsets.dtype() == at::kLong && expert_offsets.numel() == experts + 1,
                "prefill expert_offsets must be contiguous device int64[E + 1]");
    TORCH_CHECK(inverse_order.device() == A.device() && inverse_order.is_contiguous() &&
                inverse_order.dtype() == at::kLong && inverse_order.numel() == assignments,
                "prefill inverse_order must be contiguous device int64[R * 10]");
    TORCH_CHECK(expert_chunks.device() == A.device() && expert_chunks.is_contiguous() &&
                expert_chunks.dtype() == at::kInt &&
                expert_chunks.numel() >= experts * MOE_PREFILL_CHUNKS_PER_EXPERT,
                "prefill expert_chunks workspace is too small");
    TORCH_CHECK(chunk_count.device() == A.device() && chunk_count.is_contiguous() &&
                chunk_count.dtype() == at::kInt && chunk_count.numel() == 1,
                "prefill chunk_count must be a same-device int32 scalar workspace");

    check_moe_alignment(A, "A");
    check_moe_alignment(output, "output");
    check_moe_alignment(gu_had, "gu_had");
    check_moe_alignment(gu_out, "gu_out");
    check_moe_alignment(down_out, "down_out");

    const int64_t* selected_ptr = reinterpret_cast<const int64_t*>(selected.data_ptr());
    const half* weights_ptr = reinterpret_cast<const half*>(weights.data_ptr());
    const int64_t* order_ptr = reinterpret_cast<const int64_t*>(order.data_ptr());
    const int64_t* counts_ptr = reinterpret_cast<const int64_t*>(expert_count.data_ptr());
    int64_t* offsets_ptr = reinterpret_cast<int64_t*>(expert_offsets.data_ptr());
    int64_t* inverse_ptr = reinterpret_cast<int64_t*>(inverse_order.data_ptr());
    int* chunks_ptr = reinterpret_cast<int*>(expert_chunks.data_ptr());
    int* chunk_count_ptr = reinterpret_cast<int*>(chunk_count.data_ptr());
    const int64_t* gt = reinterpret_cast<const int64_t*>(gate_trellis.data_ptr());
    const int64_t* gsuh = reinterpret_cast<const int64_t*>(gate_suh.data_ptr());
    const int64_t* gsvh = reinterpret_cast<const int64_t*>(gate_svh.data_ptr());
    const int64_t* ut = reinterpret_cast<const int64_t*>(up_trellis.data_ptr());
    const int64_t* usuh = reinterpret_cast<const int64_t*>(up_suh.data_ptr());
    const int64_t* usvh = reinterpret_cast<const int64_t*>(up_svh.data_ptr());
    const int64_t* dt = reinterpret_cast<const int64_t*>(down_trellis.data_ptr());
    const int64_t* dsuh = reinterpret_cast<const int64_t*>(down_suh.data_ptr());
    const int64_t* dsvh = reinterpret_cast<const int64_t*>(down_svh.data_ptr());

    cuda_check(hipMemsetAsync(inverse_ptr, 0xff, assignments * sizeof(int64_t), stream));
    moe_prefill_metadata_kernel<<<CEIL_DIVIDE(assignments, 256), 256, 0, stream>>>
    (counts_ptr, order_ptr, offsets_ptr, inverse_ptr, chunks_ptr, chunk_count_ptr,
     experts, assignments);

    dim3 had_gu_grid(2 * assignments, MOE_HIDDEN / 128);
    moe_prefill_had_rows_kernel<true, false><<<had_gu_grid, 32, 0, stream>>>
    (A.data_ptr(), gu_had.data_ptr(), selected_ptr, order_ptr, counts_ptr, gsuh, usuh,
     assignments, MOE_HIDDEN, experts, true);

    // num_chunks is computed on device (no host readback). Two host-side upper
    // bounds are always valid: sum_e ceil(count_e/16) <= ceil(A/16) + experts and
    // sum_e ceil(count_e/16) <= (A + 15*A_active)/16 <= A. At small assignment
    // counts (decode rows 2-5) the second is far tighter — at rows=2 it is 20
    // slots vs 514 — so take the smaller of the two; large-A prefill keeps the
    // old bound.
    const int chunk_slots = std::min(
        CEIL_DIVIDE(assignments, MOE_PREFILL_ROWS_PER_CHUNK) + experts, assignments);
    const int pcfg = moe_prefill_cfg();
    const int pcols = pcfg == 0 ? 32 : 64;          // COLS = WNT * 16 in exl3_gemv_kernel_body
    dim3 gu_grid(intermediate / pcols, chunk_slots, 2);
    #define PREFILL_GU_ARGS \
        reinterpret_cast<const half*>(gu_had.data_ptr()), offsets_ptr, chunks_ptr, chunk_count_ptr, \
        gt, ut, gu_out.data_ptr(), MOE_HIDDEN, intermediate, experts, assignments
    switch (pcfg)
    {
        case 0: moe_prefill_grouped_gemv_k3_kernel<false, true, 0><<<gu_grid, 512, 0, stream>>>(PREFILL_GU_ARGS); break;
        case 1: moe_prefill_grouped_gemv_k3_kernel<false, true, 1><<<gu_grid, 256, 0, stream>>>(PREFILL_GU_ARGS); break;
        default: moe_prefill_grouped_gemv_k3_kernel<false, true, 2><<<gu_grid, 128, 0, stream>>>(PREFILL_GU_ARGS); break;
    }
    #undef PREFILL_GU_ARGS

    moe_prefill_had_rows_kernel<false, false>
        <<<dim3(2 * assignments, intermediate / 128), 32, 0, stream>>>
    (gu_out.data_ptr(), gu_out.data_ptr(), selected_ptr, order_ptr, counts_ptr, gsvh, usvh,
     assignments, intermediate, experts, false);

    const int activation_count = assignments * intermediate;
    half* gu_ptr = reinterpret_cast<half*>(gu_out.data_ptr());
    moe_silu_mul_kernel<<<CEIL_DIVIDE(activation_count, 256), 256, 0, stream>>>
    (gu_ptr, gu_ptr + activation_count, gu_ptr, activation_count);

    moe_prefill_had_rows_kernel<true, false>
        <<<dim3(assignments, intermediate / 128), 32, 0, stream>>>
    (gu_out.data_ptr(), gu_out.data_ptr(), selected_ptr, order_ptr, counts_ptr, dsuh, dsuh,
     assignments, intermediate, experts, false);

    dim3 down_grid(MOE_HIDDEN / pcols, chunk_slots, 1);
    #define PREFILL_DN_ARGS \
        reinterpret_cast<const half*>(gu_out.data_ptr()), offsets_ptr, chunks_ptr, chunk_count_ptr, \
        dt, dt, down_out.data_ptr(), intermediate, MOE_HIDDEN, experts, assignments
    switch (pcfg)
    {
        case 0: moe_prefill_grouped_gemv_k3_kernel<true, false, 0><<<down_grid, 512, 0, stream>>>(PREFILL_DN_ARGS); break;
        case 1: moe_prefill_grouped_gemv_k3_kernel<true, false, 1><<<down_grid, 256, 0, stream>>>(PREFILL_DN_ARGS); break;
        default: moe_prefill_grouped_gemv_k3_kernel<true, false, 2><<<down_grid, 128, 0, stream>>>(PREFILL_DN_ARGS); break;
    }
    #undef PREFILL_DN_ARGS

    moe_prefill_had_rows_kernel<false, true>
        <<<dim3(assignments, MOE_HIDDEN / 128), 32, 0, stream>>>
    (down_out.data_ptr(), down_out.data_ptr(), selected_ptr, order_ptr, counts_ptr, dsvh, dsvh,
     assignments, MOE_HIDDEN, experts, false);

    moe_prefill_weighted_reduce_kernel
        <<<dim3(CEIL_DIVIDE(MOE_HIDDEN, 256), rows), 256, 0, stream>>>
    (reinterpret_cast<const float*>(down_out.data_ptr()), selected_ptr, weights_ptr,
     inverse_ptr, counts_ptr, reinterpret_cast<float*>(output.data_ptr()),
     rows, experts, MOE_HIDDEN);
    cuda_check(hipPeekAtLastError());
}

#endif // USE_ROCM

static int exl3_gemv_env_mode()
{
    const char* env = std::getenv("EXL3_GEMV");
    if (!env) return 1;
    return atoi(env);
}

static bool exl3_gemv_debug()
{
    static const bool enabled = []
    {
        const char* env = std::getenv("EXL3_GEMV_DEBUG");
        return env && env[0] == '1' && env[1] == '\0';
    }();
    return enabled;
}

// Which WMMA family does this device have? The two are NOT interchangeable:
// different builtins, different A/B widths, different C row mapping.
//   0 = none, 1 = gfx12 (RDNA4), 2 = gfx11.5 (RDNA3.5 / Strix Halo)
int exl3_gemv_wmma_family(int device)
{
#if defined(USE_ROCM)
    static std::map<int, int> family_cache;
    static std::mutex family_cache_mtx;
    std::lock_guard<std::mutex> lock(family_cache_mtx);
    auto it = family_cache.find(device);
    if (it != family_cache.end()) return it->second;

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) return 0;
    const char* arch = prop.gcnArchName;
    auto is = [&](const char* name) {
        const size_t n = std::strlen(name);
        return !std::strncmp(arch, name, n) && (arch[n] == '\0' || arch[n] == ':');
    };
    int family = 0;
    if (is("gfx1200") || is("gfx1201")) family = 1;
    else if (is("gfx1150") || is("gfx1151") || is("gfx1152")) family = 2;
    family_cache[device] = family;
    return family;
#else
    (void) device;
    return 1;
#endif
}

// True when the EXL3 WMMA GEMV can run. NOTE: the gfx12-only MoE/routing entry
// points must check exl3_gemv_wmma_family(device) == 1 instead of this.
bool exl3_gemv_supported(int device)
{
#if defined(USE_ROCM)
    return exl3_gemv_wmma_family(device) != 0;
#else
    (void) device;
    return true;
#endif
}

// -1 = default per bits, 0 = force shuffle extraction, 1 = force smem staging (testing)
#if !defined(USE_ROCM)
static int exl3_gemv_env_smem()
{
    const char* env = std::getenv("EXL3_GEMV_SMEM");
    if (!env) return -1;
    return atoi(env);
}
#endif

// -1: not eligible, 0: narrow config, 1: wide config. narrow_coresident = number of narrow-config
// blocks that fit on the device at once (its grid is one block per 32 output columns)
static int exl3_gemv_cfg(int cc, int size_m, int size_k, int size_n, int K, int cb, int mode, int narrow_coresident)
{
    if (mode == 0) return -1;
#if defined(USE_ROCM)
    if (K < 2 || K > 6) return -1;
#else
    if (K < 2 || K > 4) return -1;
#endif
    if (K != 4 && cb == 0) return -1;
#if defined(USE_ROCM)
    // K2 has no HIP MMODE-2 instantiation; its established envelope ends at eight rows.
    if (K == 2 && size_m > EXL3_GEMV_MAX_M) return -1;
    if (size_m > EXL3_GEMV_ROCM_MAX_M) return -1;
#else
    if (size_m > EXL3_GEMV_MAX_M) return -1;
#endif
    if (size_k % 128 || size_n % 128) return -1;
    //if (cc != CC_AMPERE) return -1;  // measured win on Ampere; Ada/Blackwell are memory-bound here
    if (mode == 2) return size_n <= 8192 ? 0 : 1;
    if (mode == 3) return 0;   // testing: force narrow config
    if (mode == 4) return 1;   // testing: force wide config

    // The narrow config wins (up to ~30%) whenever its grid fits in a single co-resident wave;
    // in the 1..2-wave zone the trailing partial wave costs more than the kernel gains unless
    // per-group work is small (small k). The wide config covers a band of large-n shapes with
    // small-to-mid k. Everything else runs the regular block-pipelined kernel.
    // Per-bits envelopes: 2 bpw is decode-bound and won at every measured shape on both archs;
    // 3 bpw wins everywhere on Ada but only in the narrow envelope on Ampere
    if (K == 2) return size_n <= 8192 ? 0 : 1;
    if (K == 3 && cc == CC_ADA) return size_n <= 8192 ? 0 : 1;
    if (size_n / 32 <= narrow_coresident) return 0;
    if (size_k <= 2048 && size_n <= 8192) return 0;
    if (K == 3) return -1;
    if (size_n >= 8192 && size_k <= 4096) return 1;
    if (size_n >= 8192 && size_n <= 10240 && size_k <= 5120 && cc == CC_AMPERE) return 1;
    return -1;
}

static void* exl3_gemv_select_kernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem)
{
#if defined(USE_ROCM)
    // HIP: only the smem-staged extraction is instantiated. gfx12 WMMA operands must not
    // be fed through __shfl_sync (hardware exception), so SMEM_STAGE is forced true and
    // the `smem` argument is ignored.
    #define SEL(bits_, cb_, fp32_, mm_, cfg_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, true>;
    #define SEL_GRID(bits_, cb_) \
        SEL(bits_, cb_, false, 0, 0) SEL(bits_, cb_, false, 0, 1) \
        SEL(bits_, cb_, false, 1, 0) SEL(bits_, cb_, false, 1, 1) \
        SEL(bits_, cb_, true,  0, 0) SEL(bits_, cb_, true,  0, 1) \
        SEL(bits_, cb_, true,  1, 0) SEL(bits_, cb_, true,  1, 1)
    #define SEL_MMODE2_GRID(bits_, cb_) \
        SEL(bits_, cb_, false, 2, 0) SEL(bits_, cb_, false, 2, 1) \
        SEL(bits_, cb_, true,  2, 0) SEL(bits_, cb_, true,  2, 1)
    SEL_GRID(4, 0) SEL_GRID(4, 1) SEL_GRID(4, 2)
    SEL_GRID(2, 1) SEL_GRID(2, 2)
    SEL_GRID(3, 1) SEL_GRID(3, 2)
    SEL_GRID(5, 1) SEL_GRID(5, 2)
    SEL_GRID(6, 1) SEL_GRID(6, 2)
    SEL_MMODE2_GRID(3, 2) SEL_MMODE2_GRID(4, 1) SEL_MMODE2_GRID(4, 2)
    SEL_MMODE2_GRID(5, 2) SEL_MMODE2_GRID(6, 1) SEL_MMODE2_GRID(6, 2)
    #undef SEL_MMODE2_GRID
    #undef SEL_GRID
    #undef SEL
#else
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
        SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
        SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
    SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
    SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
    SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
    #undef SEL_GRID
    #undef SEL
#endif
    return nullptr;
}

#if defined(USE_ROCM)
static void* exl3_gemv_select_kernel_half(int bits, bool c_fp32, int mmode, int cfg)
{
    #define SEL_HALF(bits_, fp32_, mm_, cfg_) \
        if (bits == bits_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, 2, mm_, cfg_, true, true>;
    #define SEL_HALF_GRID(bits_) \
        SEL_HALF(bits_, false, 0, 0) SEL_HALF(bits_, false, 0, 1) \
        SEL_HALF(bits_, false, 1, 0) SEL_HALF(bits_, false, 1, 1) \
        SEL_HALF(bits_, true,  0, 0) SEL_HALF(bits_, true,  0, 1) \
        SEL_HALF(bits_, true,  1, 0) SEL_HALF(bits_, true,  1, 1)
    #define SEL_HALF_MMODE2(bits_) \
        SEL_HALF(bits_, false, 2, 0) SEL_HALF(bits_, false, 2, 1) \
        SEL_HALF(bits_, true,  2, 0) SEL_HALF(bits_, true,  2, 1)
    SEL_HALF_GRID(1) SEL_HALF_GRID(2) SEL_HALF_GRID(3)
    SEL_HALF_MMODE2(1) SEL_HALF_MMODE2(2) SEL_HALF_MMODE2(3)
    #undef SEL_HALF_MMODE2
    #undef SEL_HALF_GRID
    #undef SEL_HALF
    return nullptr;
}
#endif

bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    bool has_su_sv,
    int device,
#if defined(USE_ROCM)
    hipStream_t stream,
#else
    cudaStream_t stream,
#endif
    void** launched_kernel,
    bool force
)
{
    // Free integer checks first; the env read (~64 ns) and device queries only run for calls
    // that could actually take this path
    if (!has_su_sv) return false;
    if (half_k)
    {
#if defined(USE_ROCM)
        if (K < 1 || K > 3 || cb != 2) return false;
#else
        return false;
#endif
    }
    else
    {
#if defined(USE_ROCM)
        if (K < 2 || K > 6) return false;
#else
        if (K < 2 || K > 4) return false;
#endif
        if (K != 4 && cb == 0) return false;
    }
#if defined(USE_ROCM)
    // Integer K2 has no HIP MMODE-2 instantiation; fractional kernels do.
    if (!half_k && K == 2 && size_m > EXL3_GEMV_MAX_M) return false;
    if (size_m > EXL3_GEMV_ROCM_MAX_M) return false;
#else
    if (size_m > EXL3_GEMV_MAX_M) return false;
#endif
    if (size_k % 128 || size_n % 128) return false;
#if defined(USE_ROCM)
    if (!exl3_gemv_supported(device)) return false;
#endif

    int mode = force ? 2 : exl3_gemv_env_mode();
    if (mode == 0) return false;
    int cc = DevCtx::instance().get_cc(device);
    // if (cc != CC_AMPERE) return false;
#if defined(USE_ROCM)
    int mmode = size_m == 1 ? 0 : (size_m <= EXL3_GEMV_MAX_M ? 1 : 2);
#else
    int mmode = size_m == 1 ? 0 : 1;
#endif
    int num_sms = DevCtx::instance().get_num_sms(device);

    // CUDA's cooperative grid is capped at full co-residency. HIP retains the same grid sizing
    // for the shape heuristic and grid-stride main loop, but uses an ordinary launch.
    static std::map<void*, int> occ_cache[MAX_DEVICES];
    auto& cache = occ_cache[device];
    auto occupancy = [&] (void* kernel, int block_dim) -> int
    {
        auto it = cache.find(kernel);
        if (it != cache.end()) return it->second;
        int blocks_per_sm;
#if defined(USE_ROCM)
        cuda_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
#else
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
#endif
        cache[kernel] = blocks_per_sm;
        if (exl3_gemv_debug())
            std::fprintf(stderr, "EXL3_GEMV_DEBUG device=%d kernel=%p block_dim=%d blocks_per_sm=%d\n",
                         device, kernel, block_dim, blocks_per_sm);
        return blocks_per_sm;
    };

#if defined(USE_ROCM)
    // Extraction style on HIP: smem staging is the only safe mode (gfx12 shfl->WMMA hazard),
    // so the EXL3_GEMV_SMEM override is ignored and staging is always on
    bool smem = true;
#else
    // Extraction style: shuffle by default, smem staging selectable per call for evaluation
    bool smem = exl3_gemv_env_smem() == 1;
#endif

    auto select = [&] (int cfg_) -> void*
    {
#if defined(USE_ROCM)
        if (half_k) return exl3_gemv_select_kernel_half(K, c_fp32, mmode, cfg_);
#endif
        return exl3_gemv_select_kernel(K, cb, c_fp32, mmode, cfg_, smem);
    };
    void* narrow_kernel = select(0);
    if (!narrow_kernel) return false;
    int narrow_coresident = occupancy(narrow_kernel, 512) * num_sms;

    int cfg = exl3_gemv_cfg(cc, size_m, size_k, size_n, half_k ? K + 1 : K, cb, mode, narrow_coresident);
    if (cfg < 0) return false;

    void* kernel = cfg == 0 ? narrow_kernel : select(cfg);
    if (!kernel) return false;

    int block_dim = cfg == 0 ? 512 : 256;
    int cols = cfg == 0 ? 32 : 64;

    int max_blocks = occupancy(kernel, block_dim) * num_sms;
    // Grid sizing: the body is a grid-stride loop and every group is processed exactly
    // once by one block with a block-local reduction, so output is bit-identical for any
    // grid >= 1 and grid size is a performance knob only. The CUDA path launches
    // cooperatively, which requires grid <= co-resident blocks, so the occupancy cap
    // stays there; the HIP launch is an ordinary stream-ordered launch (extra blocks
    // queue behind the co-resident ones), where the cap measurably throttles the
    // multi-row cells (M=4: -4.5..-10.7% per call, lm_head M=16: -15.5%; M=1 neutral).
#if defined(USE_ROCM)
    int grid = size_n / cols;
#else
    int grid = MIN(size_n / cols, max_blocks);
#endif
    (void) max_blocks;
    if (grid < 1) return false;

    cuda_check
    (
#if defined(USE_ROCM)
        hipLaunchKernel
#else
        cudaLaunchCooperativeKernel
#endif
        (
            kernel,
            dim3(grid),
            dim3(block_dim),
            kernel_args,
            0,
            stream
        )
    );

    if (launched_kernel) *launched_kernel = kernel;
    return true;
}

void exl3_gemv
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
#if defined(USE_ROCM)
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#else
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#endif

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK(A.dim() >= 1 && C.dim() >= 1, "exl3_gemv requires non-scalar A and C");
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "exl3_gemv requires contiguous A, B and C");
    TORCH_CHECK(B.device() == A.device() && C.device() == A.device(),
                "exl3_gemv requires A, B and C on the same device");

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    const int tile_u16 = B.size(2);
    const bool half_k = tile_u16 % 16 != 0;
    int K = tile_u16 / 16;
    TORCH_CHECK(!half_k || (tile_u16 % 16 == 8 && mul1),
                "exl3_gemv: half-integer bitrates require a valid MUL1 tensor");

    TORCH_CHECK(suh.has_value() && A_had.has_value() && svh.has_value(),
                "exl3_gemv requires suh, A_had and svh");
    const at::Tensor& suh_t = suh.value();
    const at::Tensor& A_had_t = A_had.value();
    const at::Tensor& svh_t = svh.value();
    TORCH_CHECK_DTYPE(suh_t, kHalf);
    TORCH_CHECK_DTYPE(A_had_t, kHalf);
    TORCH_CHECK_DTYPE(svh_t, kHalf);
    TORCH_CHECK(suh_t.device() == A.device() && A_had_t.device() == A.device() &&
                svh_t.device() == A.device(),
                "exl3_gemv requires workspaces on the A device");
    TORCH_CHECK(suh_t.is_contiguous() && A_had_t.is_contiguous() && svh_t.is_contiguous(),
                "exl3_gemv requires contiguous workspaces");
    TORCH_CHECK(suh_t.numel() == size_k, "exl3_gemv suh size mismatch");
    TORCH_CHECK(A_had_t.sizes() == A.sizes(), "exl3_gemv A_had shape mismatch");
    TORCH_CHECK(svh_t.numel() == size_n, "exl3_gemv svh size mismatch");

    const half* suh_ptr = (const half*) suh_t.data_ptr();
    half* A_had_ptr = (half*) A_had_t.data_ptr();
    const half* svh_ptr = (const half*) svh_t.data_ptr();

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int device;
#if defined(USE_ROCM)
    hipGetDevice(&device);
#else
    cudaGetDevice(&device);
#endif
    int* locks = DevCtx::instance().get_locks(device);

#if defined(USE_ROCM)
    // Same gate as the CUDA exl3_gemm call site: the fused int8 kernel does its own input Hadamard
    // (from A with suh) and output Hadamard (svh), so it must run BEFORE the had_r_128 pre-pass
    // below and returns the finished C. Off unless EXL3_INT8_GEMV is set to 1 or 2 on ROCm
    if (!half_k && mul1 && exl3_gemv_int8_enabled())
    {
        if (exl3_gemv_int8(A, B, C, suh, A_had, svh, stream, nullptr))
        {
            cuda_check(hipPeekAtLastError());
            return;
        }
    }
    at::Tensor A_view = A.view({size_m, size_k});
    at::Tensor A_had_view = A_had.value().view({size_m, size_k});
    had_r_128(A_view, A_had_view, suh, c10::nullopt, 1.0f);
    const half* A_ptr = (const half*) A_had_view.data_ptr();
#else
    const half* A_ptr = (const half*) A.data_ptr();
#endif
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    void* kernel_args[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };

    bool ok = exl3_gemv_try_launch
    (
        kernel_args, size_m, size_k, size_n, K, half_k, cb, c_fp32,
        true, device, stream, nullptr, true
    );
    TORCH_CHECK(ok, "exl3_gemv: call is not eligible for the GEMV kernel");

#if defined(USE_ROCM)
    at::Tensor C_view = C.view({size_m, size_n});
    had_r_128(C_view, C_view, c10::nullopt, svh, 1.0f);
    cuda_check(hipPeekAtLastError());
#else
    cuda_check(cudaPeekAtLastError());
#endif
}
