#include <cuda_fp16.h>
#include "quantize.cuh"
#include <array>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <map>
#include <cstdlib>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"
#include "exl3_devctx.cuh"
#include <cmath>
#include <string>

#define H_INF __ushort_as_half(0x7c00)

#include "comp_units/quantize_tiles_instances.cuh"
#include "quantize_tiles_kernel.cuh"

#define __(i, cb) quantize_tiles_kernel_k##i##_cb##cb()
static const std::array<fp_quantize_tiles_kernel, 24> quantize_tiles_kernel_instances
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

// 160-length rows (n-gram embedding vectors), mul1 codebook only
#define __(i) quantize_tiles_kernel_k##i##_cb2_l160()
static const std::array<fp_quantize_tiles_kernel, 8> quantize_tiles_kernel_instances_l160
{
    __(1), __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

// Keep original instances for architectures outside the sm_120 tuning target.
#define __(i, cb) quantize_tiles_kernel_k##i##_cb##cb(true)
static const std::array<fp_quantize_tiles_kernel, 24> quantize_tiles_optimized_instances
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

// 160-length rows (n-gram embedding vectors), mul1 codebook only
#define __(i) quantize_tiles_kernel_k##i##_cb2_l160(true)
static const std::array<fp_quantize_tiles_kernel, 8> quantize_tiles_optimized_instances_l160
{
    __(1), __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __


template <int cb>
__global__ void quantize_codebook_kernel(half* table)
{
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    table[state] = decode_3inst<cb>(state);
}

static const half2* quantize_codebook(int device, int cb, const at::Tensor& input)
{
    // Per-host-thread cache; the event permits safe reuse on another CUDA stream.
    struct Entry { at::Tensor table; at::cuda::CUDAEvent ready; };
    thread_local std::map<std::pair<int, int>, Entry> tables;
    auto& entry = tables[{device, cb}];
    auto stream = at::cuda::getCurrentCUDAStream();
    if (!entry.table.defined())
    {
        entry.table = at::empty({65536}, input.options().dtype(at::kHalf));
        auto* ptr = reinterpret_cast<half*>(entry.table.data_ptr());
        if (cb == 0) quantize_codebook_kernel<0><<<256, 256, 0, stream.stream()>>>(ptr);
        if (cb == 1) quantize_codebook_kernel<1><<<256, 256, 0, stream.stream()>>>(ptr);
        if (cb == 2) quantize_codebook_kernel<2><<<256, 256, 0, stream.stream()>>>(ptr);
        cuda_check(cudaPeekAtLastError());
        entry.ready.record(stream);
    }
    else entry.ready.block(stream);
    // Protect outstanding uses if the host thread exits and releases its cached tensor.
    c10::cuda::CUDACachingAllocator::recordStream(entry.table.storage().data_ptr(), stream);
    return reinterpret_cast<const half2*>(entry.table.data_ptr());
}


// Which architectures run the dense specializations (quantize_tiles_optimized.cuh) for a given K
// and codebook. On Ada and Ampere, the register-cached decoded values and the global codebook table
// do not pay off for K=3 and K=6. K=7 depends on the codebook.
// EXL3_QT_OPTIMIZED=1/0 forces the choice for testing purposes. The Python scratch allocation asks the
// extension through quantize_tiles_scratch, so the layouts always agree
bool quantize_tiles_use_optimized(int major, int minor, int K, int cb)
{
    if (const char* env = std::getenv("EXL3_QT_OPTIMIZED"))
        return env[0] == '1';
    if (major == 12) return true;
    if (major == 8 && minor == 9)
        return K == 1 || K == 2 || K == 4 || K == 5 || (K == 7 && cb == 0) || K == 8;
    if (major == 8 && minor == 6)
        return K == 1 || K == 2 || K == 4 || K == 5 || (K == 7 && cb != 2) || K == 8;
    return false;
}

#if defined(USE_ROCM)
#include "quantize_tiles_rdna.cuh"

template <int K, int cb, int NT>
static fp_quantize_tiles_kernel qt_rdna_nt() { return quantize_tiles_rdna_kernel<K, cb, 256, NT>; }

// Grouped gfx11 kernel instances (L = 256). Block sizes per K: the default first, alternatives
// selectable with EXL3_QT_RDNA_NT for tuning
template <int cb>
static fp_quantize_tiles_kernel qt_rdna_instance(int K, int nt)
{
    if (K == 2) return nt == 512 ? qt_rdna_nt<2, cb, 512>() : qt_rdna_nt<2, cb, 1024>();
    if (K == 3) return nt == 256 ? qt_rdna_nt<3, cb, 256>() : nt == 512 ? qt_rdna_nt<3, cb, 512>() : qt_rdna_nt<3, cb, 1024>();
    return qt_rdna_nt<4, cb, 256>();
}

static int qt_rdna_shmem_for(int K)
{
    return K == 2 ? qt_rdna_shmem<2, 256>() : K == 3 ? qt_rdna_shmem<3, 256>() : qt_rdna_shmem<4, 256>();
}

static int64_t qt_rdna_history_for(int K)
{
    return K == 2 ? qt_rdna_history_bytes<2, 256>() : K == 3 ? qt_rdna_history_bytes<3, 256>() : qt_rdna_history_bytes<4, 256>();
}
#endif

enum { QT_BASE = 0, QT_OPTIMIZED = 1, QT_RDNA = 2 };

// Kernel family for (K, codebook, L). EXL3_QT_KERNEL=base/opt/rdna forces one for testing (rdna
// falls back to base where it has no instance); EXL3_QT_OPTIMIZED=1/0 keeps its old meaning.
// On ROCm the grouped kernel is the default for K = 2..3, the dense specialization for K = 4..5.
// All are measured bit-identical to the base kernel on gfx1151. K = 4 routes to dense because the
// grouped kernel only matches base there while dense is ~2.5-3x base (measured for K = 2..3
// and K = 5 as well).
static int quantize_tiles_mode(int major, int minor, int K, int cb, int L)
{
    if (const char* env = std::getenv("EXL3_QT_KERNEL"))
    {
        std::string v(env);
        if (v == "opt") return QT_OPTIMIZED;
        if (v == "base") return QT_BASE;
#if defined(USE_ROCM)
        if (v == "rdna") return (L == 256 && K >= 2 && K <= 4) ? QT_RDNA : QT_BASE;
#endif
    }
    if (std::getenv("EXL3_QT_OPTIMIZED"))
        return quantize_tiles_use_optimized(major, minor, K, cb) ? QT_OPTIMIZED : QT_BASE;
#if defined(USE_ROCM)
    if (L == 256 && K >= 2 && K <= 3) return QT_RDNA;
    if (K == 4 || K == 5) return QT_OPTIMIZED;
    return QT_BASE;
#else
    return quantize_tiles_use_optimized(major, minor, K, cb) ? QT_OPTIMIZED : QT_BASE;
#endif
}

// Kernel and launch geometry for one (K, codebook, L) on the current device: which implementation,
// its block size (read back from the compiled kernel, so a PTX-JIT'd instance launches with the
// size it was built for), dynamic shared memory and the resident blocks per SM
struct QtLaunch
{
    int mode;
    fp_quantize_tiles_kernel kernel;
    int num_threads;
    int shmem;
    int blocks_per_sm;
    int64_t history_bytes;
};

static QtLaunch qt_launch(int device, int K, int cb, int L)
{
    const auto* props = at::cuda::getDeviceProperties(device);
    const int mode = quantize_tiles_mode(props->major, props->minor, K, cb, L);
    const bool optimized = mode == QT_OPTIMIZED;
    const int edges = 65536 >> K;
    int shmem;
    fp_quantize_tiles_kernel kernel;
    int64_t history_bytes;
#if defined(USE_ROCM)
    if (mode == QT_RDNA)
    {
        const char* nt_env = std::getenv("EXL3_QT_RDNA_NT");
        const int nt = nt_env ? std::atoi(nt_env) : 0;
        kernel = cb == 0 ? qt_rdna_instance<0>(K, nt) : cb == 1 ? qt_rdna_instance<1>(K, nt) : qt_rdna_instance<2>(K, nt);
        shmem = qt_rdna_shmem_for(K);
        history_bytes = qt_rdna_history_for(K);
    }
    else
#endif
    {
#if defined(USE_ROCM)
        // gfx1151's 64 KiB LDS cannot hold K=1/2 trellis costs plus the input tile and traceback scratch.
        // Both HIP kernels use the caller-provided global cost buffers at those K values.
        const int cost_arrays = K >= 3 ? 2 : 0;
#else
        const int cost_arrays = optimized && K == 1 ? 1 : (K >= 2 ? 2 : 0);
#endif
        shmem = cost_arrays * edges * sizeof(half) + L * sizeof(half) + 64 + 128;
        const auto& instances = optimized ? quantize_tiles_optimized_instances : quantize_tiles_kernel_instances;
        const auto& instances_l160 = optimized ? quantize_tiles_optimized_instances_l160 : quantize_tiles_kernel_instances_l160;
        kernel = L == 256 ? instances[K - 1 + 8 * cb] : instances_l160[K - 1];
        history_bytes = optimized ? (int64_t) L * edges / (K == 1 ? 8 : 1) : (int64_t) L * edges * 2;
    }
    cuda_check(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    cudaFuncAttributes attr;
    cuda_check(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel)));
    int blocks_per_sm;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, attr.maxThreadsPerBlock, shmem));
    return {mode, kernel, attr.maxThreadsPerBlock, shmem, blocks_per_sm, history_bytes};
}

// For the Python scratch allocation: whether the dense specialization runs for this configuration
// (byte / bit history layout) and the number of tiles one wave keeps resident
std::tuple<bool, int64_t> quantize_tiles_scratch(int device, int K, bool mcg, bool mul1, int L)
{
    const c10::cuda::CUDAGuard device_guard(device);
    TORCH_CHECK(K >= 1 && K <= 8, "quantize_tiles_scratch: K must be 1..8");
    TORCH_CHECK(L == 256 || (L == 160 && mul1), "quantize_tiles_scratch: length 160 requires the mul1 codebook");
    auto launch = qt_launch(device, K, mul1 ? 2 : mcg ? 1 : 0, L);
    return {launch.mode != QT_BASE, (int64_t) launch.blocks_per_sm * DevCtx::instance().get_num_sms(device)};
}


void quantize_tiles
(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int K,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input_tiles.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(input_tiles, 2);
    const int L = input_tiles.size(1);
    TORCH_CHECK(L == 256 || L == 160, "quantize_tiles tile length must be 256 or 160");
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_indices);
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_tiles);
    TORCH_CHECK_DTYPE(input_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_indices, kShort);
    TORCH_CHECK(K >= 1 && K <= 8, "quantize_tiles K must be in range 1..8");

    const int edges = 65536 >> K;
    const int num_tiles = input_tiles.size(0);
    if (!num_tiles) return;

    for (const auto& tensor : {input_tiles, output_tiles, output_indices, temp_costs, temp_edges})
    {
        TORCH_CHECK(tensor.device() == input_tiles.device(), "quantize_tiles tensors must share a device");
        TORCH_CHECK(tensor.is_contiguous(), "quantize_tiles tensors must be contiguous");
    }
    TORCH_CHECK_DTYPE(temp_costs, kHalf);
    TORCH_CHECK(temp_costs.numel() > 0, "quantize_tiles requires nonempty cost scratch");
    int device;
    cuda_check(cudaGetDevice(&device));
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;
    TORCH_CHECK(L == 256 || cb == 2, "quantize_tiles length 160 requires the mul1 codebook");
    const auto launch = qt_launch(device, K, cb, L);
    const bool optimized = launch.mode == QT_OPTIMIZED;
    const bool byte_layout = launch.mode != QT_BASE;
    const auto kernel = launch.kernel;
    const int num_threads = launch.num_threads;
    const int shmem = launch.shmem;
    const int blocks_per_sm = launch.blocks_per_sm;
    int64_t scratch_tiles;
    if (byte_layout)
    {
        TORCH_CHECK(temp_edges.scalar_type() == at::kByte || temp_edges.scalar_type() == at::kShort,
                    "quantize_tiles optimized history must be byte or short storage");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(temp_edges.data_ptr()) % 8 == 0,
                    "quantize_tiles history has insufficient alignment");
        scratch_tiles = temp_edges.numel() * temp_edges.element_size() / launch.history_bytes;
    }
    else
    {
        TORCH_CHECK_DIM(temp_costs, 3);
        TORCH_CHECK_SIZE(temp_costs, 1, 2);
        TORCH_CHECK_SIZE(temp_costs, 2, edges);
        TORCH_CHECK_DTYPE(temp_edges, kShort);
        TORCH_CHECK_DIM(temp_edges, 3);
        TORCH_CHECK_SIZE(temp_edges, 1, L);
        TORCH_CHECK_SIZE(temp_edges, 2, edges);
        scratch_tiles = MIN(temp_costs.size(0), temp_edges.size(0));
    }
    const int max_batch_size = (int) MIN(scratch_tiles, (int64_t) blocks_per_sm * DevCtx::instance().get_num_sms(device));
    TORCH_CHECK(max_batch_size > 0, "quantize_tiles scratch must hold at least one tile");
    const half2* lut = optimized && K == 6 ? quantize_codebook(device, cb, input_tiles) : nullptr;

    for (int batch_i = 0; batch_i < num_tiles; batch_i += max_batch_size)
    {
        const int bsz = MIN(max_batch_size, num_tiles - batch_i);
        kernel<<<bsz, num_threads, shmem, stream>>>
        (
            ((const float*) input_tiles.data_ptr()) + (int64_t) L * batch_i,
            ((float*) output_tiles.data_ptr()) + (int64_t) L * batch_i,
            ((uint16_t*) output_indices.data_ptr()) + (int64_t) L * batch_i,
            (half*) temp_costs.data_ptr(),
            (uint16_t*) temp_edges.data_ptr(),
            lut
        );
        cuda_check(cudaPeekAtLastError());
    }
}

template <typename T>
__global__ //__launch_bounds__(64)
void decode_kernel
(
    const uint16_t* __restrict__ input_tiles_ptr,
    T* __restrict__ output_tiles_ptr,
    int cols,
    bool mcg,
    bool mul1
)
{
    int col = threadIdx.x + blockIdx.x * 64;
    if (col >= cols) return;
    int row = blockIdx.y;
    int idx = row * cols + col;

    uint32_t enc = (uint32_t) input_tiles_ptr[idx];
    half w;
    if (mcg)
        w = decode_3inst<1>(enc);
    else if (mul1)
        w = decode_3inst<2>(enc);
    else
        w = decode_3inst<0>(enc);

    if constexpr (std::is_same_v<T, float>)
        output_tiles_ptr[idx] = __half2float(w);
    else
        output_tiles_ptr[idx] = w;
}

/*
Decode tensor

input_indices: uint16_t
output_tiles: float or half
mcg: use mcg codebook
mul1: use mcg codebook
*/

void decode
(
    at::Tensor input_indices,
    at::Tensor output_tiles,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input_indices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(input_indices, 2);
    TORCH_CHECK_SHAPES_FULL(input_indices, output_tiles);
    TORCH_CHECK_DTYPE(input_indices, kShort);

    int rows = input_indices.size(0);
    int cols = input_indices.size(1);

    dim3 blockDim(64);
    dim3 gridDim(CEIL_DIVIDE(cols, 64), rows);

    if (output_tiles.dtype() == at::kFloat)
        decode_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const uint16_t*) input_indices.data_ptr(),
            (float*) output_tiles.data_ptr(),
            cols,
            mcg,
            mul1
        );
    else if (output_tiles.dtype() == at::kHalf)
        decode_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const uint16_t*) input_indices.data_ptr(),
            (half*) output_tiles.data_ptr(),
            cols,
            mcg,
            mul1
        );
}


#define NUM_THREADS_TD 1024
#define MAX_BINS 1024

__global__ __launch_bounds__(NUM_THREADS_TD)
void test_distribution_kernel
(
    const float* __restrict__ input_ptr,
    float* __restrict__ dist_output_ptr,
    float* __restrict__ ref_output_ptr,
    uint64_t numel,
    uint64_t num_bins,
    float min_value,
    float max_value,
    bool mcg,
    bool mul1
)
{
    __shared__ int histogram[MAX_BINS];
    auto reset_histogram = [&]()
    {
        for (int i = threadIdx.x; i < num_bins; i += NUM_THREADS_TD)
            histogram[i] = 0;
        __syncthreads();
    };

    auto write_histogram = [&](float* output_ptr, uint64_t sc)
    {
        float scf = (float) sc;
        for (int i = threadIdx.x; i < num_bins; i += NUM_THREADS_TD)
            output_ptr[i] = ((float) histogram[i]) / scf;
        __syncthreads();
    };

    auto count = [&](float val)
    {
        val -= min_value;
        val /= (max_value - min_value);
        val *= (float) num_bins;
        int idx = (int) val;
        if (idx < 0) idx = 0;
        if (idx > num_bins - 1) idx = num_bins - 1;
        atomicAdd(&histogram[idx], 1);
    };

    if (ref_output_ptr)
    {
        reset_histogram();
        for (uint64_t i = threadIdx.x; i < 65536; i += NUM_THREADS_TD)
        {
            if (mcg)
                count(decode_3inst_f<1>((uint16_t) (i & 0xffff)));
            else if (mul1)
                count(decode_3inst_f<2>((uint16_t) (i & 0xffff)));
            else
                count(decode_3inst_f<0>((uint16_t) (i & 0xffff)));
        }
        __syncthreads();
        write_histogram(ref_output_ptr, 65536);
    }

    reset_histogram();
    for (uint64_t i = threadIdx.x; i < numel; i += NUM_THREADS_TD)
        count(input_ptr[i]);
    __syncthreads();
    write_histogram(dist_output_ptr, numel);
}

/*
Compare tensor distribution to codebook (not optimized)

input: tensor, float, any shape
dist_output: (empty) output histogram, float, shape (num_bins,)
ref_output, optional: (empty) output codebook histogram, float, shape (num_bins,)
*/

void test_distribution
(
    at::Tensor& input,
    at::Tensor& dist_output,
    const c10::optional<at::Tensor>& ref_output,
    float min_value,
    float max_value,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(input, kFloat);

    uint64_t numel = input.numel();
    float* ref_output_ptr = (float*) OPTPTR(ref_output);
    uint64_t num_bins = dist_output.numel();
    TORCH_CHECK(num_bins <= MAX_BINS, "Too many bins");
    if (ref_output_ptr)
        TORCH_CHECK(num_bins == ref_output.value().numel());

    test_distribution_kernel<<<1, NUM_THREADS_TD, 0, stream>>>
    (
        (const float*) input.data_ptr(),
        (float*) dist_output.data_ptr(),
        (float*) ref_output_ptr,
        numel,
        num_bins,
        min_value,
        max_value,
        mcg,
        mul1
    );
}
