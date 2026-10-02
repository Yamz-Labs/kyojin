#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "dsa_topk_gfx12.cuh"
#include "util.cuh"

#include <climits>
#include <cstdint>
#include <cstring>

namespace {

constexpr int WAVE_SIZE = 32;
constexpr int THREADS = 1024;
constexpr int K_MAX = 512;   // top-k and output width bound (QSA shape)
constexpr uint16_t KEY_NEG_INF = 0x03ff;

__device__ __forceinline__ uint16_t topk_key(uint16_t bits)
{
    return bits & 0x8000 ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
}

__device__ __forceinline__ void find_descending_bucket(const int* hist, int target, int* result)
{
    int total = 0;
    for (int bucket = 255; bucket >= 0; --bucket)
    {
        if (total + hist[bucket] >= target)
        {
            result[0] = bucket;
            result[1] = total;
            return;
        }
        total += hist[bucket];
    }
    result[0] = -1;
    result[1] = total;
}

__device__ __forceinline__ int block_exclusive_scan(
    int value, int lane, int warp, int* warp_sums, int* warp_offsets)
{
    int inclusive = value;
    #pragma unroll
    for (int offset = 1; offset < WAVE_SIZE; offset <<= 1)
    {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += other;
    }
    if (lane == WAVE_SIZE - 1) warp_sums[warp] = inclusive;
    __syncthreads();

    if (warp == 0)
    {
        int warp_inclusive = warp_sums[lane];
        #pragma unroll
        for (int offset = 1; offset < WAVE_SIZE; offset <<= 1)
        {
            const int other = __shfl_up_sync(0xffffffffu, warp_inclusive, offset);
            if (lane >= offset) warp_inclusive += other;
        }
        warp_offsets[lane] = warp_inclusive - warp_sums[lane];
    }
    __syncthreads();
    return warp_offsets[warp] + inclusive - value;
}

__global__ __launch_bounds__(THREADS)
void dsa_topk_gfx12_kernel(const half* __restrict__ scores, int* __restrict__ out, int width, int stride,
                           int K, int out_w)
{
    const int tid = threadIdx.x;
    const half* row_scores = scores + static_cast<size_t>(blockIdx.x) * stride;
    int* row_out = out + static_cast<size_t>(blockIdx.x) * out_w;

    __shared__ int hist[256];
    __shared__ int search[2];
    __shared__ int warp_sums[THREADS / WAVE_SIZE];
    __shared__ int warp_offsets[THREADS / WAVE_SIZE];

    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    for (int64_t index = tid; index < static_cast<int64_t>(width); index += THREADS)
    {
        const uint16_t key = topk_key(__half_as_ushort(row_scores[index]));
        if (key > KEY_NEG_INF) atomicAdd(&hist[key >> 8], 1);
    }
    __syncthreads();
    if (tid == 0) find_descending_bucket(hist, K, search);
    __syncthreads();

    const int high_bucket = search[0];
    const int count_above_high = search[1];
    uint16_t threshold = KEY_NEG_INF;
    int ties_needed = 0;
    if (high_bucket >= 0)
    {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        for (int64_t index = tid; index < static_cast<int64_t>(width); index += THREADS)
        {
            const uint16_t key = topk_key(__half_as_ushort(row_scores[index]));
            if (key > KEY_NEG_INF && (key >> 8) == high_bucket)
                atomicAdd(&hist[key & 0xff], 1);
        }
        __syncthreads();
        if (tid == 0) find_descending_bucket(hist, K - count_above_high, search);
        __syncthreads();
        threshold = static_cast<uint16_t>((high_bucket << 8) | search[0]);
        ties_needed = K - count_above_high - search[1];
    }

    // Each thread owns one contiguous range, so prefixing its local output count produces
    // ascending global indices. A separate tie prefix lets only the earliest threshold ties
    // survive without serializing the full row scan.
    const int range_begin = static_cast<int>(static_cast<int64_t>(width) * tid / THREADS);
    const int range_end = static_cast<int>(static_cast<int64_t>(width) * (tid + 1) / THREADS);
    const int lane = tid % WAVE_SIZE;
    const int warp = tid / WAVE_SIZE;
    int high_count = 0;
    int tie_count = 0;
    for (int index = range_begin; index < range_end; ++index)
    {
        const uint16_t key = topk_key(__half_as_ushort(row_scores[index]));
        if (key > threshold && key > KEY_NEG_INF) ++high_count;
        else if (key == threshold && key > KEY_NEG_INF) ++tie_count;
    }

    const int tie_offset = block_exclusive_scan(tie_count, lane, warp, warp_sums, warp_offsets);
    const int local_ties = max(0, min(tie_count, ties_needed - tie_offset));
    const int selected_count = high_count + local_ties;
    int output = block_exclusive_scan(selected_count, lane, warp, warp_sums, warp_offsets);

    int seen_ties = 0;
    for (int index = range_begin; index < range_end; ++index)
    {
        const uint16_t key = topk_key(__half_as_ushort(row_scores[index]));
        if (key > threshold && key > KEY_NEG_INF)
            row_out[output++] = index;
        else if (key == threshold && key > KEY_NEG_INF)
        {
            if (seen_ties < local_ties) row_out[output++] = index;
            ++seen_ties;
        }
    }
    if (tid == THREADS - 1)
        for (; output < out_w; ++output) row_out[output] = -1;
}

bool is_gfx12_wave32(int device)
{
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
    const char* arch = prop.gcnArchName;
    // Only wave-width primitives here (__shfl*, warpSize); no RDNA4-only
    // instruction, so any wave32 gfx11.5 / gfx12 part is valid.
    auto is_arch = [arch](const char* name, size_t n) {
        return !std::strncmp(arch, name, n) && (arch[n] == '\0' || arch[n] == ':');
    };
    const bool wave32_arch =
        is_arch("gfx1200", 7) || is_arch("gfx1201", 7) ||
        is_arch("gfx1150", 7) || is_arch("gfx1151", 7) || is_arch("gfx1152", 7);
    return wave32_arch && prop.warpSize == WAVE_SIZE;
}

} // namespace

void dsa_topk_gfx12(const at::Tensor& scores, at::Tensor& indices, int64_t k)
{
    TORCH_CHECK(scores.is_cuda() && indices.is_cuda(), "dsa_topk_gfx12 requires device tensors");
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    const int device = scores.get_device();
    TORCH_CHECK(is_gfx12_wave32(device), "dsa_topk_gfx12 requires a wave32 gfx11.5/gfx12 part");
    TORCH_CHECK(scores.device() == indices.device(), "dsa_topk_gfx12 tensors must share a device");
    TORCH_CHECK(scores.dtype() == at::kHalf && indices.dtype() == at::kInt,
                "dsa_topk_gfx12 requires fp16 scores and int32 indices");
    TORCH_CHECK(scores.dim() == 2 && indices.dim() == 2 && scores.size(0) == indices.size(0),
                "dsa_topk_gfx12 requires matching 2D tensors");
    TORCH_CHECK(scores.size(1) <= INT_MAX && scores.stride(0) <= INT_MAX,
                "dsa_topk_gfx12 requires score width and row stride <= INT_MAX");
    TORCH_CHECK(k >= 1 && k <= K_MAX, "dsa_topk_gfx12 requires 1 <= k <= 512");
    TORCH_CHECK(scores.size(1) >= 1 && scores.stride(1) == 1 && scores.stride(0) >= scores.size(1) &&
                scores.stride(0) % 128 == 0,
                "dsa_topk_gfx12 requires QSA scores with T >= 1 and a 128-aligned row stride");
    TORCH_CHECK(indices.size(1) >= k && indices.size(1) <= K_MAX && indices.is_contiguous(),
                "dsa_topk_gfx12 requires contiguous int32 [R, W] output with k <= W <= 512");
    if (scores.size(0) == 0) return;

    hipStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    dsa_topk_gfx12_kernel<<<scores.size(0), THREADS, 0, stream>>>(
        reinterpret_cast<const half*>(scores.data_ptr()), reinterpret_cast<int*>(indices.data_ptr()),
        static_cast<int>(scores.size(1)), static_cast<int>(scores.stride(0)),
        static_cast<int>(k), static_cast<int>(indices.size(1)));
    cuda_check(hipPeekAtLastError());
}

#endif // USE_ROCM
