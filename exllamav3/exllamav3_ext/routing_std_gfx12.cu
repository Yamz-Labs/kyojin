#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "routing_std_gfx12.cuh"
#include "util.cuh"

#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr int WAVE_SIZE = 32;
constexpr int GEMV_WAVES = 2;   // 2 expert rows per block: E / 2 blocks (8 per block left most CUs idle at E = 256)
// Supported (hidden, experts, top-k) decode shapes: see routing_std_gfx12_bsz1 below and _HIP_ROUTER_SHAPES in block_sparse_mlp_routing.py
constexpr int MAX_ROUTER_ROWS = 16;
constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

__device__ __forceinline__
void wave_reduce_best(float& key, int& idx)
{
    bool valid = idx >= 0;
    #pragma unroll
    for (int offset = WAVE_SIZE / 2; offset > 0; offset >>= 1)
    {
        const float other_key = __shfl_down(key, offset, WAVE_SIZE);
        const int other_idx = __shfl_down(idx, offset, WAVE_SIZE);
        const bool other_valid = other_idx >= 0;
        if (other_valid && (!valid || other_key > key ||
                            (other_key == key && other_idx < idx)))
        {
            key = other_key;
            idx = other_idx;
            valid = true;
        }
    }
    key = __shfl(key, 0, WAVE_SIZE);
    idx = __shfl(idx, 0, WAVE_SIZE);
}

__device__ __forceinline__
float wave_reduce_max(float value)
{
    #pragma unroll
    for (int offset = WAVE_SIZE / 2; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down(value, offset, WAVE_SIZE));
    return __shfl(value, 0, WAVE_SIZE);
}

__device__ __forceinline__
float wave_reduce_sum(float value)
{
    #pragma unroll
    for (int offset = WAVE_SIZE / 2; offset > 0; offset >>= 1)
        value += __shfl_down(value, offset, WAVE_SIZE);
    return __shfl(value, 0, WAVE_SIZE);
}

template <int ROUTER_HIDDEN, int ROUTER_EXPERTS>
__global__ __launch_bounds__(GEMV_WAVES * WAVE_SIZE)
void routing_gemv_gfx12_bsz1_kernel
(
    const half* __restrict__ hidden,
    const half* __restrict__ gate_t,
    half* __restrict__ scores
)
{
    const int row = blockIdx.y;
    const int wave = threadIdx.x / WAVE_SIZE;
    const int lane = threadIdx.x % WAVE_SIZE;
    const int expert = blockIdx.x * GEMV_WAVES + wave;
    hidden += static_cast<size_t>(row) * ROUTER_HIDDEN;
    scores += static_cast<size_t>(row) * ROUTER_EXPERTS;
    if (expert >= ROUTER_EXPERTS) return;

    const half2* hidden2 = reinterpret_cast<const half2*>(hidden);
    const half2* gate2 = reinterpret_cast<const half2*>(
        gate_t + static_cast<size_t>(expert) * ROUTER_HIDDEN);

    float sum = 0.0f;
    #pragma unroll 8
    for (int col = lane; col < ROUTER_HIDDEN / 2; col += WAVE_SIZE)
    {
        const half2 h = hidden2[col];
        const half2 g = gate2[col];
        sum = fmaf(__half2float(__low2half(h)), __half2float(__low2half(g)), sum);
        sum = fmaf(__half2float(__high2half(h)), __half2float(__high2half(g)), sum);
    }

    sum = wave_reduce_sum(sum);
    if (lane == 0) scores[expert] = __float2half_rn(sum);
}

// Top-k by rank (same result as the standard routing.cu reduction): every thread holds one expert score and counts, from shared
// memory, the scores that beat it (greater, or equal with a lower expert index); a rank below K is that expert's output slot.
// Then the softmax over the K selected exactly as before: exp(score - max over all experts), lane-order sum, divide.
template <int ROUTER_EXPERTS, int ROUTER_TOP_K>
__global__ __launch_bounds__(ROUTER_EXPERTS)
void routing_std_topk_gfx12_bsz1_kernel
(
    const half* __restrict__ scores,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights
)
{
    constexpr int NUM_WAVES = ROUTER_EXPERTS / WAVE_SIZE;
    static_assert(ROUTER_EXPERTS % WAVE_SIZE == 0 && NUM_WAVES <= WAVE_SIZE && ROUTER_TOP_K < WAVE_SIZE, "router shape");

    __shared__ float logits[ROUTER_EXPERTS];
    __shared__ float wave_max[NUM_WAVES];
    __shared__ float global_max;
    __shared__ float selected_exp[ROUTER_TOP_K];
    __shared__ int selected_idx[ROUTER_TOP_K];

    const int tid = threadIdx.x;
    const int lane = tid % WAVE_SIZE;
    const int wave = tid / WAVE_SIZE;
    scores += static_cast<size_t>(blockIdx.x) * ROUTER_EXPERTS;
    topk_indices += static_cast<size_t>(blockIdx.x) * ROUTER_TOP_K;
    topk_weights += static_cast<size_t>(blockIdx.x) * ROUTER_TOP_K;
    const float logit = __half2float(scores[tid]);
    logits[tid] = logit;

    float max_logit = wave_reduce_max(logit);
    if (lane == 0) wave_max[wave] = max_logit;
    __syncthreads();
    if (wave == 0)
    {
        max_logit = lane < NUM_WAVES ? wave_max[lane] : NEG_INF;
        max_logit = wave_reduce_max(max_logit);
        if (lane == 0) global_max = max_logit;
    }

    int rank = 0;
    #pragma unroll 8
    for (int j = 0; j < ROUTER_EXPERTS; ++j)
    {
        const float other = logits[j];
        rank += (other > logit || (other == logit && j < tid)) ? 1 : 0;
    }
    __syncthreads();
    if (rank < ROUTER_TOP_K)
    {
        selected_exp[rank] = logit == global_max ? 1.0f : expf(logit - global_max);
        selected_idx[rank] = tid;
    }
    __syncthreads();

    if (wave == 0)
    {
        const float e = lane < ROUTER_TOP_K ? selected_exp[lane] : 0.0f;
        const float sum = wave_reduce_sum(e) + 1.0e-20f;
        if (lane < ROUTER_TOP_K)
        {
            topk_indices[lane] = static_cast<int64_t>(selected_idx[lane]);
            topk_weights[lane] = __float2half_rn(e / sum);
        }
    }
}

bool is_gfx12_wave32(int device)
{
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
    const char* arch = prop.gcnArchName;
    // These kernels use only wave-width primitives (__shfl*, __syncwarp) and no
    // RDNA4-only instruction, so every wave32 gfx11.5 / gfx12 part is valid.
    // gfx1150/1151/1152 = RDNA3.5 (Strix Halo), gfx1200/1201 = RDNA4.
    auto is_arch = [arch](const char* name, size_t n) {
        return !std::strncmp(arch, name, n) && (arch[n] == '\0' || arch[n] == ':');
    };
    const bool wave32_arch =
        is_arch("gfx1200", 7) || is_arch("gfx1201", 7) ||
        is_arch("gfx1150", 7) || is_arch("gfx1151", 7) || is_arch("gfx1152", 7);
    return wave32_arch && prop.warpSize == WAVE_SIZE;
}

void check_tensor
(
    const at::Tensor& tensor,
    const at::Device& device,
    at::ScalarType dtype,
    at::IntArrayRef shape,
    const char* name,
    size_t alignment
)
{
    TORCH_CHECK(tensor.is_cuda(), "routing_std_gfx12_bsz1: ", name, " must be a device tensor");
    TORCH_CHECK(tensor.device() == device, "routing_std_gfx12_bsz1: ", name,
                " must be on the hidden-state device");
    TORCH_CHECK(tensor.dtype() == dtype, "routing_std_gfx12_bsz1: ", name,
                " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == shape, "routing_std_gfx12_bsz1: ", name,
                " has the wrong shape");
    TORCH_CHECK(tensor.is_contiguous(), "routing_std_gfx12_bsz1: ", name,
                " must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0,
                "routing_std_gfx12_bsz1: ", name, " must be ", alignment, "-byte aligned");
}

} // namespace

void routing_std_gfx12_bsz1
(
    const at::Tensor& hidden,
    const at::Tensor& gate_t,
    at::Tensor& scores,
    at::Tensor& topk_indices,
    at::Tensor& topk_weights
)
{
    TORCH_CHECK(hidden.is_cuda(),
                "routing_std_gfx12_bsz1: hidden must be a device tensor");
    const at::cuda::OptionalCUDAGuard device_guard(hidden.device());
    const int device = hidden.get_device();
    TORCH_CHECK(is_gfx12_wave32(device),
                "routing_std_gfx12_bsz1 requires gfx1200/gfx1201 with wave32");

    TORCH_CHECK(hidden.dim() == 2,
                "routing_std_gfx12_bsz1: hidden must be a 2D tensor");
    const int64_t rows = hidden.size(0);
    TORCH_CHECK(rows > 0 && rows <= MAX_ROUTER_ROWS,
                "routing_std_gfx12_bsz1 supports 1 through ", MAX_ROUTER_ROWS, " rows");
    TORCH_CHECK(hidden.dim() == 2 && gate_t.dim() == 2 && topk_indices.dim() == 2,
                "routing_std_gfx12_bsz1: 2D tensors expected");
    const int64_t H = hidden.size(1), E = gate_t.size(0), K = topk_indices.size(1);
    check_tensor(hidden, hidden.device(), at::kHalf, {rows, H}, "hidden", 16);
    check_tensor(gate_t, hidden.device(), at::kHalf, {E, H}, "gate_t", 16);
    check_tensor(scores, hidden.device(), at::kHalf, {rows, E}, "scores", 16);
    check_tensor(topk_indices, hidden.device(), at::kLong, {rows, K}, "topk_indices", 16);
    check_tensor(topk_weights, hidden.device(), at::kHalf, {rows, K}, "topk_weights", 16);

    hipStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const half* h = reinterpret_cast<const half*>(hidden.data_ptr());
    const half* g = reinterpret_cast<const half*>(gate_t.data_ptr());
    half* sc = reinterpret_cast<half*>(scores.data_ptr());
    int64_t* ti = reinterpret_cast<int64_t*>(topk_indices.data_ptr());
    half* tw = reinterpret_cast<half*>(topk_weights.data_ptr());
    #define ROUTER_SHAPE(HH, EE, KK) \
        if (H == HH && E == EE && K == KK) \
        { \
            routing_gemv_gfx12_bsz1_kernel<HH, EE><<<dim3(EE / GEMV_WAVES, rows), GEMV_WAVES * WAVE_SIZE, 0, stream>>>(h, g, sc); \
            routing_std_topk_gfx12_bsz1_kernel<EE, KK><<<rows, EE, 0, stream>>>(sc, ti, tw); \
        } else
    ROUTER_SHAPE(2560, 512, 10)      // Qwen3.8-Flash-Next
    ROUTER_SHAPE(2048, 256, 8)       // Qwen3.6-35B-A3B (GGUF and EXL3 packs)
    TORCH_CHECK(false, "routing_std_gfx12_bsz1: unsupported router shape ", H, " x ", E, " top-", K);
    #undef ROUTER_SHAPE
    cuda_check(hipPeekAtLastError());
}

#endif // USE_ROCM
