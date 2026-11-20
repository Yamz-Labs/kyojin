#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "gm_router.cuh"

#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr int WAVE_SIZE = 32;
constexpr int GEMV_WAVES = 2;   // 2 expert rows per block: E / 2 blocks (8 per block left most CUs idle at E = 256)
// Supported (hidden, experts, top-k) decode shapes: see gm_routing_std_bsz1 below and _HIP_ROUTER_SHAPES in block_sparse_mlp_routing.py
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
    TORCH_CHECK(tensor.is_cuda(), "gm_routing_std_bsz1: ", name, " must be a device tensor");
    TORCH_CHECK(tensor.device() == device, "gm_routing_std_bsz1: ", name,
                " must be on the hidden-state device");
    TORCH_CHECK(tensor.dtype() == dtype, "gm_routing_std_bsz1: ", name,
                " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == shape, "gm_routing_std_bsz1: ", name,
                " has the wrong shape");
    TORCH_CHECK(tensor.is_contiguous(), "gm_routing_std_bsz1: ", name,
                " must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0,
                "gm_routing_std_bsz1: ", name, " must be ", alignment, "-byte aligned");
}

} // namespace

void gm_routing_std_bsz1
(
    const at::Tensor& hidden,
    const at::Tensor& gate_t,
    at::Tensor& scores,
    at::Tensor& topk_indices,
    at::Tensor& topk_weights
)
{
    TORCH_CHECK(hidden.is_cuda(),
                "gm_routing_std_bsz1: hidden must be a device tensor");
    const at::cuda::OptionalCUDAGuard device_guard(hidden.device());
    const int device = hidden.get_device();
    TORCH_CHECK(is_gfx12_wave32(device),
                "gm_routing_std_bsz1 requires gfx1200/gfx1201 with wave32");

    TORCH_CHECK(hidden.dim() == 2,
                "gm_routing_std_bsz1: hidden must be a 2D tensor");
    const int64_t rows = hidden.size(0);
    TORCH_CHECK(rows > 0 && rows <= MAX_ROUTER_ROWS,
                "gm_routing_std_bsz1 supports 1 through ", MAX_ROUTER_ROWS, " rows");
    TORCH_CHECK(hidden.dim() == 2 && gate_t.dim() == 2 && topk_indices.dim() == 2,
                "gm_routing_std_bsz1: 2D tensors expected");
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
    ROUTER_SHAPE(2048, 128, 8)       // Qwen3-30B-A3B (qwen3moe)
    TORCH_CHECK(false, "gm_routing_std_bsz1: unsupported router shape ", H, " x ", E, " top-", K);
    #undef ROUTER_SHAPE
    TORCH_CHECK(hipPeekAtLastError() == hipSuccess, "gm_routing_std_bsz1: launch failed");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Generic router (any hidden size / expert count <= 1024 / top-k <= 32, fp32 or fp16 gate weights, fp32 or fp16 activations, 1..16 rows):
// logits = x . gate^T (fp32 accumulation, one wave per expert), then ONE block per row selects the top-k by rank counting (ties -> lowest expert id,
// i.e. a stable descending sort) and forms the weights.
//   gating 0: softmax over all experts (selection on the logits), w = exp(l - max) / sum_all, or / (sum_selected + 1e-20) when norm (Qwen3 / Mixtral / gpt-oss)
//   gating 2: sigmoid; selection on sigmoid + bias (bias optional, selection only), w = sigmoid[sel], / max(sum_sel, eps) when norm (GLM / DeepSeek / Hy4)
//   w *= scale. Outputs: sel int64 [R, k], w half [R, k].
// ---------------------------------------------------------------------------------------------------------------------------------
namespace {

template <typename T> __device__ __forceinline__ float gm_ld_f(const T v);
template <> __device__ __forceinline__ float gm_ld_f<float>(const float v) { return v; }
template <> __device__ __forceinline__ float gm_ld_f<half>(const half v) { return __half2float(v); }

template <typename XT, typename WT>
__global__ __launch_bounds__(128) void gm_rt_logits_kernel(const XT* __restrict__ x, const WT* __restrict__ g, float* __restrict__ scores, int H, int E)
{
    const int row = blockIdx.y;
    const int wave = threadIdx.x / WAVE_SIZE;
    const int lane = threadIdx.x % WAVE_SIZE;
    const int expert = blockIdx.x * 4 + wave;
    if (expert >= E) return;
    const XT* xr = x + (size_t)row * H;
    const WT* gr = g + (size_t)expert * H;
    float sum = 0.0f;
    #pragma unroll 4
    for (int h = lane; h < H; h += WAVE_SIZE)
        sum = fmaf(gm_ld_f<XT>(xr[h]), gm_ld_f<WT>(gr[h]), sum);
    sum = wave_reduce_sum(sum);
    if (lane == 0) scores[(size_t)row * E + expert] = sum;
}

__global__ __launch_bounds__(1024) void gm_rt_topk_kernel(const float* __restrict__ scores, const float* __restrict__ bias, int64_t* __restrict__ sel,
                                                          half* __restrict__ wout, int E, int topk, int gating, int norm, float scale, float eps)
{
    __shared__ float key[1024];
    __shared__ float prob[1024];
    __shared__ float wred[32];
    __shared__ float s_max, s_sum;
    __shared__ float selw[32];
    __shared__ int selidx[32];
    const int tid = threadIdx.x, lane = tid % WAVE_SIZE, wave = tid / WAVE_SIZE;
    const int nwaves = (blockDim.x + WAVE_SIZE - 1) / WAVE_SIZE;
    const float* sc = scores + (size_t)blockIdx.x * E;
    const bool valid = tid < E;
    const float logit = valid ? sc[tid] : NEG_INF;
    float p, k;
    if (gating == 2)
    {
        p = valid ? 1.0f / (1.0f + expf(-logit)) : 0.0f;
        k = valid ? (bias != nullptr ? p + bias[tid] : p) : NEG_INF;
    }
    else
    {
        float m = wave_reduce_max(logit);
        if (lane == 0) wred[wave] = m;
        __syncthreads();
        if (wave == 0)
        {
            float mm = lane < nwaves ? wred[lane] : NEG_INF;
            mm = wave_reduce_max(mm);
            if (lane == 0) s_max = mm;
        }
        __syncthreads();
        p = valid ? expf(logit - s_max) : 0.0f;       // unnormalized exp
        k = logit;
        float t = wave_reduce_sum(p);
        __syncthreads();
        if (lane == 0) wred[wave] = t;
        __syncthreads();
        if (wave == 0)
        {
            float tt = lane < nwaves ? wred[lane] : 0.0f;
            tt = wave_reduce_sum(tt);
            if (lane == 0) s_sum = tt;
        }
    }
    key[tid] = k;
    prob[tid] = p;
    __syncthreads();
    int rank = 0;
    if (valid)
    {
        #pragma unroll 8
        for (int j = 0; j < E; ++j)
        {
            const float o = key[j];
            rank += (o > k || (o == k && j < tid)) ? 1 : 0;
        }
        if (rank < topk) { selw[rank] = p; selidx[rank] = tid; }
    }
    __syncthreads();
    if (wave == 0)
    {
        const float w = lane < topk ? selw[lane] : 0.0f;
        const float ssum = wave_reduce_sum(w);
        if (lane < topk)
        {
            float o;
            if (gating == 2) o = norm ? w / fmaxf(ssum, eps) : w;
            else o = norm ? w / (ssum + 1.0e-20f) : w / s_sum;
            sel[(size_t)blockIdx.x * topk + lane] = (int64_t)selidx[lane];
            wout[(size_t)blockIdx.x * topk + lane] = __float2half_rn(o * scale);
        }
    }
}

}  // namespace

void gm_route(const at::Tensor& x, const at::Tensor& gate, const at::Tensor& bias, at::Tensor& scores, at::Tensor& sel, at::Tensor& w,
              int64_t gating, bool norm, double scale, double eps)
{
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && gate.is_cuda() && gate.dim() == 2 && x.is_contiguous() && gate.is_contiguous(), "gm_route: x [R, H] and gate [E, H] contiguous device tensors");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const int R = (int)x.size(0), H = (int)x.size(1), E = (int)gate.size(0), topk = (int)sel.size(1);
    TORCH_CHECK(R >= 1 && R <= MAX_ROUTER_ROWS && gate.size(1) == H && E <= 1024 && topk >= 1 && topk <= 32 && topk <= E, "gm_route: unsupported shape R=", R, " H=", H, " E=", E, " k=", topk);
    TORCH_CHECK(gating == 0 || gating == 2, "gm_route: gating 0 (softmax) or 2 (sigmoid)");
    TORCH_CHECK(sel.dtype() == at::kLong && w.dtype() == at::kHalf && scores.dtype() == at::kFloat && scores.numel() >= (int64_t)R * E && sel.size(0) == R && w.size(0) == R && w.size(1) == topk,
                "gm_route: sel int64 [R, k], w half [R, k], scores float scratch [R, E]");
    TORCH_CHECK(bias.numel() == 0 || (bias.dtype() == at::kFloat && bias.numel() == E && bias.is_contiguous()), "gm_route: bias float [E] or empty");
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    float* sp = scores.data_ptr<float>();
    const dim3 grid((E + 3) / 4, R);
    const bool xf = x.dtype() == at::kFloat, gf = gate.dtype() == at::kFloat;
    TORCH_CHECK((xf || x.dtype() == at::kHalf) && (gf || gate.dtype() == at::kHalf), "gm_route: x / gate float or half");
    #define GM_RT(XT, WT) gm_rt_logits_kernel<XT, WT><<<grid, 128, 0, stream>>>(reinterpret_cast<const XT*>(x.data_ptr()), reinterpret_cast<const WT*>(gate.data_ptr()), sp, H, E)
    if (xf && gf) GM_RT(float, float); else if (xf) GM_RT(float, half); else if (gf) GM_RT(half, float); else GM_RT(half, half);
    #undef GM_RT
    const int nt = ((E + 31) / 32) * 32;
    gm_rt_topk_kernel<<<R, nt, 0, stream>>>(sp, bias.numel() ? bias.data_ptr<float>() : nullptr, sel.data_ptr<int64_t>(), reinterpret_cast<half*>(w.data_ptr()),
                                            E, topk, (int)gating, norm ? 1 : 0, (float)scale, (float)eps);
    TORCH_CHECK(hipPeekAtLastError() == hipSuccess, "gm_route: launch failed");
}

#endif // USE_ROCM
