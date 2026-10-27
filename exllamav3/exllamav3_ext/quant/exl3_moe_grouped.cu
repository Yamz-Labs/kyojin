#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_moe_common.cuh"
#include "hadamard_inner.cuh"
#include "exl3_moe_gemm.cuh"
#include "exl3_moe_grouped.cuh"

// Grouped prefill tier of the fused MoE: the same math as exl3_moe_kernel (input hadamard, gate/up
// GEMMs, activation + hadamards, down GEMM, output hadamard into the per-assignment fp32 slots that
// exl3_moe_gather sums), split into token-parallel hadamard kernels and two grouped GEMM launches
// (exl3_moe_gemm.cuh) instead of one persistent kernel with expert groups of cooperating CTAs

#define MOEG_ROW_THREADS 256

// Expert of sorted row r: the last expert whose start is <= r (empty experts share their successor's
// start, so the last of equal starts is the non-empty one)
__device__ __forceinline__ int moeg_expert_of(const int64_t* __restrict__ starts, int num_experts, int64_t r)
{
    int lo = 0, hi = num_experts;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (starts[mid] <= r) lo = mid + 1; else hi = mid;
    }
    return lo - 1;
}

__global__ __launch_bounds__(MOEG_ROW_THREADS)
void exl3_moe_grouped_in_kernel
(
    const half* __restrict__ hidden_state,
    half* __restrict__ buf_g, half* __restrict__ buf_u,
    const half** __restrict__ gate_suh, const half** __restrict__ up_suh,
    const int64_t* __restrict__ token_sorted,
    const int64_t* __restrict__ counts, const int64_t* __restrict__ starts, const int64_t* __restrict__ fused_base,
    const int num_experts, const int hidden_dim, const int count_lo, const int count_hi
)
{
    const int wpt = hidden_dim / 128;
    const int64_t gw = (int64_t) blockIdx.x * (MOEG_ROW_THREADS / 32) + threadIdx.x / 32;
    const int64_t r = gw / wpt;
    const int chunk = gw % wpt;
    const int e = moeg_expert_of(starts, num_experts, r);
    if (e < 0) return;
    const int cnt = (int) counts[e];
    if (cnt < count_lo || cnt > count_hi || r >= starts[e] + cnt) return;
    const int64_t slot = fused_base[e] + (r - starts[e]);
    const half* in_ptr = hidden_state + token_sorted[r] * hidden_dim + chunk * 128;
    had_hf_r_128_inner<true, false>(in_ptr, buf_g + slot * hidden_dim + chunk * 128, gate_suh[e] + chunk * 128, 0.088388347648f);
    had_hf_r_128_inner<true, false>(in_ptr, buf_u + slot * hidden_dim + chunk * 128, up_suh[e] + chunk * 128, 0.088388347648f);
}

__global__ __launch_bounds__(MOEG_ROW_THREADS)
void exl3_moe_grouped_act_kernel
(
    half* __restrict__ inter_g, const half* __restrict__ inter_u,
    const half** __restrict__ gate_svh, const half** __restrict__ up_svh, const half** __restrict__ down_suh,
    const int64_t* __restrict__ counts, const int64_t* __restrict__ starts, const int64_t* __restrict__ fused_base,
    const int num_experts, const int intermediate_dim, const int count_lo, const int count_hi,
    const float act_limit, const int act_function
)
{
    const int wpt = intermediate_dim / 128;
    const int64_t gw = (int64_t) blockIdx.x * (MOEG_ROW_THREADS / 32) + threadIdx.x / 32;
    const int64_t r = gw / wpt;
    const int chunk = gw % wpt;
    const int e = moeg_expert_of(starts, num_experts, r);
    if (e < 0) return;
    const int cnt = (int) counts[e];
    if (cnt < count_lo || cnt > count_hi || r >= starts[e] + cnt) return;
    const int64_t slot = fused_base[e] + (r - starts[e]);
    const int64_t o = slot * intermediate_dim + chunk * 128;
    had_hf_r_128_guad_inner
    (
        inter_g + o, inter_u + o, inter_g + o,
        gate_svh[e] + chunk * 128, up_svh[e] + chunk * 128, down_suh[e] + chunk * 128,
        0.088388347648f, act_limit, act_function
    );
}

// Output hadamard of the down GEMM rows fused with the per-token combine: one warp per (token, 128 columns) sums the token's
// grouped-tier assignments in k order and adds the total to the fp32 output row (exl3_moe_gather handles the other tiers)
__global__ __launch_bounds__(MOEG_ROW_THREADS)
void exl3_moe_grouped_out_kernel
(
    const half* __restrict__ down_out, float* __restrict__ out_state,
    const half** __restrict__ down_svh, const half* __restrict__ weight_sorted,
    const int64_t* __restrict__ flat_expert, const int64_t* __restrict__ inv_order,
    const int64_t* __restrict__ counts, const int64_t* __restrict__ starts, const int64_t* __restrict__ gbase,
    const int num_tokens, const int topk, const int num_experts, const int hidden_dim, const int count_lo, const int count_hi
)
{
    const int wpt = hidden_dim / 128;
    const int64_t gw = (int64_t) blockIdx.x * (MOEG_ROW_THREADS / 32) + threadIdx.x / 32;
    const int64_t token = gw / wpt;
    const int chunk = gw % wpt;
    if (token >= num_tokens) return;
    const int t = threadIdx.x & 31;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    bool any = false;
    for (int k = 0; k < topk; ++k)
    {
        const int64_t a = token * topk + k;
        const int64_t e64 = flat_expert[a];
        if (e64 < 0 || e64 >= num_experts) continue;
        const int e = (int) e64;
        const int cnt = (int) counts[e];
        if (cnt < count_lo || cnt > count_hi) continue;
        const int64_t pos = inv_order[a];
        const int64_t slot = gbase[e] + (pos - starts[e]);
        const float w = __half2float(weight_sorted[pos]);
        any = true;
        half4 v = ((const half4*) (down_out + slot * hidden_dim + chunk * 128))[t];
        float v0 = __half2float(__low2half(v.x)), v1 = __half2float(__high2half(v.x));
        float v2 = __half2float(__low2half(v.y)), v3 = __half2float(__high2half(v.y));
        float p0 = v0 + v1, d0 = v0 - v1, p1 = v2 + v3, d1 = v2 - v3;
        float h0 = p0 + p1, h1 = d0 + d1, h2 = p0 - p1, h3 = d0 - d1;
        shuffle_had_f4x32(h0, h1, h2, h3, t);
        const float rs = 0.088388347648f * w;
        const half4 sc = ((const half4*) (down_svh[e] + chunk * 128))[t];
        s0 += h0 * rs * __low2float(sc.x);
        s1 += h1 * rs * __high2float(sc.x);
        s2 += h2 * rs * __low2float(sc.y);
        s3 += h3 * rs * __high2float(sc.y);
    }
    if (!any) return;
    float4* o = (float4*) (out_state + token * hidden_dim + chunk * 128) + t;
    float4 cur = *o;
    cur.x += s0; cur.y += s1; cur.z += s2; cur.w += s3;
    *o = cur;
}

#define MOEG_CASE(BITS) \
    case BITS: if (bm == 64) exl3_moe_gemm_launch<BITS, 64, STAGES, 2>(A0, A1, C0, C1, B0, B1, counts, starts, E, nmat, K, N, lo, hi, max_rows, stream); \
               else exl3_moe_gemm_launch<BITS, 32, STAGES, 2>(A0, A1, C0, C1, B0, B1, counts, starts, E, nmat, K, N, lo, hi, max_rows, stream); \
               break;

template<int STAGES>
static void moeg_gemm
(
    int bits, int bm,
    const half* A0, const half* A1, half* C0, half* C1, const uint16_t** B0, const uint16_t** B1,
    const int64_t* counts, const int64_t* starts, int E, int nmat, int K, int N, int lo, int hi, int max_rows,
    cudaStream_t stream
)
{
    switch (bits)
    {
        MOEG_CASE(2) MOEG_CASE(3) MOEG_CASE(4) MOEG_CASE(5) MOEG_CASE(6)
        default: TORCH_CHECK(false, "exl3_moe_grouped: unsupported bitrate ", bits);
    }
}

bool exl3_moe_grouped_supported(int K_gate, int K_up, int K_down)
{
    auto ok = [](int k) { return k >= 2 && k <= 6; };
    return ok(K_gate) && ok(K_up) && ok(K_down);
}

void exl3_moe_grouped
(
    const at::Tensor& hidden_state,
    const at::Tensor& expert_count,
    const at::Tensor& expert_start,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const int act_function,
    const int K_gate,
    const int K_up,
    const int K_down,
    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,
    const float act_limit,
    const at::Tensor& output_state,
    const at::Tensor& flat_expert,
    const at::Tensor& inv_order,
    const at::Tensor& group_base,
    const int64_t n_rows,
    const int count_lo,
    const int count_hi,
    const int m_tile,
    const int intermediate_dim
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(hidden_state, kHalf);
    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DTYPE(expert_start, kLong);
    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_DTYPE(weight_sorted, kHalf);
    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_DTYPE(flat_expert, kLong);
    TORCH_CHECK_DTYPE(inv_order, kLong);
    TORCH_CHECK_DTYPE(group_base, kLong);
    TORCH_CHECK(m_tile == 32 || m_tile == 64, "exl3_moe_grouped: m_tile");
    TORCH_CHECK(exl3_moe_grouped_supported(K_gate, K_up, K_down), "exl3_moe_grouped: bitrate");
    const int H = hidden_state.size(1);
    const int I = intermediate_dim;
    TORCH_CHECK(H % 128 == 0 && I % 128 == 0, "exl3_moe_grouped: dims must be multiples of 128");
    const int E = gate_ptrs_trellis.size(0);
    const int64_t rows_total = token_sorted.size(0);
    const int64_t n_slots = n_rows;
    const int num_tokens = output_state.size(0);
    const int topk = rows_total / num_tokens;
    if (!rows_total || !n_slots) return;
    const int max_rows = count_hi;

    auto opt = hidden_state.options();
    at::Tensor buf = at::empty({2, n_slots, H}, opt);
    at::Tensor inter = at::empty({2, n_slots, I}, opt);
    half* buf_g = (half*) buf.data_ptr();
    half* buf_u = buf_g + n_slots * H;
    half* inter_g = (half*) inter.data_ptr();
    half* inter_u = inter_g + n_slots * I;
    // The down GEMM output reuses the gate input buffer: gate/up have fully consumed it by then (stream order)
    half* down_out = buf_g;

    const int64_t* counts = (const int64_t*) expert_count.data_ptr();
    const int64_t* starts = (const int64_t*) expert_start.data_ptr();
    const int64_t* fbase = (const int64_t*) group_base.data_ptr();

    auto grid_for = [&](int dim) { return (unsigned) ((rows_total * (dim / 128) + 7) / 8); };

    exl3_moe_grouped_in_kernel<<<grid_for(H), MOEG_ROW_THREADS, 0, stream>>>
    (
        (const half*) hidden_state.data_ptr(), buf_g, buf_u,
        (const half**) gate_ptrs_suh.data_ptr(), (const half**) up_ptrs_suh.data_ptr(),
        (const int64_t*) token_sorted.data_ptr(), counts, starts, fbase, E, H, count_lo, count_hi
    );

    if (K_gate == K_up)
        moeg_gemm<3>(K_gate, m_tile, buf_g, buf_u, inter_g, inter_u,
                     (const uint16_t**) gate_ptrs_trellis.data_ptr(), (const uint16_t**) up_ptrs_trellis.data_ptr(),
                     counts, fbase, E, 2, H, I, count_lo, count_hi, max_rows, stream);
    else
    {
        moeg_gemm<3>(K_gate, m_tile, buf_g, buf_g, inter_g, inter_g,
                     (const uint16_t**) gate_ptrs_trellis.data_ptr(), (const uint16_t**) gate_ptrs_trellis.data_ptr(),
                     counts, fbase, E, 1, H, I, count_lo, count_hi, max_rows, stream);
        moeg_gemm<3>(K_up, m_tile, buf_u, buf_u, inter_u, inter_u,
                     (const uint16_t**) up_ptrs_trellis.data_ptr(), (const uint16_t**) up_ptrs_trellis.data_ptr(),
                     counts, fbase, E, 1, H, I, count_lo, count_hi, max_rows, stream);
    }

    exl3_moe_grouped_act_kernel<<<grid_for(I), MOEG_ROW_THREADS, 0, stream>>>
    (
        inter_g, inter_u,
        (const half**) gate_ptrs_svh.data_ptr(), (const half**) up_ptrs_svh.data_ptr(), (const half**) down_ptrs_suh.data_ptr(),
        counts, starts, fbase, E, I, count_lo, count_hi, act_limit, act_function
    );

    moeg_gemm<4>(K_down, m_tile, inter_g, inter_g, down_out, down_out,
                 (const uint16_t**) down_ptrs_trellis.data_ptr(), (const uint16_t**) down_ptrs_trellis.data_ptr(),
                 counts, fbase, E, 1, I, H, count_lo, count_hi, max_rows, stream);

    exl3_moe_grouped_out_kernel<<<(unsigned) (((int64_t) num_tokens * (H / 128) + 7) / 8), MOEG_ROW_THREADS, 0, stream>>>
    (
        down_out, (float*) output_state.data_ptr(),
        (const half**) down_ptrs_svh.data_ptr(), (const half*) weight_sorted.data_ptr(),
        (const int64_t*) flat_expert.data_ptr(), (const int64_t*) inv_order.data_ptr(),
        counts, starts, fbase, num_tokens, topk, E, H, count_lo, count_hi
    );
    cuda_check(cudaPeekAtLastError());
}
