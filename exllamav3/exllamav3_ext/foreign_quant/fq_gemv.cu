// Fast decode GEMV for foreign-quant weights (M = 1..16 rows).
// out[m, n] = sum_k x[m, k] * W[n, k], fp32 accumulate, fp16 or fp32 out.
//
// Why this beats fq_gemm.cu at decode: the old kernel gives one thread a
// whole (row, col) dot product, so every output re-reads the full x row from
// global memory (N x traffic) and small-N layers launch only M*N threads
// (e.g. 4 blocks for a 1024-wide latent proj: most SMs idle). This kernel
// instead cooperates: a block stages one x row in shared memory once, 8
// lane-groups of 32 lanes each take one output, lanes split K and reduce in
// shared memory. Weight traffic is unchanged (one pass, the roofline floor);
// x traffic drops from M*N*K to M*K; occupancy is full for any N >= 8.
//
// Portable CUDA/HIP: no PTX, no warp shuffles (the 32-lane groups reduce
// through shared memory + __syncthreads, so wave32 and wave64 both work),
// only cuda_fp16.h half ops plus __fmul_rn/__fadd_rn (HIP has both).
// Accuracy: same per-element fp32 arithmetic as fq_dequant.cuh, but the K
// summation is lane-strided and the 32 partials reduce sequentially, so vs
// fq_gemm the fp32 accumulation ORDER differs (fp16 tolerance, same contract
// as the existing fused-vs-unfused path).
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
#include "fq_api.h"

namespace {

// Cooperative GEMV core. FMT selects the format: 0 = Q8_0, 1 = IQ2_XXS,
// 2 = NVFP4. IN/OUT select the x/out dtypes (half/float). PAIR selects the
// NVFP4 E2M1 pair-table dot (meaningful for FMT == 2 only).
// packed/sr point at output row n's data (strides applied by the caller).
template <int FMT, typename IN, typename OUT, bool PAIR = false>
__global__ void fq_gemv_kernel(const IN* __restrict__ x, const uint8_t* __restrict__ packed,
                               const uint8_t* __restrict__ scales, float global,
                               OUT* __restrict__ out, int M, int N, int K,
                               int64_t packed_stride, int64_t scales_stride, bool use_aux)
{
    int tile = blockIdx.x;
    int ntiles = (N + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    int m = tile / ntiles;
    int t = tile % ntiles;
    int grp = threadIdx.x / FQ_GEMV_LANES;   // 0..7 output slot in this block
    int lane = threadIdx.x % FQ_GEMV_LANES;  // 0..31 K-split lane
    int n = t * FQ_GEMV_OUT_PER_BLOCK + grp;

    extern __shared__ __half sh[];
    __half* xs = sh;                                            // K halves
    // IQ2 header scratch (8 groups x nb*16 u32) lives between xs and red.
    // NVFP4 pair table (256 float2, PAIR arm only) likewise, with the
    // scaled-e4m3 table (256 floats, use_stab arm) after it.
    uint32_t* hdrs = nullptr;
    float2* e2m1p = nullptr;
    float* stab = nullptr;
    float* red;
    int nb1 = K / FQ_IQ2_XXS_GROUP;
    // use_aux (block-uniform kernel arg): scaled-e4m3 table arm for FMT==2
    // (from BOLT_FQ_STAB). FMT==1 ignores it.
    bool use_stab = false;
    if constexpr (FMT == 1)
    {
        hdrs = reinterpret_cast<uint32_t*>(xs + K);
        red = reinterpret_cast<float*>(hdrs + 8 * nb1 * 16);
    }
    else if constexpr (FMT == 2)
    {
        use_stab = use_aux;
        if constexpr (PAIR)
        {
            e2m1p = reinterpret_cast<float2*>(xs + K);
            if (use_stab)
            {
                stab = reinterpret_cast<float*>(e2m1p + 256);
                red = reinterpret_cast<float*>(stab + 256);
            }
            else
                red = reinterpret_cast<float*>(e2m1p + 256);
        }
        else if (use_stab)
        {
            stab = reinterpret_cast<float*>(xs + K);
            red = reinterpret_cast<float*>(stab + 256);
        }
        else
            red = reinterpret_cast<float*>(xs + K);
    }
    else
        red = reinterpret_cast<float*>(xs + K);                 // 8*32 floats

    __shared__ fq_tables_sh tab;
    fq_tables_load(&tab, FMT == 2, FMT == 1);
    fq_stage_x<IN>(x + (int64_t)m * K, xs, K);
    if constexpr (FMT == 1)
        fq_stage_iq2_hdrs(packed, packed_stride, hdrs, nb1,
                           t * FQ_GEMV_OUT_PER_BLOCK, N);
    if constexpr (FMT == 2 && PAIR)
        fq_e2m1p_build(e2m1p);
    if (use_stab)
        fq_e2m1s_build(stab, global);
    __syncthreads();

    float acc = 0.0f;
    if (n < N)
    {
        if constexpr (FMT == 0)
            acc = fq_dot_q8_0(packed + (int64_t)n * packed_stride, xs, K, lane);
        else if constexpr (FMT == 1)
            acc = fq_dot_iq2_xxs(packed + (int64_t)n * packed_stride, xs, &tab,
                                  hdrs + (int64_t)grp * nb1 * 16, K, lane);
        else
        {
            const uint8_t* row = packed + (int64_t)n * packed_stride;
            const uint8_t* srow = scales + (int64_t)n * scales_stride;
            if (use_stab)
                acc = fq_dot_nvfp4<PAIR, true>(row, srow, xs, &tab, e2m1p,
                                              stab, K, lane, global);
            else
                acc = fq_dot_nvfp4<PAIR, false>(row, srow, xs, &tab, e2m1p,
                                               nullptr, K, lane, global);
        }
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    // Lane 0 of each group sums its 32 partials (sequential, portable).
    if (n < N && lane == 0)
    {
        float s = 0.0f;
        #pragma unroll
        for (int l = 0; l < 32; l++)
            s = __fadd_rn(s, red[grp * 32 + l]);
        if constexpr (std::is_same<OUT, __half>::value)
            out[(int64_t)m * N + n] = __float2half_rn(s);
        else
            out[(int64_t)m * N + n] = s;
    }
}

template <int FMT>
static void fq_gemv_q(torch::Tensor x, torch::Tensor packed, torch::Tensor scales,
                      double global_scale, torch::Tensor out, bool pair = false)
{
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kHalf || x.dtype() == torch::kFloat, "x must be half or float");
    TORCH_CHECK(out.dtype() == torch::kHalf || out.dtype() == torch::kFloat,
                "out must be half or float");
    TORCH_CHECK_DIM(x, 2);
    TORCH_CHECK_DIM(out, 2);
    int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
    TORCH_CHECK((int)out.size(0) == M, "out rows must match x rows");
    TORCH_CHECK(M >= 1 && M <= 16, "fast GEMV covers M = 1..16");
    int64_t pstride = 0, sstride = 0;
    if constexpr (FMT == 0)
    {
        TORCH_CHECK_DIM(packed, 3);
        TORCH_CHECK(K % FQ_Q8_0_GROUP == 0, "K must be a multiple of 32");
        TORCH_CHECK_SIZE(packed, 2, FQ_Q8_0_BYTES);
        TORCH_CHECK((int)packed.size(0) == N && (int)packed.size(1) == K / FQ_Q8_0_GROUP,
                    "packed must be [N, K/32, 34]");
        pstride = (int64_t)(K / FQ_Q8_0_GROUP) * FQ_Q8_0_BYTES;
    }
    else if constexpr (FMT == 1)
    {
        TORCH_CHECK_DIM(packed, 3);
        TORCH_CHECK(K % FQ_IQ2_XXS_GROUP == 0, "K must be a multiple of 256");
        TORCH_CHECK(K <= 8192, "IQ2 header scratch covers K <= 8192");
        TORCH_CHECK_SIZE(packed, 2, FQ_IQ2_XXS_BYTES);
        TORCH_CHECK((int)packed.size(0) == N && (int)packed.size(1) == K / FQ_IQ2_XXS_GROUP,
                    "packed must be [N, K/256, 66]");
        pstride = (int64_t)(K / FQ_IQ2_XXS_GROUP) * FQ_IQ2_XXS_BYTES;
    }
    else
    {
        TORCH_CHECK_DIM(packed, 2);
        TORCH_CHECK_DIM(scales, 2);
        TORCH_CHECK(K % FQ_NVFP4_GROUP == 0, "K must be a multiple of 16");
        TORCH_CHECK((int)packed.size(0) == N && (int)packed.size(1) == K / 2, "packed must be [N, K/2]");
        TORCH_CHECK((int)scales.size(0) == N && (int)scales.size(1) == K / 16, "scales must be [N, K/16]");
        // v4 consecutive dot: E = K/32 elements per lane must be a multiple
        // of 16 (one scale group) and even (half2 x reads, no k<K guards).
        TORCH_CHECK(K % 64 == 0, "NVFP4 GEMV needs K a multiple of 64");
        pstride = K / 2;
        sstride = K / 16;
    }
    at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int ntiles = (N + FQ_GEMV_OUT_PER_BLOCK - 1) / FQ_GEMV_OUT_PER_BLOCK;
    size_t shmem = (size_t)K * sizeof(__half) + FQ_GEMV_THREADS * sizeof(float);
    if constexpr (FMT == 1)
        shmem += (size_t)8 * (K / FQ_IQ2_XXS_GROUP) * 16 * sizeof(uint32_t);
    if constexpr (FMT == 2)
    {
        if (pair)
            shmem += (size_t)256 * sizeof(float2);
        if (fq_use_stab())
            shmem += (size_t)256 * sizeof(float);
    }
    const uint8_t* sc = nullptr;
    if constexpr (FMT == 2)
        sc = scales.data_ptr<uint8_t>();
    // Read once per launch (host): the kernel gets it as a uniform arg.
    bool use_aux = (FMT == 2) ? fq_use_stab() : false;
    auto launch = [&](auto* xp, auto* op, bool pr)
    {
        if constexpr (FMT == 2)
        {
            if (pr)
                fq_gemv_kernel<FMT, std::decay_t<decltype(*xp)>, std::decay_t<decltype(*op)>, true>
                    <<<M * ntiles, FQ_GEMV_THREADS, shmem, stream>>>(
                        xp, packed.data_ptr<uint8_t>(), sc, (float)global_scale,
                        op, M, N, K, pstride, sstride, use_aux);
            else
                fq_gemv_kernel<FMT, std::decay_t<decltype(*xp)>, std::decay_t<decltype(*op)>, false>
                    <<<M * ntiles, FQ_GEMV_THREADS, shmem, stream>>>(
                        xp, packed.data_ptr<uint8_t>(), sc, (float)global_scale,
                        op, M, N, K, pstride, sstride, use_aux);
        }
        else
            fq_gemv_kernel<FMT, std::decay_t<decltype(*xp)>, std::decay_t<decltype(*op)>>
                <<<M * ntiles, FQ_GEMV_THREADS, shmem, stream>>>(
                    xp, packed.data_ptr<uint8_t>(), sc, (float)global_scale,
                    op, M, N, K, pstride, sstride, use_aux);
    };
    if (x.dtype() == torch::kHalf && out.dtype() == torch::kHalf)
        launch(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(out.data_ptr<at::Half>()), pair);
    else if (x.dtype() == torch::kHalf)
        launch(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
               out.data_ptr<float>(), pair);
    else if (out.dtype() == torch::kHalf)
        launch(x.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), pair);
    else
        launch(x.data_ptr<float>(), out.data_ptr<float>(), pair);
}

}  // namespace

void fq_gemv_q8_0(torch::Tensor x, torch::Tensor packed, torch::Tensor out)
{
    if (fq_use_q8q8())
    {
        // q8_1-activation path: quantize once per row, integer dp4a dots.
        TORCH_CHECK((int)x.size(1) % FQ_Q8_1_GROUP == 0, "K must be a multiple of 32");
        int M = (int)x.size(0), K = (int)x.size(1);
        torch::Tensor qb = fq_q8_scratch(x.get_device(), M, K);
        fq_quant_q8_1_rows(x, qb);
        fq_gemv_q8_0_q8(qb, packed, out);
        return;
    }
    fq_gemv_q<0>(x, packed, {}, 1.0, out);
}

void fq_gemv_iq2_xxs(torch::Tensor x, torch::Tensor packed, torch::Tensor out)
{
    if (fq_use_q8act())
    {
        // q8_1-activation path: quantize once per row, integer dp4a dots.
        TORCH_CHECK((int)x.size(1) % FQ_Q8_1_GROUP == 0, "K must be a multiple of 32");
        int M = (int)x.size(0), K = (int)x.size(1);
        torch::Tensor qb = fq_q8_scratch(x.get_device(), M, K);
        fq_quant_q8_1_rows(x, qb);
        fq_gemv_iq2_xxs_q8(qb, packed, out);
        return;
    }
    fq_gemv_q<1>(x, packed, {}, 1.0, out);
}

void fq_gemv_nvfp4(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, double global_scale,
                   torch::Tensor out, bool pair)
{
    if (fq_use_q8nv())
    {
        // q8_1-activation path: quantize once per row, integer dp4a dots.
        int K = (int)x.size(1);
        TORCH_CHECK(K % FQ_Q8_1_GROUP == 0, "K must be a multiple of 32");
        int M = (int)x.size(0);
        torch::Tensor qb = fq_q8_scratch(x.get_device(), M, K);
        fq_quant_q8_1_rows(x, qb);
        fq_gemv_nvfp4_q8(qb, packed, scales, global_scale, out);
        return;
    }
    fq_gemv_q<2>(x, packed, scales, global_scale, out, pair);
}
