// q8_1-activation decode kernels for foreign-quant weights (Bolt phase 2):
// port of the llama.cpp mmvq design. Activations are quantized ONCE per
// row to q8_1 blocks (fq_quant_q8_1_kernel, same arithmetic as ggml's
// quantize_q8_1) into a caller-cached global scratch, then row-per-block
// dot kernels run INTEGER dp4a dots (fq_dot_iq2_xxs_q8 in fq_gemv.cuh).
//
// Same portability contract as fq_gemv.cu (no PTX, no warp shuffles,
// shared-memory reduction, __fmul_rn/__fadd_rn + fq_dp4a only).
#include <cuda_fp16.h>
#include <map>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_gemv.cuh"
#include "fq_api.h"

namespace {

// Quantize M rows (x: [M, K] half/float) to q8_1 blocks in qb ([M][nb32]
// blocks of 36B). Grid: M*nb32 blocks x 32 threads; block (m, b) owns
// 32-block b of row m. Each thread independently recomputes the block
// amax over all 32 elements (redundant but L1-served and sync-free),
// then writes its own qs byte; thread 0 additionally stores (d, sum).
// No shared memory, no __syncthreads, no cross-thread state at all.
// Matches ggml quantize_q8_1: d = amax/127, q = round-half-away(xi/d).
template <typename IN>
__global__ void fq_quant_q8_1_kernel(const IN* __restrict__ x, uint8_t* __restrict__ qb,
                                     int M, int K)
{
    int blk = blockIdx.x;
    int nb32 = K / FQ_Q8_1_GROUP;
    int m = blk / nb32;
    int b = blk % nb32;
    int tid = threadIdx.x;
    if (m >= M)
        return;
    const IN* xr = x + (int64_t)m * K + (int64_t)b * 32;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++)
    {
        float vi;
        if constexpr (std::is_same<IN, __half>::value)
            vi = __half2float(xr[i]);
        else
            vi = xr[i];
        amax = fmaxf(amax, fabsf(vi));
    }
    float d = amax / 127.0f;
    float v;
    if constexpr (std::is_same<IN, __half>::value)
        v = __half2float(xr[tid]);
    else
        v = xr[tid];
    int q = (amax == 0.0f) ? 0 : (int)roundf(v / d);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    uint8_t* qbblk = qb + ((int64_t)m * nb32 + b) * FQ_Q8_1_BYTES;
    qbblk[4 + tid] = (uint8_t)(int8_t)q;
    if (tid == 0)
    {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++)
        {
            if constexpr (std::is_same<IN, __half>::value)
                sum += __half2float(xr[i]);
            else
                sum += xr[i];
        }
        // NOTE: store dh/sh with single u16 stores, NOT memcpy. The
        // memcpy byte-split pattern (shr + 2x st.global.u8) loses the low
        // byte with this toolchain (observed constant 0x1E65 stored as
        // 0x1E00); single st.global.u16 is exact. qbblk is 2B-aligned
        // (torch base alignment, b*36 even).
        __half dh = __float2half_rn(d);
        __half sh = __float2half_rn(sum);
        uint16_t dhb, shb;
        memcpy(&dhb, &dh, 2);
        memcpy(&shb, &sh, 2);
        reinterpret_cast<uint16_t*>(qbblk)[0] = dhb;
        reinterpret_cast<uint16_t*>(qbblk)[1] = shb;
    }
}

// Row-per-block IQ2_XXS q8 dot: block bidx computes output row n of x-row
// m against the packed IQ2_XXS matrix. NT = blockDim.x threads take
// 32-elem chunks round-robin; partials reduce sequentially (portable).
// V2 (default, BOLT_FQ_Q8V2=1): the v2 dot needs no prologue tables at
// all (const grid, inline signs, direct headers), so shmem is just the
// reduction row; V2=false keeps the v1 staged path for same-binary A/B.
template <typename OUT, bool V2>
__global__ void fq_gemv_q8_iq2_kernel(const uint8_t* __restrict__ aqb,
                                      const uint8_t* __restrict__ packed,
                                      OUT* __restrict__ out,
                                      int M, int N, int K, int64_t packed_stride,
                                      int64_t qb_stride)
{
    int bidx = blockIdx.x;
    int m = bidx / N;
    int n = bidx % N;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    int nb = K / FQ_IQ2_XXS_GROUP;

    extern __shared__ uint8_t sh[];
    float* red;
    uint32_t* hdrs = nullptr;
    uint32_t* signs32 = nullptr;
    __shared__ fq_tables_sh tab;
    const uint8_t* grid8 = nullptr;
    if constexpr (!V2)
    {
        // Layout: hdrs (nb*16 u32), signs32 (256 u32), red (nt floats).
        hdrs = reinterpret_cast<uint32_t*>(sh);
        signs32 = hdrs + (size_t)nb * 16;
        red = reinterpret_cast<float*>(signs32 + 256);
        fq_tables_load(&tab, 0, 1);
        if (n < N)
            fq_stage_iq2_hdrs_row(packed, packed_stride, hdrs, nb, n, N);
        // Barrier BEFORE signs32_build consumes the staged tab: the staging
        // above is cooperative (different threads write different entries) and
        // the build reads other threads' entries. (The kernel's later
        // __syncthreads only covers the dot; without this one the build races
        // staging and signs come out zero/garbage nondeterministically.)
        __syncthreads();
        fq_signs32_build(&tab, signs32);
        __syncthreads();
        grid8 = reinterpret_cast<const uint8_t*>(tab.grid);
    }
    else
    {
        red = reinterpret_cast<float*>(sh);
    }
    float acc = 0.0f;
    if (n < N)
    {
        if constexpr (V2)
            acc = fq_dot_iq2_xxs_q8v2(packed + (int64_t)n * packed_stride,
                                      aqb + (int64_t)m * qb_stride, K, tid, nt);
        else
            acc = fq_dot_iq2_xxs_q8(packed + (int64_t)n * packed_stride, hdrs, grid8,
                                    signs32, aqb + (int64_t)m * qb_stride, K, tid, nt);
    }
    red[tid] = acc;
    __syncthreads();
    if (n < N && tid == 0)
    {
        float s = 0.0f;
        for (int l = 0; l < nt; l++)
            s = __fadd_rn(s, red[l]);
        if constexpr (std::is_same<OUT, __half>::value)
            out[(int64_t)m * N + n] = __float2half_rn(s);
        else
            out[(int64_t)m * N + n] = s;
    }
}

// Row-per-block NVFP4 q8 dot: block bidx computes output row n of x-row
// m. Same grid/mapping as the IQ2 kernel above; the dot is fq_dot_nvfp4_q8
// (4 dp4a per 16-group, split packed/scales layout). The row-uniform
// `global` multiplies once inside the dot.
template <typename OUT>
__global__ void fq_gemv_q8_nvfp4_kernel(const uint8_t* __restrict__ aqb,
                                        const uint8_t* __restrict__ packed,
                                        const uint8_t* __restrict__ scales,
                                        float global,
                                        OUT* __restrict__ out,
                                        int M, int N, int K, int64_t packed_stride,
                                        int64_t scales_stride, int64_t qb_stride)
{
    int bidx = blockIdx.x;
    int m = bidx / N;
    int n = bidx % N;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    extern __shared__ uint8_t sh[];
    // Layout: e2m1db (128 int8), red (nt floats).
    int8_t* e2m1db = reinterpret_cast<int8_t*>(sh);
    float* red = reinterpret_cast<float*>(e2m1db + 128);

    fq_e2m1db_build(e2m1db);
    __syncthreads();
    float acc = 0.0f;
    if (n < N)
    {
        acc = fq_dot_nvfp4_q8(packed + (int64_t)n * packed_stride,
                              scales + (int64_t)n * scales_stride,
                              aqb + (int64_t)m * qb_stride, e2m1db,
                              K, global, tid, nt);
    }
    red[tid] = acc;
    __syncthreads();
    if (n < N && tid == 0)
    {
        float s = 0.0f;
        for (int l = 0; l < nt; l++)
            s = __fadd_rn(s, red[l]);
        if constexpr (std::is_same<OUT, __half>::value)
            out[(int64_t)m * N + n] = __float2half_rn(s);
        else
            out[(int64_t)m * N + n] = s;
    }
}

// Row-per-block Q8_0 q8 dot: same grid; the dot is fq_dot_q8_0_q8
// (8 dp4a per 32-block, weights are int8, no tables).
template <typename OUT>
__global__ void fq_gemv_q8_q8_kernel(const uint8_t* __restrict__ aqb,
                                     const uint8_t* __restrict__ packed,
                                     OUT* __restrict__ out,
                                     int M, int N, int K, int64_t packed_stride,
                                     int64_t qb_stride)
{
    int bidx = blockIdx.x;
    int m = bidx / N;
    int n = bidx % N;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    extern __shared__ float red[];

    float acc = 0.0f;
    if (n < N)
    {
        acc = fq_dot_q8_0_q8(packed + (int64_t)n * packed_stride,
                             aqb + (int64_t)m * qb_stride, K, tid, nt);
    }
    red[tid] = acc;
    __syncthreads();
    if (n < N && tid == 0)
    {
        float s = 0.0f;
        for (int l = 0; l < nt; l++)
            s = __fadd_rn(s, red[l]);
        if constexpr (std::is_same<OUT, __half>::value)
            out[(int64_t)m * N + n] = __float2half_rn(s);
        else
            out[(int64_t)m * N + n] = s;
    }
}

// Cached global q8_1 scratch: [M][nb32(K)] 36B blocks. One buffer serves
// all layers of a decode step (single stream: no two dots live at once;
// allocated on first use = outside CUDA-graph capture, like act_cache).
torch::Tensor fq_q8_scratch_cached(int dev, int M, int K)
{
    static thread_local std::map<std::tuple<int, int, int>, torch::Tensor> cache;
    auto key = std::make_tuple(dev, M, K);
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;
    auto opts = torch::dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, dev));
    torch::Tensor t = torch::empty({M, (K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES}, opts);
    cache[key] = t;
    return t;
}

}  // namespace

torch::Tensor fq_q8_scratch(int dev, int M, int K)
{
    return fq_q8_scratch_cached(dev, M, K);
}

void fq_quant_q8_1_rows(torch::Tensor x, torch::Tensor qb)
{
    TORCH_CHECK(x.is_cuda() && qb.is_cuda(), "tensors must be CUDA");    TORCH_CHECK(x.is_contiguous() && qb.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kHalf || x.dtype() == torch::kFloat, "x must be half or float");
    TORCH_CHECK_DIM(x, 2);
    TORCH_CHECK_DIM(qb, 2);
    int M = (int)x.size(0), K = (int)x.size(1);
    TORCH_CHECK(K % FQ_Q8_1_GROUP == 0, "K must be a multiple of 32");
    TORCH_CHECK((int)qb.size(0) == M &&
                (int)qb.size(1) == (K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES,
                "qb must be [M, nb32*36]");
    at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int nb32 = K / FQ_Q8_1_GROUP;
    auto launch = [&](auto* xp)
    {
        fq_quant_q8_1_kernel<<<M * nb32, 32, 0, stream>>>(
            xp, qb.data_ptr<uint8_t>(), M, K);
    };
    if (x.dtype() == torch::kHalf)
        launch(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()));
    else
        launch(x.data_ptr<float>());
}

// Shared row-per-block q8 launch sizing: threads take 32-elem chunks
// round-robin (same as the IQ2 kernel).
static inline int fq_q8_nt(int K)
{
    int nchunks = K / 32;
    return nchunks >= 256 ? 256 : ((nchunks + 31) / 32) * 32;
}

// q8_1-activation NVFP4 decode GEMV. x is quantized by the caller into
// qb (fq_quant_q8_1_rows); out is half or float [M, N].
void fq_gemv_nvfp4_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor scales,
                      double global_scale, torch::Tensor out)
{
    TORCH_CHECK(qb.is_cuda() && packed.is_cuda() && scales.is_cuda() && out.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(qb.is_contiguous() && packed.is_contiguous() && scales.is_contiguous() &&
                out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(out.dtype() == torch::kHalf || out.dtype() == torch::kFloat,
                "out must be half or float");
    TORCH_CHECK_DIM(qb, 2);
    TORCH_CHECK_DIM(packed, 2);
    TORCH_CHECK_DIM(scales, 2);
    TORCH_CHECK_DIM(out, 2);
    int M = (int)qb.size(0);
    int N = (int)packed.size(0);
    TORCH_CHECK((int)out.size(0) == M, "out rows must match qb rows");
    TORCH_CHECK((int)out.size(1) == N, "out cols must match N");
    int K = (int)qb.size(1) / FQ_Q8_1_BYTES * FQ_Q8_1_GROUP;
    TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32");
    TORCH_CHECK((int)packed.size(1) == K / 2, "packed width must be K/2");
    TORCH_CHECK((int)scales.size(1) == K / 16, "scales width must be K/16");
    int64_t pstride = (int64_t)(K / 2);
    int64_t sstride = (int64_t)(K / 16);
    int64_t qb_stride = (int64_t)(K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
    int nt = fq_q8_nt(K);
    size_t shmem = 128 * sizeof(int8_t) + (size_t)nt * sizeof(float);
    at::cuda::CUDAGuard guard(qb.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto launch = [&](auto* op)
    {
        fq_gemv_q8_nvfp4_kernel<std::decay_t<decltype(*op)>> <<<M * N, nt, shmem, stream>>>(
            qb.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
            (float)global_scale, op, M, N, K, pstride, sstride, qb_stride);
    };
    if (out.dtype() == torch::kHalf)
        launch(reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
    else
        launch(out.data_ptr<float>());
}

// q8_1-activation Q8_0 decode GEMV. Same contract as above.
void fq_gemv_q8_0_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    TORCH_CHECK(qb.is_cuda() && packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(qb.is_contiguous() && packed.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(out.dtype() == torch::kHalf || out.dtype() == torch::kFloat,
                "out must be half or float");
    TORCH_CHECK_DIM(qb, 2);
    TORCH_CHECK_DIM(packed, 3);
    TORCH_CHECK_DIM(out, 2);
    int M = (int)qb.size(0);
    int N = (int)packed.size(0);
    TORCH_CHECK((int)out.size(0) == M, "out rows must match qb rows");
    TORCH_CHECK((int)out.size(1) == N, "out cols must match N");
    TORCH_CHECK_SIZE(packed, 2, FQ_Q8_0_BYTES);
    int nb = (int)packed.size(1);
    int K = nb * FQ_Q8_0_GROUP;
    TORCH_CHECK((int)qb.size(1) == (K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES, "qb width must match K");
    int64_t pstride = (int64_t)nb * FQ_Q8_0_BYTES;
    int64_t qb_stride = (int64_t)(K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
    int nt = fq_q8_nt(K);
    size_t shmem = (size_t)nt * sizeof(float);
    at::cuda::CUDAGuard guard(qb.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto launch = [&](auto* op)
    {
        fq_gemv_q8_q8_kernel<std::decay_t<decltype(*op)>> <<<M * N, nt, shmem, stream>>>(
            qb.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(),
            op, M, N, K, pstride, qb_stride);
    };
    if (out.dtype() == torch::kHalf)
        launch(reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
    else
        launch(out.data_ptr<float>());
}

// q8_1-activation IQ2_XXS decode GEMV. x is quantized by the caller into
// qb (fq_quant_q8_1_rows); out is half or float [M, N].
void fq_gemv_iq2_xxs_q8(torch::Tensor qb, torch::Tensor packed, torch::Tensor out)
{
    TORCH_CHECK(qb.is_cuda() && packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(qb.is_contiguous() && packed.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(out.dtype() == torch::kHalf || out.dtype() == torch::kFloat,
                "out must be half or float");
    TORCH_CHECK_DIM(qb, 2);
    TORCH_CHECK_DIM(packed, 3);
    TORCH_CHECK_DIM(out, 2);
    int M = (int)qb.size(0);
    int N = (int)packed.size(0);
    TORCH_CHECK((int)out.size(0) == M, "out rows must match qb rows");
    TORCH_CHECK((int)out.size(1) == N, "out cols must match N");
    TORCH_CHECK_SIZE(packed, 2, FQ_IQ2_XXS_BYTES);
    int nb = (int)packed.size(1);
    int K = nb * FQ_IQ2_XXS_GROUP;
    TORCH_CHECK((int)qb.size(1) == (K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES, "qb width must match K");
    TORCH_CHECK(K <= 8192, "IQ2 header scratch covers K <= 8192");
    int64_t pstride = (int64_t)nb * FQ_IQ2_XXS_BYTES;
    int64_t qb_stride = (int64_t)(K / FQ_Q8_1_GROUP) * FQ_Q8_1_BYTES;
    int nchunks = K / 32;
    int nt = fq_q8_nt_iq2(K);
    bool v2 = fq_use_q8v2();
    size_t shmem = v2 ? (size_t)nt * sizeof(float)
                      : (size_t)nb * 16 * sizeof(uint32_t) + 256 * sizeof(uint32_t) +
                        (size_t)nt * sizeof(float);
    at::cuda::CUDAGuard guard(qb.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto launch = [&](auto* op)
    {
        if (v2)
            fq_gemv_q8_iq2_kernel<std::decay_t<decltype(*op)>, true> <<<M * N, nt, shmem, stream>>>(
                qb.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(),
                op, M, N, K, pstride, qb_stride);
        else
            fq_gemv_q8_iq2_kernel<std::decay_t<decltype(*op)>, false> <<<M * N, nt, shmem, stream>>>(
                qb.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(),
                op, M, N, K, pstride, qb_stride);
    };
    if (out.dtype() == torch::kHalf)
        launch(reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
    else
        launch(out.data_ptr<float>());
}
