// Dense Q8_0 x q8_1 decode GEMV, few rows, column-tiled (HY4_DENSE_V2): one block = nt threads (the thread -> chunk map of
// fq_gemv_q8_q8_kernel / fq_gemv_q8_q8_mrows_kernel, exllamav3_ext/foreign_quant/fq_gemv_q8.cu) walks `cpb` consecutive columns.
// What changes against the stock kernels: the q8_1 activation blocks of every row are loaded ONCE per thread into registers
// (stock: reloaded for every column, 9 loads per row per chunk), the weight block of the next column is in flight while the
// current one is dotted, and one barrier + one ordered reduction per column replace one block launch per column.
// Arithmetic per output is identical to the stock kernels (same int8 dot, same per-thread fp32 chain over chunk tid, tid + nt, ...,
// same sequential sum over the nt thread slots): bit-identical at every row count.
#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "gm_dense_q8.cuh"

#define GDQ_Q8_1_BYTES 36
#define GDQ_Q8_0_BYTES 34

namespace {

__device__ __forceinline__ int gdq_dp4a(int a, int b, int c)
{
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

template <int NCH>
struct GdqW
{
    int32_t q[NCH][8];
    float d[NCH];
};

template <int NCH>
__device__ __forceinline__ void gdq_load_w(GdqW<NCH>& w, const uint8_t* wr, int tid, int nt, int nchunks, bool on)
{
    #pragma unroll
    for (int ch = 0; ch < NCH; ch++)
    {
        int c = tid + ch * nt;
        if (on && c < nchunks)
        {
            const uint8_t* blk = wr + (size_t)c * GDQ_Q8_0_BYTES;
            __half dwh;
            memcpy(&dwh, blk, 2);
            w.d[ch] = __half2float(dwh);
            const uint16_t* w16 = reinterpret_cast<const uint16_t*>(blk + 2);
            #pragma unroll
            for (int j2 = 0; j2 < 8; j2++)
                w.q[ch][j2] = (int32_t)(w16[j2 * 2] | ((uint32_t)w16[j2 * 2 + 1] << 16));
        }
    }
}

template <typename OUT, int MR, int NCH>
__global__ void gdq_kernel(const uint8_t* __restrict__ aqb, const uint8_t* __restrict__ packed, OUT* __restrict__ out,
                           int M, int N, int K, int64_t packed_stride, int64_t qb_stride, int cpb)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    const int n0 = blockIdx.x * cpb;
    const int m0 = blockIdx.y * MR;
    const int mc = min(MR, M - m0);
    const int nchunks = K / 32;
    const int rs = nt + 4;                     // padded row stride of the reduction buffer (float4 reads, spread banks)
    extern __shared__ float red[];             // [2][MR][rs]

    int32_t aq[NCH][MR][8];
    float da[NCH][MR];
    #pragma unroll
    for (int ch = 0; ch < NCH; ch++)
    {
        int c = tid + ch * nt;
        #pragma unroll
        for (int r = 0; r < MR; r++)
        {
            if (r < mc && c < nchunks)
            {
                const uint8_t* ab = aqb + (int64_t)(m0 + r) * qb_stride + (size_t)c * GDQ_Q8_1_BYTES;
                __half dah;
                memcpy(&dah, ab, 2);
                da[ch][r] = __half2float(dah);
                const int32_t* aq32 = reinterpret_cast<const int32_t*>(ab + 4);
                #pragma unroll
                for (int j2 = 0; j2 < 8; j2++)
                    aq[ch][r][j2] = aq32[j2];
            }
        }
    }

    GdqW<NCH> cur, nxt;
    gdq_load_w<NCH>(cur, packed + (int64_t)n0 * packed_stride, tid, nt, nchunks, n0 < N);
    for (int j = 0; j < cpb; j++)
    {
        const int n = n0 + j;
        if (n >= N)
            break;
        gdq_load_w<NCH>(nxt, packed + (int64_t)(n + 1) * packed_stride, tid, nt, nchunks, j + 1 < cpb && n + 1 < N);
        float acc[MR];
        #pragma unroll
        for (int r = 0; r < MR; r++) acc[r] = 0.0f;
        #pragma unroll
        for (int ch = 0; ch < NCH; ch++)
        {
            if (tid + ch * nt < nchunks)
            {
                #pragma unroll
                for (int r = 0; r < MR; r++)
                {
                    if (r < mc)
                    {
                        int sumi = 0;
                        #pragma unroll
                        for (int j2 = 0; j2 < 8; j2++)
                            sumi = gdq_dp4a(cur.q[ch][j2], aq[ch][r][j2], sumi);
                        acc[r] = __fadd_rn(acc[r], __fmul_rn(__fmul_rn(cur.d[ch], da[ch][r]), (float)sumi));
                    }
                }
            }
        }
        float* rb = red + (size_t)(j & 1) * MR * rs;
        #pragma unroll
        for (int r = 0; r < MR; r++) rb[r * rs + tid] = acc[r];
        __syncthreads();
        if (tid < mc)
        {
            const float4* p = reinterpret_cast<const float4*>(rb + tid * rs);
            float s = 0.0f;
            #pragma unroll 4
            for (int l = 0; l < nt / 4; l++)
            {
                float4 v = p[l];
                s = __fadd_rn(s, v.x);
                s = __fadd_rn(s, v.y);
                s = __fadd_rn(s, v.z);
                s = __fadd_rn(s, v.w);
            }
            if constexpr (std::is_same<OUT, __half>::value)
                out[(int64_t)(m0 + tid) * N + n] = __float2half_rn(s);
            else
                out[(int64_t)(m0 + tid) * N + n] = s;
        }
        cur = nxt;
    }
}

template <typename OUT, int MR, int NCH>
void gdq_launch(const at::Tensor& qb, const at::Tensor& packed, OUT* o, int M, int N, int K, int cpb, int nt, cudaStream_t stream)
{
    int64_t pstride = (int64_t)(K / 32) * GDQ_Q8_0_BYTES, qstride = (int64_t)(K / 32) * GDQ_Q8_1_BYTES;
    dim3 grid((N + cpb - 1) / cpb, (M + MR - 1) / MR);
    size_t shmem = (size_t)2 * MR * (nt + 4) * sizeof(float);
    gdq_kernel<OUT, MR, NCH><<<grid, nt, shmem, stream>>>(qb.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(), o, M, N, K, pstride, qstride, cpb);
}

template <typename OUT, int NCH>
void gdq_rows(const at::Tensor& qb, const at::Tensor& packed, OUT* o, int M, int N, int K, int cpb, int nt, cudaStream_t stream)
{
    if (M <= 1) gdq_launch<OUT, 1, NCH>(qb, packed, o, M, N, K, cpb, nt, stream);
    else if (M == 2) gdq_launch<OUT, 2, NCH>(qb, packed, o, M, N, K, cpb, nt, stream);
    else if (M <= 4) gdq_launch<OUT, 4, NCH>(qb, packed, o, M, N, K, cpb, nt, stream);
    else gdq_launch<OUT, 8, NCH>(qb, packed, o, M, N, K, cpb, nt, stream);
}

}  // namespace

// qb: [M, K/32*36] uint8 (fq_quant_q8_1_rows), packed: [N, K/32, 34] uint8, out: [M, N] float or half. cpb = columns per block.
// Returns false when the shape is outside the kernel (K > 16384 or K % 32): the caller keeps the stock launch.
bool gm_dense_q8(const at::Tensor& qb, const at::Tensor& packed, at::Tensor& out, int64_t cpb)
{
    TORCH_CHECK(qb.is_cuda() && packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(qb.is_contiguous() && packed.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    int M = (int)qb.size(0), N = (int)packed.size(0), K = (int)packed.size(1) * 32;
    TORCH_CHECK(packed.dim() == 3 && packed.size(2) == GDQ_Q8_0_BYTES && (int)out.size(0) == M && (int)out.size(1) == N, "shape mismatch");
    TORCH_CHECK((int)qb.size(1) == (K / 32) * GDQ_Q8_1_BYTES, "qb width must match K");
    int nchunks = K / 32;
    int nt = nchunks >= 256 ? 256 : ((nchunks + 31) / 32) * 32;
    int nch = (nchunks + nt - 1) / nt;
    if (nch > 2 || M < 1 || cpb < 1)
        return false;
    at::cuda::CUDAGuard guard(qb.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto go = [&](auto* op)
    {
        using O = std::decay_t<decltype(*op)>;
        if (nch == 1) gdq_rows<O, 1>(qb, packed, op, M, N, K, (int)cpb, nt, stream);
        else gdq_rows<O, 2>(qb, packed, op, M, N, K, (int)cpb, nt, stream);
    };
    if (out.dtype() == torch::kHalf)
        go(reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
    else if (out.dtype() == torch::kFloat)
        go(out.data_ptr<float>());
    else
        return false;
    return true;
}

#endif
