// Foreign-quant dequant kernels: Q8_0 / IQ2_XXS / NVFP4 -> fp16.
// Portable CUDA/HIP (no PTX, no wave-size assumptions). v1: one thread
// per output value, grid-stride loop, scalar loads. v2 (default,
// BOLT_FQ_DQV2=0 restores v1): streaming 2D grids over (block, row) --
// no index division, direct u16 header pairs, inline sign unpack, and a
// 3 KB global table pair for NVFP4 -- same per-element fp32 op order,
// bitwise-identical output. Bit-exact vs
// exllamav3/foreign_quant/*.py (see fq_dequant.cuh contract).
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_dequant.cuh"
#include "fq_gemv.cuh"
#include "fq_api.h"

namespace {

// 3 KB global NVFP4 tables for the streaming v2 kernel: e2m1p[256] as
// float2 (lo,hi) + e4m3f[256] as float, both bit-exact copies of
// FQ_E2M1_LUT / FQ_E4M3_BITS (exact floats / exact bits, no arithmetic).
// Built once per device on first use (input-independent constants).
__global__ void fq_dq_tables_build(float* __restrict__ e2m1p, float* __restrict__ e4m3f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256)
        return;
    e2m1p[2 * i] = FQ_E2M1_LUT[i & 15];
    e2m1p[2 * i + 1] = FQ_E2M1_LUT[i >> 4];
    e4m3f[i] = __uint_as_float(FQ_E4M3_BITS[i]);
}

torch::Tensor fq_dq_e2m1p_cache;
torch::Tensor fq_dq_e4m3f_cache;
c10::DeviceIndex fq_dq_tab_dev = -1;

void fq_dq_ensure_tables(torch::Tensor like)
{
    c10::DeviceIndex dev = like.get_device();
    if (fq_dq_tab_dev == dev && fq_dq_e2m1p_cache.defined())
        return;
    at::cuda::CUDAGuard guard(dev);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fq_dq_e2m1p_cache = torch::empty({256, 2}, torch::dtype(torch::kFloat32).device(like.device()));
    fq_dq_e4m3f_cache = torch::empty({256}, torch::dtype(torch::kFloat32).device(like.device()));
    fq_dq_tables_build<<<1, 256, 0, stream>>>(
        fq_dq_e2m1p_cache.data_ptr<float>(), fq_dq_e4m3f_cache.data_ptr<float>());
    fq_dq_tab_dev = dev;
}

// IQ2_XXS streaming dequant v2: 2D grid (Kb, N), 256 threads = one 66B
// input block -> 256 fp16 outputs. No index division (block IS the
// block), headers via direct u16 pairs, signs via inline unpack (no
// divergent const LUT), grid via global LDG. Same per-element fp32 op
// order as fq_dq_kernel<IQ2> (db = d*(0.5+(a1>>28))*0.25 left-assoc,
// v = (db*mag)*sgn), hence bitwise-identical.
__global__ void fq_dq_iq2_kernel_v2(const uint8_t* __restrict__ packed, __half* __restrict__ out,
                                    int64_t kb)
{
    int b = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;  // 0..255
    const uint8_t* blk = packed + ((int64_t)n * kb + b) * FQ_IQ2_XXS_BYTES;
    __half dh;
    memcpy(&dh, blk, 2);
    float d = __half2float(dh);
    int ib = tid >> 5;          // sub-block 0..7
    int elle = (tid >> 3) & 3;  // grid entry 0..3
    int j = tid & 7;            // lane 0..7
    const uint16_t* h16 = reinterpret_cast<const uint16_t*>(blk + 2 + ib * 8);
    uint32_t a0 = (uint32_t)h16[0] | ((uint32_t)h16[1] << 16);
    uint32_t a1 = (uint32_t)h16[2] | ((uint32_t)h16[3] << 16);
    float db = __fmul_rn(__fmul_rn(d, 0.5f + (float)(a1 >> 28)), 0.25f);
    uint64_t grid = FQ_IQ2XXS_GRID[(a0 >> (8 * elle)) & 0xFF];
    float mag = (float)((grid >> (8 * j)) & 0xFF);
    uint32_t s4 = fq_unpack_ksigns((uint8_t)((a1 >> (7 * elle)) & 127));
    // s4 replicates the 8 sign bits (KSIGNS entry) across 4 bytes;
    // 0x01010101<<j selects bit j in every byte (== old KSIGNS&KMASK).
    float sgn = (s4 & (0x01010101u << j)) ? -1.0f : 1.0f;
    float v = __fmul_rn(__fmul_rn(db, mag), sgn);
    out[((int64_t)n * kb + b) * FQ_IQ2_XXS_GROUP + tid] = __float2half_rn(v);
}

// NVFP4 streaming dequant v2: 2D grid (K/256, N), 256 threads = 256
// consecutive outputs of row n. Per thread: packed byte + scale byte +
// global table loads, 2 FMUL. Same op order as fq_nvfp4_group
// (w = (e2m1*gscale)*global left-assoc), hence bitwise-identical.
__global__ void fq_dq_nvfp4_kernel_v2(const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scales,
                                      const float* __restrict__ e2m1p, const float* __restrict__ e4m3f,
                                      __half* __restrict__ out, int64_t nrows, int64_t khalf,
                                      int64_t K, float global)
{
    int n = blockIdx.y;
    int64_t k = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (k >= K)
        return;
    (void)nrows;
    uint8_t byte = packed[(int64_t)n * khalf + (k >> 1)];
    const float* p = e2m1p + (size_t)byte * 2;
    float e = (k & 1) ? p[1] : p[0];
    float gs = e4m3f[scales[(int64_t)n * (khalf / 8) + (k >> 4)]];
    float w = __fmul_rn(__fmul_rn(e, gs), global);
    out[(int64_t)n * K + k] = __float2half_rn(w);
}

// Q8_0 streaming dequant v2: 2D grid (Kb*8, N), 256 threads = 8 blocks
// -> 256 outputs. Vectorization-friendly scalar (one qs byte + one FMUL
// per thread, d broadcast per 32). Same op order as v1 (v = qs*d),
// hence bitwise-identical.
__global__ void fq_dq_q8_kernel_v2(const uint8_t* __restrict__ packed, __half* __restrict__ out,
                                    int64_t kb)
{
    int bb = blockIdx.x;  // 8-block group
    int n = blockIdx.y;
    int tid = threadIdx.x;  // 0..255
    int64_t b = (int64_t)bb * 8 + (tid >> 5);
    int j = tid & 31;
    if (b >= kb)
        return;
    const uint8_t* blk = packed + ((int64_t)n * kb + b) * FQ_Q8_0_BYTES;
    __half dh;
    memcpy(&dh, blk, 2);
    float v = __fmul_rn((float)((int8_t)blk[2 + j]), __half2float(dh));
    out[((int64_t)n * kb + b) * FQ_Q8_0_GROUP + j] = __float2half_rn(v);
}

// out[n * K + k] for packed[n * Kb, B]; G = group size, B = block bytes.
template <int G, int B>
__global__ void fq_dq_kernel(const uint8_t* __restrict__ packed, __half* __restrict__ out,
                             int64_t rows, int64_t kb, float* tmp)
{
    (void)tmp;
    int64_t total = rows * kb * G;
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (int64_t)gridDim.x * blockDim.x)
    {
        int64_t g = t % G;
        int64_t b = (t / G) % kb;
        int64_t n = t / (G * kb);
        const uint8_t* blk = packed + (n * kb + b) * B;
        float v;
        if constexpr (G == FQ_Q8_0_GROUP)
        {
            __half dh;
            memcpy(&dh, blk, 2);
            v = __fmul_rn((float)((int8_t)blk[2 + g]), __half2float(dh));
        }
        else
        {
            // IQ2_XXS: decode the value's own (sub-block, grid entry, lane).
            // The 66-byte block is tiny; every lane recomputes its sub-block's
            // db from the shared header (L1-cached), keeping the kernel
            // wave-size agnostic with no cross-lane traffic.
            int ib = (int)(g >> 5);          // sub-block 0..7 (32 values each)
            int elle = (int)((g >> 3) & 3);  // grid entry 0..3 within sub-block
            int j = (int)(g & 7);            // lane 0..7 within grid entry
            const uint8_t* q = blk + 2 + 8 * ib;
            uint32_t q0 = (uint32_t)q[0] | ((uint32_t)q[1] << 8);
            uint32_t q1 = (uint32_t)q[2] | ((uint32_t)q[3] << 8);
            uint32_t q2 = (uint32_t)q[4] | ((uint32_t)q[5] << 8);
            uint32_t q3 = (uint32_t)q[6] | ((uint32_t)q[7] << 8);
            uint32_t a0 = q0 | (q1 << 16);
            uint32_t a1 = q2 | (q3 << 16);
            __half dh;
            memcpy(&dh, blk, 2);
            float db = __fmul_rn(__fmul_rn(__half2float(dh), 0.5f + (float)(a1 >> 28)), 0.25f);
            uint64_t grid = FQ_IQ2XXS_GRID[(a0 >> (8 * elle)) & 0xFF];
            uint8_t signs = FQ_KSIGNS_IQ2XS[(a1 >> (7 * elle)) & 127];
            float mag = (float)((grid >> (8 * j)) & 0xFF);
            float sgn = (signs & FQ_KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
            v = __fmul_rn(__fmul_rn(db, mag), sgn);
        }
        out[t] = __float2half_rn(v);
    }
}

__global__ void fq_dq_nvfp4_kernel(const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scales,
                                   __half* __restrict__ out, int64_t rows, int64_t k_half, float global)
{
    // packed [rows, K/2], scales [rows, K/16]; one thread per output value.
    int64_t total = rows * k_half * 2;
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (int64_t)gridDim.x * blockDim.x)
    {
        int64_t k = t % (k_half * 2);
        int64_t n = t / (k_half * 2);
        uint8_t byte = packed[n * k_half + (k >> 1)];
        uint8_t nib = (k & 1) ? (byte >> 4) & 0xF : byte & 0xF;
        float gs = fq_e4m3(scales[n * (k_half / 8) + (k >> 4)]);
        float w = __fmul_rn(__fmul_rn(FQ_E2M1_LUT[nib], gs), global);
        out[t] = __float2half_rn(w);
    }
}

}  // namespace

void fq_dequant_q8_0(torch::Tensor packed, torch::Tensor out)
{
    TORCH_CHECK_DTYPE(packed, kByte);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(packed.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK_DIM(packed, 3);
    TORCH_CHECK_DIM(out, 2);
    TORCH_CHECK_SIZE(packed, 2, FQ_Q8_0_BYTES);
    int64_t n = packed.size(0), kb = packed.size(1);
    TORCH_CHECK(out.size(0) == n && out.size(1) == kb * FQ_Q8_0_GROUP, "out must be [N, Kb*32]");
    at::cuda::CUDAGuard guard(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t total = n * kb * FQ_Q8_0_GROUP;
    if (fq_use_dqv2())
    {
        dim3 grid((unsigned)CEIL_DIVIDE(kb, 8), (unsigned)n);
        fq_dq_q8_kernel_v2<<<grid, 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), kb);
        return;
    }
    fq_dq_kernel<FQ_Q8_0_GROUP, FQ_Q8_0_BYTES>
        <<<CEIL_DIVIDE(total, 256), 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n, kb, nullptr);
}

void fq_dequant_iq2_xxs(torch::Tensor packed, torch::Tensor out)
{
    TORCH_CHECK_DTYPE(packed, kByte);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(packed.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK_DIM(packed, 3);
    TORCH_CHECK_DIM(out, 2);
    TORCH_CHECK_SIZE(packed, 2, FQ_IQ2_XXS_BYTES);
    int64_t n = packed.size(0), kb = packed.size(1);
    TORCH_CHECK(out.size(0) == n && out.size(1) == kb * FQ_IQ2_XXS_GROUP, "out must be [N, Kb*256]");
    at::cuda::CUDAGuard guard(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t total = n * kb * FQ_IQ2_XXS_GROUP;
    if (fq_use_dqv2())
    {
        dim3 grid((unsigned)kb, (unsigned)n);
        fq_dq_iq2_kernel_v2<<<grid, 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), kb);
        return;
    }
    fq_dq_kernel<FQ_IQ2_XXS_GROUP, FQ_IQ2_XXS_BYTES>
        <<<CEIL_DIVIDE(total, 256), 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n, kb, nullptr);
}

void fq_dequant_nvfp4(torch::Tensor packed, torch::Tensor scales, double global_scale, torch::Tensor out)
{
    TORCH_CHECK_DTYPE(packed, kByte);
    TORCH_CHECK_DTYPE(scales, kByte);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(packed.is_cuda() && scales.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(packed.is_contiguous() && scales.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK_DIM(packed, 2);
    TORCH_CHECK_DIM(scales, 2);
    TORCH_CHECK_DIM(out, 2);
    int64_t n = packed.size(0), kh = packed.size(1);
    TORCH_CHECK(scales.size(0) == n && scales.size(1) == kh / 8, "scales must be [N, K/16]");
    TORCH_CHECK(out.size(0) == n && out.size(1) == kh * 2, "out must be [N, K]");
    at::cuda::CUDAGuard guard(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t total = n * kh * 2;
    if (fq_use_dqv2())
    {
        fq_dq_ensure_tables(packed);
        dim3 grid((unsigned)CEIL_DIVIDE(kh * 2, 256), (unsigned)n);
        fq_dq_nvfp4_kernel_v2<<<grid, 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
            fq_dq_e2m1p_cache.data_ptr<float>(), fq_dq_e4m3f_cache.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n, kh, kh * 2, (float)global_scale);
        return;
    }
    fq_dq_nvfp4_kernel<<<CEIL_DIVIDE(total, 256), 256, 0, stream>>>(
        packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n, kh, (float)global_scale);
}
