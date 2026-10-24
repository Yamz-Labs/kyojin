// Foreign-quant dequant to fp16 for the GGUF K-quants (Q4_K / Q5_K / Q6_K)
// and IQ4_XS: the unfused weight path for batched prefill. One dequant pass
// over the raw packed bytes writes the [N, K] fp16 weight, then cuBLAS fp16
// GEMM does the math (foreign_linear._dequant_rows serves these formats;
// rows >= 16 run the unfused dequant + cuBLAS fp16 GEMM prefill path).
//
// Same launch shape as the v2 kernels in fq_dequant.cu: 2D grid (Kb, N),
// 256 threads = one 256-element super-block, threadIdx.x = element index
// inside the block, one fp16 store per thread. gridDim.y caps at 65535 rows
// (the lm_head is N = 248144), so the launchers chunk the rows in groups of
// at most 65535 with pointer offsets -- kernels themselves are unchanged.
// No shared memory, no __syncthreads, no index division. Each thread re-parses the few header
// bytes its own value needs (L1-resident), which keeps the kernels free of
// cross-lane traffic and of any wave-size assumption.
//
// Bit-exactness contract (see fq_dequant.cuh): float32 op-for-op in the same
// order as exllamav3/foreign_quant/k_quants.py dequant_{q4_K,q5_K,q6_K,
// iq4_xs}_block, one rounding per op. Every float op is an explicit
// __fmul_rn/__fsub_rn (never a bare a*b, which --use_fast_math would contract
// into an FMA), then a single __float2half_rn, so the output is
// bitwise-identical to dequant_row(blocks, fmt).astype(np.float16).
//
// Portable CUDA/HIP: no inline PTX, no warp shuffles, no wave-size
// assumptions, only <cuda_fp16.h> intrinsics and the __fmul_rn family. The
// Q4_K/Q5_K 6-bit scale bytes go through the shared fq_q4k_scales()
// (fq_gemv.cuh, verbatim ggml get_scale_min_k4) and IQ4_XS reads the shared
// FQ_KVALUES_IQ4NL codebook; neither bit twiddling is re-implemented here.
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "fq_dequant.cuh"
#include "fq_gemv.cuh"
#include "fq_api.h"

namespace {

// Q4_K: 144 B per 256-block (d half @0, dmin half @2, scales[12] @4,
// qs[128] @16). 256 elements = 4 groups of 64 = 8 sub-groups of 32; the
// numpy loop walks those 64-groups (j = 0, 64, 128, 192) and pulls the scale
// pair (is_+0, is_+1) per group, i.e. sub-group s = e >> 5 gets the pair
// decoded by fq_q4k_scales(..., s >> 1) at index s & 1. First 32 elements of
// a 64-group take the low nibble (d1, m1), next 32 the high nibble (d2, m2):
//     y = (d*sc) * nib - (dmin*m)          (left-assoc, two roundings).
__global__ void fq_dq_q4_k_kernel(const uint8_t* __restrict__ packed, __half* __restrict__ out,
                                   int64_t kb)
{
    int b = blockIdx.x;            // 256-block index within the row
    int n = blockIdx.y;            // row
    int e = threadIdx.x;           // 0..255, element index inside the block
    const uint8_t* blk = packed + ((int64_t)n * kb + b) * FQ_Q4_K_BYTES;
    __half dh, dmh;
    memcpy(&dh, blk, 2);
    memcpy(&dmh, blk + 2, 2);
    float d = __half2float(dh);
    float dmin = __half2float(dmh);
    int c = e >> 6;                // 64-element group 0..3
    int s = e >> 5;                // 32-element sub-group 0..7 == 2*c + (nib. hi)
    int l = e & 31;                // lane inside the sub-group
    int hi = s & 1;                // 0: low nibble of qs, 1: high nibble
    uint8_t sc[2], m[2];
    fq_q4k_scales(blk + 4, s >> 1, sc, m);
    uint8_t q = blk[16 + 32 * c + l];
    int nib = hi ? (q >> 4) : (q & 0xF);
    float d1 = __fmul_rn(d, (float)sc[hi]);
    float m1 = __fmul_rn(dmin, (float)m[hi]);
    float v = __fsub_rn(__fmul_rn(d1, (float)nib), m1);
    out[((int64_t)n * kb + b) * FQ_Q4_K_GROUP + e] = __float2half_rn(v);
}

// Q5_K: 176 B per 256-block (d @0, dmin @2, scales[12] @4, qh[32] @16,
// qs[128] @48). Identical to Q4_K plus the 5th bit, and the bit plane is the
// trap: the numpy loop shifts the masks per 64-group (u1 = 1<<(2*c),
// u2 = 2<<(2*c)) while the qh byte index restarts at 0 in every group, so the
// mask is 1 << (2*c + hi) == 1 << s and the qh byte is l, NOT l + 32*c.
//     y = (d*sc) * (nib + 16*(qh[l] >> s & 1)) - (dmin*m)
__global__ void fq_dq_q5_k_kernel(const uint8_t* __restrict__ packed, __half* __restrict__ out,
                                   int64_t kb)
{
    int b = blockIdx.x;
    int n = blockIdx.y;
    int e = threadIdx.x;
    const uint8_t* blk = packed + ((int64_t)n * kb + b) * FQ_Q5_K_BYTES;
    __half dh, dmh;
    memcpy(&dh, blk, 2);
    memcpy(&dmh, blk + 2, 2);
    float d = __half2float(dh);
    float dmin = __half2float(dmh);
    int c = e >> 6;
    int s = e >> 5;
    int l = e & 31;
    int hi = s & 1;
    uint8_t sc[2], m[2];
    fq_q4k_scales(blk + 4, s >> 1, sc, m);
    uint8_t q = blk[48 + 32 * c + l];
    int nib = hi ? (q >> 4) : (q & 0xF);
    int bit = (blk[16 + l] >> s) & 1;      // qh index is l only; mask is 1 << s
    int q5 = nib + (bit << 4);
    float d1 = __fmul_rn(d, (float)sc[hi]);
    float m1 = __fmul_rn(dmin, (float)m[hi]);
    float v = __fsub_rn(__fmul_rn(d1, (float)q5), m1);
    out[((int64_t)n * kb + b) * FQ_Q5_K_GROUP + e] = __float2half_rn(v);
}

// Q6_K: 210 B per 256-block (ql[128] @0, qh[64] @128, 16 int8 scales @192,
// d half @208). Two 128-element halves; inside a half the 32 lanes of loop
// iteration l are laid out as 4 interleaved groups of 32 outputs, group
// (q1..q4) writing y[p], y[p+32], y[p+64], y[p+96] and taking the scales
// sc[is_+0], sc[is_+2], sc[is_+4], sc[is_+6] with is_ = l/16. So the output
// index alone gives group = (e & 127) >> 5, and the scale index is
// 8*half + 2*group + (l >> 4) -- the +2*group term is the second trap. The
// numpy ql/qh/sc slices advance per half by 64/32/8 RELATIVE to their own
// planes, so the global byte offsets are 64*half, 128 + 32*half and 192 +
// 8*half -- dropping the 128 would silently read the wrong qh plane.
//     y = (d * sc) * q          (left-assoc: d*sc first, then *q, q already -32)
__global__ void fq_dq_q6_k_kernel(const uint8_t* __restrict__ packed, __half* __restrict__ out,
                                   int64_t kb)
{
    int b = blockIdx.x;
    int n = blockIdx.y;
    int e = threadIdx.x;
    const uint8_t* blk = packed + ((int64_t)n * kb + b) * FQ_Q6_K_BYTES;
    int half = e >> 7;            // 128-element half 0..1
    int t = e & 127;              // index inside the half
    int grp = t >> 5;             // 0..3 -> q1..q4, output offset 32*grp
    int l = t & 31;               // numpy loop index l
    __half dh;
    memcpy(&dh, blk + 208, 2);
    float d = __half2float(dh);
    uint8_t ql = blk[64 * half + l + 32 * (grp & 1)];
    uint8_t qh = blk[128 + 32 * half + l];   // qh plane starts at byte 128
    // q1/q2 take the LOW nibble (ql[l], ql[l+32]), q3/q4 the HIGH nibble
    // (ql[l], ql[l+32]) -- so the nibble follows grp >= 2, not grp & 1, while
    // the ql byte follows grp & 1.
    int q = (grp >> 1) ? (ql >> 4) : (ql & 0xF);
    q = q | (((qh >> (2 * grp)) & 3) << 4);
    q -= 32;
    float s = (float)(int8_t)blk[192 + 8 * half + 2 * grp + (l >> 4)];
    float v = __fmul_rn(__fmul_rn(d, s), (float)q);
    out[((int64_t)n * kb + b) * FQ_Q6_K_GROUP + e] = __float2half_rn(v);
}

// IQ4_XS: 136 B per 256-block (d half @0, scales_h u16 @2, scales_l[4] @4,
// qs[128] @8). 8 sub-blocks of 32, each with its own 6-bit-ish scale
// ls = (scales_l[ib/2] >> 4*(ib%2) & 0xF) | ((scales_h >> 2*ib & 3) << 4)
// and dl = d * (ls - 32); the first 16 lanes take the low nibble of 16 qs
// bytes, the next 16 the high nibble of the same bytes.
//     y = (d * (ls-32)) * kvalues[nib]
__global__ void fq_dq_iq4_xs_kernel(const uint8_t* __restrict__ packed, __half* __restrict__ out,
                                     int64_t kb)
{
    int b = blockIdx.x;
    int n = blockIdx.y;
    int e = threadIdx.x;
    const uint8_t* blk = packed + ((int64_t)n * kb + b) * FQ_IQ4_XS_BYTES;
    __half dh;
    memcpy(&dh, blk, 2);
    float d = __half2float(dh);
    uint16_t sh;
    memcpy(&sh, blk + 2, 2);
    int ib = e >> 5;              // sub-block 0..7
    int j = e & 31;               // lane inside the sub-block
    uint8_t sl = blk[4 + (ib >> 1)];
    int ls = (int)((sl >> (4 * (ib & 1))) & 0xF) | (int)(((uint32_t)sh >> (2 * ib)) & 3u) << 4;
    uint8_t q = blk[8 + 16 * ib + (j & 15)];
    int nib = (j & 16) ? (q >> 4) : (q & 0xF);
    float dl = __fmul_rn(d, (float)(ls - 32));
    float v = __fmul_rn(dl, (float)FQ_KVALUES_IQ4NL[nib]);
    out[((int64_t)n * kb + b) * FQ_IQ4_XS_GROUP + e] = __float2half_rn(v);
}

// Shared validation + geometry for the four launchers. packed is [N, Kb*bb]
// (a [N, Kb, bb] view is accepted too, like the fq_dequant_q8_0 caller shape);
// out is [N, Kb*256]. All four formats have 256 values per block.
static bool fq_dq_kq_check(torch::Tensor& packed, torch::Tensor& out, int64_t block_bytes,
                           int64_t& n, int64_t& kb)
{
    TORCH_CHECK_DTYPE(packed, kByte);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(packed.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(packed.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(packed.dim() == 2 || packed.dim() == 3, "packed must be [N, Kb*bb] or [N, Kb, bb]");
    TORCH_CHECK_DIM(out, 2);
    n = packed.size(0);
    if (packed.dim() == 3)
    {
        TORCH_CHECK_SIZE(packed, 2, block_bytes);
        kb = packed.size(1);
    }
    else
    {
        TORCH_CHECK(packed.size(1) % block_bytes == 0, "packed row bytes must be a multiple of the block size");
        kb = packed.size(1) / block_bytes;
    }
    TORCH_CHECK(out.size(0) == n && out.size(1) == kb * FQ_Q4_K_GROUP, "out must be [N, Kb*256]");
    return n > 0 && kb > 0;   // empty input: nothing to launch
}

}  // namespace

// gridDim.y caps at 65535: the lm_head dequant is N = 248144 rows, so every
// launcher below chunks the rows (pointer offsets, kernels unchanged).
static const int64_t FQ_DQ_KQ_MAX_ROWS = 65535;

void fq_dequant_q4_k(torch::Tensor packed, torch::Tensor out)
{
    int64_t n, kb;
    if (!fq_dq_kq_check(packed, out, FQ_Q4_K_BYTES, n, kb)) return;
    at::cuda::CUDAGuard guard(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const uint8_t* p = packed.data_ptr<uint8_t>();
    __half* o = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    for (int64_t n0 = 0; n0 < n; n0 += FQ_DQ_KQ_MAX_ROWS)
    {
        int64_t nc = (n - n0 < FQ_DQ_KQ_MAX_ROWS) ? n - n0 : FQ_DQ_KQ_MAX_ROWS;
        dim3 grid((unsigned)kb, (unsigned)nc);
        fq_dq_q4_k_kernel<<<grid, 256, 0, stream>>>(
            p + n0 * kb * FQ_Q4_K_BYTES, o + n0 * kb * FQ_Q4_K_GROUP, kb);
    }
}

void fq_dequant_q5_k(torch::Tensor packed, torch::Tensor out)
{
    int64_t n, kb;
    if (!fq_dq_kq_check(packed, out, FQ_Q5_K_BYTES, n, kb)) return;
    at::cuda::CUDAGuard guard(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const uint8_t* p = packed.data_ptr<uint8_t>();
    __half* o = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    for (int64_t n0 = 0; n0 < n; n0 += FQ_DQ_KQ_MAX_ROWS)
    {
        int64_t nc = (n - n0 < FQ_DQ_KQ_MAX_ROWS) ? n - n0 : FQ_DQ_KQ_MAX_ROWS;
        dim3 grid((unsigned)kb, (unsigned)nc);
        fq_dq_q5_k_kernel<<<grid, 256, 0, stream>>>(
            p + n0 * kb * FQ_Q5_K_BYTES, o + n0 * kb * FQ_Q5_K_GROUP, kb);
    }
}

void fq_dequant_q6_k(torch::Tensor packed, torch::Tensor out)
{
    int64_t n, kb;
    if (!fq_dq_kq_check(packed, out, FQ_Q6_K_BYTES, n, kb)) return;
    at::cuda::CUDAGuard guard(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const uint8_t* p = packed.data_ptr<uint8_t>();
    __half* o = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    for (int64_t n0 = 0; n0 < n; n0 += FQ_DQ_KQ_MAX_ROWS)
    {
        int64_t nc = (n - n0 < FQ_DQ_KQ_MAX_ROWS) ? n - n0 : FQ_DQ_KQ_MAX_ROWS;
        dim3 grid((unsigned)kb, (unsigned)nc);
        fq_dq_q6_k_kernel<<<grid, 256, 0, stream>>>(
            p + n0 * kb * FQ_Q6_K_BYTES, o + n0 * kb * FQ_Q6_K_GROUP, kb);
    }
}

void fq_dequant_iq4_xs(torch::Tensor packed, torch::Tensor out)
{
    int64_t n, kb;
    if (!fq_dq_kq_check(packed, out, FQ_IQ4_XS_BYTES, n, kb)) return;
    at::cuda::CUDAGuard guard(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const uint8_t* p = packed.data_ptr<uint8_t>();
    __half* o = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    for (int64_t n0 = 0; n0 < n; n0 += FQ_DQ_KQ_MAX_ROWS)
    {
        int64_t nc = (n - n0 < FQ_DQ_KQ_MAX_ROWS) ? n - n0 : FQ_DQ_KQ_MAX_ROWS;
        dim3 grid((unsigned)kb, (unsigned)nc);
        fq_dq_iq4_xs_kernel<<<grid, 256, 0, stream>>>(
            p + n0 * kb * FQ_IQ4_XS_BYTES, o + n0 * kb * FQ_IQ4_XS_GROUP, kb);
    }
}