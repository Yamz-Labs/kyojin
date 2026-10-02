#if defined(USE_ROCM)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <cstdint>
#include <cstdlib>

#include "fused_elt.cuh"
#include "util.h"
#include "util.cuh"

// Arithmetic mirrors the ATen kernels the fallback chains launch (no fast-math intrinsics):
//   sigmoid(x) = 1 / (1 + expf(-x)), silu(x) = x / (1 + expf(-x)), mean = sum * (1 / N)

__device__ __forceinline__ float fe_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float fe_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float fe_bf16_ld(const uint16_t* p, size_t i) { return __uint_as_float(((uint32_t) p[i]) << 16); }

// float -> bf16, round to nearest even (c10::BFloat16 semantics)
__device__ __forceinline__ uint16_t fe_bf16_rn(float f)
{
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// ------------------------------------------------------------------------------------------------
// Gated RMSNorm: one wave32 per row, GN_MAX values per lane kept in registers

#define GN_THREADS 256
#define GN_MAX 16

template <int G_T, bool W_F32, bool Y_F32>
__global__ __launch_bounds__(GN_THREADS)
void fused_gated_rms_norm_kernel
(
    const uint16_t* __restrict__ x,
    const void* __restrict__ w,
    void* __restrict__ y,
    const void* __restrict__ g,
    const int rows,
    const int dim,
    const float eps,
    const float constant_bias,
    const int w_groups,
    const bool gate_first,
    const int gate_act
)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (GN_THREADS / 32) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const size_t off = (size_t) row * dim;

    float ni[GN_MAX];
    float gt[GN_MAX];
    // ATen mean over the last dim uses the vectorized input path: lane t owns vec4 chunks
    // t, t + 32, ... with 4 per-component accumulators, combined ((a0 + a1) + a2) + a3
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int j = 0; j < GN_MAX; ++j)
    {
        const int i = 4 * (lane + (j >> 2) * 32) + (j & 3);
        ni[j] = 0.0f;
        gt[j] = 0.0f;
        if (i < dim)
        {
            const float xf = fe_bf16_ld(x, off + i);
            // G_T: 0 = bf16, 1 = fp32, 2 = fp16 (KDA z from EXL3_KDA_F16_GATES; exact widen, as the fallback's .float())
            const float gf = G_T == 1 ? ((const float*) g)[off + i]
                           : G_T == 2 ? __half2float(((const half*) g)[off + i])
                           : fe_bf16_ld((const uint16_t*) g, off + i);
            const float gate = gate_act == 1 ? fe_sigmoid(gf) : fe_silu(gf);
            gt[j] = gate;
            const float v = gate_first ? xf * gate : xf;
            ni[j] = v;
            acc[j & 3] = __fadd_rn(acc[j & 3], __fmul_rn(v, v));
        }
    }
    float ss = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
    #pragma unroll
    // ATen block_x_reduce: shfl_down tree, lane 0 holds the result; broadcast it so every
    // lane uses the same association (xor butterfly sums differ per lane)
    for (int m = 1; m < 32; m <<= 1) ss = __fadd_rn(ss, __shfl_down(ss, m));
    ss = __shfl(ss, 0);
    // separate mul / add (ATen MeanOps project, then add kernel): no fma contraction
    // torch.rsqrt(float) is correctly rounded here; the rsqrtf builtin is not (1 ulp off on ~10% of rows)
    const float r = (float) (1.0 / sqrt((double) __fadd_rn(__fmul_rn(ss, 1.0f / (float) dim), eps)));

    const int grp = row % w_groups;
    #pragma unroll
    for (int j = 0; j < GN_MAX; ++j)
    {
        const int i = 4 * (lane + (j >> 2) * 32) + (j & 3);
        if (i < dim)
        {
            float wf = W_F32 ? ((const float*) w)[(size_t) grp * dim + i]
                             : fe_bf16_ld((const uint16_t*) w, (size_t) grp * dim + i);
            if (constant_bias != 0.0f) wf = wf + constant_bias;
            float h = ni[j] * r;
            h = h * wf;
            if (!gate_first) h = h * gt[j];
            if (Y_F32) ((float*) y)[off + i] = h;
            else ((half*) y)[off + i] = __float2half_rn(h);
        }
    }
}

void fused_gated_rms_norm
(
    const at::Tensor& x,
    const at::Tensor& w,
    at::Tensor& y,
    const at::Tensor& g,
    double eps,
    double constant_bias,
    int64_t w_groups,
    bool gate_first,
    int64_t gate_act
)
{
    const at::Device device = x.device();
    c10::cuda::OptionalCUDAGuard device_guard(device);
    hipStream_t stream = c10::hip::getCurrentHIPStream(device.index()).stream();

    TORCH_CHECK_DTYPE(x, kBFloat16);
    TORCH_CHECK(y.dtype() == at::kHalf || y.dtype() == at::kFloat, "fused_gated_rms_norm: y must be f16/f32");
    TORCH_CHECK(w.dtype() == at::kBFloat16 || w.dtype() == at::kFloat, "fused_gated_rms_norm: w must be bf16/f32");
    TORCH_CHECK(g.dtype() == at::kBFloat16 || g.dtype() == at::kFloat || g.dtype() == at::kHalf,
                "fused_gated_rms_norm: g must be bf16/f32/f16");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && g.is_contiguous() && w.is_contiguous(),
                "fused_gated_rms_norm: tensors must be contiguous");
    const int dim = x.size(-1);
    const int rows = x.numel() / dim;
    TORCH_CHECK(dim <= 32 * GN_MAX && dim % 128 == 0, "fused_gated_rms_norm: dim must be <= 512 and a multiple of 128");
    TORCH_CHECK(y.numel() == x.numel() && g.numel() == x.numel() && w_groups >= 1 && w.numel() == w_groups * dim,
                "fused_gated_rms_norm: shape mismatch");
    TORCH_CHECK(gate_act == 0 || gate_act == 1, "fused_gated_rms_norm: gate_act must be 0 or 1");
    if (rows == 0) return;

    const int gt = g.dtype() == at::kFloat ? 1 : g.dtype() == at::kHalf ? 2 : 0;
    const bool wf = w.dtype() == at::kFloat, yf = y.dtype() == at::kFloat;
    const int blocks = (rows + GN_THREADS / 32 - 1) / (GN_THREADS / 32);
    #define GN_LAUNCH(GT, WF, YF) fused_gated_rms_norm_kernel<GT, WF, YF><<<blocks, GN_THREADS, 0, stream>>>( \
        (const uint16_t*) x.data_ptr(), w.data_ptr(), y.data_ptr(), g.data_ptr(), rows, dim, \
        (float) eps, (float) constant_bias, (int) w_groups, gate_first, (int) gate_act)
    #define GN_W(GT) \
        if      ( wf &&  yf) GN_LAUNCH(GT, true,  true);  \
        else if ( wf && !yf) GN_LAUNCH(GT, true,  false); \
        else if (!wf &&  yf) GN_LAUNCH(GT, false, true);  \
        else                 GN_LAUNCH(GT, false, false);
    if      (gt == 1) { GN_W(1) }
    else if (gt == 2) { GN_W(2) }
    else              { GN_W(0) }
    #undef GN_W
    #undef GN_LAUNCH
    cuda_check(hipPeekAtLastError());
}

// ------------------------------------------------------------------------------------------------
// silu(x) * y -> z (half). Half inputs: the fallback materializes silu(x) as half first.

#define SM_THREADS 256

// act_limit != 0 (clamped SwiGLU, GLM swiglu_limit): the fallback runs silu -> clamp_max(L) ->
// y.clamp(-L, L) -> mul [-> clamp +-65504 for f32] -> copy; clamps keep NaN like ATen's.
// y and z may alias (interm half: a is u), so no __restrict__ on them; each element is read
// before its own store.
__device__ __forceinline__ float fe_clamp_max(float v, float hi) { return v > hi ? hi : v; }
__device__ __forceinline__ float fe_clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

template <bool F32, bool LIM>
__global__ __launch_bounds__(SM_THREADS)
void fused_silu_mul_kernel(const void* __restrict__ x, const void* y, half* z, const size_t n, const float lim)
{
    for (size_t i = (size_t) blockIdx.x * SM_THREADS + threadIdx.x; i < n; i += (size_t) gridDim.x * SM_THREADS)
    {
        if (F32)
        {
            float s = fe_silu(((const float*) x)[i]);
            float yv = ((const float*) y)[i];
            if (LIM) { s = fe_clamp_max(s, lim); yv = fe_clamp(yv, -lim, lim); }
            float r = s * yv;
            r = r < -65504.0f ? -65504.0f : (r > 65504.0f ? 65504.0f : r);
            z[i] = __float2half_rn(r);
        }
        else
        {
            // every intermediate is a half tensor in the fallback: round after each op
            float s = __half2float(__float2half_rn(fe_silu(__half2float(((const half*) x)[i]))));
            float yv = __half2float(((const half*) y)[i]);
            if (LIM)
            {
                s = __half2float(__float2half_rn(fe_clamp_max(s, lim)));
                yv = __half2float(__float2half_rn(fe_clamp(yv, -lim, lim)));
            }
            z[i] = __float2half_rn(s * yv);
        }
    }
}

void fused_silu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    double act_limit
)
{
    const at::Device device = x.device();
    c10::cuda::OptionalCUDAGuard device_guard(device);
    hipStream_t stream = c10::hip::getCurrentHIPStream(device.index()).stream();

    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "fused_silu_mul: x must be f16/f32");
    TORCH_CHECK(y.dtype() == x.dtype(), "fused_silu_mul: x/y dtype mismatch");
    TORCH_CHECK_DTYPE(z, kHalf);
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && z.is_contiguous(), "fused_silu_mul: contiguous only");
    TORCH_CHECK(y.numel() == x.numel() && z.numel() == x.numel(), "fused_silu_mul: shape mismatch");
    const size_t n = x.numel();
    if (n == 0) return;
    const int blocks = (int) std::min<size_t>((n + SM_THREADS - 1) / SM_THREADS, 4096);
    const float lim = (float) act_limit;
    const bool f32 = x.dtype() == at::kFloat;
    #define SM_LAUNCH(F, L) fused_silu_mul_kernel<F, L><<<blocks, SM_THREADS, 0, stream>>>(x.data_ptr(), y.data_ptr(), (half*) z.data_ptr(), n, lim)
    if (act_limit != 0.0) { if (f32) SM_LAUNCH(true, true);  else SM_LAUNCH(false, true); }
    else                  { if (f32) SM_LAUNCH(true, false); else SM_LAUNCH(false, false); }
    #undef SM_LAUNCH
    cuda_check(hipPeekAtLastError());
}

// ------------------------------------------------------------------------------------------------
// KDA gates, one thread per g element; the first element of each head also writes beta

#define KG_THREADS 256

__global__ __launch_bounds__(KG_THREADS)
void kda_gate_kernel
(
    const float* __restrict__ b,
    const float* __restrict__ f,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    uint16_t* __restrict__ beta,
    float* __restrict__ g,
    const int nv,
    const int dk,
    const size_t n,
    const float beta_scale,
    const float lb,
    const bool has_lb
)
{
    const size_t i = (size_t) blockIdx.x * KG_THREADS + threadIdx.x;
    if (i >= n) return;
    const int hd = nv * dk;
    const int c = (int) (i % hd);
    const int h = c / dk;
    const float gf = f[i] + dt_bias[c];
    const float decay = expf(a_log[h]);
    if (has_lb)
    {
        const float s = fe_sigmoid(decay * gf);
        g[i] = s * lb;
    }
    else
    {
        const float sp = gf > 20.0f ? gf : log1pf(expf(gf));
        g[i] = (-decay) * sp;
    }
    if (c % dk == 0)
    {
        const size_t bi = (i / hd) * nv + h;
        const float bs = b[bi] * beta_scale;
        beta[bi] = fe_bf16_rn(fe_sigmoid(bs));
    }
}

void kda_gate
(
    const at::Tensor& b,
    const at::Tensor& f,
    const at::Tensor& dt_bias,
    const at::Tensor& a_log,
    double beta_scale,
    double lower_bound,
    bool has_lb,
    at::Tensor& beta,
    at::Tensor& g
)
{
    const at::Device device = f.device();
    c10::cuda::OptionalCUDAGuard device_guard(device);
    hipStream_t stream = c10::hip::getCurrentHIPStream(device.index()).stream();

    TORCH_CHECK_DTYPE(b, kFloat);
    TORCH_CHECK_DTYPE(f, kFloat);
    TORCH_CHECK_DTYPE(dt_bias, kFloat);
    TORCH_CHECK_DTYPE(a_log, kFloat);
    TORCH_CHECK_DTYPE(beta, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);
    TORCH_CHECK(b.is_contiguous() && f.is_contiguous() && dt_bias.is_contiguous() && a_log.is_contiguous() &&
                beta.is_contiguous() && g.is_contiguous(), "kda_gate: contiguous only");
    const int nv = a_log.numel();
    const int hd = dt_bias.numel();
    TORCH_CHECK(nv > 0 && hd % nv == 0, "kda_gate: dt_bias / a_log shape");
    const int dk = hd / nv;
    const size_t n = f.numel();
    TORCH_CHECK(n % hd == 0 && g.numel() == (int64_t) n, "kda_gate: f / g shape");
    const size_t T = n / hd;
    TORCH_CHECK(b.numel() == (int64_t) (T * nv) && beta.numel() == b.numel(), "kda_gate: b / beta shape");
    if (n == 0) return;
    const int blocks = (int) ((n + KG_THREADS - 1) / KG_THREADS);
    kda_gate_kernel<<<blocks, KG_THREADS, 0, stream>>>(
        (const float*) b.data_ptr(), (const float*) f.data_ptr(), (const float*) dt_bias.data_ptr(),
        (const float*) a_log.data_ptr(), (uint16_t*) beta.data_ptr(), (float*) g.data_ptr(),
        nv, dk, n, (float) beta_scale, (float) lower_bound, has_lb);
    cuda_check(hipPeekAtLastError());
}

// ------------------------------------------------------------------------------------------------
// kda-dec step 2: KDA decode low-rank second stages fused into their consumers (rows <= 8).
// hipBLAS HSS for these shapes (K = 128) sums k = 0..127 in order in fp32 (fp16 x fp16 products are
// exact in fp32, so fmaf == mul + add); scratch/kd2/order_probe.py proved bit-equality at rows 1/2/4/8.
// The kernels below use that exact order, then the unchanged gate / norm arithmetic: bit-exact.

#define LR_K 128
#define KDA_LR_SPLIT_DEFAULT 3
// Native 16-byte vector: a HIP uint4 array (struct with union) is left in scratch; this one stays in VGPRs.
typedef unsigned int lr_u32x4 __attribute__((ext_vector_type(4)));
// EXL3_KDA_LR_SPLIT bitmask (read per call, so an A/B can flip it between graph captures): 1 = kda_gb_norm, 2 =
// kda_fb_gate stage W in two k-halves. Both arms are bit-identical.
static inline bool kda_lr_split(int bit)
{
    const char* e = getenv("EXL3_KDA_LR_SPLIT");
    return ((e ? atoi(e) : KDA_LR_SPLIT_DEFAULT) & bit) != 0;
}
#define LR_MAX_ROWS 8
#define FG_COLS 64
#define FG_THREADS 256

// f = fa @ w_fb, then kda_gate_kernel's math on f (g, and beta from b). One block per 64 columns.
template <int NH>
__global__ __launch_bounds__(FG_THREADS)
void kda_fb_gate_kernel
(
    const half* __restrict__ fa,       // [rows, LR_K]
    const half* __restrict__ wfb,      // [LR_K, hd]
    const float* __restrict__ b,       // [rows, nv]
    const float* __restrict__ dt_bias, // [hd]
    const float* __restrict__ a_log,   // [nv]
    uint16_t* __restrict__ beta,       // [rows, nv] bf16
    float* __restrict__ g,             // [rows, hd]
    const int rows,
    const int nv,
    const int dk,
    const float beta_scale,
    const float lb,
    const bool has_lb
)
{
    // NH = 1: all of W in LDS (20 KB, 3 blocks/CU -> 120 slots < 128 blocks). NH = 2: W in two k-halves, rows
    // 0..63 to LDS and 64..127 held in VGPRs until the first half is summed (12 KB). The fp32 partial sum is carried
    // across halves, so the per-column fmaf order stays k = 0..127 ascending either way: bit-identical.
    constexpr int KH = LR_K / NH;
    __shared__ __align__(16) half ws[KH][FG_COLS];
    __shared__ float as[LR_MAX_ROWS][LR_K];
    const int hd = nv * dk;
    const int c0 = blockIdx.x * FG_COLS;
    // Dependent loads issued before the tile barrier so they overlap the weight stream (math unchanged).
    const int cc = threadIdx.x % FG_COLS;
    const int c = c0 + cc;
    const int h = c / dk;
    const float dtb = dt_bias[c];
    const float decay = expf(a_log[h]);
    const int r0 = threadIdx.x / FG_COLS;
    const float b0 = (c % dk == 0 && r0 < rows) ? b[(size_t) r0 * nv + h] : 0.0f;
    constexpr int NVT = KH * FG_COLS / 8 / FG_THREADS;
    lr_u32x4 hi[NVT];
    #pragma unroll
    for (int j = 0; j < NVT; ++j)
    {
        const int v = threadIdx.x + j * FG_THREADS;
        const int kk = v / (FG_COLS / 8);
        const int c8 = (v % (FG_COLS / 8)) * 8;
        *((lr_u32x4*) &ws[kk][c8]) = *((const lr_u32x4*) (wfb + (size_t) kk * hd + c0 + c8));
        if (NH > 1) hi[j] = *((const lr_u32x4*) (wfb + (size_t) (kk + KH) * hd + c0 + c8));
    }
    for (int v = threadIdx.x; v < rows * LR_K; v += FG_THREADS) as[v / LR_K][v % LR_K] = __half2float(fa[v]);
    __syncthreads();

    constexpr int RPT = LR_MAX_ROWS / (FG_THREADS / FG_COLS);
    float accs[RPT];
    #pragma unroll
    for (int j = 0; j < RPT; ++j)
    {
        const int r = r0 + j * (FG_THREADS / FG_COLS);
        float acc = 0.0f;
        if (r < rows)
        {
            #pragma unroll 16
            for (int kk = 0; kk < KH; ++kk) acc = fmaf(as[r][kk], __half2float(ws[kk][cc]), acc);
        }
        accs[j] = acc;
    }
    if (NH > 1)
    {
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < NVT; ++j)
        {
            const int v = threadIdx.x + j * FG_THREADS;
            *((lr_u32x4*) &ws[v / (FG_COLS / 8)][(v % (FG_COLS / 8)) * 8]) = hi[j];
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < RPT; ++j)
        {
            const int r = r0 + j * (FG_THREADS / FG_COLS);
            if (r < rows)
            {
                #pragma unroll 16
                for (int kk = 0; kk < KH; ++kk) accs[j] = fmaf(as[r][KH + kk], __half2float(ws[kk][cc]), accs[j]);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < RPT; ++j)
    {
        const int r = r0 + j * (FG_THREADS / FG_COLS);
        if (r >= rows) break;
        const float acc = accs[j];
        const size_t i = (size_t) r * hd + c;
        const float gf = acc + dtb;
        if (has_lb)
        {
            const float s = fe_sigmoid(decay * gf);
            g[i] = s * lb;
        }
        else
        {
            const float sp = gf > 20.0f ? gf : log1pf(expf(gf));
            g[i] = (-decay) * sp;
        }
        if (c % dk == 0)
        {
            const size_t bi = (size_t) r * nv + h;
            const float bs = (r == r0 ? b0 : b[bi]) * beta_scale;
            beta[bi] = fe_bf16_rn(fe_sigmoid(bs));
        }
    }
}

void kda_fb_gate
(
    const at::Tensor& fa,
    const at::Tensor& wfb,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& a_log,
    double beta_scale,
    double lower_bound,
    bool has_lb,
    at::Tensor& beta,
    at::Tensor& g
)
{
    const at::Device device = fa.device();
    c10::cuda::OptionalCUDAGuard device_guard(device);
    hipStream_t stream = c10::hip::getCurrentHIPStream(device.index()).stream();
    TORCH_CHECK_DTYPE(fa, kHalf);
    TORCH_CHECK_DTYPE(wfb, kHalf);
    TORCH_CHECK_DTYPE(b, kFloat);
    TORCH_CHECK_DTYPE(dt_bias, kFloat);
    TORCH_CHECK_DTYPE(a_log, kFloat);
    TORCH_CHECK_DTYPE(beta, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);
    TORCH_CHECK(fa.is_contiguous() && wfb.is_contiguous() && b.is_contiguous() && dt_bias.is_contiguous() &&
                a_log.is_contiguous() && beta.is_contiguous() && g.is_contiguous(), "kda_fb_gate: contiguous only");
    const int nv = a_log.numel();
    const int hd = dt_bias.numel();
    TORCH_CHECK(nv > 0 && hd % nv == 0 && hd % FG_COLS == 0, "kda_fb_gate: dt_bias / a_log shape");
    const int dk = hd / nv;
    TORCH_CHECK(fa.size(-1) == LR_K && wfb.dim() == 2 && wfb.size(0) == LR_K && wfb.size(1) == hd,
                "kda_fb_gate: fa [rows, 128], wfb [128, hd]");
    TORCH_CHECK(((uintptr_t) wfb.data_ptr()) % 16 == 0, "kda_fb_gate: wfb must be 16-byte aligned");
    const int rows = fa.numel() / LR_K;
    TORCH_CHECK(rows >= 1 && rows <= LR_MAX_ROWS, "kda_fb_gate: 1..8 rows");
    TORCH_CHECK(g.numel() == (int64_t) rows * hd && b.numel() == (int64_t) rows * nv && beta.numel() == b.numel(),
                "kda_fb_gate: b / beta / g shape");
    #define FG_LAUNCH(NH) kda_fb_gate_kernel<NH><<<hd / FG_COLS, FG_THREADS, 0, stream>>>( \
        (const half*) fa.data_ptr(), (const half*) wfb.data_ptr(), (const float*) b.data_ptr(), \
        (const float*) dt_bias.data_ptr(), (const float*) a_log.data_ptr(), (uint16_t*) beta.data_ptr(), \
        (float*) g.data_ptr(), rows, nv, dk, (float) beta_scale, (float) lower_bound, has_lb)
    if (kda_lr_split(2)) FG_LAUNCH(2);
    else                 FG_LAUNCH(1);
    #undef FG_LAUNCH
    cuda_check(hipPeekAtLastError());
}

// z = ga @ w_gb for one head (128 columns), then fused_gated_rms_norm_kernel's per-row math with G_F32
// gate read from LDS. One block per head, one wave per token row. dim (v_head_dim) must be 128.
#define GB_DIM 128

template <bool W_F32, bool Y_F32, int NH>
__global__ __launch_bounds__(GN_THREADS)
void kda_gb_norm_kernel
(
    const uint16_t* __restrict__ x,    // [rows * nv, GB_DIM] bf16
    const half* __restrict__ ga,       // [rows, LR_K]
    const half* __restrict__ wgb,      // [LR_K, nv * GB_DIM]
    const void* __restrict__ w,
    void* __restrict__ y,
    const int rows,
    const int nv,
    const float eps,
    const float constant_bias,
    const int w_groups,
    const bool gate_first,
    const int gate_act
)
{
    constexpr int KH = LR_K / NH;
    __shared__ __align__(16) half ws[KH][GB_DIM];
    __shared__ float as[LR_MAX_ROWS][LR_K];
    __shared__ float zs[LR_MAX_ROWS][GB_DIM];
    const int hh = blockIdx.x;
    const int hd = nv * GB_DIM;
    // Norm-phase operands (x slice, norm weight) loaded before the tile barrier; math unchanged.
    const int lane = threadIdx.x & 31;
    const int tr = threadIdx.x >> 5;
    const int row = tr * nv + hh;
    const int dim = GB_DIM;
    const size_t off = (size_t) row * dim;
    const int grp = row % w_groups;
    float xv[GN_MAX];
    float wv[GN_MAX];
    #pragma unroll
    for (int j = 0; j < GN_MAX; ++j)
    {
        const int i = 4 * (lane + (j >> 2) * 32) + (j & 3);
        xv[j] = 0.0f;
        wv[j] = 0.0f;
        if (tr < rows && i < dim)
        {
            xv[j] = fe_bf16_ld(x, off + i);
            wv[j] = W_F32 ? ((const float*) w)[(size_t) grp * dim + i]
                          : fe_bf16_ld((const uint16_t*) w, (size_t) grp * dim + i);
        }
    }
    // NH = 1: all of W in LDS (40 KB -> 3 blocks/WGP, 60 slots < nv = 64 blocks: a second round). NH = 2: W in two
    // k-halves, rows 0..63 to LDS and 64..127 in VGPRs until the first half is summed (16 + 8 KB, whole grid
    // co-resident). The fp32 partial sum is carried in zs (exact); per-column fmaf order stays k = 0..127 ascending.
    constexpr int NVT = KH * GB_DIM / 8 / GN_THREADS;
    lr_u32x4 hi[NVT];
    #pragma unroll
    for (int j = 0; j < NVT; ++j)
    {
        const int v = threadIdx.x + j * GN_THREADS;
        const int kk = v / (GB_DIM / 8);
        const int cc = (v % (GB_DIM / 8)) * 8;
        *((lr_u32x4*) &ws[kk][cc]) = *((const lr_u32x4*) (wgb + (size_t) kk * hd + hh * GB_DIM + cc));
        if (NH > 1) hi[j] = *((const lr_u32x4*) (wgb + (size_t) (kk + KH) * hd + hh * GB_DIM + cc));
    }
    for (int v = threadIdx.x; v < rows * LR_K; v += GN_THREADS) as[v / LR_K][v % LR_K] = __half2float(ga[v]);
    __syncthreads();
    for (int v = threadIdx.x; v < rows * GB_DIM; v += GN_THREADS)
    {
        const int r = v / GB_DIM;
        const int cc = v % GB_DIM;
        float acc = 0.0f;
        #pragma unroll 16
        for (int kk = 0; kk < KH; ++kk) acc = fmaf(as[r][kk], __half2float(ws[kk][cc]), acc);
        zs[r][cc] = acc;
    }
    __syncthreads();
    if (NH > 1)
    {
        #pragma unroll
        for (int j = 0; j < NVT; ++j)
        {
            const int v = threadIdx.x + j * GN_THREADS;
            *((lr_u32x4*) &ws[v / (GB_DIM / 8)][(v % (GB_DIM / 8)) * 8]) = hi[j];
        }
        __syncthreads();
        for (int v = threadIdx.x; v < rows * GB_DIM; v += GN_THREADS)
        {
            const int r = v / GB_DIM;
            const int cc = v % GB_DIM;
            float acc = zs[r][cc];
            #pragma unroll 16
            for (int kk = 0; kk < KH; ++kk) acc = fmaf(as[r][KH + kk], __half2float(ws[kk][cc]), acc);
            zs[r][cc] = acc;
        }
        __syncthreads();
    }

    if (tr >= rows) return;

    float ni[GN_MAX];
    float gt[GN_MAX];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int j = 0; j < GN_MAX; ++j)
    {
        const int i = 4 * (lane + (j >> 2) * 32) + (j & 3);
        ni[j] = 0.0f;
        gt[j] = 0.0f;
        if (i < dim)
        {
            const float xf = xv[j];
            const float gf = zs[tr][i];
            const float gate = gate_act == 1 ? fe_sigmoid(gf) : fe_silu(gf);
            gt[j] = gate;
            const float v = gate_first ? xf * gate : xf;
            ni[j] = v;
            acc[j & 3] = __fadd_rn(acc[j & 3], __fmul_rn(v, v));
        }
    }
    float ss = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) ss = __fadd_rn(ss, __shfl_down(ss, m));
    ss = __shfl(ss, 0);
    const float r = (float) (1.0 / sqrt((double) __fadd_rn(__fmul_rn(ss, 1.0f / (float) dim), eps)));

    #pragma unroll
    for (int j = 0; j < GN_MAX; ++j)
    {
        const int i = 4 * (lane + (j >> 2) * 32) + (j & 3);
        if (i < dim)
        {
            float wf = wv[j];
            if (constant_bias != 0.0f) wf = wf + constant_bias;
            float h = ni[j] * r;
            h = h * wf;
            if (!gate_first) h = h * gt[j];
            if (Y_F32) ((float*) y)[off + i] = h;
            else ((half*) y)[off + i] = __float2half_rn(h);
        }
    }
}

void kda_gb_norm
(
    const at::Tensor& x,
    const at::Tensor& ga,
    const at::Tensor& wgb,
    const at::Tensor& w,
    at::Tensor& y,
    double eps,
    double constant_bias,
    int64_t w_groups,
    bool gate_first,
    int64_t gate_act
)
{
    const at::Device device = x.device();
    c10::cuda::OptionalCUDAGuard device_guard(device);
    hipStream_t stream = c10::hip::getCurrentHIPStream(device.index()).stream();
    TORCH_CHECK_DTYPE(x, kBFloat16);
    TORCH_CHECK_DTYPE(ga, kHalf);
    TORCH_CHECK_DTYPE(wgb, kHalf);
    TORCH_CHECK(y.dtype() == at::kHalf || y.dtype() == at::kFloat, "kda_gb_norm: y must be f16/f32");
    TORCH_CHECK(w.dtype() == at::kBFloat16 || w.dtype() == at::kFloat, "kda_gb_norm: w must be bf16/f32");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && ga.is_contiguous() && wgb.is_contiguous() &&
                w.is_contiguous(), "kda_gb_norm: tensors must be contiguous");
    TORCH_CHECK(x.size(-1) == GB_DIM, "kda_gb_norm: dim must be 128");
    TORCH_CHECK(ga.size(-1) == LR_K && wgb.dim() == 2 && wgb.size(0) == LR_K && wgb.size(1) % GB_DIM == 0,
                "kda_gb_norm: ga [rows, 128], wgb [128, nv * 128]");
    TORCH_CHECK(((uintptr_t) wgb.data_ptr()) % 16 == 0, "kda_gb_norm: wgb must be 16-byte aligned");
    const int nv = wgb.size(1) / GB_DIM;
    const int rows = ga.numel() / LR_K;
    TORCH_CHECK(rows >= 1 && rows <= LR_MAX_ROWS && rows <= GN_THREADS / 32, "kda_gb_norm: 1..8 rows");
    TORCH_CHECK(x.numel() == (int64_t) rows * nv * GB_DIM && y.numel() == x.numel() && w_groups >= 1 &&
                w.numel() == w_groups * GB_DIM, "kda_gb_norm: shape mismatch");
    TORCH_CHECK(gate_act == 0 || gate_act == 1, "kda_gb_norm: gate_act must be 0 or 1");
    const bool wf = w.dtype() == at::kFloat, yf = y.dtype() == at::kFloat;
    const bool split = kda_lr_split(1);
    #define GB_LAUNCH_NH(WF, YF, NH) kda_gb_norm_kernel<WF, YF, NH><<<nv, GN_THREADS, 0, stream>>>( \
        (const uint16_t*) x.data_ptr(), (const half*) ga.data_ptr(), (const half*) wgb.data_ptr(), w.data_ptr(), \
        y.data_ptr(), rows, nv, (float) eps, (float) constant_bias, (int) w_groups, gate_first, (int) gate_act)
    #define GB_LAUNCH(NH) \
        if      ( wf &&  yf) GB_LAUNCH_NH(true,  true,  NH); \
        else if ( wf && !yf) GB_LAUNCH_NH(true,  false, NH); \
        else if (!wf &&  yf) GB_LAUNCH_NH(false, true,  NH); \
        else                 GB_LAUNCH_NH(false, false, NH);
    if (split) { GB_LAUNCH(2) }
    else       { GB_LAUNCH(1) }
    #undef GB_LAUNCH
    #undef GB_LAUNCH_NH
    cuda_check(hipPeekAtLastError());
}

#endif

