#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include "exl3_moe_fused_api.h"
#include "hadamard.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#if defined(USE_ROCM)
#include "exl3_moe_valu.cuh"   // mv_dot2
#include "exl3_moe_fused.cuh"  // fused MoE half-layer, opt-in EXL3_MOE_FUSED=1
#endif
#include <cstdio>
#include <cstring>
#include <algorithm>

#if defined(USE_ROCM)

// ---------------------------------------------------------------------------------------------------------------------------
// EXL3_MOE_FUSED: one persistent launch per MoE half-layer (exl3_moe_fused.cuh). Entry points: exl3_moe_fused_half (launch),
// exl3_moe_fused_supported / _ws_bytes / _ws_offsets (plan). Only shapes listed in MF_SHAPES are instantiated.
// ---------------------------------------------------------------------------------------------------------------------------
namespace {
template <class S> struct MfShapeMatch
{
    static bool eq(int D, int H, int LR, int NEXP, int TOPK, int INTER, int RB, int SB)
    { return D == S::D && H == S::H && LR == S::LR && NEXP == S::NEXP && TOPK == S::TOPK && INTER == S::INTER && RB == S::RB && SB == S::SB; }
};
// workspace layout (bytes, 256-aligned): ctl (bar, done, err u32) | dots | post | mixed | scores | sgl | gu | dn | ydbg | seldbg
template <class S> std::vector<int64_t> mf_ws_offsets()
{
    using DM = mf::Dm<S>;
    const int NARM = mf::MAXRT * S::TOPK + mf::MAXRT;   // moe8: sized for 8 rows (the <= 4-row kernels use the first part of every buffer)
    const int64_t sizes[] = {
        256,                                                   // ctl
        (int64_t) mf::MAXRT * (DM::MR + 1) * S::H * 4,         // dots f32
        (int64_t) mf::MAXRT * S::H * 4,                         // post f32
        (int64_t) mf::MAXRT * S::D * 2,                         // mixed half
        (int64_t) mf::MAXRT * S::NEXP * 2,                      // scores half
        256,                                                   // sgl f32 (64)
        (int64_t) 2 * NARM * S::INTER * 2,                     // gu half
        (int64_t) NARM * S::D * 4,                             // dn f32
        (int64_t) mf::MAXRT * S::D * 4,                         // ydbg f32
        (int64_t) mf::SELDBG_INTS * 4,                         // seldbg i32
    };
    std::vector<int64_t> off; int64_t o = 0;
    for (int64_t s : sizes) { off.push_back(o); o += (s + 255) / 256 * 256; }
    off.push_back(o);                                          // total
    return off;
}
template <class S> int mf_grid()
{
    static int G = [] {
        int nb = 0, dev = 0; hipGetDevice(&dev);
        hipDeviceProp_t prop; hipGetDeviceProperties(&prop, dev);
        cuda_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*) mf::moe_fused_kernel<S>, mf::THREADS, 0));
        return std::max(1, nb) * prop.multiProcessorCount;
    }();
    return G;
}
template <class S> void mf_launch
(
    const at::Tensor& x, const at::Tensor& fn, const at::Tensor& fn_scale, const at::Tensor& upt, const at::Tensor& up_scale, const at::Tensor& w_h,
    const at::Tensor& router, const at::Tensor& sgate, const at::Tensor& wt, const at::Tensor& svt,
    at::Tensor& ws, double rms_eps, int native, int R, hipStream_t stream
)
{
    using DM = mf::Dm<S>;
    TORCH_CHECK(fn.numel() == (int64_t) DM::MR * S::H * S::D, "fused: fn size");
    const bool gs = fn_scale.numel() != DM::MR;
    TORCH_CHECK(!gs || mf::gs_ok<S>(), "fused: group scales need D % 128 == 0");
    TORCH_CHECK(fn_scale.numel() == mf::fn_scale_n<S>(gs), "fused: fn_scale size");
    TORCH_CHECK(upt.numel() == (int64_t) S::H * DM::D4 * S::LR * 4, "fused: upt size");
    TORCH_CHECK(up_scale.numel() == mf::up_scale_n<S>(gs) && w_h.numel() == (int64_t) S::H * S::D, "fused: up_scale / w_h size");
    TORCH_CHECK(router.numel() == (int64_t) S::NEXP * S::D && sgate.numel() == S::D, "fused: router size");
    TORCH_CHECK(wt.dtype() == at::kLong && wt.numel() == 3 * (S::NEXP + 1) && svt.dtype() == at::kLong && svt.numel() == 6 * (S::NEXP + 1), "fused: pointer tables");
    TORCH_CHECK(native == 0 || native == 1, "fused: native flag");
    const auto off = mf_ws_offsets<S>();
    TORCH_CHECK(ws.numel() >= off.back(), "fused: workspace too small");
    char* wb = (char*) ws.data_ptr();
    mf::Params p{};
    p.R = R; p.G = mf_grid<S>();
    TORCH_CHECK(p.G >= 1 && p.G <= mf::CTL_MAXB, "exl3_moe_fused: grid exceeds the barrier arrival-marker words");
    p.xin = (float*) x.data_ptr(); p.xout = (float*) x.data_ptr();
    p.fn = (const int8_t*) fn.data_ptr(); p.fn_scale = (const float*) fn_scale.data_ptr(); p.upt = (const int8_t*) upt.data_ptr();
    p.gs = gs ? 1 : 0; p.up_scale = (const float*) up_scale.data_ptr(); p.w = (const half*) w_h.data_ptr();
    p.router = (const half*) router.data_ptr(); p.sgate_w = (const half*) sgate.data_ptr();
    p.wt = (const uint32_t* const*) wt.data_ptr(); p.svt = (const half* const*) svt.data_ptr(); p.native = native;
    unsigned* ctl = (unsigned*) (wb + off[0]);
    p.bar = ctl; p.done = ctl + 1; p.err = (int*) (ctl + 2);
    p.dots = (float*) (wb + off[1]); p.post = (float*) (wb + off[2]); p.mixed = (half*) (wb + off[3]); p.scores = (half*) (wb + off[4]);
    p.sgl = (float*) (wb + off[5]); p.gu = (half*) (wb + off[6]); p.dn = (float*) (wb + off[7]); p.ydbg = (float*) (wb + off[8]); p.seldbg = (int*) (wb + off[9]);
    p.rms_eps = (float) rms_eps;
    mf::moe_fused_kernel<S><<<p.G, mf::THREADS, 0, stream>>>(p);
    cuda_check(hipPeekAtLastError());
}
}  // namespace

#define MF_SHAPES(X) X(mf::QwenShape) X(mf::SmallTestShape) X(mf::SmallK3Shape) X(mf::QwenK3S4) X(mf::QwenK3S5) X(mf::QwenK4S4) X(mf::QwenK4S5) X(mf::SmallK3S4) X(mf::SmallK3S5) X(mf::SmallK4S4) X(mf::SmallK4S5) X(mf::QwenK6S6) X(mf::SmallK6S6) X(mf::QwenK4S6) X(mf::QwenK5S6) X(mf::SmallK4S6) X(mf::SmallK5S6)

bool exl3_moe_fused_supported(int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB)
{
#define X(S) if (MfShapeMatch<S>::eq(D, H, LR, NEXP, TOPK, INTER, RB, SB)) return true;
    MF_SHAPES(X)
#undef X
    return false;
}

std::vector<int64_t> exl3_moe_fused_ws_offsets(int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB)
{
#define X(S) if (MfShapeMatch<S>::eq(D, H, LR, NEXP, TOPK, INTER, RB, SB)) return mf_ws_offsets<S>();
    MF_SHAPES(X)
#undef X
    TORCH_CHECK(false, "exl3_moe_fused: shape not instantiated");
    return {};
}

int64_t exl3_moe_fused_grid(int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB)
{
#define X(S) if (MfShapeMatch<S>::eq(D, H, LR, NEXP, TOPK, INTER, RB, SB)) return mf_grid<S>();
    MF_SHAPES(X)
#undef X
    TORCH_CHECK(false, "exl3_moe_fused: shape not instantiated");
    return 0;
}

void exl3_moe_fused_half
(
    at::Tensor& x, const at::Tensor& fn, const at::Tensor& fn_scale, const at::Tensor& upt, const at::Tensor& up_scale, const at::Tensor& w_h,
    const at::Tensor& router, const at::Tensor& sgate, const at::Tensor& wt, const at::Tensor& svt,
    at::Tensor& ws, double rms_eps, int64_t native, int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == at::kFloat && x.is_contiguous() && x.dim() == 4, "fused: x must be contiguous fp32 (b, s, H, D)");
    TORCH_CHECK(x.size(2) == H && x.size(3) == D, "fused: x shape");
    const int R = (int) (x.size(0) * x.size(1));
    TORCH_CHECK(R >= 1 && R <= mf::MAXR, "fused: rows 1..4");
    TORCH_CHECK(fn.dtype() == at::kChar && upt.dtype() == at::kChar, "fused: int8 weights");
    TORCH_CHECK(fn_scale.dtype() == at::kFloat && up_scale.dtype() == at::kFloat, "fused: fp32 scales");
    TORCH_CHECK(w_h.dtype() == at::kHalf && router.dtype() == at::kHalf && sgate.dtype() == at::kHalf, "fused: fp16 tensors");
    TORCH_CHECK(ws.dtype() == at::kByte, "fused: workspace is uint8");
#define X(S) if (MfShapeMatch<S>::eq(D, H, LR, NEXP, TOPK, INTER, RB, SB)) { mf_launch<S>(x, fn, fn_scale, upt, up_scale, w_h, router, sgate, wt, svt, ws, rms_eps, (int) native, R, stream); return; }
    MF_SHAPES(X)
#undef X
    TORCH_CHECK(false, "exl3_moe_fused: shape not instantiated");
}

// ---- tile test entry: (bits, K, cf32) combinations instantiated: K = 640 / 2048 (down, float out), 2560 / 4096 (gate/up, half out); bits 2, 3, 4
namespace {
template <int BITS, bool CF, int KS>
bool mf_tt_launch(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int R, int nt, int native, hipStream_t stream)
{
    mf::mf_tile_test_kernel<BITS, CF, KS><<<1, mf::THREADS, 0, stream>>>((const uint32_t*) A.data_ptr(), (const uint32_t*) B.data_ptr(), C.data_ptr(), R, nt, native);
    cuda_check(hipPeekAtLastError());
    return true;
}
template <int BITS>
bool mf_tt_dispatch(int ks, bool cf, const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int R, int nt, int native, hipStream_t stream)
{
    if (!cf && ks == 160) return mf_tt_launch<BITS, false, 160>(A, B, C, R, nt, native, stream);
    if (!cf && ks == 256) return mf_tt_launch<BITS, false, 256>(A, B, C, R, nt, native, stream);
    if (cf && ks == 40) return mf_tt_launch<BITS, true, 40>(A, B, C, R, nt, native, stream);
    if (cf && ks == 128) return mf_tt_launch<BITS, true, 128>(A, B, C, R, nt, native, stream);
    return false;
}
}  // namespace

void exl3_moe_fused_tile_test(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int64_t bits, int64_t native)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(A.dtype() == at::kHalf && A.dim() == 2 && A.is_contiguous(), "tile test: A half [R, K]");
    TORCH_CHECK(B.dtype() == at::kShort && B.is_contiguous(), "tile test: B int16 (native [K/16, N/16, 16 bits] or repacked)");
    const bool cf = C.dtype() == at::kFloat;
    TORCH_CHECK(cf || C.dtype() == at::kHalf, "tile test: C half or float");
    TORCH_CHECK(C.is_contiguous() && C.dim() == 2, "tile test: C [R, N]");
    const int R = (int) A.size(0), K = (int) A.size(1), ks = K / 16;
    TORCH_CHECK(R >= 1 && R <= mf::MAXR && K % 16 == 0, "tile test: rows 1..4, K % 16");
    TORCH_CHECK(C.size(0) == R && C.size(1) % 16 == 0 && C.size(1) >= 16, "tile test: C shape");
    const int nt = (int) (C.size(1) / 16);
    TORCH_CHECK(B.numel() == (int64_t) ks * nt * 16 * bits, "tile test: B size vs A / C / bits");
    TORCH_CHECK(native == 1 || (native == 0 && nt % 2 == 0), "tile test: layout");
    bool ok = false;
    if (bits == 2) ok = mf_tt_dispatch<2>(ks, cf, A, B, C, R, nt, (int) native, stream);
    else if (bits == 3) ok = mf_tt_dispatch<3>(ks, cf, A, B, C, R, nt, (int) native, stream);
    else if (bits == 4) ok = mf_tt_dispatch<4>(ks, cf, A, B, C, R, nt, (int) native, stream);
    else if (bits == 5) ok = mf_tt_dispatch<5>(ks, cf, A, B, C, R, nt, (int) native, stream);
    else if (bits == 6) ok = mf_tt_dispatch<6>(ks, cf, A, B, C, R, nt, (int) native, stream);
    TORCH_CHECK(ok, "tile test: (bits, K, output dtype) not instantiated");
}
#endif
