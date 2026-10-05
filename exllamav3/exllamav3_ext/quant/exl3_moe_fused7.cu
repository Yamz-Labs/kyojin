#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "exl3_moe_fused_api.h"
#include "hadamard.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_moe_valu.cuh"
#include "exl3_moe_fused7.cuh"
#include <algorithm>
#include <type_traits>

// core7: torch.ops.mf7.half, same arguments and workspace layout as exl3_moe_fused_half (Qwen shape only), plus `timing` (1 = phase stamps into seldbg).
namespace {
template <class S7, bool TM, int ABL> int grid7()
{
    static int G = [] {
        int nb = 0, dev = 0; hipGetDevice(&dev);
        hipDeviceProp_t prop; hipGetDeviceProperties(&prop, dev);
        cuda_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*) mf7::moe_fused7_kernel<S7, TM, ABL>, mf::THREADS, 0));
        return std::max(1, nb) * prop.multiProcessorCount;
    }();
    return G;
}
template <class S> struct Match7
{
    static bool eq(int D, int H, int LR, int NEXP, int TOPK, int INTER, int RB, int SB)
    { return D == S::D && H == S::H && LR == S::LR && NEXP == S::NEXP && TOPK == S::TOPK && INTER == S::INTER && RB == S::RB && SB == S::SB; }
};
template <class S7, bool TM, int ABL> void launch7(const at::Tensor& x, const at::Tensor& fn, const at::Tensor& fn_scale, const at::Tensor& upt, const at::Tensor& up_scale,
                                const at::Tensor& w_h, const at::Tensor& router, const at::Tensor& sgate, const at::Tensor& wt, const at::Tensor& svt,
                                at::Tensor& ws, double rms_eps, int native, int R, int vmask, hipStream_t stream)
{
    using DM = mf::Dm<S7>;
    const auto off = exl3_moe_fused_ws_offsets(S7::D, S7::H, S7::LR, S7::NEXP, S7::TOPK, S7::INTER, S7::RB, S7::SB);
    TORCH_CHECK(ws.numel() >= off.back(), "mf7: workspace too small");
    const bool gs = fn_scale.numel() != DM::MR;
    TORCH_CHECK(!gs || mf::gs_ok<S7>(), "mf7: group scales need D % 128 == 0");
    TORCH_CHECK(fn.numel() == (int64_t) DM::MR * S7::H * S7::D && fn_scale.numel() == mf::fn_scale_n<S7>(gs), "mf7: fn size");
    TORCH_CHECK(upt.numel() == (int64_t) S7::H * DM::D4 * S7::LR * 4, "mf7: upt size");
    TORCH_CHECK(up_scale.numel() == mf::up_scale_n<S7>(gs) && w_h.numel() == (int64_t) S7::H * S7::D, "mf7: up_scale size");
    TORCH_CHECK(router.numel() == (int64_t) S7::NEXP * S7::D && sgate.numel() == S7::D, "mf7: router size");
    TORCH_CHECK(wt.dtype() == at::kLong && wt.numel() == 3 * (S7::NEXP + 1) && svt.dtype() == at::kLong && svt.numel() == 6 * (S7::NEXP + 1), "mf7: pointer tables");
    char* wb = (char*) ws.data_ptr();
    mf::Params p{};
    p.R = R; p.G = grid7<S7, TM, ABL>();
    TORCH_CHECK(((long long) S7::H * (DM::MR + 1) + 4LL * p.G - 1) / (4LL * p.G) + 1 <= mf7::DOTS_MAXIT, "mf7: dots items per team exceed DOTS_MAXIT");
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
    mf7::moe_fused7_kernel<S7, TM, ABL><<<p.G, mf::THREADS, 0, stream>>>(p, vmask);
    cuda_check(hipPeekAtLastError());
}
}  // namespace

void mf7_half(at::Tensor& x, const at::Tensor& fn, const at::Tensor& fn_scale, const at::Tensor& upt, const at::Tensor& up_scale, const at::Tensor& w_h,
              const at::Tensor& router, const at::Tensor& sgate, const at::Tensor& wt, const at::Tensor& svt, at::Tensor& ws, double rms_eps, int64_t native, int64_t timing, int64_t variant,
              int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == at::kFloat && x.is_contiguous() && x.dim() == 4 && x.size(2) == H && x.size(3) == D, "mf7: x");
    const int R = (int) (x.size(0) * x.size(1));
    TORCH_CHECK(R >= 1 && R <= mf::MAXR, "mf7: rows 1..4");
    TORCH_CHECK(fn.dtype() == at::kChar && upt.dtype() == at::kChar && fn_scale.dtype() == at::kFloat && up_scale.dtype() == at::kFloat, "mf7: dtypes");
    TORCH_CHECK(w_h.dtype() == at::kHalf && router.dtype() == at::kHalf && sgate.dtype() == at::kHalf && ws.dtype() == at::kByte, "mf7: dtypes 2");
    TORCH_CHECK(native == 0 || native == 1, "mf7: native flag");
#define L7(SH, TMV, ABLV) launch7<SH, TMV, ABLV>(x, fn, fn_scale, upt, up_scale, w_h, router, sgate, wt, svt, ws, rms_eps, (int) native, R, (int) (variant & 255), stream)
#define X7(SH) if (Match7<SH>::eq(D, H, LR, NEXP, TOPK, INTER, RB, SB)) { \
        if (timing) { L7(SH, true, 0); return; } \
        if (std::is_same<SH, mf::QwenShape>::value) { \
            switch (variant >> 8) { case 1: L7(mf::QwenShape, false, 1); return; case 2: L7(mf::QwenShape, false, 2); return; case 3: L7(mf::QwenShape, false, 3); return; \
                                    case 8: L7(mf::QwenShape, false, 8); return; default: break; } } \
        L7(SH, false, 0); return; }
    X7(mf::QwenShape) X7(mf::SmallTestShape)
#undef X7
    TORCH_CHECK(false, "mf7: shape not instantiated");
}

TORCH_LIBRARY_FRAGMENT(mf7, m)
{
    m.def("half(Tensor(a!) x, Tensor fn, Tensor fn_scale, Tensor upt, Tensor up_scale, Tensor w_h, Tensor router, Tensor sgate, Tensor wt, Tensor svt, Tensor(b!) ws, float rms_eps, int native, int timing, int variant, int D, int H, int LR, int NEXP, int TOPK, int INTER, int RB, int SB) -> ()");
}
TORCH_LIBRARY_IMPL(mf7, CUDA, m) { m.impl("half", &mf7_half); }
#endif
