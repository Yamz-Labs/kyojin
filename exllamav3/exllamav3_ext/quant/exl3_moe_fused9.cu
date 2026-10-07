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
#include "exl3_moe_fused9.cuh"
#include <algorithm>
#include <type_traits>

// core9: torch.ops.mf9.half (mf7 kernel + cache touches), same arguments and workspace layout as exl3_moe_fused_half (Qwen shape only), plus `timing` (1 = phase stamps into seldbg).
namespace {
template <class S7, bool TM, int OCC, int INL, int ABL = 0> int grid7()
{
    static int G = [] {
        int nb = 0, dev = 0; hipGetDevice(&dev);
        hipDeviceProp_t prop; hipGetDeviceProperties(&prop, dev);
        cuda_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (OCC ? (const void*) mf9::moe_fused9o_kernel<S7, TM, (bool) INL, ABL> : (const void*) mf9::moe_fused9_kernel<S7, TM, (bool) INL, ABL>), mf::THREADS, 0));
        return (OCC == 3 ? 1 : (OCC == 2 || OCC == 4) ? 2 : std::max(1, nb)) * prop.multiProcessorCount;   // OCC 2: forced two blocks per WGP (the occupancy API reports one); OCC 4: two blocks per WGP as six stage launches without any grid barrier; OCC 3: one block per WGP, half the VGPR / LDS free for other GPU clients
    }();
    return G;
}
template <class S> struct Match7
{
    static bool eq(int D, int H, int LR, int NEXP, int TOPK, int INTER, int RB, int SB)
    { return D == S::D && H == S::H && LR == S::LR && NEXP == S::NEXP && TOPK == S::TOPK && INTER == S::INTER && RB == S::RB && SB == S::SB; }
};
template <class S7, bool TM, int OCC, int INL, int ABL = 0> void launch7(const at::Tensor& x, const at::Tensor& fn, const at::Tensor& fn_scale, const at::Tensor& upt, const at::Tensor& up_scale,
                                const at::Tensor& w_h, const at::Tensor& router, const at::Tensor& sgate, const at::Tensor& wt, const at::Tensor& svt,
                                at::Tensor& ws, double rms_eps, int native, int R, int vmask, hipStream_t stream)
{
    using DM = mf::Dm<S7>;
    TORCH_CHECK(R >= 1 && R <= mf::TRows<S7>::v, "mf9: rows exceed this instantiation");
    const auto off = exl3_moe_fused_ws_offsets(S7::D, S7::H, S7::LR, S7::NEXP, S7::TOPK, S7::INTER, S7::RB, S7::SB);
    TORCH_CHECK(ws.numel() >= off.back(), "mf9: workspace too small");
    const bool gs = fn_scale.numel() != DM::MR;
    TORCH_CHECK(!gs || mf::gs_ok<S7>(), "mf9: group scales need D % 128 == 0");
    TORCH_CHECK(fn.numel() == (int64_t) DM::MR * S7::H * S7::D && fn_scale.numel() == mf::fn_scale_n<S7>(gs), "mf9: fn size");
    TORCH_CHECK(upt.numel() == (int64_t) S7::H * DM::D4 * S7::LR * 4, "mf9: upt size");
    TORCH_CHECK(up_scale.numel() == mf::up_scale_n<S7>(gs) && w_h.numel() == (int64_t) S7::H * S7::D, "mf9: up_scale size");
    TORCH_CHECK(router.numel() == (int64_t) S7::NEXP * S7::D && sgate.numel() == S7::D, "mf9: router size");
    TORCH_CHECK(wt.dtype() == at::kLong && wt.numel() == 3 * (S7::NEXP + 1) && svt.dtype() == at::kLong && svt.numel() == 6 * (S7::NEXP + 1), "mf9: pointer tables");
    char* wb = (char*) ws.data_ptr();
    mf::Params p{};
    p.R = R; p.G = grid7<S7, TM, OCC, INL, ABL>();
    TORCH_CHECK(p.G >= 1 && p.G <= mf::CTL_MAXB, "mf9: grid exceeds the barrier arrival-marker words");
    TORCH_CHECK(((long long) S7::H * (DM::MR + 1) + 4LL * p.G - 1) / (4LL * p.G) + 1 <= mf7::DOTS_MAXIT, "mf9: dots items per team exceed DOTS_MAXIT");
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
    if constexpr (OCC == 4)
    {
        static_assert(!TM && INL == 1, "split launches: no timing build, inline phases");
        mf9::moe_fused9s_kernel<S7, (bool) INL, ABL, 0><<<p.G, mf::THREADS, 0, stream>>>(p, vmask);
        mf9::moe_fused9s_kernel<S7, (bool) INL, ABL, 1><<<p.G, mf::THREADS, 0, stream>>>(p, vmask);
        mf9::moe_fused9s_kernel<S7, (bool) INL, ABL, 2><<<p.G, mf::THREADS, 0, stream>>>(p, vmask);
        mf9::moe_fused9s_kernel<S7, (bool) INL, ABL, 3><<<p.G, mf::THREADS, 0, stream>>>(p, vmask);
        mf9::moe_fused9s_kernel<S7, (bool) INL, ABL, 4><<<p.G, mf::THREADS, 0, stream>>>(p, vmask);
        mf9::moe_fused9s_kernel<S7, (bool) INL, ABL, 5><<<p.G, mf::THREADS, 0, stream>>>(p, vmask);
    }
    else
    {
        if (OCC) { mf9::moe_fused9o_kernel<S7, TM, (bool) INL, ABL><<<p.G, mf::THREADS, 0, stream>>>(p, vmask); }
        else { mf9::moe_fused9_kernel<S7, TM, (bool) INL, ABL><<<p.G, mf::THREADS, 0, stream>>>(p, vmask); }
    }
    cuda_check(hipPeekAtLastError());
}
}  // namespace

void mf9_half(at::Tensor& x, const at::Tensor& fn, const at::Tensor& fn_scale, const at::Tensor& upt, const at::Tensor& up_scale, const at::Tensor& w_h,
              const at::Tensor& router, const at::Tensor& sgate, const at::Tensor& wt, const at::Tensor& svt, at::Tensor& ws, double rms_eps, int64_t native, int64_t timing, int64_t variant,
              int64_t D, int64_t H, int64_t LR, int64_t NEXP, int64_t TOPK, int64_t INTER, int64_t RB, int64_t SB)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == at::kFloat && x.is_contiguous() && x.dim() == 4 && x.size(2) == H && x.size(3) == D, "mf9: x");
    const int R = (int) (x.size(0) * x.size(1));
    TORCH_CHECK(R >= 1 && R <= mf::MAXRT, "mf9: rows 1..8");
    TORCH_CHECK(fn.dtype() == at::kChar && upt.dtype() == at::kChar && fn_scale.dtype() == at::kFloat && up_scale.dtype() == at::kFloat, "mf9: dtypes");
    TORCH_CHECK(w_h.dtype() == at::kHalf && router.dtype() == at::kHalf && sgate.dtype() == at::kHalf && ws.dtype() == at::kByte, "mf9: dtypes 2");
    TORCH_CHECK(native == 0 || native == 1, "mf9: native flag");
#define L7A(SH, TMV, OCCV, INLV, ABLV) launch7<SH, TMV, OCCV, INLV, ABLV>(x, fn, fn_scale, upt, up_scale, w_h, router, sgate, wt, svt, ws, rms_eps, (int) native, R, (int) (variant & 0xffff), stream)
#define L7(SH, TMV, OCCV, INLV) launch7<SH, TMV, OCCV, INLV>(x, fn, fn_scale, upt, up_scale, w_h, router, sgate, wt, svt, ws, rms_eps, (int) native, R, (int) (variant & 0xffff), stream)
    // moe8: 5..8 rows in one launch (wide instantiation: row windows in the front end, units of <= 4 rows in the back end). Served variant only (six stage launches; the one-kernel variants spill at 8 rows).
    if (R > mf::MAXR)
    {
        TORCH_CHECK(!timing, "mf9: wide launches (rows 5..8) have no timing build");
        const int64_t vv = variant & 0x1f0000;
        TORCH_CHECK(vv == 0x160000, "mf9: wide launches (rows 5..8) exist for the default variant 0x160000 (six stage launches) only");
#define XW(SH) if (Match7<mf::Wide<SH>>::eq(D, H, LR, NEXP, TOPK, INTER, RB, SB)) { L7(mf::Wide<SH>, false, 4, 1); return; }
        XW(mf::QwenShape) XW(mf::SmallTestShape) XW(mf::QwenK3S4) XW(mf::QwenK3S5) XW(mf::QwenK4S4) XW(mf::QwenK4S5)
        XW(mf::SmallK3S4) XW(mf::SmallK3S5) XW(mf::SmallK4S4) XW(mf::SmallK4S5)
#undef XW
        TORCH_CHECK(false, "mf9: shape not instantiated for wide launches");
    }
#define X7(SH) if (Match7<SH>::eq(D, H, LR, NEXP, TOPK, INTER, RB, SB)) { \
        if (variant & 0x100000) { TORCH_CHECK((variant & 0x1f0000) == 0x160000 && !timing, "mf9: split launches (0x100000) need variant 0x160000 and no timing"); L7(SH, false, 4, 1); return; }   /* six stage launches, no grid barrier */ \
        if ((variant & 0x80000) && (variant & 0x70000) == 0x60000) { if (timing) L7(SH, true, 3, 1); else L7(SH, false, 3, 1); return; }   /* one block per WGP */ \
        const int occ = (variant & 0x40000) ? 2 : ((variant & 0x10000) != 0), inl = (variant & 0x20000) != 0; \
        if (timing) { if (occ == 2) { if (inl) L7(SH, true, 2, 1); else L7(SH, true, 2, 0); } else if (occ) { if (inl) L7(SH, true, 1, 1); else L7(SH, true, 1, 0); } else { if (inl) L7(SH, true, 0, 1); else L7(SH, true, 0, 0); } return; } \
        if (occ == 2) { if (inl) L7(SH, false, 2, 1); else L7(SH, false, 2, 0); } else if (occ) { if (inl) L7(SH, false, 1, 1); else L7(SH, false, 1, 0); } else { if (inl) L7(SH, false, 0, 1); else L7(SH, false, 0, 0); } return; }
    X7(mf::QwenShape) X7(mf::SmallTestShape)
#undef X7
    // the shapes of K3 / K4 routed packs and K4 / K5 shared experts carry the served variant only (two blocks per WGP, forced inline, no timing build)
#define X9D(SH) if (Match7<SH>::eq(D, H, LR, NEXP, TOPK, INTER, RB, SB)) { \
        TORCH_CHECK((variant & 0x70000) == 0x60000, "mf9: this shape is instantiated for the default variant 0x60000 only"); \
        if (variant & 0x100000) { TORCH_CHECK((variant & 0x1f0000) == 0x160000 && !timing, "mf9: split launches (0x100000) need variant 0x160000 and no timing"); L7(SH, false, 4, 1); return; }   /* split launches */ \
        const bool one = (variant & 0x80000) != 0;   /* one block per WGP, same kernel and arithmetic */ \
        if (timing) { if (one) L7(SH, true, 3, 1); else L7(SH, true, 2, 1); return; } \
        if (one) L7(SH, false, 3, 1); else L7(SH, false, 2, 1); return; }
    // ablation arms (timing only, garbage values by design, no index depends on them): variant bits 0x100 no decode, 0x200 every slice re-reads slice 0 (K4/S5 only)
    if (Match7<mf::QwenK4S5>::eq(D, H, LR, NEXP, TOPK, INTER, RB, SB) && (variant & 0x300) && timing) { TORCH_CHECK((variant & 0x70000) == 0x60000, "mf9: ablation needs 0x60000");
        switch ((variant >> 8) & 3) { case 1: L7A(mf::QwenK4S5, true, 2, 1, 1); break; case 2: L7A(mf::QwenK4S5, true, 2, 1, 2); break; default: L7A(mf::QwenK4S5, true, 2, 1, 3); } return; }
    X9D(mf::QwenK3S4) X9D(mf::QwenK3S5) X9D(mf::QwenK4S4) X9D(mf::QwenK4S5)
    X9D(mf::QwenK6S6) X9D(mf::SmallK6S6)
    X9D(mf::SmallK3S4) X9D(mf::SmallK3S5) X9D(mf::SmallK4S4) X9D(mf::SmallK4S5)
#undef X9D
    TORCH_CHECK(false, "mf9: shape not instantiated");
}

TORCH_LIBRARY_FRAGMENT(mf9, m)
{
    m.def("half(Tensor(a!) x, Tensor fn, Tensor fn_scale, Tensor upt, Tensor up_scale, Tensor w_h, Tensor router, Tensor sgate, Tensor wt, Tensor svt, Tensor(b!) ws, float rms_eps, int native, int timing, int variant, int D, int H, int LR, int NEXP, int TOPK, int INTER, int RB, int SB) -> ()");
}
TORCH_LIBRARY_IMPL(mf9, CUDA, m) { m.impl("half", &mf9_half); }
#endif
