// SPDX-License-Identifier: MIT
// Kyojin: torch binding of the routed (MoE) HIP MMQ (rocm/gmmq, adapted from Gufo / llama.cpp, MIT; see
// THIRD_PARTY_NOTICES.md). HIP only.
#if defined(USE_ROCM)

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "gmmq.h"
#include "gmmq_api.h"

void fq_mmq_moe(torch::Tensor x, torch::Tensor ids, torch::Tensor w_ptrs, int64_t fmt, int64_t M, torch::Tensor out,
                int64_t tile_cols, int64_t max_expert_rows)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
                (x.dtype() == torch::kFloat || x.dtype() == torch::kHalf), "fq_mmq_moe: x must be fp32/fp16 [T, K] contiguous");
    TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && ids.is_contiguous() && ids.dtype() == torch::kInt &&
                ids.size(0) == x.size(0), "fq_mmq_moe: ids must be int32 [T, n_used] contiguous");
    TORCH_CHECK(w_ptrs.is_cuda() && w_ptrs.dim() == 1 && w_ptrs.is_contiguous() && w_ptrs.dtype() == torch::kLong,
                "fq_mmq_moe: w_ptrs must be an int64 device tensor [E]");
    const int64_t T = x.size(0), K = x.size(1), n_used = ids.size(1), E = w_ptrs.size(0);
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat && out.is_contiguous() && out.dim() == 2 &&
                out.size(0) == T * n_used && out.size(1) == M, "fq_mmq_moe: out must be fp32 [T * n_used, M] contiguous");
    TORCH_CHECK(T * n_used < (int64_t) 1 << 31 && M * T * n_used < ((int64_t) 1 << 31) && E < 65536,
                "fq_mmq_moe: shape too large");
    if (T == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    torch::Tensor xf = x.dtype() == torch::kFloat ? x : x.to(torch::kFloat);
    const size_t nbytes = gmmq_moe_scratch_bytes((int) fmt, (int) K, (int) T, (int) E, (int) n_used);
    torch::Tensor scratch = torch::empty({(int64_t) nbytes}, x.options().dtype(torch::kUInt8));
    const int rc = gmmq_moe((int) fmt, (const void * const *) w_ptrs.data_ptr(), (const float *) xf.data_ptr(),
                            (const int32_t *) ids.data_ptr(), (float *) out.data_ptr(), (int) M, (int) K, (int) T,
                            (int) E, (int) n_used, scratch.data_ptr(), nbytes, (int) tile_cols, (int) max_expert_rows,
                            stream);
    TORCH_CHECK(rc == 0, "fq_mmq_moe: gmmq_moe failed (rc ", rc, ", fmt ", fmt, ")");
}

#endif
