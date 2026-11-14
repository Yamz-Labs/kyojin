// SPDX-License-Identifier: MIT
// Kyojin: routed (MoE) HIP MMQ adapter. Adapted from Gufo (github.com/gufo-org/gufo, 07e1191, MIT,
// Copyright (c) 2026 Gufo contributors) src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp
// (qfn_mmq_moe_impl), itself adapted from llama.cpp 5c0e946 ggml/src/ggml-cuda/mmq.cu (MIT, Copyright (c) 2023-2026
// The ggml authors). See THIRD_PARTY_NOTICES.md.
// Changes: routed path only (dense / vector / paired / decode adapters dropped); weights through a per-expert pointer
// table (mmq_args::x_ptrs) instead of one contiguous [E, M, K] tensor; scratch (id maps, q8_1 activations) supplied by
// the caller instead of the ggml pool; no output sanitize pass.
#include "qfn_mmq_prelude.h"
#include "gmmq_api.h"
#include "gmmq_case.cuh"
namespace qfn_mmq {
#include "quantize.hpp"
#include "mmid.hpp"
}  // namespace qfn_mmq

GMMQ_DECL_CASE(GGML_TYPE_Q4_K)
GMMQ_DECL_CASE(GGML_TYPE_Q5_K)
GMMQ_DECL_CASE(GGML_TYPE_Q5_1)
GMMQ_DECL_CASE(GGML_TYPE_Q8_0)

namespace qfn_mmq {

static ggml_backend_hip_context * gmmq_ctx_for_device(int device) {
    static std::unique_ptr<ggml_backend_hip_context> cached[GGML_HIP_MAX_DEVICES];
    static std::mutex mutex;
    if (device < 0 || device >= GGML_HIP_MAX_DEVICES) return nullptr;
    const std::lock_guard<std::mutex> lock(mutex);
    if (!cached[device]) cached[device] = std::make_unique<ggml_backend_hip_context>(device);
    return cached[device].get();
}

static bool gmmq_type_ok(int type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_0;
}

struct gmmq_layout {
    size_t off_src1, off_dst, off_bounds, off_scan, off_q8, total;
};

static size_t gmmq_align(size_t n) { return (n + 255) & ~(size_t) 255; }

static gmmq_layout gmmq_get_layout(int K, int n_tokens, int n_experts, int n_expert_used, int cc) {
    const int64_t rows = (int64_t) n_tokens * n_expert_used;
    const int64_t ne10_padded = GGML_PAD((int64_t) K, MATRIX_ROW_PADDING);
    gmmq_layout l;
    l.off_src1 = 0;
    l.off_dst = l.off_src1 + gmmq_align(rows * sizeof(int32_t));
    l.off_bounds = l.off_dst + gmmq_align(rows * sizeof(int32_t));
    l.off_scan = l.off_bounds + gmmq_align((size_t) (n_experts + 1) * sizeof(int32_t));
    l.off_q8 = l.off_scan + gmmq_align(gmmq_ids_scratch_elems(n_experts, n_tokens, n_expert_used) * sizeof(int32_t));
    const size_t q8 = rows * ne10_padded * sizeof(block_q8_1_mmq) / (4 * QK8_1) +
                      get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    l.total = l.off_q8 + gmmq_align(q8);
    return l;
}

template <ggml_type type>
static int gmmq_moe_impl(const void * const * w_ptrs, const float * X_f32, const int32_t * ids, float * out_f32,
                         int M, int K, int n_tokens, int n_experts, int n_expert_used,
                         char * scratch, const gmmq_layout & lay, int tile_cols, int max_expert_rows,
                         ggml_backend_hip_context & ctx, hipStream_t stream) {
    const int64_t ne_get_rows = (int64_t) n_tokens * n_expert_used;
    const int64_t ne00 = K;
    const int64_t ne10_padded = GGML_PAD((int64_t) K, MATRIX_ROW_PADDING);
    const int64_t ne11 = 1;
    const int64_t ne12 = n_tokens;
    const int64_t s01 = (int64_t) K / ggml_blck_size(type);

    int32_t * ids_src1 = (int32_t *) (scratch + lay.off_src1);
    int32_t * ids_dst = (int32_t *) (scratch + lay.off_dst);
    int32_t * expert_bounds = (int32_t *) (scratch + lay.off_bounds);
    int32_t * scan = (int32_t *) (scratch + lay.off_scan);
    char * src1_q8_1 = scratch + lay.off_q8;

    hipMemsetAsync(ids_src1, 0, ne_get_rows * sizeof(int32_t), stream);
    hipMemsetAsync(ids_dst, 0, ne_get_rows * sizeof(int32_t), stream);

    const int si1 = n_expert_used;
    const int sis1 = 1;
    ggml_hip_launch_mm_ids_helper(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used,
                                  (int) ne11, si1, sis1, scan, stream);

    const size_t y_block_size = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = 4 * QK8_1;
    quantize_mmq_q8_1_hip(X_f32, ids_src1, (void *) src1_q8_1, type, K, (int64_t) K, (int64_t) K * ne11,
                          (int64_t) K * ne11 * ne12, ne10_padded, ne_get_rows, 1, 1, stream);

    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size / (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    mmq_args args = {
        nullptr, type, (const int *) src1_q8_1, ids_dst, expert_bounds, out_f32,
        ne00, (int64_t) M, ne_get_rows, s01, ne_get_rows, (int64_t) M,
        (int64_t) n_experts, (int64_t) n_experts, (int64_t) M * s01, s12_mmq, (int64_t) 0,
        1, 1, 0, s13_mmq, 0,
        false, ne_get_rows,
        nullptr, 0,
    };
    args.ncols_grid_max = max_expert_rows > 0 && (int64_t) max_expert_rows < ne_get_rows ? max_expert_rows : 0;
    args.mmq_x_request = tile_cols > 0 ? tile_cols : 0;
    args.x_ptrs = (const char * const *) w_ptrs;
    mul_mat_q_case<type>(ctx, args, stream);
    return 0;
}

}  // namespace qfn_mmq

size_t gmmq_moe_scratch_bytes(int type, int K, int n_tokens, int n_experts, int n_expert_used) {
    using namespace qfn_mmq;
    (void) type;
    const int dev = ggml_hip_get_device();
    return gmmq_get_layout(K, n_tokens, n_experts, n_expert_used, ggml_hip_info().devices[dev].cc).total;
}

int gmmq_moe(int type, const void * const * w_ptrs, const float * x, const int32_t * ids, float * out,
             int M, int K, int n_tokens, int n_experts, int n_expert_used,
             void * scratch, size_t scratch_bytes, int tile_cols, int max_expert_rows, hipStream_t stream) {
    using namespace qfn_mmq;
    constexpr const char * tag = "gmmq_moe";
    if (!gmmq_type_ok(type)) {
        fprintf(stderr, "%s: unsupported ggml type %d\n", tag, type);
        return -1;
    }
    if (!w_ptrs || !x || !ids || !out || !scratch) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0 || n_expert_used > n_experts ||
        K % ggml_blck_size((ggml_type) type) != 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n", tag, M, K, n_tokens, n_experts,
                n_expert_used);
        return -1;
    }
    const int dev = ggml_hip_get_device();
    ggml_backend_hip_context * ctx = gmmq_ctx_for_device(dev);
    if (!ctx) return -1;
    const gmmq_layout lay = gmmq_get_layout(K, n_tokens, n_experts, n_expert_used, ggml_hip_info().devices[dev].cc);
    if (scratch_bytes < lay.total) {
        fprintf(stderr, "%s: scratch %zu < %zu bytes\n", tag, scratch_bytes, lay.total);
        return -1;
    }
    char * s = (char *) scratch;
    int rc = -1;
    switch (type) {
        case GGML_TYPE_Q4_K: rc = gmmq_moe_impl<GGML_TYPE_Q4_K>(w_ptrs, x, ids, out, M, K, n_tokens, n_experts, n_expert_used, s, lay, tile_cols, max_expert_rows, *ctx, stream); break;
        case GGML_TYPE_Q5_K: rc = gmmq_moe_impl<GGML_TYPE_Q5_K>(w_ptrs, x, ids, out, M, K, n_tokens, n_experts, n_expert_used, s, lay, tile_cols, max_expert_rows, *ctx, stream); break;
        case GGML_TYPE_Q5_1: rc = gmmq_moe_impl<GGML_TYPE_Q5_1>(w_ptrs, x, ids, out, M, K, n_tokens, n_experts, n_expert_used, s, lay, tile_cols, max_expert_rows, *ctx, stream); break;
        case GGML_TYPE_Q8_0: rc = gmmq_moe_impl<GGML_TYPE_Q8_0>(w_ptrs, x, ids, out, M, K, n_tokens, n_experts, n_expert_used, s, lay, tile_cols, max_expert_rows, *ctx, stream); break;
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, hipGetErrorString(err));
        return -4;
    }
    return rc;
}
