// SPDX-License-Identifier: MIT
// Thin C ABI wrapper for the pinned Composable Kernel paged-KV specialization.
// Built by tools/strix_halo/build_ck_pagedkv.sh; no binary is committed.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <hip/hip_runtime.h>
#include "bias.hpp"
#include "fmha_fwd.hpp"
#include "mask.hpp"

using ck_mask = ck_tile::SimplifiedGenericAttentionMask<true>;
using ck_trait = fmha_fwd_pagedkv_traits_<
    192, FmhaFwdFp16, false, 64, 64, 32, 128, 32, 192, true,
    ck_tile::BlockFmhaPipelineEnum::QRKSVS, false, ck_mask,
    ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, false, true,
    true, false, false, false, false, -1>;
template <>
float fmha_fwd_pagedkv_<ck_trait, ck_tile::gfx11_t>(
    const ck_tile::stream_config&, fmha_fwd_pagedkv_args);

// Keep a prior C ABI declaration so CK's native -Weverything diagnostics
// remain enabled, including -Wmissing-prototypes.
extern "C" float ck_pagedkv_forward(
    const void* q, const void* k, const void* v, void* out,
    const int32_t* block_table, const int32_t* lengths,
    int q_len, int kv_len, int pages, float scale, int q_head_major,
    int value_stride, hipStream_t stream, int time_it);

extern "C" float ck_pagedkv_forward(
    const void* q, const void* k, const void* v, void* out,
    const int32_t* block_table, const int32_t* lengths,
    int q_len, int kv_len, int pages, float scale, int q_head_major,
    int value_stride, hipStream_t stream, int time_it) try {
    constexpr int q_heads = 64;
    constexpr int kv_heads = 4;
    constexpr int qk_dim = 192;
    constexpr int value_dim = 128;
    constexpr int page_size = 256;

    // -1 is reserved for rejection before the specialization can launch.
    // Null stream is valid: HIP uses it for the default stream.
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
        block_table == nullptr || lengths == nullptr || q_len <= 0 ||
        kv_len < q_len || (q_head_major != 0 && q_head_major != 1) ||
        value_stride != qk_dim) {
        return -1.0f;
    }
    // Avoid overflowing either ceiling division or CK's int32 stride fields.
    const int required_pages = kv_len / page_size + (kv_len % page_size != 0 ? 1 : 0);
    if (pages < required_pages ||
        q_len > std::numeric_limits<ck_tile::index_t>::max() / q_heads / qk_dim) {
        return -1.0f;
    }

    fmha_fwd_pagedkv_args args{};
    args.q_ptr = const_cast<void*>(q);
    args.k_ptr = const_cast<void*>(k);
    args.v_ptr = const_cast<void*>(v);
    args.o_ptr = out;
    args.block_table_ptr = const_cast<int32_t*>(block_table);
    args.batch_stride_block_table = pages;
    args.page_block_size = page_size;
    args.seqlen_k_ptr = const_cast<int32_t*>(lengths);
    args.seqlen_q = q_len;
    args.seqlen_k = kv_len;
    args.batch = 1;
    args.max_seqlen_q = q_len;
    args.hdim_q = qk_dim;
    args.hdim_v = value_dim;
    args.nhead_q = q_heads;
    args.nhead_k = kv_heads;
    args.scale_s = scale;
    args.scale_p = 1.0f;
    args.scale_o = 1.0f;
    args.window_size_left = -1;
    args.window_size_right = 0;
    args.mask_type = static_cast<ck_tile::index_t>(mask_enum::mask_bottom_right);
    args.stride_k = qk_dim;
    args.nhead_stride_k = static_cast<ck_tile::index_t>(page_size) * qk_dim;
    args.batch_stride_k = kv_heads * page_size * qk_dim;
    args.stride_v = value_stride;
    args.nhead_stride_v = static_cast<ck_tile::index_t>(page_size) * value_stride;
    args.batch_stride_v = kv_heads * page_size * value_stride;
    if (q_head_major) {
        args.stride_q = qk_dim;
        args.nhead_stride_q = static_cast<ck_tile::index_t>(q_len) * qk_dim;
        args.batch_stride_q = q_heads * q_len * qk_dim;
        args.stride_o = value_stride;
        args.nhead_stride_o = static_cast<ck_tile::index_t>(q_len) * value_stride;
        args.batch_stride_o = q_heads * q_len * value_stride;
    } else {
        args.stride_q = q_heads * qk_dim;
        args.nhead_stride_q = qk_dim;
        args.batch_stride_q = static_cast<ck_tile::index_t>(q_len) * q_heads * qk_dim;
        args.stride_o = q_heads * value_stride;
        args.nhead_stride_o = value_stride;
        args.batch_stride_o = static_cast<ck_tile::index_t>(q_len) * q_heads * value_stride;
    }
    const ck_tile::stream_config config{stream, time_it != 0, 0, 0, 1, false};
    const float elapsed = fmha_fwd_pagedkv_<ck_trait, ck_tile::gfx11_t>(config, args);
    if (!std::isfinite(elapsed) || elapsed < 0.0f) return -2.0f;
    if (time_it == 0) return 1.0f;
    return elapsed > 0.0f ? elapsed : -2.0f;
} catch (...) {
    // CK's HIP checks throw; no C++ exception may unwind into ctypes.
    return -2.0f;
}
