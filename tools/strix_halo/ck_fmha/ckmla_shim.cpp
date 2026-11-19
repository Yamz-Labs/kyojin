// ckmla: C ABI over CK fmha_fwd for un-absorbed MLA prefill (fp16, batch 1, hd 256/256 (any hd the build covers), per-tensor strides,
// optional elementwise bias [sq, sk] (head-broadcast), optional per-head learnable sink (fp32, in scaled-logit space), causal bottom-right
// with an optional left window (window_left > 0: keys in [q - window_left, q]). Separate symbol from ckpf_fwd (that one is unchanged).
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
#include "bias.hpp"
#include "fmha_fwd.hpp"
#include "mask.hpp"

extern "C" float ckmla_fwd(const void* q, const void* k, const void* v, void* o, const void* bias, const void* sink,
                           int sq, int sk, int hq, int hk, int dq, int dv, float scale,
                           long stride_q, long nh_q, long stride_k, long nh_k, long stride_v, long nh_v, long stride_o, long nh_o,
                           long stride_bias, int window_left, int causal, hipStream_t stream, int time_it)
{
    fmha_fwd_traits t{};
    t.hdim_q = dq; t.hdim_v = dv; t.data_type = "fp16"; t.is_group_mode = false;
    t.is_v_rowmajor = true; t.has_logits_soft_cap = false;
    t.mask_type = causal ? mask_enum::mask_bottom_right : mask_enum::no_mask;
    t.bias_type = bias ? bias_enum::elementwise_bias : bias_enum::no_bias;
    t.has_lse = false; t.has_dropout = false; t.has_sink = sink != nullptr;
    t.qscale_type = quant_scale_enum::no_scale;
    fmha_fwd_args a{};
    a.q_ptr = q; a.k_ptr = k; a.v_ptr = v; a.o_ptr = o; a.bias_ptr = bias; a.sink_ptr = sink;
    a.seqlen_q = sq; a.seqlen_k = sk; a.batch = 1; a.max_seqlen_q = sq;
    a.hdim_q = dq; a.hdim_v = dv; a.nhead_q = hq; a.nhead_k = hk;
    a.scale_s = scale; a.logits_soft_cap = 0.f;
    a.stride_q = stride_q; a.stride_k = stride_k; a.stride_v = stride_v; a.stride_o = stride_o; a.stride_bias = stride_bias;
    a.nhead_stride_q = nh_q; a.nhead_stride_k = nh_k; a.nhead_stride_v = nh_v; a.nhead_stride_o = nh_o; a.nhead_stride_bias = 0;
    a.batch_stride_q = 0; a.batch_stride_k = 0; a.batch_stride_v = 0; a.batch_stride_o = 0; a.batch_stride_bias = 0;
    a.window_size_left = window_left > 0 ? window_left : -1; a.window_size_right = causal ? 0 : -1;
    a.sink_size = 0;
    a.mask_type = static_cast<ck_tile::index_t>(t.mask_type);
    a.p_drop = 0.f; a.s_randval = false;
    a.drop_seed_offset = std::pair<uint64_t, uint64_t>{0, 0};
    ck_tile::stream_config cfg{stream, time_it != 0, 0, 0, 1, false};
    return fmha_fwd(t, a, cfg);
}
