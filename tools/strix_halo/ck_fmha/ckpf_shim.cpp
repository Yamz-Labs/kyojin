// ckpf: C ABI over CK fmha_fwd (batch mode, bshd, fp16, GQA, causal bottom-right)
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
#include "bias.hpp"
#include "fmha_fwd.hpp"
#include "mask.hpp"

extern "C" float ckpf_fwd(const void* q, const void* k, const void* v, void* o,
                          int batch, int sq, int sk, int hq, int hk, int dq, int dv,
                          float scale, int causal, hipStream_t stream, int time_it)
{
    fmha_fwd_traits t{};
    t.hdim_q = dq; t.hdim_v = dv; t.data_type = "fp16"; t.is_group_mode = false;
    t.is_v_rowmajor = true; t.has_logits_soft_cap = false;
    t.mask_type = causal ? mask_enum::mask_bottom_right : mask_enum::no_mask;
    t.bias_type = bias_enum::no_bias; t.has_lse = false; t.has_dropout = false;
    t.qscale_type = quant_scale_enum::no_scale;
    fmha_fwd_args a{};
    a.q_ptr = q; a.k_ptr = k; a.v_ptr = v; a.o_ptr = o;
    a.seqlen_q = sq; a.seqlen_k = sk; a.batch = batch; a.max_seqlen_q = sq;
    a.hdim_q = dq; a.hdim_v = dv; a.nhead_q = hq; a.nhead_k = hk;
    a.scale_s = scale; a.logits_soft_cap = 0.f;
    a.stride_q = hq * dq; a.stride_k = hk * dq; a.stride_v = hk * dv; a.stride_o = hq * dv;
    a.nhead_stride_q = dq; a.nhead_stride_k = dq; a.nhead_stride_v = dv; a.nhead_stride_o = dv;
    a.batch_stride_q = (ck_tile::index_t)sq * hq * dq;
    a.batch_stride_k = (ck_tile::index_t)sk * hk * dq;
    a.batch_stride_v = (ck_tile::index_t)sk * hk * dv;
    a.batch_stride_o = (ck_tile::index_t)sq * hq * dv;
    a.window_size_left = -1; a.window_size_right = causal ? 0 : -1;
    a.mask_type = static_cast<ck_tile::index_t>(t.mask_type);
    a.p_drop = 0.f; a.s_randval = false;
    a.drop_seed_offset = std::pair<uint64_t, uint64_t>{0, 0};
    ck_tile::stream_config cfg{stream, time_it != 0, 0, 0, 1, false};
    return fmha_fwd(t, a, cfg);
}
