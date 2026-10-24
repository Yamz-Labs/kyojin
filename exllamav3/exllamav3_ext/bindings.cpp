#include <cuda_fp16.h>

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stloader.h"
#include "cuda_host.h"
#include "hadamard.h"
#include "hc_mix.cuh"
#include "activation.cuh"

// CPU expert offload: unconditional — the flag/worker/mul1 symbols are bound in both
// the CUDA and ROCm branches below.
#include "cpu/moe_mul1.h"
#include "cpu/moe_handoff.h"

#if !defined(USE_ROCM)

#include "norm.cuh"
#include "hgemm.cuh"
#include "rope.cuh"
#include "softcap.cuh"
#include "routing.cuh"
#include "gdn.cuh"
#include "add.cuh"

#include "quant/quantize.cuh"
#include "quant/pack.cuh"
#include "quant/reconstruct.cuh"
#include "quant/hadamard.cuh"
#include "quant/exl3_gemm.cuh"
#include "quant/exl3_gemv.cuh"
#include "quant/exl3_moe_fused_api.h"
#include "quant/exl3_gemv_int8.cuh"
#include "quant/exl3_moe_prefill.cuh"
#include "cpu/moe_mul1.h"
#include "cpu/moe_handoff.h"
#include "quant/exl3_kernel_map.cuh"
#include "quant/util.cuh"
#include "quant/exl3_devctx.cuh"
#include "quant/exl3_moe.cuh"
#include "quant/exl3_moe_grouped.cuh"
#include "quant/exl3_moe_coop.cuh"

#include "generator/strings.h"
#include "generator/sampling_basic.cuh"
#include "generator/sampling_extra.cuh"
#include "generator/gumbel.cuh"
#include "generator/sampling_fused.cuh"
#include "generator/rep_pen.cuh"
#include "generator/dry.cuh"
#include "moe_unswizzle.cuh"
#include "generator/cache.cuh"

#include "cache/q_cache.cuh"

#include "histogram.cuh"

#include "parallel/context.cuh"
#include "parallel/broadcast.cuh"
#include "parallel/barrier.cuh"
#include "parallel/gather.cuh"
#include "parallel/all_reduce.cuh"

#include "libtorch/gated_delta_net.h"
#include "libtorch/attention.h"
#include "libtorch/mla_attention.h"
#include "libtorch/linear.h"
#include "libtorch/gated_rmsnorm.h"
#include "libtorch/mlp.h"
#include "libtorch/blocksparse_mlp.h"
#include "libtorch/dsv4_compressor.h"
#include "libtorch/dsv4_attn.h"
#include "dsv4_compress.cuh"
#include "dsv4_pool_quant.cuh"
#include "dsa_topk.cuh"
#include "ple.cuh"
#include "dflash2.cuh"
#include "ngram.cuh"

#include "attention.cuh"

#include "sam.h"
#include "quant/exl3_dec.cuh"
#include "foreign_quant/fq_api.h"

#else

#include "hgemm.cuh"
#include "rope.cuh"
#include "gdn.cuh"
#include "routing_std_gfx12.cuh"
#include "dsa_topk_gfx12.cuh"
#include "dsa_topk_par.cuh"
#include "add.cuh"
#include "fused_elt.cuh"
#include "mla_dec_valu.cuh"
#include "kda_inter.cuh"

#include "quant/pack.cuh"
#include "quant/reconstruct.cuh"
#include "quant/quantize.cuh"
#include "quant/hadamard.cuh"
#include "quant/exl3_gemv.cuh"
#include "quant/exl3_moe_fused_api.h"
#include "quant/exl3_gemv_int8.cuh"
#include "quant/exl3_dec.cuh"
#include "quant/exl3_moe_prefill.cuh"

#include "generator/strings.h"
#include "generator/sampling_basic.cuh"
#include "generator/sampling_extra.cuh"
#include "generator/gumbel.cuh"
#include "generator/rep_pen.cuh"
#include "generator/cache.cuh"
#include "generator/dry.cuh"
#include "ngram.cuh"

#endif

#if defined(USE_ROCM)
// The ROCm dense-GEMM tuner (hgemm.cu, dtune) only runs for >= 256 rows and can take seconds on a cold
// tune cache: drop the GIL for those calls so the /health thread keeps answering. Smaller (decode)
// calls keep the GIL, as before.
static inline bool hgemm_gil_free(const at::Tensor& a)
{
    return a.dim() >= 1 && a.size(-1) > 0 && a.numel() / a.size(-1) >= 256;
}
#endif

#if defined(USE_ROCM) || defined(EXL3_PORT_DEC)
// exl3_dec decode kernels (quant/exl3_dec.cu): always bound on ROCm; on CUDA only in an EXL3_PORT_DEC build
// and only when EXL3_PORT_DEC=1 at import (Python switches them on by hasattr; default = the upstream CUDA path)
static void bind_exl3_dec(py::module_& m)
{
    m.def("exl3_dec_gemv", &exl3_dec_gemv, "exl3_dec_gemv",
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("out"),
          py::arg("scratch"), py::arg("counters"), py::arg("K"), py::arg("mcg") = false);
    m.def("exl3_dec_gemv_multi", &exl3_dec_gemv_multi, "exl3_dec_gemv_multi",
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("out"),
          py::arg("scratch"), py::arg("counters"), py::arg("K"), py::arg("mcg") = false);
    m.def("exl3_dec_gemv_x", &exl3_dec_gemv_x, "exl3_dec_gemv_x",
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("out"),
          py::arg("scratch"), py::arg("counters"), py::arg("K"), py::arg("mcg") = false,
          py::arg("x_job_stride") = 0, py::arg("strips_hint") = 0);
    m.def("exl3_dec_moe", &exl3_dec_moe, "exl3_dec_moe",
          py::arg("x"), py::arg("out"), py::arg("selected"), py::arg("weights"),
          py::arg("gate_trellis"), py::arg("gate_suh"), py::arg("gate_svh"),
          py::arg("up_trellis"), py::arg("up_suh"), py::arg("up_svh"),
          py::arg("down_trellis"), py::arg("down_suh"), py::arg("down_svh"),
          py::arg("act"), py::arg("scratch"), py::arg("counters"),
          py::arg("intermediate"), py::arg("K_gu"), py::arg("K_down"),
          py::arg("accumulate"), py::arg("act_limit") = 0.0, py::arg("mcg") = false);
    m.def("exl3_dec_moe_shared", &exl3_dec_moe_shared, "exl3_dec_moe_shared",
          py::arg("x"), py::arg("out"), py::arg("selected"), py::arg("weights"),
          py::arg("gate_trellis"), py::arg("gate_suh"), py::arg("gate_svh"),
          py::arg("up_trellis"), py::arg("up_suh"), py::arg("up_svh"),
          py::arg("down_trellis"), py::arg("down_suh"), py::arg("down_svh"),
          py::arg("sh_gate_trellis"), py::arg("sh_gate_suh"), py::arg("sh_gate_svh"),
          py::arg("sh_up_trellis"), py::arg("sh_up_suh"), py::arg("sh_up_svh"),
          py::arg("sh_down_trellis"), py::arg("sh_down_suh"), py::arg("sh_down_svh"),
          py::arg("act"), py::arg("scratch"), py::arg("counters"),
          py::arg("intermediate"), py::arg("K_gu"), py::arg("K_down"),
          py::arg("K_sh_gu"), py::arg("K_sh_down"),
          py::arg("accumulate"), py::arg("act_limit") = 0.0, py::arg("mcg") = false);
    m.def("exl3_dec_moe_union", &exl3_dec_moe_union, "exl3_dec_moe_union",
          py::arg("x"), py::arg("out"), py::arg("selected"), py::arg("weights"),
          py::arg("gate_trellis"), py::arg("gate_suh"), py::arg("gate_svh"),
          py::arg("up_trellis"), py::arg("up_suh"), py::arg("up_svh"),
          py::arg("down_trellis"), py::arg("down_suh"), py::arg("down_svh"),
          py::arg("act"), py::arg("down_part"), py::arg("scratch"), py::arg("counters"),
          py::arg("intermediate"), py::arg("K_gu"), py::arg("K_down"),
          py::arg("accumulate"), py::arg("act_limit") = 0.0, py::arg("mcg") = false,
          py::arg("device_table") = false);
    m.def("exl3_dec_gemv_strided", &exl3_dec_gemv_strided, "exl3_dec_gemv_strided",
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("out"),
          py::arg("scratch"), py::arg("counters"), py::arg("K"), py::arg("in_features"),
          py::arg("x_gstride"), py::arg("mcg") = false);
    m.def("exl3_dec_gemv_r", &exl3_dec_gemv_r, "exl3_dec_gemv_r",
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("out"),
          py::arg("scratch"), py::arg("counters"), py::arg("K"), py::arg("mcg") = false);
    m.def("exl3_dec_gemv_r_multi", &exl3_dec_gemv_r_multi, "exl3_dec_gemv_r_multi",
          py::arg("x"), py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("out"),
          py::arg("scratch"), py::arg("counters"), py::arg("K"), py::arg("mcg") = false);
    m.def("exl3_dec_router_norm", &exl3_dec_router_norm, "exl3_dec_router_norm");
    m.def("exl3_dec_router", &exl3_dec_router, "exl3_dec_router");
    m.def("exl3_dec_router_rows", &exl3_dec_router_rows, "exl3_dec_router_rows");
    m.def("exl3_dec_rms_norm", &exl3_dec_rms_norm, "exl3_dec_rms_norm");
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
#if !defined(USE_ROCM)
    m.attr("HAS_DFLASH2") = true;               // dflash2_dynconv, dflash2_selector_walk, dflash2_topk
#else
    m.attr("HAS_DFLASH2") = false;              // dflash2.cu is excluded from ROCm builds
#endif
    m.def("stloader_read", &stloader_read, "stloader_read");
    m.def("stloader_open_file", &stloader_open_file, "stloader_open_file");
    m.def("stloader_close_file", &stloader_close_file, "stloader_close_file");
    py::class_<TensorLoadJob>(m, "TensorLoadJob")
        .def(py::init<std::vector<uintptr_t>, size_t, size_t, uintptr_t, size_t, bool, bool, bool, int>());
    m.def("stloader_deferred_cpu", &stloader_deferred_cpu, py::arg("jobs"));
    m.def("stloader_deferred_cuda", &stloader_deferred_cuda, py::arg("jobs"), py::arg("max_chunk_size"));

    m.def("cuda_host_register", &cuda_host_register, py::arg("ptr"), py::arg("nbytes"), py::arg("flags"));
    m.def("cuda_host_unregister", &cuda_host_unregister, py::arg("ptr"));
    m.def("cuda_host_get_device_pointer", &cuda_host_get_device_pointer, py::arg("ptr"));
    m.def("cuda_device_get_attribute", &cuda_device_get_attribute, py::arg("attr"), py::arg("device"));
    m.def("pinned_cuda_view", &pinned_cuda_view, py::arg("t"), py::arg("device"));

    m.def("hc_mix_supported", &hc_mix_supported, "hc_mix_supported");
    m.def("hc_mix", &hc_mix, "hc_mix");
    m.def("hc_head", &hc_head, "hc_head");
    m.def("hc_mix_norm", &hc_mix_norm, "hc_mix_norm");
    m.def("hc_mix_set_onepass", &hc_mix_set_onepass, "hc_mix_set_onepass");
    m.def("hc_mix_num_chunks", &hc_mix_num_chunks, "hc_mix_num_chunks");
    m.def("hc_apply", &hc_apply, "hc_apply");
    m.def("hc_pf_apply_mix_norm", &hc_pf_apply_mix_norm, "hc_pf_apply_mix_norm");
    m.def("gr_mix", &gr_mix, "gr_mix");
    m.def("gr_mix_q8", &gr_mix_q8, "gr_mix_q8");
    m.def("add_sigmoid_gate_proj", &add_sigmoid_gate_proj, "add_sigmoid_gate_proj");

#if !defined(USE_ROCM)
    m.def("rms_norm", &rms_norm, "rms_norm",
        py::arg("x"), py::arg("w"), py::arg("y"), py::arg("epsilon"),
        py::arg("constant_bias"), py::arg("constant_scale"), py::arg("span_heads"),
        py::arg("add_residual"), py::arg("w_groups") = 1);
#endif
#if !defined(USE_ROCM)
    m.def("rms_norm_res_in", &rms_norm_res_in, "rms_norm_res_in");
    m.def("gated_rms_norm", &gated_rms_norm, "gated_rms_norm");
    m.def("softcap", &softcap, "softcap");

    m.def("routing_ds3_nogroup", &routing_ds3_nogroup, "routing_ds3_nogroup");
    m.def("routing_ds3_nogroup_logits", &routing_ds3_nogroup_logits, "routing_ds3_nogroup_logits");
    m.def("routing_sel_norm", &routing_sel_norm, "routing_sel_norm");
    m.def("moe_split_map", &moe_split_map, "moe_split_map");
    m.def("moe_split_issue", &moe_split_issue, "moe_split_issue");
    m.def("moe_split_collect_add", &moe_split_collect_add, "moe_split_collect_add");
    m.def("dsv4_compress", &dsv4_compress, "dsv4_compress");
    m.def("dsv4_pool_quant_scatter", &dsv4_pool_quant_scatter, "dsv4_pool_quant_scatter");
    m.def("dsv4_ring_append", &dsv4_ring_append, "dsv4_ring_append");
    m.def("dsa_topk", &dsa_topk, "dsa_topk");
    m.def("dsa_topk_tile", &dsa_topk_tile, "dsa_topk_tile");
    m.def("dsa_topk_merge_tiles", &dsa_topk_merge_tiles, "dsa_topk_merge_tiles");
    m.def("ple_gate", &ple_gate, "ple_gate");
    m.def("dflash2_dynconv", &dflash2_dynconv, "dflash2_dynconv");
    m.def("dflash2_selector_walk", &dflash2_selector_walk, "dflash2_selector_walk");
    m.def("dflash2_topk", &dflash2_topk, "dflash2_topk");
    m.def("ple_forward_streams", &ple_forward_streams, "ple_forward_streams");
    m.def("ngram_hash_cpu", &ngram_hash_cpu, "ngram_hash_cpu");
    m.def("ngram_gather_cpu", &ngram_gather_cpu, "ngram_gather_cpu");
    m.def("ngram_dequant", &ngram_dequant, "ngram_dequant");
    m.def("routing_std", &routing_std, "routing_std");
    m.def("routing_std_logits", &routing_std_logits, "routing_std_logits");

    m.def("had_paley", &had_paley, "had_paley");
    m.def("had_paley2", &had_paley2, "had_paley2");

    m.def("pg_init_context", &pg_init_context, "pg_init_context");
    m.def("pg_broadcast", &pg_broadcast, "pg_broadcast");
    m.def("pg_broadcast_ll", &pg_broadcast_ll, "pg_broadcast_ll");
    m.def("pg_barrier", &pg_barrier, "pg_barrier");
    m.def("pg_gather", &pg_gather, "pg_gather");
    m.def("pg_gather_small", &pg_gather_small, "pg_gather_small");
    m.def("pg_all_reduce", &pg_all_reduce, "pg_all_reduce");
    m.def("pg_all_reduce_cpu", &pg_all_reduce_cpu, "pg_all_reduce_cpu");
    m.def("run_cpu_reduce_jobs", &run_cpu_reduce_jobs, "run_cpu_reduce_jobs");
    m.def("end_cpu_reduce_jobs", &end_cpu_reduce_jobs, "end_cpu_reduce_jobs");

    m.def("quantize_tiles", &quantize_tiles, "quantize_tiles");
    m.def("quantize_tiles_scratch", &quantize_tiles_scratch, "quantize_tiles_scratch");
    m.def("test_distribution", &test_distribution, "test_distribution");
    m.def("decode", &decode, "decode");
    m.def("pack_trellis", &pack_trellis, "pack_trellis");
    m.def("unpack_trellis", &unpack_trellis, "unpack_trellis");
    m.def("pack_signs", &pack_signs, "pack_signs");
    m.def("reconstruct", &reconstruct, "reconstruct");
    m.def("reconstruct_had_slice", &reconstruct_had_slice, "reconstruct_had_slice");
#if !defined(USE_ROCM)
    m.def("reconstruct_had_batch", &reconstruct_had_batch, "reconstruct_had_batch");
    m.def("reconstruct_batch", &reconstruct_batch, "reconstruct_batch");
#endif
    m.def("reconstruct_slice", &reconstruct_slice, "reconstruct_slice");
    m.def("had_r_128", &had_r_128, "had_r_128");
    m.def("had_r_128_batch", &had_r_128_batch, "had_r_128_batch");
    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");
    m.def("exl3_gemv", &exl3_gemv, "exl3_gemv");
    m.def("exl3_gemv_supported", &exl3_gemv_supported, "exl3_gemv_supported");
    m.def("exl3_gemv_wmma_family", &exl3_gemv_wmma_family, "exl3_gemv_wmma_family");
    m.def("exl3_gemm_num_kernel_shapes", &exl3_gemm_num_kernel_shapes, "exl3_gemm_num_kernel_shapes");
    m.def("exl3_gemm_shape_compat", &exl3_gemm_shape_compat, "exl3_gemm_shape_compat");
    m.def("g_get_cc", &g_get_cc, "g_get_cc");
    m.def("g_get_num_sms", &g_get_num_sms, "g_get_num_sms");
    m.def("exl3_gemv_int8_max_k", &exl3_gemv_int8_max_k, "exl3_gemv_int8_max_k");
    m.def("exl3_moe_cpu_make_layer", &exl3_moe_cpu_make_layer, "exl3_moe_cpu_make_layer");
    m.def("exl3_moe_cpu_free_layer", &exl3_moe_cpu_free_layer, "exl3_moe_cpu_free_layer");
    m.def("exl3_moe_cpu_forward", &exl3_moe_cpu_forward, "exl3_moe_cpu_forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("exl3_moe_cpu_has_avx2", &exl3_moe_cpu_has_avx2, "exl3_moe_cpu_has_avx2");
    m.def("exl3_moe_flag_write", &exl3_moe_flag_write, "exl3_moe_flag_write");
    m.def("exl3_moe_flag_wait", &exl3_moe_flag_wait, "exl3_moe_flag_wait");
    m.def("moe_unswizzle_trellis", &moe_unswizzle_trellis, "moe_unswizzle_trellis");
    m.def("exl3_moe_cpu_set_memops", &exl3_moe_cpu_set_memops, "exl3_moe_cpu_set_memops");
    m.def("exl3_moe_cpu_set_prof", &exl3_moe_cpu_set_prof, "exl3_moe_cpu_set_prof");
    m.def("exl3_moe_cpu_pool_stress", &exl3_moe_cpu_pool_stress, "exl3_moe_cpu_pool_stress");
    m.def("exl3_moe_cpu_worker_run", &exl3_moe_cpu_worker_run, "exl3_moe_cpu_worker_run",
          py::call_guard<py::gil_scoped_release>());
    m.def("exl3_moe_cpu_has_avx512_bw", &exl3_moe_cpu_has_avx512_bw, "exl3_moe_cpu_has_avx512_bw");
    m.def("exl3_moe_cpu_has_avx512_vnni", &exl3_moe_cpu_has_avx512_vnni, "exl3_moe_cpu_has_avx512_vnni");
    m.def("exl3_moe_cpu_has_avx512_vbmi", &exl3_moe_cpu_has_avx512_vbmi, "exl3_moe_cpu_has_avx512_vbmi");
    m.def("exl3_mgemm", &exl3_mgemm, "exl3_mgemm",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("suh"), py::arg("A_had"), py::arg("svh"),
          py::arg("indices"), py::arg("weights"), py::arg("K"), py::arg("force_shape_idx"), py::arg("mcg"),
          py::arg("mul1"), py::arg("min_index"), py::arg("max_index"), py::arg("force_num_sms"),
          py::arg("num_tokens") = 1, py::arg("size_n_list") = py::none(), py::arg("c_ptrs") = py::none(),
          py::arg("n_stride_list") = py::none(), py::arg("had_src_list") = py::none(), py::arg("num_had_src") = 0);
    m.def("hgemm", &hgemm, "hgemm");
    m.def("hgemm_batched", &hgemm_batched, "hgemm_batched");
    m.def("hgemm_recon", &hgemm_recon, "hgemm_recon");
    m.def("hgemm_f16acc", &hgemm_f16acc, "hgemm_f16acc");
    m.def("hgemm_f16acc_status", &hgemm_f16acc_status, "hgemm_f16acc_status");
    m.def("rope", &rope, "rope");
    m.def("gen_mrope_pos_ids", &gen_mrope_pos_ids, "gen_mrope_pos_ids");
    m.def("silu_mul", &silu_mul, "silu_mul");
    m.def("silu_oai_mul", &silu_oai_mul, "silu_oai_mul");
    m.def("gelu_mul", &gelu_mul, "gelu_mul");
    m.def("relu2_mul", &relu2_mul, "relu2_mul");
    m.def("relu_mul", &relu_mul, "relu_mul");
    m.def("xielu", &xielu, "xielu");
    m.def("add_sigmoid_gate", &add_sigmoid_gate, "add_sigmoid_gate");
    m.def("mul_sigmoid_", &mul_sigmoid_, "mul_sigmoid_");
    m.def("deinterleave_qg", &deinterleave_qg, "deinterleave_qg");
    m.def("mul_sigmoid_broadcast_", &mul_sigmoid_broadcast_, "mul_sigmoid_broadcast_");
    m.def("mul_softplus_broadcast_", &mul_softplus_broadcast_, "mul_softplus_broadcast_");
    m.def("add", &add, "add");

    m.def("gated_delta_net_fused_op", &gated_delta_net_fused_op, "gated_delta_net_fused_op");
    m.def("gated_delta_net_fused_op_2", &gated_delta_net_fused_op_2, "gated_delta_net_fused_op_2");
    m.def("cuda_recurrent_gated_delta_rule", &cuda_recurrent_gated_delta_rule, "cuda_recurrent_gated_delta_rule");
    m.def("mamba2_dt_op", &mamba2_dt_op, "mamba2_dt_op");
    m.def("cuda_recurrent_mamba2", &cuda_recurrent_mamba2, "cuda_recurrent_mamba2");
    m.def("cuda_causal_conv1d_update", &cuda_causal_conv1d_update, "cuda_causal_conv1d_update");
    m.def("gdn_ba_gemv", &gdn_ba_gemv, "gdn_ba_gemv");
    m.def("gdn_lowrank_gemv_f", [](const at::Tensor& x, const at::Tensor& w_t, at::Tensor& y)
        { gdn_lowrank_gemv_f_gr(x, w_t, y, nullptr); }, "gdn_lowrank_gemv_f");
    m.def("kda_gate_op", [](const at::Tensor& qkv, const at::Tensor& b, const at::Tensor& f,
                            const at::Tensor& dt_bias, const at::Tensor& a_log,
                            at::Tensor& mixed_qkv, at::Tensor& beta, at::Tensor& g,
                            float lower_bound, float beta_scale)
        { kda_gate_op_gr(qkv, b, f, dt_bias, a_log, mixed_qkv, beta, g, lower_bound, beta_scale, nullptr); },
        "kda_gate_op");

    py::class_<ConvRewindJob>(m, "ConvRewindJob")
        .def(py::init<uintptr_t, uintptr_t, int, int, int>());
    py::class_<StateRewindJob>(m, "StateRewindJob")
        .def(py::init<uintptr_t, uintptr_t, int64_t>());
    m.def("batched_conv_rewind", &batched_conv_rewind, py::arg("jobs"), py::arg("device_index"));
    m.def("batched_state_rewind", &batched_state_rewind, py::arg("jobs"), py::arg("device_index"));

    m.def("argmax_sample", &argmax_sample, "argmax_sample");
    m.def("gumbel_sample", &gumbel_sample, "gumbel_sample");
    m.def("gumbel_noise_f16", &gumbel_noise_f16, "gumbel_noise_f16");
    m.def("gumbel_noise_f32", &gumbel_noise_f32, "gumbel_noise_f32");
    m.def("gumbel_noise_log", &gumbel_noise_log, "gumbel_noise_log");
    m.def("fused_sampler", &fused_sampler, "fused_sampler");
    m.def("apply_logit_bitmask", &apply_logit_bitmask, "apply_logit_bitmask");
    m.attr("FUSED_SAMPLER_MAX_BLOCKS") = FUSED_SAMPLER_MAX_BLOCKS;
    m.attr("FUSED_SAMPLER_HIST_STRIDE") = FUSED_SAMPLER_HIST_STRIDE;
    m.def("apply_rep_pens", &apply_rep_pens, "apply_rep_pens");
    m.def("dry_penalty", &dry_penalty, "dry_penalty");
    m.def("apply_pres_freq_pens", &apply_pres_freq_pens, "apply_pres_freq_pens");
    m.def("adaptivep_gumbel_noise_f32", &adaptivep_gumbel_noise_f32, "adaptivep_gumbel_noise_f32");

    m.def("cache_rotate", &cache_rotate, "cache_rotate");
    m.def("dspark_write_rows", &dspark_write_rows, "dspark_write_rows");
    m.def("paged_kv_cache_update", &paged_kv_cache_update, "paged_kv_cache_update");

    m.def("partial_strings_match", &partial_strings_match, "partial_strings_match");
    m.def("count_match_tensor", &count_match_tensor, "count_match_tensor");

    m.def("quant_cache_cont", &quant_cache_cont, "quant_cache_cont");
    m.def("dequant_cache_cont", &dequant_cache_cont, "dequant_cache_cont");
    m.def("quant_cache_paged", &quant_cache_paged, "quant_cache_paged");
    m.def("dequant_cache_paged", &dequant_cache_paged, "dequant_cache_paged");
    m.def("dequant_cache_paged_window", &dequant_cache_paged_window, "dequant_cache_paged_window");

    m.def("count_inf_nan", &count_inf_nan, "count_inf_nan");
    m.def("histogram", &histogram, "histogram");

    m.def("blocksparse_mlp_routing", &blocksparse_mlp_routing, "blocksparse_mlp_routing");
    m.def("exl3_moe_max_concurrency", &exl3_moe_max_concurrency, "exl3_moe_max_concurrency");
    m.def("exl3_moe", &exl3_moe, "exl3_moe");
    m.def("exl3_moe_gather", &exl3_moe_gather, "exl3_moe_gather");
    m.def("exl3_moe_grouped", &exl3_moe_grouped, "exl3_moe_grouped");
    m.def("exl3_moe_grouped_supported", &exl3_moe_grouped_supported, "exl3_moe_grouped_supported");
    m.def("exl3_moe_coop", &exl3_moe_coop, "exl3_moe_coop");

    m.def("bighead_attn", &bighead_attn, "bighead_attn");
    m.def("bighead_attn_paged", &bighead_attn_paged, "bighead_attn_paged");
    m.def("bighead_attn_workspace_size", &bighead_attn_workspace_size, "bighead_attn_workspace_size");

#if defined(EXL3_PORT_DEC)
    if (const char* e = getenv("EXL3_PORT_DEC"); e && atoi(e) > 0) bind_exl3_dec(m);
#endif
    #include "libtorch/linear_bc.h"
    #include "libtorch/gated_delta_net_bc.h"
    #include "libtorch/attention_bc.h"
    #include "libtorch/mla_attention_bc.h"
    #include "libtorch/gated_rmsnorm_bc.h"
    #include "libtorch/mlp_bc.h"
    #include "libtorch/blocksparse_mlp_bc.h"
    #include "libtorch/dsv4_compressor_bc.h"
    #include "libtorch/dsv4_attn_bc.h"
    #include "sam_bc.h"
#else
    m.def("had_paley", &had_paley, "had_paley");
    m.def("had_paley2", &had_paley2, "had_paley2");

    m.def("hgemm", [](at::Tensor a, at::Tensor b, at::Tensor c)
          {
              if (hgemm_gil_free(a)) { py::gil_scoped_release nogil; hgemm(a, b, c); }
              else hgemm(a, b, c);
          }, "hgemm");
    m.def("skinny_cat", &skinny_cat, "skinny_cat");
    m.def("skinny_cat_w", &skinny_cat_w, "skinny_cat_w");
    m.def("hgemm_batched", &hgemm_batched, "hgemm_batched");
    m.def("hgemm_recon", [](at::Tensor a, at::Tensor b, at::Tensor c)
          {
              if (hgemm_gil_free(a)) { py::gil_scoped_release nogil; hgemm_recon(a, b, c); }
              else hgemm_recon(a, b, c);
          }, "hgemm_recon");
    m.def("hgemm_f16acc", &hgemm_f16acc, "hgemm_f16acc");
    m.def("hgemm_f16acc_status", &hgemm_f16acc_status, "hgemm_f16acc_status");
    m.def("rope", &rope, "rope");
    m.def("gen_mrope_pos_ids", &gen_mrope_pos_ids, "gen_mrope_pos_ids");
    m.def("reconstruct", &reconstruct, "reconstruct");
    m.def("reconstruct_had_slice", &reconstruct_had_slice, "reconstruct_had_slice");
    m.def("reconstruct_slice", &reconstruct_slice, "reconstruct_slice");
    m.def("had_r_128", &had_r_128, "had_r_128");
    m.def("had_r_128_batch", &had_r_128_batch, "had_r_128_batch");
    m.def("exl3_gemv", &exl3_gemv, "exl3_gemv");
    m.def("exl3_gemv_supported", &exl3_gemv_supported, "exl3_gemv_supported");
    m.def("exl3_gemv_wmma_family", &exl3_gemv_wmma_family, "exl3_gemv_wmma_family");
    m.def("exl3_moe_fused_half", &exl3_moe_fused_half, "exl3_moe_fused_half");
    m.def("exl3_moe_valu_override", &exl3_moe_valu_override, "exl3_moe_valu_override");
    m.def("exl3_moe_silu_mul_limit", &exl3_moe_silu_mul_limit, "exl3_moe_silu_mul_limit");
    m.def("exl3_moe_fused_supported", &exl3_moe_fused_supported, "exl3_moe_fused_supported");
    m.def("exl3_moe_fused_ws_offsets", &exl3_moe_fused_ws_offsets, "exl3_moe_fused_ws_offsets");
    m.def("exl3_moe_fused_grid", &exl3_moe_fused_grid, "exl3_moe_fused_grid");
    m.def("exl3_moe_fused_tile_test", &exl3_moe_fused_tile_test, "exl3_moe_fused_tile_test");
    m.def("exl3_gemv_int8_max_k", &exl3_gemv_int8_max_k, "exl3_gemv_int8_max_k");
    m.def("exl3_moe_gfx12_k3", &exl3_moe_gfx12_k3, "exl3_moe_gfx12_k3");
    bind_exl3_dec(m);
    m.def("fused_gated_rms_norm", &fused_gated_rms_norm, "fused_gated_rms_norm");
    m.def("mul_sigmoid_pad", &mul_sigmoid_pad, "mul_sigmoid_pad");
    m.def("pad_copy", &pad_copy, "pad_copy");
    m.def("fused_silu_mul", &fused_silu_mul, "fused_silu_mul");
    m.def("silu_mul_pitched", &silu_mul_pitched, "silu_mul_pitched");
    m.def("skinny_set_onepass", &skinny_set_onepass, "skinny_set_onepass");
    m.def("kda_gate", &kda_gate, "kda_gate");
    m.def("kda_fb_gate", &kda_fb_gate, "kda_fb_gate");
#if defined(USE_ROCM)
    m.def("mla_absorb_dec", &mla_absorb_dec, "mla_absorb_dec");
    m.def("mla_unfold_dec", &mla_unfold_dec, "mla_unfold_dec");
#endif
    m.def("kda_gb_norm", &kda_gb_norm, "kda_gb_norm");
    m.def("kda_inter_off", &kda_inter_off, "kda_inter_off");
    m.def("exl3_moe_gfx12_k3_prefill", &exl3_moe_gfx12_k3_prefill,
          "exl3_moe_gfx12_k3_prefill");
    m.attr("EXL3_MPW_SHORT_A") = 1;     // (ROCm branch: the attr block near the samplers is CUDA-only) exl3_moe_prefill_wmma accepts A with fewer rows than selected (zero rows)
    m.def("exl3_moe_prefill_wmma", &exl3_moe_prefill_wmma, "exl3_moe_prefill_wmma",
          py::arg("A"), py::arg("output"), py::arg("selected"), py::arg("weights"),
          py::arg("order"), py::arg("expert_count"),
          py::arg("gate_trellis"), py::arg("gate_suh"), py::arg("gate_svh"),
          py::arg("up_trellis"), py::arg("up_suh"), py::arg("up_svh"),
          py::arg("down_trellis"), py::arg("down_suh"), py::arg("down_svh"),
          py::arg("K_gu"), py::arg("K_down"), py::arg("gu_had"), py::arg("gu_out"),
          py::arg("down_out"), py::arg("expert_offsets"), py::arg("inverse_order"),
          py::arg("tiles"), py::arg("tile_count"),
          py::arg("act_limit") = 0.0, py::arg("mcg") = false, py::arg("addend") = py::none());
    m.def("exl3_moe_prefill_wmma_mixed", &exl3_moe_prefill_wmma_mixed, "exl3_moe_prefill_wmma_mixed");
    m.def("exl3_moe_prefill_gemm_test", &exl3_moe_prefill_gemm_test, "exl3_moe_prefill_gemm_test");
    m.def("exl3_dense_mpw2", &exl3_dense_mpw2, "exl3_dense_mpw2");
    m.def("routing_std_gfx12_bsz1", &routing_std_gfx12_bsz1,
          "routing_std_gfx12_bsz1");
    m.def("dsa_topk_gfx12", &dsa_topk_gfx12, "dsa_topk_gfx12",
          py::arg("scores"), py::arg("indices"), py::arg("k") = 512);
    m.def("dsa_topk_par", &dsa_topk_par, "dsa_topk_par",
          py::arg("scores"), py::arg("indices"), py::arg("k") = 512, py::arg("nb") = 0);
    m.def("dsa_topk_gfx12_v2", &dsa_topk_gfx12_v2, "dsa_topk_gfx12_v2",
          py::arg("scores"), py::arg("indices"), py::arg("k") = 512, py::arg("threads") = 512);
    m.def("pack_trellis", &pack_trellis, "pack_trellis");
    m.def("unpack_trellis", &unpack_trellis, "unpack_trellis");
    m.def("pack_signs", &pack_signs, "pack_signs");
    // quant2: the EXL3 quantizer on ROCm (glm-int only bound it in the CUDA branch)
    m.def("quantize_tiles", &quantize_tiles, "quantize_tiles");
    m.def("quantize_tiles_scratch", &quantize_tiles_scratch, "quantize_tiles_scratch");
    m.def("test_distribution", &test_distribution, "test_distribution");
    m.def("decode", &decode, "decode");
    // CPU expert offload: registered-SHM handoff with kernel-wait flags. The moe_split_*/
    // fused-exl3_moe tiers are NOT ported; their Python call sites are gated off on HIP.
    m.def("exl3_moe_cpu_make_layer", &exl3_moe_cpu_make_layer, "exl3_moe_cpu_make_layer");
    m.def("exl3_moe_cpu_free_layer", &exl3_moe_cpu_free_layer, "exl3_moe_cpu_free_layer");
    m.def("exl3_moe_cpu_forward", &exl3_moe_cpu_forward, "exl3_moe_cpu_forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("exl3_moe_cpu_has_avx2", &exl3_moe_cpu_has_avx2, "exl3_moe_cpu_has_avx2");
    m.def("exl3_moe_flag_write", &exl3_moe_flag_write, "exl3_moe_flag_write");
    m.def("exl3_moe_flag_wait", &exl3_moe_flag_wait, "exl3_moe_flag_wait");
    m.def("exl3_moe_cpu_set_memops", &exl3_moe_cpu_set_memops, "exl3_moe_cpu_set_memops");
    m.def("exl3_moe_cpu_set_prof", &exl3_moe_cpu_set_prof, "exl3_moe_cpu_set_prof");
    m.def("exl3_moe_cpu_worker_run", &exl3_moe_cpu_worker_run, "exl3_moe_cpu_worker_run",
          py::call_guard<py::gil_scoped_release>());
    m.def("exl3_moe_cpu_has_avx512_vnni", &exl3_moe_cpu_has_avx512_vnni, "exl3_moe_cpu_has_avx512_vnni");
    m.def("exl3_moe_cpu_has_avx512_vbmi", &exl3_moe_cpu_has_avx512_vbmi, "exl3_moe_cpu_has_avx512_vbmi");

    m.def("cuda_recurrent_gated_delta_rule", &cuda_recurrent_gated_delta_rule, "cuda_recurrent_gated_delta_rule");
    m.def("cuda_recurrent_mamba2", &cuda_recurrent_mamba2, "cuda_recurrent_mamba2");
    m.def("cuda_causal_conv1d_update", &cuda_causal_conv1d_update, "cuda_causal_conv1d_update");
    m.def("gated_delta_net_fused_op", &gated_delta_net_fused_op, "gated_delta_net_fused_op");
    m.def("gated_delta_net_fused_op_2", &gated_delta_net_fused_op_2, "gated_delta_net_fused_op_2");
    m.def("mamba2_dt_op", &mamba2_dt_op, "mamba2_dt_op");
    m.def("gdn_ba_gemv", &gdn_ba_gemv, "gdn_ba_gemv");

    py::class_<ConvRewindJob>(m, "ConvRewindJob")
        .def(py::init<uintptr_t, uintptr_t, int, int, int>());
    py::class_<StateRewindJob>(m, "StateRewindJob")
        .def(py::init<uintptr_t, uintptr_t, int64_t>());
    m.def("batched_conv_rewind", &batched_conv_rewind, py::arg("jobs"), py::arg("device_index"));
    m.def("batched_state_rewind", &batched_state_rewind, py::arg("jobs"), py::arg("device_index"));
    m.def("dspark_write_rows", &dspark_write_rows, "dspark_write_rows");

    m.def("argmax_sample", &argmax_sample, "argmax_sample");
    m.def("gumbel_sample", &gumbel_sample, "gumbel_sample");
    m.def("gumbel_noise_f16", &gumbel_noise_f16, "gumbel_noise_f16");
    m.def("gumbel_noise_f32", &gumbel_noise_f32, "gumbel_noise_f32");
    m.def("gumbel_noise_log", &gumbel_noise_log, "gumbel_noise_log");
    m.def("apply_rep_pens", &apply_rep_pens, "apply_rep_pens");
    m.def("apply_pres_freq_pens", &apply_pres_freq_pens, "apply_pres_freq_pens");
    m.def("adaptivep_gumbel_noise_f32", &adaptivep_gumbel_noise_f32, "adaptivep_gumbel_noise_f32");
    m.def("dry_penalty", &dry_penalty, "dry_penalty");

    m.def("cache_rotate", &cache_rotate, "cache_rotate");
    m.def("paged_kv_cache_update", &paged_kv_cache_update, "paged_kv_cache_update");

    m.def("partial_strings_match", &partial_strings_match, "partial_strings_match");
    m.def("count_match_tensor", &count_match_tensor, "count_match_tensor");
    m.def("ngram_hash_cpu", &ngram_hash_cpu, "ngram_hash_cpu");
    m.def("ngram_gather_cpu", &ngram_gather_cpu, "ngram_gather_cpu");
    m.def("ngram_dequant", &ngram_dequant, "ngram_dequant");
#endif

#if !defined(USE_ROCM)
    // Foreign-quant kernels (foreign_quant/*.cu): CUDA builds only (the ROCm build does not compile them).
    // No env switch: they are additive (nothing upstream is replaced); Python falls back to torch when the ext
    // predates them.
    m.def("fq_dequant_q8_0", &fq_dequant_q8_0, "fq_dequant_q8_0",
          py::arg("packed"), py::arg("out"));
    m.def("fq_dequant_iq2_xxs", &fq_dequant_iq2_xxs, "fq_dequant_iq2_xxs",
          py::arg("packed"), py::arg("out"));
    m.def("fq_dequant_nvfp4", &fq_dequant_nvfp4, "fq_dequant_nvfp4",
          py::arg("packed"), py::arg("scales"), py::arg("global_scale"), py::arg("out"));
    m.def("fq_dequant_q4_k", &fq_dequant_q4_k, "fq_dequant_q4_k",
          py::arg("packed"), py::arg("out"));
    m.def("fq_dequant_q5_k", &fq_dequant_q5_k, "fq_dequant_q5_k",
          py::arg("packed"), py::arg("out"));
    m.def("fq_dequant_q6_k", &fq_dequant_q6_k, "fq_dequant_q6_k",
          py::arg("packed"), py::arg("out"));
    m.def("fq_dequant_iq4_xs", &fq_dequant_iq4_xs, "fq_dequant_iq4_xs",
          py::arg("packed"), py::arg("out"));
    m.def("fq_gemm_q8_0", &fq_gemm_q8_0, "fq_gemm_q8_0",
          py::arg("x"), py::arg("packed"), py::arg("out"));
    m.def("fq_gemm_iq2_xxs", &fq_gemm_iq2_xxs, "fq_gemm_iq2_xxs",
          py::arg("x"), py::arg("packed"), py::arg("out"));
    m.def("fq_gemm_nvfp4", &fq_gemm_nvfp4, "fq_gemm_nvfp4",
          py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("global_scale"), py::arg("out"));
    m.def("fq_gemv_q8_0", &fq_gemv_q8_0, "fq_gemv_q8_0",
          py::arg("x"), py::arg("packed"), py::arg("out"));
    m.def("fq_gemv_iq2_xxs", &fq_gemv_iq2_xxs, "fq_gemv_iq2_xxs",
          py::arg("x"), py::arg("packed"), py::arg("out"));
    m.def("fq_gemv_nvfp4", &fq_gemv_nvfp4, "fq_gemv_nvfp4",
          py::arg("x"), py::arg("packed"), py::arg("scales"), py::arg("global_scale"), py::arg("out"),
          py::arg("pair") = true);
    m.def("fq_moe_iq2xxs_grouped", &fq_moe_iq2xxs_grouped, "fq_moe_iq2xxs_grouped",
          py::arg("h"), py::arg("sel"), py::arg("wts"),
          py::arg("up_ptrs"), py::arg("down_ptrs"),
          py::arg("up_stride"), py::arg("down_stride"),
          py::arg("partial"), py::arg("out"));
    m.def("fq_mlp_shared_iq2xxs", &fq_mlp_shared_iq2xxs, "fq_mlp_shared_iq2xxs",
          py::arg("x"), py::arg("up_packed"), py::arg("h"));
    m.def("fq_q8_scratch", &fq_q8_scratch, "fq_q8_scratch",
          py::arg("dev"), py::arg("M"), py::arg("K"));
    m.def("fq_quant_q8_1_rows", &fq_quant_q8_1_rows, "fq_quant_q8_1_rows",
          py::arg("x"), py::arg("qb"));
    m.def("fq_gemv_iq2_xxs_q8", &fq_gemv_iq2_xxs_q8, "fq_gemv_iq2_xxs_q8",
          py::arg("qb"), py::arg("packed"), py::arg("out"));
    m.def("fq_gemv_nvfp4_q8", &fq_gemv_nvfp4_q8, "fq_gemv_nvfp4_q8",
          py::arg("qb"), py::arg("packed"), py::arg("scales"), py::arg("global_scale"), py::arg("out"));
    m.def("fq_gemv_q8_0_q8", &fq_gemv_q8_0_q8, "fq_gemv_q8_0_q8",
          py::arg("qb"), py::arg("packed"), py::arg("out"));
    m.def("fq_gemv_q4_k_q8", &fq_gemv_q4_k_q8, "fq_gemv_q4_k_q8",
          py::arg("qb"), py::arg("packed"), py::arg("out"));
    m.def("fq_gemv_q5_k_q8", &fq_gemv_q5_k_q8, "fq_gemv_q5_k_q8",
          py::arg("qb"), py::arg("packed"), py::arg("out"));
    m.def("fq_gemv_q6_k_q8", &fq_gemv_q6_k_q8, "fq_gemv_q6_k_q8",
          py::arg("qb"), py::arg("packed"), py::arg("out"));
    m.def("fq_gemv_iq4_xs_q8", &fq_gemv_iq4_xs_q8, "fq_gemv_iq4_xs_q8",
          py::arg("qb"), py::arg("packed"), py::arg("out"));
    m.def("fq_moe_q4k_grouped_q8", &fq_moe_q4k_grouped_q8, "fq_moe_q4k_grouped_q8",
          py::arg("h"), py::arg("sel"), py::arg("wts"),
          py::arg("gate_ptrs"), py::arg("up_ptrs"), py::arg("down_ptrs"),
          py::arg("gate_stride"), py::arg("up_stride"), py::arg("down_stride"),
          py::arg("partial"), py::arg("out"));
    m.def("fq_moe_q5k_grouped_q8", &fq_moe_q5k_grouped_q8, "fq_moe_q5k_grouped_q8",
          py::arg("h"), py::arg("sel"), py::arg("wts"),
          py::arg("gate_ptrs"), py::arg("up_ptrs"), py::arg("down_ptrs"),
          py::arg("gate_stride"), py::arg("up_stride"), py::arg("down_stride"),
          py::arg("partial"), py::arg("out"));
    m.def("fq_moe_q6k_grouped_q8", &fq_moe_q6k_grouped_q8, "fq_moe_q6k_grouped_q8",
          py::arg("h"), py::arg("sel"), py::arg("wts"),
          py::arg("gate_ptrs"), py::arg("up_ptrs"), py::arg("down_ptrs"),
          py::arg("gate_stride"), py::arg("up_stride"), py::arg("down_stride"),
          py::arg("partial"), py::arg("out"));
    m.def("fq_moe_iq4xs_grouped_q8", &fq_moe_iq4xs_grouped_q8, "fq_moe_iq4xs_grouped_q8",
          py::arg("h"), py::arg("sel"), py::arg("wts"),
          py::arg("gate_ptrs"), py::arg("up_ptrs"), py::arg("down_ptrs"),
          py::arg("gate_stride"), py::arg("up_stride"), py::arg("down_stride"),
          py::arg("partial"), py::arg("out"));
#endif
}
