"""Plain GGUF -> HF-style config for the standard dense architectures (llama, qwen3).

The GGUF metadata is the source of truth: write_config_dir() emits the config.json that the engine Config classes read,
next to a symlink to the GGUF itself (so transformers can build the tokenizer from it with gguf_file=).
"""
from __future__ import annotations
import json
import os

from .gguf_collection import GGUFCollection

# gguf general.architecture -> (HF architectures entry, HF model_type)
GGUF_ARCHS = {
    "llama": ("GGUFDenseForCausalLM", "llama"),
    "qwen3": ("GGUFDenseForCausalLM", "qwen3"),
    # unsloth / llama.cpp Qwen3.5-style hybrid MoE (GDN + attention, 256 experts + shared expert), served by the
    # MiniCPM-V-4.7 text class (same block family, GGUF-backed)
    "qwen35moe": ("MiniCPMV4_7ForConditionalGeneration", "qwen3_5_moe"),
    # dense sibling (Qwen3.6-27B class): same GDN + attention blocks, plain gated MLP per layer
    "qwen35": ("MiniCPMV4_7ForConditionalGeneration", "qwen3_5"),
    # Qwen3.8-Flash-Next family: GDN + QSA attention, gated-residual hyper-connections, PLE n-gram layer, 512-expert MoE
    "qwen4exp": ("Qwen4ExpForConditionalGeneration", "qwen4_exp"),
}

# ggml type id -> ForeignLinear format
GGML_FMT = {
    2: "q4_0", 6: "q5_0", 7: "q5_1", 8: "q8_0", 12: "q4_k", 13: "q5_k", 14: "q6_k", 18: "iq3_xxs", 16: "iq2_xxs",
    20: "iq4_nl", 21: "iq3_s", 23: "iq4_xs", 11: "q3_k", 10: "q2_k", 17: "iq2_xs", 22: "iq2_s", 39: "mxfp4", 40: "nvfp4_gguf", 0: "fp", 1: "fp",
}


def hf_config(g: GGUFCollection) -> dict:
    m = g.meta
    arch = m["general.architecture"]
    if arch not in GGUF_ARCHS:
        raise ValueError(f"GGUF architecture {arch!r} has no standard-class mapping (have: {sorted(GGUF_ARCHS)})")
    p = arch + "."
    if arch in ("qwen35moe", "qwen35"):
        return _hf_config_qwen35moe(g, m, arch)
    if arch == "qwen4exp":
        return _hf_config_qwen4exp(g, m)
    ne = m[p + "embedding_length"]
    nq = m[p + "attention.head_count"]
    cfg = {
        "architectures": [GGUF_ARCHS[arch][0]],
        "model_type": GGUF_ARCHS[arch][1],
        "gguf_arch": arch,
        "hidden_size": ne,
        "num_attention_heads": nq,
        "num_key_value_heads": m.get(p + "attention.head_count_kv", nq),
        "num_hidden_layers": m[p + "block_count"],
        "intermediate_size": m[p + "feed_forward_length"],
        "rms_norm_eps": m[p + "attention.layer_norm_rms_epsilon"],
        "max_position_embeddings": m[p + "context_length"],
        "rope_theta": m.get(p + "rope.freq_base", 10000.0),
        "hidden_act": "silu",
        "head_dim": m.get(p + "attention.key_length", ne // nq),
        "vocab_size": g.infos["token_embd.weight"][1][1],
        "tie_word_embeddings": "output.weight" not in g.infos,
    }
    if p + "rope.scaling.type" in m and m[p + "rope.scaling.type"] != "none":
        raise ValueError(f"GGUF rope scaling {m[p + 'rope.scaling.type']!r} is not mapped yet")
    return cfg


def _hf_config_qwen35moe(g: GGUFCollection, m: dict, arch: str = "qwen35moe") -> dict:
    p = arch + "."
    dense = arch == "qwen35"
    ssm_heads, ssm_groups, ssm_dim = m[p + "ssm.time_step_rank"], m[p + "ssm.group_count"], m[p + "ssm.state_size"]
    tc = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": m[p + "embedding_length"],
        "num_attention_heads": m[p + "attention.head_count"],
        "num_key_value_heads": m[p + "attention.head_count_kv"],
        "head_dim": m[p + "attention.key_length"],
        "num_hidden_layers": m[p + "block_count"],
        "full_attention_interval": m[p + "full_attention_interval"],
        "linear_conv_kernel_dim": m[p + "ssm.conv_kernel"],
        "linear_num_key_heads": ssm_groups,
        "linear_num_value_heads": ssm_heads,
        "linear_key_head_dim": ssm_dim,
        "linear_value_head_dim": m[p + "ssm.inner_size"] // ssm_heads,
        "hidden_act": "silu",
        **({"intermediate_size": m[p + "feed_forward_length"]} if dense else {
            "moe_intermediate_size": m[p + "expert_feed_forward_length"],
            "num_experts": m[p + "expert_count"],
            "num_experts_per_tok": m[p + "expert_used_count"],
            "shared_expert_intermediate_size": m[p + "expert_shared_feed_forward_length"]}),
        "rms_norm_eps": m[p + "attention.layer_norm_rms_epsilon"],
        "vocab_size": g.infos["token_embd.weight"][1][1],
        "max_position_embeddings": m[p + "context_length"],
        "rope_theta": m[p + "rope.freq_base"],
        "partial_rotary_factor": m[p + "rope.dimension_count"] / m[p + "attention.key_length"],
    }
    return {"architectures": ["MiniCPMV4_7ForConditionalGeneration"], "model_type": "qwen3_5_moe",
            "gguf_arch": arch, "tie_word_embeddings": "output.weight" not in g.infos, "text_config": tc}


def _hf_config_qwen4exp(g: GGUFCollection, m: dict) -> dict:
    p = "qwen4exp."
    L = m[p + "block_count"]
    fai = m[p + "full_attention_interval"]
    nh_ssm, ng_ssm = m[p + "ssm.time_step_rank"], m[p + "ssm.group_count"]
    hd = m[p + "attention.key_length"]
    heads_ple = len(m[p + "ple.head_offsets"])
    cr = [int(c) for c in m.get(p + "attention.compress_ratios", [4])]
    tc = {
        "model_type": "qwen4_exp_text",
        "hidden_size": m[p + "embedding_length"],
        "num_hidden_layers": L,
        "num_attention_heads": m[p + "attention.head_count"],
        "num_key_value_heads": m[p + "attention.head_count_kv"],
        "head_dim": hd,
        "full_attention_interval": fai,
        "layer_types": ["full_attention" if (i + 1) % fai == 0 else "linear_attention" for i in range(L)],
        "linear_conv_kernel_dim": m[p + "ssm.conv_kernel"],
        "linear_num_key_heads": ng_ssm,
        "linear_num_value_heads": nh_ssm,
        "linear_key_head_dim": m[p + "ssm.state_size"],
        "linear_value_head_dim": m[p + "ssm.inner_size"] // nh_ssm,
        "num_experts": m[p + "expert_count"],
        "num_experts_per_tok": m[p + "expert_used_count"],
        "moe_intermediate_size": m[p + "expert_feed_forward_length"],
        "shared_expert_intermediate_size": m[p + "expert_shared_feed_forward_length"],
        "hidden_act": "silu",
        "rms_norm_eps": m[p + "attention.layer_norm_rms_epsilon"],
        "vocab_size": g.infos["token_embd.weight"][1][1],
        "max_position_embeddings": m[p + "context_length"],
        "rope_parameters": {"rope_type": "default", "rope_theta": m[p + "rope.freq_base"],
                            "partial_rotary_factor": m[p + "rope.dimension_count"] / hd,
                            "mrope_section": [int(v) for v in m[p + "rope.dimension_sections"]][:3], "mrope_interleaved": True},
        "partial_rotary_factor": m[p + "rope.dimension_count"] / hd,
        "hc_count": m[p + "hyper_connection.count"],
        "hc_lowrank": m[p + "hyper_connection.low_rank"],
        "indexer_n_heads": m[p + "attention.indexer.head_count"],
        "indexer_kv_heads": 1,
        "indexer_head_dim": m[p + "attention.indexer.key_length"],
        "indexer_budget": m[p + "attention.indexer.top_k"],
        "indexer_compress_ratio": max(cr) if cr else 4,
        "output_gate_type": "sigmoid",
        "ple_layer_ids": [int(v) + 1 for v in m[p + "ple.layers"]],     # GGUF 0-based, engine 1-based
        "ple_embed_dim": m[p + "embedding_length_per_layer_input"] * heads_ple,
        "ple_conv_kernel_size": m[p + "ple.conv_kernel"],
        "ngram_size": m[p + "ple.ngram_size"],
        "heads_per_ngram": m[p + "ple.heads_per_ngram"],
        "eos_token_id": m[p + "ple.eos_token_id"],
        "bos_token_id": m.get("tokenizer.ggml.bos_token_id", m[p + "ple.eos_token_id"]),
        "mtp_num_hidden_layers": 0,
        "tie_word_embeddings": "output.weight" not in g.infos,
    }
    return {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "gguf_arch": "qwen4exp",
            "image_token_id": m.get(p + "ple.image_token_id"), "language_model_only": True,
            "tie_word_embeddings": tc["tie_word_embeddings"], "text_config": tc}


def name_map_qwen4exp(cfg: dict) -> dict:
    """engine key -> gguf tensor name for qwen4exp (tensor keys incl. .weight; experts as '<gguf 3d>.e<idx>.weight' slices)."""
    tc = cfg["text_config"]
    P = "model.language_model"
    nm = {"lm_head.weight": "output.weight", f"{P}.embed_tokens.weight": "token_embd.weight",
          f"{P}.hyper_connection_mixer.input_mix_weight_down.weight": "output_hc_down.weight",
          f"{P}.hyper_connection_mixer.input_mix_weight_up.weight": "output_hc_up.weight",
          f"{P}.hyper_connection_mixer.hc_norm.weight": "output_hc_norm.weight"}
    for i, lt in enumerate(tc["layer_types"]):
        L, G = f"{P}.layers.{i}", f"blk.{i}"
        for site, gs in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            nm.update({f"{L}.{site}.input_mix_weight_down.weight": f"{G}.{gs}_down.weight",
                       f"{L}.{site}.input_mix_weight_up.weight": f"{G}.{gs}_up.weight",
                       f"{L}.{site}.block_inject_weight.weight": f"{G}.{gs}_inject.weight",
                       f"{L}.{site}.hc_norm.weight": f"{G}.{gs}_norm.weight"})
        if lt == "linear_attention":
            A = f"{L}.linear_attn"
            nm.update({f"{A}.in_proj_qkv.weight": f"{G}.attn_qkv.weight", f"{A}.in_proj_z.weight": f"{G}.attn_gate.weight",
                       f"{A}.in_proj_b.weight": f"{G}.ssm_beta.weight", f"{A}.in_proj_a.weight": f"{G}.ssm_alpha.weight",
                       f"{A}.out_proj.weight": f"{G}.ssm_out.weight", f"{A}.conv1d.weight": f"{G}.ssm_conv1d.weight",
                       f"{A}.A_log": f"{G}.ssm_a", f"{A}.dt_bias": f"{G}.ssm_dt.bias", f"{A}.norm.weight": f"{G}.ssm_norm.weight"})
        else:
            A = f"{L}.self_attn"
            nm.update({f"{A}.q_proj.weight": f"{G}.attn_q.weight", f"{A}.k_proj.weight": f"{G}.attn_k.weight",
                       f"{A}.v_proj.weight": f"{G}.attn_v.weight", f"{A}.o_proj.weight": f"{G}.attn_output.weight",
                       f"{A}.q_norm.weight": f"{G}.attn_q_norm.weight", f"{A}.k_norm.weight": f"{G}.attn_k_norm.weight",
                       f"{A}.indexer.q_layernorm.weight": f"{G}.indexer.q_norm.weight",
                       f"{A}.indexer.k_layernorm.weight": f"{G}.indexer.k_norm.weight"})
        M = f"{L}.mlp"
        nm.update({f"{M}.gate.weight": f"{G}.ffn_gate_inp.weight", f"{M}.shared_expert_gate.weight": f"{G}.ffn_gate_inp_shexp.weight",
                   f"{M}.shared_expert.gate_proj.weight": f"{G}.ffn_gate_shexp.weight",
                   f"{M}.shared_expert.up_proj.weight": f"{G}.ffn_up_shexp.weight",
                   f"{M}.shared_expert.down_proj.weight": f"{G}.ffn_down_shexp.weight"})
        for e in range(tc["num_experts"]):
            for proj, gp in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")):
                nm[f"{M}.experts.{e}.{proj}.weight"] = f"{G}.{gp}.e{e}.weight"
        if (i + 1) in tc["ple_layer_ids"]:
            Q = f"{L}.ple"
            nm.update({f"{Q}.key_proj.weight": f"{G}.ple_key.weight", f"{Q}.value_proj.weight": f"{G}.ple_value.weight",
                       f"{Q}.conv1d.weight": f"{G}.ple_conv1d.weight", f"{Q}.norm_conv.weight": f"{G}.ple_norm_conv.weight",
                       f"{Q}.norm_key.weight": f"{G}.ple_norm_key.weight", f"{Q}.norm_query.weight": f"{G}.ple_norm_query.weight",
                       f"{Q}.ple_embedding.ngram_embedding.weight": "per_layer_token_embd.weight"})
    return nm


def name_map(cfg: dict) -> dict:
    """engine key -> gguf tensor name (tensor keys, with .weight)."""
    nm = {"model.norm.weight": "output_norm.weight", "model.embed_tokens.weight": "token_embd.weight"}
    if not cfg["tie_word_embeddings"]:
        nm["lm_head.weight"] = "output.weight"
    for i in range(cfg["num_hidden_layers"]):
        L, G = f"model.layers.{i}", f"blk.{i}"
        nm.update({
            f"{L}.input_layernorm.weight": f"{G}.attn_norm.weight",
            f"{L}.post_attention_layernorm.weight": f"{G}.ffn_norm.weight",
            f"{L}.self_attn.q_proj.weight": f"{G}.attn_q.weight",
            f"{L}.self_attn.k_proj.weight": f"{G}.attn_k.weight",
            f"{L}.self_attn.v_proj.weight": f"{G}.attn_v.weight",
            f"{L}.self_attn.o_proj.weight": f"{G}.attn_output.weight",
            f"{L}.self_attn.q_norm.weight": f"{G}.attn_q_norm.weight",
            f"{L}.self_attn.k_norm.weight": f"{G}.attn_k_norm.weight",
            f"{L}.mlp.gate_proj.weight": f"{G}.ffn_gate.weight",
            f"{L}.mlp.up_proj.weight": f"{G}.ffn_up.weight",
            f"{L}.mlp.down_proj.weight": f"{G}.ffn_down.weight",
        })
    return nm


def write_config_dir(gguf_path: str, outdir: str) -> str:
    """Create outdir/{config.json, model.gguf -> gguf_path}; return outdir."""
    from .gguf_tokenizer import write_tokenizer_files
    g = GGUFCollection(gguf_path)
    os.makedirs(outdir, exist_ok=True)
    try:
        cfg = hf_config(g)
        write_tokenizer_files(g, outdir)
    finally:
        g.close()
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf8") as f:
        json.dump(cfg, f, indent=1)
    link = os.path.join(outdir, "model.gguf")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(gguf_path), link)
    return outdir
