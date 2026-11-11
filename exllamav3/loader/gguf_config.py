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
}

# ggml type id -> ForeignLinear format
GGML_FMT = {
    2: "q4_0", 6: "q5_0", 8: "q8_0", 12: "q4_k", 13: "q5_k", 14: "q6_k", 18: "iq3_xxs", 16: "iq2_xxs",
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
