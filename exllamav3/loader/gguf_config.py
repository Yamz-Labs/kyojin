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
}

# ggml type id -> ForeignLinear format
GGML_FMT = {
    2: "q4_0", 6: "q5_0", 8: "q8_0", 12: "q4_k", 13: "q5_k", 14: "q6_k", 18: "iq3_xxs", 16: "iq2_xxs",
    20: "iq4_nl", 21: "iq3_s", 23: "iq4_xs", 39: "mxfp4", 40: "nvfp4_gguf", 0: "fp", 1: "fp",
}


def hf_config(g: GGUFCollection) -> dict:
    m = g.meta
    arch = m["general.architecture"]
    if arch not in GGUF_ARCHS:
        raise ValueError(f"GGUF architecture {arch!r} has no standard-class mapping (have: {sorted(GGUF_ARCHS)})")
    p = arch + "."
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
