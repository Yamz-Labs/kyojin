"""Blocks around modules/mla_attn.py: the MLA GGUF adapter and its modifiers. Predicates + configs only (Hy4 steps 3-4):
the builds come later. Selection is by keys / tensor names checked on the hy_v4_flash and glm5-next headers."""
from __future__ import annotations
from ..gguf_registry import Block, register

_MLA = frozenset({"attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight", "attn_q.weight", "attn_kv_a_mqa.weight",
                  "attn_kv_a_norm.weight", "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight"})


def is_mla(ctx, i):
    """GGUF MLA (llama.cpp deepseek2 layout): kv_lora_rank key + compressed kv + per-head absorbed k_b / v_b"""
    return ctx.key("attention.kv_lora_rank") is not None and all(
        ctx.has(i, s) for s in ("attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight"))


def _mla_layer(ctx, i):
    nh = ctx.per_layer("attention.head_count")
    kb = ctx.dims(f"blk.{i}.attn_k_b.weight")
    return {"n_head": nh[i], "q_lora_rank": ctx.key("attention.q_lora_rank"), "kv_lora_rank": ctx.key("attention.kv_lora_rank"),
            "qk_head_dim": ctx.key("attention.key_length_mla"), "v_head_dim": ctx.key("attention.value_length_mla"),
            "rope_dim": ctx.key("rope.dimension_count"), "k_b_dims": kb, "v_b_dims": ctx.dims(f"blk.{i}.attn_v_b.weight"),
            "kv_a_dims": ctx.dims(f"blk.{i}.attn_kv_a_mqa.weight"), "kq_scale_dim": ctx.key("attention.key_length_mla"),
            "rope_theta": ctx.key("rope.freq_base_swa") if _swa(ctx, i) else ctx.key("rope.freq_base")}


register(Block("mla", "attn", select=is_mla, config=_mla_layer, claims=_MLA, build=None, step="step 3", priority=1))


def _swa(ctx, i):
    pat = ctx.key("attention.sliding_window_pattern")
    return bool(pat[i]) if isinstance(pat, list) and i < len(pat) else False


def _gguf(config):
    return getattr(config, "gguf", config)      # the GGUFCollection, from a model config or passed directly


def _build_hy4_attn(config, plan, i, key=None):
    """Hy4 attention layer (CPU fp32 maths, loader/mla_hy4.py): MLA + sinks + window + sigmoid gate before o_proj + post norm.
    Selected through the tensors attn_gate / attn_sinks (+ the window keys), never a model name; other MLA GGUFs keep build=None."""
    from ..mla_hy4 import Hy4MLA
    return Hy4MLA(_gguf(config), i)


register(Block(
    "mla_window_sinks", "attn_mod", step="step 3",
    select=lambda ctx, i: is_mla(ctx, i) and (ctx.has(i, "attn_sinks.weight") or (_swa(ctx, i) and ctx.key("attention.sliding_window"))),
    config=lambda ctx, i: {"swa": _swa(ctx, i), "window": ctx.key("attention.sliding_window") if _swa(ctx, i) else 0,
                           "rope_theta": ctx.key("rope.freq_base_swa") if _swa(ctx, i) else ctx.key("rope.freq_base"),
                           "sinks": ctx.has(i, "attn_sinks.weight")},
    claims=frozenset({"attn_sinks.weight"}),
    build=lambda config, plan, i, key=None: _build_hy4_attn(config, plan, i, key) if plan is None or plan.ctx.has(i, "attn_gate.weight") else None))

register(Block(
    "mla_output_gate", "attn_mod", step="step 3",
    select=lambda ctx, i: is_mla(ctx, i) and ctx.has(i, "attn_gate.weight"),
    config=lambda ctx, i: {"gate": "sigmoid_full", "order": "before_o_proj", "dims": ctx.dims(f"blk.{i}.attn_gate.weight"),
                           "post_norm": ctx.has(i, "post_attention_norm.weight")},
    claims=frozenset({"attn_gate.weight"}), build=_build_hy4_attn))


def _indexer_layer(ctx, i):
    types = ctx.key("attention.indexer.types")
    if ctx.has(i, "indexer_compressor_value.weight"):
        variant = "kpool_value"          # hy_v4_flash: pooled projection of the key, gate off the key, norm after the pool
    elif ctx.has(i, "indexer_compressor_gate.weight"):
        variant = "kpool_gate"           # glm5-next
    else:
        variant = "dsa"                  # deepseek v3.2 / v4 indexer, no pooling
    return {"variant": variant, "k_norm": "layernorm" if ctx.has(i, "indexer.k_norm.bias") else "rms_weight",
            "n_head": ctx.key("attention.indexer.head_count"), "head_dim": ctx.key("attention.indexer.key_length"),
            "top_k": ctx.key("attention.indexer.top_k"), "kpool": ctx.key("attention.indexer.kpool"),
            "select_tail": ctx.key("attention.indexer.kpool_select_tail"),
            "typed": bool(types[i]) if isinstance(types, list) and i < len(types) else None,
            "rope_tail": bool(ctx.feats.get("indexer_rope_tail"))}


register(Block(
    "dsa_indexer", "attn_mod", step="step 4",
    select=lambda ctx, i: ctx.has(i, "indexer.attn_k.weight") or ctx.has(i, "indexer.attn_q_b.weight"),
    config=_indexer_layer,
    claims=frozenset({"indexer.attn_k.weight", "indexer.attn_q_b.weight", "indexer.k_norm.weight", "indexer.k_norm.bias",
                      "indexer.proj.weight", "indexer_compressor_ape.weight", "indexer_compressor_gate.weight",
                      "indexer_compressor_norm.weight", "indexer_compressor_value.weight"}), build=None))
