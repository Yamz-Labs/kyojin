"""Blocks around modules/transformer.py, embedding.py, rmsnorm.py: token embedding, LM head, plain residual stream."""
from __future__ import annotations
from ..gguf_registry import Block, register, CONFIG_ONLY

_NORMS = ("post_attention_norm.weight", "post_ffw_norm.weight", "layer_output_scale.weight", "ffn_norm.weight")


def residual_global(ctx):
    """sandwich norms: llama.cpp names the pre-FFN norm ffn_norm; gpt-oss has no ffn_norm and uses post_attention_norm there"""
    def f(ctx):
        for sfx in _NORMS:
            if ctx.has_any(sfx) and not ctx.has_all(sfx):
                ctx.gaps.append(f"{sfx} present on only some layers")
        sandwich = ctx.has_all("post_attention_norm.weight") and ctx.has_all("post_ffw_norm.weight")
        pre_ffn_from_post = (not ctx.has_any("ffn_norm.weight")) and ctx.has_all("post_attention_norm.weight") \
            and not ctx.has_any("post_ffw_norm.weight")
        if not ctx.has_any("ffn_norm.weight") and not pre_ffn_from_post:
            ctx.gaps.append("no pre-FFN norm tensor (ffn_norm / post_attention_norm)")
        return {"sandwich_norms": sandwich, "pre_ffn_from_post": pre_ffn_from_post, "layer_scalar": ctx.has_all("layer_output_scale.weight")}
    return ctx.once("residual", f)


def _residual_names(cfg, i, L, G):
    pre_ffn = "post_attention_norm" if cfg.get("pre_ffn_from_post") else "ffn_norm"
    nm = {f"{L}.input_layernorm.weight": f"{G}.attn_norm.weight", f"{L}.post_attention_layernorm.weight": f"{G}.{pre_ffn}.weight"}
    if cfg.get("sandwich_norms"):
        nm[f"{L}.attn_post_norm.weight"] = f"{G}.post_attention_norm.weight"
        nm[f"{L}.mlp_post_norm.weight"] = f"{G}.post_ffw_norm.weight"
    if cfg.get("layer_scalar"):
        nm[f"{L}.layer_scalar"] = f"{G}.layer_output_scale.weight"
    return nm


def _residual_build(config, plan, idx, L, attn, mlp):
    import torch
    from ...modules import RMSNorm, TransformerBlock
    post = {}
    if config.sandwich_norms:
        post = dict(attn_post_norm=RMSNorm(config=config, key=f"{L}.attn_post_norm", rms_norm_eps=config.rms_norm_eps, out_dtype=torch.float),
                    mlp_post_norm=RMSNorm(config=config, key=f"{L}.mlp_post_norm", rms_norm_eps=config.rms_norm_eps, out_dtype=torch.float))
    if config.layer_scalar:
        post["key_layer_scalar"] = "layer_scalar"
    return TransformerBlock(config=config, key=L, layer_idx=idx,
                            attn_norm=RMSNorm(config=config, key=f"{L}.input_layernorm", rms_norm_eps=config.rms_norm_eps),
                            attn=attn,
                            mlp_norm=RMSNorm(config=config, key=f"{L}.post_attention_layernorm", rms_norm_eps=config.rms_norm_eps),
                            **post, mlp=mlp)


register(Block(
    "residual", "stream",
    select=lambda ctx, i: not ctx.has(i, "hc_attn_fn.weight"),
    config=lambda ctx, i: residual_global(ctx),
    claims=frozenset({"attn_norm.weight", *_NORMS}),
    build=_residual_build, names=_residual_names))


def _embed_build(config, P):
    from ...modules import Embedding
    return Embedding(config=config, key=f"{P}.embed_tokens", vocab_size=config.vocab_size, hidden_size=config.hidden_size,
                     multiplier=config.embed_scale)


register(Block(
    "token_embd", "embed",
    select=lambda ctx, i: "token_embd.weight" in ctx.glob,
    config=lambda ctx, i: {"vocab_size": ctx.dims("token_embd.weight")[1],
                           "embed_scale": float(ctx.key("embedding_length")) ** 0.5 if ctx.feats.get("embed_scale") else 1.0},
    claims=frozenset({"token_embd.weight"}), build=_embed_build,
    names=lambda cfg, i, L, G: {"model.embed_tokens.weight": "token_embd.weight"}))


def _head_build(config, P):
    import torch
    from ...modules import RMSNorm
    from ...architecture.gguf_generic import _fl
    norm = RMSNorm(config=config, key=f"{P}.norm", rms_norm_eps=config.rms_norm_eps, out_dtype=torch.half)
    if config.tie_word_embeddings:
        config.gguf.name_map["lm_head.weight"] = "token_embd.weight"
    head = _fl(config, "lm_head", config.hidden_size, config.vocab_size, post_scale=config.logit_scale, softcap=config.final_softcap)
    head.caps["logits_output"] = True
    return [norm, head]


def _head_names(cfg, i, L, G):
    nm = {"model.norm.weight": "output_norm.weight"}
    if not cfg["tie_word_embeddings"]:
        nm["lm_head.weight"] = "output.weight"
    return nm


register(Block(
    "lm_head", "head",
    select=lambda ctx, i: "output_norm.weight" in ctx.glob,
    config=lambda ctx, i: {"tie_word_embeddings": "output.weight" not in ctx.infos,
                           "final_logit_softcapping": float(ctx.key("final_logit_softcapping", 0.0)),
                           "logit_scale": float(ctx.key("logit_scale", 1.0))},
    claims=frozenset({"output_norm.weight", "output.weight"}), build=_head_build, names=_head_names))

# rope_freqs.weight: llama.cpp freq_factors on the non-SWA layers; read by the attention config (attn.py gqa_global)
register(Block(
    "rope_freqs", "embed_mod",
    select=lambda ctx, i: "rope_freqs.weight" in ctx.glob,
    config=None, claims=frozenset({"rope_freqs.weight"}), build=CONFIG_ONLY))
