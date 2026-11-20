"""Blocks around modules/hyperconnections.py: HyperConnection stream (comb = identity, or Sinkhorn mHC) and HyperHead.
Selection is by keys / tensors only. Builds: the per-site HC modules (build_site) and the head reduction (hyper_head)."""
from __future__ import annotations
from ..gguf_registry import Block, register, GenericUnsupported

_RESID_NORMS = ("attn_norm.weight", "ffn_norm.weight", "post_attention_norm.weight", "post_ffw_norm.weight")


def hc_layer(ctx, i):
    H = ctx.key("hyper_connection.count")
    fn = ctx.dims(f"blk.{i}.hc_attn_fn.weight")         # ggml dims (H * n_embd, rows)
    rows = fn[1] if fn and len(fn) > 1 else None
    sk = ctx.key("hyper_connection.sinkhorn_iterations")
    if sk is None and rows == 2 * H:
        mode = "identity"                                 # pre + post only (hy_v4_flash; M:5431 comb is the identity)
    elif sk and rows == (2 + H) * H:
        mode = "sinkhorn"                                 # pre + post + H x H comb, Sinkhorn-normalised (deepseek41, glm5-next)
    else:
        mode = "?"
    if mode == "identity" and not ctx.feats.get("hc_comb_identity"):
        mode = "?"                                        # identity only where llama.cpp hard-codes it
    return {"count": H, "eps": ctx.key("hyper_connection.epsilon"), "magnitude": ctx.key("hyper_connection.magnitude"),
            "sinkhorn_iterations": sk, "fn_dims": fn, "fn_type": ctx.infos[f"blk.{i}.hc_attn_fn.weight"][0] if fn else None,
            "mode": mode, "norms": [s for s in _RESID_NORMS if ctx.has(i, s)]}


def _site_names(cfg, i, L, G):
    return {f"{L}.hc_{s}_{t}": f"{G}.hc_{s}_{t}.weight" for s in ("attn", "ffn") for t in ("fn", "base", "scale")}


def build_site(config, plan, idx, L, tag):
    """one HC site module (tag attn | ffn) from the layer's hyper_connection config; tensors {L}.hc_<tag>_fn / _base / _scale"""
    from ...modules.hyperconnections import HyperConnection, HyperConnectionPP
    c = plan.configs[("hyper_connection", idx)]
    kw = dict(config=config, key=f"{L}.hc_{tag}", hc_mult=c["count"], hidden_size=config.hidden_size,
              hc_eps=c["eps"], rms_norm_eps=config.rms_norm_eps)
    if c["mode"] == "identity":
        return HyperConnectionPP(**kw, magnitude=c["magnitude"] if c["magnitude"] is not None else 2.0)
    if c["mode"] == "sinkhorn":
        if c["magnitude"] not in (None, 2.0):
            raise GenericUnsupported(f"hyper_connection.magnitude {c['magnitude']} (the Sinkhorn HyperConnection hard-codes 2.0)")
        return HyperConnection(**kw, sinkhorn_iters=c["sinkhorn_iterations"])
    raise GenericUnsupported(f"hyper_connection layout not recognised at layer {idx}: fn dims {c['fn_dims']}, "
                             f"sinkhorn_iterations {c['sinkhorn_iterations']}")


def _stream_names(cfg, i, L, G):
    """sites + the norms of the layer: attn_norm / ffn_norm before the sublayers, post_attention_norm / post_ffw_norm on their output"""
    nm = _site_names(cfg, i, L, G)
    norms = cfg.get("norms") or []
    for e, g in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")):
        if f"{g}.weight" in norms:
            nm[f"{L}.{e}.weight"] = f"{G}.{g}.weight"
    if "post_attention_norm.weight" in norms and "post_ffw_norm.weight" in norms:
        nm[f"{L}.attn_post_norm.weight"] = f"{G}.post_attention_norm.weight"
        nm[f"{L}.mlp_post_norm.weight"] = f"{G}.post_ffw_norm.weight"
    return nm


def _stream_build(config, plan, idx, L, attn, mlp):
    """TransformerBlock with the HC sites. Order (llama.cpp hy_v4_flash, M:6064-6091): collapse (pre) -> attn_norm -> attn ->
    post_attention_norm -> post mix into the streams; the same with ffn_norm / ffn / post_ffw_norm. Engram is outside the block
    (it writes the streams before the attn pre-mix, M:6059)."""
    import torch
    from ...modules import RMSNorm, TransformerBlock
    norms = plan.configs[("hyper_connection", idx)]["norms"]
    if "attn_norm.weight" not in norms or "ffn_norm.weight" not in norms:
        raise GenericUnsupported(f"hyper_connection layer {idx}: attn_norm / ffn_norm missing ({norms})")
    sandwich = "post_attention_norm.weight" in norms and "post_ffw_norm.weight" in norms
    if ("post_attention_norm.weight" in norms) != ("post_ffw_norm.weight" in norms):
        raise GenericUnsupported(f"hyper_connection layer {idx}: only one of post_attention_norm / post_ffw_norm ({norms})")
    eps = config.rms_norm_eps
    post = {}
    if sandwich:
        post = dict(attn_post_norm=RMSNorm(config=config, key=f"{L}.attn_post_norm", rms_norm_eps=eps, out_dtype=torch.float),
                    mlp_post_norm=RMSNorm(config=config, key=f"{L}.mlp_post_norm", rms_norm_eps=eps, out_dtype=torch.float))
    return TransformerBlock(config=config, key=L, layer_idx=idx,
                            attn_hc=build_site(config, plan, idx, L, "attn"),
                            attn_norm=RMSNorm(config=config, key=f"{L}.input_layernorm", rms_norm_eps=eps), attn=attn,
                            mlp_hc=build_site(config, plan, idx, L, "ffn"),
                            mlp_norm=RMSNorm(config=config, key=f"{L}.post_attention_layernorm", rms_norm_eps=eps), mlp=mlp,
                            **post)


register(Block(
    "hyper_connection", "stream", step="step 5", priority=1, names=_stream_names,
    select=lambda ctx, i: bool(ctx.key("hyper_connection.count")) and ctx.has(i, "hc_attn_fn.weight"),
    config=hc_layer,
    claims=frozenset({f"hc_{s}_{t}.weight" for s in ("attn", "ffn") for t in ("fn", "base", "scale")} | set(_RESID_NORMS)),
    build=_stream_build))


def _head_names(cfg, i, L, G):
    return {f"model.hc_head_{t}": f"output_hc_{t}.weight" for t in ("fn", "base", "scale")}


def _head_build(config, P):
    """[HyperHead], to sit before the final norm: mixes = fn @ rms(flat streams); pre = sigmoid(mixes * scale + base) + eps
    (llama.cpp build_hc_head, no magnitude); collapsed = sum_i pre_i * stream_i"""
    from ...modules.hyperconnections import HyperHead
    c = config.gguf_plan.configs[("hyper_head", None)]
    return [HyperHead(config=config, key=f"{P}.hc_head", hc_mult=c["count"], rms_norm_eps=config.rms_norm_eps, hc_eps=c["eps"])]


register(Block(
    "hyper_head", "head_mod", step="step 5",
    select=lambda ctx, i: "output_hc_fn.weight" in ctx.glob,
    config=lambda ctx, i: {"fn_dims": ctx.dims("output_hc_fn.weight"), "count": ctx.key("hyper_connection.count"),
                           "eps": ctx.key("hyper_connection.epsilon")},
    claims=frozenset({"output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight"}),
    build=_head_build, names=_head_names))
