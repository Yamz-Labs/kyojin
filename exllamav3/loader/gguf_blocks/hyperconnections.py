"""Blocks around modules/hyperconnections.py: HyperConnection stream (comb = identity, or Sinkhorn mHC) and HyperHead.
Predicates + configs only (Hy4 step 5)."""
from __future__ import annotations
from ..gguf_registry import Block, register

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


register(Block(
    "hyper_connection", "stream", step="step 5", priority=1,
    select=lambda ctx, i: bool(ctx.key("hyper_connection.count")) and ctx.has(i, "hc_attn_fn.weight"),
    config=hc_layer,
    claims=frozenset({f"hc_{s}_{t}.weight" for s in ("attn", "ffn") for t in ("fn", "base", "scale")} | set(_RESID_NORMS)),
    build=None))

register(Block(
    "hyper_head", "head_mod", step="step 5",
    select=lambda ctx, i: "output_hc_fn.weight" in ctx.glob,
    config=lambda ctx, i: {"fn_dims": ctx.dims("output_hc_fn.weight"), "count": ctx.key("hyper_connection.count")},
    claims=frozenset({"output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight"}), build=None))
