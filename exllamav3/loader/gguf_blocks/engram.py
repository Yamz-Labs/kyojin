"""Blocks around modules/dsv41_fn/engram_dev.py (Engram n-gram rows) and the F8_E4M3 row table (ggml type 43).
Predicates + configs only (Hy4 step 2). Layout comes from the ple.* / engram.* keys, shared by hy_v4_flash and deepseek41."""
from __future__ import annotations
from ..gguf_registry import Block, register, GenericUnsupported

GGML_F8_E4M3 = 43      # ggml_master.h:262; row-lookup-only: never a Linear, never dequantised resident

_HY4_GATE = ("engram_kv_b.weight", "engram_lookup_norm.weight", "engram_pre_gamma.weight", "engram_hc_scale.weight")


def _engram_layer(ctx, i):
    pl = list(ctx.key("ple.layers") or [])
    n, h = ctx.key("ple.ngram_size"), ctx.key("ple.heads_per_ngram")
    cols = (n - 1) * h if n and h else None
    offs = ctx.key("ple.head_offsets") or []
    return {"module": pl.index(i), "n_modules": len(pl), "ngram_size": n, "heads_per_ngram": h, "columns": cols,
            "row_width": ctx.key("embedding_length_per_layer_input"), "n_head_offsets": len(offs),
            "offsets_cover_modules": bool(cols) and len(offs) == cols * len(pl),
            "variant": "hy4_gate" if all(ctx.has(i, s) for s in _HY4_GATE) else "ds41_gate",
            "kv_dims": ctx.dims(f"blk.{i}.engram_kv.weight"), "pad_token_id": ctx.key("engram.pad_token_id"),
            "token_map": len(ctx.key("engram.token_map") or []), "eps": ctx.feats.get("engram_eps")}


def _is_engram_layer(ctx, i):
    return i in (ctx.key("ple.layers") or []) and ctx.has(i, "engram_kv.weight")


def _hy4_gate(ctx, i):
    return all(ctx.has(i, s) for s in _HY4_GATE)


def _gguf(config):
    return getattr(config, "gguf", config)      # the GGUFCollection, from a model config or passed directly


def _build_engram_hy4(config, plan, i, key=None):
    """Hy4 Engram pre-hook (CPU fp32 maths, loader/engram_rows.py); a module with forward(x [T, hc, n_embd], tokens, start)."""
    from ..engram_rows import Hy4Engram
    eps = (plan.ctx.feats if plan is not None else {}).get("engram_eps", 1.1920929e-07)
    return Hy4Engram(_gguf(config), i, eps=eps)


# DS4.1 gate variant (wkv + norm_key/query gate, no kv bias): still a predicate only
register(Block(
    "engram", "pre", step="step 2",
    select=lambda ctx, i: _is_engram_layer(ctx, i) and not _hy4_gate(ctx, i),
    config=_engram_layer, claims=frozenset({"engram_kv.weight", "engram_gate.weight", *_HY4_GATE}), build=None))

# Hy4 gate variant: selected by the tensors engram_kv_b / lookup_norm / pre_gamma / hc_scale, not by a model name
register(Block(
    "engram_hy4", "pre", step="step 2",
    select=lambda ctx, i: _is_engram_layer(ctx, i) and _hy4_gate(ctx, i),
    config=_engram_layer, claims=frozenset({"engram_kv.weight", "engram_gate.weight", *_HY4_GATE}), build=_build_engram_hy4))


def _f8_cfg(ctx, i):
    ty, dims, off = ctx.infos["per_layer_token_embd.weight"]
    part = off >> 48
    bases = getattr(ctx.g, "bases", None)
    offs = ctx.key("ple.head_offsets") or []
    sizes = ctx.key("ple.head_vocab_sizes") or []
    layout, row_bytes = None, None
    if ty == GGML_F8_E4M3:
        from ..engram_rows import f8_layout, tensor_span, F8LayoutError, QB
        try:
            layout = f8_layout(dims[0] * dims[1], tensor_span(ctx.g, "per_layer_token_embd.weight"), dims[0])
            row_bytes = dims[0] if layout == "plain" else dims[0] // QB * 33
        except (F8LayoutError, OSError, AttributeError) as ex:
            layout = f"refused: {ex}"
    return {"type": ty, "row_width": dims[0], "rows": dims[1], "layout": layout, "row_bytes": row_bytes,
            "part": part, "data_offset": (bases[part] if bases else 0) + (off & ((1 << 48) - 1)),
            "width_matches_ple": dims[0] == ctx.key("embedding_length_per_layer_input"),
            "rows_match_offsets": bool(offs) and offs[-1] + sizes[-1] == dims[1]}


def _build_f8(config, plan, i, key=None):
    """row-lookup-only F8_E4M3 table: F8Rows.lookup(ids) -> float32 rows; refuses a span that is neither layout"""
    from ..engram_rows import F8Rows
    return F8Rows(_gguf(config), "per_layer_token_embd.weight")


register(Block(
    "f8_rows", "embed_mod", step="step 2",
    select=lambda ctx, i: "per_layer_token_embd.weight" in ctx.infos and ctx.infos["per_layer_token_embd.weight"][0] == GGML_F8_E4M3,
    config=_f8_cfg, claims=frozenset({"per_layer_token_embd.weight"}), build=_build_f8))
