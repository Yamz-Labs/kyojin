"""Blocks around modules/dsv41_fn/engram_dev.py (Engram n-gram rows) and the F8_E4M3 row table (ggml type 43).
Predicates + configs only (Hy4 step 2). Layout comes from the ple.* / engram.* keys, shared by hy_v4_flash and deepseek41."""
from __future__ import annotations
from ..gguf_registry import Block, register

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


register(Block(
    "engram", "pre", step="step 2",
    select=lambda ctx, i: i in (ctx.key("ple.layers") or []) and ctx.has(i, "engram_kv.weight"),
    config=_engram_layer, claims=frozenset({"engram_kv.weight", "engram_gate.weight", *_HY4_GATE}), build=None))


def _f8_cfg(ctx, i):
    ty, dims, off = ctx.infos["per_layer_token_embd.weight"]
    part = off >> 48
    bases = getattr(ctx.g, "bases", None)
    offs = ctx.key("ple.head_offsets") or []
    sizes = ctx.key("ple.head_vocab_sizes") or []
    return {"type": ty, "row_width": dims[0], "rows": dims[1], "row_bytes": dims[0] if ty == GGML_F8_E4M3 else None,
            "part": part, "data_offset": (bases[part] if bases else 0) + (off & ((1 << 48) - 1)),
            "width_matches_ple": dims[0] == ctx.key("embedding_length_per_layer_input"),
            "rows_match_offsets": bool(offs) and offs[-1] + sizes[-1] == dims[1]}


register(Block(
    "f8_rows", "embed_mod", step="step 2",
    select=lambda ctx, i: "per_layer_token_embd.weight" in ctx.infos and ctx.infos["per_layer_token_embd.weight"][0] == GGML_F8_E4M3,
    config=_f8_cfg, claims=frozenset({"per_layer_token_embd.weight"}), build=None))
