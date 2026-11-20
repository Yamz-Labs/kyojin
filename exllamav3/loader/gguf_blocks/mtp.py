"""MTP / nextn block (blk.<n>.nextn.*): predicate only, built after the trunk is correct (Hy4 step 8)."""
from __future__ import annotations
from ..gguf_registry import Block, register

register(Block(
    "mtp_nextn", "layer_mod", step="step 8",
    select=lambda ctx, i: ctx.has(i, "nextn.eh_proj.weight"),
    config=lambda ctx, i: {"nextn_predict_layers": ctx.key("nextn_predict_layers")},
    claims=frozenset({"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight",
                      "nextn.embed_tokens.weight", "nextn.shared_head_head.weight"}), build=None))
