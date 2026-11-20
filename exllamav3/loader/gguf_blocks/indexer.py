"""Block around modules/kpool_indexer.py: the Hy4 (hy_v4_flash) k-pool DSA indexer, row-list form (design §2, indexer row).
Selected by tensors / keys only: a layer with indexer.attn_k + indexer.attn_q_b + indexer_compressor_value and a weight-only
indexer.k_norm (no bias), with the indexer.top_k / kpool keys. The other indexer variants (dsa with no pooling, glm5-next
kpool_gate, LayerNorm k_norm) keep the plain 'dsa_indexer' block from mla_attn.py, which this file narrows so the two never
select the same layer (the only touch on that registration; mla_attn.py itself is not edited)."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import torch
from ..gguf_registry import Block, REGISTRY, register, GenericUnsupported
from . import mla_attn as _mla                       # noqa: F401  (registers dsa_indexer first)

_KP = ("indexer.attn_k.weight", "indexer.attn_q_b.weight", "indexer_compressor_value.weight")
# engine name -> GGUF suffix
_W = {"attn_k": "indexer.attn_k.weight", "k_norm": "indexer.k_norm.weight", "q_b": "indexer.attn_q_b.weight",
      "proj": "indexer.proj.weight", "value": "indexer_compressor_value.weight", "gate": "indexer_compressor_gate.weight",
      "ape": "indexer_compressor_ape.weight", "pool_norm": "indexer_compressor_norm.weight"}


def is_kpool(ctx, i):
    return (all(ctx.has(i, s) for s in (*_KP, "indexer_compressor_gate.weight", "indexer_compressor_ape.weight",
                                        "indexer_compressor_norm.weight", "indexer.k_norm.weight", "indexer.proj.weight"))
            and not ctx.has(i, "indexer.k_norm.bias")
            and all(ctx.key(f"attention.indexer.{k}") for k in ("top_k", "kpool", "head_count", "key_length")))


_orig = REGISTRY["dsa_indexer"]
register(replace(_orig, select=lambda ctx, i: _orig.select(ctx, i) and not is_kpool(ctx, i)))


def _cfg(ctx, i):
    c = _mla._indexer_layer(ctx, i)
    swa = _mla._swa(ctx, i)
    c.update(rope_dim=ctx.key("rope.dimension_count"), rope_base=ctx.key("rope.freq_base"), eps=ctx.key("attention.layer_norm_rms_epsilon"),
             swa_layer=swa, dims={k: ctx.dims(f"blk.{i}.{s}") for k, s in _W.items()},
             types={k: ctx.infos[f"blk.{i}.{s}"][0] for k, s in _W.items()})
    return c


def _w(g, name, ty, dims):
    """fp32 (out, in) weight of one GGUF tensor: F32 as is, Q8_0 dequantised (34-byte blocks: fp16 d + 32 int8). Other types are refused."""
    t = g.get_tensor(name)
    if ty == 0:
        return t.float().reshape(dims[1] if len(dims) > 1 else 1, dims[0]) if len(dims) > 1 else t.float().reshape(-1)
    if ty == 8:
        raw = t.numpy().reshape(dims[1], dims[0] // 32, 34)
        d = np.ascontiguousarray(raw[..., :2]).view(np.float16).astype(np.float32)
        q = np.ascontiguousarray(raw[..., 2:]).view(np.int8).astype(np.float32)
        return torch.from_numpy((q * d).reshape(dims[1], dims[0]))
    raise GenericUnsupported(f"indexer tensor {name}: ggml type {ty} (only F32 and Q8_0 are read)")


def load_weights(g, i):
    out = {}
    for k, s in _W.items():
        ty, dims, _ = g.infos[f"blk.{i}.{s}"]
        out[k] = _w(g, f"blk.{i}.{s}", ty, dims)
    return out


def _build(config, plan, i, key=None):
    """KPoolIndexer for layer i (CPU fp32): forward(x, qr) -> IndexerSelection (row lists + scores). Weights read by offset."""
    from ...modules.kpool_indexer import KPoolIndexer
    c = plan.configs[("dsa_kpool_indexer", i)]
    if c["swa_layer"]:
        raise GenericUnsupported(f"layer {i}: an indexer on a sliding-window layer (the cache mirrors the full-attention child)")
    if c["rope_dim"] is None or c["rope_base"] is None or c["eps"] is None or c["rope_dim"] >= c["head_dim"]:
        raise GenericUnsupported(f"layer {i}: indexer needs rope.dimension_count < indexer.key_length, rope.freq_base, rms eps")
    g = getattr(config, "gguf", config)
    return KPoolIndexer(load_weights(g, i), n_head=c["n_head"], head_dim=c["head_dim"], top_k=c["top_k"], kpool=c["kpool"],
                        select_tail=bool(c["select_tail"]), rope_dim=c["rope_dim"], rope_base=c["rope_base"], eps=c["eps"],
                        q8_weights=tuple(k for k, t in c["types"].items() if t == 8))


register(Block("dsa_kpool_indexer", "attn_mod", select=is_kpool, config=_cfg, claims=_orig.claims, build=_build, priority=1))
