"""Allocation file (per-layer / per-expert K, no pruning), generic over the adapter.

Same rules as tools/ds41/alloc_file.py (which stays untouched); the expert count, the layer set and the schema tag
come from the adapter. `expert_bits` keys are `adapter.ffn_key(layer_prefix)`.
"""
from __future__ import annotations

K_MIN, K_MAX = 1, 8
DENSE_DEFAULTS = {"attn": 5, "shared": 5, "head": 6, "mtp": 4}
TOP_KEYS = {"schema", "dense", "experts", "layers", "comment"}
LAYER_KEYS = {"attn", "shared", "experts"}


def _k(v, where):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{where}: K must be an integer, got {v!r}")
    if v < K_MIN:
        raise ValueError(f"{where}: K={v}; K >= {K_MIN} everywhere, pruning is not allowed")
    if v > K_MAX:
        raise ValueError(f"{where}: K={v} > {K_MAX}")
    return v


def _experts(v, where, n_experts):
    if isinstance(v, list):
        if len(v) != n_experts:
            raise ValueError(f"{where}: {len(v)} entries, expected {n_experts}")
        for i, k in enumerate(v):
            _k(k, f"{where}[{i}]")
    else:
        _k(v, where)


def backbone_layers(adapter, tmap=None) -> list:
    if tmap is None:
        return list(range(adapter.n_layers))
    return adapter.backbone_layers(tmap)


def layer_prefixes(adapter, tmap) -> dict:
    """{layer index: layer prefix} of the layers with routed experts."""
    out = {}
    for n in tmap["tensors"]:
        e = adapter.expert_of(n)
        if e is not None:
            out[e.layer] = e.layer_prefix
    return out


def validate(alloc: dict, adapter, tmap=None) -> dict:
    """Raise ValueError on any violation; fill the missing `dense` defaults in place and return alloc."""
    if alloc.get("schema") != adapter.alloc_schema:
        raise ValueError(f"schema must be {adapter.alloc_schema!r}, got {alloc.get('schema')!r}")
    extra = set(alloc) - TOP_KEYS
    if extra:
        raise ValueError(f"unknown top-level keys {sorted(extra)} (a pruning section is not part of this format)")
    dense = alloc.setdefault("dense", {})
    if set(dense) - set(DENSE_DEFAULTS) - {"mtp_exp"}:
        raise ValueError(f"dense: unknown keys {sorted(set(dense) - set(DENSE_DEFAULTS) - {'mtp_exp'})}; "
                         f"allowed {sorted(DENSE_DEFAULTS) + ['mtp_exp']}")
    if "mtp_exp" in dense:
        _k(dense["mtp_exp"], "dense.mtp_exp")
    for k, d in DENSE_DEFAULTS.items():
        _k(dense.setdefault(k, d), f"dense.{k}")
    layers = backbone_layers(adapter, tmap)
    n = adapter.n_experts
    if "experts" in alloc:
        _experts(alloc["experts"], "experts", n)
    lay = alloc.get("layers", {})
    for key, ent in lay.items():
        if not str(key).isdigit() or int(key) not in layers:
            raise ValueError(f"layers.{key}: not a backbone layer (0-{layers[-1]})")
        if set(ent) - LAYER_KEYS:
            raise ValueError(f"layers.{key}: unknown keys {sorted(set(ent) - LAYER_KEYS)}; allowed {sorted(LAYER_KEYS)}")
        for f in ("attn", "shared"):
            if f in ent:
                _k(ent[f], f"layers.{key}.{f}")
        if "experts" in ent:
            _experts(ent["experts"], f"layers.{key}.experts", n)
    missing = [l for l in layers if "experts" not in lay.get(str(l), {}) and "experts" not in alloc]
    if missing:
        raise ValueError(f"no routed-expert K for layers {missing}: give layers.L.experts or a top-level experts")
    return alloc


def expert_bits(alloc: dict, adapter, tmap=None) -> dict:
    """{ffn_key: [K per expert]} for every backbone layer."""
    pref = layer_prefixes(adapter, tmap) if tmap is not None else {l: f"layers.{l}" for l in range(adapter.n_layers)}
    out = {}
    for l in backbone_layers(adapter, tmap):
        v = alloc.get("layers", {}).get(str(l), {}).get("experts", alloc.get("experts"))
        out[adapter.ffn_key(pref[l])] = list(v) if isinstance(v, list) else [v] * adapter.n_experts
    return out


def layer_dense(alloc: dict, adapter, tmap=None) -> dict:
    """{layer unit: {'attn': K, 'shared': K}} for the layers that override the dense defaults."""
    return {f"layers.{l}": {k: v for k, v in e.items() if k in ("attn", "shared")}
            for l, e in alloc.get("layers", {}).items() if set(e) & {"attn", "shared"}}
