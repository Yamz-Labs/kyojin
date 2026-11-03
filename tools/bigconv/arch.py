"""ArchAdapter: the only place that knows an architecture.

Three seams:
  * naming / roles   classify(name, T) -> Cls: what to do with every source tensor (plan side, no torch).
  * source format    how a linear is stored (fp8 block + scale tensor, bf16, ...): part of Cls (scale name, in/out).
  * capture forward  capture_runner(...) -> LayerRunner: the reference (unquantized) math of one block, in
                     pieces the generic streamed driver calls (capture side, needs torch + the model code).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

ADAPTERS: dict = {}


def register(cls):
    ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str, config: Optional[dict] = None) -> "ArchAdapter":
    if name not in ADAPTERS:
        raise KeyError(f"unknown arch {name!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[name](config)


@dataclass
class Expert:
    """One routed-expert linear."""
    layer_prefix: str      # e.g. "layers.3" (DS4.1) or "model.language_model.layers.3"
    layer: int
    e: int
    proj: str              # arch name of the projection (w1 / gate_proj / ...)


@dataclass
class Cls:
    """Classification of one source tensor.

    action  quant   quantizable linear (kind says which group)
            copy    stored as it is
            raw     copied byte for byte, never touched (engram tables, ...)
            skip    consumed with another tensor (the .scale of a weight)
    kind    routed | mtp_routed | mtp | shared | attn | dense | head | engram | copy
    role    attention role name (per-role bits), or None
    slices  [(suffix, (row_lo, row_hi))] when one stored weight is quantized as several linears
    """
    action: str
    kind: str = "copy"
    weight: Optional[str] = None
    scale: Optional[str] = None
    out_f: int = 0
    in_f: int = 0
    base: Optional[str] = None
    expert: Optional[Expert] = None
    role: Optional[str] = None
    slices: Optional[list] = None
    fmt: Optional[str] = None       # source storage format (vendor_input.classify_linear): fp8_block, nvfp4, mxfp4, dense_bf16 ...
    scale2: Optional[str] = None    # NVFP4 global scale tensor


class ArchAdapter:
    name = "base"
    # --- sizes (filled from the config)
    n_layers = 0            # trunk layers that capture runs
    n_experts = 0
    topk = 0
    n_mtp = 0               # MTP / draft units
    hidden = 0
    expert_inter = 0        # routed expert intermediate size (w1/w3 out)
    hc_mult = 1             # hyper-connection streams (row state is [L, hc, hidden])
    default_bits = {"attn": 5, "shared": 5, "mtp": 4, "head": 6, "hot": 3, "cold": 2, "mtp_exp": None}
    attn_roles: tuple = ()
    alloc_schema = "bigconv-alloc/1"
    alloc_k_min, alloc_k_max = 1, 8
    # per-expert linears: (name, in_dim_attr, out_dim_attr) used by the sheet (w1/w3 up, w2 down)
    source_bytes_per_weight = 1.0   # 1.0 fp8, 2.0 bf16, 0.5 fp4

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.config_vocab = (self.config.get("text_config", self.config) or {}).get("vocab_size", 0)

    # ----------------------------------------------------------------------------- plan side
    def unit_of(self, name: str) -> str:
        raise NotImplementedError

    def unit_sort_key(self, unit: str):
        if unit == "global":
            return (0, 0, "")
        if unit.endswith(".engram"):
            return (3, int(unit.split(".")[1]), "")
        pre, idx = unit.rsplit(".", 1)
        return (1 if pre.endswith("layers") else 2, int(idx), "")

    def classify(self, name: str, T: dict) -> Cls:
        raise NotImplementedError

    def ffn_key(self, layer_prefix: str) -> str:
        return f"{layer_prefix}.ffn"

    def backbone_layers(self, tmap) -> list:
        """Layers that carry routed experts (sorted)."""
        out = set()
        for n in tmap["tensors"]:
            c = self.expert_of(n)
            if c is not None:
                out.add(c.layer)
        return sorted(out)

    expert_first_proj = "w1"

    def expert_of(self, name: str) -> Optional[Expert]:
        """Routed-expert linear match by name only (no tensor map needed), trunk layers only."""
        raise NotImplementedError

    # ----------------------------------------------------------------------------- sheet side
    def sheet_dims(self) -> dict:
        """Dims the resource sheet needs: hidden, expert_inter, n_experts, topk, n_layers_moe, ...."""
        raise NotImplementedError

    def shape_of(self, name: str, names=None):
        """Shape [out, in] of a quantizable linear derived from the config, or None when unknown (sheet without map).
        `names` = set of all tensor names of the index (lets an adapter tell two layer types apart)."""
        return None

    def dense_hessian_dims(self) -> list:
        """Input widths of the non-expert Hessian groups of the largest layer type (spool + RAM sizing)."""
        return [self.hidden]

    # ----------------------------------------------------------------------------- capture side
    def capture_runner(self, *a, **k):
        raise NotImplementedError(f"{self.name}: capture forward not implemented (see TODO_OPS)")

    TODO_OPS: list = []


class TableAdapter(ArchAdapter):
    """Adapter driven by an ordered role table (HF-style checkpoints with one `.weight` per linear).

    Subclasses set:
      layer_re      regex with one group = layer index, matching the per-layer name prefix
      expert_re     regex (groups: layer prefix, layer idx, expert idx, proj) of a routed expert weight
      ROLES         [(regex, kind, role, shape_fn)] first match wins; kind in routed|shared|attn|dense|mtp|head|copy;
                    shape_fn(adapter) -> (out, in) used by the sheet when no tensor map has shapes
      scale_suffix  suffix of the scale tensor next to a weight (".weight_scale_inv"), or None for bf16 sources
    """
    layer_re = None
    expert_re = None
    ROLES: list = []
    scale_suffix = None
    mtp_from_layer = None       # layer index >= this is an MTP unit "mtp.<i - this>"
    lang_prefixes: tuple = ()   # only names under these prefixes are language-model linears (vision etc. = copy)

    def unit_of(self, name):
        m = self.layer_re.match(name)
        if not m:
            return "global"
        i = int(m.group(1))
        if self.mtp_from_layer is not None and i >= self.mtp_from_layer:
            return f"mtp.{i - self.mtp_from_layer}"
        return f"layers.{i}"

    def expert_of(self, name):
        m = self.expert_re.match(name)
        if not m:
            return None
        lp, li = m.group(1), int(m.group(2))
        if self.mtp_from_layer is not None and li >= self.mtp_from_layer:
            return None
        return Expert(lp, li, int(m.group(3)), m.group(4))

    _SCALE_SFX = (".weight_scale_inv", ".weight_scale", ".scale", ".weight_scale_2", ".weight_global_scale",
                  ".input_scale", ".input_global_scale")

    def _src_info(self, T):
        """{name: (dtype, shape)} of the tensor map, cached per map (vendor_input.classify_linear works on it)."""
        c = getattr(self, "_info_cache", None)
        if c is None or c[0] is not T:
            from .formats import info_from_map
            c = self._info_cache = (T, info_from_map({"tensors": T}))
        return c[1]

    def classify(self, name, T):
        t = T[name]
        if self.scale_suffix and name.endswith(self.scale_suffix) and name[:-len(self.scale_suffix)] + ".weight" in T:
            return Cls("skip")
        sfx = next((x for x in self._SCALE_SFX if name.endswith(x)), None)
        if sfx and (name[:-len(sfx)] + ".weight" in T or name[:-len(sfx)] + ".weight_packed" in T):
            return Cls("skip")           # scale / global scale / activation scale consumed with its weight (any vendor layout)
        if not name.endswith((".weight", ".weight_packed")) or len(t.get("shape") or [0, 0]) != 2:
            return Cls("copy")
        if self.lang_prefixes and not name.startswith(self.lang_prefixes):
            return Cls("copy")
        for rx, kind, role, _ in self.ROLES:
            if not rx.search(name):
                continue
            if kind == "copy":
                return Cls("copy")
            sname = name[:-7] + self.scale_suffix if self.scale_suffix else None
            if sname not in T:
                sname = None
            src = None
            if name.endswith(".weight") or name.endswith(".weight_packed"):
                from .formats import vendor_input
                base = name.rsplit(".", 1)[0]
                src = vendor_input().classify_linear(self._src_info(T), base)
            in_f = t["shape"][1]
            if src and src["fmt"] in ("nvfp4", "mxfp4"):
                in_f *= 2                # two e2m1 nibbles per stored byte
            if sname is None and src and src.get("scale"):
                sname = src["scale"]
            c = dict(action="quant", weight=name, scale=sname, out_f=t["shape"][0], in_f=in_f, base=name.rsplit(".", 1)[0],
                     role=role, fmt=src["fmt"] if src else None, scale2=(src or {}).get("scale2"))
            if kind == "routed":
                ex = self.expert_of(name)
                if ex is not None:
                    return Cls(kind="routed", expert=ex, **c)
                return Cls(kind="mtp_routed", **c)
            if kind in ("shared", "attn", "dense", "head"):
                if kind == "head":
                    c["base"] = "head"
                if self.is_mtp_name(name) and kind != "head":
                    return Cls(kind="mtp", **c)
                return Cls(kind=kind, **c)
            return Cls(kind=kind, **c)
        return Cls("copy")

    def is_mtp_name(self, name):
        m = self.layer_re.match(name)
        return bool(m and self.mtp_from_layer is not None and int(m.group(1)) >= self.mtp_from_layer)

    def shape_of(self, name, names=None):
        if self.lang_prefixes and not name.startswith(self.lang_prefixes):
            return None
        for rx, kind, role, fn in self.ROLES:
            if rx.search(name) and name.endswith(".weight"):
                return fn(self) if fn else None
        return None
