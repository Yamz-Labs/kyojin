"""GGUF block registry: the generic loader is the composition of these blocks (no per-model class).

A Block is (name, slot, select, config, claims, build):
  select(ctx, i) -> bool      ctx = the GGUF header (metadata + tensor suffixes per layer); i = layer index, None for global slots
  config(ctx, i) -> dict      the per-layer (or global) module config
  claims                      tensor suffixes the block owns (blk.N.<suffix>, or full names for global slots)
  build(config, plan, i, key) -> Module, or None while the block exists only as a predicate (its step is named)
Blocks live in loader/gguf_blocks/<module>.py, one file per engine module family; their builds import the modules lazily
so the registry itself imports without torch or the extension (CPU header tests).

Composition (architecture/gguf_generic.py): per layer pre -> stream(attn [+ attn_mod], ffn [+ ffn_mod]); global embed
[+ embed_mod], head [+ head_mod]. In each base slot the first selected block (priority order) wins; *_mod slots take every
selected block. Coverage audit: the union of the selected blocks' claims must cover every tensor of the header, else the
'no builder for tensors' gap fires (an unknown arch still fails loudly). Tensors owned by a selected block that has no
build yet are reported per block ('block X not built').

Per-arch residue: ARCH_FEATS holds only what llama.cpp hard-codes for an arch with no GGUF key (each entry names its source).
"""
from __future__ import annotations
import importlib
import re
from dataclasses import dataclass
from typing import Callable

BASE_SLOTS = ("pre", "stream", "attn", "ffn")
MOD_SLOTS = ("attn_mod", "ffn_mod", "layer_mod")
GLOBAL_SLOTS = ("embed", "embed_mod", "head", "head_mod")
CONFIG_ONLY = "config-only"      # build marker: the block feeds module configs, no module of its own
BLOCK_MODULES = ("transformer", "attn", "mlp")

# Per-arch features llama.cpp hard-codes with no GGUF key. Every entry names the llama.cpp source it mirrors
# (M = the model patch published with the model, else the llama.cpp file).
ARCH_FEATS = {
    # llama.cpp convert_hf_to_gguf.py LlamaModel.permute: Q/K rows permuted to adjacent-pair rotary (only this converter)
    "llama": {"rope_gptj": True},
    # llama.cpp llama-model.cpp load_hparams: expert_weights_norm default when the key is absent
    "qwen3moe": {"norm_topk": True},
    "mixtral": {"norm_topk": True},
    "olmoe": {"norm_topk": False},
    "gpt-oss": {
        "norm_topk": True,                # softmax over the top-k logits == softmax over all, renormalised over the picked experts
        "swa_pattern": 2,                 # llama-model.cpp LLM_ARCH_OPENAI_MOE: hparams.set_swa_pattern(2)
        "yarn_no_truncate": True,         # HF gpt-oss rope_scaling.truncate = false (llama.cpp yarn has no truncation)
        "moe_act": ("swiglu_oai", 7.0),   # llama-graph.cpp build_moe_ffn: swiglu_oai alpha 1.702 / limit 7.0 (no key; <arch>.swiglu_limit overrides)
    },
    # src/models/gemma4.cpp (see GEMMA4_UNCHECKED.md): LLM_FFN_GELU (tanh) dense act; build_inp_embd(.., sqrtf(n_embd));
    # unweighted rms_norm on V of every layer (K=V layers: V = raw k_proj output); hparams.f_attention_scale = 1.0
    "gemma4": {"act": "gelu", "embed_scale": True, "v_norm": True, "sm_scale": 1.0},
}


class GenericUnsupported(ValueError):
    """The GGUF needs a piece the generic builder does not have (the message names it)."""


def req(m: dict, key: str):
    if key not in m:
        raise GenericUnsupported(f"missing GGUF metadata key {key!r}")
    return m[key]


@dataclass(frozen=True)
class Block:
    name: str
    slot: str
    select: Callable
    config: Callable | None
    claims: frozenset
    build: Callable | None = None
    names: Callable | None = None      # (cfg, i, L, G) -> {engine key: gguf name}: the block's part of the name map
    step: str = ""                     # design §8 build step, for blocks that are predicates only
    priority: int = 0


REGISTRY: dict[str, Block] = {}


def register(b: Block) -> Block:
    assert b.slot in BASE_SLOTS + MOD_SLOTS + GLOBAL_SLOTS + ("model",), b.slot
    REGISTRY[b.name] = b
    return b


def _load_blocks():
    for mod in BLOCK_MODULES:
        importlib.import_module(f"{__package__}.gguf_blocks.{mod}")


class Ctx:
    """GGUF header view shared by every block predicate / config (cached derived values in .memo)."""

    def __init__(self, g):
        self.g, self.meta, self.infos = g, g.meta, g.infos
        self.arch = self.meta.get("general.architecture")
        self.p = (self.arch or "") + "."
        self.nl = self.meta.get(self.p + "block_count", 0)
        self.feats = ARCH_FEATS.get(self.arch, {})
        self.layers: dict[int, set] = {}
        self.glob: set = set()
        for name in self.infos:
            mt = re.match(r"blk\.(\d+)\.(.+)$", name)
            if mt:
                self.layers.setdefault(int(mt.group(1)), set()).add(mt.group(2))
            else:
                self.glob.add(name)
        self.memo: dict = {}
        self.gaps: list = []

    def key(self, k, default=None):
        return self.meta.get(self.p + k, default)

    def has(self, i, sfx) -> bool:
        return sfx in self.layers.get(i, ())

    def has_all(self, sfx) -> bool:
        return all(sfx in self.layers.get(i, ()) for i in range(self.nl))

    def has_any(self, sfx) -> bool:
        return any(sfx in self.layers.get(i, ()) for i in range(self.nl))

    def dims(self, name):
        return self.infos[name][1] if name in self.infos else None

    def once(self, name, fn):
        if name not in self.memo:
            self.memo[name] = fn(self)
        return self.memo[name]

    def per_layer(self, k, default=None):
        """<arch>.<k> as a per-layer list (scalar keys repeated)."""
        v = self.key(k, default)
        return list(v) if isinstance(v, list) else [v] * self.nl


@dataclass
class Plan:
    arch: str
    layers: list          # per layer: {slot: block name} for base slots, {slot: [names]} for mod slots
    glob: dict            # {slot: block name / [names]}
    configs: dict         # (block name, i or None) -> config dict
    claimed: dict         # tensor name -> block name
    unclaimed: list
    unbuilt: dict         # block name -> sorted tensor patterns (blk.N.<sfx> with the layer list)
    cov_gaps: list
    ctx: object = None

    @property
    def gaps(self):
        """block-config gaps (ctx.gaps grows while the composer reads global configs) + coverage / unbuilt gaps"""
        return list(self.ctx.gaps) + list(self.cov_gaps)

    def blocks_at(self, i):
        lp = self.layers[i]
        return [lp[s] for s in BASE_SLOTS if lp.get(s)] + [b for s in MOD_SLOTS for b in lp.get(s, [])]

    def block(self, i, slot):
        return REGISTRY[self.layers[i][slot]] if self.layers[i].get(slot) else None


def _pick(ctx, slot, i):
    cands = sorted((b for b in REGISTRY.values() if b.slot == slot), key=lambda b: -b.priority)
    sel = [b for b in cands if b.select(ctx, i)]
    return sel if slot in MOD_SLOTS or slot.endswith("_mod") else sel[:1]


def _compress(pats: dict) -> list:
    out = []
    for sfx, ls in sorted(pats.items()):
        ls = sorted(ls)
        if ls == [None]:
            out.append(sfx)
            continue
        rng, s = [], ls[0]
        for a, b in zip(ls, ls[1:] + [None]):
            if b != a + 1:
                rng.append(f"{s}" if s == a else f"{s}-{a}")
                s = b
        out.append(f"blk.[{','.join(rng)}].{sfx}")
    return out


def build_plan(g) -> Plan:
    """GGUF header -> block plan (selection, configs, claims, coverage audit)."""
    _load_blocks()
    ctx = Ctx(g)
    layers, configs, claimed = [], {}, {}

    def take(b, i):
        if b.config is not None:
            configs[(b.name, i)] = b.config(ctx, i)
        names = (ctx.glob if i is None else ctx.layers.get(i, ()))
        for sfx in b.claims:
            if sfx in names:
                claimed.setdefault(sfx if i is None else f"blk.{i}.{sfx}", b.name)

    glob = {}
    for slot in GLOBAL_SLOTS:
        sel = _pick(ctx, slot, None)
        for b in sel:
            take(b, None)
        glob[slot] = [b.name for b in sel] if slot.endswith("_mod") else (sel[0].name if sel else None)
    for i in range(ctx.nl):
        lp = {}
        for slot in BASE_SLOTS + MOD_SLOTS:
            sel = _pick(ctx, slot, i)
            for b in sel:
                take(b, i)
            lp[slot] = [b.name for b in sel] if slot in MOD_SLOTS else (sel[0].name if sel else None)
        layers.append(lp)
    unclaimed = []
    for name in ctx.infos:
        if name not in claimed:
            unclaimed.append(name if not name.startswith("blk.") or name.startswith("blk.0.") else re.sub(r"blk\.\d+", "blk.N", name))
    unclaimed = sorted(set(unclaimed))
    gaps = []
    if unclaimed:
        gaps.append(f"no builder for tensors: {', '.join(unclaimed[:12])}")
    unbuilt: dict = {}
    for name, bn in claimed.items():
        if REGISTRY[bn].build is None:
            mt = re.match(r"blk\.(\d+)\.(.+)$", name)
            unbuilt.setdefault(bn, {}).setdefault(mt.group(2) if mt else name, []).append(int(mt.group(1)) if mt else None)
    selected = {bn for lp in layers for v in lp.values() for bn in ([v] if isinstance(v, str) else (v or []))}
    selected |= {bn for v in glob.values() for bn in ([v] if isinstance(v, str) else (v or []))}
    for bn in sorted(selected):
        if REGISTRY[bn].build is None and bn not in unbuilt:
            unbuilt[bn] = {"(no tensor: key-selected)": [None]}
    unbuilt = {bn: _compress(p) for bn, p in unbuilt.items()}
    for bn, pats in sorted(unbuilt.items()):
        gaps.append(f"block {bn} not built ({REGISTRY[bn].step}): {', '.join(pats)}")
    return Plan(ctx.arch, layers, glob, configs, claimed, unclaimed, unbuilt, gaps, ctx)


def class_block(arch: str) -> Block | None:
    """Dedicated-class archs (loader/gguf_config.GGUF_ARCHS) are one 'model' block each: config = their mapping function."""
    return REGISTRY.get("class:" + arch)
