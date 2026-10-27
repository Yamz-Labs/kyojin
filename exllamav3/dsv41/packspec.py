"""
Single specification of a converted DS4.1 pack: tensor names, shapes, shard layout, `quantization_config` fields and the
Engram table files. The converter (tools/ds41), the synthetic pack writer (synth_pack.py), the pack validator
(tools/ds41/validate_pack.py) and the tests all derive their names from here; the serving model's module tree
(architecture/deepseek_v41.py, modules/dsv41.py) is checked against it by tests/dsv41/test_dsv41_packwire.py.

Standard library only (no torch, no exllamav3 imports): tools/ds41 loads this file by path (`load_by_path`) so the
converter's --dry-run keeps working without the engine installed. `cfg` is a `DSV41Config` (duck typed).

Pack layout
    <pack>/config.json                      HF config, `text_config` + `quantization_config` (see `build_qcfg`)
    <pack>/model.safetensors.index.json     weight_map: tensor -> shard
    <pack>/global.safetensors               embed.weight, norm.weight, head.{trellis,suh,svh,mul1}
    <pack>/layers.N.safetensors             one trunk layer (attention, hc params, router, shared + routed experts,
                                            engram wkv / q_weight / k_weight)
    <pack>/mtp.J.safetensors                dspark stage J (+ main_proj / main_norm on the first, norm / markov head /
                                            confidence head on the last stage)
    <pack>/engram/layers.N.engram.embed.{weight,scale}.bin   Engram tables, NOT in the shards, NOT in the index; read with
                                            O_DIRECT row gathers (`quantization_config.engram` names the directory)
    <pack>/plan.json, quantization_config.json   converter bookkeeping (optional for the loader)
A linear is stored as `<key>.{trellis,suh,svh,mul1}` (EXL3) or `<key>.weight` (plain, synthetic packs / unquantized).
`wo_a` is stored as G slices `attn.wo_a.slice.g` (G = o_groups). A pruned routed expert (K = 0 in
`quantization_config.expert_bits`) has no tensors; the router bias of the layer masks it (value <= MASK_THRESHOLD).
"""
from __future__ import annotations
import importlib.util
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field

PACK_FORMAT = 1
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mul1")
MUL1_MULT = 0x83DCD12D                # codebook multiplier of the "mul1" codebook (stored per linear as an int32)
MASK_BIAS = -1e4                      # router bias written for pruned experts (never selected)
MASK_THRESHOLD = -1e3                 # validator: a pruned expert's bias must be at most this
ENGRAM_DIR = "engram"
ENGRAM_FMT = "mxfp4"                  # rows: head_dim/2 value bytes (e2m1 pairs) + head_dim/32 e8m0 scale bytes
FLOAT_DTYPES = ("F32", "BF16", "F16")
K_MIN, K_MAX = 1, 8                   # trellis bits per weight the format can store (pack.cu instances 1..8)

RE_EXPERT_KEY = re.compile(r"^((?:layers|mtp)\.\d+)\.ffn\.experts\.(\d+)\.(w[123])$")
RE_LAYER = re.compile(r"^(layers|mtp)\.(\d+)\.")


def load_by_path(name: str = "packspec"):
    """Import this module from its file (tools use it without importing the exllamav3 package)."""
    if __name__ in sys.modules and __name__ != "__main__":
        return sys.modules[__name__]
    key = f"_ds41_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.abspath(__file__))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ EXL3 container
def exl3_tensors(in_f: int, out_f: int, k: int) -> dict:
    """{suffix: (safetensors dtype, shape)} of one EXL3 linear at k bits per weight. in_f / out_f are the 128-aligned
    (padded) dims; mul1 is a 0-dim int32 (the codebook multiplier), see quantize_exl3."""
    assert in_f % 16 == 0 and out_f % 16 == 0 and K_MIN <= k <= K_MAX, (in_f, out_f, k)
    return {"trellis": ("I16", [in_f // 16, out_f // 16, 16 * k]), "suh": ("F16", [in_f]), "svh": ("F16", [out_f]),
            "mul1": ("I32", [])}


def exl3_bytes(in_f: int, out_f: int, k: int) -> int:
    """Bytes of one EXL3 linear (trellis int16 [in/16, out/16, 16k], suh, svh fp16, mul1 int32)."""
    return (in_f // 16) * (out_f // 16) * 16 * k * 2 + in_f * 2 + out_f * 2 + 4


def exl3_trellis_shape(in_f: int, out_f: int, k: int) -> list:
    return [in_f // 16, out_f // 16, 16 * k]


def k_of_trellis_shape(shape) -> int:
    assert len(shape) == 3 and shape[2] % 16 == 0, shape
    return shape[2] // 16


def pad128(n: int) -> int:
    return -(-n // 128) * 128


# ------------------------------------------------------------------------------------------------ units / shards
def unit_of(name: str) -> str:
    """Shard unit of a tensor name: global | layers.N | mtp.J (Engram tables are not in shards)."""
    m = RE_LAYER.match(name)
    return f"{m.group(1)}.{m.group(2)}" if m else "global"


def shard_file(unit: str) -> str:
    return f"{unit}.safetensors"


def unit_order(unit: str):
    if unit == "global":
        return (0, 0)
    pre, idx = unit.split(".")
    return (1 if pre == "layers" else 2, int(idx))


def is_engram_table(name: str) -> bool:
    """`layers.N.engram.embed.{weight,scale}`: the hashed n-gram tables (the engram projection and q/k weights are
    ordinary layer tensors)."""
    return re.match(r"^layers\.\d+\.engram\.embed\.(weight|scale)$", name) is not None


def engram_files(layer: int) -> tuple:
    """(value file, scale file) of the Engram tables of a layer inside the engram directory."""
    return f"layers.{layer}.engram.embed.weight.bin", f"layers.{layer}.engram.embed.scale.bin"


def engram_row_bytes(head_dim: int, fmt: str = ENGRAM_FMT) -> tuple:
    """(value bytes, scale bytes) per table row."""
    return (head_dim // 2 if fmt == "mxfp4" else head_dim), head_dim // 32


# ------------------------------------------------------------------------------------------------ tensor inventory
@dataclass(frozen=True)
class Lin:
    key: str                 # pack key (EXL3: key.trellis ...; plain: key.weight)
    in_f: int
    out_f: int               # unpadded dims as the framework Linear sees them
    role: str                # head | attn | wo_a | indexer | shared | routed | engram | mtp | mtp_routed | main_proj
    unit: str
    src: str                 # official tensor name (`<src>.weight` + `<src>.scale` in the source checkpoint)
    rows: tuple = None       # row range of the source weight (wo_a slices)
    layer: int = None        # graph layer id (trunk layer, or n_layers + j for mtp stages)
    expert: int = None

    @property
    def in_pad(self):
        return pad128(self.in_f)

    @property
    def out_pad(self):
        return pad128(self.out_f)


@dataclass(frozen=True)
class Raw:
    name: str
    shape: tuple
    unit: str
    optional: bool = False


def _stage(cfg, l: int):
    """(key prefix, is_mtp, stage index) of graph layer l."""
    n = cfg.num_hidden_layers
    return (f"layers.{l}", False, l) if l < n else (f"mtp.{l - n}", True, l - n)


def block_linears(cfg, l: int):
    """Linears of graph layer l (a trunk layer or a dspark stage), routed experts included."""
    key, mtp, _ = _stage(cfg, l)
    d, hd, H, G = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.o_groups
    u = key
    a = f"{key}.attn"
    role_mtp = "mtp" if mtp else "attn"
    yield Lin(f"{a}.wq_a", d, cfg.q_lora_rank, role_mtp, u, f"{a}.wq_a", layer=l)
    yield Lin(f"{a}.wkv", d, hd, role_mtp, u, f"{a}.wkv", layer=l)
    yield Lin(f"{a}.wq_b", cfg.q_lora_rank, H * hd, role_mtp, u, f"{a}.wq_b", layer=l)
    gw = H * hd // G
    for g in range(G):
        yield Lin(f"{a}.wo_a.slice.{g}", gw, cfg.o_lora_rank, "wo_a" if not mtp else "mtp", u, f"{a}.wo_a",
                  (g * cfg.o_lora_rank, (g + 1) * cfg.o_lora_rank), layer=l)
    yield Lin(f"{a}.wo_b", G * cfg.o_lora_rank, d, role_mtp, u, f"{a}.wo_b", layer=l)
    if not mtp and cfg.is_index_source(l):
        yield Lin(f"{a}.indexer.wq_b", cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim, "indexer", u,
                  f"{a}.indexer.wq_b", layer=l)
    isz = cfg.moe_intermediate_size * cfg.n_shared_experts
    for nm, i, o in (("w1", d, isz), ("w3", d, isz), ("w2", isz, d)):
        yield Lin(f"{key}.ffn.shared_experts.{nm}", i, o, "shared" if not mtp else "mtp", u,
                  f"{key}.ffn.shared_experts.{nm}", layer=l)
    if not mtp and l in cfg.engram_layer_ids:
        nc = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        yield Lin(f"{key}.engram.wkv", nc * cfg.engram_head_dim, d * (cfg.hc_mult + 1), "engram", u,
                  f"{key}.engram.wkv", layer=l)
    ne, _ = cfg.moe_counts(l)
    ei = cfg.moe_intermediate_size
    for e in range(ne):
        for nm, i, o in (("w1", d, ei), ("w3", d, ei), ("w2", ei, d)):
            yield Lin(f"{key}.ffn.experts.{e}.{nm}", i, o, "mtp_routed" if mtp else "routed", u,
                      f"{key}.ffn.experts.{e}.{nm}", layer=l, expert=e)


def block_raws(cfg, l: int):
    key, mtp, _ = _stage(cfg, l)
    d, hd, H = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads
    mix = (2 + cfg.hc_mult) * cfg.hc_mult
    a = f"{key}.attn"
    u = key
    for t in ("attn", "ffn"):
        yield Raw(f"{key}.hc_{t}_fn", (mix, cfg.hc_mult * d), u)
        yield Raw(f"{key}.hc_{t}_base", (mix,), u)
        yield Raw(f"{key}.hc_{t}_scale", (3,), u)
    yield Raw(f"{key}.attn_norm.weight", (d,), u)
    yield Raw(f"{key}.ffn_norm.weight", (d,), u)
    yield Raw(f"{a}.attn_sink", (H,), u)
    yield Raw(f"{a}.q_norm.weight", (cfg.q_lora_rank,), u)
    yield Raw(f"{a}.kv_norm.weight", (hd,), u)
    if not mtp and cfg.is_kv_source(l):
        yield Raw(f"{a}.compressor.wkv.weight", (hd, d), u)
        yield Raw(f"{a}.compressor.norm.weight", (hd,), u)
        yield Raw(f"{a}.indexer.wk.weight", (cfg.index_head_dim, hd), u)
        yield Raw(f"{a}.indexer.k_norm.weight", (cfg.index_head_dim,), u)
        if cfg.ratio(l) > 1:
            yield Raw(f"{a}.compressor.wgate.weight", (hd, d), u)
    if not mtp and cfg.is_index_source(l):
        yield Raw(f"{a}.indexer.weights_proj.weight", (cfg.index_n_heads, d), u)
    ne, _ = cfg.moe_counts(l)
    yield Raw(f"{key}.ffn.gate.weight", (ne, d), u)
    yield Raw(f"{key}.ffn.gate.bias", (ne,), u)
    yield Raw(f"{key}.ffn.gate.bias_vl", (ne,), u, optional=True)
    if not mtp and l in cfg.engram_layer_ids:
        yield Raw(f"{key}.engram.q_weight", (cfg.hc_mult, d), u)
        yield Raw(f"{key}.engram.k_weight", (cfg.hc_mult, d), u)


def global_linears(cfg):
    yield Lin("head", cfg.hidden_size, cfg.vocab_size, "head", "global", "head")


def global_raws(cfg):
    yield Raw("embed.weight", (cfg.vocab_size, cfg.hidden_size), "global")
    yield Raw("norm.weight", (cfg.hidden_size,), "global")


def mtp_extra_linears(cfg):
    if cfg.n_mtp_layers:
        yield Lin("mtp.0.main_proj", cfg.hidden_size * len(cfg.dspark_target_layer_ids), cfg.hidden_size, "main_proj",
                  "mtp.0", "mtp.0.main_proj", layer=cfg.num_hidden_layers)


def mtp_extra_raws(cfg):
    if cfg.n_mtp_layers:
        d, last = cfg.hidden_size, f"mtp.{cfg.n_mtp_layers - 1}"
        yield Raw("mtp.0.main_norm.weight", (d,), "mtp.0")
        yield Raw(f"{last}.norm.weight", (d,), last)
        yield Raw(f"{last}.markov_head.embed.weight", (cfg.vocab_size, cfg.dspark_markov_rank), last)
        yield Raw(f"{last}.markov_head.head.weight", (cfg.vocab_size, cfg.dspark_markov_rank), last)
        yield Raw(f"{last}.confidence_head.proj.weight", (1, d + cfg.dspark_markov_rank), last)


def linears(cfg, mtp: bool = True):
    """Every linear the serving model builds (trunk, head, and the dspark stages when `mtp`)."""
    yield from global_linears(cfg)
    for l in range(cfg.num_hidden_layers):
        yield from block_linears(cfg, l)
    if mtp:
        for j in range(cfg.n_mtp_layers):
            yield from block_linears(cfg, cfg.num_hidden_layers + j)
        yield from mtp_extra_linears(cfg)


def raws(cfg, mtp: bool = True):
    yield from global_raws(cfg)
    for l in range(cfg.num_hidden_layers):
        yield from block_raws(cfg, l)
    if mtp:
        for j in range(cfg.n_mtp_layers):
            yield from block_raws(cfg, cfg.num_hidden_layers + j)
        yield from mtp_extra_raws(cfg)


def engram_rows(cfg) -> list:
    """Rows of each Engram table (cfg.engram_num_embeddings, one per engram layer)."""
    return list(cfg.engram_num_embeddings or [])


# ------------------------------------------------------------------------------------------------ quantization_config
def layer_ffn_key(layer_key: str) -> str:
    return f"{layer_key}.ffn"


def expert_bits_from_entries(entries, n_experts: int = None) -> dict:
    """{"layers.L.ffn": [K per routed expert, 0 = pruned]} from plan entries (tools/ds41/plan.py). Only layers whose
    experts are all present are listed. `n_experts` None: take it from the entries."""
    per = {}
    for e in entries:
        if e.get("kind") != "routed" or e.get("action") not in ("quant", "pruned"):
            continue
        k = e.get("key") or e.get("weight", "").removesuffix(".weight")
        m = RE_EXPERT_KEY.match(k)
        if m and m.group(1).startswith("layers.") and m.group(3) == "w1":
            per.setdefault(m.group(1), {})[int(m.group(2))] = int(e["bits"])
    out = {}
    for layer, d in sorted(per.items(), key=lambda kv: int(kv[0].split(".")[1])):
        n = n_experts or (max(d) + 1)
        if len(d) == n and set(d) == set(range(n)):
            out[layer_ffn_key(layer)] = [d[i] for i in range(n)]
    return out


def pruned_from_expert_bits(expert_bits: dict) -> dict:
    return {k: [i for i, b in enumerate(v) if b == 0] for k, v in expert_bits.items() if any(b == 0 for b in v)}


def build_qcfg(*, bits, head_bits, mtp_bits, expert_bits=None, engram=None, plan=None, original=None, extra=None) -> dict:
    """The `quantization_config` the converter writes and the loader reads.
    expert_bits {"layers.L.ffn": [K...]} (0 = pruned); engram {"dir", "format", "layers", "rows", "head_dim"}."""
    q = {"quant_method": "exl3", "pack_format": PACK_FORMAT, "bits": bits, "head_bits": head_bits, "mtp_bits": mtp_bits,
         "codebook": "mul1", "out_scales": "always"}
    if plan is not None:
        q["plan"] = plan
    if expert_bits:
        q["expert_bits"] = {k: list(v) for k, v in expert_bits.items()}
        pr = pruned_from_expert_bits(q["expert_bits"])
        if pr:
            q["pruned_experts"] = pr
    if engram:
        q["engram"] = dict(engram)
    if original is not None:
        q["original_quantization_config"] = original
    q.update(extra or {})
    return q


def expert_bits_of(qcfg: dict, layer_key: str, n_experts: int):
    """K per expert of `layer_key` ("layers.3.ffn") or None (the layer's experts then all carry the tensors' own K)."""
    t = (qcfg or {}).get("expert_bits") or {}
    if layer_key not in t:
        return None
    b = t[layer_key]
    if not isinstance(b, list) or len(b) != n_experts or any(not isinstance(x, (int, float)) for x in b):
        raise ValueError(f"expert_bits[{layer_key!r}]: expected {n_experts} numbers")
    return [int(x) if float(x) == int(x) else x for x in b]


def pruned_set(qcfg: dict, layer_key: str, n_experts: int) -> frozenset:
    b = expert_bits_of(qcfg, layer_key, n_experts)
    return frozenset(i for i, k in enumerate(b) if k == 0) if b else frozenset()


def engram_cfg(qcfg: dict) -> dict:
    return dict((qcfg or {}).get("engram") or {})


def resolve_engram_dir(pack_dir: str, qcfg: dict = None, override: str = None):
    """Directory holding the Engram table files, in this order: explicit `override` (kwarg), $EXL3_DSV41_ENGRAM_DIR,
    `quantization_config.engram.dir` (relative to the pack), `<pack>/engram`. None when none exists (the loader then
    falls back to tables inside the shards, the older layout)."""
    cands = [override, os.environ.get("EXL3_DSV41_ENGRAM_DIR")]
    d = engram_cfg(qcfg).get("dir")
    if d:
        cands.append(d if os.path.isabs(d) else os.path.join(pack_dir, d))
    cands.append(os.path.join(pack_dir, ENGRAM_DIR))
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


# ------------------------------------------------------------------------------------------------ expected pack
@dataclass
class Expected:
    name: str
    dtype: str                # safetensors dtype ("*" = any float)
    shape: tuple
    kind: str                 # exl3 | plain | raw
    key: str = None           # owning linear key
    optional: bool = False


def lin_k(lin: Lin, expert_bits=None, module_bits=None, default_k: int = None):
    """K of a linear: expert_bits for routed experts, module_bits (key -> K) for the rest, else `default_k`."""
    if lin.expert is not None and expert_bits:
        lk = expert_bits.get(layer_ffn_key(lin.unit))
        if lk is not None:
            return lk[lin.expert]
    if module_bits and lin.key in module_bits:
        return module_bits[lin.key]
    return default_k


def expected_tensors(cfg, *, exl3: bool = True, expert_bits=None, module_bits=None, default_k: int = None,
                     mtp: bool = True) -> dict:
    """{tensor name: Expected} of a complete pack. exl3=False: plain `.weight` linears (synthetic fp16 pack).
    `module_bits` / `default_k` give K for non-expert linears when known (shape checks only if K is given)."""
    out = {}
    for lin in linears(cfg, mtp=mtp):
        k = lin_k(lin, expert_bits, module_bits, default_k)
        if lin.expert is not None and k == 0:
            continue                                    # pruned expert: no tensors
        if not exl3:
            out[lin.key + ".weight"] = Expected(lin.key + ".weight", "*", (lin.out_f, lin.in_f), "plain", lin.key)
            continue
        if k is None:                                   # K unknown: the suffix set and the dims that do not depend on K
            for sfx, (dt, shp) in (("suh", ("F16", [lin.in_pad])), ("svh", ("F16", [lin.out_pad])),
                                   ("mul1", ("I32", [])), ("trellis", ("I16", None))):
                out[f"{lin.key}.{sfx}"] = Expected(f"{lin.key}.{sfx}", dt, tuple(shp) if shp is not None else None,
                                                   "exl3", lin.key)
            continue
        for sfx, (dt, shp) in exl3_tensors(lin.in_pad, lin.out_pad, k).items():
            out[f"{lin.key}.{sfx}"] = Expected(f"{lin.key}.{sfx}", dt, tuple(shp), "exl3", lin.key)
    for r in raws(cfg, mtp=mtp):
        out[r.name] = Expected(r.name, "*", tuple(r.shape), "raw", None, r.optional)
    return out


def shard_of(cfg, name: str) -> str:
    return shard_file(unit_of(name))


# ------------------------------------------------------------------------------------------------ index / safetensors headers
def write_index(path: str, weight_map: dict, total_size: int):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)
    os.replace(tmp, path)


def read_header(path: str) -> dict:
    """safetensors header ({name: {dtype, shape, data_offsets}}, `__metadata__` dropped)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def header_data_start(path: str) -> int:
    with open(path, "rb") as f:
        return 8 + struct.unpack("<Q", f.read(8))[0]
