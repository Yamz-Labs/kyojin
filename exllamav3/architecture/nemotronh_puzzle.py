"""Nemotron-H Puzzle config adapter (torch-free).

Parses the per-layer ("Puzzle") block layout of NVIDIA Nemotron-H checkpoints
such as ``nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`` and builds a
config-only module graph (no weights, no device tensors) that the engine's
model builder can later materialize on meta/CUDA/HIP devices.

Supported checkpoint shapes
---------------------------
* Raw Puzzle config: ``model_type == "nemotron_h_puzzle"``
  (``NemotronHPuzzleForCausalLM``), tensor keys like
  ``backbone.layers.0.mixer.in_proj``.
* Omni wrapper: ``NemotronH_Omni_Reasoning_V3`` whose ``llm_config`` holds the
  Puzzle config above. Checkpoint keys carry the ``language_model.`` prefix,
  which this adapter strips. Vision keys (``vision_model.*``, ``mlp1.*``) are
  recognized and skipped (text-only port; vision tower is deferred).

Source: ``config.json`` + ``modeling_nemotron_h*.py`` of the Bolt checkpoint
(model_type ``nemotron_h_puzzle``, 88 blocks: 40 Mamba2 / 40 latent-MoE /
8 GQA-NoPE attention).

This module deliberately imports only the standard library so the adapter and
its unit test run anywhere (no torch, no compiled extension). The torch-based
engine wiring (``NemotronHPuzzleConfig`` / model builder reusing the
``Mamba2`` / ``Attention`` / ``BlockSparseMLP`` GENERIC blocks) lives in
``nemotronh_puzzle_engine.py`` and is built from the spec produced here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Top-level architecture string of the shipped Omni checkpoint.
OMNI_ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"
#: Architecture string of the raw Puzzle causal-LM config.
PUZZLE_ARCHITECTURE = "NemotronHPuzzleForCausalLM"
#: model_type of the LLM sub-config.
PUZZLE_MODEL_TYPE = "nemotron_h_puzzle"

#: Checkpoint key prefix of every LLM tensor inside the Omni wrapper.
LLM_KEY_PREFIX = "language_model."
#: Checkpoint key prefixes of the (currently skipped) vision tower + projector.
VISION_KEY_PREFIXES = ("vision_model.", "mlp1.")

#: Block types used in layers_block_type / block_configs.
BLOCK_MAMBA = "mamba"
BLOCK_MOE = "moe"
BLOCK_ATTENTION = "attention"

#: Bolt reference geometry (asserted by the CPU unit test).
BOLT_NUM_LAYERS = 88
BOLT_BLOCK_COUNTS = {BLOCK_MAMBA: 40, BLOCK_MOE: 40, BLOCK_ATTENTION: 8}
BOLT_ATTENTION_INDICES = (7, 16, 25, 36, 47, 58, 69, 78)


# --------------------------------------------------------------------------
# Key mapping
# --------------------------------------------------------------------------

def is_vision_key(key: str) -> bool:
    """True for checkpoint keys belonging to the skipped vision tower."""
    return key.startswith(VISION_KEY_PREFIXES)


def is_llm_key(key: str) -> bool:
    """True for checkpoint keys belonging to the language model."""
    return key.startswith(LLM_KEY_PREFIX)


def strip_llm_prefix(key: str) -> str:
    """Strip the Omni wrapper's ``language_model.`` prefix.

    Vision keys raise (they are skipped, not mapped); keys without any known
    prefix (bare Puzzle checkpoints) pass through unchanged.
    """
    if is_vision_key(key):
        raise ValueError(f"Vision key has no LLM mapping (skipped): {key!r}")
    if key.startswith(LLM_KEY_PREFIX):
        return key[len(LLM_KEY_PREFIX):]
    return key


def map_checkpoint_key(key: str) -> str | None:
    """Map a raw checkpoint key to the engine's (stripped) namespace.

    Returns None for vision keys (skip), the stripped key otherwise.
    """
    if is_vision_key(key):
        return None
    return strip_llm_prefix(key)


# --------------------------------------------------------------------------
# Parsed spec
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PuzzleBlockSpec:
    """Per-layer block description from ``block_configs``."""
    index: int
    block_type: str
    n_routed_experts: int = 0
    moe_intermediate_size: int = 0
    num_experts_per_tok: int = 0


@dataclass
class PuzzleSpec:
    """Full parsed LLM config: globals + one spec per layer."""
    model_type: str = PUZZLE_MODEL_TYPE
    from_omni_wrapper: bool = False
    hidden_size: int = 0
    vocab_size: int = 0
    num_hidden_layers: int = 0
    max_position_embeddings: int = 0
    layer_norm_epsilon: float = 1e-5
    # Attention (NoPE GQA)
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    # Mamba2
    mamba_num_heads: int = 0
    mamba_head_dim: int = 0
    ssm_state_size: int = 0
    n_groups: int = 0
    conv_kernel: int = 4
    # Latent MoE
    moe_latent_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    routed_scaling_factor: float = 0.0
    scoring_func: str = ""
    norm_topk_prob: bool = False
    blocks: list[PuzzleBlockSpec] = field(default_factory=list)

    @property
    def block_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in self.blocks:
            counts[b.block_type] = counts.get(b.block_type, 0) + 1
        return counts

    @property
    def attention_indices(self) -> tuple[int, ...]:
        return tuple(b.index for b in self.blocks if b.block_type == BLOCK_ATTENTION)

    def block_at(self, index: int) -> PuzzleBlockSpec:
        return self.blocks[index]

    def moe_blocks(self) -> list[PuzzleBlockSpec]:
        return [b for b in self.blocks if b.block_type == BLOCK_MOE]

    def distinct_moe_specs(self) -> list[tuple[int, int]]:
        return sorted({(b.n_routed_experts, b.moe_intermediate_size) for b in self.moe_blocks()})


def load_llm_config_dict(config_dict: dict) -> tuple[dict, bool]:
    """Unwrap the LLM sub-config.

    Returns (llm_config_dict, from_omni_wrapper). Raises ValueError on
    unknown architectures.
    """
    archs = config_dict.get("architectures", [])
    model_type = config_dict.get("model_type", "")
    if OMNI_ARCHITECTURE in archs or model_type == "nemotron_h_omni":
        llm = config_dict.get("llm_config")
        if not isinstance(llm, dict):
            raise ValueError("Omni config has no llm_config dict")
        return llm, True
    if PUZZLE_ARCHITECTURE in archs or model_type == PUZZLE_MODEL_TYPE:
        return config_dict, False
    raise ValueError(f"Unsupported config architectures={archs} model_type={model_type!r}")


def load_llm_config_file(path: str) -> tuple[dict, bool]:
    """Read config.json (or a directory containing it) and unwrap it."""
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    with open(path, encoding="utf8") as f:
        config_dict = json.load(f)
    return load_llm_config_dict(config_dict)


def parse_puzzle_spec(llm_config: dict, from_omni_wrapper: bool = False) -> PuzzleSpec:
    """Parse + structurally validate a raw Puzzle LLM config dict."""
    model_type = llm_config.get("model_type", "")
    if model_type != PUZZLE_MODEL_TYPE:
        raise ValueError(f"Expected model_type {PUZZLE_MODEL_TYPE!r}, got {model_type!r}")

    layers_block_type = llm_config.get("layers_block_type")
    block_configs = llm_config.get("block_configs")
    if not isinstance(layers_block_type, list) or not isinstance(block_configs, list):
        raise ValueError("Puzzle config needs layers_block_type + block_configs lists")
    if len(layers_block_type) != len(block_configs):
        raise ValueError(
            f"layers_block_type ({len(layers_block_type)}) != "
            f"block_configs ({len(block_configs)})"
        )

    blocks: list[PuzzleBlockSpec] = []
    for i, (btype, bcfg) in enumerate(zip(layers_block_type, block_configs)):
        if not isinstance(bcfg, dict) or bcfg.get("block_type") != btype:
            raise ValueError(f"Layer {i}: block_configs entry does not match {btype!r}")
        if btype not in (BLOCK_MAMBA, BLOCK_MOE, BLOCK_ATTENTION):
            raise ValueError(f"Layer {i}: unknown block_type {btype!r}")
        blocks.append(PuzzleBlockSpec(
            index=i,
            block_type=btype,
            n_routed_experts=int(bcfg.get("n_routed_experts", 0) or 0),
            moe_intermediate_size=int(bcfg.get("moe_intermediate_size", 0) or 0),
            num_experts_per_tok=int(bcfg.get("num_experts_per_tok", 0) or 0),
        ))

    num_q = int(llm_config["num_attention_heads"])
    hidden = int(llm_config["hidden_size"])
    return PuzzleSpec(
        model_type=model_type,
        from_omni_wrapper=from_omni_wrapper,
        hidden_size=hidden,
        vocab_size=int(llm_config["vocab_size"]),
        num_hidden_layers=len(blocks),
        max_position_embeddings=int(llm_config.get("max_position_embeddings", 0) or 0),
        layer_norm_epsilon=float(llm_config.get("layer_norm_epsilon", 1e-5)),
        num_attention_heads=num_q,
        num_key_value_heads=int(llm_config.get("num_key_value_heads", num_q)),
        head_dim=int(llm_config.get("head_dim", 0) or hidden // num_q),
        mamba_num_heads=int(llm_config.get("mamba_num_heads", 0) or 0),
        mamba_head_dim=int(llm_config.get("mamba_head_dim", 0) or 0),
        ssm_state_size=int(llm_config.get("ssm_state_size", 0) or 0),
        n_groups=int(llm_config.get("n_groups", 0) or 0),
        conv_kernel=int(llm_config.get("conv_kernel", 4)),
        moe_latent_size=int(llm_config.get("moe_latent_size", 0) or 0),
        moe_shared_expert_intermediate_size=int(
            llm_config.get("moe_shared_expert_intermediate_size", 0) or 0),
        routed_scaling_factor=float(llm_config.get("routed_scaling_factor", 0.0) or 0.0),
        scoring_func=str(llm_config.get("scoring_func", "")),
        norm_topk_prob=bool(llm_config.get("norm_topk_prob", False)),
        blocks=blocks,
    )


def load_puzzle_spec(path: str) -> PuzzleSpec:
    """Read config.json (file or directory) and return the validated spec."""
    llm_config, wrapped = load_llm_config_file(path)
    return parse_puzzle_spec(llm_config, from_omni_wrapper=wrapped)


# --------------------------------------------------------------------------
# Config-only module graph (no weights, no tensors)
# --------------------------------------------------------------------------

def block_key(index: int) -> str:
    return f"backbone.layers.{index}"


def iter_block_tensor_keys(node: PuzzleBlockSpec) -> "object":
    """Yield the expected (stripped) weight keys of one block.

    Names mirror both the HF reference (modeling_nemotron_h.py:
    NemotronHBlock/NemotronHMoE) and the engine's GENERIC blocks
    (Mamba2 in_proj/conv1d/A_log/dt_bias/D/norm/out_proj, Attention
    q/k/v/o_proj, BlockSparseMLP experts.{i}.up/down + gate +
    fc1/fc2_latent_proj + shared_experts), so the graph doubles as the
    loader's key checklist. Generator: MoE layers enumerate per-expert keys.
    """
    base = block_key(node.index)
    yield base + ".norm.weight"
    mixer = base + ".mixer"
    if node.block_type == BLOCK_MAMBA:
        for name in ("in_proj.weight", "out_proj.weight",
                     "conv1d.weight", "conv1d.bias",
                     "A_log", "D", "dt_bias", "norm.weight"):
            yield mixer + "." + name
    elif node.block_type == BLOCK_ATTENTION:
        for name in ("q_proj.weight", "k_proj.weight",
                     "v_proj.weight", "o_proj.weight"):
            yield mixer + "." + name
    elif node.block_type == BLOCK_MOE:
        yield mixer + ".gate.weight"
        yield mixer + ".gate.e_score_correction_bias"
        yield mixer + ".fc1_latent_proj.weight"
        yield mixer + ".fc2_latent_proj.weight"
        yield mixer + ".shared_experts.up_proj.weight"
        yield mixer + ".shared_experts.down_proj.weight"
        for e in range(node.n_routed_experts):
            yield f"{mixer}.experts.{e}.up_proj.weight"
            yield f"{mixer}.experts.{e}.down_proj.weight"
    else:  # pragma: no cover - parse_puzzle_spec rejects these
        raise ValueError(f"Unknown block_type {node.block_type!r}")


TOP_LEVEL_TENSOR_KEYS = (
    "backbone.embeddings.weight",
    "backbone.norm_f.weight",
    "lm_head.weight",
)


@dataclass
class PuzzleGraph:
    """Module graph from config only: one node per block + top-level keys.

    No weights are loaded and no tensors are allocated; per-layer MoE dims
    ride on the nodes so the engine builder can size each BlockSparseMLP.
    """
    spec: PuzzleSpec
    nodes: list[PuzzleBlockSpec] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        return len(self.nodes)

    def find(self, key: str) -> PuzzleBlockSpec:
        for n in self.nodes:
            if block_key(n.index) == key:
                return n
        raise KeyError(key)

    def count_tensors(self, materialize_moe_experts: bool = False) -> int:
        """Number of expected weight tensors (top-level + blocks).

        MoE expert keys are counted arithmetically unless materialized.
        """
        total = len(TOP_LEVEL_TENSOR_KEYS) + len(self.nodes)  # block norms
        for n in self.nodes:
            if n.block_type == BLOCK_MAMBA:
                total += 8
            elif n.block_type == BLOCK_ATTENTION:
                total += 4
            else:
                total += 6 + 2 * n.n_routed_experts
        return total


def build_puzzle_graph(spec: PuzzleSpec) -> PuzzleGraph:
    """Build the config-only module graph for a parsed spec."""
    return PuzzleGraph(spec=spec, nodes=list(spec.blocks))


def summarize(spec: PuzzleSpec) -> str:
    counts = spec.block_counts
    return (
        f"Nemotron-H Puzzle: {spec.num_hidden_layers} blocks "
        f"({counts.get(BLOCK_MAMBA, 0)} mamba / "
        f"{counts.get(BLOCK_MOE, 0)} moe / "
        f"{counts.get(BLOCK_ATTENTION, 0)} attn), "
        f"hidden={spec.hidden_size} vocab={spec.vocab_size} "
        f"moe_latent={spec.moe_latent_size} "
        f"{len(spec.distinct_moe_specs())} distinct MoE specs"
        + (" [omni wrapper]" if spec.from_omni_wrapper else "")
    )


def strip_llm_prefix_from_maps(
    tensor_file_map: dict,
    file_headers: dict,
) -> tuple[dict, dict]:
    """Rewrite loader maps into the engine's (stripped) key namespace.

    Pure-dict helper implementing the ``language_model.`` strip for the
    engine's SafetensorsCollection: both ``tensor_file_map`` and every per-file
    header (``get_tensor`` reads the tensor entry back out of the header).
    Vision keys (``vision_model.*``, ``mlp1.*``) are dropped - the text-only
    port does not load them. ``_header_offset`` entries are preserved.
    """
    new_map: dict = {}
    for key, filename in tensor_file_map.items():
        mapped = map_checkpoint_key(key)
        if mapped is None:
            continue
        new_map[mapped] = filename
    new_headers: dict = {}
    for filename, header in file_headers.items():
        renamed: dict = {}
        for key, entry in header.items():
            if key in ("__metadata__", "_header_offset"):
                continue
            mapped = map_checkpoint_key(key)
            if mapped is None:
                continue
            renamed[mapped] = entry
        if "_header_offset" in header:
            renamed["_header_offset"] = header["_header_offset"]
        new_headers[filename] = renamed
    return new_map, new_headers


# --------------------------------------------------------------------------
# Engine wiring (torch required; skipped on torch-free imports so the spec
# adapter above stays usable - and unit-testable - without torch)
# --------------------------------------------------------------------------

try:
    import torch as _torch
    from ..model.config import Config as _EngineConfig
    from ..model.model import Model as _EngineModel
    from ..modules import (
        RMSNorm as _RMSNorm,
        Embedding as _Embedding,
        TransformerBlock as _TransformerBlock,
        BlockSparseMLP as _BlockSparseMLP,
        MLP as _MLP,
        Linear as _Linear,
        GDNState as _GDNState,
    )
    from ..modules.attn import prepare_for_attn as _prepare_for_attn
    from ..cache.recurrent_util import (
        prepare_for_recurrence as _prepare_for_recurrence,
    )
    from .nemotronh import nemotronh_block as _nemotronh_block
    from ..modules.foreign_linear import ForeignLinear as _ForeignLinear
    _ENGINE_AVAILABLE = True
except ImportError:  # pragma: no cover - no torch on CPU-only test hosts
    _ENGINE_AVAILABLE = False

#: BOLT_FOREIGN=1 runs the shipped QAT weights as-is (ForeignLinear over the
#: U8 Q8_0 / IQ2_XXS / NVFP4 checkpoint tensors) instead of native Linear.
BOLT_FOREIGN = os.environ.get("BOLT_FOREIGN", "0") != "0"


def _fl(config, key: str, in_features: int, out_features: int, format: str,
        out_dtype=None):
    return _ForeignLinear(config, key, in_features, out_features, format,
                          out_dtype=out_dtype)


if _ENGINE_AVAILABLE:

    class NemotronHPuzzleConfig(_EngineConfig):
        """Engine config for the Omni-wrapped Puzzle checkpoint.

        Top-level architecture is ``NemotronH_Omni_Reasoning_V3``; the LLM
        hyper-parameters live in the nested ``llm_config`` dict, which becomes
        this config's ``config_dict`` after init. The loader maps are
        rewritten into the stripped namespace (``language_model.`` removed,
        vision keys dropped).
        """

        arch_string = OMNI_ARCHITECTURE

        def __init__(self, directory: str, **kwargs):
            super().__init__(directory, {"text": NemotronHPuzzleModel}, **kwargs)

            llm_config, wrapped = load_llm_config_dict(self.config_dict)
            if not wrapped:
                raise ValueError(
                    "NemotronHPuzzleConfig expects the Omni wrapper config "
                    f"({OMNI_ARCHITECTURE}); got a bare Puzzle config."
                )
            self.config_dict = llm_config

            # Re-read the base fields from the LLM dict (mirror Config).
            self.bos_token_id = self.read_cfg(int, ["bos_token_id"], None)
            self.eos_token_id = self.read_cfg([int, list], ["eos_token_id"], None)
            self.pad_token_id = self.read_cfg(int, ["pad_token_id"], None)
            self.vocab_size = self.read_cfg(int, ["vocab_size"], None)
            if isinstance(self.eos_token_id, list):
                self.eos_token_id_list = self.eos_token_id
                self.eos_token_id = self.eos_token_id[0]
            else:
                self.eos_token_id_list = [self.eos_token_id]
            self.eos_token_id_list = [e for e in self.eos_token_id_list if e is not None]
            self.max_position_embeddings = self.read_cfg(
                int, ["max_position_embeddings"],
                self.default_max_position_embeddings(),
            )

            self.puzzle_spec = parse_puzzle_spec(llm_config, from_omni_wrapper=True)
            spec = self.puzzle_spec

            self.hidden_size = spec.hidden_size
            self.num_q_heads = spec.num_attention_heads
            self.num_kv_heads = spec.num_key_value_heads
            self.head_dim = spec.head_dim
            self.mamba_num_heads = spec.mamba_num_heads
            self.mamba_head_dim = spec.mamba_head_dim
            self.ssm_state_size = spec.ssm_state_size
            self.n_groups = spec.n_groups
            self.conv_kernel = spec.conv_kernel
            self.time_step_limit = self.read_cfg(
                list, "time_step_limit", [0.0, float("inf")])
            self.intermediate_size = self.read_cfg(int, "intermediate_size", 0)
            self.moe_latent_size = spec.moe_latent_size or None
            self.shared_expert_intermediate_size = (
                spec.moe_shared_expert_intermediate_size)
            self.routed_scaling_factor = spec.routed_scaling_factor or 2.5
            self.layer_norm_epsilon = spec.layer_norm_epsilon
            self.num_hidden_layers = spec.num_hidden_layers
            self.tie_word_embeddings = self.read_cfg(
                bool, "tie_word_embeddings", False)

            (self.stc.tensor_file_map, self.stc.file_headers) = (
                strip_llm_prefix_from_maps(
                    self.stc.tensor_file_map, self.stc.file_headers))

        def get_tensor_name_fixes(self):
            return {}


    def nemotronh_puzzle_block(config, key: str, layer_idx: int,
                               block_spec=None):
        """One hybrid block with per-layer (Puzzle) MoE dims.

        Mamba ("M") and attention ("*") reuse the GENERIC ``nemotronh_block``
        builders; MoE ("E") sizes this layer's ``BlockSparseMLP`` from its own
        ``block_configs`` entry instead of a global MoE size.
        """
        spec = block_spec or config.puzzle_spec.block_at(layer_idx)
        if spec.block_type == BLOCK_MAMBA:
            return _nemotronh_block(config, key, layer_idx, "M", foreign=BOLT_FOREIGN)
        if spec.block_type == BLOCK_ATTENTION:
            return _nemotronh_block(config, key, layer_idx, "*", foreign=BOLT_FOREIGN)
        if spec.block_type != BLOCK_MOE:
            raise ValueError(f"Unknown Puzzle block_type {spec.block_type!r}")
        norm = _RMSNorm(
            config=config,
            key=f"{key}.norm",
            rms_norm_eps=config.layer_norm_epsilon,
        )
        if BOLT_FOREIGN:
            hid, lat = config.hidden_size, config.moe_latent_size
            inter, sh_inter = spec.moe_intermediate_size, config.shared_expert_intermediate_size
            n_exp = spec.n_routed_experts
            fups = [_fl(config, f"{key}.mixer.experts.{i}.up_proj", lat, inter, "iq2_xxs")
                    for i in range(n_exp)]
            fdowns = [_fl(config, f"{key}.mixer.experts.{i}.down_proj", inter, lat, "iq2_xxs",
                           out_dtype=_torch.float) for i in range(n_exp)]
            flat_in = _fl(config, f"{key}.mixer.fc1_latent_proj", hid, lat, "q8_0")
            flat_out = _fl(config, f"{key}.mixer.fc2_latent_proj", lat, hid, "q8_0",
                            out_dtype=_torch.float)
            sh_up = _fl(config, f"{key}.mixer.shared_experts.up_proj", hid, sh_inter, "iq2_xxs")
            sh_down = _fl(config, f"{key}.mixer.shared_experts.down_proj", sh_inter, hid, "iq2_xxs",
                           out_dtype=_torch.float)
        else:
            fups = fdowns = flat_in = flat_out = sh_up = sh_down = None
        # Prebuilt foreign lists bypass the internal builders, which are also
        # the only register_submodule call sites: register them here so
        # Module.load() recurses into them (routing gate + latent projs
        # register themselves; MLP prebuilts never do).
        mlp = _BlockSparseMLP(
            config=config,
            key=f"{key}.mixer",
                hidden_size=config.hidden_size,
                intermediate_size=spec.moe_intermediate_size,
                num_experts=spec.n_routed_experts,
                num_experts_per_tok=spec.num_experts_per_tok,
                latent_size=config.moe_latent_size,
                key_latent_in=None if BOLT_FOREIGN else ("fc1_latent_proj" if config.moe_latent_size else None),
                key_latent_out=None if BOLT_FOREIGN else ("fc2_latent_proj" if config.moe_latent_size else None),
                latent_in=flat_in,
                latent_out=flat_out,
                latent_hq_bits=2,
                key_up=None if BOLT_FOREIGN else "experts.{expert_idx}.up_proj",
                key_down=None if BOLT_FOREIGN else "experts.{expert_idx}.down_proj",
                gates=[] if BOLT_FOREIGN else None,
                ups=fups,
                downs=fdowns,
                key_routing_gate="gate",
                activation_fn="relu2",
                router_type="dots",
                routed_scaling_factor=config.routed_scaling_factor,
                n_group=1,
                topk_group=1,
                qmap="block.mlp",
                interm_dtype=_torch.half,
                out_dtype=_torch.float,
                shared_experts=_MLP(
                    config=config,
                    key=f"{key}.mixer.shared_experts",
                    hidden_size=config.hidden_size,
                    intermediate_size=config.shared_expert_intermediate_size,
                    key_up=None if BOLT_FOREIGN else "up_proj",
                    key_down=None if BOLT_FOREIGN else "down_proj",
                    ups=[sh_up] if BOLT_FOREIGN else None,
                    downs=[sh_down] if BOLT_FOREIGN else None,
                    activation_fn="relu2",
                    qmap="block.mlp",
                    out_dtype=_torch.float,
                    select_hq_bits=2,
                ),
            )
        if BOLT_FOREIGN:
            for m in fups + fdowns:
                mlp.register_submodule(m)
            mlp.shared_experts.register_submodule(sh_up)
            mlp.shared_experts.register_submodule(sh_down)
        return _TransformerBlock(
            config=config,
            key=key,
            layer_idx=layer_idx,
            mlp_norm=norm,
            mlp=mlp,
        )


    class NemotronHPuzzleModel(_EngineModel):
        """Text-only model over the stripped key namespace."""

        config_class = NemotronHPuzzleConfig

        def __init__(self, config: NemotronHPuzzleConfig, **kwargs):
            super().__init__(config, **kwargs)

            self.modules += [
                _Embedding(
                    config=config,
                    key="backbone.embeddings",
                    vocab_size=config.vocab_size,
                    hidden_size=config.hidden_size,
                )
            ]

            self.first_block_idx = len(self.modules)

            for idx in range(config.num_hidden_layers):
                self.modules.append(nemotronh_puzzle_block(
                    config, f"backbone.layers.{idx}", idx))

            self.last_kv_module_idx = len(self.modules) - 1

            head_alt_key = None
            if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
                head_alt_key = "backbone.embeddings"

            if BOLT_FOREIGN:
                head = _fl(config, "lm_head", config.hidden_size, config.vocab_size, "q8_0")
                head.caps["logits_output"] = True
            else:
                head = _Linear(
                    config=config,
                    key="lm_head",
                    qbits_key="head_bits",
                    alt_key=head_alt_key,
                    in_features=config.hidden_size,
                    out_features=config.vocab_size,
                    qmap="block",
                    caps={"logits_output": True},
                )
            self.modules += [
                _RMSNorm(
                    config=config,
                    key="backbone.norm_f",
                    rms_norm_eps=config.layer_norm_epsilon,
                    out_dtype=_torch.half,
                ),
                head,
            ]

            self.logit_layer_idx = len(self.modules) - 1
            self.calibration_all_experts = True

            self.caps.update({
                "recurrent_states": True,
                "default_recurrent_checkpoint_interval": 2048,
                "linear_attn": True,
            })
            self.recurrent_state_cls = _GDNState
            self.caps.update({"supports_tp": True})

        def prepare_inputs(self, input_ids, params):
            input_ids = _prepare_for_attn(input_ids, params)
            _prepare_for_recurrence(input_ids, params, self)
            return input_ids

        def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
            p = ""
            if system_prompt:
                p += "<|im_start|>system\n"
                p += f"{system_prompt}<|im_end|>\n"
            p += "<|im_start|>user\n"
            p += f"{prompt}<|im_end|>\n"
            p += "<|im_start|>assistant\n"
            return p
