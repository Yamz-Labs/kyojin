"""Generic GGUF loader: the composition of the block registry (loader/gguf_registry.py, loader/gguf_blocks/), no per-model class.

Off by default: only active with EXL3_GGUF_GENERIC=1, and only for a GGUF whose general.architecture has no entry in
loader/gguf_config.GGUF_ARCHS. Everything is derived from the GGUF header:
  metadata  <arch>.block_count / embedding_length / attention.head_count(_kv) / key_length / rope.* / expert_count / ...
  tensors   blk.N.attn_{q,k,v,output,q_norm,k_norm} / ffn_{gate,up,down}[_exps] / ffn_gate_inp / *_norm
Anything it cannot map is reported in cfg["gaps"] (named missing piece); the Config refuses to build while gaps remain
unless EXL3_GGUF_GENERIC_ALLOW_GAPS=1 (dry-run / test use).
"""
from __future__ import annotations
import os
import re
from typing_extensions import override

import torch

from ..loader.gguf_collection import GGUFCollection
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..loader.gguf_registry import ARCH_FEATS, GenericUnsupported, REGISTRY, CONFIG_ONLY, build_plan, req
from ..loader.gguf_blocks.attn import gqa_global
from ..loader.gguf_blocks.transformer import residual_global
from ..loader.gguf_blocks.mlp import moe_global, moe_layers
from ..modules.attn import prepare_for_attn

ARCH_STRING = "GGUFGenericForCausalLM"
_ARCH_FEATS = ARCH_FEATS     # per-arch residue: loader/gguf_registry.ARCH_FEATS (only what llama.cpp hard-codes with no key)


def enabled() -> bool:
    return os.environ.get("EXL3_GGUF_GENERIC", "0") == "1"


def _req(m: dict, key: str):
    return req(m, key)


def hf_config_generic(g: GGUFCollection) -> dict:
    """GGUF header -> HF-style config dict for GGUFGenericForCausalLM, composed from the block plan; unmapped pieces go to cfg['gaps']."""
    m = g.meta
    arch = _req(m, "general.architecture")
    p = arch + "."
    _req(m, p + "embedding_length"), _req(m, p + "attention.head_count")
    nl = _req(m, p + "block_count")
    plan = build_plan(g)
    ctx = plan.ctx
    a = gqa_global(ctx)
    gaps = ctx.gaps
    if sorted(ctx.layers) != list(range(nl)):
        gaps.append(f"blk indices {sorted(ctx.layers)[:3]}.. do not cover block_count={nl} (nextn/MTP or non-uniform stack)")
    if p + "ssm.state_size" in m:
        gaps.append(f"metadata {p}ssm.state_size (ssm / mamba layers) has no module wiring")
    if m.get(p + "attention.shared_kv_layers", 0):
        gaps.append(f"shared_kv_layers={m[p + 'attention.shared_kv_layers']} (KV sharing across layers) not wired")
    if m.get(p + "embedding_length_per_layer_input", 0) and not any(lp["pre"] == "engram" for lp in plan.layers):
        gaps.append("per-layer input embeddings (PLE) not wired")
    res = residual_global(ctx)
    emb = plan.configs.get(("token_embd", None)) or REGISTRY["token_embd"].config(ctx, None)
    head = plan.configs.get(("lm_head", None)) or REGISTRY["lm_head"].config(ctx, None)
    ml = moe_layers(ctx)
    moe = moe_global(ctx)
    cfg = {"architectures": [ARCH_STRING], "model_type": "gguf_generic", "gguf_arch": arch, **a,
           "hidden_act": ctx.feats.get("act", "silu"), "embed_scale": emb["embed_scale"], **res, **head,
           "vocab_size": emb["vocab_size"], "moe_layers": ml, "intermediate_size": m.get(p + "feed_forward_length", 0), **moe}
    if len(ml) not in (0, nl):
        cfg["dense_layers"] = [i for i in range(nl) if i not in ml]
    cfg["gaps"] = plan.gaps
    return cfg


def name_map_generic(cfg: dict, plan=None) -> dict:
    """engine key -> gguf tensor name, the union of the selected blocks' name maps; routed experts use per-expert slice keys (.eN)."""
    nm = {}
    if plan is None:      # config-only callers (tests): the plan of the stack the config describes
        return _name_map_fixed(cfg)
    for slot in ("embed", "head"):
        b = REGISTRY[plan.glob[slot]]
        nm.update(b.names(cfg, None, None, None))
    for i in range(cfg["num_hidden_layers"]):
        L, G = f"model.layers.{i}", f"blk.{i}"
        for bn in plan.blocks_at(i):
            b = REGISTRY[bn]
            if b.names is not None:
                nm.update(b.names(cfg, i, L, G))
    return nm


def _name_map_fixed(cfg: dict) -> dict:
    """the plain residual / gqa / dense-or-moe stack the generic config dict describes (no header at hand)"""
    from ..loader.gguf_blocks import transformer as T, attn as A, mlp as MM
    nm = {**T._head_names(cfg, None, None, None), "model.embed_tokens.weight": "token_embd.weight"}
    moe = set(cfg["moe_layers"])
    for i in range(cfg["num_hidden_layers"]):
        L, G = f"model.layers.{i}", f"blk.{i}"
        nm.update(T._residual_names(cfg, i, L, G))
        nm.update(A._names(cfg, i, L, G))
        nm.update(MM._moe_names(cfg, i, L, G) if i in moe else MM._dense_names(cfg, i, L, G))
    return nm


def _fl(config, eng_key: str, i: int, o: int, out_dtype=None, bias=False, **kw):
    from ..modules.foreign_linear import ForeignLinear
    from ..loader.gguf_config import GGML_FMT
    gname = config.gguf.name_map[eng_key + ".weight"]
    base = re.sub(r"\.e\d+(\.weight)$", r"\1", gname) if "_exps.e" in gname else gname
    fmt = config.gguf_formats[base]
    if fmt == "fp" or fmt.startswith("unk"):
        raise GenericUnsupported(f"{base}: ggml type {fmt!r} has no ForeignLinear format (add it to GGML_FMT)")
    if bias:
        # Linear.load_gguf_ builds the ForeignLinear on the stored blocks and loads <key>.bias into _ForeignInner (inner.bias / inner.fl,
        # what the grouped MXFP4 MoE path and Attention read). pad_to=1: the GGUF payload is not padded to 128 (gpt-oss 2880).
        from ..modules.linear import Linear
        assert not kw
        return Linear(config, eng_key, i, o, out_dtype=out_dtype, pad_to=1)
    return ForeignLinear(config, eng_key, i, o, fmt, out_dtype=out_dtype, **kw)


class GGUFGenericConfig(Config):
    arch_string = ARCH_STRING

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": GGUFGenericModel}, **kwargs)
        from ..loader.gguf_config import GGML_FMT
        self.gguf_arch = self.read_cfg(str, "gguf_arch", no_default)
        self.head_dim = self.read_cfg(int, "head_dim", no_default)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", 0)
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.qk_norm = self.config_dict["qk_norm"]
        self.moe_layers = set(self.config_dict["moe_layers"])
        self.num_experts = self.config_dict.get("num_experts", 0)
        self.num_experts_per_tok = self.config_dict.get("num_experts_per_tok", 0)
        self.moe_intermediate_size = self.config_dict.get("moe_intermediate_size", 0)
        self.norm_topk_prob = bool(self.config_dict.get("norm_topk_prob", True))
        self.shared_intermediate = self.config_dict.get("shared_intermediate_size", 0)
        self.shared_gate = bool(self.config_dict.get("shared_gate", False))
        self.router_bias = bool(self.config_dict.get("router_bias", False))
        self.expert_bias = {pj: bool(self.config_dict.get("expert_bias_" + pj, False)) for pj in ("gate", "up", "down")}
        self.moe_activation = self.config_dict.get("moe_activation", "silu")
        self.moe_act_limit = float(self.config_dict.get("moe_act_limit", 0.0))
        self.attn_bias = self.config_dict.get("attn_bias") or [""] * self.config_dict["num_hidden_layers"]
        self.gaps = self.config_dict["gaps"]
        if self.gaps and os.environ.get("EXL3_GGUF_GENERIC_ALLOW_GAPS", "0") != "1":
            raise GenericUnsupported(f"{self.gguf_arch}: generic loader is missing: " + "; ".join(self.gaps))
        cd = self.config_dict
        self.hidden_act, self.embed_scale, self.v_norm, self.sm_scale = cd["hidden_act"], cd["embed_scale"], cd["v_norm"], cd["sm_scale"]
        self.kv_eq_v = set(cd["kv_eq_v_layers"]) if cd["v_norm"] else set()
        self.head_dim_l, self.n_kv_l = cd["head_dim_layers"], cd["n_kv_layers"]
        self.swa_l = [cd["sliding_window"] - 1 if sw else -1 for sw in cd["swa_layers"]]   # kernels keep window + 1 keys incl. the query
        self.sandwich_norms, self.layer_scalar = cd["sandwich_norms"], cd["layer_scalar"]
        self.sinks_layers = set(cd["sinks_layers"])
        self.final_softcap, self.attn_softcap, self.logit_scale = cd["final_logit_softcapping"], cd["attn_logit_softcapping"], cd["logit_scale"]
        style = RopeStyle.GPTJ if ARCH_FEATS.get(self.gguf_arch, {}).get("rope_gptj") else RopeStyle.NEOX
        self.layer_rope = []
        rope_prop, rope_pf = set(cd["rope_prop_layers"]), cd["rope_partial_factor"]
        if cd["rope_freqs_pending"]:      # header-only cut at config build time: read the factors now, or refuse on a real run
            try:
                ff = GGUFCollection(os.path.join(directory, "model.gguf"))
                rope_pf = None
                f = ff._pread(ff.infos["rope_freqs.weight"][2], 4 * ff.infos["rope_freqs.weight"][1][0]).view("<f4")
                ff.close()
                rot = int((f == 1.0).sum())
                if rot == 0 or not (f[:rot] == 1.0).all() or not (f[rot:] >= 1e9).all():
                    raise GenericUnsupported("rope_freqs.weight is not a proportional rope pattern")
                rope_pf = 2.0 * rot / cd["head_dim_layers"][min(rope_prop)]
            except (IOError, OSError, ValueError):
                if os.environ.get("EXL3_GGUF_GENERIC_ALLOW_GAPS", "0") != "1":
                    raise GenericUnsupported("rope_freqs.weight data unreadable (truncated file?)")
                rope_pf = 1.0     # CPU dry-run on a header-only cut only: structure is built, numerics are not
        for i in range(self.num_hidden_layers):
            rcd = {"rope_theta": cd["rope_theta_layers"][i], "rope_scaling": cd.get("rope_scaling"),
                   "max_position_embeddings": cd.get("max_position_embeddings")}
            if i in rope_prop:
                rcd["rope_scaling"] = {"rope_type": "proportional", "factor": 1.0}
                rcd["partial_rotary_factor"] = rope_pf
            if cd["rotary_dim_layers"][i] != self.head_dim_l[i]:
                rcd["rotary_dim"] = cd["rotary_dim_layers"][i]
            self.layer_rope.append(self.read_rope_settings_default(style, config_dict=rcd, override_head_dim=self.head_dim_l[i]))
        self.rope_settings = self.layer_rope[0]
        self.gguf = GGUFCollection(os.path.join(directory, "model.gguf"))
        self.gguf_plan = build_plan(self.gguf)
        self.gguf.name_map = name_map_generic(self.config_dict, self.gguf_plan)
        self.stc = self.gguf
        self.gguf_formats = {n: GGML_FMT.get(ty, f"unk{ty}") for n, (ty, _, _) in self.gguf.infos.items()}
        # Quantized token_embd: dequantize once on the GPU and serve it from memory (as gguf_dense does).
        # Skipped without CUDA so the CPU dry-run tests stay CPU-only.
        if torch.cuda.is_available() and os.environ.get("EXL3_GGUF_GENERIC_NO_EMB_DEQ") != "1":
            self.embedding_override()

    def embedding_override(self, device=None):
        """Dequantise token_embd once (GPU kernels) and serve it from memory. Not run at construction (CPU dry-run)."""
        from ..modules.foreign_linear import ForeignLinear
        from ..loader.gguf_config import GGML_FMT
        ty, (K, N), _ = self.gguf.infos["token_embd.weight"]
        fmt = GGML_FMT.get(ty, f"unk{ty}")
        if fmt == "fp":
            return
        fl = ForeignLinear(self, "model.embed_tokens", K, N, fmt)
        fl.load(device or torch.device("cuda:0"))
        w = torch.cat([fl._dequant_rows(i, min(4096, N - i)).cpu() for i in range(0, N, 4096)])
        fl.unload()
        self.gguf.new_tensors = {"model.embed_tokens.weight": w}

    def get_tensor_name_fixes(self):
        return {}


def _built(name):
    """the registry block, refusing a block that is still a predicate only (its build step is named)"""
    b = REGISTRY.get(name) if name else None
    if b is None or b.build is None:
        raise GenericUnsupported(f"no built block for {name!r}" + (f" ({b.step})" if b else ""))
    return b


class GGUFGenericModel(Model):
    config_class = GGUFGenericConfig

    def __init__(self, config: GGUFGenericConfig, key_prefix: str = "model", **kwargs):
        super().__init__(config, **kwargs)
        P, plan = key_prefix, config.gguf_plan
        self.modules += [_built(plan.glob["embed"]).build(config, P)]
        for bn in plan.glob["embed_mod"] + plan.glob["head_mod"]:
            _built(bn)
        self.first_block_idx = len(self.modules)
        for idx in range(config.num_hidden_layers):
            L, lp = f"{P}.layers.{idx}", plan.layers[idx]
            for bn in ([lp["pre"]] if lp["pre"] else []) + lp["attn_mod"] + lp["ffn_mod"] + lp["layer_mod"]:
                _built(bn)
            attn = _built(lp["attn"]).build(config, plan, idx, L)
            mlp = _built(lp["ffn"]).build(config, plan, idx, L)
            self.modules += [_built(lp["stream"]).build(config, plan, idx, L, attn, mlp)]
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += _built(plan.glob["head"]).build(config, P)
        self.logit_layer_idx = len(self.modules) - 1

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)
