"""Generic GGUF loader (first cut): llama-family attention + dense gated MLP or standard top-k MoE, no per-model class.

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
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, BlockSparseMLP
from ..modules.attn import prepare_for_attn

ARCH_STRING = "GGUFGenericForCausalLM"


def enabled() -> bool:
    return os.environ.get("EXL3_GGUF_GENERIC", "0") == "1"


class GenericUnsupported(ValueError):
    """The GGUF needs a piece the generic builder does not have (the message names it)."""


# llama.cpp permutes Q/K rows (adjacent-pair rotary) only for these converters; every other arch is stored in HF/NEOX order.
_ROPE_PERMUTED = {"llama"}
# routing weights renormalised over the picked experts (llama.cpp hparams.expert_weights_norm) when the GGUF lacks the key
# gpt-oss: softmax over the top-k logits == softmax over all, renormalised over the picked experts
_NORM_TOPK_DEFAULT = {"qwen3moe": True, "mixtral": True, "olmoe": False, "gpt-oss": True}

# arch default for <arch>.attention.sliding_window_pattern when the GGUF lacks it (llama.cpp set_swa_pattern); 0/absent = all layers sliding
_SWA_DEFAULT_PATTERN = {"gpt-oss": 2}
# yarn without truncation of the correction range (HF gpt-oss rope_scaling.truncate = false)
_YARN_NO_TRUNCATE = {"gpt-oss"}

# activation + clamp per arch: llama.cpp hard-codes swiglu_oai alpha 1.702 / limit 7.0 for gpt-oss (llama-graph.cpp build_moe_ffn, no GGUF key);
# the engine's "swiglu_oai" has the same alpha and the +1 on up. <arch>.swiglu_limit overrides when a GGUF carries it.
_MOE_ACT = {"gpt-oss": ("swiglu_oai", 7.0)}

_KNOWN = {
    "attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias", "attn_sinks.weight",
    "post_attention_norm.weight", "post_ffw_norm.weight", "layer_output_scale.weight",
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "attn_q_norm.weight", "attn_k_norm.weight", "ffn_norm.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight",
    "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
    "ffn_gate_inp.bias", "ffn_gate_exps.bias", "ffn_up_exps.bias", "ffn_down_exps.bias",
}
_KNOWN_GLOBAL = {"token_embd.weight", "output_norm.weight", "output.weight", "rope_freqs.weight"}

# arch features llama.cpp hard-codes per arch (no GGUF key): gemma4 = src/models/gemma4.cpp (see GEMMA4_UNCHECKED.md)
#   act: dense FFN activation (LLM_FFN_GELU = tanh approx); embed_scale: inp = tok_embd * sqrt(n_embd) (build_inp_embd(.., sqrtf(n_embd)));
#   v_norm: unweighted rms_norm on V of every layer (K=V layers: V = raw k_proj output); sm_scale: hparams.f_attention_scale = 1.0
_ARCH_FEATS = {"gemma4": {"act": "gelu", "embed_scale": True, "v_norm": True, "sm_scale": 1.0}}


def _req(m: dict, key: str):
    if key not in m:
        raise GenericUnsupported(f"missing GGUF metadata key {key!r}")
    return m[key]


def hf_config_generic(g: GGUFCollection) -> dict:
    """GGUF header -> HF-style config dict for GGUFGenericForCausalLM; unmapped pieces go to cfg['gaps']."""
    m, infos = g.meta, g.infos
    arch = _req(m, "general.architecture")
    p = arch + "."
    ne, nq = _req(m, p + "embedding_length"), _req(m, p + "attention.head_count")
    nl = _req(m, p + "block_count")
    hd = m.get(p + "attention.key_length", ne // nq)
    gaps, unmapped = [], []
    # per-layer kv heads (gemma4: array)
    kvr = m.get(p + "attention.head_count_kv", nq)
    n_kv_l = list(kvr) if isinstance(kvr, list) else [kvr] * nl
    if len(n_kv_l) != nl:
        gaps.append(f"head_count_kv array has {len(n_kv_l)} entries for {nl} layers")
        n_kv_l = (n_kv_l * nl)[:nl]
    # tensor-name audit: every tensor must be one the builder knows
    layers: dict[int, set] = {}
    for name in infos:
        mt = re.match(r"blk\.(\d+)\.(.+)$", name)
        if mt:
            layers.setdefault(int(mt.group(1)), set()).add(mt.group(2))
            if mt.group(2) not in _KNOWN:
                unmapped.append(name if int(mt.group(1)) == 0 else re.sub(r"blk\.\d+", "blk.N", name))
        elif name not in _KNOWN_GLOBAL:
            unmapped.append(name)
    unmapped = sorted(set(unmapped))
    if unmapped:
        gaps.append(f"no builder for tensors: {', '.join(unmapped[:12])}")
    if sorted(layers) != list(range(nl)):
        gaps.append(f"blk indices {sorted(layers)[:3]}.. do not cover block_count={nl} (nextn/MTP or non-uniform stack)")
    moe_layers = [i for i in range(nl) if "ffn_gate_exps.weight" in layers.get(i, ())]
    if p + "ssm.state_size" in m:
        gaps.append(f"metadata {p}ssm.state_size (ssm / mamba layers) has no module wiring")
    if m.get(p + "attention.shared_kv_layers", 0):
        gaps.append(f"shared_kv_layers={m[p + 'attention.shared_kv_layers']} (KV sharing across layers) not wired")
    if m.get(p + "embedding_length_per_layer_input", 0):
        gaps.append("per-layer input embeddings (PLE) not wired")
    # (1) sandwich norms: llama.cpp names the pre-FFN norm ffn_norm; gpt-oss has no ffn_norm and uses post_attention_norm there
    def has_all(sfx):
        return all(sfx in layers.get(i, ()) for i in range(nl))
    def has_any(sfx):
        return any(sfx in layers.get(i, ()) for i in range(nl))
    for sfx in ("post_attention_norm.weight", "post_ffw_norm.weight", "layer_output_scale.weight", "ffn_norm.weight"):
        if has_any(sfx) and not has_all(sfx):
            gaps.append(f"{sfx} present on only some layers")
    sandwich = has_all("post_attention_norm.weight") and has_all("post_ffw_norm.weight")
    pre_ffn_from_post = (not has_any("ffn_norm.weight")) and has_all("post_attention_norm.weight") and not has_any("post_ffw_norm.weight")
    if not has_any("ffn_norm.weight") and not pre_ffn_from_post:
        gaps.append("no pre-FFN norm tensor (ffn_norm / post_attention_norm)")
    layer_scalar = has_all("layer_output_scale.weight")
    # (2) sliding window, per-layer pattern, per-layer head dim / rope base
    win = m.get(p + "attention.sliding_window", 0) or 0
    swa = [False] * nl
    if win:
        pat = m.get(p + "attention.sliding_window_pattern", _SWA_DEFAULT_PATTERN.get(arch, 0))
        if isinstance(pat, list):
            if len(pat) != nl:
                gaps.append(f"sliding_window_pattern has {len(pat)} entries for {nl} layers")
            swa = [bool(pat[i % len(pat)]) for i in range(nl)]
        elif pat:
            swa = [i % pat < pat - 1 for i in range(nl)]
        else:
            swa = [True] * nl
    hd_swa = m.get(p + "attention.key_length_swa", hd)
    hd_l = [hd_swa if swa[i] else hd for i in range(nl)]
    bad_q = [i for i in range(nl) if f"blk.{i}.attn_q.weight" in infos and infos[f"blk.{i}.attn_q.weight"][1][1] != nq * hd_l[i]]
    if bad_q:
        gaps.append(f"attn_q width != n_heads*head_dim on layers {bad_q[:4]} (gated q / unknown per-layer head dim)")
    fb, fbs = m.get(p + "rope.freq_base", 10000.0), m.get(p + "rope.freq_base_swa")
    theta_l = [fbs if (swa[i] and fbs) else fb for i in range(nl)]
    rd, rds = m.get(p + "rope.dimension_count"), m.get(p + "rope.dimension_count_swa")
    rdim_l = [(rds if (swa[i] and rds) else rd) or hd_l[i] for i in range(nl)]
    kv_eq_v = [i for i in range(nl) if "attn_k.weight" in layers.get(i, ()) and "attn_v.weight" not in layers.get(i, ())]
    feats = _ARCH_FEATS.get(arch, {})
    if kv_eq_v and not feats.get("v_norm"):    # K=V is only wired together with the unweighted v_norm of that arch family
        gaps.append(f"K=V layers without attn_v (use_k_as_v + unweighted v_norm) on {len(kv_eq_v)} layers, first {kv_eq_v[:3]}")
    if arch.startswith("gemma") and arch not in _ARCH_FEATS:
        gaps.append(f"gemma family arch {arch!r}: GELU-tanh MLP act, embedding * sqrt(hidden), unweighted v_norm, sm_scale 1.0 not mapped")
    # rope_freqs.weight (llama.cpp: freq_factors on the non-SWA layers, theta_i / ff_i): ff = 1 on the rotated angles, huge on the rest
    # == engine "proportional" rope (partial_rotary_factor = rotated angles * 2 / head_dim)
    rope_prop, rope_pf, rope_pending = [], None, False
    if "rope_freqs.weight" in infos:
        rope_prop = [i for i in range(nl) if not swa[i]]
        nff = infos["rope_freqs.weight"][1][0]
        hd_g = {hd_l[i] for i in rope_prop}
        if infos["rope_freqs.weight"][0] != 0 or not rope_prop or hd_g != {2 * nff}:
            gaps.append(f"rope_freqs.weight (type {infos['rope_freqs.weight'][0]}, n={nff}) does not match head_dim {sorted(hd_g)} of the non-SWA layers")
        else:
            try:
                ff = g._pread(infos["rope_freqs.weight"][2], 4 * nff).view("<f4")
            except (IOError, OSError, ValueError):
                ff, rope_pending = None, True      # header-only cut: no tensor data; GGUFGenericConfig re-reads it on the full file
            if ff is not None:
                rot = int((ff == 1.0).sum())
                if rot == 0 or not (ff[:rot] == 1.0).all() or not (ff[rot:] >= 1e9).all():
                    gaps.append(f"rope_freqs.weight is not (1.0 x k, then >=1e9): not a proportional rope ({rot} ones)")
                else:
                    rope_pf = 2.0 * rot / hd_l[rope_prop[0]]
    # (4) attention biases + sinks
    attn_bias = ["".join(c for c, t in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output"))
                         if f"{t}.bias" in layers.get(i, ())) for i in range(nl)]
    sinks_layers = [i for i in range(nl) if "attn_sinks.weight" in layers.get(i, ())]
    if p + "attention.layer_norm_rms_epsilon" not in m:
        gaps.append("no rms epsilon (LayerNorm-style arch)")
    # (6) rope scaling: yarn + linear mapped; llama3 / longrope / rope_freqs factors stay gaps
    rs = m.get(p + "rope.scaling.type", "none")
    rope_scaling = None
    if rs in ("yarn", "linear"):
        rope_scaling = {"rope_type": rs, "factor": float(m.get(p + "rope.scaling.factor", 1.0))}
        if rs == "yarn":
            rope_scaling["original_max_position_embeddings"] = int(m.get(p + "rope.scaling.original_context_length", m.get(p + "context_length", 4096)))
            for k_gguf, k_hf in (("yarn_beta_fast", "beta_fast"), ("yarn_beta_slow", "beta_slow"), ("attn_factor", "attention_factor")):
                if p + "rope.scaling." + k_gguf in m:
                    rope_scaling[k_hf] = float(m[p + "rope.scaling." + k_gguf])
            if arch in _YARN_NO_TRUNCATE:
                rope_scaling["truncate"] = False
    elif rs != "none":
        gaps.append(f"rope.scaling.type={rs!r} not mapped (llama3/longrope factors)")
    if any(rdim_l[i] != hd_l[i] for i in range(nl)):
        gaps.append(f"partial rotary: rope.dimension_count={sorted(set(rdim_l))} != head_dim={sorted(set(hd_l))}")
    # qk norm: per head (width head_dim, qwen3) or spanning all heads (width nq*hd, olmoe)
    qn = infos.get("blk.0.attn_q_norm.weight")
    qk_norm = "none" if qn is None else ("head" if qn[1][0] == hd_l[0] else "span" if qn[1][0] == nq * hd_l[0] else "?")
    if qk_norm == "?":
        gaps.append(f"attn_q_norm width {qn[1][0]} is neither head_dim nor n_heads*head_dim")
    cfg = {
        "architectures": [ARCH_STRING], "model_type": "gguf_generic", "gguf_arch": arch,
        "hidden_size": ne, "num_attention_heads": nq, "num_key_value_heads": n_kv_l[0],
        "num_hidden_layers": nl, "head_dim": hd,
        "rms_norm_eps": m.get(p + "attention.layer_norm_rms_epsilon", 1e-6),
        "max_position_embeddings": m.get(p + "context_length", 4096),
        "rope_theta": fb, "hidden_act": feats.get("act", "silu"),
        "embed_scale": float(ne) ** 0.5 if feats.get("embed_scale") else 1.0, "v_norm": bool(feats.get("v_norm")),
        "sm_scale": feats.get("sm_scale"), "rope_prop_layers": rope_prop, "rope_partial_factor": rope_pf, "rope_freqs_pending": rope_pending,
        "n_kv_layers": n_kv_l, "head_dim_layers": hd_l, "rope_theta_layers": theta_l, "rotary_dim_layers": rdim_l,
        "sliding_window": win, "swa_layers": swa, "rope_scaling": rope_scaling,
        "sandwich_norms": sandwich, "pre_ffn_from_post": pre_ffn_from_post, "layer_scalar": layer_scalar,
        "attn_bias": attn_bias, "sinks_layers": sinks_layers, "kv_eq_v_layers": kv_eq_v,
        "final_logit_softcapping": float(m.get(p + "final_logit_softcapping", 0.0)),
        "attn_logit_softcapping": float(m.get(p + "attn_logit_softcapping", 0.0)),
        "logit_scale": float(m.get(p + "logit_scale", 1.0)),
        "vocab_size": infos["token_embd.weight"][1][1],
        "tie_word_embeddings": "output.weight" not in infos,
        "qk_norm": qk_norm, "moe_layers": moe_layers, "gaps": gaps,
        "intermediate_size": m.get(p + "feed_forward_length", 0),
    }
    if moe_layers:
        gd = infos[f"blk.{moe_layers[0]}.ffn_gate_exps.weight"][1]      # gguf dims (H, I, E)
        cfg.update(num_experts=gd[2], moe_intermediate_size=gd[1],
                   num_experts_per_tok=_req(m, p + "expert_used_count"))
        if m.get(p + "expert_count", gd[2]) != gd[2]:
            gaps.append("expert_count metadata disagrees with ffn_gate_exps shape")
        # (5) shared experts: GatedMLP beside the routed experts, optional sigmoid gate (ffn_gate_inp_shexp)
        sh = [i for i in moe_layers if "ffn_gate_shexp.weight" in layers.get(i, ())]
        if sh and len(sh) != len(moe_layers):
            gaps.append("shared experts present on only some MoE layers")
        elif sh:
            cfg["shared_intermediate_size"] = infos[f"blk.{sh[0]}.ffn_gate_shexp.weight"][1][1]
            cfg["shared_gate"] = "ffn_gate_inp_shexp.weight" in layers.get(sh[0], ())
        elif m.get(p + "expert_shared_count", 0):
            gaps.append("expert_shared_count > 0 but no ffn_*_shexp tensors")
        # (7) router logit bias + stacked per-expert biases (gpt-oss): all MoE layers or none
        for sfx, nmk in (("ffn_gate_inp.bias", "router_bias"), ("ffn_gate_exps.bias", "expert_bias_gate"),
                         ("ffn_up_exps.bias", "expert_bias_up"), ("ffn_down_exps.bias", "expert_bias_down")):
            have = [i for i in moe_layers if sfx in layers.get(i, ())]
            if have and len(have) != len(moe_layers):
                gaps.append(f"{sfx} present on only some MoE layers")
            cfg[nmk] = bool(have) and len(have) == len(moe_layers)
        if cfg["router_bias"]:
            bd = infos[f"blk.{moe_layers[0]}.ffn_gate_inp.bias"][1]
            if tuple(bd) != (gd[2],):
                gaps.append(f"ffn_gate_inp.bias dims {tuple(bd)} != (num_experts,)")
        for pj, nmk, shp in (("gate", "expert_bias_gate", (gd[1], gd[2])), ("up", "expert_bias_up", (gd[1], gd[2])),
                             ("down", "expert_bias_down", (gd[0], gd[2]))):
            if cfg[nmk] and tuple(infos[f"blk.{moe_layers[0]}.ffn_{pj}_exps.bias"][1]) != shp:
                gaps.append(f"ffn_{pj}_exps.bias dims {tuple(infos[f'blk.{moe_layers[0]}.ffn_{pj}_exps.bias'][1])} != {shp}")
        act, lim = _MOE_ACT.get(arch, ("silu", 0.0))
        cfg["moe_activation"], cfg["moe_act_limit"] = act, float(m.get(p + "swiglu_limit", lim))
        norm = m.get(p + "expert_weights_norm", _NORM_TOPK_DEFAULT.get(arch))
        if norm is None:
            gaps.append(f"unknown expert_weights_norm for arch {arch!r}")
        if m.get(p + "expert_gating_func", 1) not in (0, 1):
            gaps.append("router gating func is not softmax")
        cfg["norm_topk_prob"] = bool(norm)
    if len(moe_layers) not in (0, nl):
        cfg["dense_layers"] = [i for i in range(nl) if i not in moe_layers]
    return cfg


def name_map_generic(cfg: dict) -> dict:
    """engine key -> gguf tensor name; routed experts use per-expert slice keys (.eN) like the minicpm builder."""
    nm = {"model.norm.weight": "output_norm.weight", "model.embed_tokens.weight": "token_embd.weight"}
    if not cfg["tie_word_embeddings"]:
        nm["lm_head.weight"] = "output.weight"
    moe = set(cfg["moe_layers"])
    for i in range(cfg["num_hidden_layers"]):
        L, G = f"model.layers.{i}", f"blk.{i}"
        pre_ffn = "post_attention_norm" if cfg.get("pre_ffn_from_post") else "ffn_norm"
        nm.update({
            f"{L}.input_layernorm.weight": f"{G}.attn_norm.weight",
            f"{L}.post_attention_layernorm.weight": f"{G}.{pre_ffn}.weight",
            f"{L}.self_attn.q_proj.weight": f"{G}.attn_q.weight", f"{L}.self_attn.k_proj.weight": f"{G}.attn_k.weight",
            f"{L}.self_attn.v_proj.weight": f"{G}.attn_v.weight", f"{L}.self_attn.o_proj.weight": f"{G}.attn_output.weight",
            })
        if cfg.get("qk_norm", "none") != "none":
            nm.update({f"{L}.self_attn.q_norm.weight": f"{G}.attn_q_norm.weight", f"{L}.self_attn.k_norm.weight": f"{G}.attn_k_norm.weight"})
        if cfg.get("sandwich_norms"):
            nm[f"{L}.attn_post_norm.weight"] = f"{G}.post_attention_norm.weight"
            nm[f"{L}.mlp_post_norm.weight"] = f"{G}.post_ffw_norm.weight"
        if cfg.get("layer_scalar"):
            nm[f"{L}.layer_scalar"] = f"{G}.layer_output_scale.weight"
        if i in cfg.get("sinks_layers", ()):
            nm[f"{L}.self_attn.sinks"] = f"{G}.attn_sinks.weight"
        for c, pj, t in (("q", "q_proj", "attn_q"), ("k", "k_proj", "attn_k"), ("v", "v_proj", "attn_v"), ("o", "o_proj", "attn_output")):
            if c in cfg.get("attn_bias", [""] * (i + 1))[i]:
                nm[f"{L}.self_attn.{pj}.bias"] = f"{G}.{t}.bias"
        if i in cfg.get("kv_eq_v_layers", ()):     # no V projection: the attention reads K (use_k_as_v) when the arch has v_norm, else dry-run alias (gap flagged)
            if cfg.get("v_norm"):
                nm.pop(f"{L}.self_attn.v_proj.weight", None)
            else:
                nm[f"{L}.self_attn.v_proj.weight"] = f"{G}.attn_k.weight"
        if i in moe:
            nm[f"{L}.mlp.gate.weight"] = f"{G}.ffn_gate_inp.weight"
            if cfg.get("router_bias"):
                nm[f"{L}.mlp.gate.bias"] = f"{G}.ffn_gate_inp.bias"
            if cfg.get("shared_intermediate_size"):
                for proj in ("gate", "up", "down"):
                    nm[f"{L}.mlp.shared_experts.{proj}_proj.weight"] = f"{G}.ffn_{proj}_shexp.weight"
                if cfg.get("shared_gate"):
                    nm[f"{L}.mlp.shared_expert_gate.weight"] = f"{G}.ffn_gate_inp_shexp.weight"
            for e in range(cfg["num_experts"]):
                for proj, gp in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")):
                    nm[f"{L}.mlp.experts.{e}.{proj}.weight"] = f"{G}.{gp}.e{e}.weight"
                    if cfg.get("expert_bias_" + proj[:-5]):
                        nm[f"{L}.mlp.experts.{e}.{proj}.bias"] = f"{G}.{gp}.e{e}.bias"
        else:
            nm.update({f"{L}.mlp.gate_proj.weight": f"{G}.ffn_gate.weight", f"{L}.mlp.up_proj.weight": f"{G}.ffn_up.weight",
                       f"{L}.mlp.down_proj.weight": f"{G}.ffn_down.weight"})
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
        style = RopeStyle.GPTJ if self.gguf_arch in _ROPE_PERMUTED else RopeStyle.NEOX
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
        self.gguf.name_map = name_map_generic(self.config_dict)
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


class GGUFGenericModel(Model):
    config_class = GGUFGenericConfig

    def __init__(self, config: GGUFGenericConfig, key_prefix: str = "model", **kwargs):
        super().__init__(config, **kwargs)
        H, nq = config.hidden_size, config.num_q_heads
        P = key_prefix
        self.modules += [Embedding(config=config, key=f"{P}.embed_tokens", vocab_size=config.vocab_size, hidden_size=H,
                                   multiplier=config.embed_scale)]
        self.first_block_idx = len(self.modules)
        span = config.qk_norm == "span"
        for idx in range(config.num_hidden_layers):
            L = f"{P}.layers.{idx}"
            hd_i, nkv_i = config.head_dim_l[idx], config.n_kv_l[idx]
            norms = {}
            if config.qk_norm != "none":
                norms = {n: RMSNorm(config=config, key=f"{L}.self_attn.{n}", rms_norm_eps=config.rms_norm_eps, span_heads=span)
                         for n in ("q_norm", "k_norm")}
            if config.v_norm:
                norms["v_norm"] = RMSNorm(config=config, key=f"{L}.self_attn.v_norm", rms_norm_eps=config.rms_norm_eps, unweighted=True)
            post = {}
            if config.sandwich_norms:
                post = dict(attn_post_norm=RMSNorm(config=config, key=f"{L}.attn_post_norm", rms_norm_eps=config.rms_norm_eps, out_dtype=torch.float),
                            mlp_post_norm=RMSNorm(config=config, key=f"{L}.mlp_post_norm", rms_norm_eps=config.rms_norm_eps, out_dtype=torch.float))
            if config.layer_scalar:
                post["key_layer_scalar"] = "layer_scalar"
            self.modules += [TransformerBlock(
                config=config, key=L, layer_idx=idx,
                attn_norm=RMSNorm(config=config, key=f"{L}.input_layernorm", rms_norm_eps=config.rms_norm_eps),
                attn=Attention(
                    config=config, key=f"{L}.self_attn", layer_idx=idx, hidden_size=H, head_dim=hd_i,
                    num_q_heads=nq, num_kv_heads=nkv_i, rope_settings=config.layer_rope[idx], sm_scale=config.sm_scale,
                    use_k_as_v=idx in config.kv_eq_v,
                    qmap="block.attn", out_dtype=torch.float,
                    sliding_window=config.swa_l[idx], logit_softcapping=config.attn_softcap,
                    key_sinks="sinks" if idx in config.sinks_layers else None,
                    q_proj=_fl(config, f"{L}.self_attn.q_proj", H, nq * hd_i, bias="q" in config.attn_bias[idx]),
                    k_proj=_fl(config, f"{L}.self_attn.k_proj", H, nkv_i * hd_i, bias="k" in config.attn_bias[idx]),
                    v_proj=None if idx in config.kv_eq_v else _fl(config, f"{L}.self_attn.v_proj", H, nkv_i * hd_i, bias="v" in config.attn_bias[idx]),
                    o_proj=_fl(config, f"{L}.self_attn.o_proj", nq * hd_i, H, out_dtype=torch.float, bias="o" in config.attn_bias[idx]), **norms),
                mlp_norm=RMSNorm(config=config, key=f"{L}.post_attention_layernorm", rms_norm_eps=config.rms_norm_eps),
                **post,
                mlp=self._moe(config, L, H) if idx in config.moe_layers else self._dense(config, L, H))]
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [RMSNorm(config=config, key=f"{P}.norm", rms_norm_eps=config.rms_norm_eps, out_dtype=torch.half)]
        if config.tie_word_embeddings:
            config.gguf.name_map["lm_head.weight"] = "token_embd.weight"
        self.modules += [_fl(config, "lm_head", H, config.vocab_size, post_scale=config.logit_scale, softcap=config.final_softcap)]
        self.modules[-1].caps["logits_output"] = True
        self.logit_layer_idx = len(self.modules) - 1

    @staticmethod
    def _dense(config, L, H):
        I = config.intermediate_size
        mlp = GatedMLP(config=config, key=f"{L}.mlp", hidden_size=H, intermediate_size=I, qmap="block.mlp", activation_fn=config.hidden_act,
                       interm_dtype=torch.half, out_dtype=torch.float,
                       gates=[_fl(config, f"{L}.mlp.gate_proj", H, I)], ups=[_fl(config, f"{L}.mlp.up_proj", H, I)],
                       downs=[_fl(config, f"{L}.mlp.down_proj", I, H, out_dtype=torch.float)])
        for m in mlp.gates + mlp.ups + mlp.downs:
            mlp.register_submodule(m)
        return mlp

    @staticmethod
    def _moe(config, L, H):
        I, E = config.moe_intermediate_size, config.num_experts
        eb = config.expert_bias
        gates = [_fl(config, f"{L}.mlp.experts.{e}.gate_proj", H, I, bias=eb["gate"]) for e in range(E)]
        ups = [_fl(config, f"{L}.mlp.experts.{e}.up_proj", H, I, bias=eb["up"]) for e in range(E)]
        downs = [_fl(config, f"{L}.mlp.experts.{e}.down_proj", I, H, out_dtype=torch.float, bias=eb["down"]) for e in range(E)]
        moe_kw = {}
        if config.router_bias:   # logit bias before top-k, softmax over the picked biased logits (engine gpt-oss: routing_std_bias, no e_score bias)
            moe_kw.update(router_type="std_bias", key_e_score_bias=None)
        if config.moe_activation != "silu":
            moe_kw.update(activation_fn=config.moe_activation, act_limit=config.moe_act_limit)
        shared = None
        if config.shared_intermediate:
            SI, SK = config.shared_intermediate, f"{L}.mlp.shared_experts"
            shared = GatedMLP(config=config, key=SK, hidden_size=H, intermediate_size=SI, qmap="block.mlp",
                              interm_dtype=torch.half, out_dtype=torch.float,
                              gates=[_fl(config, f"{SK}.gate_proj", H, SI)], ups=[_fl(config, f"{SK}.up_proj", H, SI)],
                              downs=[_fl(config, f"{SK}.down_proj", SI, H, out_dtype=torch.float)])
            for m in shared.gates + shared.ups + shared.downs:
                shared.register_submodule(m)
        mlp = BlockSparseMLP(config=config, key=f"{L}.mlp", hidden_size=H, intermediate_size=I, num_experts=E,
                             shared_experts=shared, key_shared_gate="shared_expert_gate" if config.shared_gate else None,
                             num_experts_per_tok=config.num_experts_per_tok, norm_topk=bool(config.norm_topk_prob), key_up=None, key_gate=None, key_down=None,
                             key_routing_gate="gate", qmap="block.mlp", interm_dtype=torch.half, out_dtype=torch.float,
                             gates=gates, ups=ups, downs=downs, **moe_kw)
        for m in gates + ups + downs:
            mlp.register_submodule(m)
        return mlp

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)
