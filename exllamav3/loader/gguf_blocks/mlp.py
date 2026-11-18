"""Blocks around modules/mlp.py and block_sparse_mlp.py: dense gated MLP, standard top-k MoE (+shared, +biases),
and the predicates of the latent MoE / sigmoid-bias router (not built yet)."""
from __future__ import annotations
from ..gguf_registry import Block, register, req

_EXPS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
_MOE_CLAIMS = frozenset({"ffn_gate_inp.weight", "ffn_gate_inp.bias", *(f"{e}.weight" for e in _EXPS), *(f"{e}.bias" for e in _EXPS),
                         "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"})


def moe_layers(ctx):
    return ctx.once("moe_layers", lambda c: [i for i in range(c.nl) if c.has(i, "ffn_gate_exps.weight")])


def moe_global(ctx):
    """standard MoE config (first MoE layer's shapes); empty when the header has no routed experts"""
    def f(ctx):
        m, infos, p, arch, gaps, ml = ctx.meta, ctx.infos, ctx.p, ctx.arch, ctx.gaps, moe_layers(ctx)
        if not ml:
            return {}
        gd = infos[f"blk.{ml[0]}.ffn_gate_exps.weight"][1]      # gguf dims (H, I, E)
        cfg = {"num_experts": gd[2], "moe_intermediate_size": gd[1], "num_experts_per_tok": req(m, p + "expert_used_count")}
        if m.get(p + "expert_count", gd[2]) != gd[2]:
            gaps.append("expert_count metadata disagrees with ffn_gate_exps shape")
        sh = [i for i in ml if ctx.has(i, "ffn_gate_shexp.weight")]
        if sh and len(sh) != len(ml):
            gaps.append("shared experts present on only some MoE layers")
        elif sh:
            cfg["shared_intermediate_size"] = infos[f"blk.{sh[0]}.ffn_gate_shexp.weight"][1][1]
            cfg["shared_gate"] = ctx.has(sh[0], "ffn_gate_inp_shexp.weight")
        elif m.get(p + "expert_shared_count", 0):
            gaps.append("expert_shared_count > 0 but no ffn_*_shexp tensors")
        for sfx, nmk in (("ffn_gate_inp.bias", "router_bias"), ("ffn_gate_exps.bias", "expert_bias_gate"),
                         ("ffn_up_exps.bias", "expert_bias_up"), ("ffn_down_exps.bias", "expert_bias_down")):
            have = [i for i in ml if ctx.has(i, sfx)]
            if have and len(have) != len(ml):
                gaps.append(f"{sfx} present on only some MoE layers")
            cfg[nmk] = bool(have) and len(have) == len(ml)
        if cfg["router_bias"]:
            bd = infos[f"blk.{ml[0]}.ffn_gate_inp.bias"][1]
            if tuple(bd) != (gd[2],):
                gaps.append(f"ffn_gate_inp.bias dims {tuple(bd)} != (num_experts,)")
        for pj, nmk, shp in (("gate", "expert_bias_gate", (gd[1], gd[2])), ("up", "expert_bias_up", (gd[1], gd[2])),
                             ("down", "expert_bias_down", (gd[0], gd[2]))):
            if cfg[nmk] and tuple(infos[f"blk.{ml[0]}.ffn_{pj}_exps.bias"][1]) != shp:
                gaps.append(f"ffn_{pj}_exps.bias dims {tuple(infos[f'blk.{ml[0]}.ffn_{pj}_exps.bias'][1])} != {shp}")
        act, lim = ctx.feats.get("moe_act", ("silu", 0.0))
        cfg["moe_activation"], cfg["moe_act_limit"] = act, float(m.get(p + "swiglu_limit", lim))
        norm = m.get(p + "expert_weights_norm", ctx.feats.get("norm_topk"))
        if norm is None:
            gaps.append(f"unknown expert_weights_norm for arch {arch!r}")
        if m.get(p + "expert_gating_func", 1) not in (0, 1):
            gaps.append("router gating func is not softmax")
        cfg["norm_topk_prob"] = bool(norm)
        return cfg
    return ctx.once("moe", f)


def _dense_names(cfg, i, L, G):
    return {f"{L}.mlp.gate_proj.weight": f"{G}.ffn_gate.weight", f"{L}.mlp.up_proj.weight": f"{G}.ffn_up.weight",
            f"{L}.mlp.down_proj.weight": f"{G}.ffn_down.weight"}


def _dense_build(config, plan, idx, L):
    import torch
    from ...modules import GatedMLP
    from ...architecture.gguf_generic import _fl
    H, I = config.hidden_size, config.intermediate_size
    mlp = GatedMLP(config=config, key=f"{L}.mlp", hidden_size=H, intermediate_size=I, qmap="block.mlp", activation_fn=config.hidden_act,
                   interm_dtype=torch.half, out_dtype=torch.float,
                   gates=[_fl(config, f"{L}.mlp.gate_proj", H, I)], ups=[_fl(config, f"{L}.mlp.up_proj", H, I)],
                   downs=[_fl(config, f"{L}.mlp.down_proj", I, H, out_dtype=torch.float)])
    for m in mlp.gates + mlp.ups + mlp.downs:
        mlp.register_submodule(m)
    return mlp


register(Block(
    "dense_ffn", "ffn",
    select=lambda ctx, i: not ctx.has(i, "ffn_gate_exps.weight"),
    config=lambda ctx, i: {"intermediate_size": ctx.key("feed_forward_length", 0), "act": ctx.feats.get("act", "silu")},
    claims=frozenset({"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}), build=_dense_build, names=_dense_names))


def _moe_names(cfg, i, L, G):
    nm = {f"{L}.mlp.gate.weight": f"{G}.ffn_gate_inp.weight"}
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
    return nm


def _moe_build(config, plan, idx, L):
    import torch
    from ...modules import GatedMLP, BlockSparseMLP
    from ...architecture.gguf_generic import _fl
    H, I, E = config.hidden_size, config.moe_intermediate_size, config.num_experts
    eb = config.expert_bias
    gates = [_fl(config, f"{L}.mlp.experts.{e}.gate_proj", H, I, bias=eb["gate"]) for e in range(E)]
    ups = [_fl(config, f"{L}.mlp.experts.{e}.up_proj", H, I, bias=eb["up"]) for e in range(E)]
    downs = [_fl(config, f"{L}.mlp.experts.{e}.down_proj", I, H, out_dtype=torch.float, bias=eb["down"]) for e in range(E)]
    moe_kw = {}
    if config.router_bias:   # logit bias before top-k, softmax over the picked biased logits (engine gpt-oss: routing_std_bias)
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
                         num_experts_per_tok=config.num_experts_per_tok, norm_topk=bool(config.norm_topk_prob), key_up=None, key_gate=None,
                         key_down=None, key_routing_gate="gate", qmap="block.mlp", interm_dtype=torch.half, out_dtype=torch.float,
                         gates=gates, ups=ups, downs=downs, **moe_kw)
    for m in gates + ups + downs:
        mlp.register_submodule(m)
    return mlp


def _moe_layer(ctx, i):
    g = moe_global(ctx)
    keys = ("num_experts", "moe_intermediate_size", "num_experts_per_tok", "shared_intermediate_size", "shared_gate", "router_bias",
            "expert_bias_gate", "expert_bias_up", "expert_bias_down", "moe_activation", "moe_act_limit", "norm_topk_prob")
    return {k: g.get(k) for k in keys}


register(Block(
    "moe_ffn", "ffn",
    select=lambda ctx, i: ctx.has(i, "ffn_gate_exps.weight") and not ctx.has(i, "ffn_routed_down.weight"),
    config=_moe_layer, claims=_MOE_CLAIMS, build=_moe_build, names=_moe_names))


# ---- predicates only (Hy4 step 5): latent MoE (routed experts at expert_latent_length between ffn_routed_down/up),
#      sigmoid router (gating_func 2) + exp_probs_b, weights_scale, per-layer swiglu_clamp_exp; engine module: block_sparse_mlp latent_in/out
def _latent_layer(ctx, i):
    gd = ctx.dims(f"blk.{i}.ffn_gate_exps.weight")
    sh = ctx.dims(f"blk.{i}.ffn_gate_shexp.weight")
    clamp = ctx.per_layer("swiglu_clamp_exp", 0.0)
    return {"latent": ctx.key("expert_latent_length"), "latent_down": ctx.dims(f"blk.{i}.ffn_routed_down.weight"),
            "expert_dims": gd, "num_experts": gd[2] if gd else None, "top_k": ctx.key("expert_used_count"),
            "gating_func": ctx.key("expert_gating_func"), "weights_scale": ctx.key("expert_weights_scale"),
            "weights_norm": ctx.key("expert_weights_norm"), "e_score_bias": ctx.has(i, "exp_probs_b.bias"),
            "clamp": clamp[i] if i < len(clamp) else None, "clamp_routed_only": bool(ctx.feats.get("clamp_routed_only")),
            "shared_dims": sh}


register(Block(
    "latent_moe", "ffn", step="step 5",
    select=lambda ctx, i: ctx.has(i, "ffn_routed_down.weight") and ctx.has(i, "ffn_gate_exps.weight"),
    config=_latent_layer, claims=_MOE_CLAIMS | {"ffn_routed_down.weight", "ffn_routed_up.weight", "exp_probs_b.bias"}, build=None))

# sigmoid / sqrt-softplus router with a per-expert score bias on a plain MoE layer (glm5-next, deepseek41); predicate only
register(Block(
    "router_score_bias", "ffn_mod", step="step 5",
    select=lambda ctx, i: ctx.has(i, "exp_probs_b.bias") and not ctx.has(i, "ffn_routed_down.weight"),
    config=lambda ctx, i: {"gating_func": ctx.key("expert_gating_func"), "weights_scale": ctx.key("expert_weights_scale"),
                           "vl_bias": ctx.has(i, "exp_probs_b_vl.bias")},
    claims=frozenset({"exp_probs_b.bias", "exp_probs_b_vl.bias"}), build=None))
