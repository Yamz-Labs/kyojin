"""Blocks around modules/attn.py: GQA / MHA attention (q/k/v/o, qk-norm, biases, sinks, sliding window, rope variants)."""
from __future__ import annotations
from ..gguf_registry import Block, register, req

_CLAIMS = frozenset({"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q.bias", "attn_k.bias",
                     "attn_v.bias", "attn_output.bias", "attn_q_norm.weight", "attn_k_norm.weight", "attn_sinks.weight"})


def gqa_global(ctx):
    """header -> attention config over all layers (lists indexed by layer); gaps go to ctx.gaps"""
    def f(ctx):
        m, infos, p, arch, nl, gaps, feats = ctx.meta, ctx.infos, ctx.p, ctx.arch, ctx.nl, ctx.gaps, ctx.feats
        ne, nq = req(m, p + "embedding_length"), req(m, p + "attention.head_count")
        nq = nq[0] if isinstance(nq, list) else nq
        hd = m.get(p + "attention.key_length", ne // nq)
        kvr = m.get(p + "attention.head_count_kv", nq)
        n_kv_l = list(kvr) if isinstance(kvr, list) else [kvr] * nl
        if len(n_kv_l) != nl:
            gaps.append(f"head_count_kv array has {len(n_kv_l)} entries for {nl} layers")
            n_kv_l = (n_kv_l * nl)[:nl]
        win = m.get(p + "attention.sliding_window", 0) or 0
        swa = [False] * nl
        if win:
            pat = m.get(p + "attention.sliding_window_pattern", feats.get("swa_pattern", 0))
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
        kv_eq_v = [i for i in range(nl) if ctx.has(i, "attn_k.weight") and not ctx.has(i, "attn_v.weight")]
        if kv_eq_v and not feats.get("v_norm"):    # K=V is only wired together with the unweighted v_norm of that arch family
            gaps.append(f"K=V layers without attn_v (use_k_as_v + unweighted v_norm) on {len(kv_eq_v)} layers, first {kv_eq_v[:3]}")
        if arch.startswith("gemma") and not feats:
            gaps.append(f"gemma family arch {arch!r}: GELU-tanh MLP act, embedding * sqrt(hidden), unweighted v_norm, sm_scale 1.0 not mapped")
        # rope_freqs.weight: ff = 1 on the rotated angles, huge on the rest == engine "proportional" rope
        rope_prop, rope_pf, rope_pending = [], None, False
        if "rope_freqs.weight" in infos:
            rope_prop = [i for i in range(nl) if not swa[i]]
            nff = infos["rope_freqs.weight"][1][0]
            hd_g = {hd_l[i] for i in rope_prop}
            if infos["rope_freqs.weight"][0] != 0 or not rope_prop or hd_g != {2 * nff}:
                gaps.append(f"rope_freqs.weight (type {infos['rope_freqs.weight'][0]}, n={nff}) does not match head_dim {sorted(hd_g)} of the non-SWA layers")
            else:
                try:
                    ff = ctx.g._pread(infos["rope_freqs.weight"][2], 4 * nff).view("<f4")
                except (IOError, OSError, ValueError):
                    ff, rope_pending = None, True      # header-only cut: GGUFGenericConfig re-reads it on the full file
                if ff is not None:
                    rot = int((ff == 1.0).sum())
                    if rot == 0 or not (ff[:rot] == 1.0).all() or not (ff[rot:] >= 1e9).all():
                        gaps.append(f"rope_freqs.weight is not (1.0 x k, then >=1e9): not a proportional rope ({rot} ones)")
                    else:
                        rope_pf = 2.0 * rot / hd_l[rope_prop[0]]
        attn_bias = ["".join(c for c, t in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output"))
                             if ctx.has(i, f"{t}.bias")) for i in range(nl)]
        sinks_layers = [i for i in range(nl) if ctx.has(i, "attn_sinks.weight")]
        if p + "attention.layer_norm_rms_epsilon" not in m:
            gaps.append("no rms epsilon (LayerNorm-style arch)")
        rs = m.get(p + "rope.scaling.type", "none")
        rope_scaling = None
        if rs in ("yarn", "linear"):
            rope_scaling = {"rope_type": rs, "factor": float(m.get(p + "rope.scaling.factor", 1.0))}
            if rs == "yarn":
                rope_scaling["original_max_position_embeddings"] = int(m.get(p + "rope.scaling.original_context_length", m.get(p + "context_length", 4096)))
                for k_gguf, k_hf in (("yarn_beta_fast", "beta_fast"), ("yarn_beta_slow", "beta_slow"), ("attn_factor", "attention_factor")):
                    if p + "rope.scaling." + k_gguf in m:
                        rope_scaling[k_hf] = float(m[p + "rope.scaling." + k_gguf])
                if feats.get("yarn_no_truncate"):
                    rope_scaling["truncate"] = False
        elif rs != "none":
            gaps.append(f"rope.scaling.type={rs!r} not mapped (llama3/longrope factors)")
        if any(rdim_l[i] != hd_l[i] for i in range(nl)):
            gaps.append(f"partial rotary: rope.dimension_count={sorted(set(rdim_l))} != head_dim={sorted(set(hd_l))}")
        qn = infos.get("blk.0.attn_q_norm.weight")
        qk_norm = "none" if qn is None else ("head" if qn[1][0] == hd_l[0] else "span" if qn[1][0] == nq * hd_l[0] else "?")
        if qk_norm == "?":
            gaps.append(f"attn_q_norm width {qn[1][0]} is neither head_dim nor n_heads*head_dim")
        return {
            "hidden_size": ne, "num_attention_heads": nq, "num_key_value_heads": n_kv_l[0], "num_hidden_layers": nl, "head_dim": hd,
            "rms_norm_eps": m.get(p + "attention.layer_norm_rms_epsilon", 1e-6), "max_position_embeddings": m.get(p + "context_length", 4096),
            "rope_theta": fb, "v_norm": bool(feats.get("v_norm")), "sm_scale": feats.get("sm_scale"),
            "rope_prop_layers": rope_prop, "rope_partial_factor": rope_pf, "rope_freqs_pending": rope_pending,
            "n_kv_layers": n_kv_l, "head_dim_layers": hd_l, "rope_theta_layers": theta_l, "rotary_dim_layers": rdim_l,
            "sliding_window": win, "swa_layers": swa, "rope_scaling": rope_scaling,
            "attn_bias": attn_bias, "sinks_layers": sinks_layers, "kv_eq_v_layers": kv_eq_v,
            "attn_logit_softcapping": float(m.get(p + "attn_logit_softcapping", 0.0)), "qk_norm": qk_norm,
        }
    return ctx.once("gqa", f)


def gqa_layer(ctx, i):
    a = gqa_global(ctx)
    return {"head_dim": a["head_dim_layers"][i], "n_q": a["num_attention_heads"], "n_kv": a["n_kv_layers"][i],
            "swa": a["swa_layers"][i], "window": a["sliding_window"], "rope_theta": a["rope_theta_layers"][i],
            "rotary_dim": a["rotary_dim_layers"][i], "rope_prop": i in a["rope_prop_layers"], "bias": a["attn_bias"][i],
            "sinks": i in a["sinks_layers"], "kv_eq_v": i in a["kv_eq_v_layers"], "qk_norm": a["qk_norm"]}


def _names(cfg, i, L, G):
    nm = {f"{L}.self_attn.q_proj.weight": f"{G}.attn_q.weight", f"{L}.self_attn.k_proj.weight": f"{G}.attn_k.weight",
          f"{L}.self_attn.v_proj.weight": f"{G}.attn_v.weight", f"{L}.self_attn.o_proj.weight": f"{G}.attn_output.weight"}
    if cfg.get("qk_norm", "none") != "none":
        nm.update({f"{L}.self_attn.q_norm.weight": f"{G}.attn_q_norm.weight", f"{L}.self_attn.k_norm.weight": f"{G}.attn_k_norm.weight"})
    if i in cfg.get("sinks_layers", ()):
        nm[f"{L}.self_attn.sinks"] = f"{G}.attn_sinks.weight"
    for c, pj, t in (("q", "q_proj", "attn_q"), ("k", "k_proj", "attn_k"), ("v", "v_proj", "attn_v"), ("o", "o_proj", "attn_output")):
        if c in cfg.get("attn_bias", [""] * (i + 1))[i]:
            nm[f"{L}.self_attn.{pj}.bias"] = f"{G}.{t}.bias"
    if i in cfg.get("kv_eq_v_layers", ()):     # no V projection: the attention reads K (use_k_as_v) when the arch has v_norm, else dry-run alias
        if cfg.get("v_norm"):
            nm.pop(f"{L}.self_attn.v_proj.weight", None)
        else:
            nm[f"{L}.self_attn.v_proj.weight"] = f"{G}.attn_k.weight"
    return nm


def _build(config, plan, idx, L):
    import torch
    from ...modules import RMSNorm, Attention
    from ...architecture.gguf_generic import _fl
    H, nq = config.hidden_size, config.num_q_heads
    hd_i, nkv_i = config.head_dim_l[idx], config.n_kv_l[idx]
    norms = {}
    if config.qk_norm != "none":
        norms = {n: RMSNorm(config=config, key=f"{L}.self_attn.{n}", rms_norm_eps=config.rms_norm_eps, span_heads=config.qk_norm == "span")
                 for n in ("q_norm", "k_norm")}
    if config.v_norm:
        norms["v_norm"] = RMSNorm(config=config, key=f"{L}.self_attn.v_norm", rms_norm_eps=config.rms_norm_eps, unweighted=True)
    return Attention(
        config=config, key=f"{L}.self_attn", layer_idx=idx, hidden_size=H, head_dim=hd_i,
        num_q_heads=nq, num_kv_heads=nkv_i, rope_settings=config.layer_rope[idx], sm_scale=config.sm_scale,
        use_k_as_v=idx in config.kv_eq_v, qmap="block.attn", out_dtype=torch.float,
        sliding_window=config.swa_l[idx], logit_softcapping=config.attn_softcap,
        key_sinks="sinks" if idx in config.sinks_layers else None,
        q_proj=_fl(config, f"{L}.self_attn.q_proj", H, nq * hd_i, bias="q" in config.attn_bias[idx]),
        k_proj=_fl(config, f"{L}.self_attn.k_proj", H, nkv_i * hd_i, bias="k" in config.attn_bias[idx]),
        v_proj=None if idx in config.kv_eq_v else _fl(config, f"{L}.self_attn.v_proj", H, nkv_i * hd_i, bias="v" in config.attn_bias[idx]),
        o_proj=_fl(config, f"{L}.self_attn.o_proj", nq * hd_i, H, out_dtype=torch.float, bias="o" in config.attn_bias[idx]), **norms)


def _select(ctx, i):
    ls = ctx.layers.get(i, ())
    return "attn_q.weight" in ls and "attn_kv_a_mqa.weight" not in ls and not any(s.startswith("ssm_") for s in ls)


register(Block("gqa", "attn", select=_select, config=gqa_layer, claims=_CLAIMS, build=_build, names=_names))
