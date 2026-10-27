"""Release-day check of a new checkpoint against the closest adapter (config.json + model.safetensors.index.json only, no weights).

    python -m bigconv check --arch glm5 --config new/config.json --index new/model.safetensors.index.json

Reports: config keys added / removed / changed against the adapter's bundled config; tensor-name patterns (digits -> N)
new or gone against the bundled index; `.weight` patterns no role matches (possible unquantized linears); expert and layer
counts versus the config; MTP units; bytes the config-derived shapes do not explain. Exit 1 when the adapter needs work.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import re

from .arch import get_adapter

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS_THAT_MATTER = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                    "num_local_experts", "n_routed_experts", "num_experts_per_tok", "n_shared_experts", "intermediate_size",
                    "moe_intermediate_size", "first_k_dense_replace", "hidden_act", "scoring_func", "routed_scaling_factor",
                    "hc_mult", "num_nextn_predict_layers", "num_mtp_modules", "layer_types", "moe_layer_freq", "architectures",
                    "model_type", "quantization_config")


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def config_diff(new: dict, old: dict):
    a, b = _flat(old), _flat(new)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return dict(added=added, removed=removed, changed=[(k, a[k], b[k]) for k in changed])


def pattern(name):
    return re.sub(r"\d+", "N", name)


def check(arch, config, index, base_config=None, base_index=None):
    from .sheet import synth_map
    ad = get_adapter(arch, config)
    names = set(index["weight_map"])
    rep = dict(arch=arch, problems=[], notes=[])
    if base_config:
        d = config_diff(config, base_config)
        rep["config"] = dict(added=d["added"], removed=d["removed"],
                             changed=[c for c in d["changed"] if any(c[0].endswith(k) or k in c[0] for k in KEYS_THAT_MATTER)],
                             changed_other=len(d["changed"]))
        if d["added"] or d["removed"]:
            rep["notes"].append(f"{len(d['added'])} new / {len(d['removed'])} removed config keys: check each against the capture forward")
    tmap, known, total = synth_map(ad, index)
    T = tmap["tensors"]
    kinds = collections.Counter()
    unclassified = collections.Counter()
    for n in names:
        c = ad.classify(n, T)
        kinds[f"{c.action}:{c.kind}" if c.action == "quant" else c.action] += 1
        if c.action == "copy" and n.endswith(".weight") and T[n]["shape"] is None:
            unclassified[pattern(n)] += 1
    rep["classes"] = dict(kinds)
    if base_index:
        bp = {pattern(n) for n in base_index["weight_map"]}
        np_ = collections.Counter(pattern(n) for n in names)
        rep["new_patterns"] = sorted(p for p in np_ if p not in bp)
        rep["gone_patterns"] = sorted(bp - set(np_))
        if rep["new_patterns"]:
            rep["problems"].append(f"{len(rep['new_patterns'])} tensor-name patterns not in the predecessor: classify each (quant / copy / raw)")
        # an unclassified weight pattern that already existed in the predecessor was a deliberate copy
        unclassified = collections.Counter({p: c for p, c in unclassified.items() if p in rep["new_patterns"]})
    rep["unclassified_weights"] = dict(unclassified)
    if unclassified:
        rep["problems"].append(f"{sum(unclassified.values())} `.weight` tensors match no role (quantize or copy on purpose): "
                               f"{sorted(unclassified)[:5]}")
    # experts / layers versus the config
    per_layer = collections.Counter()
    for n in names:
        e = ad.expert_of(n)
        if e is not None:
            per_layer[e.layer] += 1
    if per_layer:
        projs = len({ad.expert_of(n).proj for n in names if ad.expert_of(n)})
        bad = {L: c for L, c in per_layer.items() if c != ad.n_experts * projs}
        rep["moe_layers"] = len(per_layer)
        if bad:
            rep["problems"].append(f"layers with an unexpected routed-linear count (want {ad.n_experts} x {projs}): {dict(list(bad.items())[:4])}")
    else:
        rep["problems"].append("no routed expert tensors matched: expert regex/naming changed")
    if ad.n_layers:
        ls = {int(m.group(1)) for n in names if (m := getattr(ad, "layer_re", None) and ad.layer_re.match(n))}
        if ls and max(ls) + 1 != ad.n_layers + (ad.n_mtp if getattr(ad, "mtp_from_layer", None) is not None else 0):
            rep["notes"].append(f"layer indices 0..{max(ls)} vs config num_hidden_layers {ad.n_layers} (+{ad.n_mtp} mtp)")
    if total:
        res = total - known
        rep["residual_gb"] = round(res / 1e9, 1)
        rep["total_gb"] = round(total / 1e9, 1)
    rep["todo_ops"] = list(ad.TODO_OPS)
    return rep


def _load(p):
    op = gzip.open if p.endswith(".gz") else open
    return json.load(op(p, "rt"))


def cli(a):
    cfg = json.load(open(a.config))
    idx = json.load(open(a.index))
    bc = os.path.join(HERE, "configs", f"{a.arch}.config.json")
    bi = os.path.join(HERE, "fixtures", f"{a.arch}.index.json.gz")
    rep = check(a.arch, cfg, idx, json.load(open(bc)) if os.path.exists(bc) else None, _load(bi) if os.path.exists(bi) else None)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        for k, v in rep.items():
            if k == "todo_ops":
                continue
            print(f"{k}: {json.dumps(v, default=str)[:900]}")
        print("TODO ops (capture forward):")
        for t in rep["todo_ops"]:
            print("  -", t)
    raise SystemExit(1 if rep["problems"] else 0)
