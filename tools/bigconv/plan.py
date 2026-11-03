"""Bits-per-weight plan for the streaming conversion, generic over the ArchAdapter.

    python -m bigconv plan --arch ds41 --map tensor_map.json --target 2.3 [--freq freq.json] --out plan.json

Same semantics (and, for DS4.1, the same bytes) as tools/ds41/plan.py: every source tensor gets one action
(quant at an integer K / copy / copy_raw), routed experts are mixed hot K / cold K to reach the target average,
the hot set follows an expert-frequency file when given.
"""
from __future__ import annotations

import json
import os
import sys

from . import alloc as AL
from .arch import get_adapter


def exl3_bytes(in_f: int, out_f: int, k: int) -> int:
    """Size of the EXL3 tensors of one linear: trellis int16 [in/16, out/16, 16k], suh, svh, mul1."""
    return (in_f // 16) * (out_f // 16) * 16 * k * 2 + in_f * 2 + out_f * 2 + 4


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_freq(path):
    """Expert-frequency JSON -> {layer_int: {expert_int: count}}. Accepts '3' or 'layers.3' keys."""
    raw = json.load(open(path))
    out = {}
    for lk, d in raw.items():
        out[int(str(lk).split(".")[-1])] = {int(e): float(c) for e, c in d.items()}
    return out


def hot_experts(layer: int, n_hot: int, freq, n_experts: int) -> set:
    if n_hot <= 0:
        return set()
    if n_hot >= n_experts:
        return set(range(n_experts))
    if freq and layer in freq:
        counts = freq[layer]
        order = sorted(range(n_experts), key=lambda e: (-counts.get(e, 0.0), e))
        return set(order[:n_hot])
    return {e for e in range(n_experts) if ((e + 1) * n_hot) // n_experts > (e * n_hot) // n_experts}


def parse_layer_bits(spec: str) -> dict:
    out = {}
    for part in spec.split(","):
        rng, _, k = part.partition(":")
        a, _, b = rng.partition("-")
        for l in range(int(a), int(b or a) + 1):
            out.setdefault(l, int(k))
    return out


def parse_ks(spec) -> list:
    if not spec:
        return []
    return sorted({int(x) for x in str(spec).split(",") if x.strip()})


def parse_role_bits(spec, roles) -> dict:
    out = {}
    for part in (spec or "").split(","):
        if part.strip():
            r, _, k = part.partition("=")
            if r.strip() not in roles:
                raise ValueError(f"unknown attention role {r!r}; roles: {roles}")
            out[r.strip()] = int(k)
    return out


def load_expert_bits(path) -> dict:
    d = json.load(open(path))
    return d.get("expert_bits", d)


def expert_bits_from_entries(entries, adapter, n_experts=None) -> dict:
    """quantization_config.expert_bits: {ffn_key: [K per routed expert]} for every backbone layer fully in `entries`."""
    n_experts = n_experts or adapter.n_experts
    per = {}
    for e in entries:
        ex = adapter.expert_of(e.get("weight", "")) if e.get("kind") == "routed" and \
            e.get("action") in ("quant", "pruned") else None
        if ex is not None and ex.proj == adapter.expert_first_proj:
            per.setdefault(adapter.ffn_key(ex.layer_prefix), {})[ex.e] = e["bits"]
    return {k: [d[i] for i in range(n_experts)] for k, d in sorted(per.items())
            if len(d) == n_experts and set(d) == set(range(n_experts))}


def build_plan(tmap: dict, target: float, adapter, freq=None, bits=None, layer_bits=None, skip_engram=False,
               mtp_queue=False, alt=None, role_bits=None, expert_bits=None, alloc=None) -> dict:
    """role_bits {role: K} overrides the attention K per role; expert_bits {ffn_key: [K per expert, 0 = pruned]} fixes
    the routed K per expert, taking precedence over layer_bits and the hot/cold tiers."""
    b = dict(adapter.default_bits)
    b.update({k: v for k, v in (bits or {}).items() if v is not None})
    layer_dense = {}
    N = adapter.n_experts
    if alloc is not None:
        assert not skip_engram, "an allocation file keeps the original raw tables: drop --no-engram"
        AL.validate(alloc, adapter, tmap)
        b.update(alloc["dense"])
        expert_bits, layer_dense = AL.expert_bits(alloc, adapter, tmap), AL.layer_dense(alloc, adapter, tmap)
    if b.get("mtp_exp") is None:
        b["mtp_exp"] = b["mtp"]
    role_bits = role_bits or {}
    hot, cold = b["hot"], b["cold"]
    frac = 0.0 if hot == cold else min(1.0, max(0.0, (target - cold) / (hot - cold)))
    n_hot = round(frac * N)
    T = tmap["tensors"]
    hot_sets = {}
    entries = []
    alt = alt or {}
    unit_of = adapter.unit_of

    def add_quant(key, kind, wname, sname, rows, in_f, out_f, k, tier=None):
        ent = {
            "action": "quant", "key": key, "kind": kind, "unit": unit_of(wname),
            "weight": wname, "scale": sname, "rows": rows, "in": in_f, "out": out_f,
            "bits": k, "tier": tier, "src_bytes": T[wname]["length"] * (out_f / T[wname]["shape"][0]),
            "out_bytes": exl3_bytes(in_f, out_f, k),
        }
        alts = [x for x in alt.get(kind, []) if x != k]
        if alts and not wname.startswith("mtp."):
            ent["alt_bits"] = alts
        entries.append(ent)

    for name in sorted(T):
        t = T[name]
        unit = unit_of(name)
        c = adapter.classify(name, T)
        if c.action == "raw":
            if skip_engram:
                continue
            entries.append({"action": "copy_raw", "name": name, "unit": unit, "kind": c.kind,
                            "out_bytes": t["length"], "src_bytes": t["length"]})
            continue
        if c.action == "skip":
            continue
        if c.action == "quant":
            base, sname, in_f, out_f = c.base, c.scale, c.in_f, c.out_f
            if c.kind == "routed":
                ex = c.expert
                eb = (expert_bits or {}).get(adapter.ffn_key(ex.layer_prefix))
                if eb is not None:
                    k_e = int(eb[ex.e])
                    assert alloc is None or k_e >= 1, f"{name}: K={k_e}; the allocation never prunes (K >= 1)"
                    if k_e == 0:
                        entries.append({"action": "pruned", "key": base, "kind": "routed", "unit": unit,
                                        "weight": name, "scale": sname, "rows": None, "in": in_f, "out": out_f,
                                        "bits": 0, "tier": "pruned", "src_bytes": 0, "out_bytes": 0})
                    else:
                        add_quant(base, "routed", name, sname, None, in_f, out_f, k_e, "expert")
                    continue
                if layer_bits is not None:
                    add_quant(base, "routed", name, sname, None, in_f, out_f, layer_bits[ex.layer], "layer")
                    continue
                if ex.layer not in hot_sets:
                    hot_sets[ex.layer] = hot_experts(ex.layer, n_hot, freq, N)
                is_hot = ex.e in hot_sets[ex.layer]
                add_quant(base, "routed", name, sname, None, in_f, out_f, hot if is_hot else cold,
                          "hot" if is_hot else "cold")
            elif c.kind == "mtp_routed":
                if mtp_queue:
                    add_quant(base, "routed", name, sname, None, in_f, out_f, b["mtp_exp"], "mtp")
                else:
                    add_quant(base, "mtp", name, sname, None, in_f, out_f, b["mtp"])
            elif c.kind == "mtp":
                add_quant(base, "mtp", name, sname, None, in_f, out_f, b["mtp"])
            elif c.kind == "shared":
                add_quant(base, "shared", name, sname, None, in_f, out_f,
                          layer_dense.get(unit, {}).get("shared", b["shared"]))
            elif c.kind == "head":
                add_quant("head", "head", name, None, None, in_f, out_f, b["head"])
            else:   # attn, dense: attention-class dense linears
                k_dense = role_bits.get(c.role, layer_dense.get(unit, {}).get("attn", b["attn"]))
                if c.slices:
                    for suffix, rng in c.slices:
                        add_quant(f"{base}{suffix}", "attn", name, sname, list(rng), in_f, rng[1] - rng[0], k_dense)
                else:
                    add_quant(base, "attn", name, sname, None, in_f, out_f, k_dense)
            continue
        entries.append({"action": "copy", "name": name, "unit": unit, "kind": "copy",
                        "out_bytes": t["length"], "src_bytes": t["length"]})

    if alloc is not None:
        assert not any(e["action"] == "pruned" for e in entries)
        assert all(e["bits"] >= 1 for e in entries if e["action"] == "quant")
        n_raw = sum(1 for n in T if adapter.classify(n, T).action == "raw")
        assert n_raw == sum(1 for e in entries if e["action"] == "copy_raw"), "raw tensors missing from the plan"
    return {"target": target, "bits": b, "role_bits": role_bits, "hot_fraction": frac, "experts_hot_per_layer": n_hot,
            "freq": bool(freq), "entries": entries, "summary": summarize(entries),
            **({"layer_bits": {str(k): v for k, v in sorted(layer_bits.items())}} if layer_bits else {})}


def summarize(entries) -> dict:
    groups = {}
    params = {}
    for e in entries:
        g = e["kind"]
        if e["action"] == "quant":
            g = f"{g}:{e['bits']}b"
            params[g] = params.get(g, 0) + e["in"] * e["out"]
        groups[g] = groups.get(g, 0) + e["out_bytes"]
    total = sum(groups.values())
    routed_w = sum(e["in"] * e["out"] for e in entries if e["kind"] == "routed" and e.get("tier") != "mtp")
    routed_bits = sum(e["in"] * e["out"] * e["bits"] for e in entries if e["kind"] == "routed" and e.get("tier") != "mtp")
    return {
        "total_bytes": total, "total_gb": total / 1e9,
        "groups_gb": {k: v / 1e9 for k, v in sorted(groups.items())},
        "routed_avg_bpw": routed_bits / routed_w if routed_w else None,
        "quant_params": params,
    }


def units(plan, adapter) -> list:
    seen = []
    for e in plan["entries"]:
        if e["unit"] not in seen:
            seen.append(e["unit"])
    return sorted(seen, key=adapter.unit_sort_key)


def cli(a):
    adapter = get_adapter(a.arch, json.load(open(a.config)) if a.config else None)
    tm = json.load(open(a.map))
    bits = {k: getattr(a, f"{k}_bits") for k in adapter.default_bits}
    p = build_plan(tm, a.target, adapter, load_freq(a.freq) if a.freq else None, bits,
                   parse_layer_bits(a.layer_bits) if a.layer_bits else None, a.no_engram, a.mtp_queue,
                   {"attn": parse_ks(a.alt_attn), "shared": parse_ks(a.alt_shared)},
                   parse_role_bits(a.role_bits, adapter.attn_roles),
                   load_expert_bits(a.expert_bits_file) if a.expert_bits_file else None,
                   json.load(open(a.alloc)) if a.alloc else None)
    if a.out:
        atomic_write_json(a.out, p)
    s = p["summary"]
    print(f"target {a.target}: hot fraction {p['hot_fraction']:.3f}, routed avg {s['routed_avg_bpw']:.4f} bpw, "
          f"total {s['total_gb']:.1f} GB")
    for k, v in s["groups_gb"].items():
        print(f"  {k:14s} {v:9.2f} GB")
