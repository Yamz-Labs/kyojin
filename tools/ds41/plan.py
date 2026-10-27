"""Bits-per-weight plan for the streaming conversion.

    python tools/ds41/plan.py --map tensor_map.json --target 2.3 [--freq freq.json] --out plan.json

Every tensor of the checkpoint is assigned one action:

* quant      linear weight quantized to EXL3 at an integer bit width K (the format stores K bits per
             weight plus per-row/column sign vectors)
* copy       small tensor stored as it is (norms, router gate, hyper-connection params, embeddings,
             vision tower, bf16 projections)
* copy_raw   engram tables only (layers.N.engram.embed.{weight,scale}), raw bytes, never quantized; the engram projection
             (wkv) is an ordinary quantized linear of its layer (kind engram_proj)

Defaults: attention and indexer 5 bits, shared experts 5, MTP block (attention, shared and routed
experts, input projection) 4, output head 6, routed experts in two tiers (hot 3 bits, cold 2 bits).
EXL3 bit widths are integers per tensor, so the routed average is reached by mixing tiers: the hot
fraction is (target - cold) / (hot - cold). With an expert-frequency file ({layer: {expert: count}})
the hot tier goes to the most used experts of each layer; without it the hot experts are spread
evenly over the expert index.

The target applies to the backbone routed experts; the plan summary lists every group separately.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PS = C.packspec()      # tensor names / sizes / quantization_config fields: exllamav3/dsv41/packspec.py (one spec)

DEFAULT_BITS = {"attn": 5, "shared": 5, "mtp": 4, "head": 6, "hot": 3, "cold": 2, "mtp_exp": None, "engram": None}
ATTN_ROLES = ("q_a", "kv", "q_b", "wo_a", "wo_b", "idx_q_b")
RE_ATTN_ROLE = [
    ("q_a", re.compile(r"\.attn\.wq_a\.weight$")), ("kv", re.compile(r"\.attn\.wkv\.weight$")),
    ("q_b", re.compile(r"\.attn\.wq_b\.weight$")), ("wo_a", re.compile(r"\.attn\.wo_a\.weight$")),
    ("wo_b", re.compile(r"\.attn\.wo_b\.weight$")), ("idx_q_b", re.compile(r"\.attn\.indexer\.wq_b\.weight$")),
]
N_EXPERTS = 384
O_GROUPS = 8

RE_EXPERT = re.compile(r"^(layers\.\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
RE_MTP_EXPERT = re.compile(r"^mtp\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
RE_LAYER = re.compile(r"^(layers|mtp)\.(\d+)\.")
RE_ENGRAM = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")   # the hashed tables ONLY (not wkv / q_weight / k_weight)

exl3_bytes = PS.exl3_bytes


def unit_of(name: str) -> str:
    """Shard unit. Engram tables get their own (legacy, raw-copy) unit `layers.N.engram`; the engram projection and the
    q / k weights are ordinary tensors of layer N."""
    m = RE_ENGRAM.match(name)
    if m:
        return f"layers.{m.group(1)}.engram"
    return PS.unit_of(name)


def load_freq(path):
    """Expert-frequency JSON -> {layer_int: {expert_int: count}}. Accepts '3' or 'layers.3' keys."""
    raw = json.load(open(path))
    out = {}
    for lk, d in raw.items():
        li = int(str(lk).split(".")[-1])
        out[li] = {int(e): float(c) for e, c in d.items()}
    return out


def hot_experts(layer: int, n_hot: int, freq, n: int = N_EXPERTS) -> set:
    """Pick the hot expert set of one layer."""
    if n_hot <= 0:
        return set()
    if n_hot >= n:
        return set(range(n))
    if freq and layer in freq:
        counts = freq[layer]
        order = sorted(range(n), key=lambda e: (-counts.get(e, 0.0), e))
        return set(order[:n_hot])
    # evenly spread: expert e is hot when floor((e+1) * n_hot / N) increases
    return {e for e in range(n) if ((e + 1) * n_hot) // n > (e * n_hot) // n}


def infer_n_experts(T: dict) -> int:
    """Routed experts per backbone layer, counted in a tensor map (384 for V4.1-Flash)."""
    rx = re.compile(r"^layers\.\d+\.ffn\.experts\.(\d+)\.w1\.weight$")
    ids = [int(m.group(1)) for m in (rx.match(n) for n in T) if m]
    return max(ids) + 1 if ids else N_EXPERTS


def parse_layer_bits(spec: str) -> dict:
    """'0-9:3,10-29:2,30-39:3' -> {layer: K}. One K for all routed experts of a layer (the serving loader asserts one K
    per layer); the first matching range wins."""
    out = {}
    for part in spec.split(","):
        rng, _, k = part.partition(":")
        a, _, b = rng.partition("-")
        for l in range(int(a), int(b or a) + 1):
            out.setdefault(l, int(k))
    return out


def parse_ks(spec) -> list:
    """'3,5' -> [3, 5]; '' / None -> []."""
    if not spec:
        return []
    return sorted({int(x) for x in str(spec).split(",") if x.strip()})


def parse_role_bits(spec) -> dict:
    """'q_b=3,wo_b=3,kv=5' -> {role: K} (attention roles of ATTN_ROLES); '' / None -> {}."""
    out = {}
    for part in (spec or "").split(","):
        if part.strip():
            r, _, k = part.partition("=")
            if r.strip() not in ATTN_ROLES:
                raise ValueError(f"unknown attention role {r!r}; roles: {ATTN_ROLES}")
            out[r.strip()] = int(k)
    return out


def load_expert_bits(path) -> dict:
    """JSON {"layers.L.ffn": [K per expert, 0 = pruned]} (ramfit.json / alloc.json 'expert_bits') -> same dict."""
    d = json.load(open(path))
    return d.get("expert_bits", d)


def expert_bits_from_entries(entries, n_experts: int = None) -> dict:
    """quantization_config.expert_bits: {"layers.L.ffn": [K per routed expert]} for every backbone layer whose experts
    are all in `entries` (the spec's function; K = 0 marks a pruned expert: no tensors in the pack, the router bias
    masks it)."""
    return PS.expert_bits_from_entries(entries, n_experts or N_EXPERTS)


def build_plan(tmap: dict, target: float, freq=None, bits=None, layer_bits=None, skip_engram=False, mtp_queue=False,
               alt=None, role_bits=None, expert_bits=None, o_groups: int = O_GROUPS, n_experts: int = None) -> dict:
    """role_bits {role: K} overrides the attention K per role (wq_a / wkv / wq_b / wo_a / wo_b / indexer wq_b);
    expert_bits {"layers.L.ffn": [K per expert, 0 = pruned]} fixes the routed K per expert (K = 1 allowed), taking
    precedence over layer_bits and the hot/cold tiers. Layers not listed there fall back to those."""
    b = dict(DEFAULT_BITS)
    b.update({k: v for k, v in (bits or {}).items() if v is not None})
    if b.get("mtp_exp") is None:
        b["mtp_exp"] = b["mtp"]
    if b.get("engram") is None:
        b["engram"] = b["attn"]          # the engram projection (wkv) is a plain linear of its layer
    N = n_experts or infer_n_experts(tmap["tensors"])
    role_bits = role_bits or {}
    hot, cold = b["hot"], b["cold"]
    frac = 0.0 if hot == cold else min(1.0, max(0.0, (target - cold) / (hot - cold)))
    n_hot = round(frac * N)
    T = tmap["tensors"]
    hot_sets = {}
    entries = []
    handled = set()

    alt = alt or {}

    def add_quant(key, kind, wname, sname, rows, in_f, out_f, k, tier=None):
        ent = {
            "action": "quant", "key": key, "kind": kind, "unit": unit_of(wname),
            "weight": wname, "scale": sname, "rows": rows, "in": in_f, "out": out_f,
            "bits": k, "tier": tier, "src_bytes": T[wname]["length"] * (out_f / T[wname]["shape"][0]),
            "out_bytes": exl3_bytes(in_f, out_f, k),
        }
        alts = [x for x in alt.get(kind, []) if x != k]
        if alts and not wname.startswith("mtp."):
            ent["alt_bits"] = alts     # also quantized at these K from the same Hessian (candidates, not in the shard)
        entries.append(ent)

    for name in sorted(T):
        if name in handled:
            continue
        t = T[name]
        unit = unit_of(name)
        if RE_ENGRAM.match(name):
            if skip_engram:
                continue     # the engram tables (MXFP4 transcode) are added to the pack outside this conversion
            entries.append({"action": "copy_raw", "name": name, "unit": unit, "kind": "engram",
                            "out_bytes": t["length"], "src_bytes": t["length"]})
            continue
        if name.endswith(".scale") and name[:-6] + ".weight" in T:
            continue  # consumed with its weight
        if name.endswith(".weight") and name[:-7] + ".scale" in T:
            sname = name[:-7] + ".scale"
            handled.add(sname)
            out_f = t["shape"][0]
            in_f = t["shape"][1] * (2 if t["dtype"] == "I8" else 1)
            base = name[:-7]
            m = RE_EXPERT.match(name)
            if m:
                layer, e = int(m.group(1).split(".")[1]), int(m.group(2))
                eb = (expert_bits or {}).get(f"{m.group(1)}.ffn")
                if eb is not None:
                    k_e = int(eb[e])
                    if k_e == 0:
                        entries.append({"action": "pruned", "key": base, "kind": "routed", "unit": unit_of(name),
                                        "weight": name, "scale": sname, "rows": None, "in": in_f, "out": out_f,
                                        "bits": 0, "tier": "pruned", "src_bytes": 0, "out_bytes": 0})
                    else:
                        add_quant(base, "routed", name, sname, None, in_f, out_f, k_e, "expert")
                    continue
                if layer_bits is not None:
                    add_quant(base, "routed", name, sname, None, in_f, out_f, layer_bits[layer], "layer")
                    continue
                if layer not in hot_sets:
                    hot_sets[layer] = hot_experts(layer, n_hot, freq, N)
                is_hot = e in hot_sets[layer]
                add_quant(base, "routed", name, sname, None, in_f, out_f, hot if is_hot else cold,
                          "hot" if is_hot else "cold")
            elif name.startswith("mtp."):
                if mtp_queue and RE_MTP_EXPERT.match(name):
                    # MTP routed experts go through the expert queue (uncalibrated unless a capture feeds them)
                    add_quant(base, "routed", name, sname, None, in_f, out_f, b["mtp_exp"], "mtp")
                elif name.endswith("attn.wo_a.weight"):          # sliced per output group like the backbone's
                    g = out_f // o_groups
                    for i in range(o_groups):
                        add_quant(f"{base}.slice.{i}", "mtp", name, sname, [i * g, (i + 1) * g], in_f, g, b["mtp"])
                else:
                    add_quant(base, "mtp", name, sname, None, in_f, out_f, b["mtp"])
            elif ".shared_experts." in name:
                add_quant(base, "shared", name, sname, None, in_f, out_f, b["shared"])
            elif ".engram.wkv." in name:
                add_quant(base, "engram_proj", name, sname, None, in_f, out_f, b["engram"])
            elif name.endswith("attn.wo_a.weight"):
                g = out_f // o_groups
                for i in range(o_groups):
                    add_quant(f"{base}.slice.{i}", "attn", name, sname, [i * g, (i + 1) * g], in_f, g,
                              role_bits.get("wo_a", b["attn"]))
            else:
                role = next((r for r, rx in RE_ATTN_ROLE if rx.search(name)), None)
                add_quant(base, "attn", name, sname, None, in_f, out_f, role_bits.get(role, b["attn"]))
            continue
        if name == "head.weight":
            add_quant("head", "head", name, None, None, t["shape"][1], t["shape"][0], b["head"])
            continue
        ent = {"action": "copy", "name": name, "unit": unit, "kind": "copy",
               "out_bytes": t["length"], "src_bytes": t["length"]}
        mg = re.match(r"^(layers\.\d+\.ffn)\.gate\.bias(_vl)?$", name)
        if mg and expert_bits and mg.group(1) in expert_bits:      # pruned experts: the router must never pick them
            pr = [i for i, k in enumerate(expert_bits[mg.group(1)]) if int(k) == 0]
            if pr:
                ent["mask_experts"] = pr
        entries.append(ent)

    return {"target": target, "bits": b, "role_bits": role_bits, "hot_fraction": frac, "n_experts": N, "o_groups": o_groups, "experts_hot_per_layer": n_hot,
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
    routed_bits = sum(e["in"] * e["out"] * e["bits"] for e in entries
                      if e["kind"] == "routed" and e.get("tier") != "mtp")
    return {
        "total_bytes": total, "total_gb": total / 1e9,
        "groups_gb": {k: v / 1e9 for k, v in sorted(groups.items())},
        "routed_avg_bpw": routed_bits / routed_w if routed_w else None,
        "quant_params": params,
    }


def units(plan) -> list:
    """Unit ids in processing order (global, layers, mtp, engram tables last)."""
    seen = []
    for e in plan["entries"]:
        if e["unit"] not in seen:
            seen.append(e["unit"])

    def order(u):
        if u == "global":
            return (0, 0, "")
        if u.endswith(".engram"):
            return (3, int(u.split(".")[1]), "")
        pre, idx = u.split(".")
        return (1 if pre == "layers" else 2, int(idx), "")
    return sorted(seen, key=order)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--target", type=float, default=2.3, help="average routed-expert bpw (2.0/2.3/2.6)")
    ap.add_argument("--freq", help="expert frequency JSON: layer -> expert -> count")
    ap.add_argument("--out")
    ap.add_argument("--layer-bits", help="per-layer K of the routed experts, e.g. '0-9:3,10-29:2,30-39:3'")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--mtp-queue", action="store_true")
    ap.add_argument("--alt-attn", help="extra candidate K for attention/indexer linears, e.g. '3,5'")
    ap.add_argument("--alt-shared", help="extra candidate K for shared-expert linears, e.g. '3,5'")
    ap.add_argument("--role-bits", help="attention K per role, e.g. 'q_b=3,wo_a=3,wo_b=3,q_a=5,kv=5' (default: --attn-bits)")
    ap.add_argument("--expert-bits-file", help="JSON {'layers.L.ffn': [K per expert, 0 = pruned]} (ramfit / alloc output)")
    for k, v in DEFAULT_BITS.items():
        ap.add_argument(f"--{k.replace('_', '-')}-bits", type=int, default=v)
    a = ap.parse_args()
    tm = json.load(open(a.map))
    p = build_plan(tm, a.target, load_freq(a.freq) if a.freq else None,
                   {k: getattr(a, f"{k}_bits") for k in DEFAULT_BITS},
                   parse_layer_bits(a.layer_bits) if a.layer_bits else None, a.no_engram, a.mtp_queue,
                   {"attn": parse_ks(a.alt_attn), "shared": parse_ks(a.alt_shared)},
                   parse_role_bits(a.role_bits), load_expert_bits(a.expert_bits_file) if a.expert_bits_file else None)
    if a.out:
        C.atomic_write_json(a.out, p)
    s = p["summary"]
    print(f"target {a.target}: hot fraction {p['hot_fraction']:.3f}, routed avg {s['routed_avg_bpw']:.4f} bpw, "
          f"total {s['total_gb']:.1f} GB")
    for k, v in s["groups_gb"].items():
        print(f"  {k:14s} {v:9.2f} GB")
