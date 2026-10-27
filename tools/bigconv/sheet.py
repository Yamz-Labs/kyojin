"""Resource sheet: what a conversion needs, from the index + config, before any weight is downloaded.

    bigconv sheet --arch ds41 --index model.safetensors.index.json --config config.json --map tensor_map.json \
        --calib-tokens 2000000 --bpw 2.3 [--ram-gib 100]

Units: GB = 1e9 bytes, GiB = 2**30 bytes; every column says which.

Model (each term is a buffer of the code, nothing is fitted except one base term):
  pack            plan.build_plan at the target bpw (hot/cold K = floor/ceil of the bpw): shards = one per unit.
  RAM / SSD split dense + copy tensors always in RAM; routed experts fill what is left of --ram-gib, the rest is the SSD tier.
  capture RAM     current code (non-memmap): tokens x (hc*hidden + 2*hidden) x 4 B  [row states h, pooled MoE input X, output y]
                  + base (embed fp32 + one layer dense fp32 + 3 experts fp32 + runtime).
                  memmap design (capture_stream.py): one row + chunk buffers + Hessian accumulators, independent of tokens.
  spool / layer   384-like experts x (tri(hidden) + tri(inter)) x 4 B + dense group Hessians (+ 2 tri(inter) per expert with kron-ho).
  tokens/expert   uniform-routing bound: mean = N*topk/E, p10 = mean - 1.2816*sqrt(mean) (real routing is skewed: worse).
  quantize time   3.5 s per expert-linear equivalent (hidden x inter), summed over the plan, divided by workers.
"""
from __future__ import annotations

import json
import math

from . import plan as PL
from .arch import get_adapter

GB, GIB = 1e9, 2.0 ** 30
QUANT_S_PER_EXPERT_LIN = 3.5
RUNTIME_GIB = 2.0           # torch + driver, not data
TRI = lambda n: n * (n + 1) // 2


def load_json(path):
    import gzip
    return json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))


def synth_map(adapter, index: dict) -> dict:
    """tensor_map-like dict from an HF index + config: shapes for the quantizable linears (adapter.shape_of), nothing
    for the rest. Returns the map and the residual bytes (index total_size minus what the shapes explain)."""
    names = set(index["weight_map"])
    T = {}
    known = 0
    for n in sorted(names):
        sh = adapter.shape_of(n, names)
        if sh is None:
            T[n] = {"file": index["weight_map"][n], "dtype": "?", "shape": None, "offset": 0, "length": 0}
            continue
        has_scale = getattr(adapter, "scale_suffix", None) and (n[:-7] + adapter.scale_suffix) in names
        bpw = 1.0 if has_scale else (2.0 if adapter.source_bytes_per_weight != 0.5 else 0.5)
        if adapter.source_bytes_per_weight == 2.0:
            bpw = 2.0
        ln = int(sh[0] * sh[1] * bpw)
        T[n] = {"file": index["weight_map"][n], "dtype": "F8_E4M3" if bpw == 1.0 else "BF16", "shape": list(sh),
                "offset": 0, "length": ln}
        known += ln
    total = index.get("metadata", {}).get("total_size")
    return {"tensors": T, "shards": {}}, known, total


def tokens_per_expert(adapter, n_tokens: int) -> dict:
    mean = n_tokens * adapter.topk / adapter.n_experts
    return {"mean": mean, "p10": max(0.0, mean - 1.2816 * math.sqrt(mean))}


def spool_gb(adapter, n_tokens: int, kron_ho: bool = False, ho_min_tokens: int = 2304) -> dict:
    d, i, E = adapter.hidden, adapter.expert_inter, adapter.n_experts
    per_in, per_mid = TRI(d) * 4, TRI(i) * 4
    exp = E * (per_in + per_mid)
    ho = 0.0
    if kron_ho:
        mean = n_tokens * adapter.topk / E
        sd = math.sqrt(mean) or 1.0
        frac = 0.5 * math.erfc((ho_min_tokens - mean) / (sd * math.sqrt(2)))     # P(tokens >= ho_min), uniform bound
        ho = E * frac * 2 * TRI(i) * 4
    dense = sum(TRI(n) * 4 for n in adapter.dense_hessian_dims())
    return {"experts_gb": exp / GB, "ho_gb": ho / GB, "dense_gb": dense / GB, "total_gb": (exp + ho + dense) / GB}


def capture_ram_gib(adapter, n_tokens: int, row_len: int, plan_summary: dict, memmap: bool, chunk_tokens: int = 16384,
                    state_bytes: int = 4) -> dict:
    d, hc, i = adapter.hidden, adapter.hc_mult, adapter.expert_inter
    embed = adapter.config_vocab * d * 4 if getattr(adapter, "config_vocab", 0) else 0
    dense_fp32 = plan_summary.get("largest_unit_dense_params", 0) * 4
    expert_fp32 = 3 * d * i * 4
    base = embed + dense_fp32 + 2 * expert_fp32 + RUNTIME_GIB * GIB
    hess = sum(n * n * 4 for n in adapter.dense_hessian_dims()) + (d * d + 3 * i * i) * 4
    if not memmap:
        tok = n_tokens * (hc * d * 4 + 2 * d * 4)
        return {"tokens_gib": tok / GIB, "base_gib": base / GIB, "hess_gib": hess / GIB,
                "total_gib": (tok + base + hess) / GIB}
    row = row_len * hc * d * state_bytes
    ctx = 2 * row_len * hc * d * 4                      # attention-half intermediates of one row (fp32)
    chunk = chunk_tokens * (d * 4 * 4 + i * 4 * 2)      # gathered x, out, gate/up/mid temporaries
    routing = n_tokens * adapter.topk * 12              # idx int64 + weight float32, kept in RAM (could be memmap too)
    tot = base + hess + row + ctx + chunk + routing
    return {"row_gib": row / GIB, "ctx_gib": ctx / GIB, "chunk_gib": chunk / GIB, "routing_gib": routing / GIB,
            "base_gib": base / GIB, "hess_gib": hess / GIB, "total_gib": tot / GIB}


def capture_disk_gb(adapter, n_tokens: int, state_bytes: int = 4) -> dict:
    d, hc = adapter.hidden, adapter.hc_mult
    st = n_tokens * hc * d * state_bytes
    x = n_tokens * d * 4
    # writes per trunk layer: h rewritten in place, X once, Y once (lower bound: page cache coalesces the scatter-adds)
    # or topk+1 times (upper bound: every scatter-add reaches the disk)
    w_lo = st + 2 * x
    w_hi = st + x + (adapter.topk + 1) * x
    return {"state_gb": st / GB, "x_gb": x / GB, "y_gb": x / GB, "total_gb": (st + 2 * x) / GB,
            "write_per_layer_lo_gb": w_lo / GB, "write_per_layer_hi_gb": w_hi / GB}


def param_counts(adapter, tmap_tensors: dict, index: dict = None) -> dict:
    """Total / active parameters two ways. (1) header sum: elements of every non-scale tensor of the language model
    (packed 4-bit int8 weights count twice); active = total - routed + routed * topk / E (trunk, no MTP).
    (2) config sum: elements of the linears the adapter derives from the config (no header shapes) + embedding table
    + the copy-tensor residual of the index total_size at 2 bytes/element."""
    names = set(tmap_tensors)
    sfx = getattr(adapter, "scale_suffix", None)
    lang = getattr(adapter, "lang_prefixes", ())
    tot = routed = mtp_routed = engram = 0
    for n, t in tmap_tensors.items():
        if not t.get("shape") or (sfx and n.endswith(sfx)) or n.endswith(".scale"):
            continue
        if lang and not n.startswith(lang):
            continue
        el = 1
        for v in t["shape"]:
            el *= v
        if t.get("dtype") == "I8" and n.endswith(".weight") and adapter.source_bytes_per_weight == 0.5:
            el *= 2
        tot += el
        c = adapter.classify(n, tmap_tensors)
        if c.action == "quant" and c.kind == "routed":
            routed += el
        elif c.action == "quant" and c.kind == "mtp_routed":
            mtp_routed += el
        elif ("engram" in n) and "table" in n.lower() or c.kind == "engram" and c.action != "quant":
            engram += el
    active = tot - routed - mtp_routed - engram + routed * adapter.topk / adapter.n_experts
    out = {"total_b": tot / 1e9, "routed_b": routed / 1e9, "active_b": active / 1e9, "mtp_routed_b": mtp_routed / 1e9, "engram_b": engram / 1e9}
    if index is not None:
        inames = set(index["weight_map"])
        q = 0
        for n in inames:
            if lang and not n.startswith(lang):
                continue
            sh = adapter.shape_of(n, inames)
            if sh and n.endswith(".weight"):
                q += sh[0] * sh[1] * (2 if adapter.source_bytes_per_weight == 0.5 and ".experts." in n else 1)
        emb = getattr(adapter, "config_vocab", 0) * adapter.hidden
        out["config_linears_b"] = q / 1e9
        out["config_b_approx"] = (q + emb) / 1e9 if q > 0.5 * (tot - engram) else None
    return out


def build_sheet(adapter, index: dict, tmap: dict = None, n_tokens: int = 2_000_000, bpw: float = 2.3,
                ram_gib: float = 100.0, row_len: int = 2048, kron_ho: bool = False, bits: dict = None) -> dict:
    residual = None
    if tmap is None:
        tmap, known, total = synth_map(adapter, index)
        residual = (total - known) if total else None
    cold = int(math.floor(bpw))
    hot = cold if bpw == cold else cold + 1
    b = {"hot": hot, "cold": cold}
    b.update(bits or {})
    pl = PL.build_plan(tmap, bpw, adapter, bits=b)
    ents = pl["entries"]
    quant = [e for e in ents if e["action"] == "quant"]
    by = lambda f: sum(e["out_bytes"] for e in ents if f(e))
    routed = by(lambda e: e["action"] == "quant" and e["kind"] == "routed" and e.get("tier") != "mtp")
    mtp_routed = by(lambda e: e["action"] == "quant" and e["kind"] in ("mtp", "routed") and
                    (e.get("tier") == "mtp" or ".experts." in (e["weight"] or "") and e["unit"].startswith("mtp.")))
    quant_b = by(lambda e: e["action"] == "quant")
    copy_b = by(lambda e: e["action"] == "copy")
    raw_b = by(lambda e: e["action"] == "copy_raw")
    units = PL.units(pl, adapter)
    unit_sz = {}
    for e in ents:
        if e["action"] != "copy_raw":
            unit_sz[e["unit"]] = unit_sz.get(e["unit"], 0) + e["out_bytes"]
    dense_q = quant_b - routed
    always_ram = dense_q + copy_b + (residual or 0)
    exp_ram = max(0.0, min(routed, ram_gib * GIB - always_ram))
    # largest unit dense params (capture base term)
    per_unit_dense = {}
    for e in quant:
        if e["kind"] != "routed" and adapter.unit_sort_key(e["unit"])[0] == 1:     # trunk layers only
            per_unit_dense[e["unit"]] = per_unit_dense.get(e["unit"], 0) + e["in"] * e["out"]
    ps = {"largest_unit_dense_params": max(per_unit_dense.values(), default=0)}
    lin_eq = adapter.hidden * adapter.expert_inter
    q_params = sum(e["in"] * e["out"] for e in quant)
    q_s = QUANT_S_PER_EXPERT_LIN * q_params / lin_eq
    n_exp_units = len({e["unit"] for e in quant if e["kind"] == "routed"})
    out = {
        "arch": adapter.name, "bpw": bpw, "calib_tokens": n_tokens, "row_len": row_len,
        "pack": {
            "quantized_gb": quant_b / GB, "copy_gb": copy_b / GB, "unmodeled_copy_gb": (residual or 0) / GB,
            "pack_gb": (quant_b + copy_b + (residual or 0)) / GB, "pack_gib": (quant_b + copy_b + (residual or 0)) / GIB,
            "raw_tensors_gb": raw_b / GB, "raw_tensors": sum(1 for e in ents if e["action"] == "copy_raw"),
            "shards": len(unit_sz), "max_shard_gb": max(unit_sz.values()) / GB, "routed_gb": routed / GB,
            "mtp_routed_gb": mtp_routed / GB, "routed_avg_bpw": pl["summary"]["routed_avg_bpw"],
            "moe_layers": n_exp_units,
        },
        "serve_split": {
            "ram_budget_gib": ram_gib, "always_ram_gib": always_ram / GIB, "routed_gib": routed / GIB,
            "mode": "RAM-only" if routed + always_ram <= ram_gib * GIB else "SSD tier",
            "experts_in_ram_gib": exp_ram / GIB, "experts_on_ssd_gib": (routed - exp_ram) / GIB,
            "ssd_fraction": (routed - exp_ram) / routed if routed else 0.0,
        },
        "tokens_per_expert": tokens_per_expert(adapter, n_tokens),
        "spool_per_layer": spool_gb(adapter, n_tokens, kron_ho=kron_ho),
        "capture_ram_current": capture_ram_gib(adapter, n_tokens, row_len, ps, memmap=False),
        "capture_ram_memmap": capture_ram_gib(adapter, n_tokens, row_len, ps, memmap=True),
        "capture_disk_memmap": capture_disk_gb(adapter, n_tokens),
        "params": param_counts(adapter, tmap["tensors"], index),
        "quantize": {"expert_lin_equivalents": q_params / lin_eq, "single_worker_h": q_s / 3600,
                     "per_moe_layer_min": q_s / max(1, n_exp_units) / 60},
    }
    if residual is not None:
        out["pack"]["note"] = ("shapes derived from the config; copy tensors whose shape the adapter does not know are "
                               "the residual of the index total_size (source bytes, upper bound of their pack bytes)")
    return out


def write_line(s: dict, disk_free_gb: float = None) -> str:
    cd = s["capture_disk_memmap"]
    layers = s["pack"]["moe_layers"]
    lo, hi = cd["write_per_layer_lo_gb"], cd["write_per_layer_hi_gb"]
    r = (f"capture writes {lo:.0f}-{hi:.0f} GB per layer, {lo * layers / 1000:.1f}-{hi * layers / 1000:.1f} TB over {layers} MoE layers "
         f"(footprint {cd['total_gb']:.0f} GB)")
    if hi > 140:
        r += "; above the ~140 GB sustained-write cliff of QLC NVMe drives: run it on a drive without that cliff or chunk X/Y rows"
    return r


def params_line(s: dict) -> str:
    q = s["params"]
    r = f"params          total {q['total_b']:.1f} B, active {q['active_b']:.1f} B (header sum; routed {q['routed_b']:.1f} B"
    if q["mtp_routed_b"]:
        r += f", MTP routed {q['mtp_routed_b']:.1f} B"
    r += ")"
    if q.get("engram_b"):
        r = r.replace("(header sum;", f"(header sum, engram tables {q['engram_b']:.1f} B counted in total only;")
    if q.get("config_b_approx"):
        r += f"; config-derived linears+embed {q['config_b_approx']:.1f} B"
    elif "config_b_approx" in q:
        r += "; config-derived n/a (this adapter needs header shapes)"
    return r


def fmt(s: dict) -> str:
    p, sv, t, sp, cr, cm, cd, q = (s[k] for k in ("pack", "serve_split", "tokens_per_expert", "spool_per_layer",
                                                  "capture_ram_current", "capture_ram_memmap", "capture_disk_memmap",
                                                  "quantize"))
    L = [f"bigconv sheet: {s['arch']} at {s['bpw']} bpw routed, {s['calib_tokens']:,} calibration tokens (rows of {s['row_len']})",
         f"pack            {p['pack_gb']:.1f} GB = {p['pack_gib']:.1f} GiB  (quantized {p['quantized_gb']:.1f} GB, copy {p['copy_gb']:.1f} GB"
         + (f", unmodeled copy {p['unmodeled_copy_gb']:.1f} GB" if p["unmodeled_copy_gb"] else "") + f"); routed avg {p['routed_avg_bpw']:.3f} bpw",
         f"                {p['shards']} shards (one per unit, max {p['max_shard_gb']:.2f} GB), {p['moe_layers']} MoE units, "
         f"routed {p['routed_gb']:.1f} GB, MTP routed {p['mtp_routed_gb']:.1f} GB",
         f"raw tensors     {p['raw_tensors']} tensors, {p['raw_tensors_gb']:.1f} GB (copied byte for byte, outside the quantized pack)",
         f"serve split     {sv['mode']} at {sv['ram_budget_gib']:.0f} GiB: always-RAM {sv['always_ram_gib']:.1f} GiB, "
         f"experts {sv['routed_gib']:.1f} GiB -> RAM {sv['experts_in_ram_gib']:.1f} GiB / SSD {sv['experts_on_ssd_gib']:.1f} GiB "
         f"({100 * sv['ssd_fraction']:.0f} % on SSD)",
         f"tokens/expert   mean {t['mean']:.0f}, p10 {t['p10']:.0f} (uniform-routing bound; skewed routing is worse)",
         f"spool / layer   {sp['total_gb']:.1f} GB (experts {sp['experts_gb']:.1f} + dense groups {sp['dense_gb']:.2f} + kron-ho {sp['ho_gb']:.1f})",
         f"capture RAM     current code {cr['total_gib']:.0f} GiB (tokens {cr['tokens_gib']:.0f} + base {cr['base_gib']:.1f} + H {cr['hess_gib']:.1f}) | "
         f"memmap {cm['total_gib']:.1f} GiB (row {cm['row_gib']:.2f} + ctx {cm['ctx_gib']:.2f} + chunk {cm['chunk_gib']:.2f} + routing {cm['routing_gib']:.2f} + base {cm['base_gib']:.1f})",
         f"capture disk    memmap {cd['total_gb']:.0f} GB (row states {cd['state_gb']:.0f} read+write in place, X {cd['x_gb']:.0f}, Y {cd['y_gb']:.0f}); current code 0",
         write_line(s),
         params_line(s),
         f"quantize        {q['expert_lin_equivalents']:.0f} expert-linear equivalents x {QUANT_S_PER_EXPERT_LIN} s = {q['single_worker_h']:.1f} h on 1 worker "
         f"({q['per_moe_layer_min']:.0f} min per MoE layer)"]
    return "\n".join(L)


def cli(a):
    cfg = load_json(a.config)
    adapter = get_adapter(a.arch, cfg)
    index = load_json(a.index)
    tmap = load_json(a.map) if a.map else None
    if tmap is None and a.arch == "ds41":
        raise SystemExit("ds41 needs --map (shapes are not derived from the config)")
    s = build_sheet(adapter, index, tmap, a.calib_tokens, a.bpw, a.ram_gib, a.row_len, kron_ho=a.kron_ho)
    print(json.dumps(s, indent=1) if a.json else fmt(s))
