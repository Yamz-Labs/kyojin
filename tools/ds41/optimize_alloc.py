"""Bit allocation optimizer for the DS4.1 EXL3 pack: choose K per routed expert and per attention/shared module AFTER
the conversion, from the candidates the conversion measured (alloc/<unit>.json tables + cands/).

    python optimize_alloc.py --tables OUT --plan OUT/plan.json --min-toks 25 --out OUT/alloc.json

Model
-----
quality  D = sum over layers of s_l * (relative output error of the MoE block + of the attention block), first order,
         errors of independent modules add:
           expert e, K:   share_e * (eps_w2 + nu (eps_w1 + eps_w3)),    share_e = w2sum_e * E_w2,e / (n_tokens * Y_l)
                          (E_w2 = mean energy of the expert's output on its routed tokens, w2sum = sum of squared routing
                          weights, Y_l = mean ||MoE output||^2 of the layer; so share_e = the expert's share of the MoE
                          output energy and the bracket is its relative error, proxy error = Hessian-weighted)
           shared expert: (E_sh / Y_l) * (same bracket), one K for w1/w2/w3
           attention:     one K per role (q_a, kv, q_b, wo_a x8, wo_b, indexer q_b), D = w_attn * sum of the proxy errors of
                          the role's linears (relative errors of a chain add to first order)
         s_l = per-layer sensitivity (default 1, `--layer-weights`); the global scalars w_attn / w_shared are the knobs
         the KLD validation calibrates. D is a ranking proxy, not a KLD; validate the chosen mix with kld_eval.
speed    seconds per decoded token = sum over modules of P(read) * bytes / bandwidth. Dense modules are read every token
         (RAM bandwidth). A routed expert is read with P = 1 - (1 - p_e)^rows (p_e = count_e / n_tokens, rows = tokens per
         step: 1 plain, ~5 with MTP verify), from RAM if it is in the resident set, else from the NVMe (bw_ssd, times
         (1 - ssd_hit) for what a dynamic cache would catch) plus the RAM read. Hot experts therefore cost speed per
         bit, cold experts cost only RAM (resident) or only disk (streamed).
RAM      resident set: attention, shared, head, MTP, misc, runtime (KV + arenas) + the routed experts chosen resident
         (tier decision per expert, made by the optimizer together with K) <= ram_gib.
disk     total pack bytes (optional cap).
solver   multiple-choice knapsack with 3 linear constraints (RAM, disk, time): Lagrangian relaxation (cyclic bisection of
         the multipliers on the dual), then greedy repair of any violation and greedy fill of the slack. Compared with
         brute force on small instances in tests/ds41/test_alloc.py.
"""

import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GiB = 1024 ** 3
INF = float("inf")

RE_LIN = [
    ("q_a", re.compile(r"^(layers\.\d+)\.attn\.wq_a$")),
    ("kv", re.compile(r"^(layers\.\d+)\.attn\.wkv$")),
    ("q_b", re.compile(r"^(layers\.\d+)\.attn\.wq_b$")),
    ("wo_a", re.compile(r"^(layers\.\d+)\.attn\.wo_a\.slice\.\d+$")),
    ("wo_b", re.compile(r"^(layers\.\d+)\.attn\.wo_b$")),
    ("idx_q_b", re.compile(r"^(layers\.\d+)\.attn\.indexer\.wq_b$")),
    ("shared", re.compile(r"^(layers\.\d+)\.ffn\.shared_experts\.w[123]$")),
]


# ------------------------------------------------------------------------------------------------ loading
def load_tables(out_dir: str) -> dict:
    d = os.path.join(out_dir, "alloc")
    tabs = {}
    for fn in sorted(os.listdir(d)):
        if fn.startswith("layers.") and fn.endswith(".json"):
            t = json.load(open(os.path.join(d, fn)))
            tabs[t["unit"]] = t
    return tabs


def layer_idx(unit: str) -> int:
    return int(unit.split(".")[1])


# ------------------------------------------------------------------------------------------------ items
class Items:
    """Choice items with their options. Arrays are [N, O] (O = max options, padded with D = inf)."""

    def __init__(self):
        self.label, self.kind, self.meta = [], [], []
        self.D, self.ram, self.disk, self.time = [], [], [], []

    def add(self, label, kind, opts, meta=None):
        """opts: list of (name, D, ram_bytes, disk_bytes, seconds)."""
        self.label.append(label)
        self.kind.append(kind)
        self.meta.append({"opts": [o[0] for o in opts], **(meta or {})})
        self.D.append([o[1] for o in opts])
        self.ram.append([o[2] for o in opts])
        self.disk.append([o[3] for o in opts])
        self.time.append([o[4] for o in opts])

    def finish(self):
        n = len(self.D)
        o = max(len(r) for r in self.D) if n else 1
        pad = lambda rows, v: np.array([r + [v] * (o - len(r)) for r in rows], dtype=np.float64).reshape(n, o)
        self.D_ = pad(self.D, INF)
        self.A = np.stack([pad(self.ram, 0.0), pad(self.disk, 0.0), pad(self.time, 0.0)])   # [3, N, O]
        self.N, self.O = n, o
        return self


def read_prob(count: float, n_tokens: float, rows: int) -> float:
    p = min(1.0, count / max(1.0, n_tokens))
    return 1.0 - (1.0 - p) ** rows


def build_items(tabs: dict, prm: dict) -> Items:
    """Items from the allocation tables. prm: nu, w_attn, w_shared, layer_weights, ks, attn_ks, bw_ram, bw_ssd, ssd_hit,
    rows, smooth."""
    it = Items()
    nu = prm["nu"]
    c_ram = 1.0 / prm["bw_ram"]
    c_ssd = (1.0 - prm["ssd_hit"]) / prm["bw_ssd"]
    for unit, t in sorted(tabs.items(), key=lambda kv: layer_idx(kv[0])):
        L = layer_idx(unit)
        sl = prm["layer_weights"].get(str(L), prm["layer_weights"].get("default", 1.0))
        st = t.get("stats") or {}
        n_tok = float(st.get("n_tokens", 0) or 0)
        Y = st.get("moe_out_energy")
        experts = t.get("experts") or {}
        counts = {int(e): float(r.get("count", 0)) for e, r in experts.items()}
        mean_c = (sum(counts.values()) / len(counts)) if counts else 0.0
        for e_s, r in sorted(experts.items(), key=lambda kv: int(kv[0])):
            e = int(e_s)
            c = counts[e]
            # Dirichlet-style smoothing: an expert the calibration never routed to is not assumed to be never used
            c_eff = c + prm["smooth"] * mean_c
            p = read_prob(c_eff, n_tok or 1.0, prm["rows"])
            en = (r.get("E") or {}).get("w2")
            share = 0.0
            if Y and n_tok and en is not None and c > 0:
                share = (r["w2sum"] * en) / (n_tok * Y)
            opts = []
            for k_s, kr in sorted(r["k"].items(), key=lambda kv: int(kv[0])):
                k = int(k_s)
                if prm["ks"] and k not in prm["ks"]:
                    continue
                px = kr.get("proxy") or {}
                if not all(w in px for w in ("w1", "w3", "w2")) and share > 0:
                    continue   # incomplete candidate: not selectable
                eps = px.get("w2", 0.0) + nu * (px.get("w1", 0.0) + px.get("w3", 0.0))
                d = sl * share * eps
                size = float(kr["bytes"])
                opts.append((f"K{k}:ram", d, size, size, p * size * c_ram))
                if not prm.get("resident_only"):
                    opts.append((f"K{k}:ssd", d, 0.0, size, p * size * (c_ssd + c_ram)))
            if opts:
                it.add(f"{unit}.e{e}", "expert", opts, {"unit": unit, "e": e, "p": p,
                                                       "ks": sorted({int(o[0][1:].split(':')[0]) for o in opts})})
        # dense modules
        groups = {}
        for key, row in (t.get("linears") or {}).items():
            for role, rx in RE_LIN:
                m = rx.match(key)
                if m:
                    groups.setdefault(role, []).append((key, row))
                    break
        sh_energy = None
        if "shared" in groups:
            for key, row in groups["shared"]:
                if key.endswith(".w2") and row.get("E") is not None:
                    sh_energy = row["E"]
        for role, lins in sorted(groups.items()):
            kset = None
            for _, row in lins:
                ks_ = {int(k) for k, v in row["k"].items() if v.get("proxy") is not None}
                kset = ks_ if kset is None else (kset & ks_)
            kset = sorted(kset or [])
            if role == "shared":
                scale = (sh_energy / Y) if (sh_energy and Y) else 0.0
                w = prm["w_shared"] * sl * scale
            else:
                w = prm["w_attn"] * sl
            opts = []
            for k in kset:
                if prm["attn_ks"] and k not in prm["attn_ks"]:
                    continue
                if role == "shared":
                    px = {key.rsplit(".", 1)[1]: row["k"][str(k)]["proxy"] for key, row in lins}
                    eps = px.get("w2", 0.0) + nu * (px.get("w1", 0.0) + px.get("w3", 0.0))
                else:
                    eps = sum(row["k"][str(k)]["proxy"] for _, row in lins)
                size = float(sum(row["k"][str(k)]["bytes"] for _, row in lins))
                opts.append((f"K{k}:ram", w * eps, size, size, size * c_ram))
            if opts:
                it.add(f"{unit}.{role}", role if role == "shared" else "attn", opts,
                       {"unit": unit, "role": role, "keys": [k for k, _ in lins]})
    return it.finish()


# ------------------------------------------------------------------------------------------------ solver
def usage(it: Items, ch: np.ndarray) -> np.ndarray:
    idx = np.arange(it.N)
    return np.array([it.A[c][idx, ch].sum() for c in range(3)])


def total_D(it: Items, ch: np.ndarray) -> float:
    return float(it.D_[np.arange(it.N), ch].sum())


def lagr_pick(it: Items, lam: np.ndarray) -> np.ndarray:
    cost = it.D_ + lam[0] * it.A[0] + lam[1] * it.A[1] + lam[2] * it.A[2]
    return cost.argmin(axis=1)


def solve(it: Items, caps: np.ndarray, verbose=False, max_moves=6000, iters=None):
    """min sum D s.t. usage <= caps (caps[c] = inf: unconstrained). Returns (option index per item, feasible).

    Dual: one multiplier per finite cap, nested bisection (outermost time, then disk, innermost RAM; for fixed outer
    multipliers the inner usage is monotone in its own multiplier, so each level is a plain bisection). The pick for the
    smallest feasible multipliers is then repaired / filled greedily (items on the threshold move one at a time)."""
    caps = np.asarray(caps, dtype=np.float64)
    scale = [max(1.0, float(np.nanmax(np.where(np.isfinite(it.A[c]), it.A[c], 0)))) for c in range(3)]
    dmax = float(np.max(it.D_[np.isfinite(it.D_)])) if np.isfinite(it.D_).any() else 1.0
    levels = [c for c in (2, 1, 0) if np.isfinite(caps[c])]
    iters = iters or {0: 70, 1: 60, 2: 50}.get(len(levels), 40)
    iters = {1: 70, 2: 48, 3: 34}.get(len(levels), 34)
    lam = np.zeros(3)

    def pick(lv):
        if not lv:
            return lagr_pick(it, lam)
        c, rest = lv[0], lv[1:]
        lam[c] = 0.0
        ch = pick(rest)
        if usage(it, ch)[c] <= caps[c]:
            return ch
        lo, hi = 0.0, dmax / scale[c] * 1e-4
        while True:
            lam[c] = hi
            ch = pick(rest)
            if usage(it, ch)[c] <= caps[c] or hi > 1e30:
                break
            lo, hi = hi, hi * 8.0
        best = ch
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lam[c] = mid
            ch = pick(rest)
            if usage(it, ch)[c] > caps[c]:
                lo = mid
            else:
                hi, best = mid, ch
        lam[c] = hi
        return best

    ch = pick(levels)
    ch = repair(it, ch, caps, max_moves)
    ch = fill(it, ch, caps, max_moves)
    ok = bool(np.all(usage(it, ch) <= caps * (1 + 1e-9)))
    if verbose or not ok:
        print("lambda", lam, "usage", usage(it, ch), "caps", caps, "feasible", ok)
    return ch, ok


def _moves(it: Items, ch: np.ndarray):
    """For every item and option: (dD, dA[3]) of switching from the current option."""
    idx = np.arange(it.N)
    dD = it.D_ - it.D_[idx, ch][:, None]
    dA = it.A - it.A[:, idx, ch][:, :, None]
    return dD, dA


def repair(it: Items, ch: np.ndarray, caps: np.ndarray, max_moves: int) -> np.ndarray:
    ch = ch.copy()
    fin = np.isfinite(caps)
    for _ in range(max_moves):
        u = usage(it, ch)
        over = np.where(fin, np.maximum(0.0, u / np.where(fin, caps, 1.0) - 1.0), 0.0)
        if not np.any(over > 1e-12):
            break
        dD, dA = _moves(it, ch)
        # normalized reduction of the violated constraints (positive = better)
        gain = np.zeros_like(dD)
        for c in range(3):
            if fin[c] and over[c] > 1e-12:
                gain += np.minimum(-dA[c], over[c] * caps[c]) / caps[c]
        ok = np.isfinite(dD) & (gain > 1e-15)
        if not ok.any():
            break
        ratio = np.where(ok, np.maximum(dD, 0.0) / np.where(ok, gain, 1.0) + (dD < 0) * -1e-30, INF)
        i, o = np.unravel_index(np.argmin(ratio), ratio.shape)
        ch[i] = o
    return ch


def fill(it: Items, ch: np.ndarray, caps: np.ndarray, max_moves: int) -> np.ndarray:
    """Use the remaining slack: best quality gain per normalized resource, moves that keep every cap."""
    ch = ch.copy()
    fin = np.isfinite(caps)
    for _ in range(max_moves):
        u = usage(it, ch)
        slack = np.where(fin, caps - u, INF)
        dD, dA = _moves(it, ch)
        feas = np.isfinite(dD) & (dD < -1e-18)
        for c in range(3):
            if fin[c]:
                feas &= dA[c] <= slack[c] + 1e-9
        if not feas.any():
            break
        cost = np.zeros_like(dD)
        for c in range(3):
            if fin[c]:
                cost += np.maximum(dA[c], 0.0) / caps[c]
        score = np.where(feas, -dD / np.maximum(cost, 1e-30), -INF)
        i, o = np.unravel_index(np.argmax(score), score.shape)
        ch[i] = o
    return ch


# ------------------------------------------------------------------------------------------------ evaluation
def describe(it: Items, ch: np.ndarray, fixed: dict) -> dict:
    idx = np.arange(it.N)
    kinds = np.array(it.kind)
    names = [it.meta[i]["opts"][ch[i]] for i in range(it.N)]
    ram = np.array([it.A[0][i, ch[i]] for i in range(it.N)])
    dsk = np.array([it.A[1][i, ch[i]] for i in range(it.N)])
    tm = np.array([it.A[2][i, ch[i]] for i in range(it.N)])
    D = it.D_[idx, ch]
    ssd = np.array([n.endswith(":ssd") for n in names])
    res = {"D": float(D.sum())}
    for kd in ("expert", "attn", "shared"):
        m = kinds == kd
        res[f"D_{kd}"] = float(D[m].sum())
        res[f"disk_gb_{kd}"] = float(dsk[m].sum() / 1e9)
        res[f"time_ms_{kd}"] = float(tm[m].sum() * 1e3)
    me = kinds == "expert"
    res["time_ms_expert_ram"] = float(tm[me & ~ssd].sum() * 1e3)
    res["time_ms_expert_ssd"] = float(tm[me & ssd].sum() * 1e3)
    res["ram_gib_experts"] = float(ram[me].sum() / GiB)
    res["ram_gib_dense"] = float(ram[~me].sum() / GiB)
    res["n_experts_k"] = {}
    for i in np.where(me)[0]:
        k = names[i].split(":")[0]
        res["n_experts_k"][k] = res["n_experts_k"].get(k, 0) + 1
    res["n_experts_resident"] = int((me & ~ssd).sum())
    res["n_experts_ssd"] = int((me & ssd).sum())
    res["disk_gb_items"] = float(dsk.sum() / 1e9)
    res["disk_gb_total"] = res["disk_gb_items"] + fixed["disk_gb"]
    res["ram_gib_total"] = res["ram_gib_experts"] + res["ram_gib_dense"] + fixed["ram_gib"]
    t_ms = float(tm.sum() * 1e3) + fixed["time_ms"]
    res["time_ms"] = t_ms
    res["ceiling_toks"] = 1e3 / t_ms
    # bytes read per token (GB): dense + experts; the speed number that matters for the roofline
    prm = fixed["prm"]
    res["read_gb_dense"] = float(sum(it.A[0][i, ch[i]] for i in range(it.N) if not me[i]) / 1e9) + fixed["read_gb"]
    pe = np.array([it.meta[i].get("p", 0.0) for i in range(it.N)])
    sz = np.array([it.A[1][i, ch[i]] for i in range(it.N)])
    res["read_gb_experts"] = float((pe[me] * sz[me]).sum() / 1e9)
    res["read_gb_ssd_experts"] = float((pe[me & ssd] * sz[me & ssd]).sum() / 1e9)
    return res


def prior_choice(it: Items, ram_cap: float, prior_k: dict) -> np.ndarray:
    """The conversion's own plan (primary K of every item = the 'where: shard' candidate, given in meta) with the RAM
    tier filled greedily by read probability. Used as the comparison point ("the guess")."""
    ch = np.zeros(it.N, dtype=np.int64)
    for i in range(it.N):
        k = prior_k.get(i)
        names = it.meta[i]["opts"]
        ch[i] = names.index(f"K{k}:ram") if k is not None and f"K{k}:ram" in names else 0
    me = [i for i in range(it.N) if it.kind[i] == "expert"]
    dense = [i for i in range(it.N) if it.kind[i] != "expert"]
    used = sum(it.A[0][i, ch[i]] for i in dense)
    for i in sorted(me, key=lambda i: -it.meta[i]["p"]):
        ram_opt = ch[i]
        nm = it.meta[i]["opts"]
        ssd_opt = ram_opt + 1 if nm[ram_opt].endswith(":ram") and ram_opt + 1 < len(nm) \
            and nm[ram_opt + 1].endswith(":ssd") else ram_opt
        sz = it.A[0][i, ram_opt]
        if used + sz <= ram_cap:
            used += sz
        else:
            ch[i] = ssd_opt
    return ch


def primary_ks(tabs: dict, it: Items) -> dict:
    prim = {}
    pos = {}
    for unit, t in tabs.items():
        for e, r in (t.get("experts") or {}).items():
            ks = [int(k) for k, v in r["k"].items() if v.get("where") == "shard"]
            pos[f"{unit}.e{e}"] = ks[0] if ks else None
        for key, row in (t.get("linears") or {}).items():
            pass
    for i, lab in enumerate(it.label):
        if lab in pos:
            prim[i] = pos[lab]
        elif "keys" in it.meta[i]:
            u = it.meta[i]["unit"]
            row = tabs[u]["linears"][it.meta[i]["keys"][0]]
            ks = [int(k) for k, v in row["k"].items() if v.get("where") == "shard"]
            prim[i] = ks[0] if ks else None
    return prim


def fixed_from_plan(plan_path, prm) -> dict:
    """Bytes the optimizer does not choose: head, MTP, copied tensors (norms, router, hc, ...), engram cache, runtime."""
    disk = ram = read = 0.0
    if plan_path and os.path.exists(plan_path):
        plan = json.load(open(plan_path))
        for e in plan["entries"]:
            if e["action"] == "quant" and e["kind"] in ("head", "mtp") or e.get("tier") == "mtp":
                disk += e["out_bytes"]
                ram += e["out_bytes"]
                if e["kind"] == "head":
                    read += e["out_bytes"]
            elif e["action"] == "copy":
                disk += e["out_bytes"]
                ram += e["out_bytes"]
        read += 0.1e9            # misc per-token reads (norms, router gates, hc, indexer small tensors)
    else:
        disk = ram = 3.0e9
        read = 0.4e9
    ram_gib = ram / GiB + prm["runtime_gib"] + prm["engram_cache_gib"]
    return {"disk_gb": disk / 1e9, "ram_gib": ram_gib, "read_gb": read / 1e9,
            "time_ms": read / prm["bw_ram"] * 1e3, "prm": prm}


def skew_report(out_dir: str, tops=(32, 64, 96, 128, 160, 224)) -> str:
    """Routing skew of the layers captured so far (stats/<unit>.routing.json): share of all expert picks held by the hottest
    N experts of a layer, and the number of experts never routed. Readable after the first layers of the conversion."""
    d = os.path.join(out_dir, "stats")
    rows = []
    for fn in sorted(os.listdir(d), key=lambda f: (len(f), f)):
        if fn.startswith("layers.") and fn.endswith(".routing.json"):
            st = json.load(open(os.path.join(d, fn)))
            c = sorted(st["counts"], reverse=True)
            tot = float(sum(c)) or 1.0
            rows.append((fn.split(".routing")[0], [sum(c[:n]) / tot for n in tops], sum(1 for x in c if x == 0)))
    if not rows:
        return "no routing statistics yet"
    out = ["layer       " + " ".join(f"top{n:<4d}" for n in tops) + "  never-routed"]
    for u, m, z in rows:
        out.append(f"{u:11s} " + " ".join(f"{x * 100:6.1f}%" for x in m) + f"  {z}")
    mean = [sum(r[1][i] for r in rows) / len(rows) for i in range(len(tops))]
    out.append("mean        " + " ".join(f"{x * 100:6.1f}%" for x in mean))
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ outputs
def write_alloc(it: Items, ch: np.ndarray, tabs: dict, res: dict, prm: dict, path: str):
    expert_bits, resident, module_bits = {}, {}, {}
    for i in range(it.N):
        m = it.meta[i]
        name = m["opts"][ch[i]]
        k = int(name[1:].split(":")[0])
        if it.kind[i] == "expert":
            key = f"{m['unit']}.ffn"
            n = max(int(e) for e in tabs[m["unit"]]["experts"]) + 1
            expert_bits.setdefault(key, [None] * n)[m["e"]] = k
            if name.endswith(":ram"):
                resident.setdefault(key, []).append(m["e"])
        else:
            for key in m["keys"]:
                module_bits[key] = k
    out = {"expert_bits": expert_bits, "resident_experts": resident, "module_bits": module_bits, "summary": res,
           "params": {k: v for k, v in prm.items() if k != "layer_weights"}}
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, path)
    return out


def make_params(a) -> dict:
    lw = json.load(open(a.layer_weights)) if a.layer_weights else {}
    return {"nu": a.nu, "w_attn": a.w_attn, "w_shared": a.w_shared, "layer_weights": lw,
            "ks": [int(x) for x in a.ks.split(",")] if a.ks else [],
            "attn_ks": [int(x) for x in a.attn_ks.split(",")] if a.attn_ks else [],
            "bw_ram": a.bw_ram * 1e9, "bw_ssd": a.bw_ssd * 1e9, "ssd_hit": a.ssd_hit, "rows": a.rows,
            "smooth": a.smooth, "ram_gib": a.ram_gib, "runtime_gib": a.runtime_gib,
            "engram_cache_gib": a.engram_cache_gib, "resident_only": a.resident_only}


def optimize(tabs, prm, fixed, min_toks=None, max_disk_gb=None):
    it = build_items(tabs, prm)
    ram_cap = prm["ram_gib"] * GiB - fixed["ram_gib"] * GiB
    caps = np.array([ram_cap, INF if max_disk_gb is None else max_disk_gb * 1e9 - fixed["disk_gb"] * 1e9,
                     INF if min_toks is None else 1.0 / min_toks - fixed["time_ms"] * 1e-3])
    ch, ok = solve(it, caps)
    it.feasible = ok
    if not ok:
        # the caps cannot all be met: return the fastest allocation that still respects RAM / disk, flagged infeasible
        it2 = build_items(tabs, prm)
        it2.D_ = np.where(np.isfinite(it2.D_), it2.A[2] * 1e3 + 1e-9 * it2.D_, INF)
        ch2, _ = solve(it2, np.array([caps[0], caps[1], INF]))
        print("WARNING: no feasible allocation for these caps; returning the fastest one under the RAM / disk caps")
        ch = ch2
    return it, ch, caps


def fastest(tabs, prm, fixed, max_disk_gb=None):
    """The highest modeled decode ceiling reachable with the measured candidates (quality ignored): minimize the time under
    the RAM (and disk) caps. Tells which --min-toks values are feasible at all."""
    it = build_items(tabs, prm)
    it.D_ = np.where(np.isfinite(it.D_), it.A[2] * 1e3 + 1e-9 * it.D_, INF)     # objective = ms per token
    caps = np.array([prm["ram_gib"] * GiB - fixed["ram_gib"] * GiB,
                     INF if max_disk_gb is None else max_disk_gb * 1e9 - fixed["disk_gb"] * 1e9, INF])
    ch, ok = solve(it, caps)
    t_ms = float(it.A[2][np.arange(it.N), ch].sum() * 1e3) + fixed["time_ms"]
    return 1e3 / t_ms, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", required=True, help="conversion output dir (contains alloc/)")
    ap.add_argument("--plan", help="plan.json (fixed bytes: head, MTP, copied tensors)")
    ap.add_argument("--out", help="alloc.json to write")
    ap.add_argument("--ram-gib", type=float, default=118.0, help="usable RAM")
    ap.add_argument("--runtime-gib", type=float, default=5.0, help="KV rings + pools + arenas + context")
    ap.add_argument("--engram-cache-gib", type=float, default=2.0, help="engram row cache (the tables live on the NVMe)")
    ap.add_argument("--bw-ram", type=float, default=200.0, help="effective RAM read GB/s (PERF_MODEL: ~200)")
    ap.add_argument("--bw-ssd", type=float, default=5.0, help="NVMe GB/s for streamed experts (3 measured cold, 6.5 spec)")
    ap.add_argument("--ssd-hit", type=float, default=0.0, help="fraction of NVMe reads a dynamic cache would serve")
    ap.add_argument("--rows", type=int, default=1, help="tokens per step (1 plain decode, ~5 with MTP verify)")
    ap.add_argument("--min-toks", type=float, help="constraint: modeled bandwidth ceiling in tok/s")
    ap.add_argument("--max-disk-gb", type=float, help="constraint: total pack bytes (without the engram tables)")
    ap.add_argument("--nu", type=float, default=0.5, help="weight of the gate/up relative errors vs down")
    ap.add_argument("--w-attn", type=float, default=1.0)
    ap.add_argument("--w-shared", type=float, default=1.0)
    ap.add_argument("--layer-weights", help="JSON {layer: s_l, default: 1.0}")
    ap.add_argument("--ks", default="", help="restrict routed-expert K, e.g. 2,3")
    ap.add_argument("--attn-ks", default="", help="restrict attention/shared K, e.g. 3,4,5")
    ap.add_argument("--smooth", type=float, default=0.05, help="routing-count prior, fraction of the layer mean count")
    ap.add_argument("--resident-only", action="store_true",
                    help="RAM-fit mode: every expert stays in RAM (no NVMe option); the experts' K is whatever the "
                         "conversion chose (ramfit), the optimizer then only picks attention / shared K")
    ap.add_argument("--skew", action="store_true", help="only print the routing skew of the captured layers and exit")
    ap.add_argument("--sweep", help="comma list of tok/s ceilings: print the quality/size/speed frontier")
    a = ap.parse_args(argv)
    if a.skew:
        print(skew_report(a.tables))
        return None, None
    prm = make_params(a)
    tabs = load_tables(a.tables)
    fixed = fixed_from_plan(a.plan, prm)
    it0 = build_items(tabs, prm)
    ram_cap = prm["ram_gib"] * GiB - fixed["ram_gib"] * GiB
    base = describe(it0, prior_choice(it0, ram_cap, primary_ks(tabs, it0)), fixed)
    print(f"{it0.N} items from {len(tabs)} layers; fixed: RAM {fixed['ram_gib']:.1f} GiB, disk {fixed['disk_gb']:.1f} GB, "
          f"read {fixed['read_gb']:.2f} GB/token")
    show("plan (primary K, hot-first residency)", base)
    fs, fok = fastest(tabs, prm, fixed, a.max_disk_gb)
    print(f"\nfastest reachable with these candidates and RAM: {fs:.1f} tok/s" + ("" if fok else "  (RAM cap infeasible!)"))
    rows = []
    for tk in ([float(x) for x in a.sweep.split(",")] if a.sweep else []) + [a.min_toks]:
        it, ch, caps = optimize(tabs, prm, fixed, tk, a.max_disk_gb)
        res = describe(it, ch, fixed)
        res["feasible"] = bool(getattr(it, "feasible", True))
        rows.append((tk, res))
        show((f"optimized, ceiling >= {tk} tok/s" if tk else "optimized (RAM only)")
             + ("" if res["feasible"] else "  ** INFEASIBLE: fastest reachable shown **"), res)
        if tk == a.min_toks and a.out:
            write_alloc(it, ch, tabs, res, prm, a.out)
            print("wrote", a.out)
    return rows, base


def show(title, r):
    print(f"\n[{title}]\n  D (relative error proxy) {r['D']:.5f}  (experts {r['D_expert']:.5f}, attn {r['D_attn']:.5f}, "
          f"shared {r['D_shared']:.5f})\n  experts by K {r['n_experts_k']}, resident {r['n_experts_resident']}, "
          f"streamed {r['n_experts_ssd']}\n  pack {r['disk_gb_total']:.1f} GB, RAM {r['ram_gib_total']:.1f} GiB, "
          f"read {r['read_gb_dense'] + r['read_gb_experts']:.2f} GB/token (dense {r['read_gb_dense']:.2f}, experts "
          f"{r['read_gb_experts']:.2f}, of which NVMe {r['read_gb_ssd_experts']:.3f})\n  modeled ceiling "
          f"{r['ceiling_toks']:.1f} tok/s ({r['time_ms']:.1f} ms/token)")


if __name__ == "__main__":
    main()
