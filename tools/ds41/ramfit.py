"""RAM-fit planner for the DS4.1 EXL3 pack: the WHOLE routed-expert set lives in RAM (no NVMe on the decode path).

    python tools/ds41/ramfit.py budget [--map tensor_map.json] [--attn-bits 4 ...]     # RAM table, bytes/token, tok/s
    python tools/ds41/ramfit.py curves [--scenario ref] [--out curves.json]            # quality proxy vs RAM, 3 schemes
    python tools/ds41/ramfit.py plan   [--out ramfit.json]                             # recommended plan + conversion flags

Two layers of code:

* pure functions (no torch, numpy only) used by the converter at queue time: `decide_layer` chooses K in {0 (pruned), 1, 2,
  3, ...} for every routed expert of ONE layer from the capture's routing statistics, so that the layer's experts fill a
  byte budget with the least proxy error. One Lagrangian solve per layer (a few ms), no extra quantizations, no candidates:
  conversion time = the time of the chosen Ks.
* the offline model (budget, synthetic routing, curves) used to pick the byte budget and the K set.

Quality proxy (same family as optimize_alloc.py, NOT a KLD): per layer D_l = sum_e share_e * 2 * eps(K_e) with share_e =
the expert's share of the routed-MoE output energy (routing mass x output energy) and eps(K) the Hessian-weighted relative
error of one linear (K=2 0.0489 and K=3 0.0120 measured by the real quantizer on synthetic Hessians, x4.07 per bit; K=1
from the CPU trellis measurement scratch/k1_trellis.py). Factor 2 = eps_w2 + 0.5 (eps_w1 + eps_w3). A pruned expert costs
share_e * prune_cost (its whole contribution is lost, plus the substitute the router picks instead: prune_cost = 1 + beta).
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GiB = 1024 ** 3
IN, MID = 5120, 2304
N_EXPERTS, N_LAYERS, TOPK = 384, 40, 6

# relative proxy error of ONE linear at K bits. K=2 / K=3 measured (real quantizer, synthetic SPD Hessian);
# K=1 = K=2 * (CPU trellis ratio, scratch/k1_trellis.log); K=4 extrapolated at x4.07 per bit.
EPS = {1: 0.0489 * 4.00, 2: 0.0489, 3: 0.0120, 4: 0.0120 / 4.07, 5: 0.0120 / 4.07 ** 2, 6: 0.0120 / 4.07 ** 3}
NU = 0.5


def exl3_bytes(in_f: int, out_f: int, k: int) -> int:
    return (in_f // 16) * (out_f // 16) * 16 * k * 2 + in_f * 2 + out_f * 2 + 4


def expert_bytes(k: int, dims=(IN, MID)) -> int:
    """One routed expert (w1, w3: hidden -> inter, w2: inter -> hidden; DS4.1 5120 / 2304) at K bits; K = 0 is a pruned
    expert (nothing stored)."""
    if k == 0:
        return 0
    h, m = dims
    return 2 * exl3_bytes(h, m, k) + exl3_bytes(m, h, k)


def expert_err(k: int, prune_cost: float = 1.3, eps=None) -> float:
    """Relative output error of one expert at K (0 = pruned): eps_w2 + NU (eps_w1 + eps_w3) with equal per-linear errors."""
    if k == 0:
        return prune_cost
    return (eps or EPS)[k] * (1.0 + 2.0 * NU)


# ------------------------------------------------------------------------------------------------ per-layer decision
def decide_layer(share, budget_bytes: float, ks=(1, 2, 3), prune_cost: float = 1.3, allow_prune: bool = False,
                 eps=None, min_k=None, prunable=None, dims=(IN, MID)):
    """K per routed expert of one layer (dims = (hidden, inter) of the expert, for the byte sizes).

    share: per-expert weight of the routed-MoE output energy (routing count x gate^2 x output energy; any positive scale).
    Minimizes sum_e share_e * err(K_e) subject to sum_e bytes(K_e) <= budget_bytes. Lagrangian: each expert picks
    argmin_K share_e err(K) + lam bytes(K); lam is bisected until the layer fits, then the slack is spent by upgrading the
    experts with the best error reduction per byte. Deterministic (ties by id).
    allow_prune / prunable: K = 0 is an option (the expert is dropped, router bias -> -inf at pack time); `prunable` is a
    boolean mask of the experts allowed to be dropped (e.g. routed to by < 0.1 % of the calibration tokens), None with
    allow_prune = every expert.
    min_k: experts never go below this K unless pruned.
    Returns (list of K per expert, bytes used, D = sum share err / sum share)."""
    share = np.asarray(share, dtype=np.float64)
    n = len(share)
    tot = float(share.sum())
    sh = share / tot if tot > 0 else np.full(n, 1.0 / n)
    do_prune = allow_prune or prunable is not None
    can0 = np.ones(n, dtype=bool) if prunable is None else np.asarray(prunable, dtype=bool)
    opts = sorted(set(int(k) for k in ks) | ({0} if do_prune else set()))
    err = np.array([expert_err(k, prune_cost, eps) for k in opts])
    byt = np.array([expert_bytes(k, dims) for k in opts], dtype=np.float64)
    cost_d = sh[:, None] * err[None, :]                          # [n, O]
    if do_prune:
        cost_d[~can0, 0] = 1e30                                  # option 0 is K = 0
    idx = np.arange(n)
    kmin_b = float(byt[1 if do_prune else 0])
    floor_bytes = kmin_b * float((~can0).sum()) if do_prune else kmin_b * n
    if budget_bytes < floor_bytes - 1e-6:
        raise ValueError(f"budget {budget_bytes / 1e6:.1f} MB below the floor {floor_bytes / 1e6:.1f} MB")

    def pick(lam):
        return (cost_d + lam * byt[None, :]).argmin(axis=1)

    def used(ch):
        return float(byt[ch].sum())

    ch = pick(0.0)
    if used(ch) > budget_bytes:
        lo, hi = 0.0, 1.0
        while used(pick(hi)) > budget_bytes:
            lo, hi = hi, hi * 4.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if used(pick(mid)) > budget_bytes:
                lo = mid
            else:
                hi = mid
        ch = pick(hi)
    # spend the slack: best error reduction per byte, one expert at a time
    slack = budget_bytes - used(ch)
    for _ in range(4 * n):
        cur_d = cost_d[idx, ch]
        gain = cur_d[:, None] - cost_d                           # error reduction of every move
        dbytes = byt[None, :] - byt[ch][:, None]
        ok = (gain > 1e-18) & (dbytes > 0) & (dbytes <= slack + 1e-9)
        if not ok.any():
            break
        score = np.where(ok, gain / np.maximum(dbytes, 1.0), -1.0)
        i, o = np.unravel_index(np.argmax(score), score.shape)
        slack -= byt[o] - byt[ch[i]]
        ch[i] = o
    kk = [opts[o] for o in ch]
    if min_k:
        kk = [k if k == 0 else max(k, min_k) for k in kk]
    byt_used = float(sum(expert_bytes(k, dims) for k in kk))
    d = float(sum(sh[i] * expert_err(k, prune_cost, eps) for i, k in enumerate(kk)))
    return kk, byt_used, d


def layer_budgets(total_bytes: float, n_layers: int = N_LAYERS, weights=None) -> list:
    """Per-layer expert byte budgets: uniform, or proportional to `weights` (e.g. per-layer sensitivity)."""
    w = np.ones(n_layers) if weights is None else np.asarray(weights, dtype=np.float64)
    return list(total_bytes * w / w.sum())


# ------------------------------------------------------------------------------------------------ RAM budget
def dense_budget(tmap: dict, modules: dict, ctx: int = 131072, ram_gib: float = 118.0, runtime_gib: float = 5.0,
                 engram_cache_gib: float = 0.5, text_only: bool = False, embed_resident: bool = True) -> dict:
    """Everything that is not a backbone routed expert: RAM resident bytes, bytes read per decode token, and the budget
    left for the experts. modules: attn / shared / head / mtp / mtp_exp bits (+ role_bits)."""
    import plan as P
    bits = {"attn": modules.get("attn", 4), "shared": modules.get("shared", 4), "head": modules.get("head", 6),
            "mtp": modules.get("mtp", 4), "mtp_exp": modules.get("mtp_exp", 2)}
    lb = {l: 2 for l in range(N_LAYERS)}
    pl = P.build_plan(tmap, 2.0, bits=bits, layer_bits=lb, skip_engram=True, mtp_queue=True,
                      role_bits=modules.get("role_bits"))
    g = {}
    for e in pl["entries"]:
        if e["action"] == "quant":
            if e["kind"] == "routed" and e.get("tier") != "mtp":
                continue
            kd = "mtp_experts" if (e["kind"] == "routed") else e["kind"]
            if e["kind"] == "mtp" or kd == "mtp_experts":
                kd = "mtp_experts" if e["kind"] == "routed" else "mtp_dense"
            g[kd] = g.get(kd, 0) + e["out_bytes"]
        else:
            nm = e["name"]
            if nm.startswith("vision.") or nm.startswith("aligner."):
                kd = "vision"
            elif nm == "embed.weight":
                kd = "embed"
            elif ".ffn.gate." in nm:
                kd = "router"
            elif ".hc_" in nm:
                kd = "hc"
            elif nm.startswith("mtp."):
                kd = "mtp_copy"
            else:
                kd = "other_copy"
            g[kd] = g.get(kd, 0) + e["out_bytes"]
    T = tmap["tensors"]
    g["engram_wkv"] = sum(v["length"] for n, v in T.items() if re.match(r"^layers\.\d+\.engram\.(wkv|q_weight|k_weight)", n))
    kv = kv_bytes(ctx)
    resident = dict(g)
    if text_only:
        resident.pop("vision", None)
    if not embed_resident:
        resident.pop("embed", None)
    # bytes read from RAM every decoded token (everything that is not an expert, plus the one embedding row)
    read = {k: g.get(k, 0) for k in ("attn", "shared", "head", "router", "hc", "engram_wkv", "other_copy")}
    read["kv"] = kv_read_bytes(ctx)
    fixed = sum(resident.values()) + kv + runtime_gib * GiB + engram_cache_gib * GiB
    return {"groups_bytes": g, "resident_bytes": sum(resident.values()), "kv_bytes": kv,
            "runtime_gib": runtime_gib, "engram_cache_gib": engram_cache_gib,
            "fixed_gib": fixed / GiB, "ram_gib": ram_gib, "experts_gib": ram_gib - fixed / GiB,
            "experts_bytes": ram_gib * GiB - fixed, "read_bytes": read, "read_total": sum(read.values()),
            "modules": bits, "ctx": ctx, "text_only": text_only}


def kv_bytes(ctx: int) -> int:
    """DS4.1 KV state in fp16 (fake-quantized values stored as fp16): sliding window rings of 128 x 512 in 40 layers, the 4
    shared latent pools (layers 2, 8, 14 ratio 2; layer 20 ratio 1), the 8 indexer key planes (5 ratio 1 + 3 ratio 2, 128
    wide). ctx = 128K -> 0.56 GB; 1M -> 4.5 GB."""
    ring = 40 * 128 * 512 * 2
    pools = (3 * (ctx // 2) + ctx) * 512 * 2
    idx = (5 * ctx + 3 * (ctx // 2)) * 128 * 2
    return ring + pools + idx


def kv_read_bytes(ctx: int) -> int:
    """Per decoded token: window + 640 gathered entries per layer (31.5 MB) + the indexer key scan."""
    return 31_500_000 + (5 * ctx + 3 * (ctx // 2)) * 128 * 2 // 2     # scan of half the keys on average (candidate-limited)


# ------------------------------------------------------------------------------------------------ synthetic routing
def zipf_weights(rng, s: float, n_dead: int = 0, dead_pi: float = 3e-5, n: int = N_EXPERTS):
    """Per-token inclusion probability pi_e (sum = TOPK) of a Zipf(s) router over a random permutation of the experts,
    plus `n_dead` experts that are almost never routed. Top-k without replacement: pi from the Gumbel top-6 simulation
    would be exact, here the standard conditional approximation pi = 1 - (1 - p)^TOPK corrected to sum to TOPK."""
    live = n - n_dead
    p = 1.0 / np.arange(1, live + 1) ** s
    p /= p.sum()
    pi = 1.0 - (1.0 - p) ** TOPK
    for _ in range(40):                      # rescale so that sum(pi_live) = TOPK - dead mass
        tgt = TOPK - n_dead * dead_pi
        c = pi.sum()
        pi = np.minimum(0.999, pi * tgt / c)
    pi_all = np.concatenate([pi, np.full(n_dead, dead_pi) * np.exp(rng.normal(0, 0.5, n_dead))])
    perm = rng.permutation(n)
    out = np.empty(n)
    out[perm] = pi_all
    return out


def simulate_counts(rng, pi, tokens: int = 4096):
    """Per-expert routed-token counts of `tokens` tokens: each token routes to TOPK distinct experts (Gumbel top-k with
    weights ~ pi), the way the reference run's `experts_hit` / `max_expert_tokens` were measured."""
    g = rng.gumbel(size=(tokens, len(pi))) + np.log(np.maximum(pi, 1e-12))[None, :]
    top = np.argpartition(-g, TOPK, axis=1)[:, :TOPK]
    return np.bincount(top.ravel(), minlength=len(pi))


# real DS4.1 reference run (ds41ref scratch/refcpu/progress.jsonl, 2 rows x 2048 tokens = 4096 tokens, CPU track):
# layer: (experts hit of 384, max tokens on one expert). Layers 0-2 use hash routing.
REF_RUN = {0: (366, 693), 1: (357, 991), 2: (345, 2144), 3: (344, 2618), 4: (359, 1503)}


def fit_reference(seed: int = 0) -> dict:
    """Zipf exponent and dead-expert count that reproduce the two learned-router layers of the reference run (3 and 4)
    on 4096 tokens: (hit, max). Grid search, 2 simulations per cell."""
    rng = np.random.default_rng(seed)
    best = {}
    for layer in (3, 4):
        hit, mx = REF_RUN[layer]
        cand = None
        for s in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
            for nd in (0, 10, 20, 30, 40, 50, 60):
                hs, ms = [], []
                for _ in range(2):
                    c = simulate_counts(rng, zipf_weights(rng, s, nd), 4096)
                    hs.append((c > 0).sum())
                    ms.append(c.max())
                e = ((np.mean(hs) - hit) / 10.0) ** 2 + ((np.mean(ms) - mx) / max(mx, 200) * 3.0) ** 2
                if cand is None or e < cand[0]:
                    cand = (e, s, nd, float(np.mean(hs)), float(np.mean(ms)))
        best[layer] = {"s": cand[1], "dead": cand[2], "sim_hit": cand[3], "sim_max": cand[4], "ref_hit": hit,
                       "ref_max": mx}
    return best


SCENARIOS = {
    # name: (zipf s range over layers, dead experts range, description)
    "ref": ((0.8, 1.1), (30, 50), "fitted to the real reference run (layers 3-4, fit_reference): Zipf 0.8-1.1 + 30-50 near-dead experts"),
    "flat": ((0.5, 0.7), (0, 0), "pessimistic: flat routing, no dead experts"),
    "steep": ((1.0, 1.3), (30, 60), "concentrated router (DS4 Pro-like reachable set)"),
}


def make_layers(scenario: str, seed: int = 0, n_layers: int = N_LAYERS):
    """share[l, e] (sum 1 per layer) and pi[l, e] for a scenario. share = routing mass x gate^2 x output energy:
    pi_e times a lognormal factor (energy spread across experts, sigma 0.3)."""
    (s0, s1), (d0, d1), _ = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    pis, shares = [], []
    for l in range(n_layers):
        s = rng.uniform(s0, s1)
        nd = int(rng.integers(d0, d1 + 1)) if d1 > 0 else 0
        pi = zipf_weights(rng, s, nd)
        sh = pi * np.exp(rng.normal(0, 0.3, len(pi)))
        pis.append(pi)
        shares.append(sh / sh.sum())
    return np.array(shares), np.array(pis)


# ------------------------------------------------------------------------------------------------ schemes and curves
def solve_scheme(shares, budget_bytes: float, ks, allow_prune: bool, prune_cost: float, layer_w=None, glob: bool = False,
                 eps=None, prunable=None):
    """Per-layer byte budgets (uniform) -> K per expert. glob=True: ONE multiplier for the whole network (the ideal
    allocation, for the cost of per-layer budgets). Returns (K [L, E], bytes, D mean over layers)."""
    L, E = shares.shape
    if not glob:
        bl = layer_budgets(budget_bytes, L, layer_w)
        Ks, used, D = [], 0.0, 0.0
        for l in range(L):
            kk, b, d = decide_layer(shares[l], bl[l], ks, prune_cost, allow_prune, eps,
                                    prunable=None if prunable is None else prunable[l])
            Ks.append(kk)
            used += b
            D += d
        return np.array(Ks), used, D / L
    opts = sorted(set(ks) | ({0} if allow_prune else set()))
    err = np.array([expert_err(k, prune_cost, eps) for k in opts])
    byt = np.array([expert_bytes(k) for k in opts], dtype=np.float64)
    # shares are per layer normalized: D_l weights equal, so the global objective is sum over layers of sum share err
    cost = shares[:, :, None] * err[None, None, :]
    if prunable is not None and allow_prune:
        cost[:, :, 0] = np.where(prunable, cost[:, :, 0], 1e30)

    def pick(lam):
        return (cost + lam * byt[None, None, :]).argmin(axis=2)

    lo, hi = 0.0, 1.0
    used = lambda ch: float(byt[ch].sum())
    if byt.min() * shares.size > budget_bytes:
        raise ValueError("budget below the floor")
    while used(pick(hi)) > budget_bytes:
        lo, hi = hi, hi * 4
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if used(pick(mid)) > budget_bytes:
            lo = mid
        else:
            hi = mid
    ch = pick(hi)
    K = np.vectorize(lambda o: opts[o])(ch)
    D = float((shares * np.vectorize(lambda k: expert_err(k, prune_cost, eps))(K)).sum(axis=1).mean())
    return K, used(ch), D


def read_bytes_per_token(K, pi, prune: bool = True) -> float:
    """Expected routed-expert bytes read per decoded token (6 experts per layer per token). Pruned experts: their routing
    mass moves to the remaining experts proportionally (the router re-selects among the rest)."""
    tot = 0.0
    for l in range(K.shape[0]):
        p = pi[l].copy()
        kept = K[l] > 0
        if not kept.all():
            p = np.where(kept, p, 0.0)
            p = p * TOPK / p.sum()
        tot += float(sum(p[e] * expert_bytes(int(K[l, e])) for e in range(K.shape[1])))
    return tot


def curve(shares, pi, budgets_gib, prune_cost, scheme: str, uniform_k: int = 2, glob: bool = False, ks_mixed=(1, 2, 3),
          tau: float = 1e-4):
    """scheme: 'mixedk' (K in ks_mixed, no pruning), 'prune' (K = uniform_k or pruned), 'both' (K in ks_mixed or pruned)."""
    rows = []
    pm = pi < tau            # rarely routed: fewer than tau of the tokens pick the expert (None = any expert)
    free = np.ones_like(pi, dtype=bool)
    for b in budgets_gib:
        bb = b * GiB
        try:
            if scheme == "mixedk":
                K, used, D = solve_scheme(shares, bb, ks_mixed, False, prune_cost, glob=glob)
            elif scheme == "prune":
                K, used, D = solve_scheme(shares, bb, (uniform_k,), True, prune_cost, glob=glob, prunable=free)
            elif scheme == "both":
                K, used, D = solve_scheme(shares, bb, ks_mixed, True, prune_cost, glob=glob, prunable=free)
            else:       # both_safe: only rarely routed experts may be pruned
                K, used, D = solve_scheme(shares, bb, ks_mixed, True, prune_cost, glob=glob, prunable=pm)
        except ValueError:
            rows.append({"gib": b, "feasible": False})
            continue
        n = K.size
        hist = {int(k): int((K == k).sum()) for k in sorted(set(K.ravel().tolist()))}
        rows.append({"gib": float(b), "feasible": True, "D": float(D), "used_gib": float(used / GiB), "hist": hist,
                     "bpw": float(sum(expert_bytes(int(k)) * c for k, c in hist.items()) * 8 / (n * 3 * IN * MID)),
                     "read_gb": float(read_bytes_per_token(K, pi) / 1e9), "pruned_frac": hist.get(0, 0) / n})
    return rows


# ------------------------------------------------------------------------------------------------ reports
def _load_map(path):
    return json.load(open(path))


def cmd_budget(a):
    tm = _load_map(a.map)
    mods = _modules(a)
    b = dense_budget(tm, mods, a.ctx, a.ram_gib, a.runtime_gib, a.engram_cache_gib, a.text_only, not a.embed_nvme)
    print(f"RAM budget {a.ram_gib:.0f} GiB usable, ctx {a.ctx}, modules {b['modules']}"
          + (f" role_bits {mods['role_bits']}" if mods.get("role_bits") else ""))
    for k, v in sorted(b["groups_bytes"].items(), key=lambda kv: -kv[1]):
        skip = (k == "vision" and a.text_only) or (k == "embed" and a.embed_nvme)
        print(f"  {k:12s} {v / GiB:7.2f} GiB" + ("   (not resident)" if skip else ""))
    print(f"  {'kv':12s} {b['kv_bytes'] / GiB:7.2f} GiB   (ctx {a.ctx})\n  {'runtime':12s} {a.runtime_gib:7.2f} GiB\n"
          f"  {'engram buf':12s} {a.engram_cache_gib:7.2f} GiB   (tables on NVMe)")
    print(f"  fixed total  {b['fixed_gib']:7.2f} GiB  ->  routed experts get {b['experts_gib']:.2f} GiB "
          f"= {b['experts_bytes'] / 1e9:.1f} GB  ({b['experts_bytes'] * 8 / (N_LAYERS * N_EXPERTS * 3 * IN * MID):.3f} bpw "
          f"avg over 15360 experts)")
    print("decode bytes/token outside the experts: " + ", ".join(f"{k} {v / 1e9:.2f}" for k, v in b["read_bytes"].items())
          + f"  = {b['read_total'] / 1e9:.2f} GB")
    return b


def _modules(a):
    rb = {}
    if getattr(a, "role_bits", None):
        for part in a.role_bits.split(","):
            k, _, v = part.partition("=")
            rb[k.strip()] = int(v)
    return {"attn": a.attn_bits, "shared": a.shared_bits, "head": a.head_bits, "mtp": a.mtp_bits,
            "mtp_exp": a.mtp_exp_bits, **({"role_bits": rb} if rb else {})}


def cmd_curves(a):
    tm = _load_map(a.map)
    b = dense_budget(tm, _modules(a), a.ctx, a.ram_gib, a.runtime_gib, a.engram_cache_gib, a.text_only, not a.embed_nvme)
    eb = b["experts_gib"]
    shares, pi = make_layers(a.scenario, a.seed)
    grid = sorted(set([round(x, 1) for x in np.arange(40, 131, 10)] + [round(eb, 1)]))
    pc = 1.0 + a.beta
    out = {"scenario": a.scenario, "desc": SCENARIOS[a.scenario][2], "expert_budget_gib": eb, "prune_cost": pc,
           "eps": {str(k): v for k, v in EPS.items()}, "curves": {}}
    all_k2 = 15360 * expert_bytes(2) / GiB
    print(f"scenario {a.scenario}: {SCENARIOS[a.scenario][2]}\nexpert budget {eb:.1f} GiB (all K=2 = {all_k2:.1f}, "
          f"all K=3 = {15360 * expert_bytes(3) / GiB:.1f}, all K=1 = {15360 * expert_bytes(1) / GiB:.1f}), prune_cost {pc}")
    schemes = (("mixedk", "mixed K only"), ("prune", f"prune only (K={a.uniform_k})"), ("both", "both"),
               ("both_safe", f"both, prune pi<{a.tau:g}"))
    for scheme, _ in schemes:
        out["curves"][scheme] = curve(shares, pi, grid, pc, scheme, a.uniform_k, a.glob, tau=a.tau)
    print("GiB experts | " + " | ".join(f"{t:>31s}" for _, t in schemes))
    for i, g in enumerate(grid):
        cells = []
        for scheme, _ in schemes:
            r = out["curves"][scheme][i]
            if not r["feasible"]:
                cells.append(f"{'infeasible':>31s}")
            else:
                cells.append(f"D{r['D'] * 100:6.2f}% {r['bpw']:.2f}bpw pr{r['pruned_frac'] * 100:4.1f}% {r['read_gb']:4.2f}GB")
        mark = " <== fit" if abs(g - round(eb, 1)) < 1e-9 else ""
        print(f"{g:11.1f} | " + " | ".join(f"{c:>31s}" for c in cells) + mark)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return out


def cmd_plan(a):
    tm = _load_map(a.map)
    mods = _modules(a)
    b = dense_budget(tm, mods, a.ctx, a.ram_gib, a.runtime_gib, a.engram_cache_gib, a.text_only, not a.embed_nvme)
    shares, pi = make_layers(a.scenario, a.seed)
    pc = 1.0 + a.beta
    ks = tuple(int(x) for x in a.ks.split(","))
    eb = b["experts_bytes"] * (1.0 - a.margin)
    pm = (pi < a.tau) if a.prune else None
    K, used, D = solve_scheme(shares, eb, ks, a.prune, pc, glob=False, prunable=pm)
    Kg, usedg, Dg = solve_scheme(shares, eb, ks, a.prune, pc, glob=True, prunable=pm)
    hist = {int(k): int((K == k).sum()) for k in sorted(set(K.ravel().tolist()))}
    n = K.size
    rd = read_bytes_per_token(K, pi)
    dense_read = b["read_total"]
    total_read = rd + dense_read
    plan = {"version": 1, "scenario": a.scenario, "ctx": a.ctx, "ram_gib": a.ram_gib, "modules": b["modules"],
            "experts": {"ks": list(ks), "prune": bool(a.prune), "prune_tau": a.tau, "prune_cost": pc,
                        "bytes_per_expert": eb / (N_LAYERS * N_EXPERTS), "layer_bytes": eb / N_LAYERS,
                        "margin": a.margin, "layer_weights": None},
            "predicted": {"expert_gib": used / GiB, "fixed_gib": b["fixed_gib"], "total_gib": used / GiB + b["fixed_gib"],
                          "hist_k": hist, "routed_bpw": float(sum(expert_bytes(k) * c for k, c in hist.items()) * 8
                                                              / (n * 3 * IN * MID)),
                          "D_expert_pct": D * 100, "D_expert_global_pct": Dg * 100,
                          "decode_gb_per_token": total_read / 1e9, "routed_read_gb": rd / 1e9,
                          "dense_read_gb": dense_read / 1e9,
                          "ceiling_tok_s": {str(bw): bw * 1e9 / total_read for bw in (190, 200, 230)}}}
    p = plan["predicted"]
    print(f"recommended plan ({a.scenario}): experts K hist {hist}, routed {p['routed_bpw']:.3f} bpw, "
          f"experts {p['expert_gib']:.1f} GiB + fixed {p['fixed_gib']:.1f} = {p['total_gib']:.1f} GiB of {a.ram_gib:.0f}\n"
          f"  D_expert {p['D_expert_pct']:.2f}% (ideal global multiplier {p['D_expert_global_pct']:.2f}%)\n"
          f"  decode {p['decode_gb_per_token']:.2f} GB/token (routed {p['routed_read_gb']:.2f} + other {p['dense_read_gb']:.2f}) "
          f"-> ceiling " + ", ".join(f"{bw} GB/s {v:.1f} tok/s" for bw, v in p["ceiling_tok_s"].items()))
    if a.out:
        json.dump(plan, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print("wrote", a.out)
    return plan


LEVER_SETS = [
    ("attn 5 / shared 5", {"attn": 5, "shared": 5}),
    ("attn 4 / shared 4 (plan default)", {"attn": 4, "shared": 4}),
    ("big attn 3 (q_b wo_a wo_b), small 5 (q_a kv)", {"attn": 4, "shared": 4,
                                                      "role_bits": {"q_b": 3, "wo_a": 3, "wo_b": 3, "q_a": 5, "kv": 5}}),
    ("big attn 3, small 4, indexer 4", {"attn": 4, "shared": 4,
                                        "role_bits": {"q_b": 3, "wo_a": 3, "wo_b": 3}}),
    ("attn 3 / shared 3", {"attn": 3, "shared": 3}),
    ("attn 3 / shared 4", {"attn": 3, "shared": 4}),
]


def attn_proxy(mods) -> float:
    """Sum over the 5 attention roles of the per-linear relative error at their K (per layer): errors of a chain add."""
    rb = mods.get("role_bits", {})
    return sum(EPS[rb.get(r, mods["attn"])] for r in ("q_a", "kv", "q_b", "wo_a", "wo_b"))


def cmd_levers(a):
    tm = _load_map(a.map)
    shares, pi = make_layers(a.scenario, a.seed)
    pc = 1.0 + a.beta
    print(f"scenario {a.scenario}, ctx {a.ctx}, text-only {a.text_only}, embed on NVMe {a.embed_nvme}, mtp experts K{a.mtp_exp_bits}, "
          f"head K{a.head_bits}\n{'attention / shared bits':46s} {'expGiB':>6s} {'bpw':>5s} {'D_exp%':>6s} {'D_attn%':>7s} "
          f"{'GB/tok':>6s} {'tok/s@190':>9s} {'@230':>6s}")
    rows = []
    for name, m in LEVER_SETS:
        mods = {"attn": m["attn"], "shared": m["shared"], "head": a.head_bits, "mtp": a.mtp_bits, "mtp_exp": a.mtp_exp_bits,
                **({"role_bits": m["role_bits"]} if "role_bits" in m else {})}
        b = dense_budget(tm, mods, a.ctx, a.ram_gib, a.runtime_gib, a.engram_cache_gib, a.text_only, not a.embed_nvme)
        K, used, D = solve_scheme(shares, b["experts_bytes"] * 0.99, (1, 2, 3), True, pc, prunable=pi < a.tau)
        rd = read_bytes_per_token(K, pi) + b["read_total"]
        bpw = used * 8 / (K.size * 3 * IN * MID)
        r = {"name": name, "experts_gib": b["experts_gib"], "bpw": bpw, "D_expert_pct": D * 100,
             "D_attn_pct": attn_proxy(mods) * 100, "gb_per_token": rd / 1e9,
             "toks_190": 190e9 / rd, "toks_230": 230e9 / rd}
        rows.append(r)
        print(f"{name:46s} {r['experts_gib']:6.1f} {bpw:5.2f} {r['D_expert_pct']:6.2f} {r['D_attn_pct']:7.2f} "
              f"{r['gb_per_token']:6.2f} {r['toks_190']:9.1f} {r['toks_230']:6.1f}")
    return rows


def cmd_ksets(a):
    tm = _load_map(a.map)
    b = dense_budget(tm, _modules(a), a.ctx, a.ram_gib, a.runtime_gib, a.engram_cache_gib, a.text_only, not a.embed_nvme)
    pc = 1.0 + a.beta
    eb = b["experts_bytes"] * 0.99
    print(f"expert budget {eb / GiB:.1f} GiB; per-layer budgets vs one network-wide multiplier; D in % of MoE output energy")
    print(f"{'scenario':8s} {'K set':14s} {'prune':5s} {'D layer-budget':>14s} {'D global':>9s} {'hist K':>34s} {'GB/tok':>7s}")
    for sc in ("ref", "flat", "steep"):
        shares, pi = make_layers(sc, a.seed)
        for ks, pr in (((2, 3), False), ((2, 3), True), ((1, 2), False), ((1, 2, 3), False), ((1, 2, 3, 4), False),
                       ((1, 2, 3), True), ((1, 2, 3, 4), True)):
            try:
                pm = (pi < a.tau) if pr else None
                K, used, D = solve_scheme(shares, eb, ks, pr, pc, prunable=pm)
                Kg, ug, Dg = solve_scheme(shares, eb, ks, pr, pc, glob=True, prunable=pm)
            except ValueError:
                print(f"{sc:8s} {','.join(map(str, ks)):14s} {str(pr):5s} {'infeasible':>14s}")
                continue
            hist = {int(k): int((K == k).sum()) for k in sorted(set(K.ravel().tolist()))}
            print(f"{sc:8s} {','.join(map(str, ks)):14s} {str(pr):5s} {D * 100:13.2f}% {Dg * 100:8.2f}% "
                  f"{json.dumps(hist):>34s} {read_bytes_per_token(K, pi) / 1e9:7.2f}")


def add_common(ap):
    ap.add_argument("--map", default=os.path.join(HERE, "..", "..", "scratch", "tensor_map.json"))
    ap.add_argument("--ram-gib", type=float, default=118.0)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--runtime-gib", type=float, default=5.0)
    ap.add_argument("--engram-cache-gib", type=float, default=0.5)
    ap.add_argument("--text-only", action="store_true", help="vision tower + aligner not resident")
    ap.add_argument("--embed-nvme", action="store_true", help="embedding matrix not resident (one row per token)")
    ap.add_argument("--attn-bits", type=int, default=4)
    ap.add_argument("--shared-bits", type=int, default=4)
    ap.add_argument("--head-bits", type=int, default=6)
    ap.add_argument("--mtp-bits", type=int, default=4, help="MTP dense linears")
    ap.add_argument("--mtp-exp-bits", type=int, default=2, help="MTP routed experts (3 x 128)")
    ap.add_argument("--role-bits", help="attention role overrides, e.g. q_a=5,kv=5,q_b=3,wo_a=3,wo_b=3,idx_q_b=4")
    ap.add_argument("--scenario", default="ref", choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=1e-4, help="rarely routed: pruning is allowed only for experts picked by "
                    "fewer than this fraction of the tokens (model: pi < tau; converter: count / tokens < tau)")
    ap.add_argument("--beta", type=float, default=0.3, help="prune cost = 1 + beta (substitute expert's extra error)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("budget")
    add_common(s)
    s = sub.add_parser("curves")
    add_common(s)
    s.add_argument("--uniform-k", type=int, default=2)
    s.add_argument("--glob", action="store_true", help="one multiplier for the whole network instead of per-layer budgets")
    s.add_argument("--out")
    for nm in ("levers", "ksets"):
        s = sub.add_parser(nm)
        add_common(s)
    s = sub.add_parser("plan")
    add_common(s)
    s.add_argument("--ks", default="1,2,3")
    s.add_argument("--prune", action="store_true")
    s.add_argument("--margin", type=float, default=0.01, help="keep this fraction of the expert budget free")
    s.add_argument("--out")
    a = ap.parse_args(argv)
    return {"budget": cmd_budget, "curves": cmd_curves, "plan": cmd_plan, "levers": cmd_levers, "ksets": cmd_ksets}[a.cmd](a)


if __name__ == "__main__":
    main()
