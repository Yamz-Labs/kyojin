#!/usr/bin/env python
"""DS4.1 tiered-expert model: RAM budget -> resident experts -> hit rate -> decode tok/s (CPU only, no GPU).

  python tools/tier/model.py report [--json out.json]        budget table + tok/s grids + the >= 90 % budget
  python tools/tier/model.py hotset CAL.json [EVAL.json] --budget-gib 100 [--warm-out warm.json]
        CAL/EVAL = {layer key: [pick counts per expert]} (what EXL3_EXPERT_STATS_OUT writes, or a calibration
        capture); picks the resident set the store's warm start will pick and reports its hit rate on EVAL
  python tools/tier/model.py sim [--budgets 80,90,100,110] simulates the store policy on SYNTHETIC routing traces

Tags: [M] measured on this box (tools/tier/ssd_bench.py, scratch/ssd_bench.json), [Q] measured on Qwen3.8 with the
store (tier1 runs), [D] derived, [A] assumption (routing skew of DS4.1 is NOT measured yet: the grids are the answer
until the calibration capture exists).
"""
import argparse, json, math, sys
import numpy as np

GiB = 1 << 30
# ---- DS4.1-Flash geometry [config.json via ds41/DESIGN.md]
L, E, TOPK = 40, 384, 6
HID, INTER = 5120, 2304


def expert_bytes(K: float) -> int:
    """Bytes of one routed expert at trellis bitrate K: 3 linears of HID x INTER (trellis 16-bit words,
    16K words per 16x16 tile = 32K bytes per 256 weights = K/8 B per weight) + suh/svh fp16 vectors."""
    trellis = 3 * HID * INTER * K / 8
    vec = 2 * 2 * (HID + INTER) + 2 * (INTER + HID)     # gate/up: suh(in) svh(out); down: suh(INTER) svh(HID)
    return int(trellis + vec)


# ---- measured NVMe numbers [M] (scratch/ssd_bench.json, Kingston OM8, O_DIRECT, random, box under other tenants' load)
SSD_BW = 5.2e9            # B/s, 9-16 MiB reads, QD 4-8 (4.9-5.4 over 2-16 threads)
SSD_LAT = 0.5e-3          # s, first-byte penalty of a read burst: single 8.85 MiB read = 2.2 ms vs 1.7 ms at 5.2 GB/s
ENGRAM_IOPS = 78000       # 4 KiB random, QD 8-16 (engram rows are served by this, not by the expert path)

# ---- resident speed [A]: DS4-Flash Q2 resident on gfx1151 measured 16.79 tok/s (ds41/DESIGN.md 3d) -> V4.1 62.3 ms;
# MTP step from ds41/est/PERF_MODEL.md (A = 2.25 tok/step, 50 % efficiency)
T_PLAIN = 0.0623
T_MTP_STEP, MTP_ACC, MTP_UNION = 0.100, 2.25, 13

# ---- prefetch [Q]: Qwen3.8 next-layer guess: 53 % of the misses were already in flight, 47 % of issued reads used
PF_COVER, PF_PREC = 0.53, 0.47


def layer_ks(plan: str):
    """K per layer. ds41split: K=3 layers 0-7 and 28-39, K=2 elsewhere (2.50 bpw average)."""
    if plan == "ds41split":
        return [3 if (l < 8 or l >= 28) else 2 for l in range(L)]
    if plan == "k2":
        return [2] * L
    if plan == "k3":
        return [3] * L
    raise ValueError(plan)


# ------------------------------------------------------------------------------------------------ memory budget
def budget_table(total_gib=118.0, nonexp_gib=8.5, mtp_gib=2.7, kv_gib=1.5, runtime_gib=6.0, ring=48, plan="ds41split"):
    ks = layer_ks(plan)
    routed = sum(E * expert_bytes(k) for k in ks) / GiB
    avail = total_gib - nonexp_gib - mtp_gib - kv_gib - runtime_gib - ring * expert_bytes(3) / GiB
    return dict(routed_gib = routed, avail_gib = avail, resident_frac = min(1.0, avail / routed),
                total_gib = total_gib, nonexp_gib = nonexp_gib, mtp_gib = mtp_gib, kv_gib = kv_gib,
                runtime_gib = runtime_gib, ring_gib = ring * expert_bytes(3) / GiB)


# ------------------------------------------------------------------------------------------------ hit rates
def zipf_q(s: float):
    """Per-call probability q_r that rank r is among the top-6 picks: q_r = min(1, a r^-s), sum q = TOPK."""
    r = np.arange(1, E + 1, dtype = np.float64)
    lo, hi = 0.0, 1e9
    for _ in range(200):
        a = (lo + hi) / 2
        if np.minimum(1.0, a * r ** -s).sum() > TOPK:
            hi = a
        else:
            lo = a
    return np.minimum(1.0, lo * r ** -s)


def resident_per_layer(budget_gib: float, plan: str, hot3_frac: float | None):
    """Experts per layer the budget holds (same N in every layer: identical Zipf layers have equal marginals).
    plan 'ds41split' = K per layer; hot3_frac = per-expert hot-3/cold-2 (hottest fraction of ranks at K=3)."""
    if hot3_frac is not None:
        by = np.array([expert_bytes(3) if r < hot3_frac * E else expert_bytes(2) for r in range(E)], dtype = np.float64)
        per_layer_bytes = np.tile(by, (L, 1))
    else:
        per_layer_bytes = np.array([[expert_bytes(k)] * E for k in layer_ks(plan)], dtype = np.float64)
    cum = np.cumsum(per_layer_bytes.sum(0))               # bytes of the top-N experts over all layers
    n = int(np.searchsorted(cum, budget_gib * GiB, side = "right"))
    return min(n, E), per_layer_bytes


def zipf_hit(budget_gib, s, plan = "ds41split", hot3_frac = None):
    n, _ = resident_per_layer(budget_gib, plan, hot3_frac)
    q = zipf_q(s)
    return float(q[:n].sum() / TOPK), n


# ------------------------------------------------------------------------------------------------ decode time
def decode_tok_s(h, mode = "plain", plan = "ds41split", cover = 0.0, depth = 1, prec = PF_PREC, bw = SSD_BW,
                 miss_bytes = None, tokens = 400, seed = 7, prio = False):
    """Monte-Carlo decode tok/s at per-use hit rate h (one NVMe = one FIFO server, BW bytes/s, SSD_LAT per burst).
    Per layer the router picks n uses (6 plain, MTP_UNION in an MTP step); each is a miss with probability 1-h.
    A miss is prefetched with probability `cover`, issued `depth` layers before the layer's router (depth 1 = the
    next-layer guess, depth 40 = token-level), together with useless reads so that only `prec` of the issued
    prefetch reads are used; the others are read on demand after the router. GPU time per layer is t_res / L.
    Returns (tok/s, resident tok/s)."""
    rng = np.random.default_rng(seed)
    if mode == "plain":
        t_res, n_uses, per_tok = T_PLAIN, TOPK, 1.0
    else:
        t_res, n_uses, per_tok = T_MTP_STEP, MTP_UNION, MTP_ACC
    tl = t_res / L
    ks = layer_ks(plan)
    size = [float(miss_bytes if miss_bytes else expert_bytes(k)) for k in ks]
    M = rng.binomial(n_uses, 1.0 - h, size = (tokens, L))
    MP = rng.binomial(M, cover) if cover > 0 else np.zeros_like(M)
    W = np.where(MP > 0, np.round(MP * (1.0 / prec - 1.0)), 0).astype(int)
    M, MP, W = M.tolist(), MP.tolist(), W.tolist()
    lat = SSD_LAT
    tot = 0.0
    for ti in range(tokens):
        m, mp, w = M[ti], MP[ti], W[ti]
        free = 0.0
        g = 0.0
        ready = [0.0] * L
        look = [[] for _ in range(L)]          # prefetches issued at the router of layer l (targets)
        for l in range(L):
            if mp[l] > 0:
                look[max(l - depth, 0) if depth < L else 0].append(l)
        for l in range(L):
            for tgt in look[l]:
                start = max(g, free) + (lat if free <= g else 0.0)
                free = start + (mp[tgt] + w[tgt]) * size[tgt] / bw
                ready[tgt] = free
            md = m[l] - mp[l]
            done = g
            if md > 0:
                start = max(g, free) + (lat if free <= g else 0.0)
                free = start + md * size[l] / bw
                done = free
            if mp[l] > 0 and ready[l] > done:
                done = ready[l]
            g = (done if done > g else g) + tl
        tot += g
    return per_tok / (tot / tokens), per_tok / t_res


def hit_needed(frac, mode, plan, cover, depth = 1, prec = PF_PREC, bw = SSD_BW, miss_bytes = None):
    lo, hi = 0.0, 1.0
    for _ in range(13):
        h = (lo + hi) / 2
        t, t0 = decode_tok_s(h, mode, plan, cover, depth, prec, bw, miss_bytes, tokens = 150)
        if t / t0 >= frac:
            hi = h
        else:
            lo = h
    return hi


# ------------------------------------------------------------------------------------------------ hot set from counts
def hotset(cal: dict, ev: dict | None, budget_gib: float, k_of=None, topk=TOPK):
    """Greedy by pick probability per byte (the store's _split_budget / _warm_fill rule). cal/ev: {layer: counts}.
    k_of(layer_key, e) -> K (default: ds41split per-layer K from the layer number). Returns (picked, hit_cal, hit_eval)."""
    import re
    def lk(key):
        m = re.search(r"layers\.(\d+)", key)
        return int(m.group(1)) if m else 0
    ks = layer_ks("ds41split")
    cand = []
    for key, cnt in cal.items():
        c = np.asarray(cnt, dtype = np.float64)
        t = c.sum()
        if t <= 0:
            continue
        p = np.minimum(1.0, c * topk / t)
        l = lk(key)
        for e in np.nonzero(c > 0)[0]:
            K = k_of(key, int(e)) if k_of else ks[min(l, L - 1)]
            cand.append((p[e] / expert_bytes(K), key, int(e), expert_bytes(K)))
    cand.sort(key = lambda x: -x[0])
    left = budget_gib * GiB
    picked = {}
    for _, key, e, b in cand:
        if b <= left:
            picked.setdefault(key, set()).add(e)
            left -= b

    def hr(counts):
        hit = tot = 0.0
        for key, cnt in counts.items():
            c = np.asarray(cnt, dtype = np.float64)
            sel = list(picked.get(key, ()))
            hit += c[sel].sum()
            tot += c.sum()
        return hit / max(tot, 1.0)
    return picked, hr(cal), (hr(ev) if ev else None)


# ------------------------------------------------------------------------------------------------ synthetic traces
def synth_trace(tokens, s, sigma, domains, switch, seed, layers = 8):
    """Routing trace (tokens, layers, 6): per layer a Zipf(s) popularity over a random permutation, per domain a
    log-normal(sigma) modulation; the session stays in a domain and switches with probability `switch` per token.
    [A] nothing here is measured on DS4.1: it only exercises the policy."""
    rng = np.random.default_rng(seed)
    base = np.arange(1, E + 1, dtype = np.float64) ** -s
    P = np.zeros((layers, domains, E))
    for l in range(layers):
        perm = rng.permutation(E)
        for d in range(domains):
            mod = np.exp(sigma * rng.standard_normal(E))
            P[l, d, perm] = base * mod
        P[l] /= P[l].sum(1, keepdims = True)
    dom = 0
    out = np.empty((tokens, layers, TOPK), dtype = np.int32)
    doms = np.empty(tokens, dtype = np.int32)
    for t in range(tokens):
        if rng.random() < switch:
            dom = int(rng.integers(domains))
        doms[t] = dom
        for l in range(layers):
            out[t, l] = rng.choice(E, TOPK, replace = False, p = P[l, dom])
    return out, doms


def sim_policy(trace, slots_per_layer, policy, halflife = 256.0, warm_counts = None, pin_frac = 0.0):
    """One pool shared by all traced layers (identical classes), the store's policy: decayed use counter (a use adds 1,
    half-life `halflife` tokens), victim = lowest counter. policy: static (warm set never changes) | decay | lru (recency
    only, cold start). Returns the hit rate over the second half of the trace (uses = unique picks of a token and layer)."""
    T, nl, _ = trace.shape
    nslots = slots_per_layer * nl
    ncell = nl * E
    resident = np.zeros(ncell, dtype = bool)
    pinned = np.zeros(ncell, dtype = bool)
    cnt = np.zeros(ncell)
    st = np.zeros(ncell)
    if warm_counts is not None:
        w = warm_counts.reshape(-1).astype(np.float64)
        order = np.argsort(-w)[:nslots]
        order = order[w[order] > 0]
        resident[order] = True
        cnt[order] = w[order] / max(w.sum() / nl, 1.0) * TOPK * halflife / math.log(2)
        pinned[order[:int(pin_frac * nslots)]] = True
    lam = math.log(2) / halflife
    hits = uses = 0
    nres = int(resident.sum())
    for t in range(T):
        for l in range(nl):
            for e in set(trace[t, l].tolist()):
                k = l * E + e
                count = t >= T // 2
                if count:
                    uses += 1
                if resident[k]:
                    if count:
                        hits += 1
                    cnt[k] = cnt[k] * math.exp(-lam * (t - st[k])) + 1.0
                    st[k] = t
                    continue
                if policy == "static":
                    continue
                if nres >= nslots:
                    sc = st.copy() if policy == "lru" else cnt * np.exp(-lam * (t - st))
                    sc[~resident | pinned] = np.inf
                    v = int(np.argmin(sc))
                    resident[v] = False
                    nres -= 1
                resident[k] = True
                nres += 1
                cnt[k] = cnt[k] * math.exp(-lam * (t - st[k])) + 1.0
                st[k] = t
    return hits / max(uses, 1)


# ------------------------------------------------------------------------------------------------ report
def report(args):
    out = {}
    bt = budget_table()
    print("== RAM budget (GiB), DS4.1 2.5 bpw pack (K=3 layers 0-7,28-39; K=2 elsewhere) ==")
    print(f"  usable {bt['total_gib']:.0f} - non-expert {bt['nonexp_gib']} - MTP resident {bt['mtp_gib']} - KV {bt['kv_gib']} - "
          f"runtime/arena {bt['runtime_gib']} - staging ring {bt['ring_gib']:.2f} = {bt['avail_gib']:.1f} GiB for experts "
          f"of {bt['routed_gib']:.1f} GiB routed = {100 * bt['resident_frac']:.0f} % resident by bytes")
    for plan in ("k2", "ds41split", "k3"):
        r = sum(E * expert_bytes(k) for k in layer_ks(plan)) / GiB
        print(f"  routed {plan:10s}: {r:6.1f} GiB, {expert_bytes(2 if plan=='k2' else 3)/2**20:.2f} MiB/expert (K={2 if plan=='k2' else 3})")
    out["budget"] = bt
    print(f"\n== SSD [M]: {SSD_BW/1e9:.1f} GB/s pipelined, {SSD_LAT*1e3:.1f} ms burst latency; K=2 expert {expert_bytes(2)/2**20:.2f} MiB = "
          f"{expert_bytes(2)/SSD_BW*1e3:.2f} ms, K=3 {expert_bytes(3)/2**20:.2f} MiB = {expert_bytes(3)/SSD_BW*1e3:.2f} ms ==")
    print("\n== per-use hit rate needed for X % of the RAM-resident decode speed (miss = K=2/K=3 per layer plan) ==")
    print("  mode   prefetch (cover, layers ahead)        >=95 %   >=90 %   >=80 %")
    cases = ((0.0, 1, "none (demand reads only)"), (0.53, 1, "Qwen next-layer guess [Q]"), (0.5, 3, "guess 3 layers ahead"),
             (0.8, 3, "good predictor, 3 layers"), (1.0, 40, "perfect, whole-token window"))
    for mode in ("plain", "mtp"):
        for cov, dep, name in cases:
            row = [hit_needed(f, mode, "ds41split", cov, dep) for f in (0.95, 0.90, 0.80)]
            print(f"  {mode:5s}  c={cov:4.2f} D={dep:<2d} {name:27s} " + "   ".join(f"{100*x:6.2f}%" for x in row))
            out[f"need_{mode}_{cov}_{dep}"] = row
    print("\n== decode tok/s vs per-use hit rate (resident = %.1f plain / %.1f MTP) ==" % (1 / T_PLAIN, MTP_ACC / T_MTP_STEP))
    print("  hit    plain none  plain c=.53,D=1  plain c=.8,D=3 | MTP none  MTP c=.53,D=1  MTP c=.8,D=3")
    for h in (0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0):
        v = [decode_tok_s(h, m, "ds41split", c, d)[0] for m in ("plain", "mtp") for c, d in ((0.0, 1), (0.53, 1), (0.8, 3))]
        print(f"  {100*h:5.1f}  {v[0]:9.1f}  {v[1]:15.1f}  {v[2]:14.1f} | {v[3]:8.1f}  {v[4]:14.1f}  {v[5]:13.1f}")
    print("\n== static hit rate vs budget under Zipf(s) routing [A] (ranked resident set, same N per layer) ==")
    print("  budget GiB   resident/layer   " + "  ".join(f"s={s:<4}" for s in (0.0, 0.4, 0.8, 1.0, 1.2, 1.5)))
    grid = {}
    for b in (60, 80, 90, 100, 105, 110, 118, 130):
        n = resident_per_layer(b, "ds41split", None)[0]
        hs = [zipf_hit(b, s)[0] for s in (0.0, 0.4, 0.8, 1.0, 1.2, 1.5)]
        grid[b] = hs
        print(f"  {b:9d}   {n:5d} ({100*n/E:3.0f}%)      " + "  ".join(f"{100*x:5.1f}%" for x in hs))
    out["zipf_grid"] = grid
    print("\n== per-expert hot-3/cold-2 (hot 50 % of ranks at K=3 = 2.5 bpw): misses are K=2 reads ==")
    for b in (100, 105, 110):
        n = resident_per_layer(b, "ds41split", 0.5)[0]
        hs = [zipf_hit(b, s, hot3_frac = 0.5)[0] for s in (0.8, 1.0, 1.2)]
        tk = [decode_tok_s(h, "plain", "ds41split", 0.0, miss_bytes = expert_bytes(2))[0] for h in hs]
        print(f"  {b} GiB: {n} experts/layer ({100*n/E:.0f}%), hit(s=.8/1/1.2) = " + "/".join(f"{100*h:.1f}%" for h in hs)
              + ", plain tok/s " + "/".join(f"{x:.1f}" for x in tk))
    # the budget that keeps >= 90 % under each skew
    print("\n== smallest budget (GiB) with >= 90 % of resident decode speed, plain, no overlap [A skew] ==")
    need = hit_needed(0.90, "plain", "ds41split", 0.0)
    need_o = hit_needed(0.90, "plain", "ds41split", 0.8, 3)
    res = {}
    for s in (0.8, 1.0, 1.2, 1.5):
        row = []
        for nd in (need, need_o):
            bmin = None
            for b in np.arange(20, 175, 1.0):
                if zipf_hit(b, s)[0] >= nd:
                    bmin = float(b)
                    break
            row.append(bmin)
        res[s] = row
        print(f"  s={s}: need hit {100*need:.1f}% -> {row[0]} GiB;  with c=.8 D=3 prefetch need {100*need_o:.1f}% -> {row[1]} GiB")
    out["min_budget_90"] = {str(k): v for k, v in res.items()}
    # prefill
    print("\n== prefill [D]: one chunk touches ~every expert, reads all non-resident bytes once ==")
    miss_gib = max(0.0, bt["routed_gib"] - bt["avail_gib"])
    ssd_s = miss_gib * GiB / SSD_BW
    for rate in (340.0,):
        print(f"  non-resident {miss_gib:.1f} GiB = {ssd_s:.1f} s per chunk at {SSD_BW/1e9:.1f} GB/s; compute {rate:.0f} tok/s")
        for rows in (1024, 2048, 4096, 8192, 16384):
            comp = rows / rate
            no = comp / (comp + ssd_s)
            ov = comp / max(comp, ssd_s)
            print(f"  chunk {rows:6d}: compute {comp:5.1f} s -> no overlap {100*no:4.0f} %, next-layer prefetch {100*ov:4.0f} % of resident speed")
    if args.json:
        json.dump(out, open(args.json, "w"), indent = 1, default = float)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest = "cmd", required = True)
    r = sub.add_parser("report")
    r.add_argument("--json")
    h = sub.add_parser("hotset")
    h.add_argument("cal")
    h.add_argument("eval", nargs = "?")
    h.add_argument("--budget-gib", type = float, required = True)
    h.add_argument("--warm-out")
    s = sub.add_parser("sim")
    s.add_argument("--budgets", default = "0.5,0.64,0.8")
    s.add_argument("--tokens", type = int, default = 1500)
    a = ap.parse_args()
    if a.cmd == "report":
        report(a)
    elif a.cmd == "hotset":
        cal = json.load(open(a.cal))
        ev = json.load(open(a.eval)) if a.eval else None
        picked, hc, he = hotset(cal, ev, a.budget_gib)
        n = sum(len(v) for v in picked.values())
        print(f"resident experts {n}; hit rate on calibration {100*hc:.2f} %" + (f", on eval {100*he:.2f} %" if he is not None else ""))
        if a.warm_out:
            json.dump(cal, open(a.warm_out, "w"))
    elif a.cmd == "sim":
        layers = 8
        print("synthetic traces [A]: Zipf s, domain modulation sigma, hit rate over the 2nd half; slots = fraction of 384/layer")
        print("  s    sigma  frac   static(warm)  decay(warm)  decay+pin30(warm)  lru(cold)")
        for s_ in (0.8, 1.2):
            for sig in (0.0, 1.0):
                tr, doms = synth_trace(a.tokens, s_, sig, 6, 1 / 300, 1, layers)
                warm = np.zeros((layers, E))
                for t in range(tr.shape[0] // 2):         # calibration = a first, different slice of the session mix
                    for l in range(layers):
                        warm[l, tr[t, l]] += 1
                for fr in [float(x) for x in a.budgets.split(",")]:
                    n = int(fr * E)
                    v = [sim_policy(tr, n, "static", warm_counts = warm), sim_policy(tr, n, "decay", warm_counts = warm),
                         sim_policy(tr, n, "decay", warm_counts = warm, pin_frac = 0.3), sim_policy(tr, n, "lru")]
                    print(f"  {s_:<4} {sig:<5} {fr:<5}  " + "  ".join(f"{100*x:11.1f}%" for x in v))


if __name__ == "__main__":
    main()
