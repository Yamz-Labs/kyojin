#!/usr/bin/env python
"""Decode speed of the RAM-cache + NVMe expert tier WITH prefetch overlap, replayed on real routing.

  python tools/tier/prefetch_model.py OUT_DIR [OUT_DIR2] --calib-json calib_ds41.json --table

Input: pred_probe.py output (real DS4.1 routing of 17 rows x 1024 tokens and the guesses of each predictor).
Replay: rows in domain-interleaved order, ONE per-layer LRU for the whole stream (warm-up rows discarded). A step verifies
T tokens (T = 1: plain decode; T = N+1 with DSpark) and advances by the accepted length. Per layer the step has
  attention phase a, router, [demand reads of the experts missing from the LRU], expert phase t_e x (unique experts).
NVMe is ONE server (4.1 GB/s with >= 2 reads waiting, 3.4 GB/s alone: 10 MB QD1), non-preemptive, demand queue before prefetch
queue, queued prefetch can be promoted/cancelled. Prefetch (the store's design): guesses of layers l+1..l+D from the router
input of layer l, read into staging entries (no LRU pollution: a staged expert enters the cache only when used), bounded by
the staging ring. Layers 0-2 are hash routed: their experts are known from the token ids at step start (exact prefetch).
Assumptions (flagged in the output): accepted length per DSpark step, draft cost, expert/attention split of the layer time.
"""
import argparse, glob, itertools, json, os, sys
from collections import OrderedDict
import numpy as np

E, NL = 384, 40
LAYERS = list(range(NL))      # replayed layers (the bench replays a subset)
P_EXPERT = 3 * 5120 * 2304            # parameters per expert
GiB = 1 << 30
DOM = {"wiki_en": "en", "web_en_fineweb_edu": "en", "wiki_fr": "fr", "web_fr_fineweb2": "fr", "code_github_clean": "code",
       "math_finemath3plus": "math", "tool_hermes_function_calling": "tool", "chat_ultrachat_multiturn": "chat",
       "reasoning_openthoughts": "reason", "stories_tinystories": "stories", "zh_fineweb2": "zh"}


def expert_bytes(bpw):
    return P_EXPERT * bpw / 8 * 1.004          # + per-expert signs/scales


def load(dirs):
    lay, rid, T = {}, [], None
    have = [set(int(os.path.basename(f)[5:-4]) for f in glob.glob(os.path.join(d, "layer*.npz"))) for d in dirs]
    common = set.intersection(*have)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "layer*.npz"))):
            l = int(os.path.basename(f)[5:-4])
            if l not in common:
                continue
            z = np.load(f)
            e = lay.setdefault(l, {})
            for k in z.files:
                if k in ("lens", "row_ids"):
                    continue
                e.setdefault(k, []).append(z[k])
            if l == 3:
                rid += z["row_ids"].tolist()
                T = int(z["lens"][0])
    for l in lay:
        lay[l] = {k: np.concatenate(v) for k, v in lay[l].items()}
    return lay, rid, T


def interleave(dom):
    names = ["tool", "code", "fr", "chat", "math", "en", "reason", "zh", "stories"]
    pools = {}
    for i, d in enumerate(dom):
        pools.setdefault(d, []).append(i)
    seq = []
    while any(pools.values()):
        for d in names:
            if pools.get(d):
                seq.append(pools[d].pop(0))
    return seq


class Trace:
    """Per (window length T, advance, cache fraction): the true unique experts, the LRU state before the step, and the
    expert-miss sets of every step and layer. Independent of the prefetch policy."""

    def __init__(self, lay, rows_order, rowlen, T, adv, cap_frac):
        self.lay, self.T, self.adv, self.rowlen = lay, T, adv, rowlen
        cap = max(1, int(round(cap_frac * E)))
        self.cap = cap
        self.steps = []         # (row, t0)
        for r in rows_order:
            for t in range(0, rowlen - T + 1, adv):
                self.steps.append((r, t))
        self.layers = sorted(lay)
        nst = len(self.steps)
        self.true = [[None] * NL for _ in range(nst)]
        self.cached = np.zeros((nst, NL, E // 8), np.uint8)
        self.miss = [[None] * NL for _ in range(nst)]
        caches = [OrderedDict() for _ in range(NL)]
        for si, (r, t) in enumerate(self.steps):
            a, b = r * rowlen + t, r * rowlen + t + T
            for l in LAYERS:
                u = np.unique(lay[l]["idx"][a:b])
                c = caches[l]
                m = np.zeros(E, bool)
                if c:
                    m[list(c.keys())] = True
                self.cached[si, l] = np.packbits(m)
                self.true[si][l] = u
                self.miss[si][l] = u[~m[u]]
                for e in u.tolist():
                    if e in c:
                        c.move_to_end(e)
                    else:
                        c[e] = 1
                        if len(c) > cap:
                            c.popitem(last=False)


class IO:
    """One NVMe server; jobs = one expert read each. Demand queue strictly before prefetch queue."""

    def __init__(self, nbytes, bw_hi=4.1e9, bw_lo=3.4e9):
        self.B, self.hi, self.lo = nbytes, bw_hi, bw_lo
        self.free_at = 0.0
        self.dq, self.pq = [], []            # jobs not yet started
        self.read_bytes = 0

    def add(self, job, prio):
        (self.dq if prio == 0 else self.pq).append(job)

    def promote(self, job):
        if job["start"] is None and job in self.pq:
            self.pq.remove(job)
            job["issue"] = max(job["issue"], 0)
            self.dq.append(job)

    def cancel(self, job):
        if job["start"] is None and job in self.pq:
            self.pq.remove(job)
            return True
        return False

    def _pick(self, t):
        for q in (self.dq, self.pq):
            for j in q:
                if j["issue"] <= t:
                    return j, q
        return None, None

    def _elig(self, t):
        return sum(1 for q in (self.dq, self.pq) for j in q if j["issue"] <= t)

    def step(self, limit):
        """Start the next job if its start time < limit. Returns False when nothing can start."""
        t = self.free_at
        j, q = self._pick(t)
        if j is None:
            nxt = min([x["issue"] for qq in (self.dq, self.pq) for x in qq] or [None], default=None) if (self.dq or self.pq) else None
            if nxt is None or nxt >= limit:
                return False
            t = nxt
            j, q = self._pick(t)
        if t >= limit:
            return False
        q.remove(j)
        rate = self.hi if self._elig(t) >= 1 else self.lo     # others waiting after this one
        j["start"] = t
        j["done"] = t + self.B / rate
        self.free_at = j["done"]
        self.read_bytes += self.B
        return True

    def run_to(self, limit):
        while self.step(limit):
            pass

    def finish(self, jobs):
        """Run until every job in `jobs` has started; returns the latest completion time."""
        while any(j["start"] is None for j in jobs):
            if not self.step(float("inf")):
                raise RuntimeError("job never started")
        return max(j["done"] for j in jobs) if jobs else 0.0


def simulate(tr, cfg, pol):
    """cfg: compute + io; pol: depth list [(variant, k)] per depth d=1..D, early flag, ring. Returns dict of means over
    the measured steps (after warm-up)."""
    B = expert_bytes(cfg["bpw"])
    c1, s = cfg["c1"], cfg["s_exp"]
    T = tr.T
    te = c1 * s / 6.0
    att = c1 * (1 - s) * (1 + 0.04 * (T - 1))
    head = cfg["head_ms"] * 1e-3
    c1, te, att = c1 * 1e-3, te * 1e-3, att * 1e-3
    draft = cfg["draft_ms"] * 1e-3 if T > 1 else 0.0
    deps = pol.get("deps", [])
    early = pol.get("early", True)
    ring = pol.get("ring", 32)
    nwarm = cfg["warm_steps"]
    tot_t, nst = 0.0, 0
    st = dict(demand=0, pf_used=0, pf_issued=0, pf_wasted=0, pf_cancel=0, stall=0.0, io_bytes=0.0, miss=0)
    lay = tr.lay
    rowlen = tr.rowlen
    LSET = set(LAYERS)
    for si, (r, t0) in enumerate(tr.steps):
        io = IO(B)
        t = 0.0
        a0, b0 = r * rowlen + t0, r * rowlen + t0 + T
        cachedm = np.unpackbits(tr.cached[si], axis=1).astype(bool)         # [NL, E]
        staged = {}         # (layer, expert) -> job
        out = 0
        stall = 0.0
        pf_issue = pf_used = pf_wasted = pf_cancel = demand = 0
        # hash layers 0-2: exact, known from the token ids at step start
        for l in [x for x in LAYERS if x < 3]:
            for e in tr.miss[si][l].tolist():
                j = dict(issue=0.0, start=None, done=None, l=l, e=e)
                staged[(l, e)] = j
                io.add(j, 1)
                out += 1
        pf_issue += out

        def issue(target, variant, k, when, depth_rank):
            nonlocal pf_issue
            if target < 3 or target not in LSET:
                return
            rk = np.full(E, 99, np.int64)
            if variant[0] == "O":          # oracle: the true experts (upper bound of any predictor with this lead)
                rk[tr.true[si][target]] = 0
            elif variant not in lay[target]:
                return
            else:
                P = lay[target][variant][a0:b0, :k].astype(np.int64)
                np.minimum.at(rk, P.ravel(), np.tile(np.arange(P.shape[1]), P.shape[0]))
            c = np.nonzero((rk < 99) & ~cachedm[target])[0]
            c = c[np.argsort(rk[c], kind="stable")]
            for e in c.tolist():
                if (target, e) in staged:
                    continue
                if len(staged) >= ring:
                    return
                j = dict(issue=when, start=None, done=None, l=target, e=e)
                staged[(target, e)] = j
                io.add(j, 1)
                pf_issue += 1

        for l in LAYERS:
            t_start = t
            # guesses that exist at the start of layer l (residual-stream predictors "A<offset>")
            for d, (variant, k) in enumerate(deps, start=1):
                if variant[0] == "A":
                    issue(l + int(variant[1:]), variant, k, t_start, d)
            t_r = t_start + att
            io.run_to(t_r)
            u = tr.true[si][l]
            miss = tr.miss[si][l]
            late_issue = []
            if early:
                for d, (variant, k) in enumerate(deps, start=1):
                    if variant[0] != "A":
                        issue(l + int(variant[1:]), variant, k, t_r, d)
            need = []
            for e in miss.tolist():
                j = staged.pop((l, e), None)
                if j is not None:
                    pf_used += 1
                    io.promote(j)
                else:
                    j = dict(issue=t_r, start=None, done=None, l=l, e=e)
                    io.add(j, 0)
                    demand += 1
                need.append(j)
            # staged guesses for this layer that the router did not pick: cancel what has not started
            for k_ in [k_ for k_ in staged if k_[0] == l]:
                j = staged.pop(k_)
                pf_wasted += 1
                if io.cancel(j):
                    pf_cancel += 1
            done = io.finish(need) if need else 0.0
            t_x = max(t_r, done)
            stall += t_x - t_r
            if not early:
                for d, (variant, k) in enumerate(deps, start=1):
                    if variant[0] != "A":
                        issue(l + int(variant[1:]), variant, k, t_x, d)
            t = t_x + te * len(u)
        t += head + draft
        if si >= nwarm:
            tot_t += t
            nst += 1
            st["demand"] += demand
            st["pf_used"] += pf_used
            st["pf_issued"] += pf_issue
            st["pf_wasted"] += pf_wasted
            st["pf_cancel"] += pf_cancel
            st["stall"] += stall
            st["io_bytes"] += io.read_bytes
            st["miss"] += sum(len(tr.miss[si][l]) for l in LAYERS)
    n = max(nst, 1)
    step_t = tot_t / n
    acc = cfg["accept"] if T > 1 else 1.0
    return dict(step_ms=step_t * 1e3, tok_s=acc / step_t, stall_ms=st["stall"] / n * 1e3,
                miss=st["miss"] / n, demand=st["demand"] / n, pf_issued=st["pf_issued"] / n, pf_used=st["pf_used"] / n,
                pf_wasted=st["pf_wasted"] / n, nvme_gbs=st["io_bytes"] / n / step_t / 1e9, steps=nst)


def accepted(N, p):
    """tokens produced per verify step with N draft tokens, per-token acceptance p (geometric): includes the bonus token"""
    return (1 - p ** (N + 1)) / (1 - p)


def run_cell(args):
    (lay_key, bpw, c1, N, pol_name, pol, p, ram_gib, ring) = args
    cap_override = G.get("cap_override")
    lay, rid, rowlen, dom = G["lay"], G["rid"], G["rowlen"], G["dom"]
    ebytes = expert_bytes(bpw)
    f = min(1.0, ram_gib * GiB / (NL * E * ebytes)) if cap_override is None else cap_override
    T = 1 if N == 0 else N + 1
    acc = accepted(N, p) if N else 1.0
    adv = max(1, int(round(acc))) if N else 1
    key = (T, adv, round(f, 4))
    if key not in G["tr"]:
        G["tr"][key] = Trace(lay, G["order"], rowlen, T, adv, f)
    tr = G["tr"][key]
    cfg = dict(bpw=bpw, c1=c1, s_exp=0.18 + (0.45 - 0.18) * max(0.0, min(1.0, (2.4 - c1) / (2.4 - 0.67))), head_ms=3.0,
               draft_ms=3 * c1 + 2.0, accept=acc, warm_steps=int(len(tr.steps) * 0.25))
    pol = dict(pol, ring=ring)
    r = simulate(tr, cfg, pol)
    return (bpw, c1, N, pol_name, ring), dict(r, cap_frac=f, accept=acc)


G = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--calib-json", required=True)
    ap.add_argument("--ram", type=float, default=100.0)
    ap.add_argument("--bpw", default="2.2,2.3,2.5")
    ap.add_argument("--c1", default="2.4,1.2,0.7", help="layer time ms: measured today / HIP graph / engine target")
    ap.add_argument("--N", default="0,3,4,5")
    ap.add_argument("--accept-p", type=float, default=0.70, help="per-draft-token acceptance (0.70 gives 2.5 tokens/step at N=3)")
    ap.add_argument("--ring", default="32")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--policies", default="all")
    ap.add_argument("--layers", default="0-39", help="replayed layers lo-hi (the bench replays a subset)")
    ap.add_argument("--cap", type=float, help="LRU fraction of the experts per layer (default: --ram / total expert bytes)")
    ap.add_argument("--out")
    a = ap.parse_args()
    global LAYERS
    lo, _, hi = a.layers.partition("-")
    LAYERS = list(range(int(lo), int(hi or lo) + 1))
    lay, rid, rowlen = load(a.dirs)
    LAYERS = [l for l in LAYERS if l in lay]
    G["cap_override"] = a.cap
    rs = json.load(open(a.calib_json))["row_sources"]
    dom = [DOM[rs[i]] for i in rid]
    G.update(lay=lay, rid=rid, rowlen=rowlen, dom=dom, tr={}, order=interleave(dom))
    pols = {
        "off": dict(deps=[]),
        "X1k6-late": dict(deps=[("X1", 6)], early=False),            # the store today (prefetch after the layer's demand)
        "X1k6": dict(deps=[("X1", 6)]),
        "X1k12": dict(deps=[("X1", 12)]),
        "X1k16": dict(deps=[("X1", 16)]),
        "X12k12": dict(deps=[("X1", 12), ("X2", 12)]),
        "X123k12": dict(deps=[("X1", 12), ("X2", 12), ("X3", 12)]),
        "A0A1k12": dict(deps=[("A0", 12), ("A1", 12)]),
        "A0A1A2k12": dict(deps=[("A0", 12), ("A1", 12), ("A2", 12)]),
        "A012+X1 k12": dict(deps=[("A0", 12), ("A1", 12), ("X1", 12), ("X2", 12)]),
        "A0A1k16": dict(deps=[("A0", 16), ("A1", 16)]),
        "oracle d1": dict(deps=[("O1", 6)]),
        "oracle d2": dict(deps=[("O2", 6)]),
        "oracle d3": dict(deps=[("O3", 6)]),
        "oracle d4": dict(deps=[("O4", 6)]),
        "oracle d6": dict(deps=[("O6", 6)]),
        "oracle d8": dict(deps=[("O8", 6)]),
    }
    if a.policies != "all":
        pols = {k: v for k, v in pols.items() if k in a.policies.split(";")}
    cells = []
    for bpw, c1, N, (pn, pol), ring in itertools.product([float(x) for x in a.bpw.split(",")], [float(x) for x in a.c1.split(",")],
                                                         [int(x) for x in a.N.split(",")], pols.items(), [int(x) for x in a.ring.split(",")]):
        cells.append((None, bpw, c1, N, pn, pol, a.accept_p, a.ram, ring))
    # group cells by trace key so each process builds a trace once
    from multiprocessing import Pool
    res = {}
    if a.procs > 1:
        # fork after loading: traces are built lazily per process; order cells to maximise reuse
        cells.sort(key=lambda c: (c[1], c[3]))
        with Pool(a.procs) as p:
            for k, v in p.imap_unordered(run_cell, cells, chunksize=1):
                res[k] = v
    else:
        for c in cells:
            k, v = run_cell(c)
            res[k] = v
    if a.out:
        json.dump({"|".join(map(str, k)): v for k, v in res.items()}, open(a.out, "w"), indent=1)
    keys = sorted(res)
    print(f"RAM for experts {a.ram:g} GiB as per-layer LRU; accept p={a.accept_p}; rows {len(rid)} x {rowlen} tokens\n")
    hdr = "bpw  cap  c1ms  N  acc  {:<14}ring  tok/s  step_ms stall_ms miss/s dem/s pfI/s pfU/s pfW/s NVMeGB/s"
    for bpw in sorted({k[0] for k in keys}):
        for c1 in sorted({k[1] for k in keys}, reverse=True):
            for N in sorted({k[2] for k in keys}):
                print(hdr.format("policy"))
                for k in keys:
                    if k[:3] != (bpw, c1, N):
                        continue
                    v = res[k]
                    print(f"{bpw:<4} {v['cap_frac']:.2f} {c1:<5} {N:<2} {v['accept']:.2f} {k[3]:<14}{k[4]:<5}{v['tok_s']:6.1f} {v['step_ms']:7.1f} {v['stall_ms']:7.1f} "
                          f"{v['miss']:6.1f} {v['demand']:5.1f} {v['pf_issued']:5.1f} {v['pf_used']:5.1f} {v['pf_wasted']:5.1f} {v['nvme_gbs']:6.2f}")
                print()


if __name__ == "__main__":
    main()
