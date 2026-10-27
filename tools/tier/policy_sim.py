#!/usr/bin/env python
"""Offline cache-policy simulator for the tiered expert store (CPU only).

  python tools/tier/policy_sim.py TRACE.npz --frac 0.25 0.5 [--halflife 4 16 64 256] [--phase decode|all]

TRACE.npz comes from EXL3_EXPERT_TRACE_OUT (one entry per layer call: unique expert ids). One pool of
`frac * (layers * experts)` slots shared by all layers (what the store does). A call's own experts are never
victims (the whole union must be resident). Policies: lru, decay<h> (frequency x recency, half-life h steps,
the store's policy), lfu (no decay), belady (offline optimum for equal-size items, an upper bound).
Hit rates are counted on the chosen phase (decode = after the last stats reset, prefill = before it).
"""
import argparse, heapq, math, collections
import numpy as np


def load(path):
    z = np.load(path)
    off, ids = z["off"], z["ids"]
    calls = [(int(z["step"][i]), int(z["layer"][i]), int(z["rows"][i]), ids[off[i]:off[i + 1]].astype(np.int64))
             for i in range(len(z["step"]))]
    return calls, z["mark"]


def sim(calls, nslots, policy, count_from, E=1 << 12, halflife=256.0):
    key = lambda l, e: l * E + e
    nxt = None
    if policy == "belady":      # next use index (call number) of each access, scanning backwards
        nxt = [None] * len(calls)
        last = {}
        for i in range(len(calls) - 1, -1, -1):
            _, l, _, u = calls[i]
            nxt[i] = np.array([last.get(key(l, int(e)), 1 << 60) for e in u])
            for e in u:
                last[key(l, int(e))] = i
    res = {}            # key -> priority (larger = keep)
    heap = []           # (priority, key) lazy min-heap
    hits = uses = 0
    lru = collections.OrderedDict()
    lval = {}           # decay: key -> L (log2 of value normalized to t=0)
    for i, (step, l, rows, u) in enumerate(calls):
        cur = {key(l, int(e)) for e in u}
        t = step / halflife
        for j, e in enumerate(u):
            k = key(l, int(e))
            hit = k in res
            if i >= count_from:
                uses += 1
                hits += hit
            if policy == "lru":
                pr = i
            elif policy == "lfu":
                pr = (res.get(k, (0,))[0] if hit else lval.get(k, 0)) + 1
            elif policy.startswith("decay"):
                old = lval.get(k)
                lv = t if old is None else float(np.logaddexp2(old, t))
                lval[k] = lv
                pr = lv
            else:  # belady: keep the soonest next use
                pr = -int(nxt[i][j])
            if hit:
                res[k] = (pr,)
                heapq.heappush(heap, (pr, k))
                continue
            while len(res) >= nslots:
                keep = []
                while heap:
                    p, vk = heapq.heappop(heap)
                    if vk not in res or res[vk][0] != p:
                        continue                    # stale
                    if vk in cur:
                        keep.append((p, vk))        # the call's own expert: protected
                        continue
                    del res[vk]
                    break
                for x in keep:
                    heapq.heappush(heap, x)
                if len(res) >= nslots and not heap:
                    raise RuntimeError("every resident entry is protected: nslots < the call's union")
            res[k] = (pr,)
            heapq.heappush(heap, (pr, k))
        if policy == "lfu":
            for e in u:
                lval[key(l, int(e))] = res[key(l, int(e))][0]
    return hits / max(uses, 1), uses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--frac", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--halflife", type=float, nargs="+", default=[8, 64, 256])
    ap.add_argument("--phase", default="decode")
    ap.add_argument("--policies", nargs="+", default=["lru", "lfu", "decay", "belady"])
    a = ap.parse_args()
    calls, marks = load(a.trace)
    layers = max(c[1] for c in calls) + 1
    E = int(max(int(c[3].max()) for c in calls if len(c[3]))) + 1
    cf = int(marks[-1]) if (a.phase == "decode" and len(marks) > 1) else 0
    print(f"{len(calls)} calls, {layers} layers, ~{E} experts/layer, counting from call {cf} ({a.phase})")
    for f in a.frac:
        ns = int(f * layers * E)
        row = []
        for p in a.policies:
            if p == "decay":
                for h in a.halflife:
                    r, n = sim(calls, ns, f"decay", cf, halflife=h)
                    row.append((f"decay h={h:g}", r))
            else:
                r, n = sim(calls, ns, p, cf)
                row.append((p, r))
        print(f"frac {f:.2f} ({ns} slots): " + "  ".join(f"{k} {v * 100:.1f}%" for k, v in row), flush=True)


if __name__ == "__main__":
    main()
