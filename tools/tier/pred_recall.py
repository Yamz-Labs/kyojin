#!/usr/bin/env python
"""Recall tables of the prefetch predictors (output of pred_probe.py, layers >= 3, hash layers 0-2 are exact from token ids).

  python tools/tier/pred_recall.py OUT_DIR [OUT_DIR2 ...] --calib-json calib_ds41.json [--cap 0.30,0.65] [--N 4]

Predictor names (target layer l): X<d> = router input of layer l-d through router l (what the expert store does today with
EXL3_EXPERT_AHEAD_DEPTH=d); A<d> = residual stream entering layer l-d, attention delta skipped; B<d> = residual after the
attention of layer l-d. Offset d = how many layers earlier the guess is available (window to hide the read).
recall@k = share of the true top-6 inside the guess' top-k. "miss recall" = share of the true experts that MISS an
LRU of cap x 384 per layer that the guess covers (experts already cached need no prefetch). "waste" = uncached guessed
experts per token and layer that the router did not pick.
"""
import argparse, glob, json, os
from collections import OrderedDict
import numpy as np

E = 384
GROUPS = {"3-4": range(3, 5), "5-19": range(5, 20), "20-22": range(20, 23), "23-36": range(23, 37), "37-39": range(37, 40)}
DOM = {"wiki_en": "en", "web_en_fineweb_edu": "en", "wiki_fr": "fr", "web_fr_fineweb2": "fr", "code_github_clean": "code",
       "math_finemath3plus": "math", "tool_hermes_function_calling": "tool", "chat_ultrachat_multiturn": "chat",
       "reasoning_openthoughts": "reason", "stories_tinystories": "stories", "zh_fineweb2": "zh"}


def load(dirs):
    """-> {layer: dict(idx [N,6], pred name -> [N,16])}, row ids, row length"""
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


def hit_frac(true, P):
    """per token: how many of the 6 true experts are in P [N,k]"""
    return (true[:, :, None] == P[:, None, :]).any(2)


def lru_miss_mask(true, cap, T):
    """[N,6] True where the true expert missed an LRU(cap) of the stream (rows concatenated, one cache), + cached-before mask is
    recomputed by callers through `before` snapshots only for waste; returns miss mask."""
    c = OrderedDict()
    miss = np.zeros(true.shape, bool)
    for t in range(true.shape[0]):
        for j, e in enumerate(true[t].tolist()):
            if e in c:
                c.move_to_end(e)
            else:
                miss[t, j] = True
                c[e] = 1
                if len(c) > cap:
                    c.popitem(last=False)
    return miss


def lru_states(true, cap):
    """bool [N,E]: experts cached before token t is served."""
    c = OrderedDict()
    st = np.zeros((true.shape[0], E), bool)
    for t in range(true.shape[0]):
        st[t, list(c.keys())] = True
        for e in true[t].tolist():
            if e in c:
                c.move_to_end(e)
            else:
                c[e] = 1
                if len(c) > cap:
                    c.popitem(last=False)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--calib-json")
    ap.add_argument("--cap", default="0.30,0.65")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--ks", default="6,8,12,16")
    ap.add_argument("--out")
    a = ap.parse_args()
    lay, rid, T = load(a.dirs)
    ks = [int(x) for x in a.ks.split(",")]
    caps = [float(x) for x in a.cap.split(",")]
    names = sorted({k for l in lay for k in lay[l] if k != "idx" and k not in ("X0",)}, key=lambda s: (s[0], int(s[1:])))
    layers = sorted(l for l in lay if l >= 3 and "X1" in lay[l])
    res = {}
    # sanity: X0 (true router input through the router) must reproduce the true top-6
    bad = sum(int((~hit_frac(lay[l]["idx"], lay[l]["X0"][:, :6]).all(1)).sum()) for l in layers if "X0" in lay[l])
    print(f"sanity X0 == true top-6: tokens off = {bad} of {len(rid) * T * len(layers)}")
    N = a.N
    nw = (T // N) * N
    print(f"rows {len(rid)} x {T} tokens, layers {layers[0]}-{layers[-1]} ({len(layers)}), window N={N}")
    for l in layers:
        true = lay[l]["idx"].astype(np.int64)
        ntok = true.shape[0]
        tm = np.zeros((ntok, E), bool)
        np.put_along_axis(tm, true, True, 1)
        tw = tm.reshape(-1, T, E)[:, :nw].reshape(-1, N, E).any(1)
        caches = {c: lru_states(true, max(1, int(round(c * E)))) for c in caps}
        for v in names:
            if v not in lay[l]:
                continue
            P = lay[l][v].astype(np.int64)
            for k in ks:
                pk = P[:, :k]
                h = hit_frac(true, pk)
                r = res.setdefault((v, k), {"rec": [], "win": [], "layer": []})
                r["layer"].append(l)
                r["rec"].append(h.mean())
                pm = np.zeros((ntok, E), bool)
                np.put_along_axis(pm, pk, True, 1)
                pw = pm.reshape(-1, T, E)[:, :nw].reshape(-1, N, E).any(1)
                r["win"].append(((pw & tw).sum() / tw.sum(), pw.sum(1).mean(), tw.sum(1).mean()))
                for c, st in caches.items():
                    miss = tm & ~st
                    got = (miss & pm).sum() / max(miss.sum(), 1)
                    waste = (pm & ~st & ~tm).sum() / ntok
                    r.setdefault(f"miss{c}", []).append((got, waste, miss.sum() / ntok))
    # table 1: recall@k, all learned layers + by depth group, X/A/B variants
    print("\nrecall@k of the true top-6 (mean over tokens and layers 3-39)")
    print("pred    " + "".join(f"  @{k:<5}" for k in ks) + "   " + "  ".join(f"{g:>6}@6" for g in GROUPS))
    for v in names:
        if (v, ks[0]) not in res:
            continue
        row = f"{v:<8}"
        for k in ks:
            row += f" {np.mean(res[(v, k)]['rec']):.3f}  "
        row += "  "
        for g, rg in GROUPS.items():
            idx = [i for i, l in enumerate(res[(v, ks[0])]['layer']) if l in rg]
            row += f" {np.mean([res[(v, ks[0])]['rec'][i] for i in idx]) if idx else float('nan'):8.3f}"
        print(row)
    print(f"\nwindow-union recall (N={N} consecutive tokens, union of per-token guesses) | mean guessed union size vs true union")
    print("pred    " + "".join(f"  @{k:<12}" for k in ks))
    for v in names:
        if (v, ks[0]) not in res:
            continue
        row = f"{v:<8}"
        for k in ks:
            w = np.array([x for x in res[(v, k)]["win"]])
            row += f" {w[:, 0].mean():.3f} ({w[:, 1].mean():4.1f}/{w[:, 2].mean():4.1f})"
        print(row)
    for c in caps:
        print(f"\nLRU {c:.0%} of the experts per layer: recall of the true experts that MISS the cache | uncached wrongly guessed experts per token and layer | misses/token/layer")
        print("pred    " + "".join(f"  @{k:<16}" for k in ks))
        for v in names:
            if (v, ks[0]) not in res:
                continue
            row = f"{v:<8}"
            for k in ks:
                m = np.array(res[(v, k)][f"miss{c}"])
                row += f" {m[:, 0].mean():.3f} w{m[:, 1].mean():5.2f} m{m[:, 2].mean():.2f}"
            print(row)
    if a.out:
        json.dump({f"{v}|{k}": {kk: np.asarray(vv, float).tolist() for kk, vv in d.items()} for (v, k), d in res.items()}, open(a.out, "w"))


if __name__ == "__main__":
    main()
