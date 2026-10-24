#!/usr/bin/env python3
"""DS4.1 quality gate: top-1 agreement and KLD of a served pack against the ORIGINAL model's stored logits.

  python3 gate.py --base-url http://127.0.0.1:8399 --ref DIR/gate_ref [--out gate_result.json]

Reference: gate_ref/ref_rowNNN.npz written during the conversion (tools/ds41/evalref.py in the converter): per position
the original model's top-64 log-probs + ids, argmax. Candidate: POST /v1/ds41/score teacher-forces the same token ids
(gate_ids.npz) and returns its full-vocabulary log-prob of each reference top-64 id and its own argmax.

Definitions (same as eval/ds41/kld_lib.py, the GLM/Qwen gate): position t predicts token t+1;
  top-1  = candidate argmax == reference argmax
  KLD    = sum_{i in ref top-64} p_i (log p_i - log q_i) + p_tail (log p_tail - log q_tail), q_i the candidate's full softmax
           probability of reference id i, q_tail = 1 - sum q_i (a lower bound of the full-vocabulary KLD, tight when peaked)
Positions < start (64) are not scored. Overall numbers weight every scored position equally; CI = row-cluster bootstrap.
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Pass bars = our published quant quality (Kyojin docs and notebooks, scorer = same top-1 / KLD definitions):
#   PASS   GLM-5.3-Flash pack (~100 GB, ~2.5 bpw MoE, the closest to this pack): top-1 89.3 %, KLD 0.151
#   STRONG Qwen3.8-Flash-Next pack (doc/benchmarks.md "Fidelity to the original model"): top-1 93.81 %, KLD 0.0216
BARS = {"pass": {"top1": 89.3, "kld": 0.151}, "strong": {"top1": 93.81, "kld": 0.0216}}


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def row_metrics(ref, pos, start):
    """ref: npz of the row; pos: server positions for t = start.. -> per-position kld, top1 (float arrays)."""
    n = len(pos)
    lp = ref["top_lp"][start:start + n].astype(np.float64)
    lq = np.array([p["cand_logps"] for p in pos], dtype=np.float64)
    top = (np.exp(lp) * (lp - lq)).sum(-1)
    lsp, lsq = np.logaddexp.reduce(lp, -1), np.logaddexp.reduce(lq, -1)
    ptail = np.clip(-np.expm1(np.minimum(lsp, 0.0)), 0.0, None)
    qtail = np.clip(-np.expm1(np.minimum(lsq, 0.0)), 1e-12, None)
    tail = np.where(ptail > 1e-12, ptail * (np.log(np.clip(ptail, 1e-300, None)) - np.log(qtail)), 0.0)
    t1 = (np.array([p["top1"] for p in pos]) == ref["argmax"][start:start + n]).astype(np.float64)
    return top + tail, t1


def boot(rows, key, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    sums = np.array([r[key].sum() for r in rows]); cnt = np.array([len(r[key]) for r in rows])
    out = []
    for _ in range(n):
        i = rng.integers(0, len(rows), len(rows))
        out.append(sums[i].sum() / cnt[i].sum())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def verdict(top1_pct, kld):
    if top1_pct >= BARS["strong"]["top1"] and kld <= BARS["strong"]["kld"]:
        return "STRONG"
    if top1_pct >= BARS["pass"]["top1"] and kld <= BARS["pass"]["kld"]:
        return "PASS"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--ref", required=True, help="gate_ref directory (manifest.json + ref_rowNNN.npz)")
    ap.add_argument("--ids", default=os.path.join(HERE, "data", "gate_ids.npz"))
    ap.add_argument("--out", default="gate_result.json")
    ap.add_argument("--rows", help="comma list of row indices (default all)")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.ref, "manifest.json")))
    ids = np.load(a.ids)
    keys = sorted(ids.files)
    meta = json.load(open(os.path.join(os.path.dirname(a.ids), "gate_meta.json")))
    if man["sha256"] != meta["sha256"]:
        sys.exit("reference was computed on a different eval set than gate_ids.npz")
    start = int(man.get("start", 64))
    sel = [int(x) for x in a.rows.split(",")] if a.rows else range(len(keys))
    cache = a.out + ".rows"
    os.makedirs(cache, exist_ok=True)
    rows, names = [], []
    t0 = time.time()
    for i in sel:
        row = ids[keys[i]].astype(np.int64)
        ref = np.load(os.path.join(a.ref, f"ref_row{i:03d}.npz"))
        cf = os.path.join(cache, f"row{i:03d}.json")
        if os.path.exists(cf):
            pos = json.load(open(cf))
        else:
            L = len(row)
            body = {"prompt_ids": row[:start + 1].tolist(), "continuation_ids": row[start + 1:].tolist(),
                    "candidate_ids": ref["top_idx"][start:L - 1].tolist()}
            r = post(a.base_url.rstrip("/") + "/v1/ds41/score", body)
            pos = r["positions"]
            json.dump(pos, open(cf, "w"))
        kld, t1 = row_metrics(ref, pos, start)
        rows.append({"kld": kld, "top1": t1}); names.append(meta["names"][i])
        print(f"row {i} {meta['names'][i]}: top-1 {100 * t1.mean():.2f} %  KLD {kld.mean():.4f}  [{time.time() - t0:.0f}s]", flush=True)
    n = sum(len(r["kld"]) for r in rows)
    kld = sum(r["kld"].sum() for r in rows) / n
    top1 = 100 * sum(r["top1"].sum() for r in rows) / n
    ci_k, ci_t = boot(rows, "kld"), boot(rows, "top1")
    v = verdict(top1, kld)
    res = {"verdict": v, "top1_pct": top1, "top1_ci95": [100 * x for x in ci_t], "kld": kld, "kld_ci95": list(ci_k),
           "positions": n, "rows": names, "bars": BARS,
           "per_row": [{"name": nm, "top1_pct": 100 * r["top1"].mean(), "kld": r["kld"].mean()} for nm, r in zip(names, rows)]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n{n} positions: top-1 {top1:.2f} % (95 % CI {100 * ci_t[0]:.2f}-{100 * ci_t[1]:.2f})  "
          f"KLD {kld:.4f} (95 % CI {ci_k[0]:.4f}-{ci_k[1]:.4f})")
    print(f"bars: PASS top-1 >= {BARS['pass']['top1']} and KLD <= {BARS['pass']['kld']} (GLM pack); "
          f"STRONG top-1 >= {BARS['strong']['top1']} and KLD <= {BARS['strong']['kld']} (Qwen pack)")
    print("VERDICT:", v)
    sys.exit(0 if v != "FAIL" else 1)


if __name__ == "__main__":
    main()
