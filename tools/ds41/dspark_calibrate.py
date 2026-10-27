"""Fit the DSpark serving knobs (draft depth N, confidence stop, DraftTuner) on real token streams.

    # 1) once the pack exists: one pass over the token rows records, for every position, which draft slots match the text
    #    and their confidence scores (GPU or CPU, minutes for a few thousand positions)
    python3 dspark_calibrate.py trace --pack <pack-folder> --ids gate_ids.npz --rows 0,1,2,6,7 --max-len 2048 \
        --depth 10 --device cuda:0 --out traces.npz
    # 2) pure numpy, seconds: replay the traces under every setting, score with the bandwidth cost model
    python3 dspark_calibrate.py fit --traces traces.npz [--overlap 0.5] [--eff 0.9] --out best.json

`trace` teacher-forces the stream through the target (one prefill, hidden of every row), then runs the drafter at every
position p exactly as the serving loop would (feed accepted rows, draft `--depth` tokens) and records
    match[p, i] = draft token i == text token p+1+i        (text acceptance: a proxy of greedy agreement, a lower bound)
    conf[p, i]  = raw confidence score of slot i (logit of acceptance)
`fit` replays rounds on the traces: at position p it drafts k tokens (fixed N, or the DraftTuner's choice), cuts the
draft at the first slot whose confidence probability is below the threshold, accepts the longest matching prefix, advances
accepted+1 positions. Round cost comes from dspark_cost.Cost. It prints the best (N, threshold) and the best auto setting
(DraftTuner k_max, cost_per_position, decay) and the matching server flags. Exactness is not at stake: all of these knobs
only change how many tokens one verify yields.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import dspark as ctl  # noqa: E402
from dspark_cost import Cost  # noqa: E402


# ---- trace --------------------------------------------------------------------------------------------------------


def trace_stream(spec, ids, depth: int, start: int = 2, stride: int = 1):
    """Match / confidence records of one token stream. `spec` is a PortSpec (or any target/drafter pair with the same API)."""
    import torch
    L = len(ids)
    out = spec.target.prefill(ids)
    hid = out.hidden                                         # [L, ...]
    positions = list(range(start, L - 1, stride))
    match = np.zeros((len(positions), depth), bool)
    conf = np.zeros((len(positions), depth), np.float32)
    spec.drafter._next = 0
    fed = 0
    for r, p in enumerate(positions):
        # the drafter needs every row up to p-1 once, in order; strided positions feed the rows in between
        rows = hid[fed:p] if fed else hid[:p]
        d = spec.drafter.draft(int(ids[p]), rows, p - 1)
        fed = p
        toks = d.tokens[:depth]
        for i, t in enumerate(toks):
            if p + 1 + i < L:
                match[r, i] = t == ids[p + 1 + i]
        conf[r, :len(toks)] = [float(c) for c in d.conf[:depth]]
    return np.array(positions), match, conf


def cmd_trace(a):
    import torch
    from ds41_serving import build_model
    from exllamav3.dsv41.spec import PortSpec
    if a.threads:
        torch.set_num_threads(min(8, a.threads))
    model = build_model(a.pack, device=a.device, dtype=a.dtype, mtp=True, moe=a.moe)
    z = np.load(a.ids, allow_pickle=True)
    rows = [f"row{int(r):03d}" for r in a.rows.split(",")] if a.rows else list(z.files)
    spec = PortSpec(model, a.max_len + a.depth + 16, max_depth=a.depth)
    res = {}
    for key in rows:
        ids = z[key][: a.max_len].tolist()
        pos, m, c = trace_stream(spec, ids, a.depth, stride=a.stride)
        res[f"{key}_pos"], res[f"{key}_match"], res[f"{key}_conf"] = pos, m, c
        print(f"{key}: {len(pos)} positions, slot0 match {m[:, 0].mean():.3f}, slot-wise {np.round(m.mean(0), 2).tolist()}", flush=True)
    np.savez_compressed(a.out, depth=a.depth, keys=np.array(rows), **res)


# ---- fit ----------------------------------------------------------------------------------------------------------


def load_traces(path):
    z = np.load(path, allow_pickle=True)
    return [(z[f"{k}_pos"], z[f"{k}_match"], z[f"{k}_conf"]) for k in z["keys"]]


def simulate(traces, cost: Cost, n_max: int, conf_thr: float | None, tuner: dict | None, stride_ok: bool = True):
    """Replay rounds. Returns dict(tok_s, tokens_per_round, rounds, accept_rate). Positions are the trace's own; with
    stride > 1 a round that advances past the next recorded position uses the nearest recorded one at or before it."""
    cut = None if conf_thr is None else math.log(conf_thr / (1 - conf_thr))
    tokens = 0
    time_s = 0.0
    rounds = prop = acc = 0
    for pos, match, conf in traces:
        tn = ctl.DraftTuner(1, n_max, **tuner) if tuner else None
        depth = match.shape[1]
        r = 0
        n = len(pos)
        while r < n:
            k = k0 = min(n_max, depth)
            if tn is not None:
                k = min(k, tn.choose())
            if cut is not None:
                kk = 0
                while kk < k and conf[r, kk] >= cut:
                    kk += 1
                k = kk
            m = 0
            while m < k and match[r, m]:
                m += 1
            tokens_round = m + 1
            time_s += cost.round_s(k, k0)
            if k:
                prop += k; acc += m
                if tn is not None:
                    tn.observe(k, min(tokens_round, k), m)
            tokens += tokens_round
            rounds += 1
            r = int(np.searchsorted(pos, pos[r] + tokens_round, side="left"))
    return {"tok_s": tokens / time_s, "tokens_per_round": tokens / max(rounds, 1), "rounds": rounds,
            "accept_rate": acc / max(prop, 1)}


def cmd_fit(a):
    traces = load_traces(a.traces)
    cost = Cost(a.gb_per_token, a.bw, a.eff, overlap=a.overlap, overhead_ms=a.overhead_ms, chain_decay=a.chain_decay,
                block=a.block)
    depth = traces[0][1].shape[1]
    base = 1 / cost.decode_s()
    rows = []
    for n in range(1, depth + 1):
        for thr in [None] + list(a.thresholds):
            s = simulate(traces, cost, n, thr, None)
            rows.append({"n": n, "conf": thr, "auto": False, **s})
    for n, cpp, dec in itertools.product(range(2, depth + 1), a.tuner_cost, a.tuner_decay):
        for thr in [None] + list(a.thresholds):
            s = simulate(traces, cost, n, thr, {"cost_per_position": cpp, "decay": dec})
            rows.append({"n": n, "conf": thr, "auto": True, "cost_per_position": cpp, "decay": dec, **s})
    rows.sort(key=lambda r: -r["tok_s"])
    best_fixed = next(r for r in rows if not r["auto"])
    best_auto = next((r for r in rows if r["auto"]), None)
    print(f"plain decode {base:.1f} tok/s (cost model)")
    for tag, r in (("fixed", best_fixed), ("auto", best_auto)):
        if r:
            print(f"best {tag}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})}")
    flags = f"--num-draft {best_fixed['n']}" + (f" --draft-conf {best_fixed['conf']}" if best_fixed["conf"] else "")
    if best_auto:
        flags_auto = f"--num-draft {best_auto['n']} --draft-auto" + (f" --draft-conf {best_auto['conf']}" if best_auto["conf"] else "")
        print("flags (fixed):", flags, "\nflags (auto): ", flags_auto,
              f"  (tuner cost_per_position={best_auto['cost_per_position']}, decay={best_auto['decay']})")
    if a.out:
        json.dump({"plain_tok_s": base, "best_fixed": best_fixed, "best_auto": best_auto, "top": rows[:20]}, open(a.out, "w"),
                  indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--pack", required=True)
    t.add_argument("--ids", required=True)
    t.add_argument("--rows", default="", help="comma list of row numbers (default all)")
    t.add_argument("--max-len", type=int, default=2048)
    t.add_argument("--depth", type=int, default=10)
    t.add_argument("--stride", type=int, default=1)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--dtype", default="float16")
    t.add_argument("--moe", default="auto")
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--out", default="traces.npz")
    f = sub.add_parser("fit")
    f.add_argument("--traces", required=True)
    f.add_argument("--out", default="")
    f.add_argument("--gb-per-token", type=float, default=7.2)
    f.add_argument("--bw", type=float, default=200.0)
    f.add_argument("--eff", type=float, default=0.9)
    f.add_argument("--overlap", type=float, default=0.0)
    f.add_argument("--overhead-ms", type=float, default=0.0)
    f.add_argument("--chain-decay", type=float, default=0.85)
    f.add_argument("--block", type=int, default=5)
    f.add_argument("--thresholds", type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    f.add_argument("--tuner-cost", type=float, nargs="*", default=[0.05, 0.1, 0.2])
    f.add_argument("--tuner-decay", type=float, nargs="*", default=[0.8, 0.9, 0.95])
    a = ap.parse_args()
    {"trace": cmd_trace, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
