#!/usr/bin/env python3
"""Score a DS4.1 pack on the Lucebox held-out tokens with the Lucebox metric definitions.

  python3 luce_eval.py run   --data DIR --base-url http://127.0.0.1:8399 --label OURS --out ours_kl.json
  python3 luce_eval.py run   --data DIR --model PACK_DIR [--loader ds41_serving:load_model] ...   (in process)
  python3 luce_eval.py table --data DIR --ours ours_kl.json [more.json ...]

DIR holds the public files of huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF:
  eval/heldout_8x1024.npy   int32 (R, L) token ids, R sequences of L tokens
  eval/teacher_8x1024.npz   idx int32 (R*(L-1), K), lp float32 (R*(L-1), K): the unquantized model's top-K ids (best first)
                            and full-vocabulary natural-log probabilities; nll, tokens: scalar mean NLL and count
  eval/teacher_kl.json, results/*_kl.json   their raw scores (table mode reads these)
Flat row i = sequence i // (L-1), position i % (L-1); it predicts heldout[seq, pos + 1]. (Checked on the public file:
the teacher argmax equals the next token on 54 % of rows, the next token is inside the top-256 on 97.5 %.)

Metrics (see doc/ds41_luce_eval.md for what is proven and what is inferred):
  kl_top256   mean over positions of sum_{i in teacher top-K} p_i (log p_i - log q_i); p = teacher, q = the candidate's
              FULL-vocabulary softmax at the same ids, no renormalization, no tail term (KL(teacher || candidate))
  top1_agree  candidate argmax == teacher argmax (idx[:, 0])
  nll / ppl   mean of -log q(next token) over all positions, ppl = exp(nll); next token scored under the full softmax
Extra, ours: kl_top256_renorm (p and q both renormalized on the top-K set), kld_tail (kl_top256 + the tail bucket
p_tail log(p_tail / q_tail), our gate's definition at K = 256, a lower bound of the full-vocabulary KLD).
"""
import argparse
import asyncio
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ---- data -----------------------------------------------------------------------------------------------------------

def load_teacher(data_dir):
    """-> tokens (R, L) int64, idx (N, K) int64, lp (N, K) float64, nll (float, teacher), N = R * (L - 1)."""
    ev = os.path.join(data_dir, "eval") if os.path.isdir(os.path.join(data_dir, "eval")) else data_dir
    tokens = np.load(os.path.join(ev, "heldout_8x1024.npy")).astype(np.int64)
    z = np.load(os.path.join(ev, "teacher_8x1024.npz"))
    idx, lp = z["idx"].astype(np.int64), z["lp"].astype(np.float64)
    r, l = tokens.shape
    if idx.shape != lp.shape or idx.shape[0] != r * (l - 1):
        raise ValueError(f"teacher {idx.shape} does not match tokens {tokens.shape}")
    return tokens, idx, lp, float(z["nll"])


def next_tokens(tokens):
    return tokens[:, 1:].reshape(-1)


# ---- metrics (pure numpy) -------------------------------------------------------------------------------------------

def compute_metrics(t_idx, t_lp, target, q_lp, q_target_lp, q_top1):
    """t_idx/t_lp (N, K): teacher top-K ids and log-probs; target (N,) the next token; q_lp (N, K) candidate log-probs at
    t_idx; q_target_lp (N,) candidate log-prob of target; q_top1 (N,) candidate argmax. Returns a dict of floats."""
    t_lp, q_lp = np.asarray(t_lp, np.float64), np.asarray(q_lp, np.float64)
    p = np.exp(t_lp)
    kl = (p * (t_lp - q_lp)).sum(-1)
    lt, lq = np.logaddexp.reduce(t_lp, -1), np.logaddexp.reduce(q_lp, -1)
    kl_renorm = (np.exp(t_lp - lt[:, None]) * ((t_lp - lt[:, None]) - (q_lp - lq[:, None]))).sum(-1)
    ptail = np.clip(-np.expm1(np.minimum(lt, 0.0)), 0.0, None)
    qtail = np.clip(-np.expm1(np.minimum(lq, 0.0)), 1e-12, None)
    tail = np.where(ptail > 1e-12, ptail * (np.log(np.clip(ptail, 1e-300, None)) - np.log(qtail)), 0.0)
    nll = float(-np.mean(q_target_lp))
    return {"tokens": int(len(target)), "nll": nll, "ppl": float(np.exp(nll)),
            "kl_top256": float(kl.mean()), "top1_agree": float((np.asarray(q_top1) == t_idx[:, 0]).mean()),
            "kl_top256_renorm": float(kl_renorm.mean()), "kld_tail": float((kl + tail).mean()),
            "kl_top256_per_pos": kl, "top1_per_pos": (np.asarray(q_top1) == t_idx[:, 0])}


def teacher_consistency(t_idx, t_lp, target, nll_file):
    """Facts about the teacher file that pin the alignment and the NLL definition (no candidate needed)."""
    hit = t_idx == target[:, None]
    seen = hit.any(1)
    nll_seen = float(-(np.where(hit, t_lp, 0.0).sum(1))[seen].sum())
    total = nll_file * len(target)
    miss_n = int((~seen).sum())
    implied_miss = (total - nll_seen) / max(miss_n, 1)
    floor = float((-t_lp[~seen, -1]).mean()) if miss_n else float("nan")       # a miss has p <= the K-th probability
    return {"next_token_in_topk": float(seen.mean()), "teacher_argmax_is_next": float((t_idx[:, 0] == target).mean()),
            "misses": miss_n, "implied_mean_miss_nll": implied_miss, "miss_nll_lower_bound": floor,
            "ppl_from_nll": float(np.exp(nll_file))}


# ---- backends: score(prompt_ids, cont_ids, cand_ids) -> [{top1, cand_logps}] -----------------------------------------

class HttpBackend:
    def __init__(self, base_url, timeout=7200):
        self.url, self.timeout = base_url.rstrip("/") + "/v1/ds41/score", timeout

    def score(self, prompt_ids, cont_ids, cand_ids):
        body = {"prompt_ids": prompt_ids, "continuation_ids": cont_ids, "candidate_ids": cand_ids}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["positions"]


class EngineBackend:
    """In-process PortEngine (ds41_engine.load_model with a loader), same code path as the /v1/ds41/score route."""

    def __init__(self, model, ctx, loader, **opts):
        import ds41_engine
        self.engine = ds41_engine.load_model(os.path.expanduser(model), ctx, loader=loader, **opts)

    def score(self, prompt_ids, cont_ids, cand_ids):
        return asyncio.run(self.engine.score("", cont_ids, cand_ids, prompt_ids=prompt_ids))


def score_rows(backend, tokens, idx, target, rows=None, max_len=None, cache_dir=None, log=print):
    """Teacher-force every selected sequence. Candidates per position = teacher top-K ids + the next token (for ppl).
    Returns arrays aligned with the selected flat rows: q_lp (n, K), q_target_lp (n,), q_top1 (n,), sel (n,) flat indices."""
    r, l = tokens.shape
    per = l - 1
    use = per if max_len is None else min(per, max_len)
    q_lp, q_t, q_1, sel = [], [], [], []
    t0 = time.time()
    for s in (range(r) if rows is None else rows):
        base = s * per
        cf = os.path.join(cache_dir, f"row{s:03d}_{use}.json") if cache_dir else None
        if cf and os.path.exists(cf):
            pos = json.load(open(cf))
        else:
            cands = [idx[base + p].tolist() + [int(target[base + p])] for p in range(use)]
            pos = backend.score(tokens[s, :1].tolist(), tokens[s, 1:1 + use].tolist(), cands)
            if cf:
                os.makedirs(cache_dir, exist_ok=True)
                json.dump(pos, open(cf, "w"))
        for p, d in enumerate(pos):
            c = d["cand_logps"]
            q_lp.append(c[:-1]); q_t.append(c[-1]); q_1.append(d["top1"]); sel.append(base + p)
        log(f"sequence {s}: {len(pos)} positions  [{time.time() - t0:.0f}s]")
    return np.array(q_lp, np.float64), np.array(q_t, np.float64), np.array(q_1, np.int64), np.array(sel, np.int64)


def evaluate(backend, data_dir, label, rows=None, max_len=None, cache_dir=None, log=print):
    tokens, idx, lp, t_nll = load_teacher(data_dir)
    target = next_tokens(tokens)
    q_lp, q_t, q_1, sel = score_rows(backend, tokens, idx, target, rows, max_len, cache_dir, log)
    m = compute_metrics(idx[sel], lp[sel], target[sel], q_lp, q_t, q_1)
    m.pop("kl_top256_per_pos"); m.pop("top1_per_pos")
    full = len(sel) == len(target)
    t_sel_nll = t_nll if full else None
    return {"label": label, **m, "teacher_nll": t_sel_nll, "teacher_ppl": float(np.exp(t_nll)) if full else None,
            "comparable": full, "definition": "luce-inferred-v1"}


# ---- table ----------------------------------------------------------------------------------------------------------

def table(data_dir, ours):
    base = os.path.join(data_dir, "results") if os.path.isdir(os.path.join(data_dir, "results")) else data_dir
    rows = []
    tk = os.path.join(data_dir, "eval", "teacher_kl.json")
    if os.path.exists(tk):
        t = json.load(open(tk))
        rows.append(("teacher (1017, unquantized)", 0.0, 1.0, t["ppl"], None))
    for f in sorted(glob.glob(os.path.join(base, "*_kl.json"))):
        d = json.load(open(f))
        if "kl_top256" not in d:
            continue
        name = d["label"].replace("DeepSeek-V4.1-Flash-1017-", "")
        size = d.get("bytes")
        rows.append((name, d["kl_top256"], d["top1_agree"], d["ppl"], size))
    for f in ours:
        d = json.load(open(f))
        flag = "" if d.get("comparable", True) else " (partial set, not comparable)"
        rows.append((d["label"] + flag, d["kl_top256"], d["top1_agree"], d["ppl"], d.get("bytes")))
    out = ["| model | KL top-256 vs teacher | top-1 | ppl | size |", "|---|---|---|---|---|"]
    for n, kl, t1, ppl, size in rows:
        out.append(f"| {n} | {kl:.3f} | {100 * t1:.1f}% | {ppl:.2f} | {'%.0f GB' % (size / 1e9) if size else ''} |")
    return "\n".join(out)


# ---- CLI ------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", default="luce_eval.json")
    r.add_argument("--base-url")
    r.add_argument("--model")
    r.add_argument("--loader", default="ds41_serving:load_model")
    r.add_argument("--opt", action="append", default=[], help="loader keyword KEY=VALUE (JSON value or string)")
    r.add_argument("--ctx", type=int, default=2048)
    r.add_argument("--rows", help="comma list of sequence indices (default all 8)")
    r.add_argument("--max-len", type=int, help="score only the first N positions per sequence (result not comparable)")
    r.add_argument("--cache", help="directory for per-sequence score caches (resume)")
    r.add_argument("--bytes", type=int, help="pack size in bytes, shown in the table")
    t = sub.add_parser("table")
    t.add_argument("--data", required=True)
    t.add_argument("--ours", nargs="*", default=[])
    c = sub.add_parser("check", help="teacher-file consistency + their reported numbers (no candidate needed)")
    c.add_argument("--data", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "table":
        print(table(a.data, a.ours))
        return 0
    if a.cmd == "check":
        tokens, idx, lp, nll = load_teacher(a.data)
        print(json.dumps(teacher_consistency(idx, lp, next_tokens(tokens), nll), indent=1))
        return 0
    if bool(a.base_url) == bool(a.model):
        ap.error("give exactly one of --base-url / --model")
    if a.base_url:
        backend = HttpBackend(a.base_url)
    else:
        opts = {}
        for kv in a.opt:
            k, _, v = kv.partition("=")
            try:
                opts[k] = json.loads(v)
            except json.JSONDecodeError:
                opts[k] = v
        backend = EngineBackend(a.model, a.ctx, a.loader, **opts)
    res = evaluate(backend, a.data, a.label, [int(x) for x in a.rows.split(",")] if a.rows else None, a.max_len, a.cache)
    if a.bytes:
        res["bytes"] = a.bytes
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
