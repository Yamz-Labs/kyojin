#!/usr/bin/env python3
"""Interleaved A/B of run-time knobs against ONE resident server (started with --admin-knobs). No reload per arm.

  ab_inproc.py PORT --arm A:EXL3_GEMV_TRI=1 --arm B:EXL3_GEMV_TRI=0 --order ABBAAB --synth 4096 [--users 1]

Each arm is NAME:K=V,K2=V2 (`K=` or `K=default` unsets K; an arm with no knobs is `NAME:`). A knob that some arm sets and another
does not is unset (its default) in the latter. The order string gives the visits (letters = arm names); every visit POSTs
/admin/knobs, discards one warm-up request (graphs recapture there), then runs every prompt once, greedy, streamed, under a new cache_salt. Per arm the report has
the median over visits of the per-visit median chunk gap (ms, one chunk = one decode step), the aggregate tok/s with --users > 1, and
the md5 of each prompt's reply; `md5 stable` says whether every visit of the arm produced the same bytes (meaningful with --users 1; concurrent users change the batch composition run to run).
Prompts: --prompts FILE (JSONL of strings/ {"messages": [...]}, or texts separated by a line `=====`) or --synth DEPTH[,DEPTH] (text of
~DEPTH tokens cut from --corpus).
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

CORPUS = os.path.expanduser("~/hermes-work/leadk/fleet/wt_roof3090/exllamav3/conversion/standard_cal_data")


def http(port, path, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", None if body is None else json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def parse_arms(specs):
    arms = {}
    for s in specs:
        name, _, kv = s.partition(":")
        if not name or name in arms:
            sys.exit(f"bad or repeated arm name in {s!r}")
        d = {}
        for item in filter(None, kv.split(",")):
            k, _, v = item.partition("=")
            d[k] = None if v in ("", "default") else v
        arms[name] = d
    return arms


def arm_values(arms, name):
    """the full knob dict an arm POSTs: its own values, every other arm's knob it does not mention unset"""
    names = {k for d in arms.values() for k in d}
    return {k: arms[name].get(k) for k in sorted(names)}


def load_prompts(args):
    if args.prompts:
        raw = open(args.prompts, encoding="utf8").read()
        if raw.lstrip().startswith(("\"", "{")):
            items = [json.loads(l) for l in raw.splitlines() if l.strip()]
        else:
            items = [t.strip("\n") for t in raw.split("\n=====\n") if t.strip()]
        return [{"messages": it["messages"]} if isinstance(it, dict) else {"messages": [{"role": "user", "content": it}]} for it in items]
    txt = "".join(open(f"{args.corpus}/{f}", encoding="utf8").read() for f in ("wiki.utf8", "c4.utf8", "technical.utf8", "code.utf8"))
    out = []
    for i, depth in enumerate(int(x) for x in args.synth.split(",")):
        chars, off = int(depth * 4.2), 7919 * (i + 1) + depth
        out.append({"messages": [{"role": "user", "content": (txt * 8)[off:off + chars] +
                                  "\n\nWrite a long, detailed continuation of the story about a lighthouse keeper."}]})
    return out


def request(port, model, prompt, max_tokens, extra):
    """one streamed greedy request -> (reply text, chunk timestamps, request start)"""
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": True, "enable_thinking": False, **prompt, **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    text, ts, t0 = [], [], time.time()
    for line in urllib.request.urlopen(req, timeout=3600):
        if not line.startswith(b"data: {"):
            continue
        ch = json.loads(line[6:])["choices"]
        d = ch[0].get("delta", {}) if ch else {}
        piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
        if piece:
            text.append(piece)
            ts.append(time.time())
    return "".join(text), ts, t0


def visit(port, model, prompts, users, max_tokens, extra, salt = None):
    """every prompt once (users > 1: the prompts run as `users` concurrent requests, round-robin) -> per-prompt md5, median gap ms, agg tok/s.
    salt: a cache_salt new for this visit (one for all its requests: a salt change drains the server), so no request finds its prompt
    in the prefix cache (prefill / TTFT arms)"""
    jobs = [prompts[i % len(prompts)] for i in range(max(users, len(prompts)))] if users > 1 else prompts
    res = [None] * len(jobs)
    def run(i):
        res[i] = request(port, model, jobs[i], max_tokens, {**extra, "cache_salt": salt} if salt else extra)
    t0 = time.time()
    if users > 1:
        th = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
        [t.start() for t in th]
        [t.join() for t in th]
    else:
        for i in range(len(jobs)):
            run(i)
    wall = time.time() - t0
    gaps, md5s, chunks, ttft = [], [], 0, []
    for text, ts, t_req in res:
        ttft.append(ts[0] - t_req if ts else float("nan"))
        md5s.append(hashlib.md5(text.encode()).hexdigest()[:10])
        mid = [b - a for a, b in zip(ts[len(ts) // 5:], ts[len(ts) // 5 + 1:])]      # drop the first fifth: prefill tail, graph warm
        gaps += mid
        chunks += len(ts)
    return {"md5": md5s, "gap_ms": statistics.median(gaps) * 1000 if gaps else float("nan"), "chunks": chunks, "wall": wall,
            "ttft_ms": statistics.median(ttft) * 1000, "ttft_all_ms": [round(x * 1000) for x in ttft]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", type=int)
    ap.add_argument("--arm", action="append", required=True, help="NAME:K=V,K2=V2")
    ap.add_argument("--order", required=True, help="visits, e.g. ABBAAB")
    ap.add_argument("--prompts")
    ap.add_argument("--synth", default="4096")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--model", default=None, help="model id (default: the first of /v1/models)")
    ap.add_argument("--users", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--warm-tokens", type=int, default=24)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request body")
    ap.add_argument("--fresh", action="store_true", help="warm up under another cache_salt than the visit: nothing, not even prompt 0, is served from the prefix cache (prefill / TTFT arms)")
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args()
    arms = parse_arms(args.arm)
    if set(args.order) - set(arms):
        sys.exit(f"order uses arms not declared: {sorted(set(args.order) - set(arms))}")
    extra = json.loads(args.extra)
    model = args.model or http(args.port, "/v1/models")["data"][0]["id"]
    prompts = load_prompts(args)
    visits = []
    for n, a in enumerate(args.order):
        t0 = time.time()
        eff = http(args.port, "/admin/knobs", arm_values(arms, a))["effective"]
        # a new cache_salt per visit: the server drops its prefix cache, so every visit starts from the same state (a prompt answered from the
        # cache and the same prompt prefilled give different last bits, and two visits of one arm in a row would otherwise differ). The
        # warm-up shares the visit's salt (it leaves prompt 0 cached, the same in every visit); --fresh warms under another salt so nothing is cached
        salt = f"ab{n}-{time.time():.0f}"
        request(args.port, model, prompts[0], args.warm_tokens, {**extra, "cache_salt": salt + ("w" if args.fresh else "")})   # warm-up per switch: graph recapture, discarded
        v = visit(args.port, model, prompts, args.users, args.max_tokens, extra, salt)
        v.update(arm=a, n=n)
        visits.append(v)
        print(f"visit {n} arm {a} gap {v['gap_ms']:.2f} ms  ttft {v['ttft_ms']:.0f} ms  ttft per prompt {'/'.join(str(x) for x in v['ttft_all_ms'])} ms  md5 {','.join(v['md5'])}  ({time.time() - t0:.0f} s)", flush=True)
    http(args.port, "/admin/knobs", {k: None for k in arm_values(arms, args.order[0])})          # leave the server on its defaults
    print("\narm  visits  median-gap-ms  ttft-ms  agg tok/s  md5 stable  md5 per prompt")
    summary = {}
    for a in arms:
        vs = [v for v in visits if v["arm"] == a]
        if not vs:
            continue
        gap = statistics.median(v["gap_ms"] for v in vs)
        agg = statistics.median(v["chunks"] / v["wall"] for v in vs)
        ttft = statistics.median(v["ttft_ms"] for v in vs)
        stable = all(v["md5"] == vs[0]["md5"] for v in vs)
        summary[a] = {"knobs": arms[a], "visits": len(vs), "gap_ms": gap, "ttft_ms": ttft, "agg_chunks_per_s": agg, "md5_stable": stable, "md5": vs[0]["md5"],
                      "gaps": [v["gap_ms"] for v in vs]}
        print(f"{a:<4} {len(vs):<7} {gap:<14.2f} {ttft:<8.0f} {agg:<10.1f} {str(stable):<11} {','.join(vs[0]['md5'])}   {arms[a]}")
    names = list(summary)
    if len(names) == 2:
        x, y = names
        print(f"\n{y} vs {x}: {summary[y]['gap_ms'] / summary[x]['gap_ms'] - 1:+.1%} step time, {summary[y]['ttft_ms'] / summary[x]['ttft_ms'] - 1:+.1%} ttft; reply bytes {'IDENTICAL' if summary[x]['md5'] == summary[y]['md5'] else 'DIFFER'}")
    if args.json:
        json.dump({"arms": summary, "visits": visits, "order": args.order}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
