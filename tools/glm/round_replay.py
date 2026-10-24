#!/usr/bin/env python3
"""Replay long-context decode points against a running GLM server and break each one into rounds.

Needs the server started with EXL3_ROUND_LOG=<file> (tools/glm/round_probe.py: one line per decode round)
and optionally KYOJIN_REQLOG_DIR=<dir> (lifecycle events with timestamps). The script runs on the same box,
reads the round file by byte offset between requests, and per point reports:

  decode_srv   server-reported predicted_per_second (generate window = first token -> eos event)
  decode_cli   client (tokens-1) / (last chunk - first chunk)
  rounds, acc/round (accepted drafts per round), tok/round, ms/round (mean, p50, p95, max), kinds (dr / ng rounds)
  gap_first_s  server first-token event -> client first chunk, i.e. delay before the stream shows the first token
  idle_s       round-file wall time not covered by the rounds themselves (sum of gaps between rounds)
  chunk_gap_max the biggest gap between two streamed chunks (and when it happened, in tokens)
  text         the generated text (JSON out) plus repetition stats (distinct 4-gram ratio of the token pieces)

Points: --chars-from curve json gives the exact chars of the 131072 and 262144 points; --scale-from-131k
derives the other depths (120K, 140K, 160K) from the 131K chars/token ratio and reports the real token count.
Same request as curve.py (REVIEW + doc[:chars]), default thinking, temperature 0, max_tokens 256, cold
(unique cache_salt per request). Order is interleaved: round r of every point before round r+1.

usage: round_replay.py --base http://localhost:8011/v1 --model glm-5.3-exl3 --roundlog RL --out out.json \
         --doc DOC.txt --chars-from curve.json [--reps 2] [--points 131072,262144,120000,140000,160000] [--reqlog-dir DIR]
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = ("Task: read the long document below, then write a very long, detailed essay (at least 800 words) "
          "reviewing its topics one by one, in plain prose, no tool calls.\n\n")


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def read_rounds(path, off):
    """New lines of the round file since byte offset off -> (list of dicts, new offset)."""
    out = []
    with open(path, "rb") as f:
        f.seek(off)
        data = f.read()
    end = data.rfind(b"\n") + 1
    for line in data[:end].splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out, off + end


def reqlog_events(d, t_from):
    """Lifecycle lines of the newest requests-*.log with a timestamp >= t_from (event name + raw line)."""
    files = sorted(glob.glob(os.path.join(d, "requests-*.log")), key=os.path.getmtime)
    if not files:
        return []
    ev = []
    for line in open(files[-1], errors="replace"):
        if any(k in line for k in (" job_start", " job_first_token", " job_done", " job_error")):
            ev.append(line.strip())
    return ev[-4:]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def distinct4(toks):
    g = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return round(len(set(g)) / len(g), 3) if g else None


def run_point(a, doc, chars, tag, off):
    body = {"model": a.model, "messages": [{"role": "user", "content": REVIEW + doc[:chars]}],
            "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "cache_prompt": False,
            "cache_salt": f"round-{tag}-{int(time.time())}" if a.salt == "cold" else "round-shared"}
    body.update(json.loads(a.think))
    chunks = []          # (t_wall, piece)
    usage = timings = None
    t_req = time.time()
    for raw in post(a.base, "/chat/completions", body):
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        pay = line[5:].strip()
        if pay == "[DONE]":
            break
        try:
            d = json.loads(pay)
        except json.JSONDecodeError:
            continue
        usage = d.get("usage") or usage
        timings = d.get("timings") or timings
        delta = (d.get("choices") or [{}])[0].get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content") or ""
        if piece:
            chunks.append((time.time(), piece))
    t_end = time.time()
    time.sleep(0.3)
    recs, off = read_rounds(a.roundlog, off)
    rounds = [r for r in recs if "ms" in r]
    comp = (usage or {}).get("completion_tokens") or len(chunks)
    tm = timings or {}
    r = {"tag": tag, "chars": chars, "prompt_tokens": (usage or {}).get("prompt_tokens"),
         "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"), "completion_tokens": comp,
         "decode_srv": tm.get("predicted_per_second"), "predicted_ms": tm.get("predicted_ms"),
         "prefill_srv": tm.get("prompt_per_second"), "prompt_ms": tm.get("prompt_ms")}
    if len(chunks) > 1:
        r["decode_cli"] = (comp - 1) / (chunks[-1][0] - chunks[0][0])
        gaps = [(chunks[i + 1][0] - chunks[i][0], i) for i in range(len(chunks) - 1)]
        gm = max(gaps)
        r["chunk_gap_max_s"], r["chunk_gap_max_at"] = round(gm[0], 3), gm[1]
        r["first_chunk_after_req_s"] = round(chunks[0][0] - t_req, 3)
        r["stream_s"] = round(chunks[-1][0] - chunks[0][0], 3)
    if rounds:
        ms = [x["ms"] for x in rounds]
        prod = [x["produced"] for x in rounds]
        accs = [x["acc"] for x in rounds if x["acc"] is not None]
        wins = [x["window"] for x in rounds if x["window"] is not None]
        t0 = rounds[0]["t"]
        tl = rounds[-1]["t"] + rounds[-1]["ms"] / 1e3
        r.update(rounds=len(rounds), tok_per_round=round(sum(prod) / len(prod), 3),
                 acc_per_round=round(sum(accs) / len(accs), 3) if accs else None,
                 window_mean=round(sum(wins) / len(wins), 3) if wins else None,
                 ms_mean=round(statistics.mean(ms), 2), ms_p50=round(statistics.median(ms), 2),
                 ms_p95=round(pct(ms, 0.95), 2), ms_max=round(max(ms), 2),
                 n_ng=sum(1 for x in rounds if x["kind"] == "ng"),
                 n_dr=sum(1 for x in rounds if x["kind"] == "dr"),
                 ms_dr=round(statistics.mean([x["ms"] for x in rounds if x["kind"] == "dr"] or [0]), 2),
                 ms_ng=round(statistics.mean([x["ms"] for x in rounds if x["kind"] == "ng"] or [0]), 2),
                 rounds_sum_s=round(sum(ms) / 1e3, 3), rounds_span_s=round(tl - t0, 3),
                 idle_s=round(tl - t0 - sum(ms) / 1e3, 3),
                 round_tps=round(sum(prod) / (sum(ms) / 1e3), 2),
                 round_first_t_minus_chunk0=round(t0 - chunks[0][0], 3) if chunks else None,
                 first_round_ms=rounds[0]["ms"], slowest5=sorted(rounds, key=lambda x: -x["ms"])[:5],
                 rounds_detail=[{k: x.get(k) for k in ("n", "produced", "window", "acc", "kind", "ms", "prof")} for x in rounds])
    text = "".join(p for _, p in chunks)
    r["text"] = text
    r["distinct4_pieces"] = distinct4([p for _, p in chunks])
    r["distinct4_words"] = distinct4(text.split())
    if a.reqlog_dir:
        r["reqlog_tail"] = reqlog_events(a.reqlog_dir, t_req)
    return r, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="glm-5.3-exl3")
    ap.add_argument("--doc", required=True, help="the text file the growing prompt is cut from")
    ap.add_argument("--chars-from", required=True, help="a served curve json (tools/glm/depth_curve.py output): its 131072 / 262144 points give the chars")
    ap.add_argument("--roundlog", required=True)
    ap.add_argument("--reqlog-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", default="131072,262144,120000,140000,160000")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prelude", default="", help="depths (tokens, from the curve json) run ONCE before the reps, recorded under "
                    "'prelude': 8192,32768,65536 reproduces the curve's order on a fresh server (tells a one-time cost from a depth cost)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--think", default="{}", help='thinking kwargs JSON; "{}" = default thinking as in the served curve')
    ap.add_argument("--salt", choices=("cold", "shared"), default="cold",
                    help="cold: unique cache_salt per request (full prefill, the served method). shared: one salt, so a repeat or a "
                         "longer prefix of an earlier point reuses its cached prompt pages (cheap decode-only replays; cached_tokens "
                         "in usage tells how much)")
    ap.add_argument("--warmup", type=int, default=1, help="discarded 8K requests first")
    a = ap.parse_args()
    a.base = a.base.rstrip("/")
    doc = open(a.doc, encoding="utf-8").read()
    ref = json.load(open(a.chars_from))["points"]
    c131 = ref["131072"]["chars"]
    ratio = c131 / ref["131072"]["usage"]["prompt_tokens"]
    pts = {}
    for t in (int(x) for x in a.points.split(",")):
        pts[t] = ref[str(t)]["chars"] if str(t) in ref else int(t * ratio)
    print("points (target tokens -> chars):", pts, flush=True)
    off = os.path.getsize(a.roundlog) if os.path.exists(a.roundlog) else 0
    res = {"argv": sys.argv, "started": time.strftime("%F %T"), "doc_chars": len(doc), "points": {str(k): [] for k in pts}}

    def save():
        json.dump(res, open(a.out + ".tmp", "w"), indent=1)
        os.replace(a.out + ".tmp", a.out)

    for w in range(a.warmup):
        _, off = run_point(a, doc, ref["8192"]["chars"], f"warm{w}", off)
    res["prelude"] = []
    for t in (int(x) for x in a.prelude.split(",") if x):
        r, off = run_point(a, doc, ref[str(t)]["chars"], f"pre{t}", off)
        res["prelude"].append(dict(r, target=t))
        save()
        print(f"prelude {t}: srv={r['decode_srv'] and round(r['decode_srv'], 1)} cli={r.get('decode_cli') and round(r['decode_cli'], 1)} "
              f"first_round_ms={r.get('first_round_ms')} rounds={r.get('rounds')} acc/rd={r.get('acc_per_round')}", flush=True)
    for rep in range(a.reps):
        for t, chars in pts.items():
            r, off = run_point(a, doc, chars, f"p{t}-r{rep}", off)
            res["points"][str(t)].append(r)
            save()
            print(f"{t} rep{rep}: tok={r['prompt_tokens']} srv={r['decode_srv'] and round(r['decode_srv'], 1)} "
                  f"cli={r.get('decode_cli') and round(r['decode_cli'], 1)} rounds={r.get('rounds')} "
                  f"acc/rd={r.get('acc_per_round')} tok/rd={r.get('tok_per_round')} ms/rd={r.get('ms_mean')} "
                  f"(p95 {r.get('ms_p95')}, max {r.get('ms_max')}) ng={r.get('n_ng')} idle={r.get('idle_s')} "
                  f"cached={r.get('cached_tokens')} first_round_ms={r.get('first_round_ms')} gap_first={r.get('round_first_t_minus_chunk0')} maxgap={r.get('chunk_gap_max_s')}@{r.get('chunk_gap_max_at')} "
                  f"d4={r.get('distinct4_words')}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
