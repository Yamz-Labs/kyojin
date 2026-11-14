#!/usr/bin/env python3
"""Concurrent-session smoke for a served model (split or single device).

Starts no server: point it at an already-running OpenAI-compatible endpoint.  The
point of the Stage 2 check is that the split still serves several requests through
one Generator: the requests must overlap, all return 200, and the server's metrics
must count every completion.  Overlap is checked from the accumulated prompt/predict
seconds in `/metrics`; the wall time alone cannot distinguish concurrency from a
queue, because a queued request's wall includes its wait.

    python tools/glm/serve.py --sessions 4 --num-draft 2 --part0-layers 4 ...
    python tools/strix_halo/split_sessions_smoke.py --port 8299 --sessions 4 \
        --max-tokens 128 --out /tmp/sessions.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request

import aiohttp

PROMPTS = [
    "Name three primary colours and explain where each appears in a rainbow.",
    "Write a Python function that returns the nth Fibonacci number, with a docstring.",
    "A bus leaves at 09:10 and arrives at 11:55. How long is the trip? Show the work.",
    "Summarize the water cycle in four short sentences.",
    "List the first five prime numbers and say why 1 is not prime.",
    "Explain the difference between RAM and a hard disk in three sentences.",
    "Write a haiku about winter, then explain the imagery in one sentence.",
    "What is the capital of Japan, and what is one well-known landmark there?",
]


def metrics(port: int) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout = 5) as r:
            return r.read().decode("utf8", "replace")
    except Exception as exc:                        # noqa: BLE001
        return f"metrics unavailable: {type(exc).__name__}: {exc}"


def parse_metrics(text: str) -> dict[str, float]:
    """Numeric sample values from a Prometheus text body (comments ignored)."""
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if not name or not value:
            continue
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


# The server accumulates one projection/prefill time per request.  If two or more
# requests ran at once those sums overlap and exceed the wall clock; a queue gives a
# ratio near 1.0, so 1.3 is a safe floor for "these requests overlapped".
MIN_OVERLAP_RATIO = 1.3


async def one(session, url, model, prompt, max_tokens, index, timeout_s):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json = body,
                                timeout = aiohttp.ClientTimeout(total = timeout_s)) as r:
            j = await r.json()
            dt = time.perf_counter() - t0
            if r.status != 200:
                return {"index": index, "status": r.status, "wall_s": dt,
                        "error": str(j)[:400]}
            u = j.get("usage", {})
            msg = (j.get("choices") or [{}])[0].get("message", {})
            return {
                "index": index,
                "status": r.status,
                "wall_s": round(dt, 3),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens")
                                 if isinstance(u.get("prompt_tokens_details"), dict) else None,
                "text": ((msg.get("content") or "") + (msg.get("reasoning_content") or ""))[:160],
                "finish_reason": (j.get("choices") or [{}])[0].get("finish_reason"),
            }
    except Exception as exc:                        # noqa: BLE001
        return {"index": index, "status": None, "wall_s": round(time.perf_counter() - t0, 3),
                "error": f"{type(exc).__name__}: {exc}"}


async def run(args):
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    prompts = (PROMPTS * ((args.sessions + len(PROMPTS) - 1) // len(PROMPTS)))[:args.sessions]
    before = metrics(args.port)
    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            one(session, url, args.model, p, args.max_tokens, i, args.timeout)
            for i, p in enumerate(prompts)
        ])
        wall = time.perf_counter() - t0
    after = metrics(args.port)
    before_m = parse_metrics(before)
    after_m = parse_metrics(after)
    ok = sum(1 for r in results if r.get("status") == 200)
    completion_total = sum(int(r.get("completion_tokens") or 0) for r in results)

    def delta(key):
        return max(0.0, after_m.get(key, 0.0) - before_m.get(key, 0.0))

    prompt_tokens = delta("llamacpp:prompt_tokens_total")
    predicted_tokens = delta("llamacpp:tokens_predicted_total")
    prompt_s = delta("llamacpp:prompt_seconds_total")
    predict_s = delta("llamacpp:tokens_predicted_seconds_total")
    server_work_s = prompt_s + predict_s
    overlap_ratio = server_work_s / wall if wall > 0 else 0.0
    result = {
        "port": args.port,
        "sessions": args.sessions,
        "max_tokens": args.max_tokens,
        "wall_s": round(wall, 3),
        "ok": ok,
        "total": len(results),
        "results": results,
        "metrics_before": before,
        "metrics_after": after,
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "predicted_tokens": predicted_tokens,
            "prompt_seconds": round(prompt_s, 3),
            "predicted_seconds": round(predict_s, 3),
            "server_work_s": round(server_work_s, 3),
            "overlap_ratio": round(overlap_ratio, 2),
            "completion_tokens_reported": completion_total,
        },
    }
    print(f"[sessions] {ok}/{len(results)} requests returned 200 in {wall:.2f} s "
          f"(sum of request walls {sum(r.get('wall_s') or 0 for r in results):.2f} s)", flush = True)
    print(f"[sessions] server work {server_work_s:.2f} s vs wall {wall:.2f} s "
          f"(overlap {overlap_ratio:.2f}x); metrics counted {predicted_tokens:.0f} predicted "
          f"tokens, responses reported {completion_total}", flush = True)
    for r in results:
        print(f"[sessions]   {r['index']}: status {r.get('status')} "
              f"wall {r.get('wall_s')} s tokens {r.get('completion_tokens')} "
              f"finish {r.get('finish_reason')}"
              + (f" error {r['error']}" if r.get("error") else ""), flush = True)
    if args.out:
        with open(args.out, "w", encoding = "utf8") as f:
            json.dump(result, f, indent = 1)
        print(f"[sessions] wrote {args.out}", flush = True)
    if ok != len(results):
        raise SystemExit(f"session smoke failed: {ok}/{len(results)} requests returned 200")
    if not after_m:
        raise SystemExit("session smoke failed: /metrics was unavailable or empty")
    if predicted_tokens < completion_total:
        raise SystemExit(f"session smoke failed: metrics counted {predicted_tokens:.0f} predicted "
                         f"tokens but responses reported {completion_total}")
    if args.sessions > 1 and overlap_ratio < MIN_OVERLAP_RATIO:
        raise SystemExit(f"session smoke failed: requests did not overlap "
                         f"(server work {server_work_s:.2f} s vs wall {wall:.2f} s, "
                         f"ratio {overlap_ratio:.2f} < {MIN_OVERLAP_RATIO})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type = int, default = 8299)
    ap.add_argument("--model", default = "glm-5.3-exl3")
    ap.add_argument("--sessions", type = int, default = 4)
    ap.add_argument("--max-tokens", type = int, default = 128)
    ap.add_argument("--timeout", type = float, default = 1200.0)
    ap.add_argument("--out", default = None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
