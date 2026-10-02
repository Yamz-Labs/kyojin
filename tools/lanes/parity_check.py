#!/usr/bin/env python3
"""Greedy parity + speed check for a running lane server (GLM or MiMo).

  parity_check.py save --port 18091 --model-id ID --out run_a.json
  parity_check.py compare run_a.json run_b.json [--min-speed-ratio 0.97]

save: sends fixed prompts at temperature 0 (non-streaming) and stores the returned text, reasoning, tool calls
      and the decode speed (completion tokens / wall seconds, after one warm-up request).
compare: exit 0 only when every output is identical and run_b is at least --min-speed-ratio of run_a's speed.
"""
import argparse, json, sys, time, urllib.request

PROMPTS = [
    "Write a Python function that merges two sorted lists, with a short explanation.",
    "Explain in four sentences why the sky is blue.",
    "List five prime numbers above 1000 and show how you checked one of them.",
    "Translate to French: The committee postponed the decision until further notice.",
]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                         "required": ["city"]}}}]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def save(a):
    cases = [{"prompt": p, "tools": None} for p in PROMPTS]
    cases.append({"prompt": "What is the weather in Paris for 3 days? Use the tool.", "tools": TOOLS})
    post(a.port, {"model": a.model_id, "messages": [{"role": "user", "content": "Say hello."}],
                  "max_tokens": 16, "temperature": 0})  # warm-up
    runs, toks, wall = [], 0, 0.0
    for c in cases:
        body = {"model": a.model_id, "messages": [{"role": "user", "content": c["prompt"]}],
                "max_tokens": a.max_tokens, "temperature": 0, "top_p": 1.0}
        if c["tools"]:
            body["tools"] = c["tools"]
        out, dt = post(a.port, body)
        msg = out["choices"][0]["message"]
        n = out["usage"]["completion_tokens"]
        toks += n; wall += dt
        runs.append({"prompt": c["prompt"], "content": msg.get("content"), "reasoning": msg.get("reasoning_content"),
                     "tool_calls": msg.get("tool_calls"), "finish": out["choices"][0].get("finish_reason"),
                     "completion_tokens": n, "wall_s": round(dt, 3)})
    json.dump({"tok_per_s": toks / wall, "completion_tokens": toks, "wall_s": wall, "runs": runs},
              open(a.out, "w"), indent=1)
    print(f"saved {a.out}: {toks} tokens, {toks / wall:.2f} tok/s")


def strip_ids(calls):
    return [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in calls or []]


def compare(a):
    x, y = (json.load(open(p)) for p in (a.a, a.b))
    bad = 0
    for i, (p, q) in enumerate(zip(x["runs"], y["runs"])):
        same = all(p[k] == q[k] for k in ("content", "reasoning", "finish", "completion_tokens")) \
            and strip_ids(p["tool_calls"]) == strip_ids(q["tool_calls"])
        print(f"case {i}: {'IDENTICAL' if same else 'DIFFERENT'} ({p['completion_tokens']} vs {q['completion_tokens']} tokens)")
        bad += not same
    ratio = y["tok_per_s"] / x["tok_per_s"]
    print(f"speed: a {x['tok_per_s']:.2f} tok/s, b {y['tok_per_s']:.2f} tok/s, ratio {ratio:.3f} (min {a.min_speed_ratio})")
    ok = bad == 0 and ratio >= a.min_speed_ratio
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save"); s.add_argument("--port", type=int, default=18091)
    s.add_argument("--model-id", required=True); s.add_argument("--out", required=True)
    s.add_argument("--max-tokens", type=int, default=256); s.set_defaults(fn=save)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    c.add_argument("--min-speed-ratio", type=float, default=0.97); c.set_defaults(fn=compare)
    a = ap.parse_args(); a.fn(a)
