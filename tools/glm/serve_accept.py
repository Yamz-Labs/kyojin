"""HTTP acceptance bench for an OpenAI-compatible EXL3 server (serve.py).

Prefill: unique wikitext slices (no prefix-cache hits), max_tokens=1, t/s = prompt_tokens / wall.
Decode: streamed greedy generation, t/s = (completion_tokens - 1) / (last chunk - first chunk).

usage: serve_accept.py --base http://127.0.0.1:18080/v1 --model glm-5.3-exl3 --out accept.json
"""
import argparse, json, random, statistics, time, urllib.request
from pathlib import Path

CORPUS = Path.home() / "bench/ppl/wiki.test.raw"
DECODE_PROMPTS = {
    "prose": "Write a 300-word story about a lighthouse.",
    "chat": "Explain how a hash map works, with its time complexity, in about 300 words.",
    "code": "Write a Python function that parses an ISO-8601 duration string into seconds, with tests.",
}


def post(base, body, stream=False):
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def prefill(base, model, words, rng):
    text = CORPUS.read_text(encoding="utf-8").split()
    start = rng.randrange(0, len(text) - words)
    prompt = f"[{rng.random()}] " + " ".join(text[start:start + words]) + "\nSummarize in one word."
    t0 = time.perf_counter()
    d = json.load(post(base, {"model": model, "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": 1, "temperature": 0}))
    wall = time.perf_counter() - t0
    n = d["usage"]["prompt_tokens"]
    return {"prompt_tokens": n, "wall_s": round(wall, 3), "tps": round(n / wall, 1)}


def decode(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    usage = None
    for raw in post(base, body):
        line = raw.decode().strip()
        if not line.startswith("data: {"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        delta = (d.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            now = time.perf_counter()
            first = first or now
            last = now
            chunks += 1
    toks = (usage or {}).get("completion_tokens") or chunks
    return {"completion_tokens": toks, "ttft_s": round(first - t0, 3),
            "tps": round((toks - 1) / (last - first), 2) if last and last > first else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--model", default="glm-5.3-exl3")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(1234)
    res = {"prefill": {}, "decode": {}}
    prefill(a.base, a.model, 300, rng)  # warm-up
    for label, words in (("4k", 3000), ("16k", 12000)):
        runs = [prefill(a.base, a.model, words, rng) for _ in range(a.reps)]
        res["prefill"][label] = {"runs": runs, "median_tps": statistics.median(r["tps"] for r in runs)}
        print("prefill", label, res["prefill"][label]["median_tps"], [r["prompt_tokens"] for r in runs], flush=True)
    for label, prompt in DECODE_PROMPTS.items():
        runs = [decode(a.base, a.model, prompt, a.max_tokens) for _ in range(a.reps)]
        res["decode"][label] = {"runs": runs, "median_tps": statistics.median(r["tps"] for r in runs)}
        print("decode", label, res["decode"][label]["median_tps"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
