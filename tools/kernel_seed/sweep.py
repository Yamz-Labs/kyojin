#!/usr/bin/env python3
"""Drive a running server through the request shapes a user reaches, so that every Triton kernel variant gets compiled
(used to build the kernel seed, see build.py). Greedy and sampled, prompt lengths from a few tokens to several chunks,
prose / code / digits (different speculative acceptance, so different verify lengths), plus a repeated prompt (cache hit)."""
import argparse, json, random, urllib.request

LENGTHS = (3, 12, 33, 70, 130, 300, 700, 1500, 3000, 4200, 4800, 6500, 9000)
SEEDS = ("The history of the Roman empire, told as a story. ", "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
         "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ", "Explain in detail how a transformer language model works. ")


def post(url, model, content, max_tokens, temperature, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.95 if temperature else 1.0}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8731")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=96)
    a = ap.parse_args()
    rnd = random.Random(0)
    n = 0
    for i, ln in enumerate(LENGTHS):
        for j, seed in enumerate(SEEDS):
            words = (seed * (ln // max(len(seed.split()), 1) + 1)).split()[:ln]
            text = " ".join(words) + "\nContinue."
            for temp in (0.0, 0.8):
                post(a.url, a.model, text, a.max_tokens, temp)
                n += 1
    post(a.url, a.model, "Say hi.", 1, 0.0)
    post(a.url, a.model, "Say hi.", 1, 0.0)
    print(f"{n + 2} requests done")


if __name__ == "__main__":
    main()
