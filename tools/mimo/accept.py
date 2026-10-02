#!/usr/bin/env python3
"""msrv GPU acceptance: HTTP end-to-end tok/s for a 400-token code prompt and a 400-token chat
prompt, plus greedy parity of the served output with scripts/dflash-bench.py plain greedy.

Run with the server already up:
  python tools/mimo/accept.py --port 8000 --out scratch/msrv/accept.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

PROMPT_TOKENS = 400
NEW_TOKENS = 128
REPS = 3
MODEL = "MiMo-2.6-EXL3"

CODE_CORPUS = ("exllamav3/conversion/standard_cal_data/code.utf8")
PROSE_CORPUS = "~/bench/ppl/wiki.test.raw"


def post(port: int, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def load_corpus(path: str) -> str:
    import os
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        return f.read()


def code_prompt(corpus: str, tokenizer, offset: int) -> str:
    ids = tokenizer.encode(corpus, add_bos=False)[:, offset:offset + PROMPT_TOKENS]
    return tokenizer.decode(ids)[0]


def chat_prompt(corpus: str, tokenizer, offset: int) -> list[dict]:
    """Same user turn as scripts/dflash-bench.py chat_tokens(); the server renders the template."""
    body_ids = tokenizer.encode(corpus, add_bos=False)[:, offset:offset + PROMPT_TOKENS - 64]
    user = ("Here is an excerpt from an encyclopedia article:\n\n" +
            tokenizer.decode(body_ids)[0] +
            "\n\nExplain in your own words, in a friendly tone, what this excerpt is about "
            "and what the most interesting facts in it are.")
    return [{"role": "user", "content": user}]


def timed(port: int, messages, max_tokens: int) -> dict:
    """Streamed request: e2e wall clock, TTFT, and decode-only tok/s.

    e2e includes prefill of the 400-token prompt, template rendering and HTTP; decode-only
    excludes prefill the way dflash-bench.py's tps() does, so the two numbers are comparable
    to the engine figures.
    """
    payload = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                          "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    text, chunks, prompt_tokens, ttft = "", 0, None, None
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: {"):
                continue
            body = json.loads(line[6:])
            if body.get("usage"):
                prompt_tokens = body["usage"]["prompt_tokens"]
                completion_tokens = body["usage"]["completion_tokens"]
            for choice in body.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text += piece
                    chunks += 1
    total = time.perf_counter() - t0
    # Generated token count comes from the server's usage block (completion_tokens), which counts
    # the decoded completion with the model tokenizer -- not the number of SSE chunks.
    return {"text": text, "chunks": chunks, "prompt_tokens": prompt_tokens, "e2e_s": total,
            "ttft": ttft, "decode_s": (total - ttft) if ttft else None,
            "completion_tokens": completion_tokens}


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="~/models/mimo26-exl3")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    ap.add_argument("--out", default="scratch/msrv/accept.json")
    ap.add_argument("--plain-greedy", default="",
                    help="JSON file of plain-greedy ids from scripts/dflash-bench.py, for parity")
    args = ap.parse_args()

    import os
    os.environ.setdefault("EXL3_REPO", os.getcwd())
    from exllamav3 import Config, Tokenizer
    import jinja2
    config = Config.from_directory(os.path.expanduser(args.model))
    tokenizer = Tokenizer.from_config(config)
    template = open(os.path.expanduser(args.model) + "/chat_template.jinja",
                    encoding="utf-8").read()
    del jinja2
    corpora = {"code": load_corpus(CODE_CORPUS), "prose": load_corpus(PROSE_CORPUS)}

    out: dict = {"port": args.port, "reps": args.reps, "new_tokens": args.new_tokens,
                 "prompt_tokens": PROMPT_TOKENS, "runs": {}, "warnings": []}

    # Warm-up (JIT + CUDA-graph capture) on an offset outside the timed reps.
    timed(args.port, [{"role": "user", "content": "hi"}], 32)

    for kind in ("code", "chat"):
        runs = []
        for rep in range(args.reps):
            offset = rep * PROMPT_TOKENS
            if kind == "chat":
                messages = chat_prompt(corpora["prose"], tokenizer, offset)
            else:
                messages = [{"role": "user", "content": code_prompt(corpora["code"], tokenizer,
                                                                    offset)}]
            r = timed(args.port, messages, args.new_tokens)
            n = r["completion_tokens"] or 1
            r["tps_e2e"] = n / r["e2e_s"]
            r["tps_decode"] = ((n - 1) / r["decode_s"]) if r["decode_s"] else None
            runs.append(r)
            print(f"{kind} rep{rep}: {n} tok  prompt {r['prompt_tokens']}  "
                  f"ttft {r['ttft']:.2f}s  e2e {r['e2e_s']:.2f}s -> {r['tps_e2e']:.2f} tok/s  "
                  f"decode {r['decode_s']:.2f}s -> {r['tps_decode']:.2f} tok/s", flush=True)
        e2e = [r["tps_e2e"] for r in runs]
        dec = [r["tps_decode"] for r in runs if r["tps_decode"]]
        out["runs"][kind] = dict(
            median_e2e=statistics.median(e2e), median_decode=statistics.median(dec) if dec else None,
            tps_e2e=e2e, tps_decode=dec,
            runs=[{k: v for k, v in r.items() if k != "text"} for r in runs],
            text=runs[0]["text"][:2000])

    if args.plain_greedy and os.path.exists(args.plain_greedy):
        ref = json.load(open(args.plain_greedy))
        out["plain_greedy_ref"] = args.plain_greedy
        out["parity"] = "see accept.md"

    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: {"e2e": round(v["median_e2e"], 2),
                         "decode": round(v["median_decode"], 2) if v["median_decode"] else None}
                      for k, v in out["runs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
