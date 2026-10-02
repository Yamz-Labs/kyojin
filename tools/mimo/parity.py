#!/usr/bin/env python3
"""msrv parity: the greedy text the server returns must equal the plain-greedy reference from
plain_greedy.py, token for token, for the first 64 tokens.

Two phases, because the box cannot hold two full MiMo loads at once:
  --phase save     (server up)   POST max_tokens=64 greedy, store ids + text
  --phase compare  (server down) compare against scratch/msrv/plain_code.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

MODEL = "MiMo-2.6-EXL3"
N = 64


def post(port: int, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--phase", default="save", choices=["save", "compare"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=os.path.expanduser("~/models/mimo26-exl3"))
    ap.add_argument("--kind", default="code", choices=["code", "chat"])
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--served", default="scratch/msrv/served_code.json")
    ap.add_argument("--plain", default="scratch/msrv/plain_code.json")
    args = ap.parse_args()

    if args.phase == "save":
        from accept import chat_prompt, code_prompt, load_corpus, CODE_CORPUS
        from exllamav3 import Config, Tokenizer
        tokenizer = Tokenizer.from_config(Config.from_directory(args.model))
        if args.kind == "chat":
            messages = chat_prompt(load_corpus(os.path.expanduser("~/bench/ppl/wiki.test.raw")),
                                   tokenizer, args.offset)
        else:
            messages = [{"role": "user", "content": code_prompt(load_corpus(CODE_CORPUS),
                                                                tokenizer, args.offset)}]
        body = post(args.port, {"model": MODEL, "messages": messages, "max_tokens": N,
                                "temperature": 0.0, "stream": False})
        # The MiMo template's generation prompt ends with an OPEN "<think>", so a 64-token greedy
        # continuation is all reasoning: it arrives as reasoning_content, and content is "". The
        # plain_greedy reference is the raw decoded text (think block included), so rebuild it.
        msg = body["choices"][0]["message"]
        reasoning = msg.get("reasoning_content") or ""
        text = (f"<think>{reasoning}</think>" if reasoning else "") + (msg.get("content") or "")
        encoded = tokenizer.encode(text, add_bos=False, encode_special_tokens=False)
        ids_t = encoded[0] if isinstance(encoded, tuple) else encoded
        ids = (ids_t[0] if ids_t.dim() > 1 else ids_t).tolist()
        out = dict(kind=args.kind, offset=args.offset, text=text, ids=ids,
                   prompt_tokens=body["usage"]["prompt_tokens"])
        os.makedirs(os.path.dirname(args.served) or ".", exist_ok=True)
        json.dump(out, open(args.served, "w"), indent=1)
        print(f"saved {len(ids)} tok, prompt_tokens={out['prompt_tokens']} -> {args.served}")
        return 0

    served = json.load(open(args.served))
    plain = json.load(open(args.plain))
    a, b = served["ids"][:N], plain["ids"][:N]
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    result = dict(kind=args.kind, n_served=len(a), n_plain=len(b), first_diff=first,
                  n_diff=sum(1 for x, y in zip(a, b) if x != y),
                  identical_64=bool(first is None and len(a) >= N and len(b) >= N))
    json.dump(result, open("scratch/msrv/parity.json", "w"), indent=1)
    print(json.dumps(result, indent=1))
    if not result["identical_64"]:
        print("MISMATCH -- served text and plain greedy differ")
        print("served:", repr(served["text"][:300]))
        print("plain :", repr(plain["text"][:300]))
        return 1
    print("parity OK: first 64 greedy tokens identical (DFlash+SpecGate == plain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
