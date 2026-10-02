#!/usr/bin/env python
"""Check: drafted greedy == plain greedy over 128 tokens, 3 prompts
(code/chat/prose). stop_conditions=[] so every run reaches exactly max_new_tokens
regardless of EOS -- token IDs, not text, decide the comparison. One model load
covers all 6 runs (3 prompts x plain/drafted).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

PROMPTS = [
    ("code", "Write a Python function `is_prime(n: int) -> bool` that returns True if n "
              "is prime, using trial division up to sqrt(n). Then write a second function "
              "`primes_up_to(limit: int) -> list[int]` that uses it. Show both with docstrings."),
    ("chat", "Give me a friendly, detailed three-paragraph explanation of how a bicycle "
              "stays upright while moving, aimed at a curious 12-year-old."),
    ("prose", "Write the opening three paragraphs of a short story about a lighthouse "
               "keeper who discovers a message in a bottle, in a literary, descriptive style."),
]


def run(generator: Generator, tok: Tokenizer, prompt: str, n: int) -> list[int]:
    ids = tok.encode(prompt)
    job = Job(input_ids=ids, max_new_tokens=n, stop_conditions=[], sampler=GreedySampler())
    generator.enqueue(job)
    out_ids: list[int] = []
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            if res.get("error"):
                raise RuntimeError(res["error"])
            tid = res.get("token_ids")
            if tid is not None:
                out_ids.extend(int(t) for t in tid.flatten().tolist())
    return out_ids


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-m", "--model_dir", default=os.path.expanduser("~/models/mimo26-exl3"))
    ap.add_argument("-dm", "--draft_model_dir", default=os.path.expanduser("~/models/mimo26-exl3/dflash"))
    ap.add_argument("--new-tokens", type=int, default=128)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    # Separate Cache per Generator: a Generator takes exclusive ownership of the Cache it's
    # built with (a newer Generator over the same Cache invalidates the older one), and we need
    # both the plain and drafted generators alive/usable across the whole prompt loop.
    cache_plain = Cache(model, max_num_tokens=4096)
    cache_draft = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)

    draft_config = Config.from_directory(args.draft_model_dir)
    draft_model = Model.from_config(draft_config)
    draft_cache = Cache(draft_model, max_num_tokens=4096)
    draft_model.load(progressbar=False)

    gen_plain = Generator(model=model, cache=cache_plain, tokenizer=tok)
    gen_draft = Generator(model=model, cache=cache_draft, tokenizer=tok,
                           draft_model=draft_model, draft_cache=draft_cache)

    all_ok = True
    for name, prompt in PROMPTS:
        plain_ids = run(gen_plain, tok, prompt, args.new_tokens)
        draft_ids = run(gen_draft, tok, prompt, args.new_tokens)
        ok = plain_ids == draft_ids
        all_ok &= ok
        if ok:
            print(f"{name}: OK  ({len(plain_ids)} tokens match)")
        else:
            n = min(len(plain_ids), len(draft_ids))
            div = next((i for i in range(n) if plain_ids[i] != draft_ids[i]), n)
            print(f"{name}: MISMATCH at token {div} "
                  f"(plain={plain_ids[div:div+5] if div < len(plain_ids) else '<end>'}, "
                  f"draft={draft_ids[div:div+5] if div < len(draft_ids) else '<end>'})")
            print(f"  plain len={len(plain_ids)} draft len={len(draft_ids)}")

    print("ALL_OK" if all_ok else "SOME_MISMATCH")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
