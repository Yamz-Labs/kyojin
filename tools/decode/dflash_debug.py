#!/usr/bin/env python
"""Root-cause probe for the drafted!=plain mismatch: print the draft window and each
verify row's argmax alongside ground-truth plain-greedy tokens for one short prompt."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

PROMPT = ("Write the opening three paragraphs of a short story about a lighthouse "
          "keeper who discovers a message in a bottle, in a literary, descriptive style.")
N = 12


def run(generator, tok, prompt, n):
    ids = tok.encode(prompt)
    job = Job(input_ids=ids, max_new_tokens=n, stop_conditions=[], sampler=GreedySampler())
    generator.enqueue(job)
    out = []
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            if res.get("error"):
                raise RuntimeError(res["error"])
            tid = res.get("token_ids")
            if tid is not None:
                out.extend(int(t) for t in tid.flatten().tolist())
    return out


def main():
    torch.set_grad_enabled(False)
    m = os.path.expanduser("~/models/mimo26-exl3")
    dm = os.path.expanduser("~/models/mimo26-exl3/dflash")
    config = Config.from_directory(m)
    model = Model.from_config(config)
    cache_plain = Cache(model, max_num_tokens=4096)
    cache_draft = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)
    draft_config = Config.from_directory(dm)
    draft_model = Model.from_config(draft_config)
    draft_cache = Cache(draft_model, max_num_tokens=4096)
    draft_model.load(progressbar=False)

    gen_plain = Generator(model=model, cache=cache_plain, tokenizer=tok)
    plain_ids = run(gen_plain, tok, PROMPT, N)
    print(f"GROUND_TRUTH plain_ids={plain_ids}")

    gen_draft = Generator(model=model, cache=cache_draft, tokenizer=tok,
                           draft_model=draft_model, draft_cache=draft_cache)
    os.environ["DFLASH_DEBUG"] = "1"
    draft_ids = run(gen_draft, tok, PROMPT, N)
    print(f"DRAFT_RESULT draft_ids={draft_ids}")


if __name__ == "__main__":
    main()
