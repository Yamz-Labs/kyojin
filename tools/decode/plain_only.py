#!/usr/bin/env python
"""Determinism check: does a fresh, plain (no draft) greedy run give the same token ids
across separate process launches with the SAME EXL3_DEC_MOE_UNION setting? If not, the
divergences seen in drafted-vs-plain comparisons may be general run-to-run nondeterminism,
not something specific to the R-row verify path."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

PROMPT = ("Write the opening three paragraphs of a short story about a lighthouse "
          "keeper who discovers a message in a bottle, in a literary, descriptive style.")
N = 10


def main():
    torch.set_grad_enabled(False)
    m = os.path.expanduser("~/models/mimo26-exl3")
    config = Config.from_directory(m)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    ids = tok.encode(PROMPT)
    job = Job(input_ids=ids, max_new_tokens=N, stop_conditions=[], sampler=GreedySampler())
    gen.enqueue(job)
    out = []
    while gen.num_remaining_jobs():
        for res in gen.iterate():
            if res.get("error"):
                raise RuntimeError(res["error"])
            tid = res.get("token_ids")
            if tid is not None:
                out.extend(int(t) for t in tid.flatten().tolist())
    print(f"PLAIN_ONLY union={os.environ.get('EXL3_DEC_MOE_UNION')} ids={out}")


if __name__ == "__main__":
    main()
