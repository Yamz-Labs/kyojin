#!/usr/bin/env python
"""Coordinator directive: at each verify128 mismatch position, what's plain's top1-top2
logit margin? margin < ~0.05 => near-tie (reduction-order numerics, acceptable). Larger
margin => real bug."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

PROMPTS = [
    ("code", "Write a Python function `is_prime(n: int) -> bool` that returns True if n "
              "is prime, using trial division up to sqrt(n). Then write a second function "
              "`primes_up_to(limit: int) -> list[int]` that uses it. Show both with docstrings.",
     20),
    ("chat", "Give me a friendly, detailed three-paragraph explanation of how a bicycle "
              "stays upright while moving, aimed at a curious 12-year-old.", 15),
    ("prose", "Write the opening three paragraphs of a short story about a lighthouse "
               "keeper who discovers a message in a bottle, in a literary, descriptive style.", 35),
]


def main():
    torch.set_grad_enabled(False)
    m = os.path.expanduser("~/models/mimo26-exl3")
    config = Config.from_directory(m)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)

    for name, prompt, pos in PROMPTS:
        ids = tok.encode(prompt)
        job = Job(input_ids=ids, max_new_tokens=pos + 2, stop_conditions=[],
                  sampler=GreedySampler(), return_logits=True)
        gen = Generator(model=model, cache=cache, tokenizer=tok)
        gen.enqueue(job)
        logits = []
        while gen.num_remaining_jobs():
            for res in gen.iterate():
                if res.get("error"):
                    raise RuntimeError(res["error"])
                if res.get("logits") is not None:
                    logits.append(res["logits"].cpu())
        L = torch.cat(logits, dim=1).float()
        top2 = L[0, pos].topk(2).values
        margin = (top2[0] - top2[1]).item()
        print(f"{name}: pos={pos} top1-top2 margin={margin:.6f} "
              f"{'NEAR-TIE' if margin < 0.05 else 'CLEAR MARGIN (real bug)'}")


if __name__ == "__main__":
    main()
