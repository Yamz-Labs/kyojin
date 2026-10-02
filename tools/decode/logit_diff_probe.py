#!/usr/bin/env python
"""Advisor-directed: bitwise per-position logit diff between plain and a perfect-draft
verify run, correlated with (round, row index) via DFLASH_DEBUG. Finds the first position
with any nonzero delta, and whether row 0 of every round is exact (attention/rope at
q_len>1 suspect) or not (some other cause)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

PROMPT = ("Write the opening three paragraphs of a short story about a lighthouse "
          "keeper who discovers a message in a bottle, in a literary, descriptive style.")
N = 40
GT_N = N + 16


def run(generator, tok, prompt, n):
    ids = tok.encode(prompt)
    job = Job(input_ids=ids, max_new_tokens=n, stop_conditions=[], sampler=GreedySampler(),
              return_logits=True)
    generator.enqueue(job)
    toks, logits = [], []
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            if res.get("error"):
                raise RuntimeError(res["error"])
            if res.get("token_ids") is not None:
                toks.append(res["token_ids"].cpu())
            if res.get("logits") is not None:
                logits.append(res["logits"].cpu())
    t = torch.cat(toks, dim=-1).flatten() if toks else torch.empty(0, dtype=torch.long)
    L = torch.cat(logits, dim=1).float() if logits else None
    return t.tolist(), L


def main():
    torch.set_grad_enabled(False)
    m = os.path.expanduser("~/models/mimo26-exl3")
    dm = os.path.expanduser("~/models/mimo26-exl3/dflash")
    config = Config.from_directory(m)
    model = Model.from_config(config)
    cache_plain = Cache(model, max_num_tokens=4096)
    cache_b = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)
    draft_config = Config.from_directory(dm)
    draft_model = Model.from_config(draft_config)
    draft_cache = Cache(draft_model, max_num_tokens=4096)
    draft_model.load(progressbar=False)

    gen_plain = Generator(model=model, cache=cache_plain, tokenizer=tok)
    plain_ids, plain_logits = run(gen_plain, tok, PROMPT, GT_N)
    print(f"GROUND_TRUTH plain_ids[:{N}]={plain_ids[:N]}")
    gt = torch.tensor(plain_ids, dtype=torch.long)

    gen_b = Generator(model=model, cache=cache_b, tokenizer=tok,
                       draft_model=draft_model, draft_cache=draft_cache)
    real_fn_b = gen_b.iterate_draftmodel_dflash_gen

    def perfect_fn(results):
        real = real_fn_b(results)
        if real is None:
            return None
        job = gen_b.active_jobs[0]
        start = job.new_tokens
        w = real.shape[-1]
        if start + w > gt.shape[0]:
            return real
        patched = real.clone()
        patched[0, :w] = gt[start:start + w].to(real.device)
        return patched

    gen_b.iterate_draftmodel_dflash_gen = perfect_fn
    os.environ["DFLASH_DEBUG"] = "1"
    b_ids, b_logits = run(gen_b, tok, PROMPT, N)
    os.environ["DFLASH_DEBUG"] = "0"

    V = min(plain_logits.shape[-1], b_logits.shape[-1])
    P = min(plain_logits.shape[1], b_logits.shape[1])
    la = plain_logits[0, :P, :V]
    lb = b_logits[0, :P, :V]
    print(f"NaN count: plain={torch.isnan(la).sum().item()} draft={torch.isnan(lb).sum().item()}")
    print(f"-inf count: plain={torch.isneginf(la).sum().item()} draft={torch.isneginf(lb).sum().item()}")
    # Bitwise int32 compare: immune to NaN/-inf (padded vocab columns can be -inf in both runs,
    # and -inf - -inf is NaN under plain subtraction, which broke the earlier float-diff version)
    bitdiff = la.view(torch.int32) != lb.view(torch.int32)
    per_pos_any = bitdiff.any(dim=-1)
    print(f"\npositions compared: {P}")
    first_nonzero = next((i for i in range(P) if per_pos_any[i]), None)
    print(f"first bitwise-differing position: {first_nonzero}")
    for i in range(P):
        if per_pos_any[i]:
            n = bitdiff[i].sum().item()
            finite = ~(torch.isnan(la[i]) | torch.isnan(lb[i]) | torch.isinf(la[i]) | torch.isinf(lb[i]))
            maxd = (la[i][finite] - lb[i][finite]).abs().max().item() if finite.any() else float("nan")
            print(f"  pos {i:3d}: {n} logits differ (bitwise), max|dlogit| over finite ones = {maxd:.6f}")
    print(f"positions with any bitwise diff: {per_pos_any.sum().item()} / {P}")


if __name__ == "__main__":
    main()
