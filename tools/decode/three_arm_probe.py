#!/usr/bin/env python
"""Advisor-directed 3-arm decisive test, one model load, 128 tokens on the prose prompt:
  A. Null draft: iterate_draftmodel_dflash_gen patched to return None (drafter never
     proposes, target never verifies a window) but draft_verifier_params (export_state_layers
     etc.) still gets merged into every forward's params, exactly as a real drafted run does.
     If A mismatches plain, an export-path kernel difference is the cause -- nothing about
     the R-row verify kernels is implicated.
  B. Perfect draft at 128 tokens: drafter patched to always propose the ground-truth
     continuation. If A matches but B mismatches, the R-row verify forward itself (attention
     at q_len>1 included) is not bit-exact.
  C. Perfect draft with one deliberately wrong token per round (forces a mid-window reject
     every round): if A and B match but C mismatches, the rollback/rewind path
     (reject_remainder, recurrent state rewind) is implicated.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

PROMPT = ("Write the opening three paragraphs of a short story about a lighthouse "
          "keeper who discovers a message in a bottle, in a literary, descriptive style.")
N = 128


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


def report(name, ids, plain_ids):
    ok = ids == plain_ids
    if ok:
        print(f"{name}: MATCH ({len(ids)} tokens)")
    else:
        n = min(len(ids), len(plain_ids))
        div = next((i for i in range(n) if ids[i] != plain_ids[i]), n)
        print(f"{name}: MISMATCH at token {div} plain={plain_ids[div:div+5]} got={ids[div:div+5]}")
    return ok


def main():
    torch.set_grad_enabled(False)
    m = os.path.expanduser("~/models/mimo26-exl3")
    dm = os.path.expanduser("~/models/mimo26-exl3/dflash")
    config = Config.from_directory(m)
    model = Model.from_config(config)
    # All Cache instances for this model must be constructed before model.load() allocates
    # its tensors (Generator asserts on this).
    cache_plain = Cache(model, max_num_tokens=4096)
    cache_a = Cache(model, max_num_tokens=4096)
    cache_b = Cache(model, max_num_tokens=4096)
    cache_c = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    tok = Tokenizer.from_config(config)
    draft_config = Config.from_directory(dm)
    draft_model = Model.from_config(draft_config)
    draft_cache = Cache(draft_model, max_num_tokens=4096)
    draft_model.load(progressbar=False)

    gen_plain = Generator(model=model, cache=cache_plain, tokenizer=tok)
    plain_ids = run(gen_plain, tok, PROMPT, N)
    print(f"GROUND_TRUTH plain_ids={plain_ids}")
    gt = torch.tensor(plain_ids, dtype=torch.long)

    # Arm A: null draft, export params still applied every forward
    gen_a = Generator(model=model, cache=cache_a, tokenizer=tok,
                       draft_model=draft_model, draft_cache=draft_cache)
    gen_a.iterate_draftmodel_dflash_gen = lambda results: None
    a_ids = run(gen_a, tok, PROMPT, N)
    a_ok = report("A (null draft, export params on)", a_ids, plain_ids)

    # Arm B: perfect draft, full 128 tokens
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
            print(f"[arm B] WARNING: ground truth too short at start={start} w={w}")
            return real
        patched = real.clone()
        patched[0, :w] = gt[start:start + w].to(real.device)
        return patched

    gen_b.iterate_draftmodel_dflash_gen = perfect_fn
    b_ids = run(gen_b, tok, PROMPT, N)
    b_ok = report("B (perfect draft, 128 tok)", b_ids, plain_ids)

    # Arm C: perfect draft but corrupt one token per round, forcing a mid-window reject
    gen_c = Generator(model=model, cache=cache_c, tokenizer=tok,
                       draft_model=draft_model, draft_cache=draft_cache)
    real_fn_c = gen_c.iterate_draftmodel_dflash_gen

    def corrupt_fn(results):
        real = real_fn_c(results)
        if real is None:
            return None
        job = gen_c.active_jobs[0]
        start = job.new_tokens
        w = real.shape[-1]
        if start + w > gt.shape[0]:
            print(f"[arm C] WARNING: ground truth too short at start={start} w={w}")
            return real
        patched = real.clone()
        patched[0, :w] = gt[start:start + w].to(real.device)
        if w > 3:
            # Corrupt position 3: forces a reject at row index 3 every round (never row 0,
            # so rows 0-2 still exercise the accept path before the reject/rewind fires)
            bad = int(patched[0, 3].item()) + 1
            patched[0, 3] = bad
        return patched

    gen_c.iterate_draftmodel_dflash_gen = corrupt_fn
    c_ids = run(gen_c, tok, PROMPT, N)
    c_ok = report("C (perfect draft + forced reject every round)", c_ids, plain_ids)

    print(f"\nSUMMARY: A={'OK' if a_ok else 'FAIL'} B={'OK' if b_ok else 'FAIL'} "
          f"C={'OK' if c_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
