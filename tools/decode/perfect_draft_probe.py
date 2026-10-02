#!/usr/bin/env python
"""Decisive test (advisor-directed): is the union=1 R-row TARGET forward itself bit-exact,
independent of the real (imperfect) drafter? Monkeypatch the drafter to always propose the
correct next tokens (read from a precomputed ground-truth plain-greedy run). With a perfect
draft, every row is accepted and the verify forward runs at full R every round with no
rollback -- if drafted still diverges from plain here, the R>1 target forward itself (most
likely attention, not yet wired for R>1) is not bit-exact. If it matches, the bug is in the
rollback/accept path instead.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402

PROMPT = ("Write the opening three paragraphs of a short story about a lighthouse "
          "keeper who discovers a message in a bottle, in a literary, descriptive style.")
N = 10


def run(generator, tok, prompt, n, debug=False):
    ids = tok.encode(prompt)
    job = Job(input_ids=ids, max_new_tokens=n, stop_conditions=[], sampler=GreedySampler())
    generator.enqueue(job)
    out = []
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            if res.get("error"):
                raise RuntimeError(res["error"])
            if debug:
                tid = res.get("token_ids")
                tid_list = tid.flatten().tolist() if tid is not None else None
                print(f"[res_debug] keys={sorted(res.keys())} token_ids={tid_list} "
                      f"eos={res.get('eos')} text={res.get('text')!r}")
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

    # Ground truth needs enough tail beyond N for the LAST round's draft window to stay fully
    # inside it (repro20 fell back to the real, imperfect draft for a tail round when this ran
    # out, which masqueraded as a target-forward bug -- it was the test's own fallback firing).
    gt_n = N + 16
    gen_plain = Generator(model=model, cache=cache_plain, tokenizer=tok)
    plain_ids_full = run(gen_plain, tok, PROMPT, gt_n)
    plain_ids = plain_ids_full[:N]
    print(f"GROUND_TRUTH plain_ids={plain_ids}")
    gt = torch.tensor(plain_ids_full, dtype=torch.long)

    gen_draft = Generator(model=model, cache=cache_draft, tokenizer=tok,
                           draft_model=draft_model, draft_cache=draft_cache)

    real_fn = gen_draft.iterate_draftmodel_dflash_gen

    def perfect_fn(results):
        real = real_fn(results)
        if real is None:
            return None
        job = gen_draft.active_jobs[0]
        start = job.new_tokens
        w = real.shape[-1]
        if start + w > gt.shape[0]:
            print(f"[perfect_fn] WARNING: fell back to real draft at start={start} w={w} "
                  f"gt_len={gt.shape[0]} -- ground truth buffer too short")
            return real  # ran past our precomputed ground truth; fall back
        patched = real.clone()
        patched[0, :w] = gt[start:start + w].to(real.device)
        return patched

    gen_draft.iterate_draftmodel_dflash_gen = perfect_fn

    draft_ids = run(gen_draft, tok, PROMPT, N, debug=True)
    print(f"PERFECT_DRAFT_RESULT draft_ids={draft_ids}")
    ok = draft_ids == plain_ids
    print("MATCH" if ok else "MISMATCH")
    if not ok:
        n = min(len(plain_ids), len(draft_ids))
        div = next((i for i in range(n) if plain_ids[i] != draft_ids[i]), n)
        print(f"first divergence at index {div}: plain={plain_ids[div:div+5]} draft={draft_ids[div:div+5]}")


if __name__ == "__main__":
    main()
