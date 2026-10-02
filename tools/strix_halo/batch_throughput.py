#!/usr/bin/env python
"""Aggregate throughput with N concurrent sequences, MTP on.

Why this can beat 47 tok/s single-stream: the measured verify cost is
    ms/forward = 32.4 + 9.04 * rows
so 47% of a 4-row forward is row-independent (dense weights + mixers, read once no matter
how many rows ride along). Batching B sequences puts B*(ndt+1) rows in ONE forward and pays
that fixed cost once, so aggregate tok/s should rise even though per-stream tok/s falls.

Reports per-stream and aggregate tok/s, plus acceptance, for each batch size.
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
NDT = int(os.environ.get("NDT", "2"))
NTOK = int(os.environ.get("NTOK", "256"))
BATCHES = [int(b) for b in os.environ.get("BATCHES", "1,2,3,4,6,8").split(",")]
CACHE = int(os.environ.get("CACHE", "16384"))

PROMPTS = [
    "Explain gradient descent in two sentences:",
    "Write a Python function that reverses a linked list:",
    "Summarize the causes of the French Revolution:",
    "What is the difference between TCP and UDP?",
    "Describe how a diesel engine differs from a petrol engine:",
    "Give three uses for a Fresnel lens:",
    "Explain why the sky is blue, briefly:",
    "What does a compiler's register allocator do?",
]


def main():
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE, max_history=NDT)
    model.load(progressbar=False)
    draft = Model.from_config(config, component="mtp")
    dcache = Cache(draft, max_num_tokens=CACHE, max_history=NDT)
    draft.load(progressbar=False)
    gen = Generator(model=model, cache=cache, tokenizer=tok,
                    draft_model=draft, draft_cache=dcache,
                    num_draft_tokens=NDT)
    print(f"ndt={NDT} ntok={NTOK} cache={CACHE}", flush=True)
    print(f"{'batch':>6} {'wall_s':>8} {'tok/s/seq':>10} {'AGGREGATE':>11} {'accept':>8} {'GiB':>6}")
    print("-" * 58)
    for B in BATCHES:
        # warm
        gen.clear_queue()
        for i in range(B):
            gen.enqueue(Job(input_ids=tok.encode(PROMPTS[i % len(PROMPTS)], add_bos=True),
                            max_new_tokens=8, sampler=GreedySampler()))
        while gen.num_remaining_jobs():
            for _ in gen.iterate(): pass
        torch.cuda.synchronize()
        gen.clear_queue()

        acc = rej = 0
        t0 = time.perf_counter()
        for i in range(B):
            gen.enqueue(Job(input_ids=tok.encode(PROMPTS[i % len(PROMPTS)], add_bos=True),
                            max_new_tokens=NTOK, sampler=GreedySampler()))
        ntok = 0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                acc += r.get("accepted_draft_tokens") or 0
                rej += r.get("rejected_draft_tokens") or 0
                if r.get("text"):
                    ntok += 1
                if r.get("error"):
                    print("JOB ERROR", r["error"], flush=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        total = B * NTOK
        a = 100 * acc / (acc + rej) if acc + rej else 0
        gib = torch.cuda.max_memory_allocated() / 2**30
        print(f"{B:>6} {wall:>8.2f} {total/B/wall:>10.1f} {total/wall:>11.1f} "
              f"{a:>7.1f}% {gib:>6.1f}", flush=True)


if __name__ == "__main__":
    main()
