#!/usr/bin/env python
"""Deterministic A/B: greedy decode, same prompt, fast WMMA vs reconstruct.

Token IDs must match EXACTLY. Also exercises a long prompt to push the
multi-row (MMODE2) epilogue path that short decodes never touch.
"""
import os, sys, json, time
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
mode = os.environ.get("EXL3_GEMV", "1")
print(f"EXL3_GEMV={mode} family={exllamav3_ext.exl3_gemv_wmma_family(0)}", flush=True)

config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=4096)
model.load(progressbar=False)

PROMPTS = [
    "The capital of France is",
    # long prompt -> multi-row prefill, exercises MMODE2
    ("Summarize the following in one sentence. " + 
     "Gradient descent is an iterative optimization algorithm used to minimize a "
     "differentiable objective function by repeatedly stepping in the direction of "
     "steepest descent as defined by the negative of the gradient. " * 8),
]

gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
out = {}
for pi, p in enumerate(PROMPTS):
    ids = tokenizer.encode(p, add_bos=True)
    job = Job(input_ids=ids, max_new_tokens=24, sampler=GreedySampler())
    gen.enqueue(job)
    toks, text = [], ""
    t0 = time.time()
    while gen.num_remaining_jobs():
        for res in gen.iterate():
            if res.get("token_ids") is not None:
                toks += res["token_ids"].flatten().tolist()
            if res.get("text"):
                text += res["text"]
    out[f"p{pi}"] = {"ntok": int(ids.numel()), "tokens": toks, "text": text,
                     "secs": round(time.time() - t0, 2)}
    print(f"  prompt{pi} ({ids.numel()} in): {len(toks)} tok in {out[f'p{pi}']['secs']}s")
    print(f"    {text!r}")

with open(f"/tmp/det_{mode}.json", "w") as f:
    json.dump(out, f)
print(f"saved /tmp/det_{mode}.json")
