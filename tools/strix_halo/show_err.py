#!/usr/bin/env python
"""Print the generator's error for the WMMA fast path."""
import os, sys, traceback
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
print(f"family={exllamav3_ext.exl3_gemv_wmma_family(0)}", flush=True)

config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=2048)
model.load(progressbar=False)

ids = tokenizer.encode("The capital of France is", add_bos=True)
gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
gen.enqueue(Job(input_ids=ids, max_new_tokens=4))

while gen.num_remaining_jobs():
    for res in gen.iterate():
        if "error" in res:
            err = res["error"]
            print("=" * 70)
            print("JOB ERROR:", type(err).__name__ if isinstance(err, BaseException) else type(err))
            print(repr(err)[:2000])
            if isinstance(err, BaseException):
                print("-" * 70)
                traceback.print_exception(type(err), err, err.__traceback__)
            print("=" * 70)
        if res.get("text"):
            print("text:", repr(res["text"]))
