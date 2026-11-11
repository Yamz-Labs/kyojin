"""Batch-1 decode (cache of one slot, as served N1): 3 prompts x 200 greedy tokens, tok/s = tokens / wall of each generate. usage: n1_hip.py MODEL_DIR"""
import sys, time
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
config = Config.from_directory(sys.argv[1]); model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192, max_batch_size=1); model.load(progressbar=False)
tok = Tokenizer.from_config(config); gen = Generator(model=model, cache=cache, tokenizer=tok)
fmt = lambda p: f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
gen.generate(prompt=fmt("Hi"), max_new_tokens=8, completion_only=True)
for p in ["Explain how a hash map handles collisions, with a short Python example.", "Écris un court paragraphe en français sur l'histoire du phare d'Alexandrie.",
          "Write a Python function that parses an ISO 8601 date string without external libraries."]:
    t = time.time(); out = gen.generate(prompt=fmt(p), max_new_tokens=200, completion_only=True, stop_conditions=[]); dt = time.time() - t
    n = tok.encode(out).shape[-1]
    print(f"N1 {n / dt:.2f} tok/s  n {n}  head {out[:70]!r}", flush=True)
