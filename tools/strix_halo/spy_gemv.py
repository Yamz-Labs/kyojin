#!/usr/bin/env python
"""Instrument the real hot path: count hip_gemv vs reconstruct_hgemm calls
during an actual forward, and A/B the two on identical inputs.

Monkey-patches LinearEXL3.forward to (a) tally which route each call takes and
(b) for the first few EXL3_GEMV-eligible calls, compute BOTH results and report
the delta. This exercises real trellis state and real shapes.
"""
import os, sys, collections
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext
from exllamav3 import Config, Model, Cache, Tokenizer
import exllamav3.modules.quant.exl3 as exl3mod
from exllamav3.modules.quant.exl3 import LinearEXL3

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
print(f"supported={exllamav3_ext.exl3_gemv_supported(0)} "
      f"family={exllamav3_ext.exl3_gemv_wmma_family(0)}", flush=True)

stats = collections.Counter()
deltas = []
CHECK_LIMIT = 12

orig_hip = LinearEXL3.hip_gemv
orig_rec = LinearEXL3.reconstruct_hgemm

def hip_spy(self, x, out_dtype):
    stats["hip_gemv"] += 1
    y = orig_hip(self, x, out_dtype)
    if len(deltas) < CHECK_LIMIT:
        try:
            yr = orig_rec(self, x, out_dtype)
            a, b = y.float(), yr.float()
            d = (a - b).abs()
            scale = b.abs().mean().item() + 1e-9
            deltas.append((self.key, tuple(x.shape), self.K, self.mcg, self.mul1,
                           d.max().item(), d.mean().item(), d.mean().item() / scale))
        except Exception as e:
            deltas.append((self.key, tuple(x.shape), self.K, self.mcg, self.mul1,
                           float("nan"), float("nan"), float("nan")))
    return y

def rec_spy(self, x, out_dtype):
    stats["reconstruct"] += 1
    return orig_rec(self, x, out_dtype)

LinearEXL3.hip_gemv = hip_spy
LinearEXL3.reconstruct_hgemm = rec_spy

config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=2048)
model.load(progressbar=False)
print("loaded", flush=True)

from exllamav3 import Generator, Job
gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
job = Job(input_ids=tokenizer.encode("The capital of France is", add_bos=True),
          max_new_tokens=6)
gen.enqueue(job)
out = ""
while gen.num_remaining_jobs():
    for r in gen.iterate():
        out += r.get("text", "")

print(f"\nroute tally: {dict(stats)}")
print(f"generated:   {out!r}")
print(f"\nfirst {len(deltas)} hip_gemv calls vs reconstruct:")
bad = 0
for key, shp, K, mcg, mul1, mx, mn, rel in deltas:
    flag = "OK" if rel < 0.02 else "** MISMATCH **"
    if not (rel < 0.02):
        bad += 1
    print(f"  {str(shp):14} K={K} mcg={int(bool(mcg))} mul1={int(bool(mul1))} "
          f"max={mx:.5f} mean={mn:.6f} rel={rel:.3e} {flag}  {key}")
print(f"\nmismatches: {bad}/{len(deltas)}")
