#!/usr/bin/env python
"""Decode-only kernel census at current HEAD. One greedy MTP run, 96 new tokens, ndt=3 dc=0.6."""
import os, sys, collections
import torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
NDT = int(os.environ.get("NDT", "3"))
NTOK = int(os.environ.get("NTOK", "96"))
config = Config.from_directory(MODEL)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
dconfig = Config.from_directory(MODEL); dconfig.arch_override = "qwen4_exp_mtp.py"
draft = Model.from_config(dconfig)
cache = Cache(model, max_num_tokens=32768, max_history=NDT); model.load(progressbar=False)
dcache = Cache(draft, max_num_tokens=32768, max_history=NDT); draft.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=NDT, dynamic_draft_tokens=True, draft_confidence=0.6)
ids = tok.encode("Explain gradient descent in two sentences:", add_bos=True)
gen.enqueue(Job(input_ids=ids, max_new_tokens=8, sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate(): pass
torch.cuda.synchronize()
gen.clear_queue()
t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
import time
wall0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    gen.enqueue(Job(input_ids=ids, max_new_tokens=NTOK, sampler=GreedySampler()))
    nacc = nrej = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            nacc += r.get("accepted_draft_tokens") or 0
            nrej += r.get("rejected_draft_tokens") or 0
    torch.cuda.synchronize()
wall = time.perf_counter() - wall0
print(f"wall {wall:.2f}s  {NTOK/wall:.1f} tok/s  accept {nacc}/{nacc+nrej}")
ka = prof.key_averages()
dev = sum((getattr(e, "self_device_time_total", 0) or 0) for e in ka)
print(f"device {dev/1e6:.2f}s ({100*dev/1e6/wall:.0f}% of wall)")
# family rollup
fam = collections.defaultdict(lambda: [0, 0.0])
def family(k):
    k = k.replace("void (anonymous namespace)::", "").replace("void at::native::", "")
    for p in ("Cijk_", "moe_prefill_grouped_gemv", "moe_prefill_had_rows", "moe_prefill_metadata",
              "exl3_gemv", "gr_mix", "gr_dots", "gr_finalize", "had_r_128", "reconstruct",
              "chunk_fwd", "recompute_w_u", "_causal_conv1d", "hgemm", "skinny",
              "aten::mm", "aten::copy_", "aten::mul", "hipLaunchKernel", "hipPointerGetAttribute"):
        if p in k: return p
    return k[:40]
for e in ka:
    d = getattr(e, "self_device_time_total", 0) or 0
    if d <= 0: continue
    f = family(e.key)
    fam[f][0] += e.count
    fam[f][1] += d
print(f"{'family':32} {'calls':>7} {'ms':>10} {'%dev':>6}")
for f, (n, d) in sorted(fam.items(), key=lambda kv: -kv[1][1])[:18]:
    print(f"{f:32} {n:7d} {d/1000:10.1f} {100*d/dev:6.1f}")
print("TOP kernels:")
for e in sorted(ka, key=lambda e: -(getattr(e, "self_device_time_total", 0) or 0))[:12]:
    d = getattr(e, "self_device_time_total", 0) or 0
    print(f"  {e.count:6d} {d/1000:8.1f} ms  {e.key[:90]}")
