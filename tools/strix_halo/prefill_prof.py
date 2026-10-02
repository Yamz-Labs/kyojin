#!/usr/bin/env python
"""Where does PREFILL time go on gfx1151? torch profiler over one cold 8k-token prefill.

Decode is bandwidth-bound; prefill is (should be) compute-bound. 315 tok/s x 3.05 bpw
weights streamed once per 2048-token chunk is nowhere near either the 31 TFLOP/s fp16
matmul rate or the 236 GB/s bandwidth. Find the kernel(s) that dominate.
"""
import os, sys, time, collections
import torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
P = int(os.environ.get("P", "8192"))
CHUNK = int(os.environ.get("CHUNK", "2048"))
config = Config.from_directory(MODEL)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=32768, max_history=0); model.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, max_chunk_size=CHUNK)

g = torch.Generator().manual_seed(1)
ids = torch.randint(1000, 100000, (1, P), generator=g, dtype=torch.long)

def run(prof=None):
    gen.clear_queue()
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler()))
    t0 = time.perf_counter()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): print("JOB ERROR", r["error"])
    torch.cuda.synchronize()
    return time.perf_counter() - t0

# warm (JIT etc.) with a short prompt
gen.enqueue(Job(input_ids=ids[:, :512], max_new_tokens=1, sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate(): pass
torch.cuda.synchronize()

# Fresh random prompt per run so nothing is prefix-cached
def fresh(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (1, P), generator=g, dtype=torch.long)
ids = fresh(1)
wall = run()
print(f"prefill {P} tokens (chunk {CHUNK}): {wall:.2f}s = {P/wall:.0f} tok/s")
ids = fresh(2)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    wall2 = run()
ka = prof.key_averages()
dev = sum((getattr(e, "self_device_time_total", 0) or 0) for e in ka)
print(f"profiled: {wall2:.2f}s wall, device {dev/1e6:.2f}s ({100*dev/1e6/wall2:.0f}% busy)")
cpu_total = sum(e.self_cpu_time_total for e in ka)
print(f"cpu self total {cpu_total/1e6:.2f}s")
rows = sorted(ka, key=lambda e: -(getattr(e, "self_device_time_total", 0) or 0))
print(f"{'kernel':60} {'calls':>6} {'ms':>8} {'us/call':>8} {'%dev':>5}")
for e in rows[:25]:
    d = (getattr(e, "self_device_time_total", 0) or 0)
    if d <= 0: continue
    print(f"{e.key[:60]:60} {e.count:>6} {d/1000:>8.1f} {d/e.count:>8.1f} {100*d/dev:>5.1f}")
print()
rows = sorted(ka, key=lambda e: -e.self_cpu_time_total)
print("top CPU self time:")
for e in rows[:8]:
    print(f"  {e.key[:60]:60} {e.count:>6} {e.self_cpu_time_total/1000:>8.1f} ms")
