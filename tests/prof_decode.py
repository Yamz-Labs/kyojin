"""prof_decode.py PACK [--depth 4096] [--kv 16|K,V] [--steps 24]: kernel-time table of steady greedy decode steps (torch profiler).
Prints per-kernel total/avg us per step (top 40) and the step wall time, to find where a decode step spends its time."""
import argparse, os, sys, time
ROOT = os.environ.get("ENG") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.presets import GreedySampler
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--depth", type=int, default=4096); ap.add_argument("--kv", default="16")
ap.add_argument("--steps", type=int, default=24); ap.add_argument("--top", type=int, default=40); ap.add_argument("--noprof", action="store_true")
a = ap.parse_args()
cfg = Config.from_directory(a.pack); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
ctx = ((a.depth + a.steps + 512 + 255) // 256) * 256
if a.kv == "16": cache = Cache(model, max_num_tokens=ctx)
else:
    kb, vb = (int(x) for x in a.kv.split(",")); cache = Cache(model, max_num_tokens=ctx, layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
model.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok)
D = os.path.join(ROOT, "exllamav3/conversion/standard_cal_data")
s = open(f"{D}/wiki.utf8", encoding="utf8").read()
ids = tok.encode(s[3000:3000 + a.depth * 8], add_bos=False).flatten()[:a.depth].view(1, -1)
job = Job(input_ids=ids, max_new_tokens=a.steps + 40, sampler=GreedySampler(), stop_conditions=[])
gen.enqueue(job)
n = 0; prof = None; t0 = None
while gen.num_remaining_jobs():
    if a.noprof:
        for r in gen.iterate():
            if r.get("text", "") != "": n += 1
        continue
    if n == 16 and prof is None:
        torch.cuda.synchronize(); prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); prof.__enter__(); t0 = time.time()
    for r in gen.iterate():
        if r.get("text", "") != "": n += 1
    if prof is not None and n >= 16 + a.steps:
        torch.cuda.synchronize(); wall = time.time() - t0; prof.__exit__(None, None, None); break
if a.noprof: print("decoded", n, "tokens"); sys.exit(0)
print(f"wall per step {wall / a.steps * 1000:.2f} ms ({a.steps / wall:.1f} tok/s under the profiler)")
ev = prof.key_averages()
rows = sorted(ev, key=lambda e: -e.device_time_total)
tot = sum(e.device_time_total for e in rows if e.device_type == torch.autograd.DeviceType.CUDA) if hasattr(torch.autograd, "DeviceType") else 0
print(f"{'kernel':80s} {'calls/step':>10s} {'us/step':>9s} {'avg us':>8s}")
tot_us = 0
for e in rows[:a.top]:
    if e.device_time_total <= 0: continue
    print(f"{e.key[:80]:80s} {e.count / a.steps:10.1f} {e.device_time_total / a.steps:9.1f} {e.device_time_total / max(e.count, 1):8.1f}")
print("sum of listed device us/step:", sum(e.device_time_total for e in rows[:a.top] if e.device_time_total > 0) / a.steps)
