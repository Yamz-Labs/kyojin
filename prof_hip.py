"""Decode-step profile on the HIP engine (decision: where the N1 step time goes). Greedy 64 tokens under torch.profiler; top kernels by
self device time, their share of the total device time, and device time vs wall (launch / host gaps).
usage: prof_hip.py MODEL_DIR"""
import sys, time
import torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator

config = Config.from_directory(sys.argv[1])
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096, max_batch_size=1)
model.load(progressbar=False)
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)
p = "<|im_start|>user\nExplain how a hash map handles collisions.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
gen.generate(prompt=p, max_new_tokens=16, completion_only=True)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    t = time.time()
    gen.generate(prompt=p, max_new_tokens=64, completion_only=True, stop_conditions=[])
    torch.cuda.synchronize()
    wall = time.time() - t
ka = prof.key_averages()
dt = lambda e: max(getattr(e, "self_device_time_total", 0) or 0, getattr(e, "self_cuda_time_total", 0) or 0)
print("events", len(ka), "with device time", sum(1 for e in ka if dt(e) > 0), "fields", [a for a in dir(ka[0]) if "time" in a][:12])
ev = [e for e in ka if dt(e) > 0]
if not ev:
    for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:15]:
        print(f"CPU {e.self_cpu_time_total / 1e3:8.1f} ms n {e.count:6d} {e.key[:90]}")
tot = sum(dt(e) for e in ev) or 1
print(f"wall {wall * 1e3:.0f} ms for 64 tokens ({wall / 64 * 1e3:.2f} ms/token), device self total {tot / 1e3:.0f} ms ({tot / 1e3 / (wall * 1e3) * 100:.0f} % of wall)")
for e in sorted(ev, key=lambda e: -dt(e))[:18]:
    s = dt(e)
    print(f"{s / 1e3:8.1f} ms {s / tot * 100:5.1f} %  n {e.count:6d}  {e.key[:95]}")
