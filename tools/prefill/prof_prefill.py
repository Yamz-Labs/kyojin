"""Profile one MiMo prefill (unique corpus slice, no prefix-cache hit) with torch.profiler.
Prints the kernel time split grouped by kernel name, plus wall time of the profiled prefill."""
import argparse, sys, time, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from exllamav3 import Generator, Job, GreedySampler, model_init

p = argparse.ArgumentParser()
model_init.add_args(p, cache = True)
p.add_argument("--prompt", type = int, default = 4096)
p.add_argument("--corpus", default = "~/bench/ppl/wiki.test.raw")
p.add_argument("--no-prof", action = "store_true")
p.add_argument("--top", type = int, default = 45)
p.add_argument("--trace", default = None)
args = p.parse_args()
torch.set_grad_enabled(False)
import exllamav3_ext
print("ext", exllamav3_ext.__file__, flush = True)

model, config, cache, tokenizer = model_init.init(args)[:4]
text = open(args.corpus, encoding = "utf-8").read()

def ids_at(off, n):
    ids = tokenizer.encode(text[off: off + n * 8])
    assert ids.shape[-1] >= n
    return ids[:, :n]

gen = Generator(model = model, cache = cache, tokenizer = tokenizer)

def prefill(ids):
    job = Job(input_ids = ids, max_new_tokens = 1, sampler = GreedySampler())
    gen.enqueue(job)
    torch.cuda.synchronize()
    t0 = time.time()
    while gen.num_remaining_jobs():
        gen.iterate()
    torch.cuda.synchronize()
    return time.time() - t0

print(f"warmup {prefill(ids_at(300000, 1024)):.2f}s", flush = True)
ids = ids_at(500000, args.prompt)
if args.no_prof:
    dt = prefill(ids)
    print(f"prefill {args.prompt} tok: {dt:.2f}s = {args.prompt/dt:.1f} tok/s", flush = True)
    sys.exit(0)

from torch.profiler import profile, ProfilerActivity
with profile(activities = [ProfilerActivity.CUDA]) as prof:
    dt = prefill(ids)
print(f"prefill (profiled) {args.prompt} tok: {dt:.2f}s = {args.prompt/dt:.1f} tok/s", flush = True)
ka = prof.key_averages()
rows = [(e.key, e.self_device_time_total, e.count) for e in ka if e.self_device_time_total > 0 and e.device_type.name == "CUDA"]
rows.sort(key = lambda r: -r[1])
tot = sum(r[1] for r in rows)
print(f"GPU total {tot/1e6:.3f}s over {sum(r[2] for r in rows)} kernel calls")
for k, t, c in rows[:args.top]:
    print(f"{t/1e3:10.1f} ms {100*t/tot:5.1f}% {c:7d}  {k[:110]}")
if args.trace:
    prof.export_chrome_trace(args.trace)
