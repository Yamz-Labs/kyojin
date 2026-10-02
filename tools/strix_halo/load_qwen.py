#!/usr/bin/env python
"""Load the Qwen3.8-Flash-Next EXL3 pack on gfx1151 and generate greedily.

Correctness-first smoke test: proves the pack loads and produces coherent
text through exllamav3's own engine on AMD Strix Halo. Speed is secondary --
gfx1151 falls back to reconstruct+hgemm (exl3_gemv_supported() == False).

The 32.6 GB n-gram/PLE table stays file-backed ("trellis_disk" mode), which
is what lets an ~80 GB pack fit in 61.4 GiB of GPU-visible memory.
"""
import argparse, os, sys, time

p = argparse.ArgumentParser()
p.add_argument("-m", "--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
p.add_argument("-p", "--prompt", default="The capital of France is")
p.add_argument("-n", "--num-tokens", type=int, default=32)
p.add_argument("-c", "--cache-tokens", type=int, default=4096)
p.add_argument("--max-gib", type=float, default=None, help="cap load budget (GiB)")
args = p.parse_args()

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

print(f"torch {torch.__version__}  hip {torch.version.hip}", flush=True)
pr = torch.cuda.get_device_properties(0)
print(f"gpu   {pr.name} / {pr.gcnArchName} / {pr.total_memory/2**30:.1f} GiB", flush=True)
try:
    import exllamav3_ext
    print(f"gemv  exl3_gemv_supported(0) = {exllamav3_ext.exl3_gemv_supported(0)} "
          f"(False => reconstruct+hgemm)", flush=True)
except Exception as e:
    print("gemv  ext probe failed:", e, flush=True)

print(f"\nloading {args.model}", flush=True)
t0 = time.time()
config = Config.from_directory(args.model)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)

load_kwargs = {}
if args.max_gib is not None:
    load_kwargs["reserve_per_device"] = 0.0
    load_kwargs["max_vram"] = [int(args.max_gib * 1024)]

cache = Cache(model, max_num_tokens=args.cache_tokens)
model.load(progressbar=True, **load_kwargs)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

# report the n-gram table mode if the model exposes one
try:
    for mod in (model.modules or []):
        m = getattr(mod, "mode", None)
        if m:
            print(f"ngram mode: {type(mod).__name__} -> {m}", flush=True)
            break
except Exception as _e:
    print(f"(ngram mode probe skipped: {_e})", flush=True)

generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

print(f"\nprompt: {args.prompt!r}", flush=True)
t0 = time.time()
job = Job(input_ids=tokenizer.encode(args.prompt, add_bos=True),
          max_new_tokens=args.num_tokens)
generator.enqueue(job)

out, ttft, ntok = "", None, 0
while generator.num_remaining_jobs():
    for res in generator.iterate():
        chunk = res.get("text", "")
        if chunk:
            if ttft is None:
                ttft = time.time() - t0
            out += chunk
            ntok += 1
            sys.stdout.write(chunk); sys.stdout.flush()
total = time.time() - t0

print("\n\n--- results ---")
print(f"output:  {out!r}")
print(f"ttft:    {ttft:.2f}s" if ttft else "ttft:    n/a")
if ttft and ntok > 1:
    print(f"decode:  {(ntok-1)/(total-ttft):.2f} tok/s  ({ntok} chunks, {total:.1f}s total)")
print(f"mem:     {torch.cuda.memory_allocated()/2**30:.1f} GiB allocated")
with open("/proc/meminfo") as f:
    for line in f:
        if line.startswith("MemAvailable"):
            print(f"host:    {int(line.split()[1])/2**20:.1f} GiB MemAvailable")
