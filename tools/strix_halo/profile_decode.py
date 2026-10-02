#!/usr/bin/env python
"""Where does decode time actually go? Measure, don't guess.

Graph capture works but gives no speedup => not launch-bound. The candidates:
  (a) memory bandwidth on the weights read per token
  (b) disk I/O on the 32.6 GB file-backed ngram_embedding table (trellis_disk)
  (c) a specific kernel dominating

Instruments:
  1. wall-clock per decode step
  2. cumulative GPU time per ext entry point (hipEvent around each call)
  3. /proc/<pid>/io read_bytes delta during decode -> real disk I/O
  4. achieved bandwidth vs the ~256 GB/s LPDDR5X ceiling
"""
import os, sys, time, collections
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext as X
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3 import ext as extwrap

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
NTOK = int(os.environ.get("NTOK", "32"))


def io_bytes():
    try:
        with open(f"/proc/{os.getpid()}/io") as f:
            d = {}
            for line in f:
                k, v = line.split(":")
                d[k.strip()] = int(v)
            return d
    except Exception:
        return {}


config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=2048)
model.load(progressbar=False)

# ---- instrument the hot ext entry points with CUDA events ----
TARGETS = ["exl3_gemv", "exl3_moe_gfx12_k3", "exl3_moe_gfx12_k3_prefill",
           "hgemm", "hgemm_recon", "had_r_128", "reconstruct",
           "routing_std", "routing_std_gfx12_bsz1", "routing_ds3_nogroup",
           "rms_norm", "silu_mul", "fused_sampler"]

stats = collections.defaultdict(lambda: {"n": 0, "ms": 0.0})
real = {}


def wrap(name):
    fn = getattr(X, name, None)
    if fn is None:
        return False
    real[name] = fn

    def spy(*a, **k):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        r = fn(*a, **k)
        e.record()
        e.synchronize()
        st = stats[name]
        st["n"] += 1
        st["ms"] += s.elapsed_time(e)
        return r

    setattr(X, name, spy)
    return True


wrapped = [n for n in TARGETS if wrap(n)]
print(f"instrumented: {wrapped}", flush=True)

gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
ids = tokenizer.encode("Explain gradient descent in two sentences:", add_bos=True)

# warm up (first token pays prefill + any JIT)
gen.enqueue(Job(input_ids=ids, max_new_tokens=4))
while gen.num_remaining_jobs():
    for _ in gen.iterate():
        pass
for st in stats.values():
    st["n"] = 0
    st["ms"] = 0.0

io0 = io_bytes()
torch.cuda.synchronize()
t0 = time.time()
gen.enqueue(Job(input_ids=ids, max_new_tokens=NTOK))
ntok = 0
first = None
while gen.num_remaining_jobs():
    for res in gen.iterate():
        if res.get("text"):
            if first is None:
                first = time.time() - t0
            ntok += 1
torch.cuda.synchronize()
total = time.time() - t0
io1 = io_bytes()

for n in wrapped:
    setattr(X, n, real[n])

decode_s = total - (first or 0)
print()
print(f"tokens={ntok}  total={total:.2f}s  ttft={first:.2f}s  "
      f"decode={decode_s:.2f}s  rate={(ntok-1)/decode_s:.2f} tok/s")
print()

rd = (io1.get("read_bytes", 0) - io0.get("read_bytes", 0)) / 2**20
print(f"disk read during run: {rd:.1f} MiB  "
      f"({rd/max(decode_s,1e-9):.1f} MiB/s)  "
      f"-> {'DISK I/O IS A FACTOR' if rd > 200 else 'negligible, weights are resident'}")
print()

rows = sorted(stats.items(), key=lambda kv: -kv[1]["ms"])
tot_ms = sum(v["ms"] for _, v in rows)
print(f"{'kernel':>28} {'calls':>8} {'total_ms':>10} {'us/call':>9} {'%GPU':>7} {'%wall':>7}")
print("-" * 76)
for name, v in rows:
    if v["n"] == 0:
        continue
    print(f"{name:>28} {v['n']:>8} {v['ms']:>10.1f} "
          f"{v['ms']*1000/v['n']:>9.1f} {100*v['ms']/max(tot_ms,1e-9):>7.1f} "
          f"{100*v['ms']/(total*1000):>7.1f}")
print("-" * 76)
print(f"{'SUM instrumented':>28} {'':>8} {tot_ms:>10.1f} {'':>9} {'':>7} "
      f"{100*tot_ms/(total*1000):>7.1f}")
print()
print(f"unaccounted wall time: {total*1000 - tot_ms:.0f} ms "
      f"({100*(1 - tot_ms/(total*1000)):.1f}%) -- Python overhead, launches, sync")
print()
# bandwidth estimate from the dominant kernel
g = stats.get("exl3_gemv")
if g and g["n"]:
    print(f"exl3_gemv: {g['n']} calls in {decode_s:.2f}s decode "
          f"= {g['n']/max(ntok-1,1):.0f} calls/token")
