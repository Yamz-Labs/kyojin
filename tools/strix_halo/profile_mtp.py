#!/usr/bin/env python
"""Profile decode WITH MTP speculation active.

Without MTP the wall was ~45% "unaccounted" host time. MTP changes the shape of
the problem: fewer trunk forwards, but each verifies multiple rows, plus the MTP
head runs every step. Re-measure so the next optimization targets reality.
"""
import os, sys, time, collections
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext as X
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
NDT = int(os.environ.get("NDT", "2"))
NTOK = int(os.environ.get("NTOK", "64"))

config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=4096, max_history=NDT)
model.load(progressbar=False)
draft_model = Model.from_config(config, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=4096, max_history=NDT)
draft_model.load(progressbar=False)

TARGETS = ["exl3_gemv", "exl3_moe_gfx12_k3", "exl3_moe_gfx12_k3_prefill",
           "hgemm", "hgemm_recon", "had_r_128", "reconstruct",
           "routing_std_gfx12_bsz1", "routing_std", "rms_norm", "silu_mul",
           "gr_mix", "hc_apply", "cuda_recurrent_gated_delta_rule"]
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
gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                draft_model=draft_model, draft_cache=draft_cache,
                num_draft_tokens=NDT, dynamic_draft_tokens=True,
                draft_confidence=0.4)
ids = tokenizer.encode("Explain gradient descent in two sentences:", add_bos=True)

# warm up
gen.enqueue(Job(input_ids=ids, max_new_tokens=6, sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate():
        pass
for st in stats.values():
    st["n"] = 0
    st["ms"] = 0.0

torch.cuda.synchronize()
t0 = time.time()
gen.enqueue(Job(input_ids=ids, max_new_tokens=NTOK, sampler=GreedySampler()))
ntok, first, acc, rej = 0, None, 0, 0
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if r.get("text"):
            if first is None:
                first = time.time() - t0
            ntok += 1
        if "accepted_draft_tokens" in r:
            acc += r["accepted_draft_tokens"]
            rej += r["rejected_draft_tokens"]
torch.cuda.synchronize()
total = time.time() - t0
for n in wrapped:
    setattr(X, n, real[n])

decode_s = total - (first or 0)
print(f"ndt={NDT} tokens={ntok} decode={decode_s:.2f}s "
      f"rate={(ntok-1)/decode_s:.2f} tok/s "
      f"acceptance={100*acc/max(acc+rej,1):.1f}%")
print()
rows = sorted(stats.items(), key=lambda kv: -kv[1]["ms"])
tot = sum(v["ms"] for _, v in rows)
print(f"{'kernel':>34} {'calls':>7} {'ms':>9} {'us/call':>9} {'%wall':>7}")
print("-" * 70)
for name, v in rows:
    if not v["n"]:
        continue
    print(f"{name:>34} {v['n']:>7} {v['ms']:>9.1f} "
          f"{v['ms']*1000/v['n']:>9.1f} {100*v['ms']/(total*1000):>7.1f}")
print("-" * 70)
print(f"{'SUM':>34} {'':>7} {tot:>9.1f} {'':>9} {100*tot/(total*1000):>7.1f}")
print(f"unaccounted: {total*1000-tot:.0f} ms "
      f"({100*(1-tot/(total*1000)):.1f}%) -- host/Python/launch")
print(f"calls per emitted token: gemv={stats['exl3_gemv']['n']/max(ntok,1):.0f} "
      f"moe={stats['exl3_moe_gfx12_k3']['n']/max(ntok,1):.0f}")
