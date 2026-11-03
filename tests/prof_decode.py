"""Decode per op class on the current device. MODEL_DIR | FMOE_GGUF, DEC_N (concurrent sessions, default 1), DEC_STEPS.
Prints step time, tok/s, achieved weight bandwidth vs a measured copy-bandwidth roof, and the kernel-time split."""
import os, sys, time
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from exllamav3 import Model, Config, Cache, Generator, Job, GreedySampler, Tokenizer
from exllamav3.loader.gguf_config import write_config_dir
from exllamav3.modules import BlockSparseMLP
from torch.profiler import profile, ProfilerActivity
N = int(os.environ.get("DEC_N", "1")); STEPS = int(os.environ.get("DEC_STEPS", "48"))
d = os.environ.get("MODEL_DIR") or write_config_dir(os.environ["FMOE_GGUF"], os.path.join(os.environ.get("FMOE_CFG", "/tmp"), "qwen35moe_cfg"))
config = Config.from_directory(d); model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096 * max(N, 1)); model.load()
tok = Tokenizer.from_config(config); gen = Generator(model=model, cache=cache, tokenizer=tok)
x = torch.empty(1 << 28, dtype=torch.uint8, device="cuda"); y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
torch.cuda.synchronize(); t = time.time()
for _ in range(10): y.copy_(x)
torch.cuda.synchronize(); bw = 10 * 2 * x.numel() / (time.time() - t) / 1e9   # read+write
del x, y
def active_params(mod, scale=1.0):
    n = 0.0
    if isinstance(mod, BlockSparseMLP):
        k = mod.num_experts_per_tok / mod.num_experts
        keep = {id(getattr(mod, a, None)) for a in ("shared_experts", "shared_gate", "routing_gate", "latent_in", "latent_out")}
        return sum(active_params(c, scale if id(c) in keep else scale * k) for c in mod.modules)
    if hasattr(mod, "in_features") and hasattr(mod, "out_features") and not mod.modules:
        return scale * mod.in_features * mod.out_features
    return sum(active_params(c, scale) for c in mod.modules)
BPW = float(os.environ.get("DEC_BPW", "4.0"))
ACT = sum(active_params(m) for m in model.modules)
g = torch.Generator().manual_seed(3)
def start(n_new):
    for _ in range(N):
        ids = torch.randint(1000, 100000, (1, 384), generator=g)
        gen.enqueue(Job(input_ids=ids, max_new_tokens=n_new, sampler=GreedySampler(), stop_conditions=[]))
def prefill_all():
    # iterate until every job has produced its first token
    seen = set()
    while len(seen) < N:
        for r in gen.iterate():
            if r.get("stage") == "error": raise SystemExit("job error: " + str(r.get("error")))
            if r.get("stage") == "streaming" and "token_ids" in r: seen.add(r["serial"])
start(STEPS + 80); prefill_all()
for _ in range(8): gen.iterate()
torch.cuda.synchronize()
t = time.time()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(STEPS): gen.iterate()
    torch.cuda.synchronize()
dt = (time.time() - t) / STEPS
tot = sum(e.device_time_total for e in p.key_averages())
print(f"N={N}: step {dt*1e3:.1f} ms wall, {N/dt:.1f} tok/s total, device-busy {tot/STEPS/1e3:.1f} ms/step ({100*tot/STEPS/1e6/dt:.0f} % of wall); copy roof {bw:.0f} GB/s (r+w)")
GBT = ACT * BPW / 8 / 1e9
print(f"active params {ACT/1e9:.2f} B x {BPW} bpw = {GBT:.2f} GB per token; one weight pass per step = {GBT/dt:.0f} GB/s read ({100*GBT/dt/(bw/2):.0f} % of half the r+w copy rate); at N={N} MoE reads more experts than one token, dense reads the same")
classes = {"gemv/linear": ("gemv", "mpw", "fq_", "exl3", "Cijk", "gemm", "mgemm", "hgemm"), "moe": ("moe", "expert", "route", "topk"),
           "gdn/ssm": ("chunk", "gated", "delta", "recompute", "conv", "gdn", "recurrent"), "attn": ("attn", "flash", "qsa", "rope", "paged"),
           "norm/elem": ("norm", "gr_", "silu", "add", "mul", "hc_apply", "copy", "fill", "elementwise", "reduce")}
acc = {k: 0.0 for k in classes}; acc["other"] = 0.0
for e in p.key_averages():
    n = e.key.lower()
    for k, subs in classes.items():
        if any(s in n for s in subs): acc[k] += e.device_time_total; break
    else: acc["other"] += e.device_time_total
print("class split (% of device time): " + ", ".join(f"{k} {100*v/tot:.0f}" for k, v in sorted(acc.items(), key=lambda kv: -kv[1])))
for e in sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:12]:
    print(f"{100*e.device_time_total/tot:5.1f}%  {e.device_time_total/STEPS:8.0f} us/step  n/step={e.count/STEPS:6.1f}  {e.key[:80]}")
