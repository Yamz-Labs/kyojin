"""Kernel count / GPU busy time per decoded token on a qwen35moe GGUF (launch-overhead profile).
    FMOE_GGUF=<file.gguf> python tests/prof_gguf_qwen35moe_decode.py"""
import os, sys, time
import torch
from torch.profiler import profile, ProfilerActivity
ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)
from exllamav3 import Model, Config, Cache, Generator, Job, GreedySampler, Tokenizer
from exllamav3.loader.gguf_config import write_config_dir
d = write_config_dir(os.environ["FMOE_GGUF"], os.path.join(os.environ.get("FMOE_CFG", "/tmp"), "qwen35moe_cfg"))
config = Config.from_directory(d); model = Model.from_config(config); cache = Cache(model, max_num_tokens=4096); model.load()
tok = Tokenizer.from_config(config); gen = Generator(model=model, cache=cache, tokenizer=tok)
ids = tok.encode("The capital of France is a city that")
def run(n):
    gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), stop_conditions=[]))
    c = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r["stage"] == "streaming" and "token_ids" in r: c += r["token_ids"].numel()
    return c
run(8); torch.cuda.synchronize()
N = 48
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    t = time.time(); c = run(N); torch.cuda.synchronize(); wall = time.time() - t
ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
busy = sum(e.device_time for e in ev) / 1000
print(f"tokens {c}  wall/token {wall / c * 1000:.2f} ms  kernels/token {len(ev) / c:.0f}  GPU busy/token {busy / c:.2f} ms")
agg = {}
for e in ev: agg.setdefault(e.name[:60], [0, 0.0]); agg[e.name[:60]][0] += 1; agg[e.name[:60]][1] += e.device_time
for n, (k, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:12]:
    print(f"{t / c / 1000:7.3f} ms/tok  {k / c:6.1f}/tok  {n}")
