"""Prefill throughput vs the GEMM roof. Prints, per prompt length: tok/s, the GEMM TFLOPs that rate implies (2 x active params
per token; attention FLOPs not counted, so the gap at 32K includes them) and its share of the device's measured fp16 matmul peak.

    MODEL_DIR=<exl3 dir> | FMOE_GGUF=<file.gguf>  [PREFILL_LENS=8192,32768] python tests/bench_prefill_roof.py
    PREFILL_PROF=1 adds a device-time split by op class and the top kernels for the last length.
"""
import os, sys, time
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)
from exllamav3 import Model, Config, Cache, Generator, Job, GreedySampler, Tokenizer
from exllamav3.loader.gguf_config import write_config_dir
from exllamav3.modules import BlockSparseMLP

lens = [int(v) for v in os.environ.get("PREFILL_LENS", "8192,32768").split(",")]
if os.environ.get("MODEL_DIR"):
    d = os.environ["MODEL_DIR"]
else:
    d = write_config_dir(os.environ["FMOE_GGUF"], os.path.join(os.environ.get("FMOE_CFG", "/tmp"), "qwen35moe_cfg"))
config = Config.from_directory(d)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=(max(lens) + 1024) // 256 * 256 + 256)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)


def active_params(mod, scale=1.0):
    n = 0.0
    if isinstance(mod, BlockSparseMLP):
        k = mod.num_experts_per_tok / mod.num_experts
        keep = {id(getattr(mod, a, None)) for a in ("shared_experts", "shared_gate", "routing_gate", "latent_in", "latent_out")}
        for c in mod.modules:
            n += active_params(c, scale if id(c) in keep else scale * k)
        return n
    if hasattr(mod, "in_features") and hasattr(mod, "out_features") and not mod.modules:
        return scale * mod.in_features * mod.out_features
    for c in mod.modules:
        n += active_params(c, scale)
    return n


dev = torch.device("cuda")
a = torch.randn(8192, 8192, device=dev, dtype=torch.half)
b = torch.randn(8192, 8192, device=dev, dtype=torch.half)
for _ in range(3): a @ b
torch.cuda.synchronize(); t = time.time()
for _ in range(20): a @ b
torch.cuda.synchronize()
peak = 20 * 2 * 8192 ** 3 / (time.time() - t) / 1e12
del a, b
act = sum(active_params(m) for m in model.modules)
print(f"device fp16 matmul peak {peak:.1f} TFLOPs; active params/token {act / 1e9:.2f} B -> GEMM roof {peak * 1e12 / (2 * act):.0f} tok/s")

g = torch.Generator().manual_seed(1)
vocab = min(config.vocab_size, 100000)
for L in [1024] + lens:
    for rep in range(2 if L == 1024 else 1):
        ids = torch.randint(1000, vocab, (1, L), generator=g)   # random ids: no prefix-cache hit
        job = Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler(), stop_conditions=[])
        gen.enqueue(job)
        torch.cuda.synchronize(); t = time.time()
        prof = os.environ.get("PREFILL_PROF") and L == lens[-1]
        if prof:
            from torch.profiler import profile, ProfilerActivity
            pctx = profile(activities=[ProfilerActivity.CUDA]); pctx.__enter__()
        while gen.num_remaining_jobs():
            gen.iterate()
        torch.cuda.synchronize(); dt = time.time() - t
        if prof:
            pctx.__exit__(None, None, None)
            tot = sum(e.device_time_total for e in pctx.key_averages())
            classes = {"gemm/linear": ("gemm", "cijk", "cutlass", "mgemm", "gemv", "exl3", "mpw", "fq_", "matmul", "tensile"), "moe": ("moe", "expert", "route", "topk", "grouped", "dq_", "sort"),
                       "gdn/ssm": ("chunk", "gated", "delta", "recompute", "conv", "gdn", "recurrent"), "attn": ("attn", "flash", "qsa", "rope", "paged", "softmax"),
                       "norm/elem": ("norm", "gr_", "silu", "add", "mul", "hc_apply", "copy", "fill", "elementwise", "reduce", "cat", "index")}
            acc = {k: 0.0 for k in classes}; acc["other"] = 0.0
            for e in pctx.key_averages():
                n = e.key.lower()
                for k, subs in classes.items():
                    if any(x in n for x in subs): acc[k] += e.device_time_total; break
                else: acc["other"] += e.device_time_total
            print(f"PROF L={L}: device {tot/1e3:.0f} ms of {dt*1e3:.0f} ms wall; class split: " + ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in sorted(acc.items(), key=lambda kv: -kv[1])))
            for e in sorted(pctx.key_averages(), key=lambda e: -e.device_time_total)[:16]:
                print(f"{100*e.device_time_total/tot:5.1f}%  {e.device_time_total/1e3:8.1f} ms  n={e.count:6d}  {e.key[:90]}")
    if L != 1024:
        r = L / dt
        print(f"prefill L={L}: {dt:.2f}s = {r:.0f} tok/s, {2 * act * r / 1e12:.1f} TFLOPs-equiv = {100 * 2 * act * r / 1e12 / peak:.0f} % of matmul peak")
