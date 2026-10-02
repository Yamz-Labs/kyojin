#!/usr/bin/env python3
"""Micro-timer: per-call us for selected mul1 modules under int8 mode 0 vs 2 (runtime toggle)."""
import os, sys, torch, time
os.environ["EXL3_INT8_GEMV"] = "0"
from exllamav3 import Config, Model
from exllamav3.modules import Linear

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
cfg = Config.from_directory(MODEL)
model = Model.from_config(cfg)
model.load(progressbar=False)

def find(keysuffix):
    stack = [model]
    while stack:
        mod = stack.pop()
        if isinstance(mod, Linear) and mod.key.endswith(keysuffix) and getattr(mod.inner, "mul1", False):
            return mod
        stack.extend(getattr(mod, "modules", []))

def timeit(inner, x, mode, iters=30):
    os.environ["EXL3_INT8_GEMV"] = str(mode)
    for _ in range(5): inner.forward(x, {})
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): inner.forward(x, {})
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters

mods = ["lm_head", "layers.47.self_attn.q_proj", "layers.47.self_attn.o_proj", "layers.47.self_attn.k_proj",
        "layers.47.mlp.shared_expert.down_proj", "layers.47.mlp.experts.511.down_proj", "layers.47.mlp.experts.511.up_proj"]
print(f"{'module':<45} {'K':>2} {'k':>5} {'n':>7} {'m':>2} {'fp16 us':>9} {'int8 us':>9} {'int8/fp16':>9}")
for s in mods:
    mod = find(s); inner = mod.inner
    for m in (1, 2):
        x = (torch.randn(m, mod.in_features) * 0.5).half().cuda()
        t0 = timeit(inner, x, 0); t2 = timeit(inner, x, 2)
        print(f"{mod.key[-45:]:<45} {inner.K:>2} {mod.in_features:>5} {mod.out_features:>7} {m:>2} {t0:>9.1f} {t2:>9.1f} {t2/t0:>9.2f}", flush=True)
os.environ["EXL3_INT8_GEMV"] = "0"
