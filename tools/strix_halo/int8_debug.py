#!/usr/bin/env python3
"""Debug harness: single module, toggle EXL3_INT8_GEMV at runtime (the C++ side re-reads getenv
on every call), inspect the relationship between int8 output and the fp16 reference."""
import os, sys, torch
os.environ["EXL3_INT8_GEMV"] = "0"
from exllamav3 import Config, Model
from exllamav3.modules import Linear
from exllamav3.ext import exllamav3_ext as ext

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

def run(inner, x, mode):
    os.environ["EXL3_INT8_GEMV"] = str(mode)
    y = inner.forward(x, {})
    torch.cuda.synchronize()
    return y.float()

def stats(name, a, b):
    d = (a - b)
    corr = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    print(f"  {name:<28} rms_rel={d.pow(2).mean().sqrt().item()/a.pow(2).mean().sqrt().item():.4g} "
          f"corr={corr:.4f} |a|={a.norm().item():.4g} |b|={b.norm().item():.4g} b/a={b.norm().item()/a.norm().item():.4g}")

os.environ["EXL3_INT8_GEMV_TRACE"] = "1"
for suffix in sys.argv[1:] or ["layers.47.self_attn.k_proj"]:
    mod = find(suffix)
    inner = mod.inner
    k, n, K = mod.in_features, mod.out_features, inner.K
    print(f"== {mod.key} K={K} k={k} n={n}")
    torch.manual_seed(0)
    x = (torch.randn(1, k) * 0.5).half().cuda()
    y0 = run(inner, x, 0)
    y2 = run(inner, x, 2)
    y1 = run(inner, x, 1)
    y2b = run(inner, x, 2)
    stats("mode2 vs ref", y0, y2)
    stats("mode1 vs ref", y0, y1)
    stats("mode2 vs mode1", y2, y1)
    stats("mode2 vs mode2 (determinism)", y2, y2b)
    # Structure probes: is the int8 output the reference with a missing/extra output hadamard?
    had = lambda v: ext.had_r_128(v.half().contiguous(), v.half().contiguous().clone(), None, None, 1.0) or None
    # constant-vector input: output should be smooth in ref
    xc = torch.full((1, k), 0.25).half().cuda()
    yc0 = run(inner, xc, 0); yc2 = run(inner, xc, 2)
    stats("const input mode2 vs ref", yc0, yc2)
    # one-hot input
    xo = torch.zeros(1, k).half().cuda(); xo[0, 5] = 1.0
    yo0 = run(inner, xo, 0); yo2 = run(inner, xo, 2)
    stats("onehot input mode2 vs ref", yo0, yo2)
    # per-128 block correlation: does the first 128-span agree?
    for blk in range(0, min(n, 512), 128):
        a = y0[0, blk:blk+128]; b = y2[0, blk:blk+128]
        c = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        print(f"  block {blk:5d}: corr={c:.3f} ref[:4]={a[:4].tolist()} int8[:4]={b[:4].tolist()}")
    # Check if int8 output is a permutation of ref by sorting
    sa, sb = y0.flatten().sort().values, y2.flatten().sort().values
    stats("sorted values", sa, sb)
os.environ["EXL3_INT8_GEMV"] = "0"
