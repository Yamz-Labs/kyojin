#!/usr/bin/env python
"""Isolate the gfx1151 WMMA GEMV: compare one real EXL3 layer's output
against the same layer forced through reconstruct+hgemm.

Loads only ONE safetensors shard's worth of layers (fast), finds the first
LinearEXL3 module, and runs it both ways on identical input.
"""
import os, sys
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext
from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3
import exllamav3.modules.quant.exl3 as exl3mod

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")

print(f"supported={exllamav3_ext.exl3_gemv_supported(0)} "
      f"family={exllamav3_ext.exl3_gemv_wmma_family(0)}", flush=True)

config = Config.from_directory(MODEL)
model = Model.from_config(config)
model.load(progressbar=False)
print("loaded", flush=True)

# Collect LinearEXL3 leaves
found = []
def walk(m, depth=0):
    if isinstance(m, LinearEXL3):
        found.append(m)
        return
    for child in (getattr(m, "modules", None) or []):
        walk(child, depth + 1)
walk(model)
print(f"found {len(found)} LinearEXL3 modules", flush=True)

if not found:
    print("no LinearEXL3 found -- structure differs; aborting")
    sys.exit(1)

torch.manual_seed(0)
bad = 0
for idx in (0, 1, 2, len(found) // 2, len(found) - 1):
    if idx >= len(found):
        continue
    lin = found[idx]
    inf = getattr(lin, "in_features", None)
    outf = getattr(lin, "out_features", None)
    K = getattr(lin, "K", None)
    if not inf:
        continue
    for m in (1, 2, 8):
        x = (torch.randn(m, inf, dtype=torch.float16, device="cuda") * 0.05)
        try:
            exl3mod.EXL3_GEMV_HIP_MAX_M  # sanity: module loaded
            # fast path (as built)
            os.environ["EXL3_GEMV"] = "1"
            exl3mod._hip_gemv_support_cache.clear()
            y_fast = lin.forward(x.clone(), {}).float()
            # force fallback by making the gate report unsupported
            saved = exl3mod._hip_gemv_supported
            exl3mod._hip_gemv_supported = lambda dev: False
            y_ref = lin.forward(x.clone(), {}).float()
            exl3mod._hip_gemv_supported = saved
        except Exception as e:
            print(f"  layer{idx} K={K} {inf}x{outf} m={m}: EXC {type(e).__name__}: {str(e)[:90]}")
            continue
        d = (y_fast - y_ref).abs()
        scale = y_ref.abs().mean().item() + 1e-9
        rel = d.mean().item() / scale
        flag = "OK" if rel < 0.02 else "** MISMATCH **"
        if rel >= 0.02:
            bad += 1
        print(f"  layer{idx} K={K} {inf}x{outf} m={m}: "
              f"max={d.max().item():.5f} mean={d.mean().item():.6f} rel={rel:.3e} {flag}")
        if rel >= 0.02 and m == 1:
            print(f"      fast[0,:6]={y_fast[0,:6].tolist()}")
            print(f"      ref [0,:6]={y_ref[0,:6].tolist()}")

print(f"\nmismatching configs: {bad}")
