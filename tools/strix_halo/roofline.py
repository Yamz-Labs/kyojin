#!/usr/bin/env python
"""Is 35 tok/s physically reachable on gfx1151? Establish the roofline.

Decode on an MoE model is dominated by streaming the ACTIVE expert weights once
per token. So:

    max_tok_s  =  achievable_read_bandwidth / active_bytes_per_token

This measures both terms instead of guessing:
  1. achievable DRAM read bandwidth (large-tensor reduction, several sizes)
  2. active bytes per token, computed from the real model config
     (top-k experts x MoE layers x 3 matrices x bpw, plus dense/attention)

If the measured decode rate sits far below the roofline, the kernel is
inefficient and there is headroom. If it sits at the roofline, no amount of
kernel work will help and the answer is "not on this hardware".
"""
import json, os, sys, time
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")

print("=" * 70)
print("1. ACHIEVABLE DRAM READ BANDWIDTH (gfx1151, LPDDR5X unified)")
print("=" * 70)
dev = torch.device("cuda:0")
for gib in (1, 2, 4, 8):
    n = int(gib * 2**30 // 2)          # fp16 elements
    x = torch.empty(n, dtype=torch.float16, device=dev).fill_(1.0)
    torch.cuda.synchronize()
    # warm
    for _ in range(2):
        x.sum()
    torch.cuda.synchronize()
    reps = 5
    t0 = time.time()
    for _ in range(reps):
        x.sum()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    gbs = (n * 2) / dt / 1e9
    print(f"  read {gib} GiB via sum(): {dt*1000:7.2f} ms  ->  {gbs:7.1f} GB/s")
    del x
    torch.cuda.empty_cache()

# a copy (read+write) for comparison
n = int(2 * 2**30 // 2)
a = torch.empty(n, dtype=torch.float16, device=dev).fill_(1.0)
b = torch.empty_like(a)
torch.cuda.synchronize()
for _ in range(2):
    b.copy_(a)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    b.copy_(a)
torch.cuda.synchronize()
dt = (time.time() - t0) / 5
print(f"  copy 2 GiB (r+w):        {dt*1000:7.2f} ms  ->  "
      f"{(n*2*2)/dt/1e9:7.1f} GB/s effective")
del a, b
torch.cuda.empty_cache()

print()
print("=" * 70)
print("2. ACTIVE BYTES PER TOKEN (from the real config)")
print("=" * 70)

cfg_path = os.path.join(MODEL, "config.json")
cfg = json.load(open(cfg_path))


def g(*names, default=None):
    for n in names:
        if n in cfg:
            return cfg[n]
        t = cfg.get("text_config") or {}
        if n in t:
            return t[n]
    return default


hidden = g("hidden_size", default=2560)
n_layers = g("num_hidden_layers", default=None)
n_experts = g("num_experts", "n_routed_experts", "num_local_experts", default=512)
top_k = g("num_experts_per_tok", default=10)
moe_inter = g("moe_intermediate_size", default=768)
inter = g("intermediate_size", default=None)
vocab = g("vocab_size", default=None)

print(f"  hidden={hidden} layers={n_layers} experts={n_experts} top_k={top_k}")
print(f"  moe_intermediate={moe_inter} intermediate={inter} vocab={vocab}")

# bitrate from the pack
BPW = 3.05

# routed experts: 3 matrices each (gate, up: hidden->moe_inter; down: moe_inter->hidden)
params_per_expert = 3 * hidden * moe_inter
bytes_per_expert = params_per_expert * BPW / 8
print()
print(f"  params/expert      = {params_per_expert/1e6:.2f} M")
print(f"  bytes/expert       = {bytes_per_expert/2**20:.2f} MiB  (at {BPW} bpw)")

# How many layers actually have routed experts? Count from the index.
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
import re
expert_layers = set()
for k in idx:
    m = re.search(r"layers\.(\d+)\..*(experts|mlp\.experts)", k)
    if m:
        expert_layers.add(int(m.group(1)))
n_moe_layers = len(expert_layers) or n_layers
print(f"  layers with routed experts (from index) = {n_moe_layers}")

active_expert_bytes = top_k * bytes_per_expert * n_moe_layers
print(f"  ACTIVE expert bytes/token = {active_expert_bytes/2**30:.3f} GiB")

# everything that is read every token regardless (attention, shared expert,
# norms, head). Approximate from total non-expert tensor bytes in the index.
total_expert_bytes = n_experts * bytes_per_expert * n_moe_layers
model_bytes = sum(os.path.getsize(os.path.join(MODEL, f))
                  for f in os.listdir(MODEL) if f.endswith(".safetensors")
                  and not f.startswith("ngram"))
dense_bytes = max(model_bytes - total_expert_bytes, 0)
print(f"  total expert bytes        = {total_expert_bytes/2**30:.2f} GiB")
print(f"  model file bytes (no ngram) = {model_bytes/2**30:.2f} GiB")
print(f"  => dense/always-read bytes  = {dense_bytes/2**30:.3f} GiB")

per_token = active_expert_bytes + dense_bytes
print()
print(f"  TOTAL bytes read per token = {per_token/2**30:.3f} GiB")

print()
print("=" * 70)
print("3. ROOFLINE")
print("=" * 70)
for bw in (100, 150, 200, 256):
    print(f"  at {bw:3d} GB/s achievable -> max {bw*1e9/per_token:6.1f} tok/s")
print()
print(f"  measured now: 29.2 tok/s  => {29.2*per_token/1e9:.1f} GB/s effective")
print(f"  target:       35.0 tok/s  => {35.0*per_token/1e9:.1f} GB/s effective")
