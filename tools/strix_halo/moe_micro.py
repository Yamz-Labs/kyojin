#!/usr/bin/env python
"""Microbench one MoE layer's GPU path at decode/MTP row counts, on REAL expert weights.

Reports per-forward GPU time for rows R = 1..4 through BlockSparseMLP.forward with the
current env (EXL3_MOE_CFG, EXL3_HIP_PREFILL_MIN_ROWS), the bytes an ideal implementation
reads (unique experts x 3 matrices), and the implied bandwidth. This is the 24%-of-device
kernel: the target is the achievable ~85-140 GB/s single-kernel rate on this GPU.
"""
import os, sys, time, collections
import torch
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model
import exllamav3.modules.block_sparse_mlp as bsm
from torch.profiler import profile, ProfilerActivity

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
LAYER = int(os.environ.get("LAYER", "10"))
config = Config.from_directory(MODEL)
model = Model.from_config(config)

def find_mlp(mod):
    if isinstance(mod, bsm.BlockSparseMLP):
        return mod
    for s in getattr(mod, "modules", []) or []:
        r = find_mlp(s)
        if r is not None:
            return r
    return None

blocks = [m for m in model.modules if find_mlp(m) is not None]
mlp = find_mlp(blocks[LAYER])
print("site:", mlp.key)
mlp.load(torch.device("cuda:0"))

H = 2560
def bytes_per_expert():
    # 3.05 bpw EXL3: gate/up 2560x640 + down 640x2560 = 3 x 1.6384M weights
    w = 3 * 2560 * 640
    K = 3  # bits (mul1 codebook K=3 for experts per earlier profiles)
    return w * K / 8

def run(R, iters=50):
    x = torch.randn(1, R, H, device="cuda", dtype=torch.half)
    params = {"attn_mode": "flash_attn"}
    for _ in range(5):
        mlp.forward(x, params)
    torch.cuda.synchronize()
    junk = torch.empty(96 << 20, device="cuda", dtype=torch.uint8)
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    tot = 0.0
    for _ in range(iters):
        junk.zero_()
        st.record(); mlp.forward(x, params); en.record(); en.synchronize()
        tot += st.elapsed_time(en)
    return tot / iters * 1000

print(f"{'R':>3} {'us/fwd':>8} {'assign':>7} {'ideal MiB':>10} {'GB/s @uniq':>11}")
for R in (1, 2, 3, 4, 8):
    us = run(R)
    assigns = R * 10
    # random x -> assume ~all unique experts (upper bound on bytes)
    mib = assigns * bytes_per_expert() / 2**20
    print(f"{R:>3} {us:>8.1f} {assigns:>7} {mib:>10.1f} {assigns*bytes_per_expert()/us/1e3:>11.1f}")

# kernel split at R=3
for R in (1, 3):
    x = torch.randn(1, R, H, device="cuda", dtype=torch.half)
    junk = torch.empty(96 << 20, device="cuda", dtype=torch.uint8)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(30):
            junk.zero_(); mlp.forward(x, {"attn_mode": "flash_attn"})
        torch.cuda.synchronize()
    rows = [(e.key, e.count, (getattr(e, "self_device_time_total", 0) or 0)) for e in prof.key_averages()]
    rows = [r for r in rows if r[2] > 0 and "zero" not in r[0] and "fill" not in r[0]]
    rows.sort(key=lambda r: -r[2])
    print(f"\nR={R} kernel split (per forward):")
    for k, n, d in rows[:10]:
        print(f"  {d/30:8.1f} us  {n/30:5.1f} calls  {k[:70]}")
