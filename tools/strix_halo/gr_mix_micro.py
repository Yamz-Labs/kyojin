#!/usr/bin/env python
"""Microbench ext.gr_mix (GatedResidual fused mix) at decode/MTP row counts.

The torch profile puts gr_dots + gr_finalize at ~17% of device time. Both launch a grid
of (.., R) blocks and re-read the fn / upt weights once PER ROW, so at MTP shapes (R=3-4)
the weight traffic is multiplied by R. This measures us/call vs R against the one-read
roofline to see what a row-looped kernel could buy.
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model
from exllamav3.modules.hyperconnections import GatedResidual
from exllamav3.ext import exllamav3_ext as ext

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL)
model = Model.from_config(config)
# find one GatedResidual site and load just it
site = None
for m in model.modules:
    for sub in m.modules if hasattr(m, "modules") else []:
        pass
def find(mod):
    global site
    if isinstance(mod, GatedResidual) and site is None:
        site = mod
    for s in getattr(mod, "modules", []) or []:
        find(s)
for m in model.modules:
    find(m)
assert site is not None
site.load(torch.device("cuda:0"))
H, D, LR = site.hc_mult, site.hidden_size, site.rank
M = site.proj_h.shape[0] + 1
if getattr(site, "use_q8", False):
    fn_bytes = site.fn_q8.numel(); upt_bytes = site.upx_q8.numel()
    def call(s3, dots, post, mixed):
        ext.gr_mix_q8(s3, site.fn_q8, site.fn_scale, site.upx_q8, site.up_scale, site.w_h, site.rms_eps, dots, post, mixed)
    print("path: gr_mix_q8 (int8)")
else:
    fn_bytes = site.fn_h.numel() * 2; upt_bytes = site.upx_h.numel() * 2
    def call(s3, dots, post, mixed):
        ext.gr_mix(s3, site.fn_h, site.upx_h, site.w_h, site.rms_eps, dots, post, mixed)
    print("path: gr_mix (fp16)")
print(f"site={site.key} H={H} D={D} rank={LR} M={M-1}  fn={fn_bytes/2**20:.1f} MiB upt={upt_bytes/2**20:.1f} MiB")
BW = 236e9
print(f"one-read roofline: fn {fn_bytes/BW*1e6:.0f} us, upt {upt_bytes/BW*1e6:.0f} us, both {(fn_bytes+upt_bytes)/BW*1e6:.0f} us")

def bench(R, iters=200):
    s3 = torch.randn(R, H, D, device="cuda", dtype=torch.float)
    dots = torch.empty(R, M, H, device="cuda", dtype=torch.float)
    post = torch.empty(R, H, device="cuda", dtype=torch.float)
    mixed = torch.empty(R, D, device="cuda", dtype=torch.half)
    # flush-ish: touch a big buffer so weights are not L2/MALL resident
    junk = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    for _ in range(5):
        call(s3, dots, post, mixed)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    tot = 0.0
    for _ in range(iters):
        junk.zero_()
        st.record()
        call(s3, dots, post, mixed)
        en.record(); en.synchronize()
        tot += st.elapsed_time(en)
    # correctness vs reference
    post_ref, mixed_ref = site._mix_ref(s3.view(1, R, H, D))
    err_m = (mixed.float() - mixed_ref.view(R, D)).abs().max().item()
    err_p = (post - post_ref.view(R, H)).abs().max().item()
    return tot / iters * 1000, err_m, err_p

print(f"{'R':>3} {'us/call':>9} {'x roofline':>11} {'max|dmixed|':>12} {'max|dpost|':>11}")
for R in (1, 2, 3, 4, 5, 8):
    us, em, ep = bench(R)
    print(f"{R:>3} {us:>9.1f} {us/((fn_bytes+upt_bytes)/BW*1e6):>11.2f} {em:>12.4f} {ep:>11.5f}")

# Split the two kernels with the torch profiler (device time per kernel)
from torch.profiler import profile, ProfilerActivity
for R in (1, 3):
    s3 = torch.randn(R, H, D, device="cuda", dtype=torch.float)
    dots = torch.empty(R, M, H, device="cuda", dtype=torch.float)
    post = torch.empty(R, H, device="cuda", dtype=torch.float)
    mixed = torch.empty(R, D, device="cuda", dtype=torch.half)
    junk = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(50):
            junk.zero_()
            call(s3, dots, post, mixed)
        torch.cuda.synchronize()
    for e in prof.key_averages():
        if "gr_" in e.key:
            d = (getattr(e, "self_device_time_total", 0) or 0)
            print(f"R={R} {e.key[:40]:40} {e.count:4d} calls  {d/e.count:8.1f} us/call")
