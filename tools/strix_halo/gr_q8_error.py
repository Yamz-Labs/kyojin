#!/usr/bin/env python
"""Where does the int8 gr_mix error come from? Torch-only emulation of per-row int8 on fn / up
separately vs the fp32 reference, plus per-row outlier stats, on the real site weights."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/kyojin"))
os.environ["EXL3_HIP_GR_MIX_Q8"] = "0"
from exllamav3 import Config, Model
from exllamav3.modules.hyperconnections import GatedResidual
import torch.nn.functional as F

config = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
model = Model.from_config(config)
sites = []
def find(mod):
    if isinstance(mod, GatedResidual): sites.append(mod)
    for s in getattr(mod, "modules", []) or []: find(s)
for m in model.modules: find(m)
site = sites[int(os.environ.get("SITE", "0"))]
site.load(torch.device("cuda:0"))
H, D, LR = site.hc_mult, site.hidden_size, site.rank
print(site.key, "rank", LR)

def q8_rows(w, group=None, qmax=127):
    w = w.float()
    if group is None:
        sc = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
        return torch.round(w / sc).clamp(-qmax, qmax) * sc
    g = w.view(w.shape[0], -1, group)
    sc = g.abs().amax(2, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.round(g / sc).clamp(-qmax, qmax) * sc).view_as(w)
def q4(w, group): return q8_rows(w, group, 7)

def ref(x, down, up, inject):
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + site.rms_eps) * site.norm_w
    flat = normed.flatten(-2)
    t = F.silu(F.linear(flat, down) / H)
    g = torch.sigmoid(F.linear(t, up))
    mixed = (g.unflatten(-1, (H, D)) * normed).mean(-2)
    post = 2 * torch.sigmoid(F.linear(flat, inject) / H)
    return mixed, post, t, F.linear(t, up)

down, up, inj = site.down_h.float(), site.up_h.float(), site.inject_h.float()
for name, w in (("down", down), ("inject", inj), ("up", up)):
    amax = w.abs().amax(1); rms = w.pow(2).mean(1).sqrt()
    print(f"{name:7} shape {tuple(w.shape)}  amax/rms per row: median {(amax/rms).median():.1f} max {(amax/rms).max():.1f}")

torch.manual_seed(0)
x = torch.randn(1, 3, H, D, device="cuda") * 3
m0, p0, t0, g0 = ref(x, down, up, inj)
print(f"ref |mixed| max {m0.abs().max():.3f}  |t| max {t0.abs().max():.2f}  |g_logit| max {g0.abs().max():.1f}")
for label, d2, u2, i2 in (
    ("q8 down only",        q8_rows(down), up, inj),
    ("q8 up only",          down, q8_rows(up), inj),
    ("q8 inject only",      down, up, q8_rows(inj)),
    ("q8 all per-row",      q8_rows(down), q8_rows(up), q8_rows(inj)),
    ("q8 all group128",     q8_rows(down, 128), q8_rows(up, 32), q8_rows(inj, 128)),
    ("q8 down/inj g128, up per-row", q8_rows(down, 128), q8_rows(up), q8_rows(inj, 128)),
    ("q4 g32 all",           q4(down, 32), q4(up, 32), q4(inj, 32)),
    ("q4 g64 all",           q4(down, 64), q4(up, 64), q4(inj, 64)),
    ("q4 g128 down/up, q8 inj", q4(down, 128), q4(up, 32), q8_rows(inj)),
    ("q4 g32 down, q8 up/inj", q4(down, 32), q8_rows(up), q8_rows(inj)),
):
    m, p, t, g = ref(x, d2, u2, i2)
    print(f"{label:32} |dmixed| max {(m-m0).abs().max():.4f} mean {(m-m0).abs().mean():.5f}   |dpost| {(p-p0).abs().max():.5f}   |dt| {(t-t0).abs().max():.4f}  |dg_logit| {(g-g0).abs().max():.3f}")
