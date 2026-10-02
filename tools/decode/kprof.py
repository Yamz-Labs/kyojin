#!/usr/bin/env python
"""Kernel-level timing of the exl3_dec kernels at MiMo shapes; run under rocprofv3 --kernel-trace."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dec import load_ext, rand_trellis, rand_sign_scale, workspace

dev = torch.device("cuda:0")
ext = load_ext(sys.argv[1])
gen = torch.Generator().manual_seed(4)
scratch, counters = workspace(dev)
flush = torch.empty(256 << 20, dtype = torch.uint8, device = dev)
H, I, E, topk = 4096, 2048, 24, 8
for Kg in (2, 2.5):
    L = {}
    for p, (kk, nn) in (("g", (H, I)), ("u", (H, I)), ("d", (I, H))):
        L[p] = [(rand_trellis(kk, nn, Kg, gen).to(dev), rand_sign_scale(kk, gen).to(dev), rand_sign_scale(nn, gen).to(dev)) for _ in range(E)]
    tabs = [torch.tensor([t[i].data_ptr() for t in L[p]], dtype = torch.long, device = dev) for p in "gud" for i in range(3)]
    x = (torch.randn(1, H, generator = gen) * 0.5).half().to(dev)
    sel = torch.tensor([[3, 17, 5, 20, 11, 2, 23, 8]], dtype = torch.long, device = dev)
    w = torch.full((1, topk), 0.125, dtype = torch.half, device = dev)
    out = torch.empty(1, H, dtype = torch.float, device = dev)
    act = torch.empty(topk, I, dtype = torch.half, device = dev)
    for i in range(20):
        flush.fill_(i)
        ext["moe"](x, out, sel, w, *tabs, act, scratch, counters, I, float(Kg), float(Kg))
    torch.cuda.synchronize()
    del L
G = (torch.randn(H, 256, generator = gen) * 0.02).half().to(dev)
bias = torch.zeros(256, device = dev)
sel = torch.empty(1, 8, dtype = torch.long, device = dev); wts = torch.empty(1, 8, dtype = torch.half, device = dev)
for i in range(20):
    flush.fill_(i)
    ext["router"](x, G, bias, sel, wts, scratch, counters, 1.0)
torch.cuda.synchronize()
print("done")
