#!/usr/bin/env python3
"""T80: gr_mix_q8 with R=1..8 rows (EXL3_GR_MIX_Q8_MAXR=8) must equal R single-row calls bitwise (dots, post, mixed). gs 0 and 1."""
import os, sys, time
sys.path.insert(0, os.environ["EXL3_ROOT"])
import torch
from exllamav3.ext import exllamav3_ext as ext
dev = torch.device("cuda:0"); g = torch.Generator(device=dev).manual_seed(3)
H, D, LR = 4, 2560, 320
def run(gs, R, reps=0):
    M = LR + H
    s3 = (torch.randn(R, H, D, device=dev, generator=g)).float().contiguous()
    fq = torch.randint(-127, 128, (M, H * D), dtype=torch.int8, device=dev, generator=g)
    uq = torch.randint(-127, 128, (H, D // 4, LR, 4), dtype=torch.int8, device=dev, generator=g)
    if gs:
        fsc = (torch.rand(M * H * D // 128, device=dev, generator=g) * 0.01 + 1e-3).float()
        usc = (torch.rand(H * D * ((LR + 63) // 64), device=dev, generator=g) * 0.01 + 1e-3).float()
    else:
        fsc = (torch.rand(M, device=dev, generator=g) * 0.01 + 1e-3).float()
        usc = (torch.rand(H * D, device=dev, generator=g) * 0.01 + 1e-3).float()
    w = (torch.rand(H * D, device=dev, generator=g) + 0.5).half()
    def call(s, n):
        dots = torch.full((n, M + 1, H), 7.0, device=dev); post = torch.full((n, H), 7.0, device=dev); mixed = torch.zeros(n, D, dtype=torch.half, device=dev)
        ext.gr_mix_q8(s.contiguous(), fq, fsc, uq, usc, w, 1e-6, dots, post, mixed)
        torch.cuda.synchronize()
        return dots, post, mixed
    d, p, m = call(s3, R)
    bad = 0
    for r in range(R):
        d1, p1, m1 = call(s3[r:r + 1], 1)
        bad += int((d1[0].view(torch.int32) != d[r].view(torch.int32)).sum()) + int((p1[0].view(torch.int32) != p[r].view(torch.int32)).sum()) + int((m1[0].view(torch.int16) != m[r].view(torch.int16)).sum())
    return bad
for gs in (0, 1):
    for R in range(1, 9):
        print(f"gs={gs} R={R} mismatching words vs {R} single-row calls: {run(gs, R)}", flush=True)
