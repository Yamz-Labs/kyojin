#!/usr/bin/env python
"""The GDN b/a projections ([R,2560] @ [2560,48] fp16 -> fp32) go through hipblaslt, which picks
a split-K algorithm (Cijk_*_HSS_* + Cijk_S_PostGSU3) costing ~60 us per call for a 245 KB
weight (~1 us of bandwidth). 36 GDN layers x 2 calls per forward = ~4.3 ms per round, ~6% of
decode. Measure alternatives that stay in torch / existing ext calls.
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3.ext import exllamav3_ext as ext

K, N = 2560, 48
dev = "cuda"
w = (torch.randn(K, N, device=dev) * 0.02).half()
w2 = (torch.randn(K, 2 * N, device=dev) * 0.02).half()
wT = w.t().contiguous()            # (N, K)
w2T = w2.t().contiguous()

def timeit(fn, iters=200):
    junk = torch.empty(64 << 20, device=dev, dtype=torch.uint8)
    for _ in range(5): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    tot = 0.0
    for _ in range(iters):
        junk.zero_()
        st.record(); fn(); en.record(); en.synchronize()
        tot += st.elapsed_time(en)
    return tot / iters * 1000

for R in (1, 3, 4):
    x = torch.randn(R, K, device=dev).half()
    y32 = torch.empty(R, N, device=dev, dtype=torch.float)
    ref = x.float() @ w.float()
    cands = {
        "ext.hgemm (current)":      lambda: ext.hgemm(x, w, y32),
        "matmul half->half":        lambda: torch.matmul(x, w),
        "matmul fp32":              lambda: torch.matmul(x.float(), w.float()),
        "F.linear half (N,K)":      lambda: torch.nn.functional.linear(x, wT),
        "mul+sum fp32 (R,K,1)*(K,N)": lambda: (x.float().unsqueeze(-1) * w.float()).sum(1),
        "einsum half->fp32":        lambda: torch.einsum("rk,kn->rn", x.float(), w.float()),
        "mv per row (N,K)@k":       lambda: torch.stack([torch.mv(wT, x[i]) for i in range(R)]),
        "merged 96: ext.hgemm":     lambda: ext.hgemm(x, w2, torch.empty(R, 2 * N, device=dev, dtype=torch.float)),
        "merged 96: matmul half":   lambda: torch.matmul(x, w2),
        "merged 96: mul+sum fp32":  lambda: (x.float().unsqueeze(-1) * w2.float()).sum(1),
        "merged 96: F.linear half": lambda: torch.nn.functional.linear(x, w2T),
    }
    print(f"\nR={R}")
    for name, fn in cands.items():
        try:
            us = timeit(fn)
            out = fn()
            if out is None: out = y32
            err = (out.float()[:, :N] - ref).abs().max().item() if out.shape[-1] >= N else float("nan")
            print(f"  {name:32} {us:8.1f} us   maxerr {err:.2e}")
        except Exception as e:
            print(f"  {name:32} FAIL {type(e).__name__}: {str(e)[:60]}")
