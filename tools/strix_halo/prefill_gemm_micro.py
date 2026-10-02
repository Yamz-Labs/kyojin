#!/usr/bin/env python
"""Prefill GEMM microbench: [M,2560] @ [2560,N] fp16 -> fp32 (what reconstruct_hgemm issues).
hipblaslt picked Cijk_..._HSS_MT64x32x8 (tiny tile) at 6.3 ms for M=2048 N=10240 = 17 TFLOP/s,
vs ~31 measured peak. Compare: fp32-out (current), fp16-out then cast, torch.matmul fp16, and
different M chunks.
"""
import torch, sys, os, time
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3.ext import exllamav3_ext as ext
dev = "cuda"
def t(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    tot = 0.0
    for _ in range(it):
        s.record(); fn(); e.record(); e.synchronize(); tot += s.elapsed_time(e)
    return tot / it
K = 2560
for M, N in ((2048, 10240), (2048, 6144), (2048, 2560), (1024, 10240), (4096, 10240), (2048, 640)):
    a = torch.randn(M, K, device=dev, dtype=torch.half); w = torch.randn(K, N, device=dev, dtype=torch.half)
    y32 = torch.empty(M, N, device=dev, dtype=torch.float); y16 = torch.empty(M, N, device=dev, dtype=torch.half)
    flop = 2 * M * N * K
    r = {}
    r["hgemm fp32 out (current)"] = t(lambda: ext.hgemm(a, w, y32))
    r["hgemm fp16 out"] = t(lambda: ext.hgemm(a, w, y16))
    r["torch.matmul fp16"] = t(lambda: torch.matmul(a, w))
    r["matmul fp16 + .float()"] = t(lambda: torch.matmul(a, w).float())
    r["F.linear fp16 (w.T contig)"] = None
    wT = w.t().contiguous()
    r["F.linear fp16 (w.T contig)"] = t(lambda: torch.nn.functional.linear(a, wT))
    r["addmm fp16->fp32 via out"] = t(lambda: torch.mm(a, w, out=y16))
    print(f"\nM={M} N={N} K={K}  ({flop/1e9:.0f} GFLOP)")
    for k, ms in r.items():
        print(f"  {k:30} {ms:8.2f} ms  {flop/ms/1e9:6.1f} TFLOP/s")
