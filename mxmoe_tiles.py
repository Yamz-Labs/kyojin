"""Tile sweep of the grouped MXFP4 expert GEMV at the gpt-oss-20b decode shape (32 experts, top-4, 2944 x 2944, 1 row)."""
import os
import torch
import triton
from exllamav3.modules.linear import _mxfp4_to_gguf
from exllamav3.foreign_quant import triton_moe_mxfp4 as MX

E, N, K, S = 32, 2944, 2944, 4
g = torch.Generator().manual_seed(1)
packs = [_mxfp4_to_gguf(torch.randint(0, 256, (N, K // 32, 16), dtype=torch.uint8, generator=g).cuda(),
                        torch.randint(118, 128, (N, K // 32), dtype=torch.uint8, generator=g).cuda()) for _ in range(E)]
wtab = torch.tensor([p.data_ptr() for p in packs], dtype=torch.int64, device="cuda")
x = torch.randn((1, K), device="cuda", dtype=torch.half)
sels = [torch.randperm(E, generator=g)[:S].to(torch.int32).cuda() for _ in range(16)]
nbytes = S * N * (K // 32) * 17
res = []
for bn in (8, 16, 32, 64):
    for nw in (1, 2, 4):
        def run(sel):
            y = torch.empty((S, N), dtype=torch.float32, device="cuda")
            MX._mxfp4_moe[(triton.cdiv(N, bn), S)](x, wtab, wtab, sel, y, K, N, S, HAS_B=False, BN=bn, num_warps=nw)
        for s_ in sels:
            run(s_)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        it = 30
        t0.record()
        for i in range(it):
            for s_ in sels:
                run(s_)
        t1.record(); torch.cuda.synchronize()
        us = t0.elapsed_time(t1) * 1000 / (it * len(sels))
        res.append((nbytes / us / 1e3, bn, nw, us))
res.sort(reverse=True)
cur = [r for r in res if r[1] == MX.BN and r[2] == MX.NW][0]
for r in res[:5]:
    print(f"BN {r[1]:3d} nw {r[2]}  {r[0]:6.1f} GB/s  {r[3]:7.1f} us")
print(f"current BN {MX.BN} nw {MX.NW}  {cur[0]:6.1f} GB/s  {cur[3]:7.1f} us")
