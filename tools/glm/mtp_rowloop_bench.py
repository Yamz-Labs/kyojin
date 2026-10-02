#!/usr/bin/env python
"""Step 2 microbench (GLM MTP verify): R-row router and lm_head variants vs the bsz-1 kernels.

router: the torch fallback (what R=2 verify uses today: the ext has no routing_ds3_nogroup) vs R x exl3_dec_router
        (the plain-decode kernel, once per row into row slices).
gemv:   exl3_dec_gemv_r (one launch, R rows) vs R x exl3_dec_gemv, at lm_head shape.
Prints time per call and bit-identity of each variant vs the bsz-1 reference.
"""
import torch
from exllamav3.ext import exllamav3_ext as ext
from types import SimpleNamespace
from exllamav3.modules.quant.exl3 import dec_workspace
from exllamav3.modules.block_sparse_mlp_routing import _routing_nogroup_torch

dev = torch.device("cuda:0")
torch.manual_seed(0)
H, E, TOPK, SCALE = 4096, 288, 8, 2.5


def bench(fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1000.0


def router(R):
    gate = (torch.randn(H, E, device=dev) * 0.02).half().contiguous()
    bias = (torch.randn(E, device=dev) * 0.01).float().contiguous()
    x = torch.randn(R, H, device=dev).half().contiguous()
    scratch, counters = dec_workspace(dev)
    sel_ref = torch.empty(R, TOPK, dtype=torch.long, device=dev)
    w_ref = torch.empty(R, TOPK, dtype=torch.half, device=dev)

    def loop():
        for r in range(R):
            ext.exl3_dec_router(x[r:r + 1], gate, bias, sel_ref[r], w_ref[r], scratch, counters, SCALE)

    cfg = SimpleNamespace(e_score_correction_bias=bias, num_experts_per_tok=TOPK, routed_scaling_factor=SCALE)
    out = {}

    def batched():
        # what routing_dots runs at bsz > 1 on this build (no routing_ds3_nogroup in the ext)
        scores = torch.sigmoid(torch.matmul(x, gate).float())
        out["sw"] = _routing_nogroup_torch(cfg, x, {}, scores)

    loop(); batched(); torch.cuda.synchronize()
    sel_b, w_b = out["sw"]
    same_sel = all(set(sel_ref[r].tolist()) == set(sel_b[r].tolist()) for r in range(R))
    print(f"RESULT router R={R} loop_us={bench(loop):.1f} torch_us={bench(batched):.1f} "
          f"torch_same_set={same_sel}")


def gemv(R, K, N=154880):
    kt, nt = H // 16, N // 16
    trellis = torch.randint(-32768, 32767, (kt, nt, 16 * K), dtype=torch.int16, device=dev)
    suh = (torch.randn(H, device=dev).sign() * 0.5).half()
    svh = (torch.randn(N, device=dev).sign() * 0.5).half()
    x = torch.randn(R, H, device=dev).half().contiguous()
    scratch, counters = dec_workspace(dev)
    y_loop = torch.empty(R, N, dtype=torch.half, device=dev)
    y_r = torch.empty(R, N, dtype=torch.half, device=dev)

    def loop():
        for r in range(R):
            ext.exl3_dec_gemv(x[r:r + 1], trellis, suh, svh, y_loop[r:r + 1], scratch, counters, K)

    def rrow():
        ext.exl3_dec_gemv_r(x, trellis, suh, svh, y_r, scratch, counters, K)

    loop(); rrow(); torch.cuda.synchronize()
    print(f"RESULT gemv K={K} R={R} N={N} loop_us={bench(loop, 50):.1f} r_us={bench(rrow, 50):.1f} "
          f"bitexact={torch.equal(y_loop, y_r)}")


for R in (1, 2, 3):
    router(R)
for K in (4, 5, 6):
    for R in (1, 2):
        gemv(R, K)
