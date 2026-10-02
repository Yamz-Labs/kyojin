#!/usr/bin/env python
"""glm-mtp step 4 microbench: exl3_dec_gemv_r (one launch, R rows) vs R x exl3_dec_gemv.

Arms per (K, N, R): loop = R batch-1 launches (reference), r<cap> = one gemv_r launch with
EXL3_GEMV_R_RPB=cap (0 = kernel default rows per block). Prints one RESULT line per shape with
us per call and bit-identity of each arm vs the loop.
"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import dec_workspace

dev = torch.device("cuda:0")
torch.manual_seed(0)
H = 4096


def bench(fn, n):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1000.0


def shape(K, N, Rs, caps, n):
    kt, nt = H // 16, N // 16
    trellis = torch.randint(-32768, 32767, (kt, nt, 16 * K), dtype=torch.int16, device=dev)
    suh = (torch.randn(H, device=dev).sign() * 0.5).half()
    svh = (torch.randn(N, device=dev).sign() * 0.5).half()
    scratch, counters = dec_workspace(dev)
    for R in Rs:
        x = torch.randn(R, H, device=dev).half().contiguous()
        y_loop = torch.empty(R, N, dtype=torch.half, device=dev)
        y_r = torch.empty(R, N, dtype=torch.half, device=dev)

        def loop():
            for r in range(R):
                ext.exl3_dec_gemv(x[r:r + 1], trellis, suh, svh, y_loop[r:r + 1], scratch, counters, K)

        def rrow():
            ext.exl3_dec_gemv_r(x, trellis, suh, svh, y_r, scratch, counters, K)

        res = {"K": K, "N": N, "R": R, "loop_us": None}
        loop(); torch.cuda.synchronize()
        # A B A B interleave: loop, then every cap, twice; keep the min per arm
        for rep in range(2):
            t = bench(loop, n)
            res["loop_us"] = t if res["loop_us"] is None else min(res["loop_us"], t)
            for cap in caps:
                os.environ["EXL3_GEMV_R_RPB"] = str(cap)
                y_r.zero_(); rrow(); torch.cuda.synchronize()
                ok = torch.equal(y_loop, y_r)
                t = bench(rrow, n)
                k = f"r{cap}"
                res[k + "_us"] = t if k + "_us" not in res else min(res[k + "_us"], t)
                res[k + "_exact"] = ok and res.get(k + "_exact", True)
        os.environ.pop("EXL3_GEMV_R_RPB", None)
        print("RESULT", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    caps = (0, 1, 2)
    shape(5, 154880, (1, 2, 3, 4), caps, 30)   # GLM lm_head
    shape(4, 154880, (2, 3, 4), caps, 30)
    shape(4, 4096, (2, 3, 4), caps, 300)
    shape(5, 4096, (2, 3, 4), caps, 300)
