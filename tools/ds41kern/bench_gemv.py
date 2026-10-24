"""GB/s of the dense EXL3 decode GEMV on every DS4.1 projection shape (random trellis bytes, batch 1 and 4 rows).
Two kernels: exl3_dec_gemv (needs in_features % 512 == 0: wq_b 1280, shared w2 2304 and indexer wq_b 1280 do NOT qualify) and the
generic WMMA exl3_gemv (the fallback those shapes take, quant/exl3.py:425-470). 190 GB/s is the practical line (k2kern: 177 on VALU).
    flock ~/ds41_work/gpu.lock python tools/ds41kern/bench_gemv.py
"""
import sys, torch
import exllamav3.ext as extmod
from exllamav3.modules.quant.exl3 import dec_workspace
ext = extmod.exllamav3_ext
dev = torch.device("cuda:0")
SHAPES = [("wq_a", 5120, 1280), ("wq_b", 1280, 32768), ("wkv", 5120, 512), ("wo_a grp", 4096, 1024), ("wo_b", 8192, 5120),
          ("shared w1/w3", 5120, 2304), ("shared w2", 2304, 5120), ("idx wq_b", 1280, 4096), ("engram wkv", 6144, 25600),
          ("head", 5120, 129280)]
junk = torch.empty(96 << 20, dtype=torch.uint8, device=dev)


def bench(call, byts, iters=30, reps=3):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    tot = 0.0
    for _ in range(reps):
        junk.zero_()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            call()
        e1.record(); torch.cuda.synchronize()
        tot += e0.elapsed_time(e1) / iters * 1e3
    us = tot / reps
    return us, byts / us * 1e-3


for Kb in (4.0, 2.0):
    print(f"--- K = {Kb} bits")
    for name, k, n in SHAPES:
        if Kb == 2.0 and name not in ("wq_b", "shared w2"):
            continue
        kb = int(Kb)
        tr = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * kb), dtype=torch.int16, device=dev)
        suh = torch.ones(k, dtype=torch.half, device=dev)
        svh = torch.ones(n, dtype=torch.half, device=dev)
        byts = k * n * Kb / 8
        for rows in (1, 4):
            x = torch.randn(rows, k, dtype=torch.half, device=dev) * 0.1
            y = torch.empty(rows, n, dtype=torch.half, device=dev)
            a_had = torch.empty(rows, k, dtype=torch.half, device=dev)
            res = []
            if k % 512 == 0 and rows == 1:
                scratch, counters = dec_workspace(dev)
                us, gb = bench(lambda: ext.exl3_dec_gemv(x, tr, suh, svh, y, scratch, counters, Kb), byts)
                res.append(f"dec_gemv {us:8.1f} us {gb:6.1f} GB/s")
            else:
                res.append("dec_gemv        n/a (K % 512)" if rows == 1 else "dec_gemv       n/a (rows)")
            try:
                us, gb = bench(lambda: ext.exl3_gemv(x, tr, y, suh, a_had, svh, False, True), byts)
                res.append(f"gemv {us:8.1f} us {gb:6.1f} GB/s")
            except Exception as e:
                res.append(f"gemv failed: {str(e)[:60]}")
            print(f"  {name:13s} [{k:5d}->{n:6d}] {byts/1e6:7.1f} MB rows={rows}:  " + "   ".join(res), flush=True)
        del tr
