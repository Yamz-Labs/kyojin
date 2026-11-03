"""Quality check of a tuning cache: per (shape, row class), time the cached autotuner choice against an exhaustive grid of forced kernel
shapes (exl3_gemm force_shape_idx 1..N x force_num_sms in {all SMs, half}). tuned/oracle ~ 1.0 means the cache picks what a hand sweep would.

    EXLLAMAV3_TUNE_CACHE=<file with the pack's winners> python oracle_gemm.py PACK OUT.json [--rows 1,2,...] [--dtype half]
Run under the GPU lock. Needs the engine venv (PYTHONPATH = engine checkout).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autotune  # noqa: E402


def run(pack, out, rows, dtype="half", repeat=60, log=print):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.util import tune_cache
    sig = tune_cache.signature()
    dev = torch.device("cuda", sig["device_index"])
    shapes = autotune.pack_shapes(pack)
    nshape = ext.exl3_gemm_num_kernel_shapes()
    res = []
    torch.manual_seed(0)

    def timeit(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        best = []
        for _ in range(5):                                 # median of 5 bursts: robust to a neighbour's clock changes
            t = time.perf_counter()
            for _ in range(repeat):
                fn()
            torch.cuda.synchronize()
            best.append((time.perf_counter() - t) / repeat * 1e6)
        return statistics.median(best)

    for (i, o, K, mcg, mul1), _n in sorted(shapes["dense"].items(), key=lambda kv: -kv[0][0] * kv[0][1]):
        if float(K) != int(K):
            continue
        K = int(K)
        tr = torch.randint(-32768, 32767, (i // 16, o // 16, 16 * K), dtype=torch.int16, device=dev)
        suh = (torch.randint(0, 2, (i,), device=dev) * 2 - 1).half()
        svh = (torch.randint(0, 2, (o,), device=dev) * 2 - 1).half()
        for r in rows:
            x = torch.randn(r, i, dtype=torch.half, device=dev) * 0.1
            xh = torch.empty_like(x)
            y = torch.empty(r, o, dtype=torch.float if dtype == "float" else torch.half, device=dev)
            tuned_tag = ext.exl3_gemm(x, tr, y, suh, xh, svh, -1, mcg, mul1, 0)
            tuned = timeit(lambda: ext.exl3_gemm(x, tr, y, suh, xh, svh, -1, mcg, mul1, 0))
            grid = {}
            for s in range(1, nshape + 1):
                if not ext.exl3_gemm_shape_compat(s, r, i, o, K):
                    continue
                for sms in (sig["sms"], sig["sms"] // 2):
                    try:
                        grid[f"{s}x{sms}"] = timeit(lambda: ext.exl3_gemm(x, tr, y, suh, xh, svh, s, mcg, mul1, sms))
                    except Exception:   # noqa: BLE001  a candidate that cannot launch at this shape is not part of the grid
                        pass
            best = min(grid, key=grid.get) if grid else None
            res.append(dict(i=i, o=o, K=K, rows=r, tuned_tag=int(tuned_tag), tuned_us=round(tuned, 1), oracle=best,
                            oracle_us=round(grid[best], 1) if best else None, ratio=round(tuned / grid[best], 3) if best else None))
            log(res[-1])
    ratios = [x["ratio"] for x in res if x["ratio"]]
    summ = dict(points=len(res), median_ratio=round(statistics.median(ratios), 3), worst_ratio=max(ratios), over_1_05=sum(1 for q in ratios if q > 1.05))
    Path(out).write_text(json.dumps(dict(summary=summ, points=res), indent=1))
    log(json.dumps(summ))
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("out")
    ap.add_argument("--rows", default="1,2,4,8,16,24,32,48,64")
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--repeat", type=int, default=60)
    a = ap.parse_args()
    run(a.pack, a.out, [int(x) for x in a.rows.split(",")], a.dtype, a.repeat)
