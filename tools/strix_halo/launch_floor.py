#!/usr/bin/env python3
"""Is the ~14us floor on small GEMVs launch overhead or kernel execution?

gemv_micro.py shows small expert tensors (2560x640, K=3) taking ~13.8us when the
bandwidth roofline says 2.6us -- 19% of peak -- and being completely insensitive
to both m (1..4) and EXL3_MOE_CFG (0/1/2). That flatness says "fixed cost", not
"streaming too slowly". There are two candidates:

  (a) kernel launch / dispatch overhead on the host or in the command processor
  (b) genuine kernel execution that is latency-bound (too few blocks: a 640-wide
      output at 64 n/block is only 10 blocks over 20 CUs)

Distinguish them by timing N back-to-back launches on one stream without
synchronising between them. Launch cost pipelines with execution; if the
per-call time collapses when calls are batched, the floor was dispatch. If it
stays flat, the floor is execution.

Also times an empty kernel (torch.empty_like on a tiny tensor is close enough to
a no-op dispatch) to get this box's raw launch cost for reference.
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch

from exllamav3 import Config, Model
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.util.tensor import g_tensor_cache
import exllamav3.ext as extmod

ext = extmod.exllamav3_ext

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")


def main():
    cfg = Config.from_directory(MODEL)
    model = Model.from_config(cfg)

    # exllamav3 modules are not torch nn.Modules; walk .modules/.inner like
    # gemv_micro.py does.
    model.load()
    found = {}
    stack = list(model.modules)
    while stack:
        mod = stack.pop()
        inner = getattr(mod, "inner", None)
        if isinstance(inner, LinearEXL3):
            key = (inner.in_features, inner.out_features, inner.K)
            found.setdefault(key, (mod.key, inner))
        for c in getattr(mod, "modules", []) or []:
            stack.append(c)

    targets = {}
    for (k_, n_, K_), (key, lin) in found.items():
        if k_ == 2560 and n_ == 640 and K_ == 3:
            targets["expert_small_2560x640_K3"] = (key, lin)
        if k_ == 640 and n_ == 2560 and K_ == 3:
            targets["expert_small_640x2560_K3"] = (key, lin)
        if k_ == 2560 and n_ == 12288:
            targets["attn_big_2560x12288"] = (key, lin)
    if not targets:
        print(f"FAIL: no targets; shapes seen: {sorted(found)[:8]}")
        return 1
    dev = torch.device("cuda:0")

    print("raw dispatch cost reference (empty 1-elem op, 2000 iters):")
    t = torch.ones(1, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2000):
        t.add_(1.0)
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"  {per:.2f}us per trivial kernel (launch-bound floor)\n")

    had = None
    for label, (name, lin) in sorted(targets.items()):
        k = lin.in_features
        n = lin.out_features
        print(f"{label}  (k={k} n={n} K={lin.K})")

        for m in (1, 3):
            x = torch.randn((m, k), device=dev, dtype=torch.half)
            y = torch.empty((m, n), device=dev, dtype=torch.half)
            A_had = g_tensor_cache.get(dev, (m, k), torch.half, "exl3_gemv_a_had")

            def once():
                ext.exl3_gemv(x, lin.trellis, y, lin.suh, A_had, lin.svh,
                              lin.mcg, lin.mul1)

            for _ in range(20):
                once()
            torch.cuda.synchronize()

            row = []
            for batch in (1, 4, 16, 64, 256):
                iters = max(1500 // batch, 8)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    for _ in range(batch):
                        once()
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / (iters * batch) * 1e6
                row.append(f"{batch}:{dt:.2f}us")
            print(f"    m={m}  " + "  ".join(row))
        print()

    print("Interpretation:")
    print("  per-call time FALLS as batch grows  => dispatch-bound (fixable on host)")
    print("  per-call time FLAT as batch grows   => execution-bound (needs more blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
