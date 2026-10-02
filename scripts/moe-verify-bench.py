#!/usr/bin/env python
"""BRIEF-14 — MoE cost at verify row counts on REAL MiMo layers (one layer resident, ~2 GB).

Paths, same layer / weights / input:
  fallback  EXL3_HIP_GROUPED_MOE=0: the per-expert loop the dflash branch used so far
  assign    grouped kernel, one GEMV per (token, expert) assignment (mimo-strix fe053fa)
  dedup     grouped kernel, assignments sorted by expert, each picked expert decoded once
            for all of its rows (this branch)

Checks: max |diff| vs fallback, and ROW INVARIANCE (row i of an R-row call bit-equal to the
1-row call on the same input), which is what makes verify rows equal the unassisted R=1 decode.
Rows are correlated (x_i = base + a * noise_i) so consecutive verify tokens share experts the
way real ones do; the mean number of distinct experts per call is printed.

  python scripts/moe-verify-bench.py -m ~/models/mimo26-exl3 --layers 1,12
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin/dflash"))
import torch  # noqa: E402
from exllamav3 import Config, Model  # noqa: E402
import exllamav3.modules.block_sparse_mlp as bsm  # noqa: E402

DEV = "cuda:0"


def set_path(p: str):
    os.environ["EXL3_HIP_GROUPED_MOE"] = "0" if p == "fallback" else "1"
    bsm._MOE_DEDUP = p == "dedup"


def fwd(mlp, x):
    return mlp.forward(x, {"attn_mode": "flash_attn"})


def bench(fn, iters: int) -> float:
    junk = torch.empty(96 << 20, device = DEV, dtype = torch.uint8)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        junk.zero_()
        e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
        e0.record(); fn(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts) * 1000.0


def make_x(R: int, H: int, alpha: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device = DEV).manual_seed(seed)
    base = torch.randn(H, device = DEV, generator = g)
    x = base[None, :] + alpha * torch.randn(R, H, device = DEV, generator = g)
    x = x / x.pow(2).mean(dim = -1, keepdim = True).sqrt()
    return x.half().view(1, R, H)


def distinct_experts(mlp, x) -> int:
    """Distinct experts the router picked for x (captured from the grouped kernel's call)."""
    seen = {}
    real = bsm.ext.exl3_moe_gfx12_k3

    def spy(*a, **k):
        seen["sel"] = a[2].clone()
        return real(*a, **k)
    bsm.ext.exl3_moe_gfx12_k3 = spy
    try:
        set_path("assign")
        fwd(mlp, x)
    finally:
        bsm.ext.exl3_moe_gfx12_k3 = real
    return len(torch.unique(seen["sel"])) if "sel" in seen else -1


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev = False)
    ap.add_argument("-m", "--model_dir", default = os.path.expanduser("~/models/mimo26-exl3"))
    ap.add_argument("--layers", default = "1,12")
    ap.add_argument("--rows", default = "1,2,3,4,5,6,7,8")
    ap.add_argument("--alpha", type = float, default = 0.6)
    ap.add_argument("--iters", type = int, default = 30)
    ap.add_argument("--out", default = "")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    res = {}
    for L in [int(v) for v in args.layers.split(",")]:
        block = model.modules[L + 1]
        mlp = block.mlp
        mlp.load(torch.device(DEV))
        H = mlp.expert_size
        print(f"\nlayer {L}: K={mlp.multi_up.K} grouped={mlp.support_hip_grouped} top_k={mlp.num_experts_per_tok} "
              f"H={H} I={mlp.intermediate_size_padded}", flush = True)
        for R in [int(v) for v in args.rows.split(",")]:
            x = make_x(R, H, args.alpha, 1000 + R)
            outs = {}
            for p in ("fallback", "assign", "dedup"):
                set_path(p)
                outs[p] = fwd(mlp, x).float().clone()
            # row invariance: every row alone, through the same path
            inv = {}
            for p in ("fallback", "assign", "dedup"):
                set_path(p)
                singles = torch.cat([fwd(mlp, x[:, i:i + 1]).float() for i in range(R)], dim = 1)
                inv[p] = bool(torch.equal(singles, outs[p]))
            ref = outs["fallback"]
            scale = ref.abs().max().item() or 1.0
            t = {}
            for p in ("fallback", "assign", "dedup"):
                set_path(p)
                t[p] = bench(lambda: fwd(mlp, x), args.iters)
            dx = distinct_experts(mlp, x)
            r = dict(us = t, distinct = dx,
                     rel_err = {p: (outs[p] - ref).abs().max().item() / scale for p in ("assign", "dedup")},
                     dedup_eq_assign = bool(torch.equal(outs["dedup"], outs["assign"])),
                     row_invariant = inv)
            res[f"L{L}_R{R}"] = r
            print(f"  R={R} distinct={dx:3d}  fallback {t['fallback']:7.0f}  assign {t['assign']:7.0f}  "
                  f"dedup {t['dedup']:7.0f} us | err assign {r['rel_err']['assign']:.2e} dedup "
                  f"{r['rel_err']['dedup']:.2e} dedup==assign {r['dedup_eq_assign']} | row-inv "
                  f"{inv}", flush = True)
        mlp.unload()
        torch.cuda.empty_cache()
    if args.out:
        json.dump(res, open(args.out, "w"), indent = 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
