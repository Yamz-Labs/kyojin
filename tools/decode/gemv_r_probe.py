#!/usr/bin/env python
"""bit-exactness probe for exl3_dec_gemv_r (R-row dense GEMV) on real MiMo
q/k/v/o projections. Kernel-level only (direct ext.* calls), no attn.py wiring yet.

exl3_dec_gemv_r_multi should be bit-exact vs R independent exl3_dec_gemv_multi calls,
by construction: same pick_ktw/kbs_of call as batch-1, same lane_gemv_r per-row arithmetic
already proven bit-exact for the MoE R-row kernels.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Config, Model  # noqa: E402
import exllamav3.modules.attn as attn_mod  # noqa: E402

DEV = "cuda:0"
ext = attn_mod.ext


def make_rows(R: int, K: int, alpha: float, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    base = torch.randn(K, device=DEV, generator=g)
    x = base[None, :] + alpha * torch.randn(R, K, device=DEV, generator=g)
    return x.half().contiguous()


def alloc_ws(R: int, kbs: int, Ns: list[int]):
    scratch = torch.zeros(R * kbs * sum(Ns), dtype=torch.float, device=DEV)
    strips = sum((n + 511) // 512 for n in Ns)
    counters = torch.zeros(R * strips + 64, dtype=torch.int32, device=DEV)
    return scratch, counters


def test_multi(lins, R, K, alpha, seed):
    trellis = [l.inner.trellis for l in lins]
    suh = [l.inner.suh for l in lins]
    svh = [l.inner.svh for l in lins]
    Ks = [float(l.inner.K) for l in lins]
    Ns = [l.out_features for l in lins]
    x = make_rows(R, K, alpha, seed)

    # Reference: R independent exl3_dec_gemv_multi calls (the proven batch-1 kernel).
    from exllamav3.modules.quant.exl3 import dec_workspace
    scratch1, counters1 = dec_workspace(DEV)
    ref_outs = [torch.empty((R, n), dtype=lins[i].inner.default_out_dtype, device=DEV) for i, n in enumerate(Ns)]
    for i in range(R):
        outs_i = [torch.empty((1, n), dtype=lins[j].inner.default_out_dtype, device=DEV) for j, n in enumerate(Ns)]
        ext.exl3_dec_gemv_multi(x[i:i + 1].contiguous(), trellis, suh, svh,
                                 [o.view(1, -1) for o in outs_i], scratch1, counters1, Ks)
        for j, o in enumerate(outs_i):
            ref_outs[j][i] = o.view(-1)

    # Test: one exl3_dec_gemv_r_multi call over all R rows.
    kbs_ub = 8
    scratch2, counters2 = alloc_ws(R, kbs_ub, Ns)
    test_outs = [torch.empty((R, n), dtype=lins[i].inner.default_out_dtype, device=DEV) for i, n in enumerate(Ns)]
    ext.exl3_dec_gemv_r_multi(x, trellis, suh, svh, test_outs, scratch2, counters2, Ks)

    maxd = 0.0
    for j in range(len(Ns)):
        d = (test_outs[j].float() - ref_outs[j].float()).abs().max().item()
        maxd = max(maxd, d)
    return maxd


def test_head(head, R, K, alpha, seed):
    """Single-matrix test at a shape wide enough that pick_ktw picks ktw=32 (row-chunking,
    rpb < R) -- the untested part of the design."""
    return test_multi([head], R, K, alpha, seed)


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-m", "--model_dir", default=os.path.expanduser("~/models/mimo26-exl3"))
    ap.add_argument("--layers", default="1,12")
    ap.add_argument("--rows", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--head", action="store_true", help="also test lm_head (row-chunking path)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    ok_all = True

    if args.head:
        head = model.modules[-1]
        head.load(torch.device(DEV))
        K = head.in_features
        print(f"\nhead: K={K} N={head.out_features}", flush=True)
        for R in [int(v) for v in args.rows.split(",")]:
            try:
                d = test_head(head, R, K, args.alpha, 5000 + R)
            except Exception as e:  # noqa: BLE001
                print(f"  head R={R}: EXC {e}", flush=True)
                ok_all = False
                continue
            ok = d == 0.0
            ok_all &= ok
            print(f"  head R={R} max|d|={d:.3e} {'OK' if ok else 'FAIL'}", flush=True)
        head.unload()
        torch.cuda.empty_cache()

    for L in [int(v) for v in args.layers.split(",") if v]:
        block = model.modules[L + 1]
        block.attn.load(torch.device(DEV))
        a = block.attn
        cand = [a.q_proj, a.k_proj] + ([] if getattr(a, "use_k_as_v", False) else [a.v_proj])
        K = a.q_proj.in_features
        print(f"\nlayer {L}: K={K} q_out={a.q_proj.out_features} "
              f"k_out={a.k_proj.out_features} v_out={cand[-1].out_features if len(cand) > 2 else 'shared'}",
              flush=True)
        for R in [int(v) for v in args.rows.split(",")]:
            try:
                d = test_multi(cand, R, K, args.alpha, 4000 + R)
            except Exception as e:  # noqa: BLE001
                print(f"  qkv R={R}: EXC {e}", flush=True)
                ok_all = False
                continue
            ok = d == 0.0
            ok_all &= ok
            print(f"  qkv R={R} max|d|={d:.3e} {'OK' if ok else 'FAIL'}", flush=True)
        block.attn.unload()
        torch.cuda.empty_cache()
    print("\nALL_OK" if ok_all else "\nSOME_FAILED", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
