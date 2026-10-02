#!/usr/bin/env python
"""row-invariance probe for exl3_dec_moe_union (R<=8) on real MiMo layers.

v2: the first version's reference re-routed every row at bsz=1, independently of the R-row
capture -- confounding "kernel not bit-exact" with "router not row-invariant" (an earlier probe already
suspected the latter). Fixed by forcing the SAME (selected, weights) on both sides via
params["dec_routed"], which forward() pops before routing. Also: R=1 for a clean kernel-vs-kernel
check, a same-input-twice noise floor, and a separate router-row-invariance measurement (needed
either way, since moe-verify-bench.py's row_inv check routes inside fwd() too).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Config, Model  # noqa: E402
import exllamav3.modules.block_sparse_mlp as bsm  # noqa: E402

DEV = "cuda:0"
ext = bsm.ext


def make_x(R: int, H: int, alpha: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    base = torch.randn(H, device=DEV, generator=g)
    x = base[None, :] + alpha * torch.randn(R, H, device=DEV, generator=g)
    x = x / x.pow(2).mean(dim=-1, keepdim=True).sqrt()
    return x.half().view(1, R, H)


def capture_routing(mlp, x, R):
    """Real routing for x: R==1 goes through the fused batch-1 path (exl3_dec_moe), R>1 through
    the grouped R-row path (exl3_moe_gfx12_k3). Spy on whichever one actually fires, and save the
    FULL call args so the caller can replay the real kernel call directly (bench()-timed region)
    instead of timing it through a full Python mlp.forward()."""
    seen = {}

    def make_spy(real):
        def spy(*a, **k):
            seen["sel"] = a[2].clone()
            seen["wts"] = a[3].clone()
            seen["args"] = a
            seen["kwargs"] = k
            return real(*a, **k)
        return spy

    if R == 1:
        real = ext.exl3_dec_moe
        ext.exl3_dec_moe = make_spy(real)
        try:
            mlp.forward(x, {"attn_mode": "flash_attn"})
        finally:
            ext.exl3_dec_moe = real
    else:
        real = ext.exl3_moe_gfx12_k3
        ext.exl3_moe_gfx12_k3 = make_spy(real)
        os.environ["EXL3_HIP_GROUPED_MOE"] = "1"
        os.environ["EXL3_HIP_GROUPED_MOE_MULTIROW"] = "1"
        try:
            mlp.forward(x, {"attn_mode": "flash_attn"})
        finally:
            ext.exl3_moe_gfx12_k3 = real
    if "sel" not in seen:
        raise RuntimeError(f"R={R} routed path did not fire -- check eligibility gates")
    return seen["sel"], seen["wts"], seen["args"], seen["kwargs"]


def router_row_invariance(mlp, x, selected, weights, R):
    """Router-only check: does the model's bsz=1 router reproduce the R-row capture's picks?"""
    sel1, wts1 = [], []
    for i in range(R):
        seen = {}
        real = ext.exl3_dec_moe

        def spy(y, out, sel, wts, *rest):
            seen["sel"] = sel.clone(); seen["wts"] = wts.clone()
            return real(y, out, sel, wts, *rest)

        ext.exl3_dec_moe = spy
        try:
            mlp.forward(x[:, i:i + 1], {"attn_mode": "flash_attn"})
        finally:
            ext.exl3_dec_moe = real
        if "sel" in seen:
            sel1.append(seen["sel"]); wts1.append(seen["wts"])
    if len(sel1) != R:
        return None  # bsz=1 fused path didn't fire (e.g. K mismatch); can't isolate the router
    sel1 = torch.cat(sel1, dim=0)
    wts1 = torch.cat(wts1, dim=0)
    # Align by expert id, not position: bsz=1 and bsz=R top-k can return the same (expert, weight)
    # pairs in a different k order (ties), which would show as a false "weight mismatch" if
    # compared positionally. row_mismatch: same expert SET. max_dw_aligned: weight diff per
    # expert after sorting both sides by expert id. order_differs: whether the k-order itself
    # differs (this is what the combine's sequential k-order sum would actually be sensitive to).
    row_mismatch = []
    max_dw_aligned = 0.0
    order_differs = []
    for i in range(R):
        s1, o1 = torch.sort(sel1[i]); s2, o2 = torch.sort(selected[i])
        row_mismatch.append(bool(not torch.equal(s1, s2)))
        if row_mismatch[-1]:
            continue
        w1 = wts1[i][o1].float(); w2 = weights[i][o2].float()
        max_dw_aligned = max(max_dw_aligned, (w1 - w2).abs().max().item())
        order_differs.append(bool(not torch.equal(sel1[i], selected[i])))
    return row_mismatch, max_dw_aligned, order_differs


def batch1_reference_routed(mlp, x, selected, weights, R):
    """R independent bsz=1 forwards, EACH forced to the R-row capture's own (selected, weights)
    for that row via dec_routed -- isolates the kernel from the router."""
    outs = []
    for i in range(R):
        params = {
            "attn_mode": "flash_attn",
            "dec_routed": (selected[i:i + 1].contiguous(), weights[i:i + 1].contiguous()),
        }
        outs.append(mlp.forward(x[:, i:i + 1], params).float().clone())
    return torch.cat(outs, dim=1)


def bench(fn, iters: int = 20) -> float:
    junk = torch.empty(96 << 20, device=DEV, dtype=torch.uint8)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        junk.zero_()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts) * 1000.0


def alloc_union_workspace(mlp, R):
    H = mlp.expert_size
    I = mlp.intermediate_size_padded
    topk = mlp.num_experts_per_tok
    out = torch.zeros((R, H), dtype=torch.float, device=DEV)
    act = torch.zeros((R * topk, I), dtype=torch.half, device=DEV)
    down_part = torch.zeros((R * topk, H), dtype=torch.float, device=DEV)
    strips_I = I // 512
    strips_H = H // 512
    kbs_ub = 8  # generous upper bound on kbs the host function may pick
    scratch = torch.zeros(R * topk * max(2 * kbs_ub * I, kbs_ub * H), dtype=torch.float, device=DEV)
    counters = torch.zeros(R * topk * (strips_I + strips_H) + 64, dtype=torch.int32, device=DEV)
    return out, act, down_part, scratch, counters


def run_union(mlp, x, selected, weights, R, ws=None, dev=False, sync=True):
    H = mlp.expert_size
    I = mlp.intermediate_size_padded
    y = x.view(R, H).half().contiguous()
    out, act, down_part, scratch, counters = ws if ws is not None else alloc_union_workspace(mlp, R)
    ext.exl3_dec_moe_union(
        y, out, selected.contiguous(), weights.contiguous(),
        mlp.multi_gate.ptrs_trellis, mlp.multi_gate.ptrs_suh, mlp.multi_gate.ptrs_svh,
        mlp.multi_up.ptrs_trellis, mlp.multi_up.ptrs_suh, mlp.multi_up.ptrs_svh,
        mlp.multi_down.ptrs_trellis, mlp.multi_down.ptrs_suh, mlp.multi_down.ptrs_svh,
        act, down_part, scratch, counters,
        I, float(mlp.multi_up.K), float(mlp.multi_down.K), False, dev,
    )
    if not sync:
        return None
    torch.cuda.synchronize()
    return out.view(1, R, H).clone()


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-m", "--model_dir", default=os.path.expanduser("~/models/mimo26-exl3"))
    ap.add_argument("--layers", default="1,12")
    ap.add_argument("--rows", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--alpha", type=float, default=0.6)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    ok_all = True
    for L in [int(v) for v in args.layers.split(",")]:
        block = model.modules[L + 1]
        mlp = block.mlp
        mlp.load(torch.device(DEV))
        H = mlp.expert_size
        print(f"\nlayer {L}: K={mlp.multi_up.K} H={H} I={mlp.intermediate_size_padded} "
              f"topk={mlp.num_experts_per_tok}", flush=True)
        for R in [int(v) for v in args.rows.split(",")]:
            x = make_x(R, H, args.alpha, 2000 + R)
            selected, weights, route_args, route_kwargs = capture_routing(mlp, x, R)

            # noise floor: same routed bsz=1 call, twice
            ref_a = batch1_reference_routed(mlp, x, selected, weights, R)
            ref_b = batch1_reference_routed(mlp, x, selected, weights, R)
            noise = (ref_a - ref_b).abs().max().item()

            got = run_union(mlp, x, selected, weights, R)
            d = (got - ref_a).abs()
            maxd = d.max().item()
            row_ok = [bool((d[:, i] == 0).all()) for i in range(R)]
            ok = maxd == 0.0
            ok_all &= ok

            got_dev = run_union(mlp, x, selected, weights, R, dev=True)
            dd = (got_dev - ref_a).abs().max().item()
            ok_dev = dd == 0.0
            ok_all &= ok_dev
            print(f"  R={R} device-table union max|d| vs batch1 = {dd:.3e} {'OK' if ok_dev else 'FAIL'}", flush=True)

            rinv = router_row_invariance(mlp, x, selected, weights, R) if R > 1 else None
            rinv_s = "" if rinv is None else (
                f" router_expert_set_mismatch={rinv[0]} max|dw|_aligned={rinv[1]:.2e} "
                f"k_order_differs={rinv[2]}")
            print(f"  R={R} max|d|={maxd:.3e} noise_floor={noise:.3e} row_ok={row_ok} "
                  f"{'OK' if ok else 'FAIL'}{rinv_s}", flush=True)

            # Kernel-only timing: preallocated workspace, direct ext.* replay of the exact
            # captured call args -- no Python forward()/routing/allocation in the timed region.
            ws = alloc_union_workspace(mlp, R)
            t_union = bench(lambda: run_union(mlp, x, selected, weights, R, ws))
            t_dev = bench(lambda: run_union(mlp, x, selected, weights, R, ws, dev=True))
            print(f"        union(device table) {t_dev:7.0f}us", flush=True)
            if R == 1:
                t_ref = bench(lambda: ext.exl3_dec_moe(*route_args, **route_kwargs))
                ref_label = "batch1(kernel)"
            else:
                t_ref = bench(lambda: ext.exl3_moe_gfx12_k3(*route_args, **route_kwargs))
                ref_label = "dedup(kernel)"
            uexp = torch.unique(selected[selected >= 0])
            print(f"        union {t_union:7.0f}us  {ref_label} {t_ref:7.0f}us"
                  f"  U={uexp.numel()}/{R * mlp.num_experts_per_tok}", flush=True)
        mlp.unload()
        torch.cuda.empty_cache()
    print("\nALL_OK" if ok_all else "\nSOME_FAILED", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
