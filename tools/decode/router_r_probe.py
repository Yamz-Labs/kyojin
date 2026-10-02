#!/usr/bin/env python
"""bit-exactness probe for dec_norm_route_r (R-row fused pre-norm + router,
block_sparse_mlp.py) and the EXL3_DEC_MOE_UNION forward() wiring, on real MiMo layers.

dec_norm_route_r loops the real batch-1 exl3_dec_router_norm kernel R times (see its docstring),
so it should be EXACTLY bit-identical to R independent dec_norm_route calls by construction --
this probe is the proof, not a tolerance check.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
import torch  # noqa: E402
from exllamav3 import Config, Model  # noqa: E402
import exllamav3.modules.block_sparse_mlp as bsm  # noqa: E402

DEV = "cuda:0"


def make_rows(R: int, H: int, alpha: float, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    base = torch.randn(H, device=DEV, generator=g)
    y = base[None, :] + alpha * torch.randn(R, H, device=DEV, generator=g)
    x = base[None, :] + alpha * torch.randn(R, H, device=DEV, generator=g)
    return y.half().contiguous(), x.float().contiguous()


def test_router(block, R, H, alpha, seed):
    mlp = block.mlp
    norm = block.mlp_norm
    y_resid, x = make_rows(R, H, alpha, seed)

    # Reference: R independent dec_norm_route (proven batch-1) calls.
    y_ref = torch.empty((R, H), dtype=torch.half, device=DEV)
    sel_ref = torch.empty((R, mlp.num_experts_per_tok), dtype=torch.long, device=DEV)
    wts_ref = torch.empty((R, mlp.num_experts_per_tok), dtype=torch.half, device=DEV)
    x_ref = x.clone()
    for i in range(R):
        params = {}
        yi = mlp.dec_norm_route(norm, y_resid[i], x_ref[i], params)
        if yi is None:
            return None  # fused path ineligible for this layer; skip
        y_ref[i] = yi.view(-1)
        sel_ref[i], wts_ref[i] = params["dec_routed"]

    # Test: one dec_norm_route_r call over all R rows.
    y_resid_t = y_resid.clone()
    x_t = x.clone()
    params_r = {}
    y_test = mlp.dec_norm_route_r(norm, y_resid_t, x_t, params_r)
    if y_test is None:
        return None
    sel_test, wts_test = params_r["dec_routed"]

    dy = (y_test.float() - y_ref.float()).abs().max().item()
    dx = (x_t - x_ref).abs().max().item()
    sel_ok = torch.equal(sel_test, sel_ref)
    dw = (wts_test.float() - wts_ref.float()).abs().max().item()
    ok = dy == 0.0 and dx == 0.0 and sel_ok and dw == 0.0
    print(f"  router  R={R} max|dy|={dy:.3e} max|dx(resid)|={dx:.3e} sel_match={sel_ok} "
          f"max|dw|={dw:.3e} {'OK' if ok else 'FAIL'}", flush=True)
    return ok, y_test, sel_test, wts_test


def test_moe_union(mlp, y_test, sel_test, wts_test, R, H):
    os.environ["EXL3_DEC_MOE_UNION"] = "1"
    out_union = mlp.forward(y_test.view(1, R, H), {
        "attn_mode": "flash_attn", "dec_routed": (sel_test.clone(), wts_test.clone()),
    }).float().clone()

    os.environ["EXL3_DEC_MOE_UNION"] = "0"
    outs = []
    for i in range(R):
        params = {"attn_mode": "flash_attn", "dec_routed": (sel_test[i:i + 1].clone(), wts_test[i:i + 1].clone())}
        outs.append(mlp.forward(y_test[i:i + 1].view(1, 1, H), params).float().clone())
    out_ref = torch.cat(outs, dim=1)

    d = (out_union - out_ref).abs().max().item()
    ok = d == 0.0
    print(f"  moe_union R={R} max|d|={d:.3e} {'OK' if ok else 'FAIL'}", flush=True)

    # accumulate=True / fuse_res: flagged untested. Real decode sets
    # mlp_residual_out whenever nothing else post-processes the sum (transformer.py's plain_add),
    # so production verify very likely takes this path, not the fresh-buffer one just tested.
    g = torch.Generator(device=DEV).manual_seed(9000 + R)
    res0 = torch.randn(R, H, device=DEV, generator=g, dtype=torch.float).contiguous()

    os.environ["EXL3_DEC_MOE_UNION"] = "1"
    res_u = res0.clone()
    p_u = {"attn_mode": "flash_attn", "dec_routed": (sel_test.clone(), wts_test.clone()), "mlp_residual_out": res_u}
    out_u = mlp.forward(y_test.view(1, R, H), p_u)
    fused_u = bool(p_u.get("mlp_residual_done"))
    res_u_final = out_u if fused_u else (res_u + out_u.float().view(R, H))

    os.environ["EXL3_DEC_MOE_UNION"] = "0"
    res_r = res0.clone()
    outs_r = []
    for i in range(R):
        res_i = res_r[i:i + 1].contiguous()
        p_i = {"attn_mode": "flash_attn", "dec_routed": (sel_test[i:i + 1].clone(), wts_test[i:i + 1].clone()),
               "mlp_residual_out": res_i}
        out_i = mlp.forward(y_test[i:i + 1].view(1, 1, H), p_i)
        fused_i = bool(p_i.get("mlp_residual_done"))
        outs_r.append(out_i if fused_i else (res_i.view(1, H) + out_i.float().view(1, H)))
    res_r_final = torch.cat(outs_r, dim=0)

    d2 = (res_u_final.view(R, H).float() - res_r_final.view(R, H).float()).abs().max().item()
    ok2 = d2 == 0.0
    print(f"  moe_union(fuse_res) R={R} fused_union={fused_u} max|d|={d2:.3e} {'OK' if ok2 else 'FAIL'}", flush=True)
    # dec_norm_route_r (R>1) is gated on this same flag (see its docstring) -- restore it before
    # the next R's test_router call, or every R after the first "0" toggle above silently skips.
    os.environ["EXL3_DEC_MOE_UNION"] = "1"
    return ok and ok2


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-m", "--model_dir", default=os.path.expanduser("~/models/mimo26-exl3"))
    ap.add_argument("--layers", default="1,12")
    ap.add_argument("--rows", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--alpha", type=float, default=0.6)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    # dec_norm_route_r (R>1) is gated behind the same flag as the union MoE kernel it feeds
    # (see its docstring) -- this probe means to exercise the fused path, so force it on.
    os.environ["EXL3_DEC_MOE_UNION"] = "1"

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    ok_all = True
    for L in [int(v) for v in args.layers.split(",")]:
        block = model.modules[L + 1]
        block.mlp.load(torch.device(DEV))
        H = block.mlp.expert_size
        print(f"\nlayer {L}: H={H} topk={block.mlp.num_experts_per_tok}", flush=True)
        for R in [int(v) for v in args.rows.split(",")]:
            r = test_router(block, R, H, args.alpha, 3000 + R)
            if r is None:
                print(f"  R={R}: fused path ineligible, skipped", flush=True)
                continue
            ok, y_test, sel_test, wts_test = r
            ok_all &= ok
            if R > 1:
                ok2 = test_moe_union(block.mlp, y_test, sel_test, wts_test, R, H)
                ok_all &= ok2
        block.mlp.unload()
        torch.cuda.empty_cache()
    print("\nALL_OK" if ok_all else "\nSOME_FAILED", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
