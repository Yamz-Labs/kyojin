#!/usr/bin/env python3
"""Compare experimental K6/K8 MoE half-layers with the unfused implementation.

Run from the patched checkout, with production stopped. Loads only one block
at a time. No model files, service configuration, or repository files are edited.
Synthetic residual inputs exercise real weights, not full-model activations.
Any numerical difference is reported for review, not silently accepted.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-rel-error", type=float, default=1e-3,
                        help="largest accepted relative update RMS error against the unfused path (default 1e-3; "
                             "0 requires bit-exact outputs)")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    assert (root / "tools/qwen/serve.py").is_file(), "Run from the experimental checkout"
    sys.path.insert(0, str(root))
    spec = importlib.util.find_spec("exllamav3_ext")
    assert spec and Path(spec.origin).resolve().parent == root, "Wrong extension path"
    print("Extension:", spec.origin, flush=True)

    # Apply the same defaults and sidecar selection as production before engine imports.
    from tools.qwen.serve import SERVE_ENV, configure_hc_sidecar
    for key, value in SERVE_ENV:
        os.environ.setdefault(key, value)
    os.environ["EXL3_MOE_FUSED_K6S8"] = "1"
    os.environ["EXL3_MOE_FUSED"] = "1"
    os.environ["EXL3_MOE_FUSED_LAYOUT"] = "native"
    os.environ["EXL3_MOE_FUSED9"] = "1441792"
    os.environ["EXL3_VERIFY_ROW_SPLIT"] = "2"
    print(configure_hc_sidecar(args.model), flush=True)

    import torch
    import exllamav3
    from exllamav3 import Config, Model
    from exllamav3.modules.moe_fused import FusedMoEHalf
    assert Path(exllamav3.__file__).resolve().parent == root / "exllamav3"
    assert torch.version.hip and hasattr(torch.ops.mf9, "half")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    dev = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(dev), flush=True)
    cfg = Config.from_directory(args.model)
    models = [Model.from_config(cfg), Model.from_config(cfg, component="mtp")]
    selected = []
    for model in models:
        blocks = [m for m in model.modules if getattr(m, "mlp_hc", None) is not None]
        assert blocks, f"No MoE blocks in {model.component}"
        indices = sorted({0, len(blocks) // 2, len(blocks) - 1})
        selected.extend(blocks[i] for i in indices)

    calls = [0]
    original_call = FusedMoEHalf.__call__

    def counted_call(self, x):
        calls[0] += 1
        return original_call(self, x)

    FusedMoEHalf.__call__ = counted_call
    comparisons = differences = 0
    worst = 0.0
    try:
        for block in selected:
            print("Loading block:", block.key, flush=True)
            block.load(dev)
            try:
                assert block.ablit is None, "Runtime projection needs a separate compatibility check"
                assert block.mlp_resid_scalar is None
                reason = FusedMoEHalf.why_not(block)
                assert reason is None, reason
                plan = FusedMoEHalf(block)
                assert plan.supported and plan.native and (plan.RB, plan.SB) == (6, 8)
                block._moe_fused = plan
                block._moe_fused_tried = True
                # Guard the workspace without changing its contents or alignment.
                size = plan.ws.numel()
                workspace = torch.full((size + 4096,), 171, dtype=torch.uint8, device=dev)
                workspace[:size].copy_(plan.ws)
                plan.ws = workspace[:size]
                print("Shape:", plan.shape, "extra_bytes:", plan.extra_bytes, flush=True)

                def run(x, fused):
                    block._moe_fused_on = fused
                    storage = torch.full((x.numel() + 128,), 12345.0, dtype=torch.float32, device=dev)
                    view = storage[64:-64].view_as(x)
                    view.copy_(x)
                    before = calls[0]
                    if fused:
                        assert plan.applicable(view, {})
                    out = block._forward_mlp(view, None, {"attn_mode": "flash_attn", "dflash_verify": True})
                    torch.cuda.synchronize()
                    assert (calls[0] > before) == fused, "Unexpected fused/fallback dispatch"
                    assert bool(torch.isfinite(out).all()), "Nonfinite output"
                    assert bool((storage[:64] == 12345).all()) and bool((storage[-64:] == 12345).all()), "Residual guard overwritten"
                    assert bool((workspace[size:] == 171).all()), "Workspace guard overwritten"
                    assert plan.ctl().tolist() == [0, 0, 0], "Kernel control error"
                    return out.clone()

                for seed in (1234, 5678):
                    generator = torch.Generator(device=dev).manual_seed(seed)
                    base = torch.randn((1, 8, plan.H, plan.D), generator=generator, device=dev)
                    # Include correlated streams, larger activations, and identical rows.
                    base = base + base[:, :, :1, :]
                    if seed == 5678:
                        base *= 8
                        base[:, 1::2] = base[:, ::2].clone()
                    for rows in range(1, 9):
                        x = base[:, :rows].contiguous()
                        ref = run(x, False)
                        got = run(x, True)
                        repeat = run(x, True)
                        assert torch.equal(got, repeat), "Repeated fused calls differ"
                        # Two-session shape must preserve the same per-row computation.
                        if rows % 2 == 0:
                            batched = run(x.view(2, rows // 2, plan.H, plan.D), True).view_as(x)
                            assert torch.equal(got, batched), "Batch reshape changes fused result"
                        error = (got.double() - ref.double())
                        update = ref.double() - x.double()
                        rms = lambda t: t.square().mean().sqrt().item()
                        exact = torch.equal(got, ref)
                        comparisons += 1
                        differences += not exact
                        relative = rms(error) / max(rms(update), 1e-30)
                        worst = max(worst, relative)
                        print(json.dumps({
                            "block": block.key, "seed": seed, "rows": rows,
                            "exact": exact, "max_abs_error": error.abs().max().item(),
                            "error_rms": rms(error), "reference_update_rms": rms(update),
                            "relative_update_rms_error": relative,
                        }), flush=True)
                print("Block checks complete:", block.key, flush=True)
                del run, plan, workspace, x, ref, got, repeat, base
            finally:
                block._moe_fused = None
                block._moe_fused_tried = False
                block.unload()
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        FusedMoEHalf.__call__ = original_call
    print(f"Comparisons: {comparisons}; non-exact: {differences}; worst relative update RMS error: {worst:.3e}", flush=True)
    print("PASS: guards, finite outputs, repeatability, and batch reshape checks.", flush=True)
    if worst > args.max_rel_error or (args.max_rel_error == 0 and differences):
        print(f"REVIEW REQUIRED: worst relative error {worst:.3e} exceeds --max-rel-error {args.max_rel_error:g}; "
              "do not enable in production yet.", flush=True)
        return 2
    if differences:
        print(f"PASS: {differences} of {comparisons} outputs differ from the unfused path at rounding level "
              f"(worst {worst:.3e} <= {args.max_rel_error:g}); see tools/qwen/K6K8.md, Numerical acceptance.", flush=True)
    else:
        print("PASS: all tested half-layer outputs exactly match the unfused path.", flush=True)
    print("Full-model and performance checks are still required.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
