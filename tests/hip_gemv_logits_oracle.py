import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "tests" / "hip_gemv_logits_worker.py"
DEFAULT_MODEL = os.path.expanduser("~/Models/Qwen3.8-27B-exl3")
MODEL = os.environ.get("EXL3_TEST_MODEL", DEFAULT_MODEL)
STEPS = 16
PROMPT = "The capital of France is"


def _run_worker(mode, out_path, forced_path=None):
    env = os.environ.copy()
    env["EXL3_GEMV"] = "0" if mode == "fallback" else "2"
    env["PYTHONPATH"] = str(REPO_ROOT) if not env.get("PYTHONPATH") else str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
    cmd = [sys.executable, str(WORKER), mode, str(out_path)]
    if forced_path is not None:
        cmd.append(str(forced_path))
    subprocess.run(cmd, check=True, env=env, timeout=600)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fallback_path = tmpdir / "fallback.pt"
        forced_tokens_path = tmpdir / "forced_tokens.pt"
        gemv_path = tmpdir / "gemv.pt"
        _run_worker("fallback", fallback_path)
        fallback = torch.load(fallback_path, map_location="cpu")
        torch.save(fallback["token_ids"].clone(), forced_tokens_path)
        _run_worker("gemv", gemv_path, forced_tokens_path)
        gemv = torch.load(gemv_path, map_location="cpu")

    fallback_tokens = fallback["token_ids"]
    gemv_tokens = gemv["token_ids"]
    fallback_logits = fallback["logits"]
    gemv_logits = gemv["logits"]
    fallback_counts = fallback["counts"]
    gemv_counts = gemv["counts"]

    assert fallback_tokens.shape == (1, STEPS)
    assert gemv_tokens.shape == (1, STEPS)
    assert fallback_logits.shape == gemv_logits.shape == (1, STEPS, fallback_logits.shape[2])
    assert torch.equal(gemv_tokens, fallback_tokens)
    assert fallback_counts["gemv"] == 0
    assert fallback_counts["reconstruct"] > 0
    assert fallback_counts["hgemm"] > 0
    if Path(MODEL).name == "Qwen3.8-27B-exl3":
        # The known Qwen3.8-27B fixture produces about 6,813 GEMV calls. Remaining hgemm
        # calls are independently ineligible architecture paths, not EXL3 reconstruction.
        assert gemv_counts["gemv"] >= 5000
        assert gemv_counts["gemv"] >= 2.5 * gemv_counts["hgemm"]
    else:
        assert gemv_counts["gemv"] > 0
    assert gemv_counts["reconstruct"] == 0
    assert gemv_counts["reconstruct_slice"] == 0
    assert gemv_counts["reconstruct_had_slice"] == 0

    for logits in (fallback_logits, gemv_logits):
        assert not torch.isnan(logits).any()
        assert not torch.isposinf(logits).any()

    fallback_finite = torch.isfinite(fallback_logits)
    gemv_finite = torch.isfinite(gemv_logits)
    mismatch = fallback_finite ^ gemv_finite
    allowed = torch.isneginf(fallback_logits) & gemv_finite
    assert not (mismatch & ~allowed).any()
    assert not (fallback_finite & ~gemv_finite).any()
    if allowed.any():
        step_fallback_top1 = fallback_logits.float().amax(dim=-1, keepdim=True)
        assert torch.all(gemv_logits[allowed].float() <= (step_fallback_top1.expand_as(gemv_logits)[allowed] - 10.0))
        gap_mass = (torch.softmax(gemv_logits.float(), dim=-1) * allowed).sum(dim=-1)
        assert gap_mass.max().item() <= 1e-6

    fallback_top1 = fallback_logits.argmax(dim=-1)
    gemv_top1 = gemv_logits.argmax(dim=-1)
    top1_agree = (fallback_top1 == gemv_top1).sum().item()

    finite = fallback_finite & gemv_finite
    abs_diff = (fallback_logits - gemv_logits).abs()
    per_step_max = abs_diff.masked_fill(~finite, float("-inf")).amax(dim=-1)
    per_step_sum = abs_diff.masked_fill(~finite, 0.0).sum(dim=-1)
    per_step_count = finite.sum(dim=-1).clamp_min(1)
    per_step_mean = per_step_sum / per_step_count
    max_diff = per_step_max.max().item()
    mean_diff = per_step_mean.max().item()
    overall_mean = abs_diff[finite].mean().item()

    top5_fallback = torch.topk(fallback_logits, 5, dim=-1).indices
    top5_gemv = torch.topk(gemv_logits, 5, dim=-1).indices
    top5_overlap = (
        top5_fallback.unsqueeze(-1) == top5_gemv.unsqueeze(-2)
    ).any(dim=-1).sum(dim=-1)
    top5_exact = (top5_overlap == 5).sum().item()
    top5_min_overlap = top5_overlap.min().item()

    print(f"route_counts fallback={fallback_counts} gemv={gemv_counts}")
    print(f"top1={top1_agree}/16 top5_exact={top5_exact}/16 top5_min_overlap={top5_min_overlap}/5")
    print(f"max={max_diff:.6f} worst_step_mean={mean_diff:.6f} overall_mean={overall_mean:.6f}")

    assert top1_agree == 16
    assert top5_min_overlap >= 4
    assert max_diff <= 0.55
    assert mean_diff <= 0.075
    assert overall_mean <= 0.05


if __name__ == "__main__":
    main()
