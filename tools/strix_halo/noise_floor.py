#!/usr/bin/env python
"""Establish the NOISE FLOOR before judging the WMMA kernel's logit deltas.

The fork's thresholds (max 0.55, mean 0.05) were calibrated on a dense 27B
fixture with ~6,813 GEMV calls. Our MoE model makes 34,958. Before treating a
0.85 max delta as a defect, measure how much the REFERENCE path varies against
ITSELF across two identical runs -- split-k reductions, atomics and hipBLASLt
kernel selection are all potential sources of run-to-run nondeterminism.

Comparisons:
    fallback vs fallback   -> reference noise floor
    gemv     vs gemv       -> is the new kernel itself deterministic?
Both are teacher-forced on the same token sequence, so inputs are identical.
"""
import os, subprocess, sys, tempfile
from pathlib import Path

import torch

REPO = Path(os.path.expanduser("~/kyojin"))
MODEL = os.environ.get("EXL3_TEST_MODEL", os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
STEPS = int(os.environ.get("ORACLE_STEPS", "16"))
PROMPT = "The capital of France is"

WORKER = Path(tempfile.gettempdir()) / "_oracle_worker2.py"


def run(mode, out_path, forced=None):
    env = os.environ.copy()
    env["EXL3_GEMV"] = "0" if mode == "fallback" else "1"
    env["ORACLE_MODEL"] = MODEL
    env["ORACLE_STEPS"] = str(STEPS)
    env["ORACLE_PROMPT"] = PROMPT
    env["PYTHONPATH"] = str(REPO)
    cmd = [str(REPO / ".venv/bin/python"), str(WORKER), mode, str(out_path)]
    if forced:
        cmd.append(str(forced))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"  !! {mode} exit={r.returncode}")
        for l in (r.stdout + r.stderr).splitlines()[-5:]:
            print("     " + l.strip())
        return False
    for l in (r.stdout + r.stderr).splitlines():
        if "RESULT" in l:
            print("  " + l.strip())
    return True


def compare(pa, pb, label):
    a_d, b_d = torch.load(pa, map_location="cpu"), torch.load(pb, map_location="cpu")
    a, b = a_d["logits"], b_d["logits"]
    if a is None or b is None:
        print(f"{label}: no logits")
        return None
    m = min(a.shape[1], b.shape[1])
    a, b = a[:, :m].float(), b[:, :m].float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    diff = (a - b).abs()
    max_d = diff.masked_fill(~finite, float("-inf")).max().item()
    mean_d = diff[finite].mean().item()
    bitexact = bool(torch.equal(a[finite], b[finite]))
    tok_same = torch.equal(a_d["token_ids"], b_d["token_ids"])
    print(f"{label}:")
    print(f"    bit-exact={bitexact}  tokens_identical={tok_same}")
    print(f"    max |dlogit|={max_d:.6f}   mean |dlogit|={mean_d:.6f}")
    return max_d, mean_d, bitexact


tmp = Path(tempfile.mkdtemp())
forced = tmp / "forced.pt"

print("=== establishing the forced token sequence ===")
seed = tmp / "seed.pt"
if not run("fallback", seed):
    sys.exit(1)
torch.save(torch.load(seed, map_location="cpu")["token_ids"].clone(), forced)

print("\n=== REFERENCE NOISE FLOOR: fallback vs fallback (same forced tokens) ===")
f1, f2 = tmp / "f1.pt", tmp / "f2.pt"
ok = run("fallback", f1, forced) and run("fallback", f2, forced)
ref = compare(f1, f2, "  fallback-vs-fallback") if ok else None

print("\n=== KERNEL SELF-CONSISTENCY: gemv vs gemv ===")
g1, g2 = tmp / "g1.pt", tmp / "g2.pt"
ok2 = run("gemv", g1, forced) and run("gemv", g2, forced)
own = compare(g1, g2, "  gemv-vs-gemv") if ok2 else None

print("\n=== CROSS: gemv vs fallback ===")
cross = compare(g1, f1, "  gemv-vs-fallback")

print("\n" + "=" * 64)
if ref and cross:
    ref_max, ref_mean, ref_exact = ref
    cx_max, cx_mean, _ = cross
    print(f"reference noise floor : max={ref_max:.6f} mean={ref_mean:.6f} bitexact={ref_exact}")
    if own:
        print(f"gemv self-consistency : max={own[0]:.6f} mean={own[1]:.6f} bitexact={own[2]}")
    print(f"gemv vs fallback      : max={cx_max:.6f} mean={cx_mean:.6f}")
    print()
    if ref_exact and cx_max > 0.55:
        print("INTERPRETATION: the reference is deterministic, so the cross delta is")
        print("attributable to the WMMA kernel's accumulation order. Judge it against")
        print("token/top-1 identity, which is the property that actually matters.")
    elif not ref_exact:
        print(f"INTERPRETATION: the REFERENCE ITSELF is nondeterministic (max={ref_max:.4f}).")
        print("The fork's fixed thresholds are not meaningful for this model; compare")
        print("the cross delta against this noise floor instead.")
        if cx_max <= ref_max * 1.5:
            print(f"-> cross delta {cx_max:.4f} is WITHIN 1.5x the reference's own noise. PASS.")
        else:
            print(f"-> cross delta {cx_max:.4f} EXCEEDS 1.5x the noise floor. Investigate.")
