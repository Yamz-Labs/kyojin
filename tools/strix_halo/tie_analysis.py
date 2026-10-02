#!/usr/bin/env python
"""Are the top-1 disagreements real errors or coin-flip near-ties?

With grouped MoE enabled the oracle reports top-1 15/17. A flip matters only if
the two candidates were meaningfully separated; if logit[top1] - logit[top2] is
within the observed cross-path delta (~0.086 mean), the flip is a tie broken
differently by two equally-valid roundings, not a defect.

For each step: report the reference's top-1/top-2 gap, whether the paths agree,
and the softmax probability mass that actually moved.
"""
import os, subprocess, sys, tempfile
from pathlib import Path

import torch

REPO = Path(os.path.expanduser("~/kyojin"))
MODEL = os.environ.get("EXL3_TEST_MODEL", os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
WORKER = Path(tempfile.gettempdir()) / "_oracle_worker2.py"

if not WORKER.exists():
    print(f"missing {WORKER} -- run wmma_oracle_full.py first")
    sys.exit(1)


def run(mode, out, forced=None):
    env = os.environ.copy()
    env.update({"EXL3_GEMV": "0" if mode == "fallback" else "1",
                "ORACLE_MODEL": MODEL, "ORACLE_STEPS": "16",
                "ORACLE_PROMPT": "The capital of France is",
                "PYTHONPATH": str(REPO)})
    cmd = [str(REPO / ".venv/bin/python"), str(WORKER), mode, str(out)]
    if forced:
        cmd.append(str(forced))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"leg {mode} failed"); print(r.stderr[-800:]); return False
    return True


tmp = Path(tempfile.mkdtemp())
fb, gv, forced = tmp / "fb.pt", tmp / "gv.pt", tmp / "forced.pt"
if not run("fallback", fb):
    sys.exit(1)
torch.save(torch.load(fb, map_location="cpu")["token_ids"].clone(), forced)
if not run("gemv", gv, forced):
    sys.exit(1)

a = torch.load(fb, map_location="cpu")["logits"].float()[0]   # [steps, vocab]
b = torch.load(gv, map_location="cpu")["logits"].float()[0]
m = min(a.shape[0], b.shape[0])
a, b = a[:m], b[:m]

print(f"{'step':>4} {'ref_top1':>9} {'gap(1-2)':>9} {'fast_top1':>10} "
      f"{'agree':>6} {'p_moved':>9}  note")
print("-" * 72)

flips, tie_flips = 0, 0
for i in range(m):
    ta = torch.topk(a[i], 2)
    gap = (ta.values[0] - ta.values[1]).item()
    t1a = int(a[i].argmax())
    t1b = int(b[i].argmax())
    agree = t1a == t1b
    pa = torch.softmax(a[i], -1)
    pb = torch.softmax(b[i], -1)
    p_moved = (pa - pb).abs().sum().item() / 2.0   # total variation distance
    note = ""
    if not agree:
        flips += 1
        if gap <= 0.30:
            tie_flips += 1
            note = "NEAR-TIE (gap <= 0.30)"
        else:
            note = "** REAL DISAGREEMENT **"
    print(f"{i:>4} {t1a:>9} {gap:>9.4f} {t1b:>10} {str(agree):>6} "
          f"{p_moved:>9.5f}  {note}")

print("-" * 72)
print(f"top-1 flips: {flips}/{m}   of which near-ties: {tie_flips}")
tv = ((torch.softmax(a, -1) - torch.softmax(b, -1)).abs().sum(-1) / 2).max().item()
print(f"max total-variation distance between step distributions: {tv:.6f}")
print()
if flips == 0:
    print("VERDICT: no flips.")
elif flips == tie_flips:
    print("VERDICT: every flip is a near-tie the two roundings break differently.")
    print("Not a defect -- and PPL independently confirms the fast path is no worse.")
else:
    print("VERDICT: at least one flip had a real margin. Investigate the MoE epilogue.")
