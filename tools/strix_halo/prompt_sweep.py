#!/usr/bin/env python
"""Prompt-averaged MTP throughput. Speculative decode is content-dependent:
predictable text accepts more drafts, so a single prompt can over- or
under-state the real figure. Runs a fixed set and reports the distribution.

Config via env (defaults reproduce the historical invocation):
    NDT=2     draft tokens
    DC=0.4    draft confidence (only used when DDS=1)
    DDS=1     1 = dynamic draft sizing (-dds), 0 = static draft depth
    NTOK=512  new tokens per prompt
    LABEL=    optional tag echoed in the summary line

Static vs dynamic is a real fork, not a tuning detail: a static depth always spends
NDT draft tokens, a dynamic one truncates when the drafter's confidence drops. Which
wins depends on the prompt, so it has to be judged on the distribution, never one prompt.
"""
import os, subprocess, sys, re, statistics, json
from pathlib import Path

REPO = Path(os.path.expanduser("~/kyojin"))
PROMPTS = [
    "Explain gradient descent in two sentences:",
    "Write a short technical summary of how GPUs execute matrix multiplication:",
    "List the first ten prime numbers and explain what makes a number prime:",
    "Describe the difference between a process and a thread:",
    "Write a Python function that reverses a linked list, with comments:",
    "Summarize the causes of the 1929 stock market crash:",
]
N = int(os.environ.get("NTOK", "512"))
NDT = os.environ.get("NDT", "2")
DC = os.environ.get("DC", "0.4")
DDS = os.environ.get("DDS", "1") != "0"
LABEL = os.environ.get("LABEL", "")

env = os.environ.copy()
env.setdefault("EXL3_MOE_CFG", "2")
env.setdefault("EXL3_HIP_PREFILL_MIN_ROWS", "2")

args = ["-n", str(N), "-ndt", NDT, "-g"]
if DDS:
    args += ["-dds", "-dc", DC]
cfg = f"ndt={NDT} " + (f"dynamic dc={DC}" if DDS else "static")

rates, accs = [], []
print(f"CONFIG {cfg}  ntok={N}  label={LABEL or '-'}")
print(f"{'tok/s':>8} {'accept':>8}  prompt")
print("-" * 78)
for p in PROMPTS:
    r = subprocess.run(
        [str(REPO / ".venv/bin/python"), str(REPO / "bench_mtp.py")] + args + ["-p", p],
        capture_output=True, text=True, env=env, timeout=1800)
    out = r.stdout + r.stderr
    m = re.search(r"decode:\s+([0-9.]+) tok/s", out)
    a = re.search(r"acceptance=([0-9.]+)%", out)
    if m:
        rates.append(float(m.group(1)))
        accs.append(float(a.group(1)) if a else 0.0)
        print(f"{rates[-1]:>8.2f} {accs[-1]:>7.1f}%  {p[:56]}", flush=True)
    else:
        print(f"{'FAIL':>8} {'':>8}  {p[:56]}", flush=True)
        err = re.search(r"(Error|Traceback|Insufficient).{0,120}", out, re.S)
        if err:
            print(f"         {err.group(0)[:120]!r}", flush=True)

if rates:
    print("-" * 78)
    print(f"mean   {statistics.mean(rates):.2f} tok/s   "
          f"median {statistics.median(rates):.2f}   "
          f"min {min(rates):.2f}   max {max(rates):.2f}")
    print(f"mean acceptance {statistics.mean(accs):.1f}%")
    print("SWEEPJSON " + json.dumps({
        "label": LABEL, "cfg": cfg, "ndt": NDT, "dc": (DC if DDS else None),
        "dds": DDS, "ntok": N, "n": len(rates),
        "mean": round(statistics.mean(rates), 2),
        "median": round(statistics.median(rates), 2),
        "min": min(rates), "max": max(rates),
        "mean_acc": round(statistics.mean(accs), 1),
        "rates": rates, "accs": accs,
    }))
