#!/usr/bin/env python
"""tie_check.py, adapted for MiMo: is the EXL3_DEC_MOE_UNION=0-vs-1 greedy divergence a
near-tie flip (legitimate summation-order difference, per AGENTS.md) or a real defect
(a flip with a multi-logit gap)? Teacher-forces the SAME token sequence (greedy under
union=0) through both flag values via job.constrain_output_now, collects full logits at
every position with return_logits=True, and reports the cross-path |dlogit| plus, at
each argmax flip, the reference top1-top2 gap.
"""
import os
import subprocess
import sys

KNOB = "EXL3_DEC_MOE_UNION"
V0, V1 = "0", "1"
N = int(os.environ.get("NTOK", "24"))
PROMPT = ("Write the opening three paragraphs of a short story about a lighthouse "
          "keeper who discovers a message in a bottle, in a literary, descriptive style.")

WORKER = r"""
import os, sys, json, torch
sys.path.insert(0, os.environ.get("EXL3_REPO", "~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
mode, prompt, n, seq_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
config = Config.from_directory(os.path.expanduser("~/models/mimo26-exl3"))
model = Model.from_config(config); tok = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=4096); model.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok)
ids = tok.encode(prompt)
job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), return_logits=True, stop_conditions=[])
gen.enqueue(job)
if mode != "gen":
    job.constrain_output_now(torch.load(seq_path).view(1, -1).contiguous())
toks, logits = [], []
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if r.get("error"): print("JOB ERROR", r["error"], file=sys.stderr)
        if r.get("token_ids") is not None: toks.append(r["token_ids"].cpu())
        if r.get("logits") is not None: logits.append(r["logits"].cpu())
t = torch.cat(toks, dim=-1).flatten()
if mode == "gen":
    torch.save(t, seq_path); print("GEN", t.numel())
else:
    L = torch.cat(logits, dim=1)
    torch.save(L.float(), seq_path + "." + mode); print("LOGITS", tuple(L.shape))
"""

py = sys.executable
seq = "/tmp/tie_seq_mimo.pt"


def run(mode, v):
    env = dict(os.environ, **{KNOB: v})
    r = subprocess.run([py, "-c", WORKER, mode, PROMPT, str(N), seq],
                        capture_output=True, text=True, env=env, timeout=1800)
    ok = [l for l in r.stdout.splitlines() if l.startswith(("GEN", "LOGITS"))]
    if not ok:
        print(f"{mode} {KNOB}={v} FAILED\n{r.stderr[-3000:]}")
        sys.exit(1)
    print(f"{mode} {KNOB}={v}: {ok[0]}")


run("gen", V0)
run("a", V0)
run("b", V1)

import torch  # noqa: E402
la = torch.load(seq + ".a")[0]
lb = torch.load(seq + ".b")[0]
V = min(la.shape[-1], lb.shape[-1])
la, lb = la[:, :V], lb[:, :V]
d = (la - lb).abs()
print(f"\npositions={la.shape[0]}  cross-path |dlogit|: max {d.max():.4f}  mean {d.mean():.5f}")
ta, tb = la.argmax(-1), lb.argmax(-1)
flips = (ta != tb).nonzero().flatten().tolist()
print(f"argmax flips: {len(flips)} / {la.shape[0]}")
top2a = la.topk(2, dim=-1).values
for p in flips[:12]:
    gap = (top2a[p, 0] - top2a[p, 1]).item()
    print(f"  pos {p:4d}: ref top1 {ta[p].item():6d} vs {tb[p].item():6d}   "
          f"ref top1-top2 gap {gap:.4f}   |dlogit| here {d[p].max():.4f}")
if flips:
    gaps = torch.tensor([(top2a[p, 0] - top2a[p, 1]).item() for p in flips])
    print(f"max gap at a flip: {gaps.max():.4f}  (cross-path max delta {d.max():.4f})")
    print("VERDICT:", "near-tie flips only" if gaps.max() < d.max() * 2
          else "REAL DEFECT: a flip with a gap beyond the numerical delta")
else:
    print("VERDICT: identical argmax everywhere")
