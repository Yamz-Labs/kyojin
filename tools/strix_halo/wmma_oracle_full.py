#!/usr/bin/env python
"""Rigorous WMMA GEMV oracle for gfx11.5.

Uses the fork's own validation methodology (tests/hip_gemv_logits_worker.py)
but tolerates a step count that differs from their dense-27B fixture:
  * teacher forcing via job.constrain_output_now() -- both legs see identical
    inputs at every step, so logits are actually comparable
  * route counting via ext.* spies -- proves the fast path ran AND that nothing
    silently fell back to reconstruct
  * the fork's own numeric thresholds on logit deltas / top-1 / top-5

Env: EXL3_TEST_MODEL, ORACLE_STEPS, ORACLE_PROMPT
"""
import os, sys, subprocess, tempfile
from pathlib import Path

import torch

REPO = Path(os.path.expanduser("~/kyojin"))
MODEL = os.environ.get("EXL3_TEST_MODEL", os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
STEPS = int(os.environ.get("ORACLE_STEPS", "16"))
PROMPT = os.environ.get("ORACLE_PROMPT", "The capital of France is")

WORKER = r'''
import os, sys, atexit, torch
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3 import ext as _extwrap
from exllamav3.generator.sampler import GreedySampler
# exllamav3.ext forwards attribute lookups to the compiled module, so the spies
# must be installed on the compiled module itself for callers to see them.
ext = _extwrap.exllamav3_ext

mode, out_path = sys.argv[1], sys.argv[2]
forced_path = sys.argv[3] if len(sys.argv) > 3 else None
MODEL = os.environ["ORACLE_MODEL"]
STEPS = int(os.environ["ORACLE_STEPS"])
PROMPT = os.environ["ORACLE_PROMPT"]

config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096, max_batch_size=1)
model.load(device="cuda")
unload = model.unload
atexit.register(unload)
tokenizer = Tokenizer.from_config(model.config)
generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

job = Job(
    input_ids=tokenizer.encode(PROMPT, add_bos=True),
    max_new_tokens=STEPS + 1,
    stop_conditions=[],
    return_logits=True,
    sampler=GreedySampler(),
)
generator.enqueue(job)

if mode == "gemv":
    if forced_path is None:
        raise SystemExit("gemv mode requires a forced-token path")
    base = torch.load(forced_path, map_location="cpu")
    if base.dim() == 1:
        base = base.unsqueeze(0)
    job.constrain_output_now(base.contiguous())

real_gemv = ext.exl3_gemv
real_reconstruct = ext.reconstruct
real_rslice = getattr(ext, "reconstruct_slice", None)
real_rhslice = getattr(ext, "reconstruct_had_slice", None)
real_hgemm = ext.hgemm
counts = {"gemv": 0, "reconstruct": 0, "reconstruct_slice": 0,
          "reconstruct_had_slice": 0, "hgemm": 0}

def mk(real, key):
    def spy(*a, **k):
        counts[key] += 1
        return real(*a, **k)
    return spy

ext.exl3_gemv = mk(real_gemv, "gemv")
ext.reconstruct = mk(real_reconstruct, "reconstruct")
if real_rslice is not None:
    ext.reconstruct_slice = mk(real_rslice, "reconstruct_slice")
if real_rhslice is not None:
    ext.reconstruct_had_slice = mk(real_rhslice, "reconstruct_had_slice")
ext.hgemm = mk(real_hgemm, "hgemm")

tok_chunks, logit_chunks = [], []
try:
    while generator.num_remaining_jobs():
        for res in generator.iterate():
            if res.get("error"):
                raise SystemExit(f"JOB ERROR: {res['error']!r}")
            if res.get("token_ids") is not None:
                tok_chunks.append(res["token_ids"].cpu())
            if res.get("logits") is not None:
                logit_chunks.append(res["logits"].cpu())
finally:
    ext.exl3_gemv = real_gemv
    ext.reconstruct = real_reconstruct
    if real_rslice is not None:
        ext.reconstruct_slice = real_rslice
    if real_rhslice is not None:
        ext.reconstruct_had_slice = real_rhslice
    ext.hgemm = real_hgemm

if not tok_chunks:
    raise SystemExit("worker produced no tokens")
token_ids = torch.cat(tok_chunks, dim=-1)
logits = torch.cat(logit_chunks, dim=1) if logit_chunks else None
torch.save({"token_ids": token_ids, "logits": logits, "counts": counts}, out_path)
print(f"RESULT {mode}: tokens={token_ids.shape} counts={counts}")
try:
    unload()
finally:
    atexit.unregister(unload)
'''


def run_leg(mode, out_path, forced=None):
    env = os.environ.copy()
    env["EXL3_GEMV"] = "0" if mode == "fallback" else "1"
    env["ORACLE_MODEL"] = MODEL
    env["ORACLE_STEPS"] = str(STEPS)
    env["ORACLE_PROMPT"] = PROMPT
    env["PYTHONPATH"] = str(REPO)
    wpath = Path(tempfile.gettempdir()) / "_oracle_worker2.py"
    wpath.write_text(WORKER)
    cmd = [str(REPO / ".venv/bin/python"), str(wpath), mode, str(out_path)]
    if forced:
        cmd.append(str(forced))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    for line in (r.stdout + r.stderr).splitlines():
        if any(s in line for s in ("RESULT", "ERROR", "Error:", "SystemExit",
                                   "AssertionError", "Traceback", "TypeError")):
            print("  " + line.strip())
    if r.returncode != 0:
        print(f"  !! leg '{mode}' exit={r.returncode}")
        tail = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()][-6:]
        for l in tail:
            print("     " + l.strip())
        return False
    return True


tmp = Path(tempfile.mkdtemp())
fb, gv, forced = tmp / "fb.pt", tmp / "gv.pt", tmp / "forced.pt"

print("=== leg 1: fallback (EXL3_GEMV=0) ===")
if not run_leg("fallback", fb):
    sys.exit(1)
d_fb = torch.load(fb, map_location="cpu")
torch.save(d_fb["token_ids"].clone(), forced)

print("\n=== leg 2: WMMA gemv (EXL3_GEMV=1), teacher-forced ===")
if not run_leg("gemv", gv, forced):
    sys.exit(1)
d_gv = torch.load(gv, map_location="cpu")

print("\n" + "=" * 64)
t_fb, t_gv = d_fb["token_ids"], d_gv["token_ids"]
c_fb, c_gv = d_fb["counts"], d_gv["counts"]
l_fb, l_gv = d_fb["logits"], d_gv["logits"]

print(f"routes fallback: {c_fb}")
print(f"routes gemv:     {c_gv}")

fails, warns = [], []
if c_fb["gemv"] != 0:
    fails.append(f"fallback leg used gemv {c_fb['gemv']}x (must be 0)")
if c_gv["gemv"] <= 0:
    fails.append("gemv leg never called exl3_gemv -- fast path did NOT run")
if c_gv["reconstruct"] != 0:
    fails.append(f"gemv leg fell back to reconstruct {c_gv['reconstruct']}x")
if c_gv["reconstruct_slice"] or c_gv["reconstruct_had_slice"]:
    fails.append("gemv leg used reconstruct_slice paths")
if c_gv["gemv"] > 0 and c_gv["hgemm"] > 0 and c_gv["gemv"] < 2.5 * c_gv["hgemm"]:
    warns.append(f"gemv/hgemm ratio {c_gv['gemv']}/{c_gv['hgemm']} below the fork's 2.5x guideline")

n = min(t_fb.numel(), t_gv.numel())
same = torch.equal(t_fb.flatten()[:n], t_gv.flatten()[:n])
print(f"tokens: fb={tuple(t_fb.shape)} gv={tuple(t_gv.shape)} identical(first {n})={same}")
if not same:
    a, b = t_fb.flatten(), t_gv.flatten()
    for i in range(n):
        if a[i] != b[i]:
            fails.append(f"token divergence at step {i}: fb={a[i].item()} gv={b[i].item()}")
            break

if l_fb is not None and l_gv is not None:
    m = min(l_fb.shape[1], l_gv.shape[1])
    a, b = l_fb[:, :m].float(), l_gv[:, :m].float()
    if torch.isnan(b).any():
        fails.append("gemv logits contain NaN")
    if torch.isposinf(b).any():
        fails.append("gemv logits contain +inf")
    finite = torch.isfinite(a) & torch.isfinite(b)
    diff = (a - b).abs()
    max_d = diff.masked_fill(~finite, float("-inf")).max().item()
    mean_d = diff[finite].mean().item()
    top1 = (a.argmax(-1) == b.argmax(-1)).sum().item()
    total = a.shape[0] * a.shape[1]
    k = min(5, a.shape[-1])
    t5a = torch.topk(a, k, dim=-1).indices
    t5b = torch.topk(b, k, dim=-1).indices
    ov = (t5a.unsqueeze(-1) == t5b.unsqueeze(-2)).any(-1).sum(-1)
    print(f"logit steps compared: {m}")
    print(f"top-1 agreement:  {top1}/{total}")
    print(f"top-5 min overlap: {ov.min().item()}/{k}")
    print(f"max |dlogit|: {max_d:.6f}   mean |dlogit|: {mean_d:.6f}")
    if top1 != total:
        fails.append(f"top-1 agreement {top1}/{total}")
    if ov.min().item() < 4:
        fails.append(f"top-5 overlap {ov.min().item()} < 4")
    if max_d > 0.55:
        fails.append(f"max |dlogit| {max_d:.4f} > 0.55")
    if mean_d > 0.05:
        fails.append(f"mean |dlogit| {mean_d:.4f} > 0.05")
else:
    warns.append("logits unavailable; token comparison only")

print("=" * 64)
for w in warns:
    print("WARN:", w)
if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(2)
print("PASS - WMMA GEMV matches the fallback on every criterion")
