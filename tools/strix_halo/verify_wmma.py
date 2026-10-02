#!/usr/bin/env python
"""End-to-end correctness gate for the gfx1151 WMMA EXL3 GEMV.

Compares per-step LOGITS from the new WMMA kernel against the
reconstruct+hgemm fallback on the SAME model, SAME prompt, forced greedy.
This is the honest test: it exercises the real trellis state, real codebooks,
real shapes, and the real epilogue.

Usage (run twice, the harness diffs the saved logits):
    EXL3_GEMV=1 python verify_wmma.py --save /tmp/logits_fast.pt
    EXL3_GEMV=0 python verify_wmma.py --save /tmp/logits_ref.pt
    python verify_wmma.py --compare /tmp/logits_fast.pt /tmp/logits_ref.pt
"""
import argparse, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
ap.add_argument("-p", "--prompt", default="The capital of France is")
ap.add_argument("-n", "--steps", type=int, default=8)
ap.add_argument("--save")
ap.add_argument("--compare", nargs=2)
args = ap.parse_args()

import torch

if args.compare:
    a = torch.load(args.compare[0])
    b = torch.load(args.compare[1])
    la, lb = a["logits"], b["logits"]
    print(f"fast: {a['mode']}   ref: {b['mode']}")
    print(f"steps compared: {min(len(la), len(lb))}")
    top1_agree = 0
    maxdiff = 0.0
    meandiff = 0.0
    n = min(len(la), len(lb))
    for i in range(n):
        x, y = la[i].float(), lb[i].float()
        d = (x - y).abs()
        maxdiff = max(maxdiff, d.max().item())
        meandiff += d.mean().item()
        if x.argmax().item() == y.argmax().item():
            top1_agree += 1
    meandiff /= max(n, 1)
    print(f"top-1 agreement:  {top1_agree}/{n}")
    print(f"max |dlogit|:     {maxdiff:.6f}")
    print(f"mean |dlogit|:    {meandiff:.6f}")
    print(f"tokens fast: {a['tokens']}")
    print(f"tokens ref:  {b['tokens']}")
    ok = (top1_agree == n) and (maxdiff < 1.0)
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}"
          f"  (require top-1 {n}/{n} and max|dlogit| < 1.0)")
    sys.exit(0 if ok else 2)

import exllamav3_ext
from exllamav3 import Config, Model, Cache, Tokenizer

mode = os.environ.get("EXL3_GEMV", "1")
print(f"EXL3_GEMV={mode}  supported={exllamav3_ext.exl3_gemv_supported(0)}  "
      f"family={exllamav3_ext.exl3_gemv_wmma_family(0)}", flush=True)

t0 = time.time()
config = Config.from_directory(args.model)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=1024)
model.load(progressbar=False)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

ids = tokenizer.encode(args.prompt, add_bos=True).to("cuda")
logits_log, toks = [], []

# Manual greedy loop so we capture raw logits each step.
params = {"attn_mode": "flash_attn_nc", "cache": cache, "past_len": 0}
cur = ids
past = 0
for step in range(args.steps):
    out = model.prefill(input_ids=cur, params={"cache": cache, "past_len": past}) \
          if hasattr(model, "prefill") else None
    logits = model.forward(input_ids=cur, params={"cache": cache, "past_len": past})
    if isinstance(logits, dict):
        logits = logits.get("logits", logits)
    last = logits[0, -1, :].detach().float().cpu()
    logits_log.append(last)
    nxt = int(last.argmax().item())
    toks.append(nxt)
    past += cur.shape[-1]
    cur = torch.tensor([[nxt]], dtype=torch.long, device="cuda")

text = tokenizer.decode(torch.tensor([toks]))[0] if toks else ""
print(f"tokens: {toks}")
print(f"text:   {text!r}")

if args.save:
    torch.save({"logits": logits_log, "tokens": toks, "mode": mode, "text": text}, args.save)
    print(f"saved -> {args.save}")
