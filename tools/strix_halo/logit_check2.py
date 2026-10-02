#!/usr/bin/env python
"""Compare step-1 logits: WMMA fast path vs reconstruct fallback.

Uses the Generator (the API that actually works for this arch) and asks for
logits back, so we see whether the fast path yields garbage / EOS.
"""
import os, sys
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
mode = os.environ.get("EXL3_GEMV", "1")
print(f"EXL3_GEMV={mode} supported={exllamav3_ext.exl3_gemv_supported(0)} "
      f"family={exllamav3_ext.exl3_gemv_wmma_family(0)}", flush=True)

config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=2048)
model.load(progressbar=False)

ids = tokenizer.encode("The capital of France is", add_bos=True)
print(f"input ids: {ids.tolist()}", flush=True)

gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
try:
    job = Job(input_ids=ids, max_new_tokens=4, return_logits=True)
except TypeError:
    job = Job(input_ids=ids, max_new_tokens=4)
gen.enqueue(job)

chunks, first_logits, all_keys = [], None, set()
while gen.num_remaining_jobs():
    for res in gen.iterate():
        all_keys |= set(res.keys())
        lg = res.get("logits")
        if lg is not None and first_logits is None:
            first_logits = lg.reshape(-1, lg.shape[-1])[-1].float().cpu()
        t = res.get("text")
        if t:
            chunks.append(t)

print(f"result keys seen: {sorted(all_keys)}")
print(f"generated chunks: {chunks}")

if first_logits is None:
    print("NO LOGITS in results -- cannot compare numerically")
    sys.exit(3)

last = first_logits
print(f"finite={torch.isfinite(last).all().item()} "
      f"nan={torch.isnan(last).sum().item()} inf={torch.isinf(last).sum().item()}")
print(f"min={last.min().item():.4f} max={last.max().item():.4f} "
      f"mean={last.mean().item():.4f} std={last.std().item():.4f}")
top = torch.topk(last, 8)
print("top-8:")
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    try:
        tok = tokenizer.decode(torch.tensor([[i]]))[0]
    except Exception:
        tok = "?"
    print(f"   id={i:7d} logit={v:9.4f}  {tok!r}")
eos = getattr(tokenizer, "eos_token_id", None)
print(f"eos_token_id={eos} argmax={last.argmax().item()} "
      f"ARGMAX_IS_EOS={last.argmax().item() == eos}")
torch.save(last, f"/tmp/logits_{mode}.pt")
print(f"saved /tmp/logits_{mode}.pt")
