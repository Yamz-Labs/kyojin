#!/usr/bin/env python
"""One forward pass; print the top-5 logits. Run with EXL3_GEMV=1 and =0 and diff.

Empty generation with a working kernel usually means the FIRST sampled token is
EOS -- i.e. the logits are wrong somewhere the per-layer spot-check didn't cover
(most likely the LM head, whose out_features is the 151k vocab).
"""
import os, sys
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext
from exllamav3 import Config, Model, Cache, Tokenizer

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
mode = os.environ.get("EXL3_GEMV", "1")
print(f"EXL3_GEMV={mode} supported={exllamav3_ext.exl3_gemv_supported(0)} "
      f"family={exllamav3_ext.exl3_gemv_wmma_family(0)}", flush=True)

config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=2048)
model.load(progressbar=False)

ids = tokenizer.encode("The capital of France is", add_bos=True).to("cuda")
print(f"input ids: {ids.tolist()}", flush=True)

logits = model.forward(input_ids=ids, params={"cache": cache, "past_len": 0})
if isinstance(logits, dict):
    logits = logits.get("logits", logits)
last = logits[0, -1, :].float()

print(f"logits shape: {tuple(logits.shape)}")
print(f"finite: {torch.isfinite(last).all().item()}  "
      f"nan: {torch.isnan(last).sum().item()}  inf: {torch.isinf(last).sum().item()}")
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
print(f"eos_token_id: {eos}  argmax: {last.argmax().item()}  "
      f"ARGMAX_IS_EOS={last.argmax().item() == eos}")

torch.save(last.cpu(), f"/tmp/logits_{mode}.pt")
print(f"saved /tmp/logits_{mode}.pt")
