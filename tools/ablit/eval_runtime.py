#!/usr/bin/env python3
"""Refusal eval through the runtime hook (EXL3_ABLIT_RUNTIME) on a quantized pack; same prompts,
detector, greedy 64 tok and batch as ablit4 (eval_ablit.refusal_eval). Hook on/off is decided by the env at load.
usage: eval_runtime.py --model DIR --n 100 --batch 16 --out out.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ablit as E
import arch as arch_mod

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-new-tokens", type=int, default=64)
ap.add_argument("--out", required=True)
ap.add_argument("--arch", default="auto", choices=["auto"] + list(arch_mod.ARCHS))
a = ap.parse_args()
_A = arch_mod.get(a.arch, a.model)
E.set_arch(None if _A["name"] == "glm" else _A)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
torch.set_grad_enabled(False)
harm = E.load_prompts(os.path.join(ROOT, "scratch/ablit/prompts/harmful_heldout.jsonl"), a.n)
config = Config.from_directory(a.model)
model = Model.from_config(config)
tok = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens = ((a.batch * (256 + a.max_new_tokens + 8) + 1023) // 1024 + 1) * 1024)
t0 = time.time()
model.load(device = "cuda:0", progressbar = False)
hooked = sum(1 for m in model.modules if getattr(m, "ablit", None))
print(f"loaded {time.time() - t0:.0f}s, hooked blocks {hooked}, runtime spec {os.environ.get('EXL3_ABLIT_RUNTIME')}", flush = True)
gen = Generator(model = model, cache = cache, tokenizer = tok)
r = E.refusal_eval(gen, tok, harm, config.eos_token_id_list, a.max_new_tokens, a.batch)
print(f"refusals {r['refusals']}/{r['n']} = {r['rate']:.3f}  empties {r['empties']}  hooked_blocks {hooked}", flush = True)
r["hooked_blocks"] = hooked
json.dump(r, open(a.out, "w"), indent = 1)
