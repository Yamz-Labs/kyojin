#!/usr/bin/env python
"""Exact bytes-per-forward accounting by module class for one trunk forward at R rows.

Wraps the tensor-reading ext entry points and sums the weight bytes each touches, grouped by
the Python module that issued the call. The result is the real byte budget the verify
forward streams, which decides which module a byte-cutting change should target next.
"""
import os, sys, collections, inspect
import torch
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache
from exllamav3.ext import exllamav3_ext as X
import exllamav3.modules.hyperconnections as hcmod

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
NDT = int(os.environ.get("NDT", "2")); DC = float(os.environ.get("DC", "0.4")); NTOK = int(os.environ.get("NTOK", "64"))

config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=2048, max_history=NDT)
model.load(progressbar=False)

agg = collections.defaultdict(lambda: [0, 0.0])
counting = {"on": False, "n": 0, "rows": 0}
def site():
    for f in inspect.stack()[2:12]:
        fn = f.filename
        if "/exllamav3/modules/" in fn or "/exllamav3/architecture/" in fn:
            return os.path.basename(fn) + ":" + f.function
    return "?"

def wrap(name, nbytes):
    fn = getattr(X, name, None)
    if fn is None: return
    def spy(*a, **k):
        if counting["on"]:
            b = nbytes(*a, **k)
            s = agg[(name, site())]; s[0] += 1; s[1] += b
        return fn(*a, **k)
    setattr(X, name, spy)

def tbytes(t): return t.numel() * t.element_size() if torch.is_tensor(t) else 0
wrap("exl3_gemv", lambda A, B, C, *a, **k: tbytes(B))
wrap("exl3_gemv_int8", lambda A, B, C, *a, **k: tbytes(B))
wrap("hgemm", lambda a, b, c, *k: tbytes(b))
wrap("gr_mix", lambda s, fn, upt, w, *a: tbytes(fn) + tbytes(upt) + tbytes(w))
wrap("hc_mix", lambda s, fn, *a: tbytes(fn))
# grouped MoE: count unique experts x 3 matrices (K=3 trellis 2560x640)
EXP = 3 * 2560 * 640 * 3 // 8
def moe_bytes(A, output, selected, *a, **k):
    return int(torch.unique(selected).numel()) * EXP
wrap("exl3_moe_gfx12_k3", moe_bytes)
wrap("exl3_moe_gfx12_k3_prefill", moe_bytes)
wrap("cuda_recurrent_gated_delta_rule", lambda *a, **k: 0)

# Drive a real MTP generation and attribute bytes to trunk-verify forwards only
from exllamav3 import Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
tok = Tokenizer.from_config(config)
draft = Model.from_config(config, component="mtp")
dcache = Cache(draft, max_num_tokens=2048, max_history=NDT)
draft.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=NDT, dynamic_draft_tokens=True, draft_confidence=DC)
real_fwd = model.forward
def fwd(input_ids, params=None):
    counting["on"] = True; counting["n"] += 1; counting["rows"] += input_ids.numel()
    try: return real_fwd(input_ids, params)
    finally: counting["on"] = False
model.forward = fwd
_agg_add = agg
ids = tok.encode("Explain gradient descent in two sentences:", add_bos=True)
gen.enqueue(Job(input_ids=ids, max_new_tokens=8, sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate(): pass
agg.clear(); counting.update(n=0, rows=0)
gen.enqueue(Job(input_ids=ids, max_new_tokens=NTOK, sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate(): pass
torch.cuda.synchronize()

tot = sum(v[1] for v in agg.values())
nf = counting["n"]
print(f"{nf} trunk forwards, mean rows {counting['rows']/nf:.2f}: {tot/nf/2**30:.3f} GiB weight reads per forward (+ draft-model reads counted separately below)")
by_mod = collections.defaultdict(lambda: [0, 0.0])
for (name, st), (n, b) in agg.items():
    m = by_mod[(st, name)]; m[0] += n; m[1] += b
for (st, name), (n, b) in sorted(by_mod.items(), key=lambda kv: -kv[1][1]):
    print(f"  {b/nf/2**20:8.1f} MiB/fwd {100*b/tot:5.1f}%  {n/nf:6.1f} calls/fwd  {name:28} {st}")
