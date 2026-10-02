#!/usr/bin/env python
"""Per-round phase breakdown of MTP speculative decode (wall, with sync).

profile_mtp.py brackets individual kernels; this brackets the *phases* of one
speculation round so the trunk-verify / MTP-head / bookkeeping split is visible:

  draft    = iterate_draftmodel_mtp_gen  (MTP head forward + confidence gate)
  verify   = model.forward inside iterate_gen (trunk, ndt+1 rows)
  dprefill = draft_model.prefill inside iterate_gen (re-sync MTP cache with accepted rows)
  other    = iterate_gen minus verify minus dprefill (sampling, page mgmt, python)
  outside  = iterate() minus (draft + iterate_gen)

Each bracket does a torch.cuda.synchronize(), so the phases are serialized and
the total is slightly pessimistic; the split is what matters.
"""
import os, sys, time, collections
import torch
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
NDT = int(os.environ.get("NDT", "2"))
NTOK = int(os.environ.get("NTOK", "256"))
DC = float(os.environ.get("DC", "0.4"))
PROMPT = os.environ.get("PROMPT", "Explain gradient descent in two sentences:")

config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=4096, max_history=NDT)
model.load(progressbar=False)
draft_model = Model.from_config(config, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=4096, max_history=NDT)
draft_model.load(progressbar=False)

T = collections.defaultdict(float)
N = collections.defaultdict(int)


def bracket(obj, name, key):
    fn = getattr(obj, name)

    def w(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        T[key] += time.perf_counter() - t
        N[key] += 1
        return r

    setattr(obj, name, w)


gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                draft_cache=draft_cache, num_draft_tokens=NDT, dynamic_draft_tokens=True,
                draft_confidence=DC)
bracket(gen, "iterate_draftmodel_mtp_gen", "draft")
bracket(gen, "iterate_gen", "itgen")
bracket(gen, "iterate", "iterate")
bracket(model, "forward", "verify")
bracket(draft_model, "forward", "dfwd")
bracket(draft_model, "prefill", "dprefill")

ids = tokenizer.encode(PROMPT, add_bos=True)
gen.enqueue(Job(input_ids=ids, max_new_tokens=8, sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate():
        pass
T.clear()
N.clear()

torch.cuda.synchronize()
t0 = time.perf_counter()
gen.enqueue(Job(input_ids=ids, max_new_tokens=NTOK, sampler=GreedySampler()))
ntok = acc = rej = 0
first = None
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if r.get("text"):
            if first is None:
                first = time.perf_counter() - t0
            ntok += 1
        if "accepted_draft_tokens" in r:
            acc += r["accepted_draft_tokens"]
            rej += r["rejected_draft_tokens"]
torch.cuda.synchronize()
total = time.perf_counter() - t0
rounds = N["itgen"]
print(f"ndt={NDT} dc={DC} tokens={ntok} rounds={rounds} tok/round={ntok/rounds:.2f} "
      f"decode={(ntok-1)/(total-first):.2f} tok/s acceptance={100*acc/max(acc+rej,1):.1f}%")
other = T["itgen"] - T["verify"] - T["dprefill"]
outside = T["iterate"] - T["draft"] - T["itgen"]
rows = [("draft (MTP head + gate)", T["draft"]), ("  of which draft fwd", T["dfwd"]),
        ("verify (trunk fwd)", T["verify"]), ("dprefill (MTP cache sync)", T["dprefill"]),
        ("iterate_gen other", other), ("outside iterate_gen/draft", outside)]
print(f"{'phase':30} {'ms total':>9} {'us/round':>9} {'%':>6}")
for k, v in rows:
    print(f"{k:30} {v*1e3:9.1f} {v*1e6/rounds:9.0f} {100*v/T['iterate']:6.1f}")
print(f"{'iterate total':30} {T['iterate']*1e3:9.1f} {T['iterate']*1e6/rounds:9.0f}")
print(f"draft fwd calls/round={N['dfwd']/rounds:.2f}  verify calls/round={N['verify']/rounds:.2f}"
      f"  dprefill calls/round={N['dprefill']/rounds:.2f}")
