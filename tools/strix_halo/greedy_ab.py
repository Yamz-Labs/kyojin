#!/usr/bin/env python
"""Greedy token-ID A/B across an env knob: the two runs must produce IDENTICAL token lists.
Usage: greedy_ab.py KNOB [v0 v1]  (runs KNOB=v0 and KNOB=v1 in separate processes; default 0 1)
NOMTP=1 disables the MTP draft model. Do that for kernel A/Bs: with speculation on, the
verify batch width varies round to round, so the SAME weights take different kernel paths
(m=1 vs m=3 GEMV, MoE decode vs prefill) and near-tie top-1 flips appear even with the knob
fixed. Run KNOB=x twice first to establish the noise floor.
"""
import os, subprocess, sys, json

KNOB = sys.argv[1]
VALS = sys.argv[2:4] if len(sys.argv) >= 4 else ["0", "1"]
N = int(os.environ.get("NTOK", "160"))
NOMTP = os.environ.get("NOMTP", "0") == "1"
PROMPTS = [
    "Explain gradient descent in two sentences:",
    "Write a Python function that reverses a linked list, with comments:",
]
WORKER = r'''
import os, sys, json
sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
prompts = json.loads(sys.argv[1]); n = int(sys.argv[2]); nomtp = sys.argv[3] == "1"
config = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
model = Model.from_config(config); tok = Tokenizer.from_config(config)
mh = 0 if nomtp else 2
cache = Cache(model, max_num_tokens=4096, max_history=mh); model.load(progressbar=False)
if nomtp:
    gen = Generator(model=model, cache=cache, tokenizer=tok)
else:
    draft = Model.from_config(config, component="mtp"); dcache = Cache(draft, max_num_tokens=4096, max_history=2); draft.load(progressbar=False)
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache, num_draft_tokens=2, dynamic_draft_tokens=True, draft_confidence=0.4)
out = []
for p in prompts:
    ids = tok.encode(p, add_bos=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler()))
    got = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): print("JOB ERROR", r["error"], file=sys.stderr)
            t = r.get("token_ids")
            if t is not None: got += t.flatten().tolist()
    out.append(got)
print("IDS " + json.dumps(out))
'''
res = {}
for tag, v in (("a", VALS[0]), ("b", VALS[1])):
    env = dict(os.environ, **{KNOB: v, "EXL3_MOE_CFG": "2", "EXL3_HIP_PREFILL_MIN_ROWS": "2"})
    r = subprocess.run([os.path.expanduser("~/kyojin/.venv/bin/python"), "-c", WORKER, json.dumps(PROMPTS), str(N), "1" if NOMTP else "0"],
                       capture_output=True, text=True, env=env, timeout=1800)
    line = [l for l in r.stdout.splitlines() if l.startswith("IDS ")]
    if not line:
        print(f"{KNOB}={v}: FAILED\n{r.stderr[-2000:]}"); sys.exit(1)
    res[tag] = json.loads(line[0][4:])
    print(f"{KNOB}={v}: {[len(x) for x in res[tag]]} tokens  (mtp={'off' if NOMTP else 'on'})")
same = res["a"] == res["b"]
print("IDENTICAL" if same else "MISMATCH")
if not same:
    for a, b in zip(res["a"], res["b"]):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  first divergence at token {i}: {x} vs {y}"); break
sys.exit(0 if same else 2)
