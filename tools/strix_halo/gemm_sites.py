#!/usr/bin/env python
"""Which Python call sites launch the hipblaslt Cijk_* GEMMs during MTP decode?

Uses the chrome trace: kernel events carry a correlation id linking them to the CPU-side
runtime launch, whose parent aten op carries the python stack (with_stack=True).
"""
import os, sys, json, collections, tempfile
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
NDT = 2
PAT = sys.argv[1] if len(sys.argv) > 1 else "Cijk"

config = Config.from_directory(MODEL)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=4096, max_history=NDT)
model.load(progressbar=False)
draft = Model.from_config(config, component="mtp")
dcache = Cache(draft, max_num_tokens=4096, max_history=NDT)
draft.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=NDT, dynamic_draft_tokens=True, draft_confidence=0.4)
PREFILL = int(os.environ.get("PREFILL", "0"))   # >0: profile a cold prefill of this many random tokens instead of decode
if PREFILL:
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(1000, 100000, (1, PREFILL), generator=g, dtype=torch.long)
    NEW = 1
else:
    ids = tokenizer.encode("Explain gradient descent in two sentences:", add_bos=True)
    NEW = 24
gen.enqueue(Job(input_ids=ids[:, :64] if PREFILL else ids, max_new_tokens=8, sampler=GreedySampler()))
while gen.num_remaining_jobs():
    for _ in gen.iterate():
        pass
gen.clear_queue()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
             with_stack=True) as prof:
    gen.enqueue(Job(input_ids=ids, max_new_tokens=NEW, sampler=GreedySampler()))
    n = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("text"):
                n += 1
    torch.cuda.synchronize()

path = os.path.join(tempfile.gettempdir(), "gemm_trace.json")
prof.export_chrome_trace(path)
tr = json.load(open(path))["traceEvents"]

# kernel events: name matches PAT, have args["External id"] (correlates to CPU op)
kern = [e for e in tr if e.get("cat") in ("kernel", "gpu_op", "Kernel") or e.get("cat", "").startswith("kernel")]
kern = [e for e in kern if PAT in e.get("name", "")]
print(f"total device kernel time {sum(e.get('dur',0) for e in tr if e.get('cat')=='kernel')/1e6:.2f}s; matching {sum(e.get('dur',0) for e in kern)/1e6:.2f}s")
cpu = [e for e in tr if e.get("cat") in ("cpu_op",) and "External id" in e.get("args", {})]
by_ext = {}
for e in cpu:
    by_ext.setdefault(e["args"]["External id"], []).append(e)
# python function events for stack lookup: (tid, ts, dur, name)
py = [e for e in tr if e.get("cat") == "python_function"]
py_by_tid = collections.defaultdict(list)
for e in py:
    py_by_tid[e["tid"]].append(e)

def stack_at(tid, ts):
    frames = [e for e in py_by_tid.get(tid, []) if e["ts"] <= ts <= e["ts"] + e.get("dur", 0)]
    frames.sort(key=lambda e: e["ts"])
    out = []
    for f in frames:
        nm = f["name"]
        if "exllamav3/" in nm and "profiler" not in nm:
            out.append(nm.split("exllamav3/")[-1])
    return out[-5:]

agg = collections.defaultdict(lambda: {"n": 0, "us": 0.0, "shapes": set()})
for k in kern:
    ext_id = k.get("args", {}).get("External id")
    ops = by_ext.get(ext_id, [])
    aten = [o for o in ops if o["name"].startswith("aten::")] or list(ops)
    aten.sort(key=lambda o: o.get("dur", 0))
    leaf = aten[0] if aten else None
    if leaf is None:
        key = ("?", "?")
    else:
        st = " <- ".join(reversed(stack_at(leaf["tid"], leaf["ts"] + 1)))
        key = (leaf["name"], st)
        agg[key]["shapes"].add(str(leaf.get("args", {}).get("Input Dims", ""))[:120])
    agg[key]["n"] += 1
    agg[key]["us"] += k.get("dur", 0)

print(f"tokens={n} matching kernels={len(kern)} ({len(kern)/max(n,1):.1f}/token)")
for (op, st), a in sorted(agg.items(), key=lambda kv: -kv[1]["us"])[:12]:
    print(f"\n{op:>12} n={a['n']} total={a['us']/1000:.1f} ms  {a['us']/a['n']:.1f} us/call")
    for s in sorted(a["shapes"])[:4]:
        print(f"   shapes {s}")
    print(f"   {st}")
