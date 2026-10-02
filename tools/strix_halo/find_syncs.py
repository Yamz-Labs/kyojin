#!/usr/bin/env python
"""Find the blocking D2H syncs. hipMemcpyWithStream = 713 ms of host time over
154 calls (4.6 ms each) -- by far the largest single cost, 6.4 per token.

A blocking device->host copy stalls the host until the GPU drains, which both
adds latency AND prevents the host from running ahead to enqueue the next
layer's kernels. Classic causes: .item(), .cpu(), .tolist(), bool(tensor),
int(tensor), or a python-side if on a device tensor.

Strategy: monkey-patch the Tensor methods that force a sync, record a stack for
each, and report the hottest call sites. Then we know exactly what to fix.
"""
import os, sys, time, collections, traceback
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
NDT = 2

sites = collections.Counter()
timing = collections.Counter()
RECORDING = False


def instrument(name, orig):
    def wrapper(self, *a, **k):
        global RECORDING
        if RECORDING and isinstance(self, torch.Tensor) and self.is_cuda:
            st = traceback.extract_stack()[:-1]
            frames = [f for f in st if "exllamav3" in f.filename]
            key = " <- ".join(
                f"{os.path.basename(f.filename)}:{f.lineno}({f.name})"
                for f in reversed(frames[-3:])) or f"<{name} outside exllamav3>"
            t0 = time.perf_counter()
            r = orig(self, *a, **k)
            dt = (time.perf_counter() - t0) * 1000
            sites[f"[{name}] {key}"] += 1
            timing[f"[{name}] {key}"] += dt
            return r
        return orig(self, *a, **k)
    return wrapper


PATCHED = {}
for nm in ("item", "tolist", "cpu", "numpy", "__bool__", "__int__", "__float__"):
    o = getattr(torch.Tensor, nm, None)
    if o is not None:
        PATCHED[nm] = o
        setattr(torch.Tensor, nm, instrument(nm, o))


def main():
    global RECORDING
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_history=NDT)
    model.load(progressbar=False)
    draft = Model.from_config(config, component="mtp")
    dcache = Cache(draft, max_num_tokens=4096, max_history=NDT)
    draft.load(progressbar=False)

    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft, draft_cache=dcache,
                    num_draft_tokens=NDT, dynamic_draft_tokens=True,
                    draft_confidence=0.4)
    ids = tokenizer.encode("Explain gradient descent in two sentences:", add_bos=True)

    gen.enqueue(Job(input_ids=ids, max_new_tokens=8, sampler=GreedySampler()))
    while gen.num_remaining_jobs():
        for _ in gen.iterate():
            pass

    sites.clear()
    timing.clear()
    RECORDING = True
    torch.cuda.synchronize()
    t0 = time.time()
    gen.enqueue(Job(input_ids=ids, max_new_tokens=32, sampler=GreedySampler()))
    n = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("text"):
                n += 1
    torch.cuda.synchronize()
    wall = time.time() - t0
    RECORDING = False

    print(f"{n} tokens in {wall:.2f}s = {n/wall:.2f} tok/s")
    print(f"total sync-forcing calls: {sum(sites.values())} "
          f"({sum(sites.values())/max(n,1):.1f} per token)")
    print(f"total time in them: {sum(timing.values()):.1f} ms "
          f"({100*sum(timing.values())/1000/wall:.1f}% of wall)")
    print()
    print(f"{'ms':>9} {'calls':>7} {'us/call':>9}  site")
    print("-" * 100)
    for key, ms in timing.most_common(16):
        c = sites[key]
        print(f"{ms:>9.1f} {c:>7} {ms*1000/max(c,1):>9.1f}  {key}")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
