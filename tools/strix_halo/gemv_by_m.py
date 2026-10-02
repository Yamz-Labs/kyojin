#!/usr/bin/env python3
"""Bucket every exl3_gemv call in a decode run by m (rows), with CUDA-event timing, to bound what
an int8 GEMV port (m <= 2 only) could save. Also counts calls by (K bits) and shape."""
import os, sys, time, argparse
os.environ.setdefault("EXL3_MOE_CFG", "2"); os.environ.setdefault("EXL3_HIP_PREFILL_MIN_ROWS", "2")
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
import exllamav3.ext as extmod
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=96); args = ap.parse_args()

E = extmod.exllamav3_ext
recs = []   # (m, k, n, e0, e1)
orig = E.exl3_gemv
def gemv(A, B, C, *a, **k):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); r = orig(A, B, C, *a, **k); e1.record()
    recs.append((A.shape[0] if A.dim() == 2 else A.numel() // A.shape[-1], A.shape[-1], C.shape[-1], e0, e1))
    return r
E.exl3_gemv = gemv

def main():
    config = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
    model = Model.from_config(config); tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_history=2); model.load(progressbar=False)
    dm = Model.from_config(config, component="mtp"); dc = Cache(dm, max_num_tokens=4096, max_history=2); dm.load(progressbar=False)
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc,
                    num_draft_tokens=2, dynamic_draft_tokens=True, draft_confidence=0.4)
    def run(n):
        recs.clear()
        ids = tok.encode("Explain gradient descent in two sentences:", add_bos=True)
        gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler()))
        t0 = time.perf_counter(); ntok = 0; ttft = None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("error"): print("JOB ERROR", r["error"])
                if r.get("text"):
                    ntok += 1
                    if ttft is None: ttft = time.perf_counter() - t0
        torch.cuda.synchronize()
        return (ntok - 1) / (time.perf_counter() - t0 - ttft), time.perf_counter() - t0 - ttft
    run(16)
    tps, wall = run(args.n)
    print(f"\ndecode {tps:.2f} tok/s   decode wall {wall*1e3:.0f} ms   gemv calls {len(recs)}")
    by_m = defaultdict(lambda: [0, 0.0]); by_shape = defaultdict(lambda: [0, 0.0])
    for m, k, n, e0, e1 in recs:
        if not e1.query(): continue
        ms = e0.elapsed_time(e1)
        by_m[m][0] += 1; by_m[m][1] += ms
        by_shape[(m, k, n)][0] += 1; by_shape[(m, k, n)][1] += ms
    tot = sum(v[1] for v in by_m.values())
    print(f"exl3_gemv total GPU {tot:.0f} ms = {100*tot/(wall*1e3):.1f}% of decode wall")
    for m in sorted(by_m):
        c, ms = by_m[m]
        print(f"  m={m:<3d} calls {c:<6d} {ms:7.0f} ms  ({100*ms/tot:4.1f}% of gemv, {100*ms/(wall*1e3):4.1f}% of wall)  {1e3*ms/c:6.1f} us/call")
    print("top shapes:")
    for (m, k, n), (c, ms) in sorted(by_shape.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  m={m:<3d} k={k:<6d} n={n:<6d} calls {c:<6d} {ms:7.0f} ms  {1e3*ms/c:6.1f} us/call")

if __name__ == "__main__":
    main()
