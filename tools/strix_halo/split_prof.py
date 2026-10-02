#!/usr/bin/env python3
"""Per-layer timing of the CPU-split handoff on HIP: brackets around issue / GPU work / collect
with CUDA events, plus host wall time for each Python call, so we can see whether the ~1.3
ms/layer loss is GPU-side (wait kernel, copies) or host-side (Python, flag ring)."""
import os, sys, time, argparse, statistics
sys.argv += []  # placeholder
os.environ.setdefault("EXL3_MOE_CFG", "2"); os.environ.setdefault("EXL3_HIP_PREFILL_MIN_ROWS", "2")
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
import exllamav3.modules.block_sparse_mlp as bsm
import exllamav3.modules.block_sparse_mlp_cpu as bsc
import exllamav3.model.moe_cpu_host as mch

ap = argparse.ArgumentParser()
ap.add_argument("-mcs", type=int, default=64)
ap.add_argument("-n", type=int, default=96)
args = ap.parse_args()

# ---- instrument -------------------------------------------------------------------------
host_t = {"submit": [], "combine": [], "issue_compute": [], "collect": []}
ev_pairs = {"submit": [], "combine": []}

def wrap_host(cls, name, key):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        host_t[key].append((time.perf_counter() - t0) * 1e3)
        e1.record(); ev_pairs[key].append((e0, e1))
        return r
    setattr(cls, name, w)

def wrap_plain(cls, name, key):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        t0 = time.perf_counter(); r = orig(self, *a, **k)
        host_t[key].append((time.perf_counter() - t0) * 1e3); return r
    setattr(cls, name, w)

wrap_host(bsc.BlockSparseMLP_CPU, "cpu_split_submit", "submit")
wrap_host(bsc.BlockSparseMLP_CPU, "cpu_split_combine", "combine")
wrap_plain(mch.MoeCpuHost, "_issue_compute", "issue_compute")
wrap_plain(mch.MoeCpuHost, "_collect_compute", "collect")

def main():
    # ---- model -----------------------------------------------------------------------------
    config = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
    if args.mcs: config.infer_params.moe_cpu_split = args.mcs
    model = Model.from_config(config); tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_history=2); model.load(progressbar=False)
    dm = Model.from_config(config, component="mtp"); dc = Cache(dm, max_num_tokens=4096, max_history=2); dm.load(progressbar=False)
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc,
                    num_draft_tokens=2, dynamic_draft_tokens=True, draft_confidence=0.4)

    def run(n):
        for k in host_t: host_t[k].clear()
        for k in ev_pairs: ev_pairs[k].clear()
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
        return (ntok - 1) / (time.perf_counter() - t0 - ttft)

    run(16)  # warm
    tps = run(args.n)
    print(f"\ndecode {tps:.2f} tok/s  (mcs={args.mcs})")
    def stats(xs): 
        xs = sorted(xs); return f"n={len(xs)} med {xs[len(xs)//2]:.3f} p90 {xs[int(len(xs)*.9)]:.3f} max {xs[-1]:.3f} sum {sum(xs):.1f} ms"
    for k, v in host_t.items():
        if v: print(f"host  {k:14s} {stats(v)}")
    for k, v in ev_pairs.items():
        ms = [a.elapsed_time(b) for a, b in v if b.query()]
        if ms: print(f"gpu   {k:14s} {stats(ms)}")


if __name__ == '__main__':
    main()
