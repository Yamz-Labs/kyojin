#!/usr/bin/env python3
"""GPU time per BlockSparseMLP.forward (CUDA events) split by batch size, for mcs=0 vs N.
Tells us whether removing experts from the GPU shrinks the grouped kernel proportionally, and
what the handoff adds. Greedy, MTP ndt=2 dds."""
import os, sys, time, argparse
os.environ.setdefault("EXL3_MOE_CFG", "2"); os.environ.setdefault("EXL3_HIP_PREFILL_MIN_ROWS", "2")
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
import exllamav3.modules.block_sparse_mlp as bsm

ap = argparse.ArgumentParser()
ap.add_argument("-mcs", type=int, default=0)
ap.add_argument("-n", type=int, default=96)
args = ap.parse_args()

evs = []   # (bsz, e0, e1, is_mtp)
orig_fwd = bsm.BlockSparseMLP.forward
def fwd(self, x, params, *a, **k):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); r = orig_fwd(self, x, params, *a, **k); e1.record()
    evs.append((x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1], e0, e1, "mtp" in self.key))
    return r
bsm.BlockSparseMLP.forward = fwd

def main():
    config = Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3"))
    if args.mcs: config.infer_params.moe_cpu_split = args.mcs
    model = Model.from_config(config); tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_history=2); model.load(progressbar=False)
    dm = Model.from_config(config, component="mtp"); dc = Cache(dm, max_num_tokens=4096, max_history=2); dm.load(progressbar=False)
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc,
                    num_draft_tokens=2, dynamic_draft_tokens=True, draft_confidence=0.4)
    def run(n):
        evs.clear()
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
    print(f"\ndecode {tps:.2f} tok/s  mcs={args.mcs}  decode wall {wall*1e3:.0f} ms")
    from collections import defaultdict
    agg = defaultdict(list)
    for bsz, e0, e1, mtp in evs:
        if e1.query(): agg[(bsz, mtp)].append(e0.elapsed_time(e1))
    tot = 0
    for (bsz, mtp), ms in sorted(agg.items()):
        ms.sort(); s = sum(ms); tot += s
        print(f"  {'mtp ' if mtp else 'main'} bsz={bsz:<3d} n={len(ms):<5d} med {ms[len(ms)//2]:.3f} p90 {ms[int(len(ms)*.9)]:.3f} ms  total {s:.0f} ms")
    print(f"  MoE layers total GPU {tot:.0f} ms = {100*tot/(wall*1e3):.0f}% of decode wall")

if __name__ == "__main__":
    main()
