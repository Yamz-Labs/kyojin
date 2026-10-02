#!/usr/bin/env python
"""Where is the 26% non-kernel time? Use the torch profiler, not guesses.

Roofline says 236 GB/s / 6.7 GiB-per-forward with ~1.63 tok/forward => ~53 tok/s
ceiling. We measure 29.2, i.e. ~40% of peak bandwidth, and CUDA-event accounting
puts only 74% of wall time inside ext kernels. So either:
  (a) host/Python time between launches (fixable), or
  (b) the kernels themselves are far from bandwidth-bound (kernel work), or
  (c) gaps/sync inside the GPU timeline (overlap work).

torch.profiler with ROCm activity gives self-CPU vs self-device time and the
launch count, which separates (a) from (b).
"""
import os, sys, time
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
NDT = 2


def main():
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

    # warm up so capture/JIT is not in the profile
    gen.enqueue(Job(input_ids=ids, max_new_tokens=8, sampler=GreedySampler()))
    while gen.num_remaining_jobs():
        for _ in gen.iterate():
            pass

    torch.cuda.synchronize()
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        gen.enqueue(Job(input_ids=ids, max_new_tokens=96, sampler=GreedySampler()))
        n = 0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("text"):
                    n += 1
        torch.cuda.synchronize()
    wall = time.time() - t0

    print(f"profiled {n} tokens in {wall:.2f}s ({n/wall:.1f} tok/s with profiler on)")
    print()
    ka = prof.key_averages()

    dev_total = sum(getattr(e, "self_device_time_total", 0) or 0 for e in ka)
    cpu_total = sum(e.self_cpu_time_total for e in ka)
    print(f"sum self_device_time = {dev_total/1000:.1f} ms")
    print(f"sum self_cpu_time    = {cpu_total/1000:.1f} ms")
    print(f"wall                 = {wall*1000:.1f} ms")
    print(f"=> device busy {100*dev_total/1e6/wall:.1f}% of wall")
    print()

    print("TOP 28 BY SELF DEVICE TIME")
    print(f"{'op':>42} {'calls':>7} {'dev_ms':>9} {'us/call':>9}")
    print("-" * 72)
    rows = sorted(ka, key=lambda e: -(getattr(e, "self_device_time_total", 0) or 0))
    for e in rows[:28]:
        d = (getattr(e, "self_device_time_total", 0) or 0) / 1000
        if d <= 0:
            continue
        print(f"{e.key[:42]:>42} {e.count:>7} {d:>9.1f} {d*1000/max(e.count,1):>9.1f}")

    print()
    print("TOP 28 BY SELF CPU TIME (host overhead candidates)")
    print(f"{'op':>42} {'calls':>7} {'cpu_ms':>9} {'us/call':>9}")
    print("-" * 72)
    rows = sorted(ka, key=lambda e: -e.self_cpu_time_total)
    for e in rows[:28]:
        c = e.self_cpu_time_total / 1000
        print(f"{e.key[:42]:>42} {e.count:>7} {c:>9.1f} {c*1000/max(e.count,1):>9.1f}")

    total_launches = sum(e.count for e in ka
                         if (getattr(e, "self_device_time_total", 0) or 0) > 0)
    print()
    print(f"total device ops in {n} tokens: {total_launches} "
          f"({total_launches/max(n,1):.0f} per token)")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
