#!/usr/bin/env python
"""Does the multi-row verification forward amortize the weight read?

Roofline: no-MTP decode achieves 55% of peak bandwidth; MTP achieves only 34%.
If a 3-row forward cost the same as a 1-row forward (weights read once, 3 rows
of math), MTP would be nearly free and we'd see ~2.5x. We see 1.63x, so the
multi-row path is losing something.

Measure trunk forward latency at m = 1, 2, 3, 4, 8 directly. Interpretation:
  * time(m) flat in m        -> weight read amortized, MTP limited elsewhere
  * time(m) ~ linear in m    -> weights RE-READ per row: the multi-row MoE path
                                is not batching, and fixing that is the win
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")


def main():
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=2048, max_history=0)
    model.load(progressbar=False)

    ids = tokenizer.encode("The capital of France is a city that", add_bos=True)
    past = ids.numel()
    # prime the cache with a prefill
    model.prefill(input_ids=ids.to("cuda"), params={"cache": cache, "past_len": 0})
    torch.cuda.synchronize()

    print(f"{'rows':>5} {'ms/forward':>12} {'ms/row':>9} {'GB/s':>8} {'%peak':>7}"
          f"  {'tok/s if all accepted':>22}")
    print("-" * 76)
    PER_FWD_GIB = 6.698
    PEAK = 236.0
    base = None
    for m in (1, 2, 3, 4, 8):
        x = torch.randint(100, 5000, (1, m), dtype=torch.long, device="cuda")
        # warm
        for _ in range(2):
            model.forward(input_ids=x, params={"cache": cache, "past_len": past})
        torch.cuda.synchronize()
        reps = 8
        t0 = time.perf_counter()
        for _ in range(reps):
            model.forward(input_ids=x, params={"cache": cache, "past_len": past})
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        gbs = PER_FWD_GIB * 1.0737 / dt
        if base is None:
            base = dt
        print(f"{m:>5} {dt*1000:>12.2f} {dt*1000/m:>9.2f} {gbs:>8.1f} "
              f"{100*gbs/PEAK:>6.1f}% {m/dt:>22.1f}")

    print()
    print("If ms/forward is ~flat, the weight read IS amortized across rows.")
    print("If it scales with rows, the multi-row path re-reads weights per row.")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
