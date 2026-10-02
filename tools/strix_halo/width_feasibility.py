#!/usr/bin/env python
"""Is a wider verification forward free? The tree-speculation feasibility test.

Decode is weight-bound: one forward streams ~4.9 GiB of weights regardless of how many
token positions ride along. If time(m) is flat from m=3 to m=8, then a *tree* draft
(multiple branches verified in one forward) buys accepted-tokens-per-forward at almost
no byte cost, and tok/s scales with it. If time(m) is linear, rows are not free and the
only lever left is making the forward itself faster.

Measured at current HEAD (int8 mixers, skinny GEMM, expert dedup, block-wide metadata
scan all in), which is NOT what the original multirow_scaling.py saw.

Reports, for each m: forward ms, and the tok/s that a tree achieving `t` accepted
tokens per forward would deliver.
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
from exllamav3 import Config, Model, Cache, Tokenizer

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.8-Flash-Next-EXL3"))
MS = [int(x) for x in os.environ.get("MS", "1,2,3,4,6,8,12,16").split(",")]
REPS = int(os.environ.get("REPS", "10"))
# Current production point, from bench_mtp -n 512 -ndt 3 -dds -g -dc 0.6:
CUR_MS, CUR_TPF, CUR_TPS = 67.1, 2.96, 44.1


def main():
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    # max_history=0: we call model.forward directly with m rows and never roll back, so no
    # rollback history is needed. (max_history=N multiplies the GDN recurrent state by N+1 and
    # OOMs above ~5 on this box -- that is what the -ndt>=6 OOM in the notes actually is.)
    cache = Cache(model, max_num_tokens=4096, max_history=0)
    model.load(progressbar=False)

    ids = tokenizer.encode("The capital of France is a city that", add_bos=True)
    past = ids.numel()
    model.prefill(input_ids=ids.to("cuda"), params={"cache": cache, "past_len": 0})
    torch.cuda.synchronize()

    print(f"{'rows':>5} {'ms/fwd':>9} {'ms/row':>8} {'GB/s':>7} {'vs m=1':>7} "
          f"{'tok/s @100% acc':>16} {'tok/s @70% acc':>15}")
    print("-" * 78)
    PER_FWD_GIB = 4.9
    base = None
    rows_out = []
    for m in MS:
        x = torch.randint(100, 5000, (1, m), dtype=torch.long, device="cuda")
        for _ in range(3):
            model.forward(input_ids=x, params={"cache": cache, "past_len": past})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            model.forward(input_ids=x, params={"cache": cache, "past_len": past})
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / REPS
        if base is None:
            base = dt
        gbs = PER_FWD_GIB * 1.0737 / dt
        # a tree of width*depth = m positions, accepting a fraction, yields ~1 + acc*(m-1)
        tps100 = m / dt
        tps70 = (1 + 0.70 * (m - 1)) / dt
        rows_out.append((m, dt, tps70))
        print(f"{m:>5} {dt*1000:>9.2f} {dt*1000/m:>8.2f} {gbs:>7.1f} {dt/base:>7.2f}x "
              f"{tps100:>16.1f} {tps70:>15.1f}")

    print()
    print(f"production now: {CUR_MS:.1f} ms/fwd, {CUR_TPF:.2f} tok/fwd = {CUR_TPS:.1f} tok/s")
    print("Verdict:")
    m1 = dict((m, dt) for m, dt, _ in rows_out)
    if 3 in m1 and 8 in m1:
        ratio = m1[8] / m1[3]
        print(f"  time(8 rows)/time(3 rows) = {ratio:.2f}x  "
              f"(1.00 = rows free -> tree wins big; 2.67 = rows cost full price)")
    best = max(rows_out, key=lambda r: r[2])
    print(f"  best @70% acceptance: m={best[0]} -> {best[2]:.1f} tok/s")
    if best[2] >= 60:
        print("  >= 60 tok/s IS REACHABLE via wider speculation. Build the tree drafter.")
    else:
        print(f"  ceiling via width alone: {best[2]:.1f} tok/s")


if __name__ == "__main__":
    main()
