#!/usr/bin/env python
"""How much expert overlap is there between speculative rows?

The grouped-MoE kernel issues assignments = rows * top_k, one GEMV per
(row, expert) pair. So at R=2 it reads up to 2x the expert weights. Measured
forward cost confirms linear row scaling:

    rows 2 -> 72 ms,  rows 3 -> 84 ms,  rows 5 -> 124 ms
    => ~38 ms fixed + ~17 ms per row

If the rows in a verification batch select OVERLAPPING experts, those reads are
redundant and dedup would cut the per-row cost. Consecutive tokens are highly
correlated, so this could be large -- or the router could be diverse by design,
in which case there is nothing to reclaim and the per-row cost is irreducible.

Measures the real selected-expert sets during MTP decode.
"""
import os, sys, collections
import torch

sys.path.insert(0, os.path.expanduser("~/kyojin"))
import exllamav3_ext as X
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

    stats = {"calls": 0, "rows": 0, "assign": 0, "unique": 0}
    hist = collections.Counter()
    real = X.exl3_moe_gfx12_k3

    def spy(*a, **k):
        sel = a[2]                      # selected experts [R, top_k] int64
        if sel.dim() == 2 and sel.shape[0] > 1:
            R, K = sel.shape
            s = sel.detach().to("cpu")
            allv = s.flatten().tolist()
            uniq = len(set(v for v in allv if v >= 0))
            stats["calls"] += 1
            stats["rows"] += R
            stats["assign"] += sum(1 for v in allv if v >= 0)
            stats["unique"] += uniq
            hist[(R, sum(1 for v in allv if v >= 0), uniq)] += 1
        return real(*a, **k)

    X.exl3_moe_gfx12_k3 = spy
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft, draft_cache=dcache,
                    num_draft_tokens=NDT, dynamic_draft_tokens=True,
                    draft_confidence=0.4)
    ids = tokenizer.encode("Explain gradient descent in two sentences:", add_bos=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=48, sampler=GreedySampler()))
    n = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("text"):
                n += 1
    X.exl3_moe_gfx12_k3 = real

    print(f"tokens generated: {n}")
    print(f"multi-row MoE calls observed: {stats['calls']}")
    if not stats["calls"]:
        print("no multi-row calls seen")
        return
    a, u = stats["assign"], stats["unique"]
    print(f"total assignments (rows x top_k): {a}")
    print(f"total UNIQUE experts needed:      {u}")
    print(f"redundant expert reads:           {a-u}  ({100*(a-u)/a:.1f}%)")
    print()
    print(f"{'(rows, assigns, unique)':>26}  {'count':>7}  {'dedup saving':>13}")
    print("-" * 54)
    for k, c in hist.most_common(8):
        R, asg, uq = k
        print(f"{str(k):>26}  {c:>7}  {100*(asg-uq)/asg:>12.1f}%")
    print()
    saving = (a - u) / a
    print(f"AVERAGE dedup opportunity: {100*saving:.1f}% of expert bytes")
    per_row_ms = 17.0
    print(f"per-row forward cost is ~{per_row_ms:.0f} ms; dedup would cut the "
          f"expert part by {100*saving:.0f}%")
    if saving < 0.05:
        print("=> NOTHING to reclaim: rows pick nearly disjoint experts.")
        print("   The per-row cost is irreducible; MoE decode must get faster instead.")
    else:
        print("=> worth implementing: dedup assignments before the GEMV.")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
