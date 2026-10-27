"""GPU timings of the dspark path on a synthetic pack at real dims (see dspark_synth_pack.py).

    python3 dspark_gpu_bench.py <pack> [--ctx 1024] [--reps 7] [--json out.json] [--profile]

Prints, median of --reps:
  decode      one-row forward (plain decode step) of the trunk in the pack
  draft(N)    PortDrafter.draft for depth 1, 5, 10 (one draft pass up to the block size, chained passes above it)
  verify(n)   PortTarget.verify of n rows (anchor + n-1 drafts) followed by truncate(keep 1): cost ratio per extra row
  round(N)    draft(N) + verify(N+1) + truncate, the full spec round
and a greedy equality check spec == plain over a short run (random weights: almost every round rejects, the cache logic
and the numerics are what is compared).
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]


def sync():
    torch.cuda.synchronize()


def med(f, reps):
    f(); sync()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); f(); sync(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--json")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--equality", type=int, default=24, help="tokens of the spec == plain greedy check (0 = skip)")
    a = ap.parse_args()
    from ds41_serving import build_model
    from exllamav3.dsv41.spec import PortSpec
    import dspark as ctl

    m = build_model(a.pack, device="cuda:0", dtype="float16", layers=None, mtp=True, moe="auto", prefill_chunk=a.ctx)
    sync()
    cfg = m.cfg
    spec = PortSpec(m, a.ctx + 64, max_depth=10)
    g = torch.Generator().manual_seed(5)
    prompt = torch.randint(0, cfg.vocab_size - 1, (a.ctx,), generator=g).tolist()
    out = spec.target.prefill(prompt)
    pos = len(prompt)
    hid = out.hidden[-1:]
    res = {"layers": cfg.num_hidden_layers, "stages": cfg.n_mtp_layers, "ctx": a.ctx,
           "backends": sorted({type(b).__name__ for b in m.moe_backends.values()})}

    def decode_step():
        spec.target.s.length = pos
        ids = torch.tensor([[1]])
        m._forward(ids, pos, spec.st, None, "last")

    res["decode_ms"] = med(decode_step, a.reps)

    def draft(depth):
        spec.drafter.max_depth = spec.drafter.block_size = depth
        spec.drafter._next = pos
        return lambda: spec.drafter.draft(1, hid, pos)

    for dpt in (1, 5, 10):
        res[f"draft{dpt}_ms"] = med(draft(dpt), a.reps)

    def verify(rows):
        def f():
            spec.target.s.length = pos
            spec.target.verify([1] * rows, pos)
            spec.target.truncate(pos + 1)
        return f

    for rows in (1, 2, 3, 4, 6, 8, 11):
        res[f"verify{rows}_ms"] = med(verify(rows), a.reps)

    def rnd(depth):
        def f():
            spec.target.s.length = pos
            d = draft(depth)()
            spec.target.verify([1, *d.tokens[:depth]], pos)
            spec.target.truncate(pos + 1)
        return f

    for dpt in (1, 2, 3, 4, 5, 6, 8, 10):
        res[f"round{dpt}_ms"] = med(rnd(dpt), a.reps)

    # draft tokens / confidence: fused hyper-connection kernels vs the torch ops
    from exllamav3.dsv41.serve_ops import TorchHC
    spec.drafter.max_depth = spec.drafter.block_size = 5
    spec.drafter._next = pos
    d_fast = spec.drafter.draft(1, hid, pos)
    hc0, m.hc = m.hc, TorchHC(cfg, m.dtype)
    spec.drafter._next = pos
    d_ref = spec.drafter.draft(1, hid, pos)
    m.hc = hc0
    res["draft_fused_vs_torch_tokens_equal"] = d_fast.tokens == d_ref.tokens
    res["draft_conf_maxdiff"] = float((d_fast.conf.float() - d_ref.conf.float()).abs().max())
    print(json.dumps(res), flush=True)
    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        spec.drafter.max_depth = spec.drafter.block_size = 5
        draft(5)()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            draft(5)(); sync()
        print(pr.key_averages().table(sort_by="self_device_time_total", row_limit=22, max_name_column_width=50))

    if a.equality:
        spec.drafter.max_depth = spec.drafter.block_size = 10
        eos = 1
        toks, dec = ctl.decode(spec.target, spec.drafter, prompt[:256], a.equality, eos, num_draft=5)
        spec.reset()
        ids = torch.tensor([prompt[:256]])
        lg, _ = m.forward(ids, 0, spec.st)
        ref, margins, p = [], [], 256
        cur = int(lg.argmax())
        for _ in range(a.equality):
            t2 = lg[0].topk(2).values
            margins.append(float(t2[0] - t2[1]))
            amax = float(lg.abs().max())
            ref.append(cur)
            if cur == eos:
                break
            lg, _ = m._forward(torch.tensor([[cur]]), p, spec.st, None, "last"); p += 1
            cur = int(lg.argmax())
        n = min(len(ref), len(toks))
        res["spec_equals_plain"] = toks[:n] == ref[:n] and n > 0
        fd = next((i for i in range(n) if toks[i] != ref[i]), None)
        res["spec_first_divergence"] = fd
        res["plain_top1_margin_at_divergence"] = None if fd is None else margins[fd]
        res["plain_logit_absmax_last"] = amax
        res["spec_equal_tokens"] = n
        res["accept_per_step"] = round(dec.stats.accepted_per_step, 3)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
