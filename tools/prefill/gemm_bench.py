"""Grouped GEMM microbench (exl3_moe_prefill_gemm_test): E experts, random trellis (decode cost does
not depend on values). Row counts per expert: uniform R, or --skew draws counts from a real-looking
routing (Dirichlet-skewed, same total = tokens * top_k). Variants are env knobs flipped in-process
(MPW_LDM is read per launch), so --compare checks every variant against variant 0 on the same data.

  big-gpu-run.sh tools/prefill/env-run.sh python tools/prefill/gemm_bench.py --tokens 4096 16384 --skew --compare
"""
import argparse, os, statistics, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from exllamav3.ext import exllamav3_ext as ext

p = argparse.ArgumentParser()
p.add_argument("--K", type = float, default = 2.0)
p.add_argument("--experts", type = int, default = 288)
p.add_argument("--topk", type = int, default = 8)
p.add_argument("--tokens", type = int, nargs = "+", default = [4096])
p.add_argument("--rows", type = int, nargs = "*", default = [], help = "uniform rows/expert instead of --tokens")
p.add_argument("--shapes", type = str, nargs = "+", default = ["4096x2048", "2048x4096"], help = "KxN")
p.add_argument("--skew", action = "store_true")
p.add_argument("--variants", type = str, nargs = "+", default = ["MPW_LDM=0", "MPW_LDM=1"])
p.add_argument("--compare", action = "store_true")
p.add_argument("--iters", type = int, default = 10)
args = p.parse_args()
DEV = "cuda:0"
E = args.experts
torch.manual_seed(0)

def counts_for(tokens):
    total = tokens * args.topk
    if not args.skew:
        c = torch.full((E,), total // E, dtype = torch.long)
        c[: total - c.sum()] += 1
        return c
    w = torch.distributions.Dirichlet(torch.full((E,), 2.0)).sample()
    c = (w * total).floor().long()
    c[torch.argsort(w, descending = True)[: total - c.sum()]] += 1
    return c

KNOBS = {kv.split("=")[0] for v in args.variants for kv in v.split(",")}

def set_variant(v):
    for k in KNOBS: os.environ.pop(k, None)     # knobs of one variant must not leak into the next
    for kv in v.split(","):
        k, val = kv.split("=")
        os.environ[k] = val

cases = [("R", r, torch.full((E,), r, dtype = torch.long)) for r in args.rows] or \
        [("T", t, counts_for(t)) for t in args.tokens]
for shape in args.shapes:
    k, n = map(int, shape.split("x"))
    words = int(16 * args.K)
    tr = [torch.randint(-32768, 32767, (k // 16, n // 16, words), dtype = torch.int16, device = DEV) for _ in range(E)]
    table = torch.tensor([t.data_ptr() for t in tr], dtype = torch.long, device = DEV)
    for tag, val, cnt_cpu in cases:
        rows = int(cnt_cpu.sum())
        A = torch.randn((rows, k), device = DEV).half()
        cnt = torch.cat([cnt_cpu, torch.zeros(1, dtype = torch.long)]).to(DEV)
        offs = torch.empty((E + 1,), dtype = torch.long, device = DEV)
        tiles = torch.empty((rows // 64 + E + 1,), dtype = torch.int, device = DEV)
        tc = torch.empty((1,), dtype = torch.int, device = DEV)
        ref = None
        for v in args.variants:
            set_variant(v)
            C = torch.empty((rows, n), device = DEV).half()
            f = lambda: ext.exl3_moe_prefill_gemm_test(A, C, cnt, table, args.K, n, offs, tiles, tc)
            for _ in range(3): f()
            ts = []
            for _ in range(args.iters):
                e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
                e0.record(); f(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
            t = statistics.median(ts)
            extra = ""
            if args.compare:
                if ref is None: ref = C.float().clone()
                else:
                    d = (C.float() - ref)
                    extra = f"  maxabs {d.abs().max().item():.3e} relL2 {(d.norm() / ref.norm()).item():.3e} equal {torch.equal(C.float(), ref)}"
            print(f"{shape} K={args.K} E={E} {tag}={val:5d} rows={rows:6d} {v:14s}: {t:7.3f} ms "
                  f"{2 * rows * k * n / t / 1e9:6.1f} TFLOP/s{extra}", flush = True)
    del tr, table
