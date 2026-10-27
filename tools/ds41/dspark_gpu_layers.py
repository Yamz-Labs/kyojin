"""Verify cost of n rows against the number of trunk layers (fit fixed + per-layer cost for each n), no dspark stages.
    python3 dspark_gpu_layers.py <pack> [--layers 1,2,3] [--ctx 1024] [--reps 7] [--json out.json]"""
import argparse, json, os, statistics, sys, time, gc
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--layers", default="1,2,3"); ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--reps", type=int, default=7); ap.add_argument("--json")
a = ap.parse_args()
from ds41_serving import build_model
ROWS = (1, 2, 3, 4, 6, 8, 11)
res = {}
def med(f):
    f(); torch.cuda.synchronize(); ts = []
    for _ in range(a.reps):
        torch.cuda.synchronize(); t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000
for L in [int(x) for x in a.layers.split(",")]:
    m = build_model(a.pack, device="cuda:0", dtype="float16", layers=L, mtp=False, moe="auto", prefill_chunk=a.ctx)
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, m.cfg.vocab_size - 1, (1, a.ctx), generator=g)
    st = m.new_state(1, a.ctx + 64); m.prefill(ids, st, 0); pos = a.ctx
    row = {}
    for n in ROWS:
        tok = torch.randint(0, m.cfg.vocab_size - 1, (1, n), generator=g)
        def f():
            st.begin_block(pos, n); m.forward_block(tok, pos, st); (st.end_block() if n == 1 else st.rollback(1))
        row[n] = med(f)
    res[L] = row; print(L, {k: round(v, 2) for k, v in row.items()}, flush=True)
    del m, st; gc.collect(); torch.cuda.empty_cache()
if a.json: json.dump(res, open(a.json, "w"), indent=1)
