"""Decode kernel trace driver. usage (under gpurun):
  rocprofv3 --kernel-trace --output-format csv -d OUT -o dec -- python prof_decode.py PACK [--layers 3] [--ctx 512]
Prints the CLOCK_MONOTONIC ns window of the measured steps (MARK lines) so analyze_trace.py can cut the CSV.
Without rocprofv3 it just reports ms/step."""
import argparse, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds41_serving import build_model

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--layers", type=int, default=3); ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--steps", type=int, default=4); ap.add_argument("--moe", default="auto")
a = ap.parse_args()
m = build_model(a.pack, device="cuda:0", dtype="float16", layers=a.layers, mtp=False, moe=a.moe, prefill_chunk=a.ctx)
print("backends", sorted({type(b).__name__ for b in m.moe_backends.values()}), "layers", a.layers, flush=True)
ids = torch.randint(0, m.cfg.vocab_size - 1, (1, a.ctx), device="cuda:0", generator=torch.Generator(device="cuda:0").manual_seed(5))
st = m.new_state(1, a.ctx + 128)
lg, _ = m.prefill(ids, st, 0, chunk=a.ctx)
tok = lg.argmax(-1).view(1, 1); pos = a.ctx
def steps(n):
    global tok, pos
    for _ in range(n):
        lg, _ = m.forward(tok, pos, st); pos += 1; tok = lg.argmax(-1).view(1, 1)
steps(8); torch.cuda.synchronize()
t = time.time(); steps(16); torch.cuda.synchronize(); print("ms/step", (time.time() - t) / 16 * 1e3, flush=True)
torch.cuda.synchronize(); t0 = time.monotonic_ns(); steps(a.steps); torch.cuda.synchronize(); t1 = time.monotonic_ns()
print(f"MARK {t0} {t1} steps {a.steps}", flush=True)
