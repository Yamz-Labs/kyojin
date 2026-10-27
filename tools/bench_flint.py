"""Flint speed bench, one (depth) row per process. usage:
  bench_flint.py PACK --depth 32768 [--gen 64] [--kv 16|8,8|6,6|4,4] [--seed 0] [-o row.json]
Prompt = real text tokens (c4/wiki/code mix). pp = depth / ttft; tg = steady decode tok/s over --gen tokens (greedy)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.presets import GreedySampler

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--depth", type=int, required=True); ap.add_argument("--gen", type=int, default=64)
ap.add_argument("--kv", default="16"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("-o", "--out")
ap.add_argument("--chunk", type=int, default=0)
a = ap.parse_args()

cfg = Config.from_directory(a.pack)
model = Model.from_config(cfg)
tok = Tokenizer.from_config(cfg)
ctx = ((a.depth + a.gen + 512 + 255) // 256) * 256
if a.kv == "16":
    cache = Cache(model, max_num_tokens=ctx)
else:
    kb, vb = (int(x) for x in a.kv.split(","))
    cache = Cache(model, max_num_tokens=ctx, layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
t0 = time.time()
model.load(progressbar=False)
print(f"loaded in {time.time() - t0:.1f}s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
kw = {"max_chunk_size": a.chunk} if a.chunk else {}
gen = Generator(model=model, cache=cache, tokenizer=tok, **kw)

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3/conversion/standard_cal_data")
def real_ids(n, seed):
    out = []
    for f in ("c4.utf8", "wiki.utf8", "code.utf8", "technical.utf8"):
        s = open(f"{D}/{f}", encoding="utf8").read()
        a0 = (seed * 7919) % max(1, len(s) // 3)
        out.append(tok.encode(s[a0:a0 + max(int(n * 6), 200000)], add_bos=False)[0])
    ids = torch.cat(out)
    # interleave blocks of 4096 tokens from each source so every source is present at every depth
    blocks = [o[:len(o) // 4096 * 4096].view(-1, 4096) for o in out]
    mix = torch.cat([blocks[i % 4][(i // 4) % blocks[i % 4].shape[0]] for i in range(n // 4096 + 2)])
    return mix[:n].view(1, -1)

def run(ids, ngen):
    job = Job(input_ids=ids, max_new_tokens=ngen, sampler=GreedySampler())
    gen.enqueue(job)
    t0 = time.time(); ttft = None; times = []; ntok = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): print("JOB ERROR", r["error"], flush=True)
            if r.get("text", ""):
                now = time.time()
                if ttft is None: ttft = now - t0
                else: times.append(now)
                ntok += 1
    return ttft, ntok, times, time.time() - t0

run(real_ids(256, 99), 8)   # warm-up (graph capture, kernels)
torch.cuda.reset_peak_memory_stats()
ids = real_ids(a.depth, 1000 + a.seed)
ttft, ntok, times, total = run(ids, a.gen)
if ttft is None:    # no token came back: the job errored (printed above), e.g. prompt longer than max_seq_len or cache OOM
    print(json.dumps({"depth": a.depth, "kv": a.kv, "error": "no tokens generated", "ctx": ctx}), flush=True)
    sys.exit(1)
tg = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 2 else 0.0
# --gen of 3 x 64 or more: also report the steady decode rate of each consecutive 64-token window and their median (shared-box noise)
wins = [(63 / (times[i + 63] - times[i])) for i in range(0, len(times) - 63, 64)] if len(times) >= 127 else []
if wins: tg = sorted(wins)[len(wins) // 2]
res = {"depth": a.depth, "kv": a.kv, "ttft": ttft, "pp": a.depth / ttft, "tg": tg, "tg_windows": wins, "ntok": ntok,
       "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
print(json.dumps(res), flush=True)
if a.out: json.dump(res, open(a.out, "w"))
r = model.__dict__.get("_cuda_graph_runner")
if r is not None: print("GRAPH", dict(r.stats.get("declines", {})), {k: v for k, v in r.stats.items() if k != "declines"}, "model_ok", r._model_ok, flush=True)
