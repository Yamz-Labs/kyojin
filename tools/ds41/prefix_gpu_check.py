"""GPU exactness of prompt-cache reuse on a real pack: next-token logits after a resumed prefill vs a full prefill, and 64 greedy
tokens (plain and drafted) with and without reuse. python prefix_gpu_check.py PACK [--engram-dir D] [--first 6144] [--extra 50]"""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--engram-dir"); ap.add_argument("--first", type=int, default=6144)
ap.add_argument("--extra", type=int, default=50); ap.add_argument("--chunk", type=int, default=16384); ap.add_argument("--new", type=int, default=64)
ap.add_argument("--data", default=os.path.expanduser("~/ds41_work/ds41ready_data")); ap.add_argument("--out")
a = ap.parse_args()
from ds41_serving import build_model
from ds41_engine import HFTokenizer
from ds41_model import PortEngine, TopPSampler
from exllamav3.dsv41 import prefix_cache as pc
m = build_model(a.pack, device="cuda:0", dtype="float16", layers=None, mtp=True, moe="auto", prefill_chunk=a.chunk, engram_dir=a.engram_dir)
tk = HFTokenizer(os.path.join(a.pack, "tokenizer.json"))
pool = np.load(os.path.join(a.data, "short.npy")).astype(np.int64).reshape(-1).tolist()
p1 = pool[:a.first]; p2 = p1 + pool[a.first:a.first + a.extra]
res = {}
# 1. logits: resumed vs full
cold = pc.prefill(m, None, p2, max_seq=len(p2) + 128)
store = pc.PrefixStore(1 << 33, 16, min_free_gib=3)
pc.prefill(m, store, p1, max_seq=len(p2) + 128)
warm = pc.prefill(m, store, p2, max_seq=len(p2) + 128)
lc, lw = cold.logits[-1].float(), warm.logits[-1].float()
res["resume_cached"] = warm.cached
res["logit_max_abs_diff"] = float((lc - lw).abs().max()); res["logit_scale"] = float(lc.abs().max())
res["argmax_equal"] = int(lc.argmax()) == int(lw.argmax())
res["top10_overlap"] = len(set(lc.topk(10).indices.tolist()) & set(lw.topk(10).indices.tolist()))
print(json.dumps(res), flush=True)
del cold, warm
# 2. greedy tokens, plain and drafted, with reuse vs cold plain
def run(eng, ids):
    out = []; t0 = time.perf_counter()
    st = eng._run(list(ids), a.new, TopPSampler(0.0), -1, None, out.extend)
    return out, st, time.perf_counter() - t0
ref, _, _ = run(PortEngine(m, tk, ctx=32768, num_draft=0, eos_id=None, prefix_cache=False), p2)
for draft in (0, 4):
    eng = PortEngine(m, tk, ctx=32768, num_draft=draft, eos_id=None, prefix_cache=True)
    run(eng, p1)
    got, st, dt = run(eng, p2)
    first_diff = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), None)
    r = {"draft": draft, "cached": st["cached_tokens"], "greedy_equal": got == ref, "first_diff": first_diff, "n": len(got), "ttft_s": round(st["time_prefill"], 2)}
    print(json.dumps(r), flush=True); res[f"draft{draft}"] = r
if a.out: open(a.out, "a").write(json.dumps(res) + "\n")
