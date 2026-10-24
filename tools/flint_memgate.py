"""Memory/gate probe: usage: flint_memgate.py PACK --depth N --kv 8,8 [--gen 128] [--prompts 2] [-o tokens.json]
Prints memory_allocated after load, peak/reserved per prompt, and dumps greedy tokens."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.presets import GreedySampler
ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--depth", type=int, required=True); ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--kv", default="8,8"); ap.add_argument("--prompts", type=int, default=2); ap.add_argument("-o", "--out")
ap.add_argument("--snap", action="store_true")
a = ap.parse_args()
G = 2**30
def nt():
    f, t = torch.cuda.mem_get_info(); return (t - f - torch.cuda.memory_reserved()) / G
cfg = Config.from_directory(a.pack); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
ctx = ((a.depth + a.gen + 512 + 255) // 256) * 256
kb, vb = (int(x) for x in a.kv.split(","))
cache = Cache(model, max_num_tokens=ctx, layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
model.load(progressbar=False)
print(f"after load: alloc {torch.cuda.memory_allocated()/G:.3f} reserved {torch.cuda.memory_reserved()/G:.3f} non_torch {nt():.3f}", flush=True)
gen = Generator(model=model, cache=cache, tokenizer=tok)
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exllamav3/conversion/standard_cal_data")
def real_ids(n, seed):
    out = []
    for f in ("c4.utf8", "wiki.utf8", "code.utf8", "technical.utf8"):
        s = open(f"{D}/{f}", encoding="utf8").read()
        a0 = (seed * 7919) % max(1, len(s) // 3)
        out.append(tok.encode(s[a0:a0 + max(int(n * 6), 200000)], add_bos=False)[0])
    blocks = [o[:len(o) // 4096 * 4096].view(-1, 4096) for o in out]
    mix = torch.cat([blocks[i % 4][(i // 4) % blocks[i % 4].shape[0]] for i in range(n // 4096 + 2)])
    return mix[:n].view(1, -1)
def run(ids, ngen):
    job = Job(input_ids=ids, max_new_tokens=ngen, sampler=GreedySampler(), return_top_tokens=0)
    gen.enqueue(job); t0 = time.time(); ttft = None; times = []; toks = []; txt = ""
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): print("JOB ERROR", r["error"], flush=True)
            if r.get("text", ""):
                now = time.time()
                if ttft is None: ttft = now - t0
                else: times.append(now)
                txt += r["text"]
    return ttft, times, txt
run(real_ids(256, 99), 8)
print(f"after warmup: alloc {torch.cuda.memory_allocated()/G:.3f} reserved {torch.cuda.memory_reserved()/G:.3f} non_torch {nt():.3f}", flush=True)
import threading, collections
free0, tot = torch.cuda.mem_get_info(); print(f"mem_get_info free {free0/G:.3f} total {tot/G:.3f}", flush=True)
cb = sum(t.numel()*t.element_size() for l in cache.layers.values() for t in l.get_tensors()) if hasattr(next(iter(cache.layers.values())), "get_tensors") else -1
print(f"cache bytes {cb/G:.3f} GiB", flush=True)
best = {"peak": 0, "snap": None}; stop = False
def poll():
    import time as _t
    while not stop:
        m = torch.cuda.memory_allocated()
        if m > best["peak"] + (48 << 20) and m > 0:
            best["peak"] = m; best["snap"] = torch.cuda.memory.memory_snapshot()
        _t.sleep(0.02)
if a.snap: threading.Thread(target=poll, daemon=True).start()
res = []
for p in range(a.prompts):
    torch.cuda.reset_peak_memory_stats()
    ttft, times, txt = run(real_ids(a.depth, 1000 + p), a.gen)
    if ttft is None: print("NO TOKENS", torch.cuda.max_memory_allocated()/G, torch.cuda.max_memory_reserved()/G); break
    tg = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 2 else 0
    print(json.dumps({"p": p, "pp": a.depth / ttft, "tg": tg, "peak": torch.cuda.max_memory_allocated()/G,
                      "resv": torch.cuda.max_memory_reserved()/G, "cur_alloc": torch.cuda.memory_allocated()/G}), flush=True)
    res.append(txt)
if a.out: json.dump(res, open(a.out, "w"))

stop = True
if best["snap"]:
    c = collections.Counter()
    for seg in best["snap"]:
        for b in seg["blocks"]:
            if b["state"] == "active_allocated": c[b["size"]] += 1
    print("snapshot at ~peak %.3f GiB; active blocks by total size:" % (best["peak"]/G))
    for sz, n in sorted(c.items(), key=lambda x: -x[0]*x[1])[:25]:
        print(f"  {n:5d} x {sz/2**20:10.2f} MiB = {n*sz/G:7.3f} GiB")
    print("segments:", len(best["snap"]), "reserved", sum(s["total_size"] for s in best["snap"])/G)
