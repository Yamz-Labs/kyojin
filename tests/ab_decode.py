"""ab_decode.py PACK SETTINGS.json [--depth 4096] [--kv 16|K,V] [--win 48] [--reps 6]
In-process A/B of decode-time environment knobs that the C++ extension reads on every call (getenv): one prefill, then one long greedy job whose
tokens are cut into windows; window i runs under settings[i % len(settings)] (a list of {"name":..., "env": {VAR: value or null}}), reps windows each.
Alternating inside one process cancels the box-wide drift that makes separate runs differ by +-10 %. Prints per-setting median/min/max tok/s."""
import argparse, json, os, sys, time, statistics
ROOT = os.environ.get("ENG") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.presets import GreedySampler
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("settings"); ap.add_argument("--depth", type=int, default=4096)
ap.add_argument("--kv", default="16"); ap.add_argument("--win", type=int, default=48); ap.add_argument("--reps", type=int, default=6)
a = ap.parse_args()
settings = json.load(open(a.settings))
cfg = Config.from_directory(a.pack); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
total = a.win * len(settings) * a.reps + 2 * a.win
ctx = ((a.depth + total + 512 + 255) // 256) * 256
if a.kv == "16": cache = Cache(model, max_num_tokens=ctx)
else:
    kb, vb = (int(x) for x in a.kv.split(",")); cache = Cache(model, max_num_tokens=ctx, layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
model.load(progressbar=False)
gen = Generator(model=model, cache=cache, tokenizer=tok)
D = os.path.join(ROOT, "exllamav3/conversion/standard_cal_data")
s = open(f"{D}/wiki.utf8", encoding="utf8").read()
ids = tok.encode(s[3000:3000 + a.depth * 8], add_bos=False).flatten()[:a.depth].view(1, -1)
job = Job(input_ids=ids, max_new_tokens=total, sampler=GreedySampler(), stop_conditions=[])
gen.enqueue(job)
base_env = {}
for st in settings:
    for k in st["env"]: base_env.setdefault(k, os.environ.get(k))
def apply(st):
    for k, v in base_env.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    for k, v in st["env"].items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = str(v)
n = -a.win; cur = None; stamps = []; results = {st["name"]: [] for st in settings}; widx = -1
apply(settings[0])
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if r.get("text", "") == "": continue
        n += 1
        if n < 0: continue                      # warm-up window
        if n % a.win == 0:
            now = time.time()
            if widx >= 0 and stamps: results[settings[widx % len(settings)]["name"]].append(a.win / (now - stamps[-1]))
            widx += 1; stamps.append(now)
            if widx < len(settings) * a.reps: apply(settings[widx % len(settings)])
            else: gen.cancel(job) if hasattr(gen, "cancel") else None
print(f"depth {a.depth} kv {a.kv} window {a.win} tokens x {a.reps} reps per setting")
b = statistics.median(results[settings[0]["name"]]) if results[settings[0]["name"]] else 1
for st in settings:
    v = results[st["name"]]
    if v: print(f"  {st['name']:28s} median {statistics.median(v):6.2f} tok/s  ({statistics.median(v) / b * 100 - 100:+5.1f} % vs {settings[0]['name']})  min {min(v):.2f} max {max(v):.2f}  n={len(v)}")
