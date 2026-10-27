"""Short tier/prefill A/B arm: one process, one model load. Prefill at --lens (fresh state each, stats delta per length),
then DSpark decode of --new tokens after a warm-up (prefix cache on, so decode-only tier stats), plus plain-greedy equality
check by token hash. Output: JSON line per arm appended to --out.
  env: the usual EXL3_EXPERT_* / EXL3_TIER_PF_CHUNK knobs (see ds41_env.sh)."""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
H = os.path.expanduser("~/ds41_work")
ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=4096); ap.add_argument("--lens", default="2048,8192,16384")
ap.add_argument("--new", type=int, default=512); ap.add_argument("--draft", type=int, default=4)
ap.add_argument("--profile", type=int, default=0, help="profile one prefill of this many tokens (torch profiler)")
ap.add_argument("--engram-dir"); ap.add_argument("--data", default=os.path.join(H, "ds41ready_data"))
a = ap.parse_args()
from ds41_serving import build_model
from ds41_engine import HFTokenizer
from ds41_model import PortEngine, TopPSampler
def mem(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1048576
t0 = time.time()
m = build_model(a.pack, device="cuda:0", dtype="float16", layers=None, mtp=True, moe="auto", prefill_chunk=a.chunk, engram_dir=a.engram_dir)
torch.cuda.synchronize(); res = {"arm": a.arm, "chunk": a.chunk, "load_s": round(time.time() - t0, 1), "memavail_after_load": round(mem(), 1),
    "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_EXPERT", "EXL3_TIER"))}}
store = None
blk = [b for b in m.pack_modules[0].modules if getattr(b, "tier", None) is not None]
if blk:
    from exllamav3.modules.expert_store import get_store
    store = get_store(blk[0].config)
import subprocess
def nvme_c():
    try:
        o = subprocess.run("sudo -n nvme smart-log /dev/nvme0", shell=True, capture_output=True, text=True, timeout=10).stdout
        return [int(l.split(":")[1].split()[0]) for l in o.splitlines() if l.startswith(("Temperature Sensor 1", "Temperature Sensor 2"))]
    except Exception:
        return []
def drop_cache():
    subprocess.run("sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null", shell=True, timeout=60)
def st_delta(f):
    if store is None: f(); return {}
    store.reset_stats(); f(); s = store.stats()
    return {k: (round(s[k], 3) if isinstance(s[k], float) else s[k]) for k in ("uses", "hits", "misses", "hit_rate", "bytes", "read_s", "wait_s", "gbps", "gbps_wall", "wall_s", "calls", "miss_calls", "sync_s", "prep_s", "ev_s", "copy_s", "ahead_issued", "ahead_used")} | {"wait_L": {k: round(v, 2) for k, v in s["wait_by_layer"].items() if k in (0, 1, 2, 3, 39)}}
s_, l_ = (np.load(os.path.join(a.data, f"{n}.npy")).astype(np.int64) for n in ("short", "long"))
res["prefill"] = {}; minmem = mem()
for L in [int(x) for x in a.lens.split(",")]:
    row = (np.concatenate(list(l_)) if L > 16384 else (l_[0] if L > 2048 else s_[0]))[:L].tolist(); ids = torch.tensor([row], device="cuda:0")
    for rep in range(2):
        t_before = nvme_c(); st = m.new_state(1, L + 32); torch.cuda.synchronize(); t = time.perf_counter()
        d = st_delta(lambda: (m.prefill(ids, st, 0), torch.cuda.synchronize()))
        dt = time.perf_counter() - t; del st
        res["prefill"][f"{L}_{'cold' if rep == 0 else 'warm'}"] = {"tok_s": round(L / dt, 1), "s": round(dt, 2), "nvme_c": [t_before, nvme_c()], **d}
        print(a.arm, L, rep, res["prefill"][f"{L}_{'cold' if rep == 0 else 'warm'}"], flush=True)
        minmem = min(minmem, mem())
if a.profile:
    L = a.profile; ids_p = torch.tensor([(l_[0])[:L].tolist()], device="cuda:0"); st = m.new_state(1, L + 32)
    m.prefill(ids_p, m.new_state(1, L + 32), 0); torch.cuda.synchronize()
    if store: store.reset_stats()
    t = time.perf_counter()
    t_ns0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    m.prefill(ids_p, st, 0); torch.cuda.synchronize()
    print("MARK", t_ns0, time.clock_gettime_ns(time.CLOCK_MONOTONIC), flush=True)
    wall = time.perf_counter() - t
    res["profile_wall_s"] = round(wall, 2)
    if store: res["profile_store"] = {k: v for k, v in store.stats().items() if not isinstance(v, dict)}
    if store: res["wait_by_layer"] = {str(k): round(v, 3) for k, v in store.stats()["wait_by_layer"].items()}; print("WAITBYLAYER", res["wait_by_layer"], flush=True)
    print("PROFILE wall", wall, res.get("profile_store"), flush=True)
    del st
tok = HFTokenizer(os.path.join(a.pack, "tokenizer.json"))
eng = PortEngine(m, tok, ctx=24576, num_draft=a.draft, eos_id=None, prefix_cache=True)
ids = s_[0].tolist()
snap = {}
def gen(n):
    out = []; snap.clear()
    def cb(t):
        if not snap and store is not None: snap.update(store.stats())
        out.extend(t)
    st = eng._run(ids, n, TopPSampler(0.0, 1.0, None), -1, None, cb); return st, out
gen(16)
holder = {}
def go(): holder["st"], holder["out"] = gen(a.new)
d = st_delta(go); st = holder["st"]
if snap:
    fin = store.stats(); dd = {k: fin[k] - snap[k] for k in ("uses", "hits", "misses", "bytes", "read_s", "wait_s", "wall_s", "calls")}
    d["decode_only"] = {**{k: round(v, 3) for k, v in dd.items()}, "hit_rate": round(dd["hits"] / max(dd["uses"], 1), 4),
                        "gbps": round(dd["bytes"] / max(dd["read_s"], 1e-9) / 1e9, 2)}
res["decode"] = {"tok_s": round((st["new_tokens"] - 1) / st["time_generate"], 2), "prefill_s": round(st["time_prefill"], 3),
    "new": st["new_tokens"], "tokhash": hashlib.md5(str(holder["out"]).encode()).hexdigest()[:12], **d,
    "stats": {k: v for k, v in st.items() if k not in ("time_prefill", "time_generate")}}
res["min_memavail"] = round(min(minmem, mem()), 1)
print(a.arm, "decode", res["decode"], flush=True)
open(a.out, "a").write(json.dumps(res, default=float) + "\n")
