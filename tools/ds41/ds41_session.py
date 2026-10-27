"""Multi-turn session through PortEngine (the engine behind serve.py) with the prompt cache on: per-turn TTFT, cached tokens,
tier misses/bytes of the prefill part and of the decode part, DSpark decode tok/s. Turn kinds: ctx (first, big), follow (30-80 tok),
tool (~2K tok). Texts are real token rows (data/short.npy, long.npy), appended to the running id list."""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
H = os.path.expanduser("~/ds41_work")
ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=16384); ap.add_argument("--new", type=int, default=96); ap.add_argument("--draft", default="4,0", help="comma list of draft depths, one engine each (0 = plain)")
ap.add_argument("--first", type=int, default=6144); ap.add_argument("--plan", default="f50,f30,t2048,f80,f40,t2000,f60,t2048,f35")
ap.add_argument("--engram-dir"); ap.add_argument("--data", default=os.path.join(H, "ds41ready_data"))
a = ap.parse_args()
from ds41_serving import build_model
from ds41_engine import HFTokenizer
from ds41_model import PortEngine, TopPSampler
m = build_model(a.pack, device="cuda:0", dtype="float16", layers=None, mtp=True, moe="auto", prefill_chunk=a.chunk, engram_dir=a.engram_dir)
blk = [b for b in m.pack_modules[0].modules if getattr(b, "tier", None) is not None]
from exllamav3.modules.expert_store import get_store
store = get_store(blk[0].config)
s_, l_ = (np.load(os.path.join(a.data, f"{n}.npy")).astype(np.int64) for n in ("short", "long"))
pool = np.concatenate([s_.reshape(-1), l_.reshape(-1)]); off = 0
def take(n):
    global off
    x = pool[off:off + n].tolist(); off += n; return x
def S(): return store.stats()
KEYS = ("uses", "hits", "misses", "bytes", "read_s", "wait_s")
def diff(x, y): return {k: y[k] - x[k] for k in KEYS}
for draft in [int(x) for x in a.draft.split(",")]:
    eng = PortEngine(m, HFTokenizer(os.path.join(a.pack, "tokenizer.json")), ctx=32768, num_draft=draft, eos_id=None, prefix_cache=True)
    off = 0
    ids = take(a.first); rows = []
    plan = ["c%d" % a.first] + a.plan.split(",")
    for i, p in enumerate(plan):
        if i > 0: ids = ids + take(int(p[1:]))
        snap = {}; marks = {}
        def cb(t):
            if "first" not in marks: marks["first"] = time.perf_counter(); snap.update(S())
        s0 = S(); t0 = time.perf_counter()
        outt = []
        def cb2(t):
            outt.extend(t); cb(t)
        st = eng._run(list(ids), a.new, TopPSampler(0.0, 1.0, None), -1, None, cb2)
        t1 = time.perf_counter(); s1 = S()
        pre, dec = diff(s0, snap), diff(snap, s1)
        nt = st["new_tokens"]
        r = {"turn": i, "kind": p, "prompt": len(ids), "cached": st.get("cached_tokens", 0), "ttft_s": round(marks["first"] - t0, 2),
             "prefix_lcp": st.get("prefix_lcp"), "prefill_misses": pre["misses"], "prefill_GB": round(pre["bytes"] / 1e9, 1), "prefill_hit": round(pre["hits"] / max(pre["uses"], 1), 3),
             "dec_tok_s": round((nt - 1) / max(t1 - marks["first"], 1e-9), 2), "dec_hit": round(dec["hits"] / max(dec["uses"], 1), 4),
             "dec_misses": dec["misses"], "dec_GB": round(dec["bytes"] / 1e9, 2), "acc_per_step": st.get("accepted_per_step")}
        print(a.arm, draft, r, flush=True); rows.append(r)
        if i % 2 == 1: ids = ids + outt[:-1]      # verbatim reply goes back (agent loop); other turns drop it
    open(a.out, "a").write(json.dumps({"arm": a.arm + "_d%d" % draft, "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_EXPERT", "EXL3_TIER"))}, "turns": rows}) + "\n")
    