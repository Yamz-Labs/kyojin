"""ab2.py: ONE load. Phase 0 exactness + bounds (graph == eager, split == unsplit, bit for bit, T = 1..8, two context buckets), phase 1 precapture cost / memory,
phase 2 A/B of the host-time levers on 5 texts (old = shipped host path, nosplit = vectorised staging + pipelined draft + prestage, new = + 3-segment graph), equal output md5
required. Run inside job.sh (box rules)."""
import hashlib, json, os, statistics, sys, time
import numpy as np
tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve
serve.apply_env(16384)
import torch
from engine import Hy4Engine
import serve_core as C
GGUF = "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf"
e = Hy4Engine(tree, GGUF, ctx=16384, k=3)
e.load(print)
GR, MG, fw = e.GR, e.MG, e.GR.fw
STG = GR.stager
V = C.Vocab(e.meta)
wiki = json.load(open("/path/to/wiki_ids.json"))
pc = time.perf_counter
sync = torch.cuda.synchronize
free = lambda: torch.cuda.mem_get_info()[0] / 2 ** 30
def set_arm(a):
    if a == "old":
        e.pipe, GR.stager, e.MH._STAGE_OV[0], GR.split = False, None, False, False
    elif a == "nosplit":
        e.pipe, GR.stager, e.MH._STAGE_OV[0], GR.split = True, STG, True, False
    else:
        e.pipe, GR.stager, e.MH._STAGE_OV[0], GR.split = True, STG, True, True

# ---------------------------------------------------------------- phase 0: exactness
def eager_rows(toks):
    """step_rows without the graph: same staging, same static inputs, the body run eagerly"""
    T, start = len(toks), fw.n
    if GR.ev_set:
        GR.ev.synchronize()
    fw.tokens.extend(int(t) for t in toks)
    GR.gs.Nb, GR.gs.T = GR.bucket(start, T), T
    GR.hnp[0, :T] = toks
    GR.hnp[1, :T] = np.arange(start, start + T)
    GR.gs.inp.copy_(GR.hin, non_blocking=True)
    GR.stage(start, T)
    fw._gs = GR.gs
    try:
        out = GR.body(GR.bounds())
    finally:
        fw._gs = None
    fw.n += T
    GR.expect_n = fw.n
    return out
bad = 0
for ctxn in (200, 700):
    e._t0 = pc(); e.reset(); e.prefill(wiki[:ctxn]); e.phase("decode")
    for T in (1, 2, 3, 4, 5, 8):
        toks = wiki[ctxn:ctxn + T]
        res = {}
        for name in ("split", "nosplit", "eager"):
            set_arm("new" if name != "nosplit" else "nosplit")
            if name == "eager":
                out = eager_rows(toks)
            else:
                out = GR.step_rows(toks)
            sync()
            res[name] = [o.clone() for o in out[:3]]
            GR.rewind(ctxn)
        for name in ("nosplit", "eager"):
            ok = all(torch.equal(a, b) for a, b in zip(res["split"], res[name]))
            if not ok and name == "nosplit":
                bad += 1
            print(f"EXACT ctx {ctxn} T {T} split vs {name}: {'bit-exact' if ok else 'MISMATCH'}", flush=True)
if bad:
    print("EXACTNESS FAILED", bad, flush=True); sys.exit(2)
print("EXACT_OK", flush=True)

# ---------------------------------------------------------------- phase 1: precapture cost + memory
NBS = [512, 1024, 1536]
m0, t0 = free(), pc()
built = []
for mode in ("new", "nosplit"):
    set_arm(mode)
    built += GR.precapture(nbs=NBS)
nd = MG.precapture(NBS)
sync()
print(f"PRECAP {len(built)} verify graphs (both modes) + {nd} draft graphs: {pc() - t0:.1f} s, {m0 - free():.2f} GiB; per verify graph {statistics.mean(x[2] for x in built):.0f} ms; "
      f"full cap = {len(range(512, GR.cap + 1, 512))} buckets x 8 T x 1 mode ~ {len(range(512, GR.cap + 1, 512)) * 8 * statistics.mean(x[2] for x in built) / 1000:.0f} s, {(m0 - free()) / len(built) * len(range(512, GR.cap + 1, 512)) * 8 / 2:.1f} GiB", flush=True)

# ---------------------------------------------------------------- phase 2: A/B
WT = V.decode(wiki[3000:3300]); WT2 = V.decode(wiki[9000:9300])
CODE = open(os.path.join(tree, "tools/hy4/serve_core.py")).read()[:2600]
chat = lambda t: V.encode(V.template.render([{"role": "user", "content": t}], None, True, reasoning_effort="no_think"))
texts = {"quote": chat("Here is a passage:\n\n" + WT + "\n\nQuote the three most important sentences verbatim, then summarize in two sentences."),
         "code": chat("Add a short docstring to every function in this Python file and return the full file.\n\n" + CODE),
         "chat": chat("Explain how a hash table handles collisions, with a short Python example and the cost of each operation."),
         "quote2": chat("Here is a passage:\n\n" + WT2 + "\n\nQuote the two most surprising sentences verbatim, then summarize in one sentence."),
         "chat2": chat("Describe how a B-tree keeps itself balanced during insertions, with a short Python sketch.")}
N = int(os.environ.get("BN", "120"))
res = {}
for i, (pn, ids) in enumerate(texts.items()):
    seq = ["old", "new", "new", "old", "nosplit", "nosplit"] if i % 2 == 0 else ["new", "old", "old", "new", "nosplit", "nosplit"]
    first = None
    for k, arm in enumerate(seq):
        set_arm(arm)
        out = []
        st = e.generate(ids, N, lambda t: out.extend(t) or False)
        md5 = hashlib.md5(str(out).encode()).hexdigest()[:8]
        first = first or md5
        r = dict(text=pn, run=k, arm=arm, cold=(k == 0), tok_s=round(st["decode_tokens"] / st["decode_s"], 2), tau=round(st["tau"], 3), rounds=st["rounds"], md5=md5, same=md5 == first,
                 round_ms=[round(x, 1) for x in st["round_ms"]], gap_ms=[round(x, 2) for x in st["gap_ms"]], busy_ms=round(st["busy_ms"], 2), caps=st["captures"])
        res.setdefault(arm, []).append(r)
        print("RES", json.dumps(r), flush=True)
for arm in ("old", "nosplit", "new"):
    rs = res[arm]
    f = lambda sel, key: round(statistics.mean(sel(x)[key] if isinstance(key, str) else sel(x)[key[0]][key[1]] for x in rs if sel(x) is not None), 2) if rs else None
    for lab, sub in (("all", rs), ("cold", [x for x in rs if x["cold"]]), ("warm", [x for x in rs if not x["cold"]])):
        if sub:
            print("MEAN", arm, lab, "tok_s", round(statistics.mean(x["tok_s"] for x in sub), 2), "gap_ms", [round(statistics.mean(x["gap_ms"][j] for x in sub), 2) for j in (0, 1)], "busy_ms",
                  round(statistics.mean(x["busy_ms"] for x in sub), 2), "verify_wait_ms", round(statistics.mean(x["round_ms"][2] for x in sub), 1), flush=True)
print("md5 all equal:", all(x["same"] for a in res.values() for x in a), flush=True)
print("AB2_DONE", flush=True)
