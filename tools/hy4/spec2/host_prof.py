"""host_prof.py: where the host time of a speculative round goes (box, ONE load, run inside job.sh). Wraps the decoder's host steps with timers and prints, per run,
captures (lazy graph builds inside the run) and per-round medians / p90 / max of: draft launch+wait, step_rows total, stage parts (hash, submit, pread wait, H2D), replay launch,
and the round wall time (gap between on_tokens calls)."""
import hashlib, json, os, statistics, sys, time
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
GR, MG = e.GR, e.MG
V = C.Vocab(e.meta)
wiki = json.load(open("/path/to/wiki_ids.json"))
pc = time.perf_counter
REC = {}
def rec(k, v):
    REC.setdefault(k, []).append(v)
def wrap(obj, name, key):
    f = getattr(obj, name)
    def w(*a, **kw):
        t = pc()
        try:
            return f(*a, **kw)
        finally:
            rec(key, (pc() - t) * 1000)
    setattr(obj, name, w)
wrap(GR, "step_rows", "step_rows")
wrap(GR, "stage", "stage")
wrap(GR, "_jobs", "hash")
wrap(GR, "_submit", "submit")
wrap(GR.stager, "hash", "hash")
wrap(GR.stager, "submit", "submit")
wrap(MG, "run", "draft_run")
wrap(MG, "_graph", "dgcall")
import ngram as _ng
wrap(_ng.NGram, "propose", "ng_propose")
wrap(_ng.NGram, "add", "ng_add")
_gk = GR._graph
CAPKEYS = []
def _gk2(Nb, T=1):
    new = (Nb, T) not in GR.graphs
    t = pc(); r = _gk(Nb, T)
    if new: CAPKEYS.append((Nb, T, round((pc() - t) * 1000)))
    return r
GR._graph = _gk2
def rbytes():
    for l in open("/proc/self/io"):
        if l.startswith("read_bytes"): return int(l.split()[1])
ev0 = GR.ev
ncap0 = lambda: GR.stats["captures"]
WT = V.decode(wiki[3000:3300]); WT2 = V.decode(wiki[9000:9300])
CODE = open(os.path.join(tree, "tools/hy4/serve_core.py")).read()[:2600]; CODE2 = open(os.path.join(tree, "tools/hy4/engine.py")).read()[1200:3800]
chat = lambda t: V.encode(V.template.render([{"role": "user", "content": t}], None, True, reasoning_effort="no_think"))
PA = {"quote": chat("Here is a passage:\n\n" + WT + "\n\nQuote the three most important sentences verbatim, then summarize in two sentences."),
      "code": chat("Add a short docstring to every function in this Python file and return the full file.\n\n" + CODE),
      "chat": chat("Explain how a hash table handles collisions, with a short Python example and the cost of each operation.")}
PB = {"quote": chat("Here is a passage:\n\n" + WT2 + "\n\nQuote the two most surprising sentences verbatim, then summarize in one sentence."),
      "code": chat("Rename every local variable of this Python file to a longer descriptive name and return the full file.\n\n" + CODE2),
      "chat": chat("Describe how a B-tree keeps itself balanced during insertions, with a short Python sketch.")}
N = int(os.environ.get("BN", "120"))
STG = GR.stager
def set_arm(a):
    if a == "old":
        e.pipe, GR.stager, e.MH._STAGE_OV[0] = False, None, False
    else:
        e.pipe, GR.stager, e.MH._STAGE_OV[0] = True, STG, True
def q(v, p):
    s = sorted(v); return s[min(len(s) - 1, int(p * len(s)))]
def summ(v):
    return "n=%d med %.2f p90 %.2f max %.2f" % (len(v), statistics.median(v), q(v, 0.9), max(v)) if v else "none"
def run(tag, arm, pn, ids):
    set_arm(arm)
    REC.clear(); CAPKEYS.clear(); rb0 = rbytes()
    stamps, out = [], []
    def cb(t):
        stamps.append(pc()); out.extend(t); return False
    st = e.generate(ids, N, cb)
    gaps = [(b - a) * 1000 for a, b in zip(stamps[1:], stamps[2:])]
    thr = 150.0
    print(f"RUN {tag} arm={arm} {pn} tok_s {st['decode_tokens'] / st['decode_s']:.2f} tau {st['tau']:.3f} rounds {st['rounds']} md5 {hashlib.md5(str(out).encode()).hexdigest()[:8]} round_ms {[round(x, 1) for x in st['round_ms']]} gap_ms {[round(x, 2) for x in st.get('gap_ms', [])]} caps {st.get('captures')} cap_s {st.get('capture_s', 0):.1f}", flush=True)
    print(f"   read_bytes {(rbytes() - rb0) / 1e6:.1f} MB; steady step_rows {summ([x for x in REC['step_rows'] if x < thr])}; stage {summ([x for x in REC.get('stage', []) if x < thr])}; hash {summ(REC.get('hash', []))}; submit {summ(REC.get('submit', []))}; ng_propose {summ(REC.get('ng_propose', []))}; ng_add {summ(REC.get('ng_add', []))}; round wall {summ([g for g in gaps if g < 450])}; capture keys {CAPKEYS}", flush=True)
    return out
for pn in PA:
    seq = [("A", "old", PA), ("A", "old", PA), ("A", "new", PA), ("A", "new", PA), ("B", "new", PB), ("B", "new", PB), ("B", "old", PB), ("B", "old", PB)]
    md = {}
    for i, (t, arm, P) in enumerate(seq):
        out = run(f"{t}{i}", arm, pn, P[pn])
        md.setdefault((t, arm), []).append(hashlib.md5(str(out).encode()).hexdigest()[:8])
    print("MD5", pn, {f"{k[0]}-{k[1]}": v for k, v in md.items()}, flush=True)
print("HP_DONE", flush=True)
