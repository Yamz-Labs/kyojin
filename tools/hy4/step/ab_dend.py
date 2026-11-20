"""ab_dend.py: ONE load; paired ABBA graph-replay verify-round wall, T = 1..6, ctx 1000 / 4000, arm A = stock dense Q8_0 GEMV, arm B = HY4_DENSE_V2 (gm_ext gm_dense_q8 for 2..16 rows, K <= 8192).
cwd = tree. Env: EXL3_GM_BUILD_DIR (the gm_ext build holding gm_dense_q8), PCTXS, PTS. Arm flip = fwm.DENSE_V2[0] + GR.clear() (graphs re-captured). Prints per-T ms for both arms, the logit equality (bit-identical expected) and the dense launch count."""
import json, os, statistics, sys, time
tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve
serve.apply_env(16384)
import torch
from engine import Hy4Engine
GGUF = os.environ.get("GGUF", "/home/kamel/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf")
e = Hy4Engine(tree, GGUF, ctx=16384, k=3)
e.load(print)
GR, M = e.GR, e.M
fwm = sys.modules[type(M).__module__]
g = fwm._gm()
print("gm_ext", None if g is None else (g.__file__, hasattr(g, "gm_dense_q8")), flush=True)
assert g is not None and hasattr(g, "gm_dense_q8"), "gm_dense_q8 missing: set EXL3_GM_BUILD_DIR to the build holding it"
wiki = json.load(open(os.environ.get("WIKI", "/home/kamel/scratch_w5/wiki_ids.json")))
pc, sync = time.perf_counter, torch.cuda.synchronize


def arm(b):
    fwm.DENSE_V2[0] = b
    GR.clear()


ctxs = [int(x) for x in os.environ.get("PCTXS", "1000,4000").split(",")]
Ts = [int(x) for x in os.environ.get("PTS", "1,2,3,4,5,6").split(",")]
res, logit = {}, {}
for ctxn in ctxs:
    e._t0 = pc(); e.reset(); e.prefill(wiki[:ctxn]); e.phase("decode")
    for a_ in ("A", "B", "B", "A"):
        arm(a_ == "B")
        for T in Ts:
            toks = wiki[ctxn:ctxn + T]
            for _ in range(3):
                GR.step_rows(toks); sync(); GR.rewind(ctxn)
            ts = []
            for _ in range(20):
                t = pc(); out = GR.step_rows(toks); sync(); ts.append((pc() - t) * 1000); GR.rewind(ctxn)
            res.setdefault((ctxn, T, a_), []).append(statistics.median(ts))
            if (ctxn, T, a_) not in logit:
                logit[(ctxn, T, a_)] = out[0].float().reshape(T, -1).clone()
        print(f"ARM {a_} ctx {ctxn} done (DENSE_V2 {fwm.DENSE_V2[0]})", flush=True)
arm(False)
for ctxn in ctxs:
    for T in Ts:
        a, b = res[(ctxn, T, "A")], res[(ctxn, T, "B")]
        eq = bool(torch.equal(logit[(ctxn, T, "A")], logit[(ctxn, T, "B")]))
        print(f"AB ctx {ctxn} T {T}: A {[round(x, 2) for x in a]} B {[round(x, 2) for x in b]} ms  delta {statistics.mean(b) - statistics.mean(a):+.2f} ms  logits_bit_equal {eq}", flush=True)
print("ABDEND_DONE", flush=True)
