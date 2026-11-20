"""ab_mla.py: ONE load (cwd = tree; HY4_KV_FP8 etc. from the job env). Graph-replay verify-step wall at T = 1..6, ctx PCTXS, levers off (A) vs on (B), ABBA, graphs re-captured at every arm
switch; logits A vs B on the same tokens (KL(A||B), top-1). Levers (env ARMB, comma list): lchb = fp8 sparse attend head-blocked + split-K (lc_sparse.HB_ON, HB_SPLIT auto),
rows = HY4_MLA_HB_ROWS rows kernel (BH 16, RG 4). ARMB empty = A vs A (reference timing of the loaded flags). env: PCTXS (1000,4000,32000), PTS (1,2,3,4,5,6)."""
import json, os, statistics, sys, time
tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve
ctxs = [int(x) for x in os.environ.get("PCTXS", "1000,4000,32000").split(",")]
CAP = max(ctxs) + 2048
serve.apply_env(CAP)
import torch
from engine import Hy4Engine
GGUF = os.environ.get("GGUF", "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf")
e = Hy4Engine(tree, GGUF, ctx=CAP, k=3)
e.load(print)
GR = e.GR
fwm = sys.modules[type(GR.fw).__module__]
rf = fwm._rf
from exllamav3.loader import lc_sparse as LS
def lchb(on):
    LS.HB_ON, LS.HB_SPLIT = bool(on), -1
def rows(on):
    rf.HB_ROWS, rf.HB_ROWS_BH, rf.HB_ROWS_RG, rf.HB_ROWS_NW = (8, 16, 4, 8) if on else (0, 16, 4, 8)
LEV = {"lchb": lchb, "rows": rows}
FL = [x for x in os.environ.get("ARMB", "lchb").split(",") if x]
def arm(fl):
    for k, f in LEV.items():
        f(k in fl)
    GR.clear()
wiki = json.load(open(os.environ.get("WIKI", "/path/to/wiki_ids.json")))
pc, sync = time.perf_counter, torch.cuda.synchronize
print("ARMB", FL, "KV_FP8", os.environ.get("HY4_KV_FP8"), "LC_SPARSE", LS.ENABLED, "HB", rf.HB, flush=True)
Ts = [int(x) for x in os.environ.get("PTS", "1,2,3,4,5,6").split(",")]
res, logit = {}, {}
for ctxn in ctxs:
    e._t0 = pc(); e.reset(); e.prefill(wiki[:ctxn]); e.phase("decode")
    for a_ in ("A", "B", "B", "A"):
        arm(FL if a_ == "B" else [])
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
    for T in Ts:
        a, b = res[(ctxn, T, "A")], res[(ctxn, T, "B")]
        la, lb = logit[(ctxn, T, "A")].log_softmax(-1), logit[(ctxn, T, "B")].log_softmax(-1)
        kl = (la.exp() * (la - lb)).sum(-1)
        agree = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
        print(f"AB ctx {ctxn} T {T}: A {a} B {b} ms  delta {statistics.mean(b) - statistics.mean(a):+.2f} ms  KL_mean {kl.mean().item():.2e} KL_max {kl.max().item():.2e} top1_agree {agree:.2f}", flush=True)
print("ABMLA_DONE", flush=True)
