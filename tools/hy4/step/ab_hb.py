"""ab_hb.py: ONE load. Graph-replay step wall at T = 1 / 2 / 6, ctx 1000 / 4000, HY4_MLA_HB off (A) vs on (B), ABBA, graphs re-captured at every arm switch;
logits of A vs B on the same tokens: KL(A||B), top-1 agreement. Run inside job.sh (box rules)."""
import json, os, statistics, sys, time
import numpy as np
tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve
serve.apply_env(16384)
import torch
from engine import Hy4Engine
from exllamav3.loader import ref_fused as rf
GGUF = "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf"
e = Hy4Engine(tree, GGUF, ctx=16384, k=3)
e.load(print)
GR = e.GR
wiki = json.load(open("/path/to/wiki_ids.json"))
pc = time.perf_counter
sync = torch.cuda.synchronize
print("REF_FUSED", rf.ENABLED, "HB_BN", rf.HB_BN, "HB_NW", rf.HB_NW, flush=True)
ctxs = [int(x) for x in os.environ.get("PCTXS", "1000,4000").split(",")]
Ts = [int(x) for x in os.environ.get("PTS", "1,2,6").split(",")]
res = {}
logit = {}
for ctxn in ctxs:
    e._t0 = pc(); e.reset(); e.prefill(wiki[:ctxn]); e.phase("decode")
    for arm in ("A", "B", "B", "A"):
        rf.HB = arm == "B"
        GR.graphs.clear()
        for T in Ts:
            toks = wiki[ctxn:ctxn + T]
            for _ in range(3):
                GR.step_rows(toks); sync(); GR.rewind(ctxn)
            ts = []
            for _ in range(20):
                t = pc(); out = GR.step_rows(toks); sync(); ts.append((pc() - t) * 1000); GR.rewind(ctxn)
            res.setdefault((ctxn, T, arm), []).append(statistics.median(ts))
            if (ctxn, T, arm) not in logit:
                logit[(ctxn, T, arm)] = out[0].float().reshape(T, -1).clone()
for ctxn in ctxs:
    for T in Ts:
        a, b = res[(ctxn, T, "A")], res[(ctxn, T, "B")]
        la, lb = logit[(ctxn, T, "A")].log_softmax(-1), logit[(ctxn, T, "B")].log_softmax(-1)
        kl = (la.exp() * (la - lb)).sum(-1)
        agree = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
        print(f"AB ctx {ctxn} T {T}: A {a} B {b} ms  delta {statistics.mean(b) - statistics.mean(a):+.2f} ms  KL_mean {kl.mean().item():.2e} KL_max {kl.max().item():.2e} top1_agree {agree:.2f}", flush=True)
print("ABHB_DONE", flush=True)
