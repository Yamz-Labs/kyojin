"""ab_exp.py: ONE load (HY4_MOE_V2=1 in the environment, code object exllamav3_ext/hip_kq/moe_dec.co built by hip_kq/build_moe.sh); cwd = tree.
Paired ABBA graph-replay step wall, T = 1..6, ctx 1000 / 4000 (PCTXS, PTS): arm A = shipped grouped kernel (fq_moe_kq_grouped_mixed_q8), arm B = HY4_MOE_V2 stages.
Prints head_ab.log style lines (ARM ... done, AB ctx T: A [...] B [...] ms delta, KL_mean top1_agree: the arms are bit-equal where the code object is bit-equal, so KL 0 is expected;
a nonzero KL is a kernel bug, not noise). cwd = tree. Env: GGUF (model path), WIKI (json list of token ids)."""
import json, os, statistics, sys, time
tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve
serve.apply_env(16384)
import torch
from engine import Hy4Engine
GGUF = os.environ["GGUF"]
e = Hy4Engine(tree, GGUF, ctx=16384, k=3)
e.load(print)
GR, M = e.GR, e.M
fwm = sys.modules[type(M).__module__]
assert fwm.MOE_V2, "run with HY4_MOE_V2=1"
wiki = json.load(open(os.environ["WIKI"]))
pc, sync = time.perf_counter, torch.cuda.synchronize


def kl_top1(la, lb):
    la, lb = la.float().log_softmax(-1), lb.float().log_softmax(-1)
    return (la.exp() * (la - lb)).sum(-1), (la.argmax(-1) == lb.argmax(-1)).float()


def arm(b):
    fwm.MOE_V2 = b
    fwm._MD.clear() if not b else None      # arm B rebuilds its scratch lazily
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
        print(f"ARM {a_} ctx {ctxn} done (MOE_V2 {fwm.MOE_V2}, v2 layers {fwm._MD[1].keys() if len(fwm._MD) > 1 else None})", flush=True)
arm(False)
for ctxn in ctxs:
    for T in Ts:
        a, b = res[(ctxn, T, "A")], res[(ctxn, T, "B")]
        k_, s_ = kl_top1(logit[(ctxn, T, "A")], logit[(ctxn, T, "B")])
        print(f"AB ctx {ctxn} T {T}: A {[round(x, 2) for x in a]} B {[round(x, 2) for x in b]} ms  delta {statistics.mean(b) - statistics.mean(a):+.2f} ms  "
              f"KL_mean {k_.mean().item():.2e} top1_agree {s_.mean().item():.2f}", flush=True)
print("ABEXP_DONE", flush=True)
