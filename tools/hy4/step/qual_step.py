"""qual_step.py: quality gate of the W5 step levers in ONE load (run inside job.sh, cwd = tree). Protocol of req_arms.py: teacher-forced wiki windows (ctx 512, first half context, second half scored, R=4 rows per step).
Arms: ref (all levers off), 3 noise nulls (1e-6 embedding noise, seeds 1234 / 77 / 5), then each lever alone and all together. Per arm: NLL, paired delta NLL vs ref, KL(ref||arm), top-1 agreement; KL / noise null."""
import json, os, sys, time
import numpy as np
tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve
serve.apply_env(16384)
import torch
from engine import Hy4Engine
GGUF = "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf"
e = Hy4Engine(tree, GGUF, ctx=16384, k=3)
e.load(print)
SG, GR = e.SG, e.GR
fwm = sys.modules[type(GR.fw).__module__]
rf, hbm, bgm = fwm._rf, fwm._HB, fwm._BGM
KVF = fwm._KVF
dev = e.dev
rg_orig = {}
def levers(fl):
    KVF.MAXT = 16 if "kvr" in fl else 1
    os.environ["HY4_ENGRAM_FUSED"] = "1" if "eng" in fl else "0"
    rf.HB = "hb" in fl
    hbm.DOT = "head" in fl
    bgm.DOT = "bg" in fl
    for i in range(GR.fw.n_layers):
        L = GR.fw._layer(i)
        if getattr(L, "router", None) is None:
            continue
        if "rg" in fl and L.router.dtype == torch.float32:
            rg_orig[i] = L.router; L.router = L.router.half().contiguous()
        elif "rg" not in fl and i in rg_orig:
            L.router = rg_orig.pop(i)

# real-shape unit check of the fused Engram block (real weights, random rows): error + time vs the torch chain, then the arms
ee = GR.fw.eng[sorted(GR.fw.ple)[0]]
for T in (1, 4):
    xx = torch.randn(T, ee.hc, ee.n_embd, device=dev); rr = torch.randn(T, ee.n_cols, ee.head_dim, device=dev); vv = (torch.rand(T, ee.n_cols, device=dev) > 0.2).float()
    os.environ["HY4_ENGRAM_FUSED"] = "0"; a = ee.apply(xx, rr, vv); os.environ["HY4_ENGRAM_FUSED"] = "1"; b = ee.apply(xx, rr, vv)
    torch.cuda.synchronize()
    ts = []
    for f in ("0", "1"):
        os.environ["HY4_ENGRAM_FUSED"] = f; torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(50): ee.apply(xx, rr, vv)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / 50 * 1e6)
    print("ENG_REAL T", T, "rel err %.2e" % ((a - b).abs().max() / a.abs().max()).item(), "us torch %.0f fused %.0f" % tuple(ts), "D", ee.n_embd, "nc", ee.n_cols, "hd", ee.head_dim, flush=True)
wiki = json.load(open("/path/to/wiki_ids.json"))
W = int(os.environ.get("WIN", "6")); CTX, HALF, R = 512, 256, 4
def run_arm(name, fl, ref=None, seed=None):
    levers(fl)
    SG.PERT = None if seed is None else (torch.randn(GR.cap, e.M.D, generator=torch.Generator().manual_seed(seed)) * 1e-6).to(dev)
    GR.clear()
    nl, kl, same, lps = [], [], [], []
    t = time.time()
    for w in range(W):
        ids = wiki[w * CTX:w * CTX + CTX + 1]
        e._t0 = time.perf_counter(); e.reset(); e.prefill(ids[:HALF]); e.phase("decode")
        for i in range(0, CTX - HALF, R):
            lg = GR.step_rows(ids[HALF + i:HALF + i + R])[0]
            lp = torch.log_softmax(lg.reshape(R, -1).float(), -1)
            tgt = torch.tensor(ids[HALF + i + 1:HALF + i + R + 1], device=dev)
            nl.extend((-lp[torch.arange(R, device=dev), tgt]).tolist())
            if ref is None:
                lps.append(lp.cpu())
            else:
                lr = ref[len(nl) - R:len(nl)].to(dev)
                kl.extend((lr.exp() * (lr - lp)).sum(-1).tolist())
                same.extend((lr.argmax(-1) == lp.argmax(-1)).tolist())
    SG.PERT = None
    GR.clear()
    return np.array(nl), kl, same, (torch.cat(lps) if ref is None else None), time.time() - t
nl_ref, _, _, lp_ref, secs = run_arm("ref", [])
print("QRES", json.dumps(dict(arm="ref", n=len(nl_ref), ppl=round(float(np.exp(nl_ref.mean())), 4), secs=round(secs, 1))), flush=True)
plan = [("noise1234", [], 1234), ("noise77", [], 77), ("noise5", [], 5)] + [(k, k.split("+"), None) for k in os.environ.get("QARMS", "kvr,eng,kvr+eng,hb+head+bg+rg+kvr+eng").split(",")]
res = {}
for name, fl, seed in plan:
    nl, kl, same, _, secs = run_arm(name, fl, lp_ref, seed)
    d = nl - nl_ref
    r = dict(arm=name, ppl=round(float(np.exp(nl.mean())), 4), d_nll=round(float(d.mean()), 5), d_se=round(float(d.std(ddof=1) / np.sqrt(len(d))), 5), kl_mean=round(float(np.mean(kl)), 6), kl_p99=round(float(np.percentile(kl, 99)), 5), top1_same=round(float(np.mean(same)), 4), secs=round(secs, 1))
    res[name] = r
    print("QRES", json.dumps(r), flush=True)
nz = max(res[k]["kl_mean"] for k in res if k.startswith("noise"))
for k in res:
    if not k.startswith("noise"):
        print("QKL_VS_NOISE", k, "kl", res[k]["kl_mean"], "x noise-null max", round(res[k]["kl_mean"] / max(nz, 1e-12), 2), flush=True)
levers([])
print("QUAL_DONE", flush=True)
