"""req_arms.py: dense Q8_0 -> Q6_K / Q5_K requant arms in ONE load (run inside the box job.sh with HY4_DENSE_REQ_ARMS=1, cwd = tree).

  ARMS=quality  teacher-forced wiki windows (llama-perplexity protocol: ctx 512, first half is context, second half scored; same wiki_ids.json): per arm NLL / PPL,
                paired delta vs the Q8_0 reference, KL(ref || arm) per position, top-1 agreement; null band = two Q8_0 arms with 1e-6 embedding noise.
  ARMS=speed    plain graph and speculative decode tok/s per (mode, kernel), q8 first and last (drift check).
Every arm switches dense_req.set_mode and drops the captured graphs."""
import json, os, sys, time
import numpy as np
tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve
serve.apply_env(16384)
import torch
from engine import Hy4Engine
import serve_core as C
GGUF = "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf"
OUT = "out"
e = Hy4Engine(tree, GGUF, ctx=16384, k=3)
e.load(print)
hf, SG, GR = e.hf, e.SG, e.GR
from exllamav3.loader import dense_req as DR
assert DR.ENABLED, "set HY4_DENSE_REQ_ARMS=1"
t0 = time.time()
n_all, n_ok, tot = DR.prepare(["q6_k", "q5_k"])
print(f"prepared owners {n_ok}/{n_all} bytes {tot} in {time.time() - t0:.1f}s", flush=True)
q8_bytes = sum(o.packed.numel() for o in DR._owners() if "alt" in o.__dict__)
print("DENSE_BYTES q8", q8_bytes, {k: v for k, v in tot.items()}, flush=True)
V = C.Vocab(e.meta)
wiki = json.load(open("/path/to/wiki_ids.json"))
ARMS = os.environ.get("ARMS", "quality").split(",")
dev = e.dev


def setmode(mode, kern="gemv"):
    DR.set_mode(mode, kern)
    GR.clear()


# ---- sanity: dequant / GEMV of each requantised encoding against the Q8_0 weight it came from
own = [o for o in DR._owners() if "alt" in o.__dict__ and type(o).__name__ == "QLinear"][:3]
for o in own:
    x = torch.randn(1, o.shape[1], device=dev)
    res = {}
    for m in ("q8", "q6_k", "q5_k"):
        DR.set_mode(m)
        w = o.dense()
        res[m] = (w, torch.empty(1, o.shape[0], device=dev))
        o.gemv(x.contiguous(), o.packed, res[m][1])
    w8, y8 = res["q8"][0], res["q8"][1]
    print("CHECK", o.gname, {m: dict(w_rel=round(float((res[m][0] - w8).norm() / w8.norm()), 5), y_rel=round(float((res[m][1] - y8).norm() / y8.norm()), 5)) for m in ("q6_k", "q5_k")}, flush=True)
setmode("q8")

# ---- quality
W = int(os.environ.get("WIN", "6")); CTX, HALF, R = 512, 256, 4


def run_arm(name, ref=None, seed=None):
    SG.PERT = None if seed is None else (torch.randn(GR.cap, e.M.D, generator=torch.Generator().manual_seed(seed)) * 1e-6).to(dev)
    GR.clear()
    nl, kl, same, lps = [], [], [], []
    t = time.time()
    for w in range(W):
        ids = wiki[w * CTX:w * CTX + CTX + 1]
        e._t0 = time.perf_counter(); e.reset(); e.prefill(ids[:HALF]); e.phase("decode")
        for i in range(0, CTX - HALF, R):
            lg, _am, _x, _h = GR.step_rows(ids[HALF + i:HALF + i + R])
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
    nl = np.array(nl)
    r = dict(arm=name, n=len(nl), nll=round(float(nl.mean()), 5), ppl=round(float(np.exp(nl.mean())), 4), secs=round(time.time() - t, 1))
    return nl, kl, same, (torch.cat(lps) if ref is None else None), r


def report(name, nl, ref_nl, kl, same, r):
    if ref_nl is not None:
        d = nl - ref_nl
        r.update(d_nll=round(float(d.mean()), 5), d_se=round(float(d.std(ddof=1) / np.sqrt(len(d))), 5),
                 kl_mean=round(float(np.mean(kl)), 6), kl_p99=round(float(np.percentile(kl, 99)), 5), top1_same=round(float(np.mean(same)), 4))
    print("QRES", json.dumps(r), flush=True)
    return r


if "quality" in ARMS:
    res = {}
    nl_ref, _, _, lp_ref, r = run_arm("q8_ref")
    report("q8_ref", nl_ref, None, None, None, r)
    json.dump({"q8_ref": nl_ref.tolist()}, open(f"{OUT}/req_nll_ref.json", "w"))
    plan = [("q8_noise1", "q8", 1), ("q8_noise2", "q8", 2)] + [(f"{m}_{k}", m, None) for m, k in (("q6_k", "gemv"), ("q5_k", "gemv"))]
    for name, mode, seed in plan:
        setmode(mode)
        nl, kl, same, _, r = run_arm(name, lp_ref, seed)
        res[name] = report(name, nl, nl_ref, kl, same, r)
        json.dump({"nll": nl.tolist(), "kl": kl}, open(f"{OUT}/req_nll_{name}.json", "w"))
    nz = [res[k]["kl_mean"] for k in res if "noise" in k]
    for k in res:
        if "noise" not in k and nz:
            print("QKL_VS_NOISE", k, "kl", res[k]["kl_mean"], "x noise-null", round(res[k]["kl_mean"] / max(max(nz), 1e-12), 1), flush=True)
    setmode("q8")

# ---- speed
if "speed" in ARMS:
    ids = V.encode(V.template.render([{"role": "user", "content": "Explain in a few short paragraphs why the sky is blue and why sunsets are red."}], None, True, reasoning_effort="no_think"))
    NT = int(os.environ.get("NT", "100"))

    def gen(spec, n):
        out = []
        st = e.generate(ids, n, lambda t: out.extend(t) or False, speculative=spec)
        return out, st["decode_tokens"] / st["decode_s"], st

    order = [("q8", "gemv"), ("q6_k", "gemv"), ("q6_k", "mmvq"), ("q5_k", "gemv"), ("q5_k", "mmvq"), ("q8", "gemv")]
    for mode, kern in order:
        setmode(mode, kern)
        gen(False, 20)
        _, tps, st = gen(False, NT)
        gen(True, 30)
        _, tps2, st2 = gen(True, 150)
        print("SRES", json.dumps(dict(mode=mode, kern=kern, plain_tok_s=round(tps, 2), spec_tok_s=round(tps2, 2), tau=round(st2["tau"], 2), rounds=st2["rounds"])), flush=True)
print("REQ_DONE", flush=True)
