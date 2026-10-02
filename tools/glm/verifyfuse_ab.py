"""verifyfuse1: EXL3_VERIFY_FUSE A/B on the served GLM config (serve.py SPEED_ENV + MTP n1f2, UNION_V2).
One load, arms switched at run time (VERIFY_FUSE["on"] + block_graph.purge()).
  R1: plain generator (no draft) greedy 32 ids x 3 prompts   -> must be bitwise (flag only acts on R > 1)
  R2: MTP generator greedy 64 ids x 3 prompts + accept        -> greedy-ids gate
  dec: served 4K decode, 128 tok, quads off/on/on/off, one unique 4K prompt per quad
Usage: python verifyfuse_ab.py <model> <corpus> <out.json> [n_quads]"""
import os, sys, json, statistics, ast, time
M, C, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NQ = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools/glm"))
sys.argv = ["glm_base", "fast", "-m", M, "--corpus", C, "-o", OUT + ".base.json"]
_src = open(os.path.join(ROOT, "tools/glm/serve.py")).read()
SPEED_ENV = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SPEED_ENV")
for k, v in SPEED_ENV.items(): os.environ.setdefault(k, v)
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
import torch, glm_base as G
torch.set_grad_enabled(False)
import exllamav3
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator import generator as GM
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules import block_graph as BG
from exllamav3.modules.quant import exl3 as EXL3

ARMS = ("off", "on")
t0 = time.perf_counter()
cfg = Config.from_directory(M)
model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(model, max_num_tokens=16384, max_history=1)
model.load(device="cuda:0", progressbar=False)
draft = Model.from_config(cfg, component="mtp")
dcache = Cache(draft, max_num_tokens=16384, max_history=1)
draft.load(device="cuda:0", progressbar=False)
GM.MTP_FUSE_CATCHUP = 2
cids = G.corpus_ids(tok, 80000)
R = {"env": {k: os.environ.get(k) for k in list(SPEED_ENV) + ["EXL3_MOE_UNION_V2"]}, "load_s": time.perf_counter() - t0}
print(f"loaded {R['load_s']:.0f}s", flush=True)
def save(): json.dump(R, open(OUT, "w"), indent=1)

def set_arm(a):
    torch.cuda.synchronize()
    EXL3.VERIFY_FUSE["on"] = a == "on"
    os.environ["EXL3_VERIFY_FUSE"] = "1" if a == "on" else "0"
    BG.purge()

def run(gen, ids, n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), stop_conditions=[])
    gen.enqueue(job)
    t0 = time.perf_counter(); ttft = None; toks = []; acc = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            tid = r.get("token_ids")
            if tid is not None and tid.numel():
                if ttft is None:
                    torch.cuda.synchronize(); ttft = time.perf_counter() - t0
                toks.append(tid.cpu())
            if r.get("eos"): acc = r.get("accepted_draft_tokens")
    torch.cuda.synchronize()
    t = torch.cat(toks, -1).flatten()
    return {"ids": t.tolist(), "ttft": ttft, "total": time.perf_counter() - t0, "ntok": int(t.numel()), "acc": acc}

def prompts3(): return [cids[:, 45000 + 900 * i: 45000 + 900 * i + 700].contiguous() for i in range(3)]

SKIP_IDS = os.environ.get("VF_SKIP_IDS", "0") != "0"
# ---- R1: plain decode, flag must be a no-op ----
gen1 = Generator(model=model, cache=cache, tokenizer=tok)
R["R1"] = {}
for a in (() if SKIP_IDS else ARMS):
    set_arm(a); run(gen1, prompts3()[0], 4)
    R["R1"][a] = [run(gen1, p, 32)["ids"] for p in prompts3()]
R["R1"]["bitwise"] = R["R1"].get("on") == R["R1"].get("off")
print(f"[R1] bitwise {R['R1']['bitwise']}", flush=True); save()
del gen1

gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=1, record_draft_stats=True)
R["R2"] = {}
for a in (() if SKIP_IDS else ARMS):
    set_arm(a); run(gen, prompts3()[0], 8)
    rs = [run(gen, p, 64) for p in prompts3()]
    R["R2"][a] = {"ids": [r["ids"] for r in rs], "acc": [r["acc"] for r in rs]}
if not SKIP_IDS:
    eq = [sum(x == y for x, y in zip(u, v)) for u, v in zip(R["R2"]["on"]["ids"], R["R2"]["off"]["ids"])]
    R["R2"]["match"] = eq
    print(f"[R2] greedy match {eq}/64 acc off {R['R2']['off']['acc']} on {R['R2']['on']['acc']}", flush=True); save()

R["dec"] = {a: [] for a in ARMS}; R["dec_acc"] = {a: [] for a in ARMS}; R["dec_msr"] = {a: [] for a in ARMS}
for q in range(NQ):
    q0 = int(os.environ.get("VF_Q0", "0")) + q
    dp = cids[:, 4096 * (q0 + 1) + 7 * q0: 4096 * (q0 + 2) + 7 * q0].contiguous()   # unique 4K prompt per quad
    for a in ("off", "on", "on", "off"):
        set_arm(a); run(gen, dp[:, :64], 4)
        r = run(gen, dp, 128)
        tps = (r["ntok"] - 1) / (r["total"] - r["ttft"])
        msr = 1000 * (r["total"] - r["ttft"]) / max(1, r["ntok"] - 1 - (r["acc"] or 0))   # ms per verify round
        R["dec"][a].append(tps); R["dec_acc"][a].append(r["acc"]); R["dec_msr"][a].append(msr)
        print(f"[dec q{q} {a}] {tps:.3f} t/s acc {r['acc']} {msr:.2f} ms/round", flush=True); save()
def ms(v):
    m = statistics.mean(v); se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
    return m, se
for a in ARMS:
    m, se = ms(R["dec"][a]); m2, se2 = ms(R["dec_msr"][a])
    print(f"RESULT {a}: {m:.3f} +- {se:.3f} t/s, {m2:.2f} +- {se2:.2f} ms/round, n={len(R['dec'][a])}", flush=True)
# paired per quad (same prompt): mean of the two on / mean of the two off
pr = [100 * ((R["dec"]["on"][2*q] + R["dec"]["on"][2*q+1]) / (R["dec"]["off"][2*q] + R["dec"]["off"][2*q+1]) - 1) for q in range(NQ)]
m, se = ms(pr); R["delta_pct"] = pr
print(f"RESULT paired delta {m:+.2f} +- {se:.2f} % (per quad {[round(x, 2) for x in pr]})", flush=True)
save(); print("DONE", flush=True)
