"""EXL3_DEC_DSA_FAST A/B on the served GLM config (serve.py SPEED_ENV + MTP n1, MTP_FUSE_CATCHUP=2).
One load. Arms: off (flag 0) / on (flag = $DECATTN2_ON, default "1"). Adapted from mla-fast tools/glm/mlafast_ab.py (a1debac4).
Phase A (target only): greedy 32 ids x 3 prompts + teacher-forced decode-path NLL (3 x 4K prompt, 128 forced tokens).
Phase B (MTP generator as served): greedy 32 ids x 3, then 4K decode interleaved off/on/on/off..., 128 tokens.
Usage: python decattn2_ab.py <model> <corpus> <out.json> [n_pairs]"""
import os, sys, json, statistics, ast
M, C, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools/glm"))
sys.argv = ["glm_base", "fast", "-m", M, "--corpus", C, "-o", OUT + ".base.json"]
_src = open(os.path.join(ROOT, "tools/glm/serve.py")).read()
SPEED_ENV = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SPEED_ENV")
for k, v in SPEED_ENV.items(): os.environ.setdefault(k, v)
import torch, glm_base as G
torch.set_grad_enabled(False)
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.generator import generator as GM
from exllamav3.modules import block_graph as BG

ARMS = {"off": "0", "on": os.environ.get("DECATTN2_ON", "1")}
cfg = Config.from_directory(M)
model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(model, max_num_tokens=16384, max_history=1)
model.load(device="cuda:0", progressbar=False)
cids = G.corpus_ids(tok, 60000)
R = {"env": {k: os.environ.get(k) for k in SPEED_ENV}, "arms": ARMS}
def save(): json.dump(R, open(OUT, "w"), indent=1)

def set_arm(a):
    torch.cuda.synchronize()
    os.environ["EXL3_DEC_DSA_FAST"] = ARMS[a]
    BG.purge()

def prompts3(): return [cids[:, 45000 + 900 * i: 45000 + 900 * i + 700].contiguous() for i in range(3)]

gen = Generator(model=model, cache=cache, tokenizer=tok)
tfp = [(cids[:, 1000 + 5000 * i: 1000 + 5000 * i + 4096].contiguous(),
        cids[:, 5200 + 5000 * i: 5200 + 5000 * i + 128].contiguous()) for i in range(3)]
R["A"] = {}
for a in ARMS:
    set_arm(a)
    G.run_job(gen, prompts3()[0], 4, stop=False)
    ids = [G.run_job(gen, p, 32, stop=False)["ids"].tolist() for p in prompts3()]
    nll = []
    for p, c in tfp:
        f = G.run_job(gen, p, 128, stop=False, return_logits=True, constrain=c[0])
        lg = f["logits"][:128].float()
        nll.append(-torch.log_softmax(lg, -1).gather(-1, c[0, :lg.shape[0], None].to(lg.device)).mean().item())
    R["A"][a] = {"ids": ids, "nll": nll}
    o = R["A"]["off"]
    eq = [sum(x == y for x, y in zip(u, v)) for u, v in zip(ids, o["ids"])]
    dn = 100 * (statistics.mean(nll) / statistics.mean(o["nll"]) - 1)
    print(f"[A {a}] greedy_match {eq} nll {[round(x, 6) for x in nll]} dNLL {dn:+.4f}% "
          f"bitwise_nll {nll == o['nll']}", flush=True)
    save()

draft = Model.from_config(cfg, component="mtp")
dcache = Cache(draft, max_num_tokens=16384, max_history=1)
draft.load(device="cuda:0", progressbar=False)
GM.MTP_FUSE_CATCHUP = 2
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=1, record_draft_stats=True)
R["B"] = {"ids": {}, "decode": {a: [] for a in ARMS}}
dprompt = cids[:, 20000:24096].contiguous()
for a in ARMS:
    set_arm(a)
    G.run_job(gen, prompts3()[0], 8, stop=False)
    ids = [G.run_job(gen, p, 32, stop=False)["ids"].tolist() for p in prompts3()]
    R["B"]["ids"][a] = ids
    eq = [sum(x == y for x, y in zip(u, v)) for u, v in zip(ids, R["B"]["ids"]["off"])]
    print(f"[B {a}] greedy_match {eq}", flush=True)
    save()

order = (["off", "on", "on", "off"] * NP)[:2 * NP]
for rep, a in enumerate(order):
    set_arm(a)
    G.run_job(gen, dprompt, 8, stop=False)
    r = G.run_job(gen, dprompt, 128, stop=False)
    tps = (r["ntok"] - 1) / (r["total"] - r["ttft"])
    same = r["ids"][:32].tolist() == R["B"]["ids"].get("_dec", r["ids"][:32].tolist())
    R["B"]["ids"].setdefault("_dec", r["ids"][:32].tolist())
    R["B"]["decode"][a].append(tps)
    print(f"[dec {rep:2d} {a}] {tps:.3f} t/s ntok {r['ntok']} ids_same {same}", flush=True)
    save()
for a, v in R["B"]["decode"].items():
    print(f"RESULT decode4K {a}: median {statistics.median(v):.3f} min {min(v):.3f} max {max(v):.3f} n={len(v)}", flush=True)
off, on = (statistics.median(R["B"]["decode"][k]) for k in ("off", "on"))
print(f"RESULT delta {100 * (on / off - 1):+.2f}%", flush=True)
save(); print("DONE", flush=True)
