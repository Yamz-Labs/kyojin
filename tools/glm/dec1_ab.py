# verifyrow1 model A/B: EXL3_GEMV_R_DEC1 off/on/on/off at MTP R=2/3/4, one load, served SPEED_ENV.
# Per (prompt, R): 4 runs of 128 greedy tokens; decode t/s + ms/round; ids must be identical across
# arms (kernel is bit-exact). Unique prompt per rep. Usage: python tools/glm/dec1_ab.py <out.json>
import os, sys, ast, json, time, statistics, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools/glm"))
_src = open(os.path.join(ROOT, "tools/glm/serve.py")).read()
for k, v in next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SPEED_ENV").items():
    os.environ.setdefault(k, os.path.expanduser(v))
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
torch.set_grad_enabled(False)
import exllamav3
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator import generator as GM
from exllamav3.generator.sampler import GreedySampler
from mtp_bench_prompts import PROMPTS
OUT = sys.argv[1]
MODEL = os.environ.get("AB_MODEL", "~/models/glm53-exl3-td205")
RS = [int(x) for x in os.environ.get("AB_RS", "3,4,2").split(",")]
NTOK = int(os.environ.get("AB_NTOK", "128"))
NP = int(os.environ.get("AB_NP", "3"))
config = Config.from_directory(MODEL)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=8192, max_history=max(RS) - 1)
model.load(device="cuda:0", progressbar=False)
dm = Model.from_config(config, component="mtp")
dcache = Cache(dm, max_num_tokens=8192, max_history=max(RS) - 1)
dm.load(device="cuda:0", progressbar=False)
GM.MTP_FUSE_CATCHUP = 2
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                num_draft_tokens=max(RS) - 1, record_draft_stats=True)
prompts = [PROMPTS[c][i] for i in range(4) for c in ("code", "chat", "prose")][:NP]

def run(p, R, arm):
    os.environ["EXL3_GEMV_R_DEC1"] = arm
    gen.num_draft_tokens = R - 1
    ids = tok.encode(f"[gMASK]<sop><|user|>\n{p}<|assistant|>\n", encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=NTOK, sampler=GreedySampler(), stop_conditions=[])
    gen.enqueue(job)
    out = []; t1 = None; n1 = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            t = r.get("token_ids")
            if t is not None and t.numel():
                if t1 is None:
                    torch.cuda.synchronize(); t1 = time.perf_counter(); n1 = int(t.numel())
                out += [int(x) for x in t.flatten().tolist()]
    torch.cuda.synchronize(); t2 = time.perf_counter()
    nr = len(job.draft_stats)
    return dict(tps=(len(out) - n1) / (t2 - t1), ms_round=1000 * (t2 - t1) / max(nr - 1, 1),
                rounds=nr, ntok=len(out), ids=out)

run(prompts[0], RS[0], "0"); run(prompts[0], RS[0], "1")   # warm both paths
RES = []
for R in RS:
    for pi, p in enumerate(prompts):
        recs = {}
        for k, arm in enumerate(("0", "1", "1", "0")):
            r = run(p, R, arm); recs.setdefault(arm, []).append(r)
            print(f"R={R} p{pi} arm={arm} tps={r['tps']:.3f} ms/round={r['ms_round']:.2f} rounds={r['rounds']}", flush=True)
        ids_eq = all(x["ids"] == recs["0"][0]["ids"] for a in recs for x in recs[a])
        off = statistics.mean(x["tps"] for x in recs["0"]); on = statistics.mean(x["tps"] for x in recs["1"])
        RES.append(dict(R=R, p=pi, off_tps=off, on_tps=on, gain_pct=100 * (on / off - 1), ids_equal=ids_eq,
                        off_ms=statistics.mean(x["ms_round"] for x in recs["0"]),
                        on_ms=statistics.mean(x["ms_round"] for x in recs["1"]),
                        ids=recs["0"][0]["ids"]))
        print(f"== R={R} p{pi} off {off:.3f} on {on:.3f} gain {RES[-1]['gain_pct']:+.2f} % ids_equal {ids_eq}", flush=True)
        json.dump(RES, open(OUT, "w"))
for R in RS:
    g = [x["gain_pct"] for x in RES if x["R"] == R]
    se = statistics.stdev(g) / len(g) ** 0.5 if len(g) > 1 else 0
    dm_ = [x["off_ms"] - x["on_ms"] for x in RES if x["R"] == R]
    print(f"SUMMARY R={R} gain {statistics.mean(g):+.2f} +- {se:.2f} % (n={len(g)}) "
          f"ms/round {statistics.mean(x['off_ms'] for x in RES if x['R']==R):.2f} -> "
          f"{statistics.mean(x['on_ms'] for x in RES if x['R']==R):.2f} (-{statistics.mean(dm_):.2f}) "
          f"ids_equal {all(x['ids_equal'] for x in RES if x['R']==R)}", flush=True)
# R-invariance: same prompt, ids at each R vs R=2 (greedy spec decode must match plain greedy up to ties)
for pi in range(len(prompts)):
    base = next((x["ids"] for x in RES if x["R"] == 2 and x["p"] == pi), None)
    if base:
        print(f"RINV p{pi}: " + " ".join(f"R{x['R']}={sum(a == b for a, b in zip(x['ids'], base))}/{len(base)}"
                                        for x in RES if x["p"] == pi), flush=True)
print("DONE")
