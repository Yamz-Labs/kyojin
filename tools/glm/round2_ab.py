"""round2: served-config A/B for decode-round levers, one load, arms switched at run time.
Served config = serve.py SPEED_ENV + MTP n1f2 + UNION_V2. Cache max_history=2 so R=3 (n2) fits in the same load.
Arms (R2_ARMS, comma list; the first is the reference):
  base   served defaults
  glue   + mla_attn.EXL3_DSA_GLUE_FUSE (merged, default off)
  graph  + BLOCK_GRAPH_VERIFY + DEC_MOE_UNION_DEV (B2-1 arm B on today's tree)
  router + EXL3_DEC_ROUTER_ROWS=2 (one router launch for all verify rows)
  combine + EXL3_MOE_COMBINE_WIDE=1 (union MoE combine over H/512 column blocks per row)
  absorbX + EXL3_MLA_ABSORB_CFG from env R2_ABSORBX (default "32,4")
  n2     base with num_draft_tokens=2 (R=3 verify)
  dyn    num_draft_tokens=2 truncated to 1 by the confidence calibrator (R2_DYN_CONF, default 0.45)
  arms can be joined with '+' (e.g. glue+router+n2)
Phases:
  ids: greedy R2_IDS_TOK tokens x 3 unique chat prompts per arm, compared to the reference arm
  dec: 12 unique chat prompts (code/chat/prose), R2_DEC_TOK tokens, arms in ABC..CBA order per prompt
Metrics per run: t/s after first token, accepted drafts, verify rounds, ms/round, accept hist by position.
Usage: python round2_ab.py <model> <out.json>"""
import os, sys, json, statistics, ast, time
M, OUT = sys.argv[1], sys.argv[2]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools/glm"))
_src = open(os.path.join(ROOT, "tools/glm/serve.py")).read()
SPEED_ENV = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SPEED_ENV")
for k, v in SPEED_ENV.items(): os.environ.setdefault(k, os.path.expanduser(v))
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
import torch
torch.set_grad_enabled(False)
import exllamav3
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator import generator as GM
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules import block_graph as BG
from exllamav3.modules import mla_attn as MLA
import serve as SV
from mtp_bench_prompts import PROMPTS

ARMS = os.environ.get("R2_ARMS", "base,glue,graph,n2").split(",")
IDS_TOK = int(os.environ.get("R2_IDS_TOK", "128"))
DEC_TOK = int(os.environ.get("R2_DEC_TOK", "128"))
SKIP_IDS = os.environ.get("R2_SKIP_IDS", "0") != "0"
t0 = time.perf_counter()
cfg = Config.from_directory(M)
model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(model, max_num_tokens=16384, max_history=2)
model.load(device="cuda:0", progressbar=False)
draft = Model.from_config(cfg, component="mtp")
dcache = Cache(draft, max_num_tokens=16384, max_history=2)
draft.load(device="cuda:0", progressbar=False)
GM.MTP_FUSE_CATCHUP = 2
template = open(os.path.join(M, "chat_template.jinja")).read()
def enc(text):
    p = SV.render_prompt(template, [{"role": "user", "content": text}], None)
    return tok.encode(p, encode_special_tokens=True)
R = {"env": {k: os.environ.get(k) for k in list(SPEED_ENV) + ["EXL3_MOE_UNION_V2"]}, "arms": ARMS,
     "load_s": time.perf_counter() - t0}
print(f"loaded {R['load_s']:.0f}s arms {ARMS}", flush=True)
def save(): json.dump(R, open(OUT, "w"), indent=1)

gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=2, record_draft_stats=True)

DEFAULTS = {"EXL3_DEC_MOE_UNION_DEV": os.environ.get("EXL3_DEC_MOE_UNION_DEV", "0"),
            "EXL3_DEC_ROUTER_ROWS": os.environ.get("EXL3_DEC_ROUTER_ROWS", "1"),
            "EXL3_MLA_ABSORB_CFG": os.environ.get("EXL3_MLA_ABSORB_CFG", ""),
            "EXL3_MOE_COMBINE_WIDE": os.environ.get("EXL3_MOE_COMBINE_WIDE", "0"),
            "EXL3_HOST_LEAN": os.environ.get("EXL3_HOST_LEAN", "0"),
            "EXL3_HOST_CUTS": os.environ.get("EXL3_HOST_CUTS", "0")}

CALS = {}
def set_arm(name):
    torch.cuda.synchronize()
    parts = set(name.split("+"))
    for k, v in DEFAULTS.items(): os.environ[k] = v
    MLA.EXL3_DSA_GLUE_FUSE = "glue" in parts
    BG.BLOCK_GRAPH_VERIFY = "graph" in parts
    if "lean" in parts or "hostcuts" in parts: os.environ["EXL3_HOST_LEAN"] = "1"
    # hostidle2: launch/copy trimming, only reachable under EXL3_HOST_LEAN (see _host_cuts)
    if "hostcuts" in parts: os.environ["EXL3_HOST_CUTS"] = "1"
    if "graph" in parts: os.environ["EXL3_DEC_MOE_UNION_DEV"] = "1"
    if "router" in parts: os.environ["EXL3_DEC_ROUTER_ROWS"] = "2"
    if "combine" in parts: os.environ["EXL3_MOE_COMBINE_WIDE"] = "1"
    for p in parts:
        if p.startswith("absorb"): os.environ["EXL3_MLA_ABSORB_CFG"] = os.environ.get("R2_" + p.upper(), "32,4")
    gen.num_draft_tokens = 2 if ("n2" in parts or "dyn" in parts) else 1
    # dyn: R=3 only when the online calibrator expects the 2nd draft to pay (EXL3 DraftConfidenceCalibrator,
    # one persistent calibrator per arm so it keeps learning across interleaved runs)
    if "dyn" in parts:
        if name not in CALS:
            from exllamav3.generator.draft_confidence import DraftConfidenceCalibrator
            CALS[name] = DraftConfidenceCalibrator(float(os.environ.get("R2_DYN_CONF", "0.45")))
        gen.draft_calibrator = CALS[name]
    else:
        gen.draft_calibrator = None
    BG.purge()
    torch.cuda.synchronize()

def run(ids, n):
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
    st = list(getattr(job, "draft_stats", []) or [])
    return {"ids": t.tolist(), "ttft": ttft, "total": time.perf_counter() - t0, "ntok": int(t.numel()),
            "acc": acc, "stats": [list(map(int, s)) for s in st]}

flat = [(c, p) for c in ("code", "chat", "prose") for p in PROMPTS[c]]
id_prompts = [enc(PROMPTS[c][0]) for c in ("code", "chat", "prose")]

# ---- ids gate ----
R["ids"] = {}
if not SKIP_IDS:
    for a in list(ARMS):
        try:
            set_arm(a); run(id_prompts[0][:, :32], 4)
            R["ids"][a] = [run(p, IDS_TOK)["ids"] for p in id_prompts]
        except Exception as ex:
            import traceback; traceback.print_exc()
            R.setdefault("errors", {})[a] = repr(ex); ARMS.remove(a); save()
            torch.cuda.synchronize(); gen.clear_queue() if hasattr(gen, "clear_queue") else None
            continue
        ref = R["ids"][ARMS[0]]
        eq = [sum(x == y for x, y in zip(u, v)) for u, v in zip(R["ids"][a], ref)]
        print(f"[ids {a}] match vs {ARMS[0]} {eq}/{IDS_TOK}", flush=True); save()

# ---- decode ----
R["dec"] = {a: [] for a in ARMS}
for i, (cat, p) in enumerate(flat):
    ids = enc(p)
    order = ARMS if i % 2 == 0 else ARMS[::-1]
    for a in order:
        try:
            set_arm(a); run(ids[:, :16], 4)
            r = run(ids, DEC_TOK)
        except Exception as ex:
            import traceback; traceback.print_exc()
            R.setdefault("errors", {})[a] = repr(ex); ARMS.remove(a); del R["dec"][a]; save()
            continue
        dt = r["total"] - r["ttft"]
        rounds = len(r["stats"]) or max(1, r["ntok"] - 1 - (r["acc"] or 0))
        rec = {"i": i, "cat": cat, "tps": (r["ntok"] - 1) / dt, "acc": r["acc"], "ntok": r["ntok"],
               "rounds": rounds, "ms_round": 1000 * dt / rounds, "stats": r["stats"]}
        R["dec"][a].append(rec)
        print(f"[dec {i} {cat} {a}] {rec['tps']:.3f} t/s acc {r['acc']} rounds {rounds} {rec['ms_round']:.2f} ms/round", flush=True)
        save()

def ms(v):
    m = statistics.mean(v); se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
    return m, se
ref = ARMS[0]
R["summary"] = {}
for a in ARMS:
    d = R["dec"][a]
    m, se = ms([x["tps"] for x in d]); m2, se2 = ms([x["ms_round"] for x in d])
    pr = [100 * (x["tps"] / y["tps"] - 1) for x, y in zip(d, R["dec"][ref])]
    pm, pse = ms(pr)
    # accept per draft position: stats rows are (position, window, accepted)
    pos = {}
    for x in d:
        for s in x["stats"]:
            w, k = s[1], s[2]
            for j in range(w):
                pos.setdefault(j, [0, 0]); pos[j][1] += 1; pos[j][0] += int(k > j)
    acc_pos = {j: round(v[0] / v[1], 4) for j, v in sorted(pos.items()) if v[1]}
    tpr = statistics.mean([(x["ntok"] - 1) / x["rounds"] for x in d])
    R["summary"][a] = {"tps": m, "tps_se": se, "ms_round": m2, "ms_round_se": se2, "paired_pct": pm,
                       "paired_se": pse, "acc_pos": acc_pos, "tok_per_round": tpr}
    print(f"RESULT {a}: {m:.3f} +- {se:.3f} t/s | {m2:.2f} +- {se2:.2f} ms/round | paired vs {ref} {pm:+.2f} +- {pse:.2f} % "
          f"| tok/round {tpr:.3f} | accept by pos {acc_pos} | n={len(d)}", flush=True)
save()

# ---- optional: torch.profiler op census of one served decode (which Python sites launch glue) ----
PROF_ARM = os.environ.get("R2_PROF", "")
if PROF_ARM:
    from torch.profiler import profile, ProfilerActivity
    set_arm(PROF_ARM); ids = enc(flat[1][1]); run(ids, 8)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        r = run(ids, 32)
    rounds = max(1, len(r["stats"]))
    rows = []
    for e in prof.key_averages(group_by_stack_n=6):
        if e.device_type.name != "CPU" or not e.stack: continue
        n = getattr(e, "count", 0)
        rows.append((n / rounds, e.key, e.stack[:6]))
    rows.sort(key=lambda t: -t[0])
    with open(OUT + ".prof.txt", "w") as f:
        f.write(f"arm {PROF_ARM} rounds {rounds}\n")
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total" if hasattr(prof.key_averages()[0], "self_cuda_time_total") else "self_device_time_total", row_limit=60))
        f.write("\n-- CPU ops per round by stack (top 150) --\n")
        for c, k, st in rows[:150]:
            f.write(f"{c:8.2f}/rd {k:40s} {' <- '.join(st)}\n")
    print(f"prof written {OUT}.prof.txt", flush=True)
print("DONE", flush=True)
