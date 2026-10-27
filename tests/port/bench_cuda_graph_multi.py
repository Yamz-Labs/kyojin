# Multi-job whole-forward CUDA graph check/bench (EXL3_CUDA_GRAPH_MULTI), Qwen3.8-27B + DFlash2 drafter. Needs a GPU; not run by CI.
# usage: python -I bench_cuda_graph_multi.py ENGINE_DIR [OUT.json] [NEW_TOKENS=128]
#   env: BENCH_MODEL (required: target model dir), BENCH_DRAFT (required: DFlash2 drafter dir), BENCH_NDRAFT (7),
#        BENCH_BS ("2,4"), BENCH_ROUNDS (2 = off/on/off/on)
# One process, one load. Arm "off" = runner enabled, MULTI off (B>1 declines to eager); arm "on" = MULTI on, via
# cuda_graph.set_multi() + purge() between arms. For B in BENCH_BS: B concurrent greedy jobs (distinct short prompts,
# NEW tokens each, DFlash2 verify Q=1+NDRAFT).
# (a) every job's token ids must be identical in all arms (+ runner stats incl. declines by reason),
# (b) aggregate tok/s per arm, interleaved off/on/off/on, (c) logits of the first LOGIT_STEPS model.forward calls (the
#     batches are identical step by step when the streams are, so graph vs eager logits are comparable): max abs diff.
# Prints JSON lines; exit code 1 if tokens or logits differ.
import sys, os, time, json
eng = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else None; NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 128
sys.path.insert(0, eng)
os.environ["EXL3_CUDA_GRAPH"] = "1"
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.model import cuda_graph as cg
import exllamav3

MDIR = os.environ["BENCH_MODEL"]
DFL = os.environ["BENCH_DRAFT"]
ND = int(os.environ.get("BENCH_NDRAFT", "7"))
BS = [int(b) for b in os.environ.get("BENCH_BS", "2,4").split(",")]
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "2"))
LOGIT_STEPS = 24
P = lambda d: print(json.dumps(d), flush = True)
res = {"engine": exllamav3.__file__, "model": MDIR, "draft": DFL, "num_draft_tokens": ND, "new_tokens": NEW,
       "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3") or k.startswith("EXLLAMAV3")}, "arms": []}

cfg = Config.from_directory(MDIR); model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens = 8192, max_batch_size = max(BS), max_history = max(3, ND))
model.load(); tok = Tokenizer.from_config(cfg)
dcfg = Config.from_directory(DFL); dm = Model.from_config(dcfg); dc = Cache(dm, max_num_tokens = 16 * 256); dm.load()
dc.max_num_tokens = 8192   # ring cache as in a served drafter: the generator only compares sizes
gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc, num_draft_tokens = ND)
S = ArgmaxSampler()
Q = lambda s: "<|im_start|>user\n" + s + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prompts = [Q(s) for s in (
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain how a hash map handles collisions, with a small C example.",
    "List ten facts about the planet Mars and one about each of its moons.")]
assert len(prompts) >= max(BS)

rec = None
_orig_forward = model.forward
def _fwd(input_ids, params = None):
    y = _orig_forward(input_ids, params)
    if rec is not None and len(rec) < LOGIT_STEPS:
        rec.append(y.detach().float().cpu().clone())
    return y
model.forward = _fwd

def run(b, new):
    """b concurrent jobs; returns per-job token lists, wall seconds (enqueue to last token)"""
    jobs = [Job(input_ids = tok.encode(prompts[i]), max_new_tokens = new, sampler = S, stop_conditions = [], identifier = i) for i in range(b)]
    outs = [[] for _ in range(b)]
    torch.cuda.synchronize(); t0 = time.time()
    gen.enqueue(jobs)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                t = r.get("token_ids")
                if t is not None and t.numel():
                    outs[r["identifier"]] += t.flatten().tolist()
    torch.cuda.synchronize()
    return outs, time.time() - t0

def arm(name, multi, b):
    global rec
    cg.purge(); cg.set_multi(multi); cg.CUDA_GRAPH_ENABLED = True
    r0 = model.__dict__.get("_cuda_graph_runner")
    if r0 is not None:
        r0.stats = {"captures": 0, "replays": 0, "warmups": 0, "declines": type(r0.stats["declines"])()}
    run(b, 24)                          # per-shape warmups + capture + autotune (graph arm), tuner (off arm)
    outs, dt = run(b, NEW)              # timed: includes the prefill of B short prompts (same in both arms)
    n = sum(len(o) for o in outs)
    rec = []; run(b, LOGIT_STEPS + 1); logits = rec; rec = None
    a = {"arm": name, "multi": multi, "B": b, "agg_tps": n / dt, "wall_s": dt, "n_tokens": n, "ids": outs, "logits": logits}
    r = model.__dict__.get("_cuda_graph_runner")
    a["runner_stats"] = {k: (dict(v) if hasattr(v, "items") else v) for k, v in r.stats.items()} if r is not None else None
    P({k: v for k, v in a.items() if k not in ("ids", "logits")})
    res["arms"].append(a)
    return a

ok = True
res["checks"] = []
for b in BS:
    arms = []
    for rd in range(ROUNDS):                              # interleaved: off, on, off, on
        arms.append(arm(f"B{b}_off{rd}", False, b)); arms.append(arm(f"B{b}_on{rd}", True, b))
    ref = arms[0]
    for a in arms[1:]:
        for j in range(b):
            x, y = ref["ids"][j], a["ids"][j]
            div = next((k for k, (u, v) in enumerate(zip(x, y)) if u != v), None)
            if div is None and len(x) != len(y): div = min(len(x), len(y))
            if div is not None: ok = False
            P({"check": "tokens", "B": b, "a": ref["arm"], "b": a["arm"], "job": j, "n": len(x), "first_divergence": div})
        md = max((float((u - v).abs().max()) if u.shape == v.shape else float("inf")) for u, v in zip(ref["logits"], a["logits"])) \
            if ref["logits"] and len(ref["logits"]) == len(a["logits"]) else float("inf")
        if md != 0.0: ok = False
        P({"check": "logits", "B": b, "a": ref["arm"], "b": a["arm"], "n_steps": len(a["logits"]), "max_abs_diff": md})
        res["checks"].append({"B": b, "a": ref["arm"], "b": a["arm"], "logits_max_abs_diff": md})
    off = [a["agg_tps"] for a in arms if not a["multi"]]; on = [a["agg_tps"] for a in arms if a["multi"]]
    P({"summary": True, "B": b, "mean_tps_off": sum(off) / len(off), "mean_tps_on": sum(on) / len(on),
       "speedup": (sum(on) / len(on)) / (sum(off) / len(off))})
cg.purge()
for a in res["arms"]: a.pop("logits")
res["identical"] = ok
if out: json.dump(res, open(out, "w"), indent = 1)
P({"identical": ok})
sys.exit(0 if ok else 1)
