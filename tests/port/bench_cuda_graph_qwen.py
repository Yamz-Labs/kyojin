# Whole-forward CUDA graph bench/check for Qwen3.8-27B (GatedDeltaNet hybrid, mrope), B=1. Needs a GPU; not run by CI.
# usage: [BENCH_DRAFT=DFLASH2_DIR [BENCH_NDRAFT=7]] EXL3_CUDA_GRAPH=1 python -I bench_cuda_graph_qwen.py ENGINE_DIR MODEL_DIR OUT.json [NEW_TOKENS]
#   plain : model=<Qwen3.8 EXL3 dir>          (Q=1 decode graph)
#   dflash: + BENCH_DRAFT=<DFlash2 dir>   (Q=1+NDRAFT verify graphs, recurrent_history/dflash_verify,
#           target hidden taps exported per replay; drafter + ring cache + Cache(max_batch_size=1, max_history) as smoke_qwen.py)
# Arms in one process, same load: eager, graph, eager, graph (module flag flipped + purge between arms).
# Per arm and prompt: decode tok/s (smoke_qwen.py method), token stream. Asserts graph stream == eager stream, and
# compares the logits returned by model.forward on the first LOGIT_STEPS decode steps (separate untimed pass).
import sys, os, time, json
eng, mdir, out = sys.argv[1], sys.argv[2], sys.argv[3]; NEW = int(sys.argv[4]) if len(sys.argv) > 4 else 256
LOGIT_STEPS = 16
sys.path.insert(0, eng)
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.model import cuda_graph as cg
import exllamav3

res = {"engine": exllamav3.__file__, "model": mdir, "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
       "arms": [], "new_tokens": NEW}
cfg = Config.from_directory(mdir); model = Model.from_config(cfg)
DFL = os.environ.get("BENCH_DRAFT"); ND = int(os.environ.get("BENCH_NDRAFT", "7"))
res["draft"] = DFL; res["num_draft_tokens"] = ND if DFL else 0
cache = Cache(model, max_num_tokens=8192, max_batch_size=1, **({"max_history": max(3, ND)} if DFL else {}))
model.load(); tok = Tokenizer.from_config(cfg)
kw = {}
if DFL:
    dcfg = Config.from_directory(DFL); dm = Model.from_config(dcfg); dc = Cache(dm, max_num_tokens=16 * 256); dm.load()
    dc.max_num_tokens = 8192   # ring cache as in a served drafter: the generator only compares sizes
    kw = dict(draft_model=dm, draft_cache=dc, num_draft_tokens=ND)
gen = Generator(model=model, cache=cache, tokenizer=tok, **kw)
S = ArgmaxSampler()
pl = os.environ["PROMPT_LONG"]   # a long text file (summary prompt)
base = open(pl).read()
prompts = [
  "<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a message in a bottle.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "<|im_start|>user\nWrite a Python function that parses an ISO 8601 date string without using datetime, with tests.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "<|im_start|>user\nSummarise the following text in five bullet points:\n" + base + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
]

def run(ids, new):
    job = Job(input_ids=ids, max_new_tokens=new, sampler=S, stop_conditions=[])
    torch.cuda.synchronize(); t0 = time.time(); tf = None; o = []
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                t = r.get("token_ids")
                if t is not None and t.numel():
                    if tf is None: torch.cuda.synchronize(); tf = time.time()
                    o += t.flatten().tolist()
    torch.cuda.synchronize(); te = time.time()
    return o, tf - t0, te - tf

# logits recorder around Model.forward (decode calls only; prefill does not go through forward)
rec = None
_orig_forward = model.forward
def _fwd(input_ids, params=None):
    y = _orig_forward(input_ids, params)
    if rec is not None and len(rec) < LOGIT_STEPS:
        rec.append(y.detach().float().cpu().clone())
    return y
model.forward = _fwd

def arm(name, graph):
    global rec
    cg.purge(); cg.CUDA_GRAPH_ENABLED = graph
    stats0 = None
    a = {"arm": name, "graph": graph, "prompts": [], "logits": []}
    r0 = model.__dict__.get("_cuda_graph_runner")
    if r0 is not None:
        r0.stats = {"captures": 0, "replays": 0, "warmups": 0, "declines": type(r0.stats["declines"])()}
    run(tok.encode("Hello, warm-up."), 16)
    for p in prompts:
        ids = tok.encode(p)
        run(ids, 8)                       # warm this prompt's block-table width (graph arm: warmups + capture)
        o, tp, td = run(ids, NEW)
        a["prompts"].append({"n_in": ids.shape[-1], "decode_tps": (len(o) - 1) / td, "decode_s": td, "ttft_s": tp, "ids": o})
        print(json.dumps({"arm": name, "n_in": ids.shape[-1], "decode_tps": round((len(o) - 1) / td, 2)}), flush=True)
        rec = []; run(ids, LOGIT_STEPS + 1); a["logits"].append(rec); rec = None
    r = model.__dict__.get("_cuda_graph_runner")
    if graph and r is not None:
        a["graph_stats"] = {k: (dict(v) if isinstance(v, dict) or hasattr(v, "items") else v) for k, v in r.stats.items()}
        print(json.dumps({"graph_stats": a["graph_stats"]}), flush=True)
    res["arms"].append(a)
    return a

arms = [arm("eager1", False), arm("graph1", True), arm("eager2", False), arm("graph2", True)]
cg.purge()

ok = True
res["checks"] = []
for ea, ga in ((arms[0], arms[1]), (arms[2], arms[3]), (arms[0], arms[2]), (arms[1], arms[3])):
    for i, (pe, pg) in enumerate(zip(ea["prompts"], ga["prompts"])):
        a_, b_ = pe["ids"], pg["ids"]
        div = next((j for j, (u, v) in enumerate(zip(a_, b_)) if u != v), None)
        if div is None and len(a_) != len(b_): div = min(len(a_), len(b_))
        md = max((float((x - y).abs().max()) if x.shape == y.shape else float("inf")) for x, y in zip(ea["logits"][i], ga["logits"][i])) \
            if ea["logits"][i] else None
        c = {"a": ea["arm"], "b": ga["arm"], "prompt": i, "first_divergence": div, "logits_max_abs_diff": md,
             "n_logit_steps": len(ea["logits"][i])}
        res["checks"].append(c); print(json.dumps(c), flush=True)
        if div is not None or md not in (0.0,): ok = False
for a in arms: a.pop("logits")
for name, sel in (("eager", (0, 2)), ("graph", (1, 3))):
    res[f"mean_tps_{name}"] = [sum(arms[k]["prompts"][i]["decode_tps"] for k in sel) / 2 for i in range(len(prompts))]
res["identical"] = ok
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("mean_tps_eager", "mean_tps_graph", "identical")}), flush=True)
assert ok, "graph != eager (see checks)"
