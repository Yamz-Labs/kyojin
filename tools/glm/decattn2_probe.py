"""probe: which DSA decode kernel does the served GLM config hit, at which R, and what
do the real top-k indices look like at 4K (valid count, are the -1 tail-only)?
Served env: serve.py SPEED_ENV + MTP (num_draft_tokens=1, MTP_FUSE_CATCHUP=2).
Usage: python decattn2_probe.py <model> <corpus> <out.json>"""
import os, sys, json, ast, collections
M, C, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
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
import exllamav3.modules.attention_fn.dsa_triton as D

calls = collections.Counter()
idx_stats = []
_orig = D.dsa_attn
def wrapped(q, *a, **kw):
    ind = kw.get("indices")
    R = ind.shape[0] if ind is not None else -1
    dt = D._dsa_dec_dt_cfg() is not None and R <= int(os.environ.get("EXL3_DSA_DEC_DT_R", "4"))
    cap = torch.cuda.is_current_stream_capturing()
    calls[(R, bool(dt), cap, kw.get("qc") is None)] += 1
    if ind is not None and R <= 4 and not cap and len(idx_stats) < 400:
        i = ind.detach().cpu()
        v = (i >= 0)
        nv = v.sum(1).tolist()
        # tail-only: every row is valid-prefix then -1 suffix
        tail = all(bool(v[r, :n].all()) and not bool(v[r, n:].any()) for r, n in enumerate(nv))
        idx_stats.append({"R": R, "valid": nv, "tail_only": tail, "pool_len": kw.get("pool_len"),
                          "sorted": all(bool((i[r, 1:n] >= i[r, :n - 1]).all()) for r, n in enumerate(nv))})
    return _orig(q, *a, **kw)
D.dsa_attn = wrapped

cfg = Config.from_directory(M)
model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(model, max_num_tokens=40960, max_history=1)
model.load(device="cuda:0", progressbar=False)
draft = Model.from_config(cfg, component="mtp")
dcache = Cache(draft, max_num_tokens=40960, max_history=1)
draft.load(device="cuda:0", progressbar=False)
GM.MTP_FUSE_CATCHUP = 2
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=1, record_draft_stats=True)
cids = G.corpus_ids(tok, 60000)
res = {}
for L in (4096, 32768):
    calls.clear(); idx_stats.clear()
    r = G.run_job(gen, cids[:, 1000:1000 + L].contiguous(), 24, stop=False)
    dec = [s for s in idx_stats if s["pool_len"] and s["R"] <= 4]
    res[L] = {"calls": {str(k): v for k, v in calls.items()},
              "n_idx": len(dec), "R_hist": dict(collections.Counter(s["R"] for s in dec)),
              "valid_min": min((min(s["valid"]) for s in dec), default=None),
              "valid_max": max((max(s["valid"]) for s in dec), default=None),
              "tail_only_all": all(s["tail_only"] for s in dec),
              "sorted_all": all(s["sorted"] for s in dec),
              "pool_len": sorted({s["pool_len"] for s in dec})[:4],
              "tps": (r["ntok"] - 1) / (r["total"] - r["ttft"])}
    print(f"[{L}] {json.dumps(res[L])}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", flush=True)
