"""ng_arms.py: n-gram gates in ONE load. (1) teacher-forced NLL with 1 / 4 / 8-row verify chunks and 1e-6 embedding-noise null arms (chunk size and noise are the null spread);
(2) served-style generate arms on repetitive text: MTP only, n-gram rows (shipped 4 / 8 gate), long-match-replaces-chain (HY4_NGRAM_LONG). Run inside job.sh."""
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
e = Hy4Engine(tree, GGUF, ctx=16384, k=3)
e.load(print)
hf, SG, GR = e.hf, e.SG, e.GR
V = C.Vocab(e.meta)
wiki = json.load(open("/path/to/wiki_ids.json"))
PRE, NN = 200, int(os.environ.get("NNLL", "240"))


def nll(R, seed=None):
    SG.PERT = None if seed is None else (torch.randn(GR.cap, e.M.D, generator=torch.Generator().manual_seed(seed)) * 1e-6).to(e.dev)
    GR.clear()
    e._t0 = time.perf_counter(); e.reset(); e.prefill(wiki[:PRE]); e.phase("decode")
    out, i = [], 0
    while i < NN:
        r = min(R, NN - i)
        lg, _am, _x, _h = GR.step_rows(wiki[PRE + i:PRE + i + r])
        lp = torch.log_softmax(lg.reshape(r, -1).float(), -1)
        tgt = torch.tensor(wiki[PRE + i + 1:PRE + i + r + 1], device=e.dev)
        out.extend((-lp[torch.arange(r, device=e.dev), tgt]).tolist())
        i += r
    SG.PERT = None
    GR.clear()          # captured graphs reference the noise table: drop them before it is freed
    return np.array(out)


arms = [("R4", 4, None), ("R8", 8, None), ("R1", 1, None), ("R4_n1", 4, 1), ("R4_n2", 4, 2), ("R8_n1", 8, 1), ("R8_n2", 8, 2)]
nl = {}
for name, R, sd in ([] if os.environ.get("SKIPNLL") else arms):
    nl[name] = nll(R, sd)
    print("NLL", name, round(float(nl[name].mean()), 5), flush=True)
a = nl.get("R4")
for k in (nl if a is not None else []):
    if k != "R4":
        d = nl[k] - a
        print("NLLD", k, "vs R4 mean", round(float(d.mean()), 5), "pct", round(100 * float(d.mean()) / float(a.mean()), 3), "se", round(float(d.std(ddof=1) / np.sqrt(len(d))), 5), flush=True)
nulls = [nl[k].mean() for k in ("R4", "R4_n1", "R4_n2", "R1")] if nl else [0]
if nl: print("NULL band (chunk4 + noise + chunk1) spread", round(float(max(nulls) - min(nulls)), 5), "; R8 arms", [round(float(nl[k].mean()), 5) for k in ("R8", "R8_n1", "R8_n2")], flush=True)

wiki_txt = V.decode(wiki[3000:3300])
CODE = open(os.path.join(tree, "tools/hy4/serve_core.py")).read()[:2600]
def chat(t):
    return V.encode(V.template.render([{"role": "user", "content": t}], None, True, reasoning_effort="no_think"))
prompts = {"quote_passage": chat("Here is a passage:\n\n" + wiki_txt + "\n\nQuote the three most important sentences verbatim, then summarize in two sentences."),
           "code_docstrings": chat("Add a short docstring to every function in this Python file and return the full file.\n\n" + CODE),
           "wiki_cont": wiki[1000:1200]}
cfgs = [("mtp", 0, 8, 0, 0), ("ng4", 4, 8, 1, 0), ("long8", 4, 8, 1, 8), ("long12", 4, 8, 1, 12), ("long12b", 4, 8, 1, 12), ("long8b", 4, 8, 1, 8), ("ng4b", 4, 8, 1, 0), ("mtpb", 0, 8, 0, 0)]      # ABBA order: mean of the two runs per arm
N = int(os.environ.get("BN", "200"))
for warm in (True, False):
    base = {}
    for pn, ids in prompts.items():
        for name, ext, mn, gate, lg in cfgs:
            e.ngram_ext, e.ngram_min, e.ngram_gate, e.ngram_long, e.ngram_fill, e.ngram_adapt = ext, mn, bool(gate), lg, False, False
            out = []
            st = e.generate(ids, N if not warm else 40, lambda t: out.extend(t) or False)
            if warm:
                continue
            base.setdefault(pn, out)
            same = next((i for i, (x, y) in enumerate(zip(out, base[pn])) if x != y), min(len(out), len(base[pn])))
            print("RES", json.dumps(dict(prompt=pn, cfg=name, tok_s=round(st["decode_tokens"] / st["decode_s"], 2), tau=round(st["tau"], 2), rounds=st["rounds"], same_prefix=same, ng=st["ngram"])), flush=True)
print("NG_DONE", flush=True)
