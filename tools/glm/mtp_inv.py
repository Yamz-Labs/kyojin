# glm-mtp step 7: batch-invariance probe. One load. For each case (ctx:kind:prompt), run the plain
# generator (R=1 first decode forward) and the MTP n1 generator (R=2 first verify forward, ids
# [last prompt token, draft]) from the same prefilled cache state, record every hooked op's output
# (Linear, norms, MLA/DSA kernels, block/attn/mlp outputs) during that one target forward, and
# report per op whether row 0 of the R=2 pass equals the R=1 pass bitwise (first differing op in
# execution order + a list). Graphs off for the probe (hooks must run eagerly): EXL3_BLOCK_GRAPH=0,
# bc_mla_step patched out, so the eager _attend path = the kernels BLOCK_GRAPH_MLA=2 replays.
# Prints RESULT json per case and DONE.
import argparse, json, os, sys, time, traceback, torch
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="~/models/glm53-exl3-td205")
ap.add_argument("--corpus", default="~/bench/ppl/wiki.test.raw")
ap.add_argument("--out", default="scratch/mtp_inv_7.json")
ap.add_argument("--cases", default="2048:code:1,1024:chat:3")
ap.add_argument("--graph", action="store_true", help="keep graphs (no op hooks; logits row-0 check only)")
args = ap.parse_args()
torch.set_grad_enabled(False)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exllamav3
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules import block_graph as BG
from exllamav3.modules import Linear, RMSNorm
from exllamav3.modules.layernorm import LayerNorm
from exllamav3.modules import mla_attn as MA
from exllamav3.modules.attention_fn import dsa_triton as DT
from exllamav3.ext import exllamav3_ext as ext
from mtp_bench_prompts import PROMPTS

os.environ.setdefault("EXL3_DEC_MOE_UNION", "1"); os.environ.setdefault("EXL3_DEC_MOE_UNION_DEV", "1")
t0 = time.perf_counter()
config = Config.from_directory(args.model)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=4096, max_history=3)
model.load(device="cuda:0", progressbar=False)
draft_model = Model.from_config(config, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=4096, max_history=3)
draft_model.load(device="cuda:0", progressbar=False)
RES = {"env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}, "load_s": time.perf_counter() - t0,
       "args": vars(args), "cases": [], "errors": []}
print("loaded", round(RES["load_s"], 1), flush=True)
def save(): json.dump(RES, open(args.out, "w"), indent=1)

CAP = {"armed": False, "on": False, "rec": [], "logits": None, "cnt": {}}
def cap(name, t):
    if not CAP["on"] or not isinstance(t, torch.Tensor) or torch.cuda.is_current_stream_capturing(): return
    k = CAP["cnt"].get(name, 0); CAP["cnt"][name] = k + 1
    CAP["rec"].append((f"{name}#{k}", t.detach().clone().cpu()))
def hook_method(cls, meth, label):
    f0 = getattr(cls, meth)
    def w(self, *a, **k):
        o = f0(self, *a, **k)
        if CAP["on"]:
            o0 = o[0] if isinstance(o, tuple) else o
            cap(f"{label(self)}", o0)
        return o
    setattr(cls, meth, w)
def hook_fn(mod, name):
    f0 = getattr(mod, name)
    def w(*a, **k):
        o = f0(*a, **k)
        cap(name, o[0] if isinstance(o, tuple) else o)
        return o
    setattr(mod, name, w)
if not args.graph:
    MA.MLAttention.bc_mla_step = lambda self, *a, **k: None
    hook_method(Linear, "forward", lambda s: s.key)
    hook_method(RMSNorm, "forward", lambda s: s.key)
    hook_method(LayerNorm, "forward", lambda s: s.key)
    for n in ("mla_attn_triton_decode", "mla_absorb", "mla_unfold"): hook_fn(MA, n)
    for n in ("dsa_indexer_scores", "dsa_attn"): hook_fn(DT, n)
    hook_method(MA.MLAttention, "_indexer_keys", lambda s: f"L{s.layer_idx}.idx_keys")
    hook_method(MA.MLAttention, "_indexer_topk_kpool", lambda s: f"L{s.layer_idx}.idx_sel")
    hook_method(MA.MLAttention, "_indexer_topk", lambda s: f"L{s.layer_idx}.idx_sel")
    _topk = ext.dsa_topk
    def topk_w(sc, out, *a, **k):
        r = _topk(sc, out, *a, **k); cap("dsa_topk", out); return r
    ext.dsa_topk = topk_w
    def wrap_obj(m, name):
        f0 = m.forward
        def w(*a, **k):
            o = f0(*a, **k); cap(name, o[0] if isinstance(o, tuple) else o); return o
        m.forward = w
    for i, mm in enumerate(model.modules):
        wrap_obj(mm, f"{i}:{mm.key}")
        for sub in ("attn", "mlp"):
            sm = getattr(mm, sub, None)
            if sm is not None and hasattr(sm, "forward"): wrap_obj(sm, f"{i}:{mm.key}.{sub}")

_fwd = model.forward
def fwd(*a, **k):
    x = a[0] if a else k.get("input_ids")
    arm = CAP["armed"] and x.shape[-1] <= 2 and not torch.cuda.is_current_stream_capturing()
    if arm: CAP["on"] = True; CAP["rec"] = []; CAP["cnt"] = {}
    try: r = _fwd(*a, **k)
    finally:
        if arm: CAP["on"] = False; CAP["armed"] = False
    if arm:
        lg = r["logits"] if isinstance(r, dict) else r
        CAP["logits"] = lg.float().reshape(x.shape[-1], -1).cpu()
    return r
model.forward = fwd

GENS = {}
def gen(ndt):
    if ndt not in GENS:
        GENS.clear(); import gc; gc.collect()
        GENS[ndt] = (Generator(model=model, cache=cache, tokenizer=tok) if ndt == 0 else
                     Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft_model,
                               draft_cache=draft_cache, num_draft_tokens=ndt))
    return GENS[ndt]
CORPUS = tok.encode(open(args.corpus, encoding="utf-8").read()[:40000], add_bos=False)[0]
def enc(p, wiki):
    p = f"Reference notes (ignore unless relevant):\n{wiki}\n\nTask: {p}"
    return tok.encode(f"[gMASK]<sop><|user|>\n{p}<|assistant|>\n", encode_special_tokens=True)
def run(ndt, ids, ntok, armed):
    g = gen(ndt); CAP["armed"] = armed
    job = Job(input_ids=ids, max_new_tokens=ntok, sampler=GreedySampler(), stop_conditions=[])
    g.enqueue(job)
    while g.num_remaining_jobs():
        for _ in g.iterate(): pass
    CAP["armed"] = False
    return list(CAP["rec"]), CAP["logits"]

def row0(a, b):
    """b's row-0 slice matching a (R=1) along the first dim where shapes differ."""
    if a.shape == b.shape: return b
    if a.dim() != b.dim():   # e.g. (1, X) vs (1, 2, X): compare flattened rows
        if a.shape[-1] != b.shape[-1]: return None
        a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        return b2[: a2.shape[0]].reshape(a.shape) if b2.shape[0] == 2 * a2.shape[0] else None
    for d in range(a.dim()):
        if a.shape[d] != b.shape[d]:
            if b.shape[d] == 2 * a.shape[d] or (a.shape[d] == 1 and b.shape[d] == 2):
                bb = b.narrow(d, 0, a.shape[d])
                return bb if bb.shape == a.shape else None
            return None
    return None
def compare(ra, rb):
    db = dict(rb); out = []
    for name, a in ra:
        b = db.get(name)
        if b is None: out.append([name, "missing"]); continue
        if "topk" in name or "idx_sel" in name or a.dtype in (torch.int32, torch.int64):
            b0 = row0(a, b)
            if b0 is None:   # score width can differ by one column: compare row 0 sets
                sa = set(a.reshape(a.shape[0], -1)[0].tolist()) if a.dim() else set()
                sb = set(b.reshape(b.shape[0], -1)[0].tolist())
                out.append([name, "set", len(sa ^ sb)]); continue
            sa, sb = set(a.flatten().tolist()), set(b0.flatten().tolist())
            out.append([name, "set", len(sa ^ sb), bool(torch.equal(a, b0))]); continue
        b0 = row0(a, b)
        if b0 is None and a.dim() == 2 and b.dim() == 2 and a.shape[0] == 1 and b.shape[0] == 2:
            w = min(a.shape[1], b.shape[1]); a, b0 = a[:, :w], b[:1, :w]   # scores: width may differ
        if b0 is None: out.append([name, "shape", list(a.shape), list(b.shape)]); continue
        af, bf = a.float(), b0.float()
        fin = torch.isfinite(af) & torch.isfinite(bf)
        d = float((af[fin] - bf[fin]).abs().max()) if fin.any() else 0.0
        eq = bool(torch.equal(torch.nan_to_num(af, 0, 1e30, -1e30), torch.nan_to_num(bf, 0, 1e30, -1e30)))
        out.append([name, "eq" if eq else "diff", d])
    return out

for c in args.cases.split(","):
    ctx, kind, pi = c.split(":"); ctx, pi = int(ctx), int(pi)
    try:
        BG.purge(); wiki = tok.decode(CORPUS[:ctx]); ids = enc(PROMPTS[kind][pi], wiki)
        run(0, enc("Say hello.", ""), 4, False); run(1, enc("Say hello.", ""), 4, False)   # warm
        ra, la = run(0, ids, 2, True)
        rb, lb = run(1, ids, 2, True)
        ra2, la2 = run(0, ids, 2, True)   # determinism of plain
        cmp = compare(ra, rb) if ra else []
        det = compare(ra, ra2) if ra else []
        diffs = [x for x in cmp if x[1] not in ("eq",) and not (x[1] == "set" and x[2] == 0 and (len(x) < 4 or x[3]))]
        dl = float((la[0] - lb[0]).abs().max())
        out = {"case": c, "n_prompt": int(ids.shape[-1]), "n_ops": [len(ra), len(rb)],
               "logit_row0_dmax": round(dl, 5), "logit_row0_eq": bool(torch.equal(la[0], lb[0])),
               "plain_repeat_eq": bool(torch.equal(la, la2)), "det_diffs": [x for x in det if x[1] != "eq"][:5],
               "first_diff": diffs[0] if diffs else None, "n_diff": len(diffs), "diffs": diffs[:60]}
        RES["cases"].append(out); save()
        print("RESULT", json.dumps({k: out[k] for k in ("case", "n_ops", "logit_row0_dmax", "logit_row0_eq", "plain_repeat_eq", "first_diff", "n_diff")}), flush=True)
    except Exception:
        RES["errors"].append(f"{c}: " + traceback.format_exc()[-2500:]); save()
        print("ERROR", c, traceback.format_exc()[-800:], flush=True)
print("DONE errors", len(RES["errors"]), flush=True)
