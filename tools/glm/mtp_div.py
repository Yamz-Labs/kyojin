# glm-mtp step 6: audit the R=2 MTP verify vs plain greedy divergences. One load.
# Per prompt (code 0-3, chat 0-3), greedy, 2K wiki context + task, --tokens new:
#   plain   : ndt 0, every target forward's logits recorded
#   n1      : ndt 1 (default fuse), R=2 verify forwards recorded (ids + logits per row)
#   perfect : n1 with the drafted id overwritten by plain's next token (oracle draft)
#   ref     : one no-cache forward (flash_attn_nc) over prompt + plain tokens (full-prefill reference)
# At the first plain/n1 divergence: plain, verify and ref top-2 margins, verify row index, KL over the span.
# Perfect: per verify row, KL(plain||row) and max|dlogit| up to its first divergence.
# Prints RESULT json per prompt and DONE; full data in --out.
import argparse, json, os, sys, time, traceback, torch
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="~/models/glm53-exl3-td205")
ap.add_argument("--corpus", default="~/bench/ppl/wiki.test.raw")
ap.add_argument("--out", default="scratch/mtp_div_6.json")
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--ctxs", default="2048,1024", help="wiki context lengths; >2048 prompt rows = DSA sparse")
ap.add_argument("--kinds", default="code,chat")
ap.add_argument("--prompts", default="0,1,2,3")
ap.add_argument("--no-ref", action="store_true")
ap.add_argument("--repeat", action="store_true", help="decode plain twice per prompt (noise floor)")
ap.add_argument("--diag", default="", help="mode:ctx:kind:prompt,... focused cross-row / determinism / idx-detail probe, then exit")
ap.add_argument("--modes", default="sparse", help="comma list: sparse (model default) / dense (index_topk -> 2^30 on every DSA layer)")
ap.add_argument("--envs", default="", help="step 9: ';'-list of label[:K=V,K=V] env arms, outermost loop, one load (e.g. 'def;dtr1:EXL3_DSA_DEC_DT_R=1')")
ap.add_argument("--idx", action="store_true", help="record DSA top-k selections, report plain vs perfect-draft row set differences")
args = ap.parse_args()
torch.set_grad_enabled(False)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exllamav3
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules import block_graph as BG
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
       "args": vars(args), "prompts": [], "errors": []}
from exllamav3.modules import mla_attn as _MA0
RES["mla_proj_r"] = _MA0.EXL3_MLA_PROJ_R; RES["gemv_r_mcg"] = _MA0.dec_gemv_r_mcg()
if _MA0.EXL3_MLA_PROJ_R == "gemv_r" and not RES["gemv_r_mcg"]: raise SystemExit("ERROR stale .so: gemv_r lacks mcg")
print("loaded", round(RES["load_s"], 1), "proj_r", RES["mla_proj_r"], "mcg", RES["gemv_r_mcg"], flush=True)
def save(): json.dump(RES, open(args.out, "w"), indent=1)

REC = [None]; IDX = [None]
LAY = {"armed": False, "on": False, "rec": []}   # per-module row-0 outputs of the first R=2 forward
def lay_wrap(m, name):
    f0 = m.forward
    def w(x, params, *a, **k):
        o = f0(x, params, *a, **k)
        if LAY["on"] and isinstance(o, torch.Tensor) and o.dim() >= 3 and o.shape[1] == 2:
            LAY["rec"].append((name, o[:, 0].float().cpu(), o[:, 1].float().cpu()))
        return o
    m.forward = w
_fwd = model.forward
def fwd(*a, **k):
    x = a[0] if a else k.get("input_ids")
    on = REC[0] is not None and x.shape[-1] <= 8 and not torch.cuda.is_current_stream_capturing()
    IDX[0] = [] if on and args.idx else None
    lay = LAY["armed"] and x.shape[-1] == 2 and not torch.cuda.is_current_stream_capturing()
    if lay: LAY["on"] = True; LAY["rec"] = []
    r = _fwd(*a, **k)
    if lay: LAY["on"] = False; LAY["armed"] = False
    if on:
        REC[0].append((x.flatten().clone().cpu(), r.float().reshape(x.shape[-1], -1).cpu(), IDX[0]))
    IDX[0] = None
    return r
from exllamav3.modules.mla_attn import MLAttention
def idx_wrap(fn):
    def w(self, *a, **k):
        r = fn(self, *a, **k)
        if IDX[0] is not None and not torch.cuda.is_current_stream_capturing(): IDX[0].append(r.clone().cpu())
        return r
    return w
MLAttention._indexer_topk = idx_wrap(MLAttention._indexer_topk)
MLAttention._indexer_topk_kpool = idx_wrap(MLAttention._indexer_topk_kpool)
DSA = [(at, at.index_topk) for mm in list(model.modules) + list(draft_model.modules)
       for at in [getattr(mm, "attn", None)] if isinstance(at, MLAttention) and at.index_topk]
if os.environ.get("MTP_DIV_LAYERS"):
    for i, mm in enumerate(model.modules):
        lay_wrap(mm, f"{i}:{mm.key}")
        for sub in ("attn", "mlp"):
            sm = getattr(mm, sub, None)
            if sm is not None and hasattr(sm, "forward"): lay_wrap(sm, f"{i}:{mm.key}.{sub}")
def set_mode(mode):
    for at, k0 in DSA: at.index_topk = (1 << 30) if mode == "dense" else k0
model.forward = fwd

GENS = {}
def gen(ndt):
    if ndt not in GENS:
        GENS.clear(); import gc; gc.collect()
        GENS[ndt] = (Generator(model=model, cache=cache, tokenizer=tok) if ndt == 0 else
                     Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft_model,
                               draft_cache=draft_cache, num_draft_tokens=ndt, record_draft_stats=True))
    return GENS[ndt]
CORPUS = tok.encode(open(args.corpus, encoding="utf-8").read()[:40000], add_bos=False)[0]
WIKI = [None]
def enc(p):
    p = f"Reference notes (ignore unless relevant):\n{WIKI[0]}\n\nTask: {p}"
    return tok.encode(f"[gMASK]<sop><|user|>\n{p}<|assistant|>\n", encode_special_tokens=True)
def run(ndt, ids, ntok, oracle=None):
    g = gen(ndt); REC[0] = []
    if oracle is not None:
        real = Generator.iterate_draftmodel_mtp_gen.__get__(g)
        def perfect(results):
            r = real(results)
            if r is not None and g.active_jobs:
                s = g.active_jobs[0].new_tokens
                if 0 <= s < len(oracle): r[0, 0] = oracle[s]   # view of draft_ids_pinned: in place
            return r
        g.iterate_draftmodel_mtp_gen = perfect
    try:
        job = Job(input_ids=ids, max_new_tokens=ntok, sampler=GreedySampler(), stop_conditions=[])
        g.enqueue(job); toks = []
        while g.num_remaining_jobs():
            for r in g.iterate():
                tid = r.get("token_ids")
                if tid is not None and tid.numel(): toks += tid.flatten().tolist()
    finally:
        rec, REC[0] = REC[0], None
        if oracle is not None: del g.iterate_draftmodel_mtp_gen
    return toks, rec, list(job.draft_stats) if ndt else []

def emitted(rec):
    """(token, logits row, row index in its forward) per emitted token; the prefill holds back the last
    prompt token, so the first decode forward emits toks[0] and emitted j == toks[j]."""
    out = []
    for f, (ids, lg, _) in enumerate(rec):
        am = lg.argmax(-1); out.append((int(am[0]), lg[0], 0, f))
        for i in range(1, len(ids)):
            if int(ids[i]) != int(am[i - 1]): break
            out.append((int(am[i]), lg[i], i, f))
    return out
def kl(p_lg, q_lg):   # KL(p||q), natural log
    lp, lq = p_lg.float().log_softmax(-1), q_lg.float().log_softmax(-1)
    return float((lp.exp() * (lp - lq)).sum())
def top2(lg):
    v = lg.topk(2); return v.indices.tolist(), round(float(v.values[0] - v.values[1]), 4)
def first_div(a, b):
    n = min(len(a), len(b)); return next((j for j in range(n) if a[j][0] != b[j][0]), None), n

def ref_logits(ids, toks):
    full = torch.cat([ids.view(1, -1), torch.tensor([toks[:-1]], dtype=torch.long)], dim=-1)
    out = model.forward(full, {"attn_mode": "flash_attn_nc"})
    out = out["logits"] if isinstance(out, dict) else out
    L = ids.shape[-1]
    r = out[0, L - 1:].float().cpu(); del out   # row i predicts toks[i] (= emitted i)
    return r

def diag(mode, ctx, kind, pi):
    set_mode(mode); WIKI[0] = tok.decode(CORPUS[:ctx])
    BG.purge(); ids = enc(PROMPTS[kind][pi]); L = int(ids.shape[-1])
    run(0, enc("Say hello."), 16); run(1, enc("Say hello."), 16)
    tp, rp, _ = run(0, ids, args.tokens)
    LAY["armed"] = True; ta, ra, _ = run(1, ids, args.tokens); la = LAY["rec"]
    tb, rb, _ = run(1, ids, args.tokens)
    LAY["armed"] = True; to, ro, _ = run(1, ids, args.tokens, oracle=tp); lo = LAY["rec"]; LAY["armed"] = False
    lays = []
    for (na, a0, a1), (no_, o0, o1) in zip(la, lo):
        lays.append([na, round(float((a0 - o0).abs().max()), 5), round(float((a1 - o1).abs().max()), 4)])
    first = next((x for x in lays if x[1] > 0), None)
    ep, ea, eb, eo = emitted(rp), emitted(ra), emitted(rb), emitted(ro)
    out = {"mode": mode, "ctx": ctx, "case": f"{kind}{pi}", "n_prompt": L,
           "fwd_rows": {k: [len(x[0]) for x in r[:4]] for k, r in (("plain", rp), ("n1a", ra), ("n1b", rb), ("oracle", ro))},
           "fwd0_ids": {k: r[0][0].tolist() for k, r in (("plain", rp), ("n1a", ra), ("oracle", ro))},
           "ids_eq": {"n1a_n1b": ta == tb, "plain_n1a": first_div(ep, ea)[0], "plain_oracle": first_div(ep, eo)[0], "n1a_n1b_div": first_div(ea, eb)[0]},
           "kl_n1a_n1b": [round(kl(ea[j][1], eb[j][1]), 6) for j in range(min(6, len(ea), len(eb)))],
           "kl_plain_n1a": [round(kl(ep[j][1], ea[j][1]), 6) for j in range(min(6, len(ep), len(ea)))],
           "kl_plain_oracle": [round(kl(ep[j][1], eo[j][1]), 6) for j in range(min(6, len(ep), len(eo)))],
           "kl_n1a_oracle_f0r0": round(kl(ra[0][1][0], ro[0][1][0]), 6),
           "dmax_n1a_oracle_f0r0": round(float((ra[0][1][0] - ro[0][1][0]).abs().max()), 4),
           "layers_n": [len(la), len(lo)], "layer_first_row0_diff": first,
           "layer_row0_diff": [x for x in lays if x[1] > 0][:12], "layer_names_head": [x[0] for x in lays[:6]]}
    if args.idx:   # j0 selection detail, plain vs n1a first forward row 0
        det = []
        for li, (a_, b_) in enumerate(zip(rp[0][2] or [], ra[0][2] or [])):
            sa = a_[0][a_[0] >= 0]; sb = b_[0][b_[0] >= 0]
            A, B = set(sa.tolist()), set(sb.tolist())
            if A != B or len(sa) != len(sb):
                det.append({"layer": li, "shape": [list(a_.shape), list(b_.shape)], "n": [len(sa), len(sb)], "uniq": [len(A), len(B)],
                            "only_plain": sorted(A - B)[:40], "only_n1": sorted(B - A)[:40]})
        out["idx_j0"] = det; out["idx_nlayers"] = [len(rp[0][2] or []), len(ra[0][2] or [])]
    RES.setdefault("diag", []).append(out); save()
    print("RESULT diag", json.dumps(out), flush=True)
if args.diag:
    for c in args.diag.split(","):
        m_, c_, k_, p_ = c.split(":")
        try: diag(m_, int(c_), k_, int(p_))
        except Exception:
            RES["errors"].append(f"diag {c}: " + traceback.format_exc()[-2500:]); save(); print("ERROR diag", c, traceback.format_exc()[-800:], flush=True)
    print("DONE errors", len(RES["errors"]), flush=True); sys.exit(0)
ENV0 = dict(os.environ)
def set_envs(spec):
    lab, _, kv = spec.partition(":")
    for k in [k for k in os.environ if k.startswith("EXL3_") and k not in ENV0]: del os.environ[k]
    for k, v in ENV0.items():
        if k.startswith("EXL3_"): os.environ[k] = v
    for a in filter(None, kv.split(",")): k, v = a.split("="); os.environ[k] = v
    return lab
def env_cases(e):   # "label[:K=V,...][@modes/ctxs]" restricts that arm's modes and ctxs
    e, _, r = e.partition("@"); ms, _, cs = r.partition("/")
    return [(e, m, int(c)) for m in (ms or args.modes).split(",") for c in (cs or args.ctxs).split(",")]
for envs, mode, ctx, kind, pi in [(e, m, c, k, int(i)) for e0 in (args.envs.split(";") if args.envs else [""]) for e, m, c in env_cases(e0) for k in args.kinds.split(",") for i in args.prompts.split(",")]:
    try:
        elab = set_envs(envs) if args.envs else ""
        set_mode(mode)
        WIKI[0] = tok.decode(CORPUS[:ctx])
        BG.purge(); ids = enc(PROMPTS[kind][pi])
        run(0, enc("Say hello."), 16); run(1, enc("Say hello."), 16)   # warm
        tp, rp, _ = run(0, ids, args.tokens)
        tv, rv, dsv = run(1, ids, args.tokens)
        to, ro, dso = run(1, ids, args.tokens, oracle=tp)
        if args.repeat:   # noise floor: the same plain decode twice in one process
            t2, r2, _ = run(0, ids, args.tokens); e1, e2 = emitted(rp), emitted(r2)
            d2, n2 = first_div(e1, e2); sp2 = range(d2 + 1 if d2 is not None else n2)
            RES.setdefault("repeat", []).append({"mode": mode, "kind": kind, "prompt": pi, "div": d2,
                "kl0": round(kl(e1[0][1], e2[0][1]), 5), "kl_mean": sum(kl(e1[j][1], e2[j][1]) for j in sp2) / len(sp2),
                "kl_max": max(kl(e1[j][1], e2[j][1]) for j in sp2), "dmax0": round(float((e1[0][1] - e2[0][1]).abs().max()), 4),
                "at_div": None if d2 is None else {"p1": top2(e1[d2][1]), "p2": top2(e2[d2][1])}})
            print("RESULT repeat", RES["repeat"][-1], flush=True)
        ep, ev, eo = emitted(rp), emitted(rv), emitted(ro)
        rep = {"env": elab, "kind": kind, "prompt": pi, "n_prompt": int(ids.shape[-1]),
               "mode": mode, "ctx": ctx, "replay_ok": [[t for t, *_ in e] == t_[:len(e)] for e, t_ in ((ep, tp), (ev, tv), (eo, to))],
               "prefix_v": next((j for j, (x, y) in enumerate(zip(tp, tv)) if x != y), None),
               "prefix_o": next((j for j, (x, y) in enumerate(zip(tp, to)) if x != y), None),
               "acc_v": sum(s[2] for s in dsv) / max(len(dsv), 1)}
        ref = None
        if not args.no_ref:
            try: ref = ref_logits(ids, tp)
            except Exception: RES["errors"].append(f"ref {kind}{pi}: " + traceback.format_exc()[-1500:]); print("ERROR ref", traceback.format_exc()[-600:], flush=True)
        def R(j): return ref[j] if ref is not None and j < ref.shape[0] else None
        d, n = first_div(ep, ev); rep["div_j"] = d; rep["div_tok"] = d
        span = range(d + 1 if d is not None else n)
        rep["kl_plain_verify"] = {"mean": sum(kl(ep[j][1], ev[j][1]) for j in span) / len(span), "max": max(kl(ep[j][1], ev[j][1]) for j in span)}
        if ref is not None:
            rep["kl_ref_plain"] = sum(kl(R(j), ep[j][1]) for j in span) / len(span)
            rep["kl_ref_verify"] = sum(kl(R(j), ev[j][1]) for j in span) / len(span)
            rep["ref_argmax_agree_plain"] = sum(int(R(j).argmax()) == ep[j][0] for j in span) / len(span)
        rep["max_dlogit_before"] = max([float((ep[j][1] - ev[j][1]).abs().max()) for j in range(d if d is not None else n)] or [0.0])
        if d is not None:
            pl, vl = ep[d][1], ev[d][1]; a, b = ep[d][0], ev[d][0]
            rep["at_div"] = {"plain_top2": top2(pl), "verify_top2": top2(vl), "verify_row": ev[d][2], "plain_row": ep[d][2],
                             "plain_gap_a_b": round(float(pl[a] - pl[b]), 4), "verify_gap_b_a": round(float(vl[b] - vl[a]), 4),
                             "kl_plain_verify": round(kl(pl, vl), 5), "max_dlogit": round(float((pl - vl).abs().max()), 4)}
            if R(d) is not None:
                rl = R(d); rep["at_div"].update(ref_top2=top2(rl), ref_gap_a_b=round(float(rl[a] - rl[b]), 4),
                                                kl_ref_plain=round(kl(rl, pl), 5), kl_ref_verify=round(kl(rl, vl), 5))
        # perfect draft: per verify row vs plain up to the oracle run's first divergence
        do, no = first_div(ep, eo); rep["perfect_div_j"] = do
        rows = {}
        for j in range(do if do is not None else no):
            r_ = eo[j][2]; rr = rows.setdefault(r_, {"n": 0, "kl": 0.0, "kl_max": 0.0, "dmax": 0.0, "kl_ref": 0.0})
            k_ = kl(ep[j][1], eo[j][1]); rr["n"] += 1; rr["kl"] += k_; rr["kl_max"] = max(rr["kl_max"], k_)
            rr["dmax"] = max(rr["dmax"], float((ep[j][1] - eo[j][1]).abs().max()))
            if R(j) is not None: rr["kl_ref"] += kl(R(j), eo[j][1])
        for rr in rows.values():
            rr["kl"] /= max(rr["n"], 1); rr["kl_ref"] /= max(rr["n"], 1)
        rep["perfect_rows"] = rows
        if args.idx:   # per token: [row, max over DSA layers of |plain set ^ verify-row set|, layers with any diff, n layers]
            sd = []
            for j in range(min((do if do is not None else no) + 1, len(eo))):
                pi_, vi_ = rp[ep[j][3]][2], ro[eo[j][3]][2]; r_ = eo[j][2]; ds = []
                for a_, b_ in zip(pi_ or [], vi_ or []):
                    sa = set(a_[0][a_[0] >= 0].tolist()); sb = set(b_[r_][b_[r_] >= 0].tolist()); ds.append(len(sa ^ sb))
                sd.append([r_, max(ds or [0]), sum(x > 0 for x in ds), len(ds), len(pi_ or []), len(vi_ or [])])
            rep["idx_symdiff"] = sd
        if do is not None:
            rep["perfect_at_div"] = {"plain_top2": top2(ep[do][1]), "verify_top2": top2(eo[do][1]), "verify_row": eo[do][2],
                                     "ref_top2": top2(R(do)) if R(do) is not None else None}
        # plain's smallest margin in the span, and the per-row KL of the real n1 run
        rep["min_plain_margin_before"] = min([top2(ep[j][1])[1] for j in range(d if d is not None else n)] or [None])
        rv_rows = {}
        for j in span:
            rr = rv_rows.setdefault(ev[j][2], [0, 0.0]); rr[0] += 1; rr[1] += kl(ep[j][1], ev[j][1])
        rep["n1_rows_kl"] = {k: [c, s / c] for k, (c, s) in rv_rows.items()}
        rep["series_n1"] = [[ev[j][2], round(kl(ep[j][1], ev[j][1]), 5), round(top2(ep[j][1])[1], 3)] + ([round(kl(R(j), ep[j][1]), 5), round(kl(R(j), ev[j][1]), 5)] if R(j) is not None else []) for j in span]
        RES["prompts"].append(rep); save()
        print("RESULT", json.dumps({k: rep.get(k) for k in ("env", "mode", "ctx", "kind", "prompt", "prefix_v", "prefix_o", "div_j", "max_dlogit_before", "acc_v")}), flush=True)
    except Exception:
        RES["errors"].append(f"{mode} {ctx} {kind}{pi}: " + traceback.format_exc()[-2500:]); save()
        print("ERROR", mode, ctx, kind, pi, traceback.format_exc()[-800:], flush=True)
print("DONE errors", len(RES["errors"]), flush=True)
