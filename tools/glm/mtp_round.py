# glm-mtp step 5: split one MTP round (ndt 1, R=2 verify) into its parts with synced wall timers,
# and A/B the fused catch-up (EXL3_MTP_FUSE_CATCHUP) against the old catch-up prefill. One load.
#   free arms  : unsynced wall t/s (plain, n1 = fuse 0, n1f1, n1f2), interleaved A B B A
#   split arms : torch.cuda.synchronize() around each part (inflates the round by the sync gaps);
#                routed-MoE kernel ms by CUDA events (verify/plain eager), unique experts per call,
#                host->device sync count per round
# Prints one RESULT json line per arm and a final DONE line.
import argparse, json, os, sys, time, collections, traceback, torch
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="~/models/glm53-exl3-td205")
ap.add_argument("--corpus", default="~/bench/ppl/wiki.test.raw")
ap.add_argument("--out", default="scratch/mtp_round_5.json")
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--kinds", default="code,chat")
ap.add_argument("--prompts", default="0", help="comma list of prompt indices per kind")
ap.add_argument("--prof", default="", help="step 9: arm to profile the draft forward under torch.profiler after the arms (e.g. n1)")
ap.add_argument("--arms", default="plain,n1,n1f1,n1f2,n1f2,n1f1,n1,plain,plain_split,n1_split,n1f1_split,n1f2_split")
args = ap.parse_args()
torch.set_grad_enabled(False)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exllamav3
from exllamav3.ext import exllamav3_ext
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator import generator as GM
from exllamav3.generator import job as JM
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules import block_graph as BG
from mtp_bench_prompts import PROMPTS

t0 = time.perf_counter()
config = Config.from_directory(args.model)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=4096, max_history=3)
model.load(device="cuda:0", progressbar=False)
draft_model = Model.from_config(config, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=4096, max_history=3)
draft_model.load(device="cuda:0", progressbar=False)
RES = {"env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}, "load_s": time.perf_counter() - t0,
       "args": vars(args), "arms": [], "errors": []}
from exllamav3.modules import mla_attn as _MA0
RES["mla_proj_r"] = _MA0.EXL3_MLA_PROJ_R; RES["gemv_r_mcg"] = _MA0.dec_gemv_r_mcg()
if _MA0.EXL3_MLA_PROJ_R == "gemv_r" and not RES["gemv_r_mcg"]: raise SystemExit("ERROR stale .so: gemv_r lacks mcg")
print("loaded", round(RES["load_s"], 1), "proj_r", RES["mla_proj_r"], "mcg", RES["gemv_r_mcg"], flush=True)
def save(): json.dump(RES, open(args.out, "w"), indent=1)

# ---- synced split instrumentation -----------------------------------------------------------
ON = [False]                    # split timers live (decode only, split arms only)
T = collections.defaultdict(float); N = collections.defaultdict(int)
CTX = ["-"]                     # which part is running, for kernel events
KEV = []                        # (ctx, kernel, start ev, end ev, sel)
SYNC = collections.Counter()
_S = torch.cuda.synchronize

def st(key_fn, fn, ctx=None):
    def w(*a, **k):
        if not ON[0]:
            return fn(*a, **k)
        key = key_fn(a, k)
        _S(); h = time.perf_counter()
        prev = CTX[0]
        if ctx: CTX[0] = ctx if isinstance(ctx, str) else ctx(a, k)
        try: r = fn(*a, **k)
        finally: CTX[0] = prev
        _S(); T[key] += 1000 * (time.perf_counter() - h); N[key] += 1
        return r
    return w
def K(n): return lambda a, k: n

G = Generator
G.iterate = st(K("round"), G.iterate)
G.iterate_draftmodel_mtp_gen = st(K("draft_gen"), G.iterate_draftmodel_mtp_gen)
G.iterate_gen = st(K("gen"), G.iterate_gen)
def _rows(a, k):
    x = a[0] if a else k.get("input_ids"); return x.shape[-1]
draft_model.forward = st(lambda a, k: f"dfwd_R{_rows(a, k)}", draft_model.forward, "dfwd")
draft_model.prefill = st(lambda a, k: f"dprefill_R{_rows(a, k)}", draft_model.prefill, "dprefill")
draft_model.sample_from_state = st(K("dsample"), draft_model.sample_from_state, "dsample")
model.forward = st(lambda a, k: f"tfwd_R{_rows(a, k)}", model.forward, lambda a, k: f"tfwd_R{_rows(a, k)}")
Job.receive_logits = st(K("sample"), Job.receive_logits)
Job.receive_sample = st(K("sample"), Job.receive_sample)

def kev(name, fn):
    def w(*a, **k):
        if not (ON[0] or MOD[0]) or torch.cuda.is_current_stream_capturing():
            return fn(*a, **k)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = fn(*a, **k); e.record()
        KEV.append((CTX[0], name, s, e, a[2].shape[0], a[2].clone()))
        return r
    return w
exllamav3_ext.exl3_dec_moe_union = kev("moe_union", exllamav3_ext.exl3_dec_moe_union)
exllamav3_ext.exl3_dec_moe = kev("moe", exllamav3_ext.exl3_dec_moe)

# ---- step 10: per-module synced split (arms *_mod, block graphs off: every module visible) ------------
# Exclusive synced wall per module bucket, keyed "<fwd>:<bucket>" with fwd = R<rows> (target forward)
# or D<rows> (draft forward); plus CUDA-event kernel ms of the router / projection GEMV launches keyed by
# the enclosing bucket. Sync per wrapper inflates each bucket by ~one launch+sync latency, equally at
# R1 and R2, so R2 - R1 per bucket is the verify cost gap.
MOD = [False]; MT = collections.defaultdict(float); MN = collections.defaultdict(int); STK = []
FWD = ["-"]; GEV = []
def mt(keyf, fn, fwd=None):
    def w(*a, **k):
        if not MOD[0] or torch.cuda.is_current_stream_capturing():
            return fn(*a, **k)
        pf = FWD[0]
        if fwd: FWD[0] = fwd(a, k)
        key = f"{FWD[0]}:{keyf}"
        prev = CTX[0]; CTX[0] = keyf
        _S(); h = time.perf_counter(); STK.append(0.0)
        try: r = fn(*a, **k)
        finally:
            _S(); dt = 1000 * (time.perf_counter() - h); ch = STK.pop()
            MT[key] += dt - ch; MN[key] += 1
            if STK: STK[-1] += dt
            CTX[0] = prev; FWD[0] = pf
        return r
    return w
def gev(name, fn):
    def w(*a, **k):
        if not MOD[0] or torch.cuda.is_current_stream_capturing():
            return fn(*a, **k)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = fn(*a, **k); e.record(); GEV.append((f"{FWD[0]}:{CTX[0]}:{name}", s, e))
        return r
    return w
for _n in ("exl3_dec_router", "exl3_dec_router_norm", "exl3_dec_gemv", "exl3_dec_gemv_r", "exl3_dec_gemv_r_multi",
           "exl3_dec_gemv_strided", "exl3_dec_moe_union", "exl3_dec_moe"):
    if hasattr(exllamav3_ext, _n): setattr(exllamav3_ext, _n, gev(_n[5:], getattr(exllamav3_ext, _n)))
from exllamav3.modules.transformer import TransformerBlock as _TB
from exllamav3.modules.gated_delta_net import GatedDeltaNet as _GDN
from exllamav3.modules.mla_attn import MLAttention as _MLA
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP as _BSM
def _wrap(obj, meth, key):
    f = getattr(obj, meth, None)
    if f is not None: setattr(obj, meth, mt(key, f))
def _inst_mod(mdl, tag):
    for m in mdl.modules:
        if isinstance(m, _TB):
            at = m.attn
            ak = "kda" if isinstance(at, _GDN) else ("dsa" if getattr(at, "indexer_mode", None) else "mla") if isinstance(at, _MLA) else "attn"
            if at is not None:
                _wrap(at, "forward", ak)
                if isinstance(at, _MLA):
                    _wrap(at, "_attend_pre", ak + "_proj"); _wrap(at, "attend_post", ak + "_oproj")
            if m.mlp is not None:
                _wrap(m.mlp, "forward", "moe" if isinstance(m.mlp, _BSM) else "mlp")
                _wrap(m.mlp, "dec_norm_route_r", "norm_route"); _wrap(m.mlp, "dec_norm_route", "norm_route")
            for n in ("attn_norm", "mlp_norm"):
                if getattr(m, n, None) is not None: _wrap(getattr(m, n), "forward", "norm")
            for n in ("attn_hc", "mlp_hc"):
                h = getattr(m, n, None)
                if h is not None:
                    for mm in ("mix", "mix_norm", "apply_"): _wrap(h, mm, "hc")
            _wrap(m, "forward", "blk_glue")
        else:
            nm = type(m).__name__.lower()
            _wrap(m, "forward", "lm_head" if (tag == "R" and m is mdl.modules[-1]) else nm)
_inst_mod(model, "R"); _inst_mod(draft_model, "D")
model.forward = mt("fwd_host", model.forward, lambda a, k: f"R{_rows(a, k)}")
draft_model.forward = mt("fwd_host", draft_model.forward, lambda a, k: f"D{_rows(a, k)}")
_lmh = model.modules[model.logit_layer_idx]

# device->host sync points (counted, not timed); installed only while a split arm runs
_item, _cpu, _tolist, _copy, _sync = torch.Tensor.item, torch.Tensor.cpu, torch.Tensor.tolist, torch.Tensor.copy_, torch.cuda.synchronize
def _cnt(name, f, test):
    def w(*a, **k):
        if ON[0] and test(*a, **k): SYNC[name] += 1
        return f(*a, **k)
    return w
def sync_count(on):
    if on:
        torch.Tensor.item = _cnt("item", _item, lambda s: s.is_cuda)
        torch.Tensor.cpu = _cnt("cpu", _cpu, lambda s, *a, **k: s.is_cuda)
        torch.Tensor.tolist = _cnt("tolist", _tolist, lambda s: s.is_cuda)
        torch.Tensor.copy_ = _cnt("d2h_copy", _copy, lambda s, src, *a, **k: (not s.is_cuda) and src.is_cuda and not k.get("non_blocking", a[0] if a else False))
        torch.cuda.synchronize = _cnt("synchronize", _sync, lambda *a, **k: True)
    else:
        torch.Tensor.item, torch.Tensor.cpu, torch.Tensor.tolist, torch.Tensor.copy_ = _item, _cpu, _tolist, _copy
        torch.cuda.synchronize = _sync

# ---- runs -------------------------------------------------------------------------------------
GENS = {}
def gen(ndt):
    if ndt not in GENS:
        GENS.clear(); import gc; gc.collect()
        GENS[ndt] = (Generator(model=model, cache=cache, tokenizer=tok) if ndt == 0 else
                     Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft_model,
                               draft_cache=draft_cache, num_draft_tokens=ndt, record_draft_stats=True))
    return GENS[ndt]
WIKI = tok.decode(tok.encode(open(args.corpus, encoding="utf-8").read()[:40000], add_bos=False)[0, :args.ctx])
def run(ndt, prompt, ntok, split):
    g = gen(ndt)
    prompt = f"Reference notes (ignore unless relevant):\n{WIKI}\n\nTask: {prompt}"
    ids = tok.encode(f"[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n", encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=ntok, sampler=GreedySampler(), stop_conditions=[])
    g.enqueue(job); t0 = time.perf_counter(); ttft = None; toks = []
    while g.num_remaining_jobs():
        for r in g.iterate():
            tid = r.get("token_ids")
            if tid is not None and tid.numel():
                if ttft is None:
                    _sync(); ttft = time.perf_counter() - t0; ON[0] = split
                toks += tid.flatten().tolist() if not ON[0] else _tolist(tid.flatten())
    ON[0] = False; _sync(); dec = time.perf_counter() - t0 - ttft
    st_ = list(job.draft_stats)
    rounds = len(st_) if st_ else len(toks) - 1
    acc = sum(s[2] for s in st_) / max(len(st_), 1)
    return {"ndt": ndt, "ntok": len(toks), "rounds": rounds, "tps": (len(toks) - 1) / dec,
            "round_ms": 1000 * dec / max(rounds, 1), "acc": acc, "tok_per_round": 1 + acc if st_ else 1.0, "toks": toks}

# arm -> (ndt, fuse, split, graph for plain)
ARMS = {"plain": (0, 0, False), "n1": (1, 0, False), "n1f1": (1, 1, False), "n1f2": (1, 2, False),
        "plain_split": (0, 0, True), "n1_split": (1, 0, True), "n1f1_split": (1, 1, True), "n1f2_split": (1, 2, True)}
# step 7/8 verify batch-invariance: plain arm names = library default (projections via gemv_r, attention
# row loop); "<arm>_ni" = both off (EXL3_MLA_DEC_ROWLOOP_MAX=0), "_np" = projection half off (M-row route),
# "_pl" = step 7's projection loop (R batch-1 launches), "_na" = attention loop off only
for _k in list(ARMS):
    for _s in ("_ni", "_np", "_pl", "_na"): ARMS[_k + _s] = ARMS[_k]
from exllamav3.modules import mla_attn as _MA
_DG0 = BG.BLOCK_GRAPH_DRAFT; MODARM = [False]
_INV_MAX = _MA.EXL3_MLA_DEC_ROWLOOP_MAX
for _k in ("plain", "n1f2"): ARMS[_k + "_mod"] = ARMS[_k]
ARMS["plain_eg"] = ARMS["plain"]
from exllamav3.modules.transformer import TransformerBlock as _TB2
_DBLK = next(m for m in draft_model.modules if isinstance(m, _TB2))
def _drunner(): return getattr(_DBLK, "block_graph_runner", None)
DCHK = {"n": 0, "replayed": 0, "eq": 0, "dmax": 0.0}
_f_draft = [None]
def dchk_forward(*a, **k):
    # step 10 gate: same draft forward eager (graphs off) then graphed, same cache position (the
    # eager KV/indexer write is rewritten identically); compare the lm_head logits bitwise. Only
    # forwards where the draft block replayed a graph count.
    ids, params = a[0], a[1]
    BG.BLOCK_GRAPH_ENABLED = False
    try: se = _f_draft[0](ids, dict(params)).clone()
    finally: BG.BLOCK_GRAPH_ENABLED = True
    r = _drunner(); r0 = r.stats["replays"] if r else 0
    out = _f_draft[0](ids, params)
    r = _drunner(); r1 = r.stats["replays"] if r else 0
    DCHK["n"] += 1
    if r1 > r0:
        le = _lmh.forward(_lmh.prepare_for_device(se, {}), {}); lg = _lmh.forward(_lmh.prepare_for_device(out, {}), {})
        DCHK["replayed"] += 1; DCHK["eq"] += int(torch.equal(le, lg))
        DCHK["dmax"] = max(DCHK["dmax"], float((le.float() - lg.float()).abs().max()))
    return out
def setarm(name):
    name, *mods = name.split("+")
    ndt, fuse, split = ARMS[name]
    _MA.EXL3_MLA_DEC_ROWLOOP_MAX = 0 if name.endswith("_ni") else _INV_MAX
    _MA.EXL3_MLA_PROJ_R = "off" if name.endswith("_np") else "loop" if name.endswith("_pl") else "gemv_r"
    _MA.EXL3_MLA_ATTN_ROWLOOP = not name.endswith("_na")
    os.environ["EXL3_DEC_MOE_UNION"] = "1"; os.environ["EXL3_DEC_MOE_UNION_DEV"] = "0"
    os.environ["EXL3_MOE_UNION_V2"] = "0"  # arms stay explicit: +v2 opts in
    BG.BLOCK_GRAPH_VERIFY = False
    # plain_split runs eager so its MoE kernels are visible to the events (graph gain ~1 ms, step 3)
    BG.BLOCK_GRAPH_ENABLED = not (split and ndt == 0) and not name.endswith("_mod") and name != "plain_eg"
    GM.MTP_FUSE_CATCHUP = fuse
    BG.BLOCK_GRAPH_DRAFT = _DG0
    for m in mods:
        if m.startswith("dg"): BG.BLOCK_GRAPH_DRAFT = int(m[2:])
        if m == "v2": os.environ["EXL3_MOE_UNION_V2"] = "1"
    if _f_draft[0] is not None: draft_model.forward = _f_draft[0]; _f_draft[0] = None
    if "dchk" in mods:
        _f_draft[0] = draft_model.forward; draft_model.forward = dchk_forward
        DCHK.update(n=0, replayed=0, eq=0, dmax=0.0)
    MOD[0] = False; MODARM[0] = name.endswith("_mod")
    BG.purge()

base = {}
for kind, pi in [(k, int(i)) for k in args.kinds.split(",") for i in args.prompts.split(",")]:
    p = PROMPTS[kind][pi]
    for name in args.arms.split(","):
        try:
            setarm(name); ndt, fuse, split = ARMS[name.split("+")[0]]
            run(ndt, "Say hello.", 24, False)       # warm (fresh graphs per arm)
            T.clear(); N.clear(); KEV.clear(); SYNC.clear(); MT.clear(); MN.clear(); GEV.clear()
            DCHK.update(n=0, replayed=0, eq=0, dmax=0.0)
            sync_count(split)
            MOD[0] = MODARM[0]
            try: r = run(ndt, p, args.tokens, split)
            finally: sync_count(False); MOD[0] = False
            toks = r.pop("toks")
            if ndt == 0 and (kind, pi) not in base: base[(kind, pi)] = toks
            if (kind, pi) in base:
                b = base[(kind, pi)]
                r["prefix_vs_plain"] = next((j for j, (x, y) in enumerate(zip(b, toks)) if x != y), min(len(b), len(toks)))
            if (kind, pi, "n1") in base:
                b = base[(kind, pi, "n1")]
                r["prefix_vs_n1"] = next((j for j, (x, y) in enumerate(zip(b, toks)) if x != y), min(len(b), len(toks)))
            if name == "n1": base.setdefault((kind, pi, "n1"), toks)
            r.update(kind=kind, prompt=pi, arm=name, toks=toks)
            if split:
                nr = max(N["round"], 1)
                r["ms_per_round"] = {k: round(T[k] / nr, 3) for k in T}
                r["calls_per_round"] = {k: round(N[k] / nr, 3) for k in N}
                r["ms_per_call"] = {k: round(T[k] / N[k], 3) for k in T}
                r["syncs_per_round"] = {k: round(v / nr, 2) for k, v in SYNC.items()}
                kk = collections.defaultdict(list); uq = collections.defaultdict(list)
                for ctx, kn, s, e, R, sel in KEV:
                    kk[f"{ctx}:{kn}:R{R}"].append(s.elapsed_time(e)); uq[f"{ctx}:{kn}:R{R}"].append(int(torch.unique(sel).numel()))
                r["moe_kernel"] = {k: {"calls_per_round": round(len(v) / nr, 2), "ms_per_round": round(sum(v) / nr, 3),
                                       "us_per_call": round(1000 * sum(v) / len(v), 1), "uniq": round(sum(uq[k]) / len(uq[k]), 2)}
                                   for k, v in kk.items()}
            dr = _drunner()
            r["draft_graph"] = {"knob": BG.BLOCK_GRAPH_DRAFT, "captures": dr.stats["captures"], "replays": dr.stats["replays"],
                                "declines": dict(dr.stats["declines"])} if dr else {"knob": BG.BLOCK_GRAPH_DRAFT, "runner": None}
            if "dchk" in name: r["dchk"] = dict(DCHK)
            if MT:
                _S(); nr = max(r["rounds"], 1)
                r["mod_ms_per_round"] = {k: round(v / nr, 3) for k, v in sorted(MT.items())}
                r["mod_calls_per_round"] = {k: round(v / nr, 2) for k, v in sorted(MN.items())}
                g_ = collections.defaultdict(float)
                for kk_, s_, e_ in GEV: g_[kk_] += s_.elapsed_time(e_)
                r["gev_ms_per_round"] = {k: round(v / nr, 3) for k, v in sorted(g_.items())}
            RES["arms"].append(r); save()
            print("RESULT", json.dumps({k: r.get(k) for k in ("kind", "prompt", "arm", "tps", "round_ms", "tok_per_round", "prefix_vs_plain", "prefix_vs_n1",
                                                               "ms_per_round", "calls_per_round", "syncs_per_round", "moe_kernel", "draft_graph", "dchk")}), flush=True)
        except Exception:
            ON[0] = False
            RES["errors"].append(f"{kind} {name}: " + traceback.format_exc()[-2500:]); save()
            print("ERROR", kind, name, traceback.format_exc()[-800:], flush=True)
# step 9: draft-head GPU busy vs wall. record_function around each draft forward; kernels of its
# descendants summed per name. busy << wall = host/launch cost, busy ~= wall = kernel time.
if args.prof:
    try:
        from torch.profiler import profile, ProfilerActivity, record_function
        setarm(args.prof); ndt = ARMS[args.prof][0]
        run(ndt, "Say hello.", 24, False)
        f_d = draft_model.forward
        def f_p(*a, **k):
            with record_function("DFWD_R%d" % _rows(a, k)): return f_d(*a, **k)
        draft_model.forward = f_p
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run(ndt, PROMPTS["code"][0], args.tokens, False)
            torch.cuda.synchronize()
        draft_model.forward = f_d
        ev = prof.events(); P = {}
        def kern(e, acc):
            for kk in e.kernels: acc[kk.name] = acc.get(kk.name, 0.0) + kk.duration
            for c in e.cpu_children: kern(c, acc)
        for e in ev:
            if not e.name.startswith("DFWD_R"): continue
            p = P.setdefault(e.name, {"n": 0, "wall_us": 0.0, "k": {}}); p["n"] += 1
            p["wall_us"] += e.time_range.elapsed_us(); kern(e, p["k"])
        # fallback / cross-check: device events whose start lies inside a DFWD window
        from torch.autograd import DeviceType
        dev = [(e.time_range.start, e.time_range.elapsed_us(), e.name) for e in ev if e.device_type == DeviceType.CUDA and not e.name.startswith("DFWD_")]
        for e in ev:
            if not e.name.startswith("DFWD_R"): continue
            w = P[e.name].setdefault("win", {}); a, b = e.time_range.start, e.time_range.end
            for st_, du, nm in dev:
                if a <= st_ < b: w[nm] = w.get(nm, 0.0) + du
        out = {}
        for nm, p in P.items():
            n = p["n"]; busy = sum(p["k"].values()) / n
            top = sorted(p["k"].items(), key=lambda x: -x[1])[:15]
            out[nm] = {"n": n, "wall_ms": round(p["wall_us"] / n / 1000, 3), "busy_ms": round(busy / 1000, 3),
                       "n_kernels": round(sum(1 for _ in p["k"]) , 1),
                       "top_us": [[t[0][:90], round(t[1] / n, 1)] for t in top],
                       "busy_win_ms": round(sum(p.get("win", {}).values()) / n / 1000, 3), "n_win_kernels": len(p.get("win", {})),
                       "top_win_us": [[t[0][:90], round(t[1] / n, 1)] for t in sorted(p.get("win", {}).items(), key=lambda x: -x[1])[:15]]}
        RES["prof"] = out; save()
        print("RESULT prof", json.dumps({k: {x: v[x] for x in ("n", "wall_ms", "busy_ms", "busy_win_ms")} for k, v in out.items()}), flush=True)
    except Exception:
        RES["errors"].append("prof: " + traceback.format_exc()[-2500:]); save()
        print("ERROR prof", traceback.format_exc()[-800:], flush=True)
print("DONE errors", len(RES["errors"]), flush=True)
