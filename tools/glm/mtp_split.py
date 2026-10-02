# glm-mtp step 2: split the R=2 MTP verify forward by module vs the plain R=1 step, and
# count the unique routed experts R rows touch. One model load, all arms interleaved.
# Timing uses CUDA events around each module call (no host sync inside the forward), read
# after the forward: GPU-timeline ms, including the device idle between kernels.
#   coarse arms: only model.forward is timed (unperturbed step / verify ms)
#   fine arms  : blocks, attn (kda/mla), hc, moe (router, shared, routed kernel), gemv
# Prints one RESULT json line per arm and a final SUMMARY line.
import argparse, json, os, subprocess, sys, time, collections, traceback, torch
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="~/models/glm53-exl3-td205")
ap.add_argument("--corpus", default="~/bench/ppl/wiki.test.raw")
ap.add_argument("--out", default="scratch/mtp_split.json")
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--kinds", default="code,chat")
ap.add_argument("--arms", default="")
args = ap.parse_args()
torch.set_grad_enabled(False)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exllamav3
from exllamav3.ext import exllamav3_ext
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules import block_graph as BG
from exllamav3.modules.gated_delta_net import GatedDeltaNet
from exllamav3.modules.mla_attn import MLAttention
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
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
print("loaded", round(RES["load_s"], 1), flush=True)
def save(): json.dump(RES, open(args.out, "w"), indent=1)

# ---- instrumentation ------------------------------------------------------------------
MODE = ["off"]          # off | coarse | fine
CUR = {"R": 0, "verify": False}
EV = []                 # (key, start, end) for the forward in flight
ACC = collections.defaultdict(float); CNT = collections.defaultdict(int)
SELS = []              # routed expert picks (R, k) per MoE call, counted after the run (no sync)
STACK = ["top"]        # enclosing fine key, so gemv time is keyed by its module

def timed(key_fn, fn, level):
    def w(*a, **k):
        if MODE[0] == "off" or (level == "fine" and MODE[0] != "fine") or torch.cuda.is_current_stream_capturing():
            return fn(*a, **k)
        key = key_fn(a, k)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        STACK.append(key); s.record()
        try: r = fn(*a, **k)
        finally: STACK.pop()
        e.record(); EV.append((key, s, e))
        return r
    return w

REC = [None]           # list while a *_rec arm runs: (input ids, logits) per target forward (R <= 8); ids clone: the generator feeds a reused pinned CPU buffer
def fwd_wrap(fn):
    def w(*a, **k):
        x = a[0] if a else k.get("input_ids"); p = k.get("params") or (a[1] if len(a) > 1 else {})
        CUR["R"] = x.shape[-1]; CUR["verify"] = bool(p.get("dflash_verify"))
        if REC[0] is not None and x.shape[-1] <= 8 and not torch.cuda.is_current_stream_capturing():
            r = fn(*a, **k); REC[0].append((x.flatten().clone().cpu(), r.float().reshape(x.shape[-1], -1).cpu()))
            return r
        if MODE[0] == "off" or torch.cuda.is_current_stream_capturing():
            return fn(*a, **k)
        EV.clear()
        tag = f"R{CUR['R']}" + ("v" if CUR["verify"] else "")
        if MODE[0] == "coarse": torch.cuda.synchronize(); h0 = time.perf_counter()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = fn(*a, **k); e.record(); e.synchronize()
        if MODE[0] == "coarse":
            torch.cuda.synchronize(); ACC[tag + ":forward_host"] += 1000 * (time.perf_counter() - h0); CNT[tag + ":forward_host"] += 1
        ACC[tag + ":forward"] += s.elapsed_time(e); CNT[tag + ":forward"] += 1
        for key, s1, e1 in EV:
            ACC[tag + ":" + key] += s1.elapsed_time(e1); CNT[tag + ":" + key] += 1
        EV.clear()
        return r
    return w
model.forward = fwd_wrap(model.forward)

def K(name): return lambda a, k: name
for m in model.modules:
    attn, mlp = getattr(m, "attn", None), getattr(m, "mlp", None)
    if attn is None and mlp is None:
        m.forward = timed(K("other:" + type(m).__name__), m.forward, "fine"); continue
    at = "kda" if isinstance(attn, GatedDeltaNet) else "mla" if isinstance(attn, MLAttention) else type(attn).__name__
    m.forward = timed(K("blk_" + at), m.forward, "fine")
    if attn is not None: attn.forward = timed(K("attn_" + at), attn.forward, "fine")
    for hcn in ("attn_hc", "mlp_hc"):
        hc = getattr(m, hcn, None)
        if hc is not None:
            for fn in ("mix_norm", "mix", "apply_"):
                if hasattr(hc, fn): setattr(hc, fn, timed(K("hc_" + fn), getattr(hc, fn), "fine"))
    if isinstance(mlp, BlockSparseMLP):
        mlp.forward = timed(K("moe"), mlp.forward, "fine")
        if getattr(mlp, "routing_fn", None) is not None: mlp.routing_fn = timed(K("moe_router"), mlp.routing_fn, "fine")
        if mlp.shared_experts is not None: mlp.shared_experts.forward = timed(K("moe_shared"), mlp.shared_experts.forward, "fine")
    elif mlp is not None:
        mlp.forward = timed(K("dense_mlp"), mlp.forward, "fine")

def union_wrap(fn):
    t = timed(K("moe_routed_kernel"), fn, "fine")
    def w(*a, **k):
        if MODE[0] == "fine" and not torch.cuda.is_current_stream_capturing():
            SELS.append(a[2].clone())
        return t(*a, **k)
    return w
exllamav3_ext.exl3_dec_moe_union = union_wrap(exllamav3_ext.exl3_dec_moe_union)
exllamav3_ext.exl3_dec_moe = timed(K("moe_routed_kernel"), exllamav3_ext.exl3_dec_moe, "fine")
for n in ("exl3_dec_gemv", "exl3_dec_gemv_r"):
    if hasattr(exllamav3_ext, n): setattr(exllamav3_ext, n, timed(lambda a, k: "gemv@" + STACK[-1], getattr(exllamav3_ext, n), "fine"))

# ---- runs -------------------------------------------------------------------------------
GENS = {}
def gen(ndt):
    if ndt not in GENS:
        GENS.clear(); import gc; gc.collect()
        GENS[ndt] = (Generator(model=model, cache=cache, tokenizer=tok) if ndt == 0 else
                     Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft_model,
                               draft_cache=draft_cache, num_draft_tokens=ndt, record_draft_stats=True))
    return GENS[ndt]
WIKI = tok.decode(tok.encode(open(args.corpus, encoding="utf-8").read()[:40000], add_bos=False)[0, :args.ctx])
def run(ndt, prompt, ntok):
    g = gen(ndt)
    prompt = f"Reference notes (ignore unless relevant):\n{WIKI}\n\nTask: {prompt}"
    ids = tok.encode(f"[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n", encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=ntok, sampler=GreedySampler(), stop_conditions=[])
    g.enqueue(job); t0 = time.perf_counter(); ttft = None; toks = []; rounds = 0
    mode = MODE[0]; MODE[0] = "off"   # never time the prefill
    while g.num_remaining_jobs():
        for r in g.iterate():
            tid = r.get("token_ids")
            if tid is not None and tid.numel():
                if ttft is None:
                    torch.cuda.synchronize(); ttft = time.perf_counter() - t0; MODE[0] = mode
                else: rounds += 1
                toks += tid.flatten().tolist()
    torch.cuda.synchronize(); dec = time.perf_counter() - t0 - ttft; MODE[0] = "off"
    st = list(job.draft_stats)
    # pos_acc[k] = P(draft position k+1 accepted) per verify round; tok_per_round = 1 + mean accepted
    pos_acc = [round(sum(s[2] > k for s in st) / max(len(st), 1), 4) for k in range(ndt)]
    # iterate() streams one result per token, so `rounds` counts tokens; with drafts the verify
    # rounds are the draft_stats entries (one per round)
    if st: rounds = len(st)
    acc = sum(s[2] for s in st) / max(len(st), 1)
    return {"ndt": ndt, "ntok": len(toks), "rounds": rounds, "tps": (len(toks) - 1) / dec,
            "round_ms": 1000 * dec / max(rounds, 1), "acc": acc,
            "pos_acc": pos_acc, "tok_per_round": 1 + acc if st else 1.0, "toks": toks}

# arm = (name, ndt, env overrides, graph on, mode)
ALL_ARMS = {
    "plain":       (0, {}, True, "coarse"),
    "v2":          (1, {"EXL3_DEC_MOE_UNION": "1"}, True, "coarse"),
    "v2dev":       (1, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1"}, True, "coarse"),
    "plain_fine":  (0, {}, False, "fine"),
    "v2_fine":     (1, {"EXL3_DEC_MOE_UNION": "1"}, True, "fine"),
    "v2dev_fine":  (1, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1"}, True, "fine"),
    # step 2 levers: rr = per-row dec router + lm_head row loop, g = graphed verify blocks
    "s2_base":     (1, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "EXL3_DEC_ROUTER_ROWS": "0", "_gv": 0, "_lk": 99}, True, "coarse"),
    "s2_rr":       (1, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "_gv": 0}, True, "coarse"),
    "s2_g":        (1, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "EXL3_DEC_ROUTER_ROWS": "0", "_lk": 99, "_gv": 1}, True, "coarse"),
    "s2_grr":      (1, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "_gv": 1}, True, "coarse"),
    "s2_rr_fine":  (1, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1"}, False, "fine"),
    # step 3: coarse eager plain (the true plain graph gain), and logit-recording arms (untimed use)
    "plain_eager": (0, {}, False, "coarse"),
    "plain_rec":   (0, {}, True, "off"),
    "s2_rr_rec":   (1, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "_gv": 0}, True, "off"),
}
# step 4: draft depth sweep (ndt 1..3 = verify R 2..4), lm_head as a row loop (lp) or one gemv_r launch (r)
for _n in (1, 2, 3):
    ALL_ARMS[f"s4_n{_n}_lp"] = (_n, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "_gv": 0, "_lk": 5}, True, "coarse")
    ALL_ARMS[f"s4_n{_n}_r"] = (_n, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "_gv": 0, "_lk": 99}, True, "coarse")
    ALL_ARMS[f"s4_n{_n}_r2"] = (_n, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "EXL3_GEMV_R_RPB": "2", "_gv": 0, "_lk": 99}, True, "coarse")
    ALL_ARMS[f"s4_n{_n}_fine"] = (_n, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "_gv": 0, "_lk": 99}, False, "fine")
    ALL_ARMS[f"s4_n{_n}_r_rec"] = (_n, {"EXL3_DEC_MOE_UNION": "1", "EXL3_DEC_MOE_UNION_DEV": "1", "_gv": 0, "_lk": 99}, True, "off")

def emitted_rows(rec):
    """Greedy emission replay of recorded target forwards: row 0 always emits; row i emits when
    the draft token fed at row i equals row i-1's argmax. Returns (token, logits row) per token."""
    out = []
    for ids, lg in rec:
        am = lg.argmax(-1)
        out.append((int(am[0]), lg[0]))
        for i in range(1, len(ids)):
            if int(ids[i]) != int(am[i - 1]): break
            out.append((int(am[i]), lg[i]))
    return out

def margin_report(kind, a, b):
    """Teacher-forced plain vs verify at the first divergence: same context up to there."""
    ea, eb = emitted_rows(a), emitted_rows(b)
    n = min(len(ea), len(eb)); d = next((j for j in range(n) if ea[j][0] != eb[j][0]), None)
    rep = {"kind": kind, "n": n, "first_div": d}
    for j in ([d] if d is not None else []) + [max(0, (d or n) - 1)]:
        la, lb = ea[j][1], eb[j][1]
        ta, tb = la.topk(2), lb.topk(2)
        rep[f"pos{j}"] = {"plain_top2": ta.indices.tolist(), "plain_margin": round(float(ta.values[0] - ta.values[1]), 4),
                          "verify_top2": tb.indices.tolist(), "verify_margin": round(float(tb.values[0] - tb.values[1]), 4),
                          "plain_gap_between_the_two": round(float(la[ta.indices[0]] - la[tb.indices[0]]), 4),
                          "verify_gap_between_the_two": round(float(lb[tb.indices[0]] - lb[ta.indices[0]]), 4),
                          "max_abs_dlogit": round(float((la - lb).abs().max()), 4)}
    dl = [float((ea[j][1] - eb[j][1]).abs().max()) for j in range(d if d is not None else n)]
    rep["max_abs_dlogit_before_div"] = round(max(dl), 4) if dl else None
    return rep
ORDER = [a for a in (args.arms.split(",") if args.arms else ALL_ARMS)]
KNOBS = ("EXL3_DEC_MOE_UNION", "EXL3_DEC_MOE_UNION_DEV", "EXL3_DEC_ROUTER_ROWS", "EXL3_GEMV_R_RPB")
import exllamav3.modules.quant.exl3 as EXL3Q
LK0 = EXL3Q.DEC_GEMV_R_LOOP_MIN_K
GV0 = BG.BLOCK_GRAPH_VERIFY
def setarm(ndt, envs, graph):
    for k in KNOBS: os.environ.pop(k, None)
    envs = dict(envs)
    BG.BLOCK_GRAPH_VERIFY = bool(envs.pop("_gv", GV0))
    EXL3Q.DEC_GEMV_R_LOOP_MIN_K = envs.pop("_lk", LK0)
    os.environ.update(envs); BG.BLOCK_GRAPH_ENABLED = graph

def warm(name):   # fresh graphs per arm (disabled slots / captures are not keyed by the knobs)
    ndt, envs, graph, _ = ALL_ARMS[name]; BG.purge(); setarm(ndt, envs, graph)
    try: run(ndt, "Say hello.", 24)
    except Exception: RES["errors"].append(f"warmup {name}: " + traceback.format_exc()[-2500:])

def svc_state():  # background services that would contaminate a measurement: none tracked in the public tree
    return {}

RES["env"]["services_start"] = svc_state()
base = {}; RECS = {}
for kind in args.kinds.split(","):
    p = PROMPTS[kind][0]
    for name in ORDER:
        warm(name); ndt, envs, graph, mode = ALL_ARMS[name]
        g0 = BG.global_stats(); ACC.clear(); CNT.clear(); SELS.clear(); MODE[0] = mode
        try:
            if name.endswith("_rec"): REC[0] = []
            try: r = run(ndt, p, args.tokens)
            finally: rec, REC[0] = REC[0], None
            toks = r.pop("toks")
            if rec is not None:
                RECS[(kind, name)] = rec
                em = [t for t, _ in emitted_rows(rec)]
                r["rec_replay_ok"] = em[:len(toks)] == toks[:len(em)]
                if name != "plain_rec" and (kind, "plain_rec") in RECS:
                    mr = margin_report(kind, RECS[(kind, "plain_rec")], rec); mr["arm"] = name
                    RES.setdefault("margins", []).append(mr); print("MARGIN", json.dumps(mr), flush=True)
            if ndt == 0 and kind not in base: base[kind] = toks
            b = base.get(kind, toks)
            r["greedy_common_prefix"] = next((j for j, (x, y) in enumerate(zip(b, toks)) if x != y), min(len(b), len(toks)))
            if name == "s2_base" and ("s2", kind) not in base: base[("s2", kind)] = toks
            if name.startswith("s4_n"):   # step 4: every gemv_r arm must match its depth's lm_head-loop arm
                lp = (kind, name.split("_")[1])
                if name.endswith("_lp") and lp not in base: base[lp] = toks
                if lp in base:
                    r["prefix_vs_lp"] = next((j for j, (x, y) in enumerate(zip(base[lp], toks)) if x != y), min(len(base[lp]), len(toks)))
            r["toks"] = toks
            b2 = base.get(("s2", kind))
            if b2 is not None:
                r["prefix_vs_s2_base"] = next((j for j, (x, y) in enumerate(zip(b2, toks)) if x != y), min(len(b2), len(toks)))
            r["services"] = svc_state()
            g1 = BG.global_stats()
            r.update(kind=kind, arm=name, graph={k: (dict(g1[k] - g0[k]) if k == "declines" else g1[k] - g0[k]) for k in g1} if graph else None)
            UNIQ = collections.defaultdict(list)
            for sel in SELS: UNIQ[sel.shape[0]].append(int(torch.unique(sel).numel()))
            r["ms"] = {k: round(ACC[k] / CNT[k], 4) for k in sorted(ACC)}
            r["ms_per_fwd"] = {}
            for k in ACC:  # per-forward totals (sum over layers)
                tag = k.split(":")[0]; nf = CNT.get(tag + ":forward", 0)
                if nf: r["ms_per_fwd"][k] = round(ACC[k] / nf, 3)
            r["calls_per_fwd"] = {k: round(CNT[k] / max(CNT.get(k.split(":")[0] + ":forward", 1), 1), 2) for k in CNT}
            r["uniq"] = {R: {"mean": sum(v) / len(v), "n": len(v), "hist": dict(collections.Counter(v))} for R, v in UNIQ.items()}
            capture = os.environ.get("EXL3_CAPTURE_SELS")
            if capture:
                # Preserve real routing tensors per observed row count for isolated kernel benches.
                # Prefer the GLM verify union regime rather than an atypical disjoint R=2 pick.
                by_rows = {}
                for s in SELS:
                    rows_n = int(s.shape[0])
                    score = abs(int(torch.unique(s).numel()) - ({1: 8, 2: 14, 4: 22}.get(rows_n, rows_n * 8)))
                    if rows_n not in by_rows or score < by_rows[rows_n][0]:
                        by_rows[rows_n] = (score, s.detach().cpu())
                torch.save({rows_n: v[1] for rows_n, v in by_rows.items()}, capture)
            RES["arms"].append(r); save()
            print("RESULT", json.dumps({k: r.get(k) for k in ("kind", "arm", "tps", "round_ms", "acc", "pos_acc", "tok_per_round", "greedy_common_prefix", "prefix_vs_s2_base", "prefix_vs_lp", "rec_replay_ok", "services")}), json.dumps(r["graph"]),
                  json.dumps({k: v for k, v in r["ms_per_fwd"].items()}), json.dumps({R: round(u["mean"], 2) for R, u in r["uniq"].items()}), flush=True)
        except Exception:
            RES["errors"].append(f"{kind} {name}: " + traceback.format_exc()[-2500:]); save()
            print("ERROR", kind, name, traceback.format_exc()[-600:], flush=True)
MODE[0] = "off"
print("DONE errors", len(RES["errors"]), flush=True)
