#!/usr/bin/env python
"""Row invariance of the GLM MTP verify forward when two jobs are decoded together (issue 28).

One model load, serve defaults (SPEED_ENV of tools/glm/serve.py, num_draft 2). Stages:
  router : the router at R = 2,3,4,5,6,7,8 rows on real layers: routing_dots(R rows) against R batch-1 calls, bit by bit
  ops    : one target verify forward, job alone (1 x 3 rows) against the same job next to another (2 x 3 rows); the first
           op (execution order) whose output differs; graphs off so every op is hooked
  e2e    : greedy tokens of N prompts alone and in pairs (pairs also against a partner with a wider block table)
  repeat : the paired run again K times, ids must repeat
  speed  : 2 jobs together, wall time per token, ROW_INV off against on, interleaved, same load
Arms: ROW_INV is flipped in process (ROW_INV["on"]), graphs purged at every flip.
Run through ~/hermes-work/bin/gpu-guard. Writes one json per stage into --out.
"""
import argparse, ast, json, os, sys, time, traceback, hashlib
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="~/models/glm53-v2-final")
ap.add_argument("--stages", default="router,ops,e2e")
ap.add_argument("--arms", default="off,on", help="ROW_INV arms for router / ops / e2e / speed")
ap.add_argument("--n-prompts", type=int, default=20)
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--reps", type=int, default=3, help="speed reps per arm")
ap.add_argument("--out", default="out")
ap.add_argument("--cache", type=int, default=24576)
ap.add_argument("--ops-pairs", default="0:1,2:7,4:5")
ap.add_argument("--mlp-layer", type=int, default=26)
ap.add_argument("--prompt", type=int, default=4)
ap.add_argument("--solo-passes", type=int, default=2)
ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt"))
ap.add_argument("--long-lens", default="2600,4000,7000")
ap.add_argument("--long-tokens", type=int, default=64)
ap.add_argument("--long-reps", type=int, default=2)
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs(args.out, exist_ok=True)

# serve defaults: SPEED_ENV out of serve.py (parsed, not imported)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.py")).read()
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SPEED_ENV":
        for k, v in ast.literal_eval(node.value).items():
            os.environ.setdefault(k, v)
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
side = os.path.expanduser("~/models/glm53-mtp-eh-proj-bf16.safetensors")
if "EXL3_MTP_EH_FP16" not in os.environ and os.path.isfile(side):
    os.environ["EXL3_MTP_EH_FP16"] = side

import torch
torch.set_grad_enabled(False)
import exllamav3
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator import generator as GM
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules import block_graph as BG
from exllamav3.modules import Linear, RMSNorm
from exllamav3.modules.layernorm import LayerNorm
from exllamav3.modules.gated_rmsnorm import GatedRMSNorm
from exllamav3.modules import mla_attn as MA
from exllamav3.modules import block_sparse_mlp_routing as RT
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.util.row_inv import ROW_INV
from mtp_bench_prompts import PROMPTS

NDT = 2
ARMS = args.arms.split(",")
EXTRA = [
    "Summarise the causes of the 1929 stock market crash in five bullet points.",
    "Translate into French: 'The committee will meet on Thursday to review the quarterly budget.'",
    "A train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive? Show the steps.",
    "Write a SQL query that returns the three best-selling products per category from tables products and orders.",
    "You are a support agent. A customer says the app crashes at login on Android 14. Reply with a short triage plan.",
    "Explain the difference between a mutex and a semaphore to a first-year student.",
    "Write a bash one-liner that counts distinct IP addresses in an nginx access log.",
    "Give me a recipe for a vegetarian lasagna for six people, with quantities.",
]
ALL_PROMPTS = [p for k in ("code", "chat", "prose") for p in PROMPTS[k]] + EXTRA


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, obj):
    json.dump(obj, open(os.path.join(args.out, name + ".json"), "w"), indent=1)


t0 = time.perf_counter()
config = Config.from_directory(os.path.expanduser(args.model))
model = Model.from_config(config); tok = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=args.cache, max_history=NDT)
model.load(device="cuda:0", progressbar=False)
draft_model = Model.from_config(config, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=args.cache, max_history=NDT)
draft_model.load(device="cuda:0", progressbar=False)
GM.MTP_FUSE_CATCHUP = 2
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft_model, draft_cache=draft_cache,
                num_draft_tokens=NDT)
log("loaded", round(time.perf_counter() - t0, 1), "s; env", {k: v for k, v in os.environ.items() if k.startswith("EXL3_")})


def set_arm(arm):
    ROW_INV["on"] = arm == "on"
    BG.purge()


def enc(p):
    return tok.encode(f"[gMASK]<sop><|user|>\n{p}<|assistant|>\n", encode_special_tokens=True)


def mkjob(ids, n):
    return Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), stop_conditions=[])


def decode(items, stop_after=None):
    """items: list of (ids, max_new_tokens, want). Jobs enqueued together. Returns token lists (None for want=False
    jobs, cancelled once every wanted job has finished) and wall seconds of the iterate loop."""
    jobs = [mkjob(i, n) for i, n, _ in items]
    for j in jobs: gen.enqueue(j)
    toks = {id(j): [] for j in jobs}
    done = set()
    want = {id(j) for j, it in zip(jobs, items) if it[2]}
    t = time.perf_counter()
    while gen.num_remaining_jobs() and not want <= done:
        for r in gen.iterate():
            if r["stage"] == "streaming":
                if "token_ids" in r: toks[id(r["job"])] += r["token_ids"].flatten().tolist()
                if r["eos"]: done.add(id(r["job"]))
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    for j in jobs:
        if id(j) not in done:
            try: gen.cancel(j)
            except Exception: pass
    while gen.num_remaining_jobs():
        gen.iterate()
    return [toks[id(j)] for j in jobs], dt


def sha(t): return hashlib.sha256(json.dumps(t).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------------------------- router
def stage_router():
    res = {}
    mlps = [m.mlp for m in model.modules if getattr(m, "mlp", None) is not None and isinstance(m.mlp, BlockSparseMLP)
            and m.mlp.routing_fn is RT.routing_dots and m.mlp.routing_gate is not None]
    log("router: moe layers", len(mlps))
    H = mlps[0].hidden_size
    for arm in ARMS:
        set_arm(arm); out = {}
        for R in (2, 3, 4, 5, 6, 7, 8):
            bad_layers, rows_bad, rows_tot, wdiff, set_bad = [], 0, 0, 0.0, 0
            for li, mlp in enumerate(mlps):
                g = torch.Generator(device="cuda").manual_seed(1000 + li * 17 + R)
                y = torch.randn(R, H, device="cuda", generator=g).half().contiguous()
                cfg = mlp.routing_cfg
                refs = []
                for r in range(R):
                    s, w = RT.routing_dots(1, cfg, y[r:r + 1].contiguous(), {})
                    refs.append((s.clone(), w.clone()))
                s, w = RT.routing_dots(R, cfg, y, {"dflash_verify": True})
                s, w = s.clone(), w.clone()
                lb = 0
                for r in range(R):
                    rows_tot += 1
                    same = torch.equal(s[r].sort().values, refs[r][0].flatten().sort().values)
                    # pairs of (expert, weight): order inside a row may follow the kernel, so compare as a map
                    ma = dict(zip(s[r].tolist(), w[r].view(-1).tolist())); mb = dict(zip(refs[r][0].flatten().tolist(), refs[r][1].flatten().tolist()))
                    exact = (ma == mb) and torch.equal(s[r], refs[r][0].flatten()) and torch.equal(w[r].view(-1), refs[r][1].flatten())
                    if not exact:
                        rows_bad += 1; lb += 1
                        if not same: set_bad += 1
                        if same:
                            wdiff = max(wdiff, max(abs(ma[k] - mb[k]) for k in ma))
                if lb: bad_layers.append(li)
            out[R] = {"rows_total": rows_tot, "rows_not_bitexact": rows_bad, "rows_other_expert_set": set_bad, "layers_with_diff": len(bad_layers),
                      "first_layers": bad_layers[:5], "max_weight_diff_same_experts": wdiff}
            log("router", arm, "R", R, out[R])
        res[arm] = out
    save("router", res)


# ----------------------------------------------------------------------------------------------- ops
CAP = {"armed": False, "on": False, "rec": [], "cnt": {}, "bsz": 1, "ids": None, "seqlens": None}


def cap(name, t):
    if not CAP["on"] or not isinstance(t, torch.Tensor) or torch.cuda.is_current_stream_capturing(): return
    k = CAP["cnt"].get(name, 0); CAP["cnt"][name] = k + 1
    CAP["rec"].append((f"{name}#{k}", t.detach().clone().cpu()))


def hook_method(cls, meth, label):
    f0 = getattr(cls, meth)
    def w(self, *a, **k):
        o = f0(self, *a, **k)
        if CAP["on"]:
            cap(label(self), o[0] if isinstance(o, tuple) else o)
        return o
    setattr(cls, meth, w)


_hooked = [False]
def install_hooks():
    if _hooked[0]: return
    _hooked[0] = True
    hook_method(Linear, "forward", lambda s: "lin:" + s.key)
    hook_method(RMSNorm, "forward", lambda s: "norm:" + s.key)
    hook_method(LayerNorm, "forward", lambda s: "ln:" + s.key)
    # routing outputs: selected experts then weights
    for i, m in enumerate(model.modules):
        mlp = getattr(m, "mlp", None)
        if isinstance(mlp, BlockSparseMLP) and mlp.routing_fn is RT.routing_dots:
            f0 = mlp.routing_fn
            def mk(f0, i):
                def w(bsz, cfg, y, params):
                    s, wt = f0(bsz, cfg, y, params)
                    cap(f"{i}:route_sel", s); cap(f"{i}:route_w", wt)
                    return s, wt
                return w
            mlp.routing_fn = mk(f0, i)
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
    from exllamav3.modules import gated_delta_net as GDN
    for fn in ("causal_conv1d_update", "gated_delta_rule_fn"):
        def mkw(f0, fn):
            def w(*a, **k):
                if CAP["on"] and fn == "gated_delta_rule_fn" and os.environ.get("PI_LIGHT", "0") == "0":
                    cap("kda:in_g", k["g"]); cap("kda:in_beta", k["beta"]); cap("kda:in_qkv", k["mixed_qkv"].contiguous())
                    rs, sl = k.get("recurrent_state"), k.get("recurrent_slots")
                    if rs is not None and sl is not None: cap("kda:in_state", rs[sl.long()])
                o = f0(*a, **k); cap("kda:" + fn, o); return o
            return w
        setattr(GDN, fn, mkw(getattr(GDN, fn), fn))
    for meth in ("_kda_gb_norm", "_kda_dec_cat", "_kda_gates_f16"):
        if hasattr(GDN.GatedDeltaNet, meth): hook_method(GDN.GatedDeltaNet, meth, lambda s, m=meth: "kda:" + m + ":" + s.o_proj.key)
    hook_method(GatedRMSNorm, "forward", lambda s: "gnorm:" + s.key)
    f0 = MA.mla_attn_triton_decode
    def attn_w(*a, **k):
        o = f0(*a, **k); cap("mla_decode", o); return o
    MA.mla_attn_triton_decode = attn_w
    fwd0 = model.forward
    def fwd(*a, **k):
        x = a[0] if a else k.get("input_ids")
        p = k.get("params")
        arm = (CAP["armed"] and x.shape[-1] == NDT + 1 and x.shape[0] == CAP["bsz"] and p is not None and p.get("dflash_verify")
               and not torch.cuda.is_current_stream_capturing())
        if arm:
            CAP["on"] = True; CAP["rec"] = []; CAP["cnt"] = {}; CAP["ids"] = x.clone().cpu()
            CAP["seqlens"] = p["cache_seqlens"].clone().cpu().tolist()
        try: r = fwd0(*a, **k)
        finally:
            if arm: CAP["on"] = False; CAP["armed"] = False
        if arm:
            CAP["logits"] = (r["logits"] if isinstance(r, dict) else r).float().reshape(x.shape[0] * x.shape[-1], -1).cpu()
        return r
    model.forward = fwd


def capture_run(items, bsz):
    """Decode until the first verify forward of batch size bsz has been recorded; returns (rec, logits, ids, seqlens)."""
    CAP["armed"] = True; CAP["bsz"] = bsz; CAP["logits"] = None
    jobs = [mkjob(i, 6) for i in items]
    for j in jobs: gen.enqueue(j)
    while gen.num_remaining_jobs():
        for _ in gen.iterate(): pass
    CAP["armed"] = False
    return list(CAP["rec"]), CAP["logits"], CAP["ids"], CAP["seqlens"]


def pick(a, b, idx):
    """slice of the paired tensor b that corresponds to job idx of the solo tensor a (first dim where b is 2x a)"""
    if a.shape == b.shape: return b if idx == 0 else None
    if a.dim() != b.dim():
        if a.shape[-1] != b.shape[-1]: return None
        a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        n = a2.shape[0]
        return b2[idx * n:(idx + 1) * n].reshape(a.shape) if b2.shape[0] == 2 * n else None
    for d in range(a.dim()):
        if a.shape[d] != b.shape[d]:
            if b.shape[d] == 2 * a.shape[d]:
                return b.narrow(d, idx * a.shape[d], a.shape[d])
            return None
    return b


def compare(solo, paired, idx):
    """in paired's execution order: names present in both; first/all differing"""
    ds = dict(solo); out = []; missing = 0
    for name, b in paired:
        if name.startswith("mla_decode"): continue   # per-row launches: call order differs with the job count, covered by .attn
        a = ds.get(name)
        if a is None: missing += 1; continue
        b0 = pick(a, b, idx)
        if b0 is None: out.append([name, "shape", list(a.shape), list(b.shape)]); continue
        eq = torch.equal(a, b0)
        if not eq:
            d = float((a.float() - b0.float()).abs().max()) if a.dtype.is_floating_point else -1.0
            out.append([name, "diff", d])
    return out, missing


def stage_ops():
    install_hooks()
    BG.BLOCK_GRAPH_ENABLED = False
    MA.MLAttention.bc_mla_step = lambda self, *a, **k: None
    res = {}
    pairs = [tuple(int(v) for v in t.split(":")) for t in args.ops_pairs.split(",")]
    # solo records (arm independent)
    solo = {}
    for i in sorted({i for p in pairs for i in p}):
        BG.purge()
        solo[i] = capture_run([enc(ALL_PROMPTS[i])], 1)
        log("ops solo", i, "ids", solo[i][2].tolist(), "seqlens", solo[i][3], "n_ops", len(solo[i][0]))
    for arm in ARMS:
        set_arm(arm); BG.BLOCK_GRAPH_ENABLED = False
        rows = []
        for (i, j) in pairs:
            rec, lg, ids, sl = capture_run([enc(ALL_PROMPTS[i]), enc(ALL_PROMPTS[j])], 2)
            for idx, k in enumerate((i, j)):
                srec, slg, sids, ssl = solo[k]
                same_in = bool(torch.equal(ids[idx], sids[0])) and sl[idx] == ssl[0]
                diffs, missing = compare(srec, rec, idx)
                lrow = slg.reshape(1, NDT + 1, -1)[0]; prow = lg.reshape(2, NDT + 1, -1)[idx]
                ent = {"pair": [i, j], "job": k, "inputs_equal": same_in, "ids_solo": sids[0].tolist(), "ids_paired": ids[idx].tolist(),
                       "logits_rows_equal": [bool(torch.equal(lrow[r], prow[r])) for r in range(NDT + 1)],
                       "n_diff_ops": len(diffs), "n_unmatched_paired_ops": missing, "first_diff": diffs[:1], "first_10": diffs[:10],
                       "all_diffs": [d[:3] for d in diffs[:80]],
                       "first_diff_block_outputs": [d for d in diffs if d[0].split(":")[0].isdigit() and ":" in d[0] and "." not in d[0].split(":", 1)[1]][:1]}
                rows.append(ent)
                log("ops", arm, (i, j), "job", k, "inputs_equal", same_in, "logits_eq", ent["logits_rows_equal"], "n_diff", len(diffs),
                    "first", diffs[:1])
        res[arm] = rows
        save("ops", res)
    BG.BLOCK_GRAPH_ENABLED = os.environ.get("EXL3_BLOCK_GRAPH", "0") == "1"
    BG.purge()


def stage_det():
    """run-to-run determinism: the same capture repeated, compared with the first one (solo vs solo, paired vs paired)"""
    install_hooks()
    BG.BLOCK_GRAPH_ENABLED = False
    MA.MLAttention.bc_mla_step = lambda self, *a, **k: None
    set_arm("on"); BG.BLOCK_GRAPH_ENABLED = False
    res = []
    for items in ([4], [5], [4, 5], [10], [11], [10, 11]):
        runs = []
        for rep in range(args.repeats):
            BG.purge()
            runs.append(capture_run([enc(ALL_PROMPTS[i]) for i in items], len(items)))
        for rep in range(1, len(runs)):
            for idx in range(len(items)):
                diffs, missing = compare(runs[0][0], runs[rep][0], 0) if len(items) == 1 else compare_same(runs[0][0], runs[rep][0])
                break
            ent = {"items": items, "rep": rep, "n_diff": len(diffs), "first": diffs[:3], "ids_equal": bool(torch.equal(runs[0][2], runs[rep][2]))}
            res.append(ent); log("det", ent)
        save("det", res)
    BG.purge()


def stage_pre():
    """prefill determinism: each KDA chunk call (and the conv before it) executed twice from the same snapshot of its state"""
    from exllamav3.modules import gated_delta_net as GDN
    stats = {"rule": [0, 0], "conv": [0, 0], "bad": []}
    cur = {"layer": 0}
    f_rule, f_conv = GDN.gated_delta_rule_fn, GDN.causal_conv1d_update
    def rule(*a, **k):
        rs, sl = k.get("recurrent_state"), k.get("recurrent_slots")
        if rs is None or sl is None:
            return f_rule(*a, **k)
        stats.setdefault("shapes", {})[str((k["history"], tuple(k["mixed_qkv"].shape[:3])))] = 1
        idx = sl.long()
        snap = rs[idx].clone()
        o1 = f_rule(*a, **k); s1 = rs[idx].clone(); torch.cuda.synchronize()
        rs[idx] = snap
        o2 = f_rule(*a, **k); s2 = rs[idx].clone(); torch.cuda.synchronize()
        stats["rule"][0] += 1
        if not (torch.equal(o1, o2) and torch.equal(s1, s2)):
            stats["rule"][1] += 1; stats["bad"].append(["rule", cur["layer"], bool(torch.equal(o1, o2)), bool(torch.equal(s1, s2)),
                                                       float((s1 - s2).abs().max())])
        cur["layer"] += 1
        return o2
    def conv(*a, **k):
        cs, sl = k.get("conv_state"), k.get("recurrent_slots")
        if k["history"] or cs is None or sl is None or k["mixed_qkv"].shape[-1] < 8:
            return f_conv(*a, **k)
        idx = sl.long()
        snap = cs[idx].clone()
        o1 = f_conv(*a, **k).clone(); s1 = cs[idx].clone(); torch.cuda.synchronize()
        cs[idx] = snap
        o2 = f_conv(*a, **k); s2 = cs[idx].clone(); torch.cuda.synchronize()
        stats["conv"][0] += 1
        if not (torch.equal(o1, o2) and torch.equal(s1, s2)):
            stats["conv"][1] += 1; stats["bad"].append(["conv", cur["layer"]])
        return o2
    GDN.gated_delta_rule_fn, GDN.causal_conv1d_update = rule, conv
    BG.BLOCK_GRAPH_ENABLED = False
    set_arm("on"); BG.BLOCK_GRAPH_ENABLED = False
    for rep in range(args.repeats):
        for i in (4, 5, 10, 11):
            cur["layer"] = 0
            decode([(enc(ALL_PROMPTS[i]), 1, True)])
        log("pre rep", rep, stats["rule"], stats["conv"], stats["bad"][-6:], list(stats.get("shapes", {}))[:6])
        save("pre", stats)
    GDN.gated_delta_rule_fn, GDN.causal_conv1d_update = f_rule, f_conv


def stage_pre2():
    """each KDA layer forward during prefill run twice from the same state snapshot; reports the layer and the sub-op that first differs"""
    from exllamav3.modules import gated_delta_net as GDN
    ctx = {"mode": 0, "snaps": [], "caps": []}
    f_rule, f_conv = GDN.gated_delta_rule_fn, GDN.causal_conv1d_update
    def rule(*a, **k):
        rs, sl = k.get("recurrent_state"), k.get("recurrent_slots")
        if ctx["mode"] == 1 and rs is not None and sl is not None: ctx["snaps"].append((rs, sl.long(), rs[sl.long()].clone()))
        o = f_rule(*a, **k)
        if ctx["mode"]: ctx["caps"].append(("rule_out", o.detach().clone()))
        return o
    def conv(*a, **k):
        cs, sl = k.get("conv_state"), k.get("recurrent_slots")
        if ctx["mode"] == 1 and cs is not None and sl is not None: ctx["snaps"].append((cs, sl.long(), cs[sl.long()].clone()))
        o = f_conv(*a, **k)
        if ctx["mode"]: ctx["caps"].append(("conv_out", o.detach().clone()))
        return o
    GDN.gated_delta_rule_fn, GDN.causal_conv1d_update = rule, conv
    for meth in ("_kda_gb_norm", "_kda_dec_cat", "_kda_gates_f16"):
        if hasattr(GDN.GatedDeltaNet, meth):
            def mk(f0, meth):
                def w(self, *a, **k):
                    o = f0(self, *a, **k)
                    if ctx["mode"]: ctx["caps"].append((meth, [t.detach().clone() for t in (o if isinstance(o, tuple) else (o,)) if isinstance(t, torch.Tensor)]))
                    return o
                return w
            setattr(GDN.GatedDeltaNet, meth, mk(getattr(GDN.GatedDeltaNet, meth), meth))
    for cls, nm in ((Linear, "lin"),):
        f0 = cls.forward
        def lw(self, *a, f0=f0, **k):
            o = f0(self, *a, **k)
            if ctx["mode"]: ctx["caps"].append(("lin:" + self.key, o.detach().clone()))
            return o
        cls.forward = lw
    f_fwd = GDN.GatedDeltaNet.forward
    stats = {"n": 0, "bad": []}
    def fwd(self, x, params, *a, **k):
        if not self.kda or x.shape[1] < 8 or params.get("dflash_verify"):
            return f_fwd(self, x, params, *a, **k)
        ctx["mode"] = 1; ctx["snaps"] = []; ctx["caps"] = []
        o1 = f_fwd(self, x, params, *a, **k).clone()
        snaps = ctx["snaps"]; caps1 = ctx["caps"]
        s1 = [t[idx].clone() for t, idx, _ in snaps]
        torch.cuda.synchronize()
        for t, idx, sn in snaps: t[idx] = sn
        ctx["mode"] = 2; ctx["snaps"] = []; ctx["caps"] = []
        o2 = f_fwd(self, x, params, *a, **k)
        caps2 = ctx["caps"]; ctx["mode"] = 0
        torch.cuda.synchronize()
        stats["n"] += 1
        first = None
        for (n1, t1), (n2, t2) in zip(caps1, caps2):
            if isinstance(t1, list):
                if not all(torch.equal(u, v) for u, v in zip(t1, t2)): first = n1; break
            elif not torch.equal(t1, t2): first = n1; break
        st_eq = all(torch.equal(t[idx], a_) for (t, idx, _), a_ in zip(snaps, s1))
        if first is not None or not torch.equal(o1, o2) or not st_eq:
            stats["bad"].append([self.o_proj.key, first, bool(torch.equal(o1, o2)), st_eq]); log("pre2 BAD", stats["bad"][-1])
        return o2
    GDN.GatedDeltaNet.forward = fwd
    BG.BLOCK_GRAPH_ENABLED = False
    set_arm("on"); BG.BLOCK_GRAPH_ENABLED = False
    for rep in range(args.repeats):
        for i in (4, 5, 10, 11):
            decode([(enc(ALL_PROMPTS[i]), 1, True)])
        log("pre2 rep", rep, stats["n"], len(stats["bad"]))
        save("pre2", stats)


def stage_seq():
    """state hash sequence of every KDA rule call over the first tokens of a job, compared across repeats of the same job"""
    from exllamav3.modules import gated_delta_net as GDN
    def hs(t):
        v = t.contiguous().view(torch.int32).long().reshape(-1)
        return (int(v.sum()), int((v * torch.arange(1, v.numel() + 1, device=v.device) % 1000003).sum()))
    log_ = []
    f_rule = GDN.gated_delta_rule_fn
    def rule(*a, **k):
        rs, sl = k.get("recurrent_state"), k.get("recurrent_slots")
        if rs is None or sl is None: return f_rule(*a, **k)
        idx = sl.long()
        h_in = hs(rs[idx]); inp = (hs(k["g"]), hs(k["beta"]), hs(k["mixed_qkv"]))
        o = f_rule(*a, **k)
        log_.append([int(k["mixed_qkv"].shape[1]), bool(k["history"]), inp, h_in, hs(rs[idx]), hs(o), [int(v) for v in idx.tolist()]])
        return o
    GDN.gated_delta_rule_fn = rule
    BG.BLOCK_GRAPH_ENABLED = False
    set_arm("on"); BG.BLOCK_GRAPH_ENABLED = False
    out = []
    for i in (4, 5, 10, 11):
        reps = []
        for rep in range(args.repeats):
            log_.clear(); BG.purge()
            decode([(enc(ALL_PROMPTS[i]), 12, True)])
            reps.append(list(log_))
        n0 = len(reps[0])
        for rep in range(1, len(reps)):
            r = reps[rep]; first = None
            for c in range(min(n0, len(r))):
                a, b = reps[0][c], r[c]
                if a[:3] != b[:3] or a[3:6] != b[3:6]:
                    kind = ("inputs" if a[2] != b[2] else "state_in" if a[3] != b[3] else "rule")
                    first = [c, c // 34, c % 34, kind, a[0], a[1], a[6], b[6]]; break
            ent = {"prompt": i, "rep": rep, "ncalls": [n0, len(r)], "first": first}
            out.append(ent); log("seq", ent)
        save("seq", out)


def stage_seq2():
    """ordered hash event log of everything in the first forwards (every module output) of a job; first differing event across repeats"""
    from exllamav3.modules import gated_delta_net as GDN
    def hs(t):
        v = t.detach().contiguous().view(torch.uint8).long().reshape(-1)
        return (int(v.sum()), int((v * (torch.arange(1, v.numel() + 1, device=v.device) % 251)).sum()))
    ev = []
    ON = {"on": False}
    def ev_add(n, t):
        if ON["on"] and isinstance(t, torch.Tensor) and not torch.cuda.is_current_stream_capturing(): ev.append((n, hs(t)))
    hook_method_ev = lambda cls, meth, label: setattr(cls, meth, (lambda f0: (lambda self, *a, **k: (lambda o: (ev_add(label(self), o[0] if isinstance(o, tuple) else o), o)[1])(f0(self, *a, **k))))(getattr(cls, meth)))
    hook_method_ev(Linear, "forward", lambda s: "lin:" + s.key)
    hook_method_ev(RMSNorm, "forward", lambda s: "norm:" + s.key)
    hook_method_ev(GatedRMSNorm, "forward", lambda s: "gnorm:" + s.key)
    for fn in ("causal_conv1d_update", "gated_delta_rule_fn"):
        def mkw(f0, fn):
            def w(*a, **k):
                if fn == "gated_delta_rule_fn":
                    ev_add("kda_in_g", k["g"]); ev_add("kda_in_beta", k["beta"]); ev_add("kda_in_qkv", k["mixed_qkv"])
                    rs, sl = k.get("recurrent_state"), k.get("recurrent_slots")
                    if rs is not None and sl is not None: ev_add("kda_in_state", rs[sl.long()])
                o = f0(*a, **k); ev_add("kda:" + fn, o); return o
            return w
        setattr(GDN, fn, mkw(getattr(GDN, fn), fn))
    for meth in ("_kda_gb_norm", "_kda_dec_cat", "_kda_gates_f16"):
        hook_method_ev(GDN.GatedDeltaNet, meth, lambda s, m=meth: "kda:" + m + ":" + s.o_proj.key)
    for i, mm in enumerate(model.modules):
        for sub in ("attn", "mlp"):
            sm = getattr(mm, sub, None)
            if sm is not None and hasattr(sm, "forward"):
                def mk(sm, name):
                    f0 = sm.forward
                    def w(*a, **k):
                        if a: ev_add(name + ":in", a[0])
                        o = f0(*a, **k); ev_add(name, o[0] if isinstance(o, tuple) else o); return o
                    sm.forward = w
                mk(sm, f"{i}:{mm.key}.{sub}")
    from exllamav3.ext import exllamav3_ext as EXT
    from exllamav3.modules.quant.exl3 import dec_workspace
    IN = {"mlp": False}
    ws_ptrs = {t.data_ptr() for t in dec_workspace(torch.device("cuda:0"))}
    for name in dir(EXT):
        f0 = getattr(EXT, name)
        if not callable(f0) or name.startswith("__"): continue
        def mkx(f0, name):
            def w(*a, **k):
                r = f0(*a, **k)
                if IN["mlp"] and ON["on"] and not torch.cuda.is_current_stream_capturing():
                    for j, t in enumerate(list(a) + list(k.values()) + ([r] if isinstance(r, torch.Tensor) else [])):
                        if isinstance(t, torch.Tensor) and t.is_cuda and t.data_ptr() not in ws_ptrs and t.numel() < 4e6 and t.dtype in (torch.half, torch.bfloat16, torch.float, torch.int32, torch.int64):
                            ev.append((f"ext:{name}:{j}", hs(t)))
                return r
            return w
        try: setattr(EXT, name, mkx(f0, name))
        except Exception: pass
    for i, mm in enumerate(model.modules):
        sm = getattr(mm, "mlp", None)
        if isinstance(sm, BlockSparseMLP) and str(mm.key).endswith(".layers.%d" % args.mlp_layer):
            def mk3(sm):
                f0 = sm.forward
                def w(x, *a, **k):
                    IN["mlp"] = x.numel() // x.shape[-1] > 8
                    try: return f0(x, *a, **k)
                    finally: IN["mlp"] = False
                sm.forward = w
            mk3(sm)
            f1 = sm.routing_fn
            def mkr(f1):
                def w(*a, **k):
                    if IN["mlp"] and ON["on"]:
                        ev.append(("route_in_y", hs(a[2])))
                        log("ROUTEFN", getattr(f1, "__name__", str(f1)) + " nogroup_torch=" + str(RT._routing_nogroup_torch.__name__) + " has_ds3=" + str(hasattr(RT.ext, "routing_ds3_nogroup")) + " dec_ok=" + str(RT._dec_router_ok(a[0], a[1], a[2], a[3])) + " rows_ok=" + str(RT._dec_router_rows_ok(a[0], a[1], a[2], a[3])) + " gate=" + str(tuple(a[1].gate_tensor.shape)) + " bias=" + str(a[1].e_score_correction_bias is not None))
                        lg1 = torch.matmul(RT._pad_rows(a[2]), a[1].gate_tensor); lg2 = torch.matmul(RT._pad_rows(a[2]), a[1].gate_tensor)
                        ev.append(("route_matmul_twin_equal", bool(torch.equal(lg1, lg2))))
                    s_, w_ = f1(*a, **k)
                    if IN["mlp"] and ON["on"]:
                        s1_, w1_ = s_[:a[0]].clone(), w_[:a[0]].clone()
                        ev.append(("route_sel", hs(s_[:a[0]]))); ev.append(("route_w", hs(w_[:a[0]])))
                        s2_, w2_ = f1(*a, **k)
                        ev.append(("route_twin_sel_equal", bool(torch.equal(s1_, s2_[:a[0]]))))
                        ev.append(("route_twin_w_equal", bool(torch.equal(w1_, w2_[:a[0]]))))
                        ev.append(("route_shape", (tuple(s_.shape), tuple(w_.shape), a[0])))
                        ev.append(("route_sel_row0", tuple(s1_[0].tolist())))
                        s_, w_ = s2_, w2_
                    return s_, w_
                return w
            sm.routing_fn = mkr(f1)
    BG.BLOCK_GRAPH_ENABLED = False
    set_arm("on"); BG.BLOCK_GRAPH_ENABLED = False
    out = []
    for i in (args.prompt,):
        reps = []
        for rep in range(args.repeats):
            ev.clear(); BG.purge(); ON["on"] = True
            decode([(enc(ALL_PROMPTS[i]), 1, True)])
            ON["on"] = False
            reps.append(list(ev))
        groups = []
        for rep, r in enumerate(reps):
            for g in groups:
                if g[1] == r: g[0].append(rep); break
            else: groups.append([[rep], r])
        ent = {"prompt": i, "groups": [g[0] for g in groups], "diffs": []}
        for g in groups[1:]:
            a, b = groups[0][1], g[1]; first = None
            for c in range(min(len(a), len(b))):
                if a[c] != b[c]: first = [c, a[c][0], b[c][0]]; break
            ent["diffs"].append([g[0][0], len(a), len(b), first])
        ent["excerpt"] = [[[n_, str(h_)] for n_, h_ in r[ (ent["diffs"][0][3][0] - 1 if ent["diffs"] and ent["diffs"][0][3] else 0): (ent["diffs"][0][3][0] + 8 if ent["diffs"] and ent["diffs"][0][3] else 0)]] for r in reps[:3]]
        out.append(ent); log("seq2", ent)
        save("seq2", out)


def stage_seq3():
    """as seq2, but inside the prefill BlockSparseMLP forward of --mlp-layer every extension call is logged with the hashes of all its tensor arguments
    after the call; reports, per call and argument, how many repeats deviate from the majority"""
    from exllamav3.ext import exllamav3_ext as EXT
    import collections
    def hs(t):
        # stream-ordered device copy: no host sync, so a race between the module's own launches stays exposed
        return t.detach().clone()
    ev = []
    IN = {"mlp": None}
    for name in dir(EXT):
        f0 = getattr(EXT, name)
        if not callable(f0) or name.startswith("__"): continue
        def mk(f0, name):
            def w(*a, **k):
                r = f0(*a, **k)
                if IN["mlp"] is not None and not torch.cuda.is_current_stream_capturing():
                    ev.append((name, [hs(t) if (isinstance(t, torch.Tensor) and t.is_cuda and t.numel() < 4e6 and t.dtype in (torch.half, torch.bfloat16, torch.float, torch.int32, torch.int64)) else None for t in list(a) + list(k.values()) + ([r] if isinstance(r, torch.Tensor) else [])]))
                return r
            return w
        try: setattr(EXT, name, mk(f0, name))
        except Exception as e: log("cannot wrap", name, e)
    for i, mm in enumerate(model.modules):
        sm = getattr(mm, "mlp", None)
        if isinstance(sm, BlockSparseMLP) and str(mm.key).endswith(".layers.%d" % args.mlp_layer):
            def mk2(sm, name):
                f0 = sm.forward
                def w(x, *a, **k):
                    on = x.numel() // x.shape[-1] > 8
                    if on: IN["mlp"] = name
                    try: return f0(x, *a, **k)
                    finally: IN["mlp"] = None
                sm.forward = w
            mk2(sm, f"{i}:{mm.key}")
    BG.BLOCK_GRAPH_ENABLED = False
    set_arm("on"); BG.BLOCK_GRAPH_ENABLED = False
    reps = []
    for rep in range(args.repeats):
        ev.clear(); BG.purge()
        decode([(enc(ALL_PROMPTS[args.prompt]), 1, True)])
        reps.append(list(ev)); torch.cuda.synchronize(); log("seq3 rep", rep, len(ev))
    n = min(len(r) for r in reps)
    flagged = []
    for c in range(n):
        names = {r[c][0] for r in reps}
        for j in range(len(reps[0][c][1])):
            vals = [r[c][1][j] if j < len(r[c][1]) else None for r in reps]
            if any(v is None for v in vals): continue
            keys = [tuple(v.shape) + (bool(torch.equal(v, vals[0])),) for v in vals]
            dev = sum(1 for k in keys if not k[-1])
            if dev: flagged.append([c, reps[0][c][0], j, dev, len(vals), list(vals[0].shape), str(vals[0].dtype)])
    log("seq3 flagged", flagged[:12], "total", len(flagged))
    save("seq3", {"flagged": flagged[:200], "n": n})


def compare_same(a, b):
    ds = dict(a); out = []
    for name, t in b:
        if name.startswith("mla_decode"): continue
        x = ds.get(name)
        if x is None or x.shape != t.shape: out.append([name, "shape"]); continue
        if not torch.equal(x, t): out.append([name, "diff", float((x.float() - t.float()).abs().max()) if x.dtype.is_floating_point else -1.0])
    return out, 0


# ----------------------------------------------------------------------------------------------- e2e
def stage_e2e():
    n = min(args.n_prompts, len(ALL_PROMPTS))
    P = [enc(p) for p in ALL_PROMPTS[:n]]
    N = args.tokens
    res = {"solo": {}, "arms": {}}
    set_arm(ARMS[-1])
    # solo twice (cold, warm): determinism of the reference
    solo = []
    for rep in range(args.solo_passes):
        t = []
        for i in range(n):
            tk, _ = decode([(P[i], N, True)]); t.append(tk[0])
        solo.append(t)
        log("e2e solo pass", rep, "done")
    stable = [i for i in range(n) if all(sp[i] == solo[0][i] for sp in solo)]
    res["solo"] = {"stable_prompts": stable, "n": n, "sha": [sha(solo[0][i]) for i in range(n)]}
    log("e2e solo stable", len(stable), "of", n)
    ref = solo[0]
    for arm in ARMS:
        set_arm(arm)
        out = {"pairs": [], "wide": []}
        for i in range(0, n - 1, 2):
            tk, dt = decode([(P[i], N, True), (P[i + 1], N, True)])
            for k, a in enumerate((i, i + 1)):
                eq = tk[k] == ref[a]
                first = next((x for x in range(min(len(tk[k]), len(ref[a]))) if tk[k][x] != ref[a][x]), None)
                out["pairs"].append({"prompt": a, "equal": eq, "first_diff": first, "n": len(tk[k])})
        # wide partner: prompt a next to a second job whose max length needs a wider block table
        for a in range(0, n, 4):
            partner = P[(a + 3) % n]
            tk, dt = decode([(P[a], N, True), (partner, 5000, False)])
            eq = tk[0] == ref[a]
            first = next((x for x in range(min(len(tk[0]), len(ref[a]))) if tk[0][x] != ref[a][x]), None)
            out["wide"].append({"prompt": a, "equal": eq, "first_diff": first, "n": len(tk[0])})
        eqs = [p["equal"] for p in out["pairs"]]; weq = [p["equal"] for p in out["wide"]]
        out["summary"] = {"pairs_equal": f"{sum(eqs)}/{len(eqs)}", "wide_equal": f"{sum(weq)}/{len(weq)}",
                          "pairs_equal_among_stable": f"{sum(p['equal'] for p in out['pairs'] if p['prompt'] in stable)}/{sum(1 for p in out['pairs'] if p['prompt'] in stable)}"}
        res["arms"][arm] = out
        log("e2e", arm, out["summary"])
        save("e2e", res)
    # keep the solo ids for the repeat / speed stages
    save("solo_ids", {"ids": ref})


def stage_repeat():
    n = min(args.n_prompts, len(ALL_PROMPTS))
    P = [enc(p) for p in ALL_PROMPTS[:n]]; N = args.tokens
    ref = json.load(open(os.path.join(args.out, "solo_ids.json")))["ids"]
    set_arm("on")
    res = {"reps": []}
    for rep in range(args.repeats):
        # rotate the partner each repeat: the output must not depend on who the neighbour is
        shift = 1 + 2 * rep
        order = list(range(n)); order = order[shift % n:] + order[:shift % n]
        eqs = 0; tot = 0; bad = []
        for q in range(0, n - 1, 2):
            a, b = order[q], order[q + 1]
            tk, _ = decode([(P[a], N, True), (P[b], N, True)])
            for k, p in enumerate((a, b)):
                tot += 1; ok = tk[k] == ref[p]; eqs += ok
                if not ok: bad.append(p)
        res["reps"].append({"rep": rep, "equal_to_solo": f"{eqs}/{tot}", "bad": bad})
        log("repeat", rep, res["reps"][-1]); save("repeat", res)


def stage_pairrep():
    """Same pair decoded --repeats times, compared with solo ids and with each other (first differing token)."""
    n = min(args.n_prompts, len(ALL_PROMPTS))
    P = [enc(p) for p in ALL_PROMPTS[:n]]; N = args.tokens
    ref = json.load(open(os.path.join(args.out, "solo_ids.json")))["ids"]
    set_arm("on")
    a, b = [int(x) for x in args.ops_pairs.split(",")[0].split(":")]
    res = []
    for rep in range(args.repeats):
        tk, _ = decode([(P[a], N, True), (P[b], N, True)])
        ent = {"rep": rep}
        for k, p in enumerate((a, b)):
            f = next((i for i, (x, y) in enumerate(zip(tk[k], ref[p])) if x != y), None)
            ent[f"p{p}_first_diff"] = f
        res.append(ent); log("pairrep", ent); save("pairrep", res)
    for rep in range(args.repeats):
        tk, _ = decode([(P[a], N, True)])
        f = next((i for i, (x, y) in enumerate(zip(tk[0], ref[a])) if x != y), None)
        log("solorep", rep, "p%d first_diff" % a, f)


def stage_speed():
    n = min(args.n_prompts, len(ALL_PROMPTS))
    P = [enc(p) for p in ALL_PROMPTS[:n]]; N = args.tokens
    res = {"paired": {a: [] for a in ARMS}, "solo": {a: [] for a in ARMS}}
    for a in ARMS:   # warm both arms (graph capture)
        set_arm(a)
        decode([(P[0], 32, True), (P[1], 32, True)]); decode([(P[0], 32, True)])
    for rep in range(args.reps):
        for a in ARMS:
            set_arm(a)
            ntok = 0; tt = 0.0
            for i in range(0, 8, 2):
                tk, dt = decode([(P[i], N, True), (P[i + 1], N, True)]); ntok += sum(len(x) for x in tk); tt += dt
            res["paired"][a].append(ntok / tt)
            ntok = 0; tt = 0.0
            for i in range(4):
                tk, dt = decode([(P[i], N, True)]); ntok += len(tk[0]); tt += dt
            res["solo"][a].append(ntok / tt)
            log("speed rep", rep, a, "paired agg tok/s", round(res["paired"][a][-1], 2), "solo tok/s", round(res["solo"][a][-1], 2))
            save("speed", res)


def stage_long():
    """Prompts above index_topk (2048): the DSA sparse path. Solo ids (two passes) against paired ids, mixed lengths."""
    corpus = open(args.corpus).read()
    want = [int(x) for x in args.long_lens.split(",")]
    P = []; off = 0
    for w in want:
        c = int(w * 3.5); txt = None
        for _ in range(4):
            txt = "Summarise the following text in three sentences.\n\n" + corpus[off:off + c]
            n = int(enc(txt).shape[-1])
            c = int(c * w / max(n, 1))
        P.append(enc(txt)); off += c + 1000
    lens = [int(p.shape[-1]) for p in P]
    log("long prompt tokens", lens)
    short = enc(ALL_PROMPTS[0])
    N = args.long_tokens
    set_arm("on")
    solo = [[decode([(p, N, True)])[0][0] for p in P + [short]] for _ in range(2)]
    stable = [solo[0][i] == solo[1][i] for i in range(len(P) + 1)]
    log("long solo stable", stable)
    ref = solo[0]
    items = P + [short]
    res = {"lens": lens + [int(short.shape[-1])], "solo_stable": stable, "pairs": []}
    pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]
    for rep in range(args.long_reps):
        for i, j in pairs:
            tk, _ = decode([(items[i], N, True), (items[j], N, True)])
            for k, a in enumerate((i, j)):
                first = next((x for x in range(min(len(tk[k]), len(ref[a]))) if tk[k][x] != ref[a][x]), None)
                res["pairs"].append({"rep": rep, "pair": [i, j], "prompt": a, "equal": tk[k] == ref[a], "first_diff": first})
        eq = [p["equal"] for p in res["pairs"]]
        log("long rep", rep, f"{sum(eq)}/{len(eq)} equal")
        save("long", res)


stages = args.stages.split(",")
# ops wraps every module, replaces bc_mla_step and the routing functions: nothing after it is the served path
assert not ({"ops", "det", "pre", "pre2", "seq", "seq2", "seq3"} & set(stages)) or stages[-1] in ("ops", "det", "pre", "pre2", "seq", "seq2", "seq3"), "run the ops stage last"
for st in stages:
    try:
        log("stage", st); globals()["stage_" + st]()
    except Exception:
        log("ERROR", st, traceback.format_exc()[-1800:])
log("DONE")
