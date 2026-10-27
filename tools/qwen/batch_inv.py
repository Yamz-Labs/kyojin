#!/usr/bin/env python
"""Row invariance of the Qwen decode / verify forward when several requests are decoded together (--sessions N).

One engine load with the serve defaults (SERVE_ENV of serve.py, the speculative policy of the server). Stages:
  ref    : plain greedy continuation of each probe prompt, one request alone (feeds the oracle drafts of `ops`)
  rinv   : one request alone, greedy ids for the plain step, the shipped draft policy and fixed draft depths 1/3/7:
           the ids a request gets must not depend on how deep its neighbours make it draft
  ops    : one verify forward of job A alone against the same job next to N-1 others, per module and per row, in
           execution order (the first differing module is the culprit). Drafts come from the plain continuation
           (some positions corrupted on purpose, so the accept lengths differ), depth --depths, step --steps
  e2e    : greedy ids of every probe prompt alone and in groups of --sessions, shipped policy; arms --arms
  speed  : decode wall time of one request and of --sessions requests together, per arm
Arms: ROW_INV["on"] (exllamav3.util.row_inv) and GROUPS["on"] (exllamav3.generator.row_groups) flipped together in process.
Run it alone on the GPU. Writes one json per stage into --out.
"""
import argparse, hashlib, json, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
ap.add_argument("--corpus", required=True, help="text file the probe prompts are cut from")
ap.add_argument("--out", default="out")
ap.add_argument("--stages", default="ref,rinv,ops,e2e")
ap.add_argument("--arms", default="off,on")
ap.add_argument("--sessions", type=int, default=4)
ap.add_argument("--ctx", type=int, default=24576)
ap.add_argument("--plen", default="300", help="prompt lengths in tokens, comma list; one probe set per length")
ap.add_argument("--n-prompts", type=int, default=8)
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--depths", default="1,3,7")
ap.add_argument("--steps", type=int, default=3, help="ops: verify forwards compared per run")
ap.add_argument("--no-vision", action="store_true", default=True)
ap.add_argument("--policy", default="mix")
ap.add_argument("--cache-bits", type=int, default=0, choices=(0, 8), help="K/V page format: 0 = fp16, 8 = packed int8")
ap.add_argument("--planes", default="keep", choices=["keep", "zero", "nan"], help="slots/pf/e2e: what the QSA side planes hold before each run (keep = whatever the previous run left)")
ap.add_argument("--fixd", default="mix", help="e2e: draft depth policies, comma list of mix or a fixed depth (QWSPEC_FIXD)")
ap.add_argument("--subops", action="store_true", help="ops: also compare the attention internals (they run once per sequence in a grouped step, so the names do not line up)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import serve as S   # noqa: E402  (tools/qwen/serve.py: the server's engine and environment)

t_load = time.perf_counter()
eng = S.QwenEngine(args.model, args.ctx, ndt=7, draft_policy=args.policy, vision=False, cache_bits=args.cache_bits, sessions=args.sessions)
torch = eng.torch
from exllamav3.util.row_inv import ROW_INV   # noqa: E402
from exllamav3.generator.row_groups import GROUPS as ROW_GROUPS   # noqa: E402
from exllamav3.modules import block_graph as BG   # noqa: E402
SPEC = eng.draft_model is not None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, obj):
    json.dump(obj, open(os.path.join(args.out, name + ".json"), "w"), indent=1)


def sha(t):
    return hashlib.sha256(json.dumps(t).encode()).hexdigest()[:12]


def set_arm(arm):
    ROW_INV["on"] = ROW_GROUPS["on"] = arm == "on"
    BG.purge()


log("loaded", round(time.perf_counter() - t_load, 1), "s; sessions", args.sessions, "spec", SPEC,
    "env", {k: v for k, v in os.environ.items() if k.startswith("EXL3_ROW") or k.startswith("QWSPEC")})

corpus = open(args.corpus, errors="replace").read()
PLENS = [v if v.startswith("mix:") else int(v) for v in args.plen.split(",")]   # "mix:300+2300" = job k takes the k-th length, cycled


def make_prompts(plen, n):
    """n prompts of exactly plen tokens (cut from the corpus at different offsets, the first tokens are a chat header)."""
    head = eng.tokenizer.encode("<|im_start|>user\n", add_bos=False, encode_special_tokens=True)
    out = []
    stride = max(1, (len(corpus) - plen * 6) // max(n, 1))
    for i in range(n):
        seg = corpus[i * stride: i * stride + plen * 6 + 200]
        ids = eng.tokenizer.encode(seg, add_bos=False, encode_special_tokens=True)
        ids = torch.cat((head, ids), dim=-1)[:, :plen].contiguous()
        assert ids.shape[-1] == plen, (ids.shape, plen)
        out.append(ids)
    return out


def mkjob(ids, n):
    return eng.Job(input_ids=ids, max_new_tokens=n, sampler=eng._sampler({"temperature": 0.0}), stop_conditions=[])


def run_jobs(gen, items):
    """items: list of (ids, ntok); all enqueued together. Returns (token lists, wall seconds of the loop)."""
    jobs = [mkjob(i, n) for i, n in items]
    for j in jobs:
        gen.enqueue(j)
    toks = {id(j): [] for j in jobs}
    t = time.perf_counter()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r["stage"] == "streaming" and "token_ids" in r:
                toks[id(r["job"])] += r["token_ids"].flatten().tolist()
    torch.cuda.synchronize()
    return [toks[id(j)] for j in jobs], time.perf_counter() - t


def rebuild():
    """eng._build(SPEC), plus the --planes fill of the cache's QSA side planes (they outlive a Generator)."""
    eng._build(SPEC)
    if args.planes != "keep":
        with torch.inference_mode():
            for layer in eng.cache.layers.values():
                for nm in ("raw_k", "pooled"):
                    t = getattr(layer, nm, None)
                    if t is not None:
                        t.fill_(float("nan") if args.planes == "nan" else 0.0)
        torch.cuda.synchronize()


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


REF = {}   # (plen, i) -> plain greedy continuation


def stage_ref():
    eng._build(False)
    for plen in PLENS:
        for i, ids in enumerate(prompts_for(plen, args.n_prompts)):
            toks, dt = run_jobs(eng.generator, [(ids, args.tokens)])
            REF[(plen, i)] = toks[0]
        log("ref", plen, [sha(REF[(plen, i)]) for i in range(args.n_prompts)])
    save("ref", {f"{k[0]}:{k[1]}": v for k, v in REF.items()})


def stage_rinv():
    res = []
    for plen in PLENS:
        if not isinstance(plen, int):
            continue
        prompts = make_prompts(plen, min(3, args.n_prompts))
        for i, ids in enumerate(prompts):
            ref = REF.get((plen, i))
            if ref is None:
                eng._build(False)
                ref = run_jobs(eng.generator, [(ids, args.tokens)])[0][0]
            row = {"plen": plen, "prompt": i, "plain": sha(ref)}
            if SPEC:
                for name, fx in (("mix", None), ("fixd1", "1"), ("fixd3", "3"), ("fixd7", "7")):
                    if fx is None:
                        os.environ.pop("QWSPEC_FIXD", None)
                    else:
                        os.environ["QWSPEC_FIXD"] = fx
                    eng._build(True)
                    toks = run_jobs(eng.generator, [(ids, args.tokens)])[0][0]
                    row[name] = {"equal": toks == ref, "first_diff": first_diff(toks, ref), "sha": sha(toks)}
                os.environ.pop("QWSPEC_FIXD", None)
                eng._build(True)
            res.append(row)
            log("rinv", row)
    save("rinv", res)


# ---------------------------------------------------------------------------------------------------- ops
CAP = {"on": False, "armed": False, "rec": [], "cnt": {}, "gen": None, "kidx": {}, "jr": {}, "cur": 0, "snap": {}}


def cap(name, t):
    if not CAP["on"] or not isinstance(t, torch.Tensor) or torch.cuda.is_current_stream_capturing():
        return
    k = CAP["cnt"].get(name, 0)
    CAP["cnt"][name] = k + 1
    CAP["rec"].append((f"{name}#{k}", t.detach().clone().cpu()))


_hooked = [False]


def install_hooks():
    if _hooked[0]:
        return
    _hooked[0] = True
    for i, top in enumerate(eng.model.modules):
        for m in top:
            f0 = m.forward

            def mk(f0, name):
                def w(*a, **k):
                    o = f0(*a, **k)
                    if CAP["on"]:
                        cap(name, o[0] if isinstance(o, tuple) else o)
                    return o
                return w
            m.forward = mk(f0, f"{i}:{m.key}")
    # the attention core is not a module: record the dense / sparse dispatch output, the projections around it and which path ran
    from exllamav3.modules import attn as A
    d0 = A.attn_dispatch

    def d(*a, **k):
        o = d0(*a, **k)
        if CAP["on"]:
            cap("attn_dispatch", o)
        return o
    A.attn_dispatch = d
    for nm in ("project_qkv", "project_o"):
        def mkp(f0, nm):
            def w(self, *a, **k):
                o = f0(self, *a, **k)
                if CAP["on"]:
                    for i, t in enumerate(o if isinstance(o, tuple) else (o,)):
                        cap(f"{nm}.{i}", t)
                return o
            return w
        setattr(A.Attention, nm, mkp(getattr(A.Attention, nm), nm))
    b0 = A.Attention.bc_attn_step

    seen = set()

    def bs(self, *a, **k):
        o = b0(self, *a, **k)
        if o is None and (self.layer_idx, a[0].shape[0]) not in seen:
            seen.add((self.layer_idx, a[0].shape[0]))
            log("bc_attn_step declined: layer", self.layer_idx, "x", tuple(a[0].shape), a[0].dtype, "contiguous", a[0].is_contiguous(),
                "bc", {kk: (v is not False and v is not None) for kk, v in self.bc_attn.items()})
        if CAP["on"]:
            cap("bc_attn_step", torch.tensor([[1.0 if o is not None else 0.0]]) if o is None else o)
        return o
    A.Attention.bc_attn_step = bs
    fwd0 = eng.model.forward

    def fwd(*a, **k):
        x = a[0] if a else k.get("input_ids")
        p = k.get("params")
        # decode / verify forwards of the run (prefill chunks of an MTP engine pass through model.forward too: no history flag)
        eligible = (CAP["armed"] and p is not None and p.get("recurrent_history") and p.get("block_table") is not None
                    and not torch.cuda.is_current_stream_capturing())
        arm = False
        if eligible:
            g = CAP["gen"]
            ks_all = [CAP["kidx"][id(j)] for j in g.active_jobs if j.is_prefill_done()]
            cur = CAP["cur"]
            ks = ks_all[cur:cur + x.shape[0]]
            CAP["cur"] = cur + x.shape[0] if cur + x.shape[0] < len(ks_all) else 0
            rounds = [CAP["jr"].get(kk, 0) for kk in ks]
            for kk in ks:
                CAP["jr"][kk] = CAP["jr"].get(kk, 0) + 1
            arm = any(r < args.steps for r in rounds)
        if arm:
            CAP["on"] = True
            CAP["rec"] = []
            CAP["cnt"] = {}
            ids = x.clone().cpu()
            seqlens = p["cache_seqlens"].clone().cpu().tolist()
            verify = bool(p.get("dflash_verify"))
        try:
            r = fwd0(*a, **k)
        finally:
            if arm:
                CAP["on"] = False
        if arm:
            lg = (r["logits"] if isinstance(r, dict) else r).float().reshape(x.shape[0] * x.shape[-1], -1).cpu()
            rec = merge_rec(CAP["rec"])
            for kk_local, (kk, rd) in enumerate(zip(ks, rounds)):
                if rd < args.steps:
                    CAP["snap"][(kk, rd)] = dict(rec=list(rec), lg=lg, ids=ids, seqlens=seqlens, kk=kk_local, nb=len(ks), verify=verify)
        return r
    eng.model.forward = fwd


def merge_rec(rec):
    """A module the per-sequence attention calls once per sequence is recorded as name#0, name#1, ...: stack those
    one-sequence outputs back into the batched tensor, as a batched forward would have produced it."""
    by = {}
    for nm, t in rec:
        by.setdefault(nm.rsplit("#", 1)[0], []).append(t)
    out = []
    for base, ts in by.items():
        if len(ts) > 1 and all(t.dim() >= 1 and t.shape[0] == 1 and t.shape == ts[0].shape for t in ts):
            ts = [torch.cat(ts, dim=0)]
        out += [(f"{base}#{i}", t) for i, t in enumerate(ts)]
    return out


def make_oracle_gen(depth, specs):
    """A Generator whose n-gram slot answers with the plain continuation (some positions corrupted on purpose)."""
    G = eng.Generator
    val = G.__dict__["_validate_ngram_drafting_support"]
    G._validate_ngram_drafting_support = staticmethod(lambda: None)
    try:
        g = G(model=eng.model, cache=eng.cache, tokenizer=eng.tokenizer, ngram_match_min=2, num_draft_tokens=depth)
    finally:
        G._validate_ngram_drafting_support = val

    def oracle(results):
        rows = []
        for j in g.active_jobs:
            if not j.is_prefill_done():
                continue
            ref, k = specs[id(j)]
            n = j.new_tokens
            d = []
            for pos in range(depth):
                t = ref[n + pos] if n + pos < len(ref) else 0
                if (n * 7 + pos * 3 + k) % 11 == 0:
                    t = (t + 1 + pos) % eng.vocab_size
                d.append(t)
            rows.append(d)
        if not rows:
            return None
        return torch.tensor(rows, dtype=torch.long)
    g.iterate_ngram_gen = oracle
    return g


def run_capture(prompts_refs, depth, kids=None):
    """Decode the jobs together (or alone) with oracle drafts; returns {(job, round): snapshot of that verify forward}.
    kids: the job numbers (default 0..n-1); the same number gives the same corrupted draft positions alone and in a group."""
    specs = {}
    g = make_oracle_gen(depth, specs)
    jobs = []
    CAP["gen"], CAP["kidx"], CAP["jr"], CAP["cur"], CAP["snap"] = g, {}, {}, 0, {}
    for k, (ids, ref) in zip(kids if kids is not None else range(len(prompts_refs)), prompts_refs):
        j = mkjob(ids, max(8, args.steps * (depth + 1) + 2))
        specs[id(j)] = (ref, k)
        CAP["kidx"][id(j)] = k
        jobs.append(j)
    CAP["armed"] = True
    for j in jobs:
        g.enqueue(j)
    while g.num_remaining_jobs():
        for _ in g.iterate():
            pass
    CAP["armed"] = False
    return dict(CAP["snap"])


def pick(a, b, idx, nb):
    """slice of the batched tensor b (nb jobs) that belongs to job idx of the solo tensor a"""
    if a.shape == b.shape:
        return b if nb == 1 else None
    n = a.shape[0]
    if b.shape[1:] == a.shape[1:] and b.shape[0] == nb * n:
        return b[idx * n:(idx + 1) * n]
    return None


def compare(solo, batched, idx, nb, rows):
    ds = dict(solo)
    out = []
    missing = 0
    for name, b in batched:
        if ":" not in name and not args.subops:
            continue
        a = ds.get(name)
        if a is None:
            missing += 1
            continue
        b0 = pick(a, b, idx, nb)
        if b0 is None:
            out.append([name, "shape", list(a.shape), list(b.shape)])
            continue
        if not torch.equal(a, b0):
            d = float((a.float() - b0.float()).abs().max()) if a.dtype.is_floating_point else -1.0
            nrow = None
            if a.dim() >= 2 and a.shape[0] == 1:
                af, bf = a.reshape(rows, -1), b0.reshape(rows, -1)
                nrow = [bool(torch.equal(af[r], bf[r])) for r in range(rows)] if af.shape[0] == rows else None
            out.append([name, "diff", d, nrow])
    return out, missing


def prompts_for(spec, n):
    """n prompts for a --plen entry: one length, or mix:a+b (job k takes length a, b, a, ... so neighbours differ in block-table width)"""
    if isinstance(spec, int):
        return make_prompts(spec, n)
    lens = [int(v) for v in spec[4:].split("+")]
    sets = {m: make_prompts(m, n) for m in set(lens)}
    return [sets[lens[k % len(lens)]][k] for k in range(n)]


def stage_ops():
    install_hooks()
    res = {}
    for plen in PLENS:
        prompts = prompts_for(plen, args.sessions)
        refs = []
        eng._build(False)
        for i, ids in enumerate(prompts):
            refs.append(REF.get((plen, i)) or run_jobs(eng.generator, [(ids, args.tokens)])[0][0])
        for depth in [int(v) for v in args.depths.split(",")]:
            rows = depth + 1
            for arm in args.arms.split(","):
                set_arm(arm)
                solo = {}
                for k in range(args.sessions):
                    for key, v in run_capture([(prompts[k], refs[k])], depth, [k]).items():
                        solo[(k, key[1])] = v
                batched = run_capture(list(zip(prompts, refs)), depth)
                for (k, rd) in sorted(batched):
                    b = batched[(k, rd)]
                    if (k, rd) not in solo:
                        continue
                    s0 = solo[(k, rd)]
                    nb, kk = b["nb"], b["kk"]
                    same_in = bool(torch.equal(b["ids"][kk:kk + 1], s0["ids"])) and b["seqlens"][kk] == s0["seqlens"][0]
                    diffs, missing = compare(s0["rec"], b["rec"], kk, nb, rows)
                    lrow = s0["lg"].reshape(rows, -1)
                    prow = b["lg"].reshape(nb * rows, -1)[kk * rows:(kk + 1) * rows]
                    ent = {"plen": plen, "depth": depth, "rows_total": args.sessions * rows, "fwd_rows": nb * rows, "arm": arm,
                           "round": rd, "job": k, "inputs_equal": same_in, "verify_flag": [s0["verify"], b["verify"]],
                           "logits_rows_equal": [bool(torch.equal(lrow[r], prow[r])) for r in range(rows)],
                           "n_diff_ops": len(diffs), "n_unmatched": missing, "first_10": diffs[:10],
                           "first_block_diff": next((d for d in diffs if ":" in d[0] and "." not in d[0].split(":", 1)[1].split("#")[0]), None)}
                    res.setdefault(f"{plen}/{depth}/{arm}", []).append(ent)
                    log("ops", plen, "d", depth, arm, "round", rd, "job", k, "fwd_rows", nb * rows, "inputs", same_in, "logits",
                        "".join("1" if v else "0" for v in ent["logits_rows_equal"]), "n_diff", len(diffs), "first", diffs[:1])
                save("ops", res)
    eng._build(SPEC)


# ---------------------------------------------------------------------------------------------------- e2e
def stage_e2e():
    res = []
    for plen in PLENS:
        prompts = prompts_for(plen, args.n_prompts)
        for fd in args.fixd.split(","):
            if fd == "mix":
                os.environ.pop("QWSPEC_FIXD", None)
            else:
                os.environ["QWSPEC_FIXD"] = fd
            for arm in args.arms.split(","):
                set_arm(arm)
                rebuild()
                solo = [run_jobs(eng.generator, [(ids, args.tokens)])[0][0] for ids in prompts]
                rebuild()   # a fresh prompt cache: the parallel phase prefills cold, as the solo phase did
                par = []
                for g0 in range(0, len(prompts), args.sessions):
                    grp = prompts[g0:g0 + args.sessions]
                    par += run_jobs(eng.generator, [(ids, args.tokens) for ids in grp])[0]
                eq = [p == s for p, s in zip(par, solo)]
                row = {"plen": plen, "fixd": fd, "arm": arm, "sessions": args.sessions, "equal": eq, "n_equal": sum(eq), "n": len(eq),
                       "first_diff": [first_diff(p, s) for p, s in zip(par, solo)], "solo": [sha(s) for s in solo]}
                res.append(row)
                log("e2e", row)
    os.environ.pop("QWSPEC_FIXD", None)
    save("e2e", res)


# ---------------------------------------------------------------------------------------------------- slots
def stage_slots():
    """Is a difference tied to the prompt or to the slot of the group? Each prompt alone (cold), then the same prompts
    as one group in several orders (cold each time)."""
    res = []
    for plen in PLENS:
        prompts = prompts_for(plen, args.sessions)
        for arm in args.arms.split(","):
            set_arm(arm)
            rebuild()
            solo = []
            for ids in prompts:
                rebuild()
                solo.append(run_jobs(eng.generator, [(ids, args.tokens)])[0][0])
            again = []
            for ids in prompts:
                rebuild()
                again.append(run_jobs(eng.generator, [(ids, args.tokens)])[0][0])
            row = {"plen": plen, "arm": arm, "solo_repeat_equal": [a == b for a, b in zip(solo, again)], "perms": {}}
            n = len(prompts)
            for perm in ([*range(n)], [*reversed(range(n))], [*range(1, n), 0], [*range(2, n), *range(2)]):
                rebuild()
                par = run_jobs(eng.generator, [(prompts[i], args.tokens) for i in perm])[0]
                row["perms"][",".join(map(str, perm))] = [first_diff(par[slot], solo[i]) for slot, i in enumerate(perm)]
            res.append(row)
            log("slots", row)
    save("slots", res)


# ---------------------------------------------------------------------------------------------------- pf
def stage_pf():
    """Every model forward of a cold run, per prompt: input hash, cache length, rows, output hash. Alone vs as a group."""
    calls = []
    fwd0 = eng.model.forward

    def fwd(*a, **k):
        x = a[0] if a else k.get("input_ids")
        p = k.get("params") or {}
        r = fwd0(*a, **k)
        o = r["logits"] if isinstance(r, dict) else r
        h = hashlib.sha256(o.detach().cpu().float().numpy().tobytes()).hexdigest()[:8] if isinstance(o, torch.Tensor) else None
        calls.append((hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()[:8], tuple(x.shape), p["cache_seqlens"].cpu().tolist()
                      if p.get("cache_seqlens") is not None else None, bool(p.get("recurrent_history")), h))
        return r
    eng.model.forward = fwd
    for plen in PLENS:
        prompts = prompts_for(plen, args.sessions)
        for arm in args.arms.split(","):
            set_arm(arm)
            solo = []
            for ids in prompts:
                rebuild()
                del calls[:]
                toks = run_jobs(eng.generator, [(ids, args.tokens)])[0][0]
                solo.append((toks, [c for c in calls if not c[3]]))
            rebuild()
            del calls[:]
            par = run_jobs(eng.generator, [(i, args.tokens) for i in prompts])[0]
            pcalls = [c for c in calls if not c[3]]
            for slot, ids in enumerate(prompts):
                stoks, scalls = solo[slot]
                log("pf", plen, arm, "slot", slot, "tokens equal", par[slot] == stoks, "first_diff", first_diff(par[slot], stoks))
                log("pf   solo prefill forwards", scalls[:6])
                log("pf   group forwards (all)", pcalls[:12])
    eng.model.forward = fwd0


# ---------------------------------------------------------------------------------------------------- hz
def stage_hz():
    """Where is the host/device race of a group of cold prefills? Same group as the slots stage, with a device sync at one place."""
    mode = {"m": "none"}

    def sync():
        torch.cuda.synchronize()

    def wrap(obj, name, when):
        f0 = getattr(obj, name)

        def w(*a, **k):
            if mode["m"] == when + "_pre":
                sync()
            r = f0(*a, **k)
            if mode["m"] == when + "_post":
                sync()
            return r
        setattr(obj, name, w)
    wrap(eng.model, "forward", "fwd")
    wrap(eng.model, "prefill", "mprefill")
    if eng.draft_model is not None:
        wrap(eng.draft_model, "prefill", "dprefill")
    J = eng.Job
    p0 = J.prefill

    def jp(self, *a, **k):
        if mode["m"] == "job_pre":
            sync()
        r = p0(self, *a, **k)
        if mode["m"] == "job_post":
            sync()
        return r
    J.prefill = jp
    for plen in PLENS:
        prompts = prompts_for(plen, args.sessions)
        for arm in args.arms.split(","):
            set_arm(arm)
            mode["m"] = "none"
            solo = []
            for ids in prompts:
                rebuild()
                solo.append(run_jobs(eng.generator, [(ids, args.tokens)])[0][0])
            for m in ["none", "job_post", "job_pre", "fwd_post", "fwd_pre", "mprefill_post", "dprefill_post", "dprefill_pre"]:
                mode["m"] = m
                rebuild()
                par = run_jobs(eng.generator, [(i, args.tokens) for i in prompts])[0]
                log("hz", plen, arm, m, [first_diff(a, b) for a, b in zip(par, solo)])
    mode["m"] = "none"
    J.prefill = p0


# ---------------------------------------------------------------------------------------------------- speed
def stage_speed():
    res = []
    for plen in PLENS:
        prompts = prompts_for(plen, args.sessions)
        for arm in args.arms.split(","):
            set_arm(arm)
            eng._build(SPEC)
            run_jobs(eng.generator, [(prompts[0], 8)])
            ntok = args.tokens
            solo_t = []
            for _ in range(2):
                toks, dt = run_jobs(eng.generator, [(prompts[0], ntok)])
                solo_t.append((len(toks[0]), dt))
            par_t = []
            for _ in range(2):
                toks, dt = run_jobs(eng.generator, [(p, ntok) for p in prompts])
                par_t.append((sum(len(t) for t in toks), dt))
            res.append({"plen": plen, "arm": arm, "solo": solo_t, "par": par_t})
            log("speed", res[-1])
    save("speed", res)


for st in args.stages.split(","):
    log("stage", st)
    try:
        {"ref": stage_ref, "rinv": stage_rinv, "ops": stage_ops, "e2e": stage_e2e, "slots": stage_slots, "pf": stage_pf, "hz": stage_hz, "speed": stage_speed}[st]()
    except Exception:
        traceback.print_exc()
        log("stage failed", st)
log("done")
