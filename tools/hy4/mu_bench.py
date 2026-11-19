"""mu_bench.py: multi-session A/B in ONE load (run inside a job wrapper: judge idle, GPU lock, capped unit).
Phases (MU_PHASES, comma list): ref (solo legacy reference per prompt, plain + spec), plain (B = 1, 2, 4 batched plain), spec (B = 1, 2, 4 batched spec),
pf (a long request arrives mid-decode: ttft + the running sessions' largest gap per prefill chunk size), logit (batched vs single-row step logits).
Env: MU_NEW (tokens per request, default 96), MU_SLOTS, MU_CTX, MU_CHUNKS (e.g. 512,1024,2048), MU_PFLEN (long prompt tokens)."""
import json
import os
import sys
import time

tree = os.getcwd()
sys.path.insert(0, os.path.join(tree, "tools", "hy4"))
import serve  # noqa: E402

CTX = int(os.environ.get("MU_CTX", "8192"))
serve.apply_env(CTX)
import torch  # noqa: E402
import serve_core as C  # noqa: E402
from mu_engine import MuEngine, MuScheduler, Request  # noqa: E402

GGUF = "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf"
NEW = int(os.environ.get("MU_NEW", "96"))
SLOTS = int(os.environ.get("MU_SLOTS", "4"))
PH = os.environ.get("MU_PHASES", "ref,plain").split(",")
PROMPTS = ["Explain in a few short paragraphs why the sky is blue and why sunsets are red.",
           "Ecris une courte histoire sur un chat qui apprend a naviguer sur un petit bateau.",
           "Write a Python function that merges two sorted lists, then explain its complexity.",
           "List five practical tips to keep a sourdough starter healthy, with one sentence of reason each."]
mem0 = None
e = MuEngine(tree, GGUF, ctx=CTX, k=3, slots=SLOTS)
t0 = time.time()
e.load(print)
V = C.Vocab(e.meta)
ids_of = [V.encode(V.template.render([{"role": "user", "content": p}], None, True, reasoning_effort="no_think")) for p in PROMPTS]
free, tot = torch.cuda.mem_get_info()
print(f"MEM after load: free {free / 2**30:.1f} GiB of {tot / 2**30:.1f}; allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB; slabs {SLOTS} x cap {e.GR.cap}", flush=True)
RES = {}


def solo_ref(idx, spec):
    """legacy solo path of the shipped engine (GraphDecoder + MTPGraph) on slot 0: the reference stream and speed"""
    SG, MH = e.SG, e.MH
    GRm, MGm = e.GR, e.MG
    if not hasattr(e, "GR1"):
        e.GR1 = SG.GraphDecoder(e.M)
        e.GR1.prepare()
        e.MG1 = MH.MTPGraph(e.MT, e.k)
    e.GR, e.MG = e.GR1, e.MG1
    try:
        s0 = GRm.new_session(0)
        GRm.bind(s0)
        out = []
        st = e.generate(ids_of[idx], NEW, lambda t: out.extend(t) or False, speculative=spec)
    finally:
        e.GR, e.MG = GRm, MGm
    return out, st


def cmp(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) == len(b) else n


def run_batch(B, spec, chunk=1024):
    sch = MuScheduler(e, chunk=chunk, spec=spec)
    sch.start()
    reqs = [Request(ids_of[i % 4], NEW, speculative=spec) for i in range(B)]
    for r in reqs:
        sch.submit(r)
    for r in reqs:
        r.done.wait(600)
    sch.shutdown()
    sch.join(10)
    errs = [r.error for r in reqs if r.error]
    t_lo = max(r.t_first for r in reqs)
    t_hi = min(r.t_done for r in reqs)
    toks = sum(n for r in reqs for (t, n) in r.times if t_lo < t <= t_hi)
    agg = toks / max(t_hi - t_lo, 1e-9)
    per = []
    for r in reqs:
        tk = sum(n for (t, n) in r.times if t_lo < t <= t_hi)
        per.append(round(tk / max(t_hi - t_lo, 1e-9), 2))
    return reqs, dict(B=B, spec=spec, agg_tok_s=round(agg, 2), per_session=per, window_s=round(t_hi - t_lo, 2), errors=errs,
                      rounds=sch.stats["rounds"], mean_rows=round(sch.stats["round_rows"] / max(sch.stats["rounds"], 1), 2),
                      ttft=[round(r.stats.get("ttft", -1), 2) for r in reqs])


if "ref" in PH:
    REF = {}
    for spec in (False, True):
        for i in range(4):
            if spec is False and i == 0:
                solo_ref(0, False)       # warm-up (graph captures)
            if spec is True and i == 0:
                solo_ref(0, True)
            out, st = solo_ref(i, spec)
            REF[(i, spec)] = out
            tps = st["decode_tokens"] / st["decode_s"]
            print("REF", "spec" if spec else "plain", i, "tok/s", round(tps, 2), "tau", st.get("tau"), "prompt", st["prompt_tokens"], flush=True)
            RES.setdefault("ref", {})[f"{i}_{'spec' if spec else 'plain'}"] = dict(tok_s=round(tps, 2), tau=st.get("tau"))
    json.dump({f"{k[0]}_{k[1]}": v for k, v in REF.items()}, open(os.environ.get("MU_REFOUT", "/tmp/mu_ref.json"), "w"))
else:
    REF = {}
    p = os.environ.get("MU_REFOUT", "/tmp/mu_ref.json")
    if os.path.exists(p):
        REF = {(int(k.split("_")[0]), k.split("_")[1] == "True"): v for k, v in json.load(open(p)).items()}

for mode in ("plain", "spec"):
    if mode not in PH:
        continue
    spec = mode == "spec"
    # warm-up: capture every (bucket, rows) graph the arms need
    for B in (1, 2, 4):
        run_batch(B, spec)
    for B in (1, 2, 4):
        reqs, r = run_batch(B, spec)
        eq = []
        for i, q in enumerate(reqs):
            ref = REF.get((i % 4, spec))
            eq.append(cmp(q.out, ref) if ref is not None else None)
        r["equal_prefix"] = eq
        r["tokens"] = [len(q.out) for q in reqs]
        print("MU", mode, json.dumps(r), flush=True)
        RES.setdefault(mode, []).append(r)
    free, tot = torch.cuda.mem_get_info()
    print(f"MEM after {mode}: free {free / 2**30:.1f} GiB; max allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

if "logit" in PH:
    GR = e.GR
    e.phase("prefill")
    sess = []
    for i in (0, 1):
        s = GR.new_session(i)
        e.prefill_begin(s, ids_of[i], 1024)
        while not e.prefill_step(s):
            pass
        sess.append(s)
    n0 = [s.n for s in sess]
    ya, yb = sess[0].y, sess[1].y
    lg1, am1, _, _ = GR.step_slots([(sess[0], [ya])])
    a1 = lg1.clone()
    GR.rewind(sess[0], n0[0])
    lg2, am2, _, _ = GR.step_slots([(sess[0], [ya]), (sess[1], [yb])])
    a2 = lg2[0].clone()
    GR.rewind(sess[0], n0[0])
    GR.rewind(sess[1], n0[1])
    print("LOGIT row0 single vs batched: max abs diff", float((a1 - a2).abs().max()), "argmax equal", int(a1.argmax()) == int(a2.argmax()), "absmax", float(a1.abs().max()), flush=True)

if "prof" in PH:
    # eager per-class timers (synchronised): rows of 1 / 2 / 4 / 8 sessions in one step, the classes that scale with the row count are the batching cost
    GR, fw = e.GR, e.M
    e.phase("prefill")
    sess = []
    for i in range(4):
        s_ = GR.new_session(i)
        e.prefill_begin(s_, ids_of[i], 1024)
        while not e.prefill_step(s_):
            pass
        sess.append(s_)
    e.phase("decode")
    gs = GR.gs
    for T in (1, 2, 4, 8):
        rows = [(sess[r % 4], [sess[r % 4].y]) for r in range(T)] if T <= 4 else [(sess[r // 2], [sess[r // 2].y, sess[r // 2].y]) for r in range(T)][:4]
        if T == 8:
            rows = [(sess[r], [sess[r].y, sess[r].y]) for r in range(4)]
        tk, ps, bs = [], [], []
        for s_, toks in rows:
            s_.tokens.extend(toks)
            tk += toks
            ps += list(range(s_.n, s_.n + len(toks)))
            bs += [s_.slot * GR.cap] * len(toks)
        Tn = len(tk)
        gs.T, gs.Nb = Tn, 512
        GR.hin[0, :Tn] = torch.tensor(tk)
        GR.hin[1, :Tn] = torch.tensor(ps)
        GR.hin[2, :Tn] = torch.tensor(bs)
        gs.inp.copy_(GR.hin)
        fw._gs = gs
        try:
            for rep in range(3):
                fw.tm.on = True
                fw.tm.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                GR.body()
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                fw.tm.on = False
        finally:
            fw._gs = None
            for s_, toks in rows:
                del s_.tokens[s_.n:]
        print("PROF T", Tn, "eager total ms", round(dt * 1000, 1), {k: round(v * 1000, 1) for k, v in sorted(fw.tm.acc.items(), key=lambda kv: -kv[1])}, flush=True)

if "nll" in PH:
    # teacher-forced NLL per session: batched slot step (B = 4 rows, B = 1 row) vs the legacy solo graph (R = 1, R = 4 rows) and the 1e-6 embedding-noise null arms of the solo path.
    # Same text per session in every arm; per-token paired differences give the se. A batched arm inside the null band = the batched step is the same maths up to noise.
    import numpy as np
    wiki = json.load(open("/path/to/wiki_ids.json"))
    PRE, NN = int(os.environ.get("MU_PRE", "200")), int(os.environ.get("MU_NN", "240"))
    segs = [wiki[1000 * i:1000 * i + PRE + NN + 8] for i in range(4)]
    SG, MH = e.SG, e.MH
    GRm, MGm = e.GR, e.MG
    if not hasattr(e, "GR1"):
        e.GR1 = SG.GraphDecoder(e.M)
        e.GR1.prepare()
        e.MG1 = MH.MTPGraph(e.MT, e.k)

    def lp_of(lg, tgt):
        lp = torch.log_softmax(lg.reshape(len(tgt), -1).float(), -1)
        return (-lp[torch.arange(len(tgt), device=e.dev), torch.tensor(tgt, device=e.dev)]).tolist()

    def solo_nll(i, R, seed=None):
        seg = segs[i]
        e.GR, e.MG = e.GR1, e.MG1
        try:
            SG.PERT = None if seed is None else (torch.randn(e.GR1.cap, e.M.D, generator=torch.Generator().manual_seed(seed)) * 1e-6).to(e.dev)
            e.GR1.clear()
            s0 = GRm.new_session(0)
            GRm.bind(s0)
            e._t0 = time.perf_counter()
            e.reset()
            e.prefill(seg[:PRE])
            e.phase("decode")
            out, j = [], 0
            while j < NN:
                r = min(R, NN - j)
                lg, _am, _x, _h = e.GR1.step_rows(seg[PRE + j:PRE + j + r])
                out.extend(lp_of(lg, seg[PRE + j + 1:PRE + j + r + 1]))
                j += r
        finally:
            SG.PERT = None
            e.GR1.clear()
            e.GR, e.MG = GRm, MGm
        return np.array(out)

    def slot_nll(B):
        sess = []
        e.phase("prefill")
        for i in range(B):
            s_ = GRm.new_session(i)
            e.prefill_begin(s_, segs[i][:PRE], 1024)
            while not e.prefill_step(s_):
                pass
            sess.append(s_)
        e.phase("decode")
        outs = [[] for _ in range(B)]
        for j in range(NN):
            rows = [(sess[i], [segs[i][PRE + j]]) for i in range(B)]
            lg, _am, _x, _h = GRm.step_slots(rows)
            lg = lg.reshape(B, -1)
            for i in range(B):
                outs[i].extend(lp_of(lg[i], [segs[i][PRE + j + 1]]))
        return [np.array(o) for o in outs]

    arms = {}
    for i in range(4):
        for name, R, sd in (("R1", 1, None), ("R1_n1", 1, 1), ("R1_n2", 1, 2), ("R4", 4, None)):
            arms[(name, i)] = solo_nll(i, R, sd)
        print("NLL solo session", i, {n: round(float(arms[(n, i)].mean()), 4) for n in ("R1", "R1_n1", "R1_n2", "R4")}, flush=True)
    for B in (4, 1):
        o = slot_nll(B)
        for i in range(B):
            arms[(f"slotB{B}", i)] = o[i]
    for i in range(4):
        base = arms[("R1", i)]
        null = [arms[(n, i)].mean() for n in ("R1", "R1_n1", "R1_n2", "R4")]
        row = {"null_band_spread": round(float(max(null) - min(null)), 4), "R1_mean": round(float(base.mean()), 4)}
        for n in ("R1_n1", "R1_n2", "R4", "slotB4", "slotB1"):
            if (n, i) in arms:
                d = arms[(n, i)] - base
                row[n] = [round(float(d.mean()), 4), round(float(d.std(ddof=1) / np.sqrt(len(d))), 4), round(float(np.abs(d).max()), 3)]
        print("NLLD session", i, json.dumps(row), flush=True)
        RES.setdefault("nll", []).append(row)

if "pf" in PH:
    pfl = int(os.environ.get("MU_PFLEN", "2048"))
    import json as _j
    wiki = _j.load(open("/path/to/wiki_ids.json"))
    for chunk in [int(x) for x in os.environ.get("MU_CHUNKS", "512,1024,2048").split(",")]:
        for spec in (False,):
            sch = MuScheduler(e, chunk=chunk, spec=spec)
            sch.start()
            runners = [Request(ids_of[i], 200, speculative=spec) for i in range(3)]
            for r in runners:
                sch.submit(r)
            while not all(r.t_first for r in runners):
                time.sleep(0.05)
            time.sleep(1.5)
            late = Request(wiki[:pfl], 24, speculative=spec)
            ts = time.perf_counter()
            sch.submit(late)
            late.done.wait(300)
            te = time.perf_counter()
            gaps = []
            for r in runners:
                tt = [t for (t, n) in r.times if ts - 0.5 <= t <= te + 0.5]
                gaps += [b - a for a, b in zip(tt, tt[1:])]
            base = []
            for r in runners:
                tt = [t for (t, n) in r.times if ts - 3 <= t <= ts - 0.2]
                base += [b - a for a, b in zip(tt, tt[1:])]
            for r in runners:
                r.done.wait(300)
            sch.shutdown()
            sch.join(10)
            res = dict(chunk=chunk, pf_tokens=pfl, late_ttft=round(late.stats.get("ttft", -1), 2), max_gap_during_prefill=round(max(gaps), 3) if gaps else None,
                       median_gap_before=round(sorted(base)[len(base) // 2], 4) if base else None, pf_s=round(sch.stats["pf_s"], 2), late_err=late.error)
            print("PF", json.dumps(res), flush=True)
            RES.setdefault("pf", []).append(res)

print("MU_DONE", json.dumps(RES), flush=True)
