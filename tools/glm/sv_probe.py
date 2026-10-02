"""servedverify1 probe: first-call cost root cause + served-path decode A/B, one load.

Loads GLM exactly as tools/glm/serve.py (ResidentEngine, SPEED_ENV, MTP n1f2+v2) and drives
engine.generate() with chat-templated prompts, as the HTTP layer does.
Phase F: prefill sequence like serve_accept (warm 300 words, 4K x4, 16K x2, 8K, 1K), each request
         logged with every Triton autotune (key, bench time) and JIT compile (time) it triggered.
Phase D: decode A/B, arms interleaved, 3 serve_accept prompts x 400 tokens per slot,
         t/s = (tokens - 1) / (last chunk - first chunk), greedy text hash per arm.
Phase P: per-forward chunk log + cProfile. Phase S: GPU busy/sclk, CPU and thread-stack sampler, CUDA-event
         time per module class, vmstat and KFD eviction deltas, over repeated F sequences (SV_SREP).
Usage: python tools/glm/sv_probe.py <model> <out.json> [phases=FD|P|S] [n_cycles=2]
"""
import os, sys, json, time, random, statistics, hashlib, asyncio
from pathlib import Path

M, OUT = sys.argv[1], sys.argv[2]
PHASES = sys.argv[3] if len(sys.argv) > 3 else "FD"
NCYC = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools/glm"))
import serve as S
for k, v in S.SPEED_ENV.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")

R = {"env": {k: os.environ.get(k) for k in list(S.SPEED_ENV) + ["EXL3_MOE_UNION_V2", "TRITON_CACHE_AUTOTUNING"]},
     "events": [], "F": [], "D": {"slots": [], "summary": {}}}
def save(): Path(OUT).write_text(json.dumps(R, indent=1))

# ---- instrumentation ---------------------------------------------------------------------------
import triton
from triton.runtime import autotuner as AT, jit as JT
EV = []
_orig_run = AT.Autotuner.run
def _run(self, *a, **kw):
    n0 = len(self.cache)
    t0 = time.perf_counter()
    out = _orig_run(self, *a, **kw)
    if len(self.cache) != n0:
        EV.append({"kind": "autotune", "fn": self.base_fn.__name__, "dt": round(time.perf_counter() - t0, 3),
                   "bench": round(getattr(self, "bench_time", 0.0), 3), "key": str(list(self.cache)[-1])[:160],
                   "nconf": len(self.configs)})
    return out
AT.Autotuner.run = _run
_orig_cc = JT.JITFunction._do_compile
def _cc(self, *a, **kw):
    t0 = time.perf_counter()
    out = _orig_cc(self, *a, **kw)
    EV.append({"kind": "compile", "fn": self.__name__, "dt": round(time.perf_counter() - t0, 3)})
    return out
JT.JITFunction._do_compile = _cc

def take_events():
    ev = list(EV); EV.clear(); return ev

def summarize(ev):
    by = {}
    for e in ev:
        k = (e["kind"], e["fn"])
        c = by.setdefault(k, [0, 0.0]); c[0] += 1; c[1] += e["dt"]
    return sorted(([f"{k[0]}:{k[1]}", n, round(t, 2)] for k, (n, t) in by.items()), key=lambda x: -x[2])

# ---- load ------------------------------------------------------------------------------------
t0 = time.perf_counter()
eng = S.ResidentEngine(M, max_history=1, max_ctx=int(os.environ.get("SV_MAX_CTX", "524288")))
R["load_s"] = round(time.perf_counter() - t0, 1)
ev = take_events(); R["events"].append({"phase": "load", "n": len(ev), "summary": summarize(ev)})
print(f"load {R['load_s']} s, events {len(ev)}", flush=True); save()
template = (Path(M) / "chat_template.jinja").read_text(encoding="utf-8")
loop = asyncio.new_event_loop()

def request(content, max_tokens):
    prompt = S.render_prompt(template, [{"role": "user", "content": content}], None)
    ntok = eng.count_tokens(prompt)
    async def go():
        t = time.perf_counter(); first = last = None; text = ""; chunks = 0
        async for d in eng.generate(prompt, max_tokens=max_tokens, temperature=0.0, top_p=1.0, stop=[]):
            now = time.perf_counter(); first = first or now; last = now; text += d; chunks += 1
        return t, first, last, text, chunks
    t, first, last, text, chunks = loop.run_until_complete(go())
    end = time.perf_counter()
    st = getattr(eng, "last_stats", None) or {}
    n = int(st.get("new_tokens", 0)) or chunks
    return {"prompt_tokens": ntok, "wall_s": round(end - t, 3), "ttft_s": round((first or end) - t, 3),
            "new_tokens": n, "tps": round((n - 1) / (last - first), 3) if first and last and last > first else None,
            "hash": hashlib.sha1(text.encode()).hexdigest()[:12], "text": text[:80]}

# ---- phase F: first-call costs ---------------------------------------------------------------
CORPUS = Path.home() / "bench/ppl/wiki.test.raw"
words = CORPUS.read_text(encoding="utf-8").split()
rng = random.Random(4321)
def wiki(n):
    s = rng.randrange(0, len(words) - n)
    return f"[{rng.random()}] " + " ".join(words[s:s + n]) + "\nSummarize in one word."
if "F" in PHASES:
    seq = [("warm", 300)] + [("4k", 3000)] * 4 + [("16k", 12000)] * 2 + [("8k", 6000), ("1k", 800), ("32k", 24000)]
    for label, n in seq:
        r = request(wiki(n), 1)
        ev = take_events()
        r.update(label=label, n_events=len(ev), ev_time=round(sum(e["dt"] for e in ev), 2),
                 summary=summarize(ev)[:12], detail=ev[:80])
        R["F"].append(r); save()
        print(f"[F {label}] ptok {r['prompt_tokens']} wall {r['wall_s']} s ({r['prompt_tokens'] / r['wall_s']:.0f} t/s) "
              f"events {len(ev)} ev_time {r['ev_time']} top {r['summary'][:4]}", flush=True)

# ---- phase P: chunk log + cProfile on the slow-request pattern ---------------------------------
if "P" in PHASES:
    import cProfile, pstats, io, torch
    from exllamav3 import Model
    CH = []
    def wrap(name):
        orig = getattr(Model, name)
        def f(self, *a, **kw):
            ids = kw.get("input_ids", a[0] if a else None)
            torch.cuda.synchronize(); t = time.perf_counter()
            out = orig(self, *a, **kw)
            torch.cuda.synchronize()
            CH.append([name, getattr(self, "component", None) or "", tuple(ids.shape) if ids is not None else None,
                       round(time.perf_counter() - t, 3)])
            return out
        setattr(Model, name, f)
    wrap("prefill"); wrap("forward")
    from exllamav3.generator import pagetable as PTm, generator as GMm
    def twrap(cls, name, tag):
        orig = getattr(cls, name)
        def f(self, *a, **kw):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = orig(self, *a, **kw)
            torch.cuda.synchronize()
            dt = round(time.perf_counter() - t, 3)
            if dt > 0.02: CH.append([tag, "", None, dt])
            return out
        setattr(cls, name, f)
    twrap(PTm.PageTable, "defrag", "DEFRAG"); twrap(GMm.Generator, "on_queue_drained", "DRAIN")
    twrap(GMm.Generator, "iterate", "ITER")
    rng2 = random.Random(4321)
    draws = [(lbl, n) for lbl, n in [("warm", 300)] + [("4k", 3000)] * 4 + [("16k", 12000)] * 2 + [("8k", 6000), ("1k", 800)]]
    texts = []
    for lbl, n in draws:
        s0 = rng2.randrange(0, len(words) - n); texts.append((lbl, f"[{rng2.random()}] " + " ".join(words[s0:s0 + n]) + "\nSummarize in one word."))
    A8 = texts[7][1]
    seq = texts if os.environ.get("SV_PSEQ") == "F" else [("warm", texts[0][1]), ("8kA", A8), ("8kA'", "Zebra " + A8), ("8kA''", "Yak " + A8), ("1k", texts[8][1]), ("4k", texts[1][1])]
    for lbl, txt in seq:
        CH.clear(); pr = cProfile.Profile(); pr.enable()
        r = request(txt, 1)
        pr.disable()
        buf = io.StringIO(); pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(18)
        r.update(label=lbl, chunks=list(CH), reserved_gb=round(torch.cuda.memory_reserved() / 2**30, 2),
                 prof=buf.getvalue()[-6000:] if r["wall_s"] > 15 else "")
        R.setdefault("P", []).append(r); save()
        print(f"[P {lbl}] ptok {r['prompt_tokens']} wall {r['wall_s']} reserved {r['reserved_gb']} chunks "
              f"{[(c[0][0], c[1][:3], c[2][-1] if c[2] else None, c[3]) for c in CH if c[3] > 0.05]}", flush=True)
        if r["prof"]: print(r["prof"][-3500:], flush=True)

# ---- phase S: sampler (GPU busy/sclk, process CPU, main-thread Python stack) over repeated F sequences
if "S" in PHASES:
    import threading, glob, traceback, collections
    main_id = threading.main_thread().ident
    dev = (glob.glob("/sys/class/drm/card*/device/gpu_busy_percent") or [None])[0]
    sclk = dev and dev.replace("gpu_busy_percent", "pp_dpm_sclk")
    SAMP = []; stop = [False]
    def rd(p):
        try: return open(p).read()
        except Exception: return ""
    def sampler():
        last = os.times()
        while not stop[0]:
            time.sleep(0.5)
            now = os.times(); cpu = (now.user + now.system - last.user - last.system) / 0.5; last = now
            st = []
            for tid, fr in sys._current_frames().items():
                if tid in (main_id, threading.get_ident()): continue
                ex = traceback.extract_stack(fr)
                if any("exllamav3" in f.filename for f in ex):
                    st = [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in ex[-7:]]
            clk = [l for l in rd(sclk).splitlines() if "*" in l]
            SAMP.append([round(time.perf_counter(), 2), rd(dev).strip(), clk[0] if clk else "", round(cpu, 2), st])
    th = threading.Thread(target=sampler, daemon=True); th.start()
    import torch
    EVT = []
    if os.environ.get("SV_EVT", "1") == "1":
        seen = 0
        for mod in eng.model:
            cn = type(mod).__name__
            if cn in ("Linear",): continue
            of = mod.forward
            def mk(of, cn):
                def f(x, *a, **kw):
                    rows = x.shape[-2] if hasattr(x, "shape") and x.dim() >= 2 else 0
                    if rows < 64 or torch.cuda.is_current_stream_capturing():
                        return of(x, *a, **kw)
                    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                    e0.record(); out = of(x, *a, **kw); e1.record()
                    EVT.append((cn, e0, e1))
                    return out
                return f
            mod.forward = mk(of, cn); seen += 1
        print("wrapped", seen, "modules", flush=True)
    rng3 = random.Random(4321)
    base = [("warm", 300)] + [("4k", 3000)] * 4 + [("16k", 12000)] * 2 + [("8k", 6000), ("1k", 800)]
    NREP = int(os.environ.get("SV_SREP", "2"))
    for rep in range(NREP):
        for lbl, n in base:
            s0 = rng3.randrange(0, len(words) - n); txt = f"[{rng3.random()}] " + " ".join(words[s0:s0 + n]) + "\nSummarize in one word."
            def vm():
                d = {}
                for l in rd("/proc/vmstat").splitlines():
                    k, v = l.split()
                    if k.startswith(("compact_", "pgmigrate", "thp_", "numa_", "pswp", "pgsteal", "pgscan", "pgfault", "pgmajfault")): d[k] = int(v)
                for f in glob.glob(f"/sys/class/kfd/kfd/proc/{os.getpid()}/stats_*/evicted_ms"): d["kfd_evicted_ms"] = int(rd(f) or 0)
                return d
            v0 = vm()
            SAMP.clear(); t0 = time.perf_counter()
            r = request(txt, 1)
            torch.cuda.synchronize()
            fam = collections.defaultdict(float)
            for cn, e0, e1 in EVT: fam[cn] += e0.elapsed_time(e1) / 1000
            EVT.clear()
            r["fam"] = {k: round(v, 3) for k, v in sorted(fam.items(), key=lambda kv: -kv[1])}
            v1 = vm(); r["vm"] = {k: v1[k] - v0.get(k, 0) for k in v1 if v1[k] - v0.get(k, 0)}
            smp = list(SAMP)
            rate = r["prompt_tokens"] / r["wall_s"]
            slow = rate < 300 and r["prompt_tokens"] > 2000 or (r["prompt_tokens"] < 2000 and r["wall_s"] > 6)
            busy = [int(x[1]) for x in smp if x[1].isdigit()]
            cpu = [x[3] for x in smp]
            stacks = collections.Counter(" < ".join(x[4][-4:][::-1]) for x in smp)
            r.update(label=lbl, rep=rep, slow=slow, busy_mean=round(statistics.mean(busy), 1) if busy else None,
                     cpu_mean=round(statistics.mean(cpu), 2) if cpu else None, clk=collections.Counter(x[2] for x in smp).most_common(3),
                     stacks=stacks.most_common(8) if slow else stacks.most_common(2), ev=summarize(take_events())[:5])
            R.setdefault("S", []).append(r); save()
            print(f"[S{rep} {lbl}] ptok {r['prompt_tokens']} wall {r['wall_s']} ({rate:.0f} t/s) slow {slow} busy {r['busy_mean']} "
                  f"cpu {r['cpu_mean']} clk {r['clk'][:2]}", flush=True)
            print("    fam", list(r["fam"].items())[:8], flush=True)
            print("    vm", {k: v for k, v in r["vm"].items() if k in ("compact_stall", "compact_migrate_scanned", "pgmigrate_success", "kfd_evicted_ms", "thp_fault_alloc", "numa_hint_faults", "pswpout", "pgsteal_kswapd", "pgmajfault")}, flush=True)
            if slow:
                for st_, n_ in r["stacks"]: print(f"    {n_:3d} {st_}", flush=True)
    stop[0] = True

# ---- phase D: decode A/B ---------------------------------------------------------------------
from exllamav3.modules import block_graph as BG
FLAGS = ["EXL3_DEC_DSA_FAST", "EXL3_MIDCHUNK_CKPT", "EXL3_PREFILL_BIG_TAIL", "EXL3_BIG_TAIL_DENSE"]
NEW = {k: S.SPEED_ENV[k] for k in FLAGS}
ARMS = {"old": {k: "0" for k in FLAGS}, "new": dict(NEW), "nodsa": dict(NEW, EXL3_DEC_DSA_FAST="0")}
if os.environ.get("SV_ARMS"):
    ARMS = {k: v for k, v in ARMS.items() if k in os.environ["SV_ARMS"].split(",")}
PROMPTS = S_PROMPTS = {
    "prose": "Write a 300-word story about a lighthouse.",
    "chat": "Explain how a hash map works, with its time complexity, in about 300 words.",
    "code": "Write a Python function that parses an ISO-8601 duration string into seconds, with tests.",
}
def set_arm(a):
    import torch
    torch.cuda.synchronize()
    os.environ.update(ARMS[a])
    BG.purge()
if "D" in PHASES:
    names = list(ARMS)
    order = []
    for c in range(NCYC):
        order += names + names[::-1]
    R["D"]["order"] = order; R["D"]["arms"] = ARMS
    for a in names:  # untimed warm pass per arm (graphs, autotune), also the greedy reference
        set_arm(a); request(PROMPTS["prose"], 32)
    take_events()
    for i, a in enumerate(order):
        set_arm(a)
        la = os.getloadavg()[0]
        row = {"slot": i, "arm": a, "load1": round(la, 2)}
        for p, txt in PROMPTS.items():
            r = request(txt, 400)
            row[p] = {k: r[k] for k in ("tps", "new_tokens", "ttft_s", "hash")}
        row["events"] = summarize(take_events())
        R["D"]["slots"].append(row); save()
        print(f"[D {i:2d} {a:6s}] load {la:.2f} " + " ".join(f"{p} {row[p]['tps']}" for p in PROMPTS), flush=True)
    summ = {}
    for a in names:
        rows = [r for r in R["D"]["slots"] if r["arm"] == a]
        summ[a] = {p: {"median": statistics.median(r[p]["tps"] for r in rows),
                       "min": min(r[p]["tps"] for r in rows), "max": max(r[p]["tps"] for r in rows),
                       "hashes": sorted({r[p]["hash"] for r in rows})} for p in PROMPTS}
        summ[a]["mean_of_medians"] = round(statistics.mean(summ[a][p]["median"] for p in PROMPTS), 3)
        summ[a]["n"] = len(rows)
    R["D"]["summary"] = summ; save()
    for a in names:
        print(f"RESULT {a}: " + " ".join(f"{p} {summ[a][p]['median']:.2f} [{summ[a][p]['min']:.2f}-{summ[a][p]['max']:.2f}]"
                                         for p in PROMPTS) + f" mean {summ[a]['mean_of_medians']} n={summ[a]['n']}", flush=True)
save(); print("DONE", flush=True)
