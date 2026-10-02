"""stall1 probe: catch the random ~40 s served stall and record what differs on the slow request.

Loads GLM like tools/glm/serve.py (ResidentEngine, SPEED_ENV, warm-up) and sends N unique
chat-templated prompts of mixed size through engine.generate(). Per request it keeps:
  - 10 Hz sysfs sampler: sclk, mclk, fclk, socclk, power, temp, gpu busy
  - process utime/stime, /proc/vmstat deltas, kfd evicted_ms, MemFree, buddyinfo high orders
  - torch allocator deltas (reserved, device allocs, segments, retries)
  - CUDA-event time per module family (rows >= 64) and per prefill-GEMM weight buffer
    (ext.hgemm_recon, keyed by the w data_ptr), plus the reconstruct kernels
Usage: python tools/glm/stall_probe.py <model> <out.json> [n_req=60] [seed=7]
Env: SP_WORDS=800,3000,6000,12000 (prompt sizes, words), SP_MAXTOK=1, SP_EVT=1, SP_PRIME=0, SP_MAX_CTX=524288
"""
import os, sys, json, time, random, statistics, hashlib, asyncio, threading, glob, collections
from pathlib import Path

M, OUT = sys.argv[1], sys.argv[2]
NREQ = int(sys.argv[3]) if len(sys.argv) > 3 else 60
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools/glm"))
import serve as S
for k, v in S.SPEED_ENV.items():
    os.environ.setdefault(k, v)
import torch

R = {"env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("EXL3_", "PYTORCH_", "HSA_", "HIP_", "SP_"))},
     "argv": sys.argv, "req": []}
def save(): Path(OUT).write_text(json.dumps(R, indent=1))
def rd(p):
    try: return open(p).read()
    except Exception: return ""

DEV = (glob.glob("/sys/class/drm/card*/device/gpu_busy_percent") or [""])[0].replace("gpu_busy_percent", "")
HW = (glob.glob(DEV + "hwmon/hwmon*/") or [""])[0]
def cur(f):
    for l in rd(DEV + f).splitlines():
        if "*" in l: return int(l.split(":")[1].strip().rstrip("*").strip().lower().replace("mhz", ""))
    return None
def sample():
    return [round(time.perf_counter(), 2), cur("pp_dpm_sclk"), cur("pp_dpm_mclk"), cur("pp_dpm_fclk"), cur("pp_dpm_socclk"),
            int(rd(HW + "power1_average").strip() or rd(HW + "power1_input").strip() or 0) // 1000000,
            int(rd(HW + "temp1_input").strip() or 0) // 1000, int(rd(DEV + "gpu_busy_percent").strip() or -1)]
SAMP = []; stop = [False]
def sampler():
    while not stop[0]:
        SAMP.append(sample()); time.sleep(0.1)

VMK = ("compact_", "pgmigrate", "thp_", "pswp", "pgsteal", "pgscan", "pgfault", "pgmajfault", "allocstall", "kswapd", "workingset_refault")
def sysstate():
    d = {}
    for l in rd("/proc/vmstat").splitlines():
        k, v = l.split()
        if k.startswith(VMK): d[k] = int(v)
    for f in glob.glob(f"/sys/class/kfd/kfd/proc/{os.getpid()}/stats_*/evicted_ms"): d["kfd_evicted_ms"] = int(rd(f) or 0)
    t = os.times(); d["utime_ms"] = int(t.user * 1000); d["stime_ms"] = int(t.system * 1000)
    for l in rd("/proc/meminfo").splitlines():
        if l.startswith(("MemFree:", "MemAvailable:", "SwapFree:")): d[l.split(":")[0]] = int(l.split()[1]) // 1024
    for l in rd("/proc/self/status").splitlines():
        if l.startswith("VmRSS:"): d["rss_mb"] = int(l.split()[1]) // 1024
    for l in rd("/proc/buddyinfo").splitlines():
        if "Normal" in l: c = [int(v) for v in l.split()[4:]]; d["free_o9plus"] = sum(c[9:]); d["free_o0_3"] = sum(c[:4])
    ms = torch.cuda.memory_stats()
    for k in ("num_device_alloc", "num_device_free", "num_alloc_retries", "reserved_bytes.all.current",
              "segment.all.current", "num_sync_all_streams", "allocated_bytes.all.peak"):
        d["t_" + k] = int(ms.get(k, 0))
    return d
ABS = ("MemFree", "MemAvailable", "SwapFree", "rss_mb", "free_o9plus", "free_o0_3", "t_reserved_bytes.all.current", "t_segment.all.current")

# ---- load + warm-up exactly as serve.py ------------------------------------------------------
t0 = time.perf_counter()
eng = S.ResidentEngine(M, max_history=1, max_ctx=int(os.environ.get("SP_MAX_CTX", "524288")))
template = (Path(M) / "chat_template.jinja").read_text(encoding="utf-8")
R["load_s"] = round(time.perf_counter() - t0, 1)
if os.environ.get("EXL3_SERVE_WARMUP", "1") != "0":
    R["warmup_s"] = round(S.warmup(eng, template), 1)
    if os.environ.get("SP_PRIME", "0") == "1":  # the serve.py dense GEMM prime (stall1 fix)
        R["prime"] = S.prime_dense_tune(float(os.environ.get("EXL3_SERVE_DTUNE_PRIME_S", "600")))
print(f"load {R['load_s']} s warmup {R.get('warmup_s')} prime {R.get('prime')}", flush=True); save()
loop = asyncio.new_event_loop()

# ---- CUDA-event instrumentation ----------------------------------------------------------------
EVT = []; GEMM = []
if os.environ.get("SP_EVT", "1") == "1":
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
                e0.record(); out = of(x, *a, **kw); e1.record(); EVT.append((cn, e0, e1))
                return out
            return f
        mod.forward = mk(of, cn)
    from exllamav3.modules.quant import exl3 as QX
    class ExtProxy:
        def __init__(self, real): self._r = real
        def __getattr__(self, n):
            f = getattr(self._r, n)
            if n not in ("hgemm_recon", "reconstruct_had_slice", "reconstruct", "reconstruct_slice"): return f
            def g(*a):
                if torch.cuda.is_current_stream_capturing(): return f(*a)
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record(); out = f(*a); e1.record()
                w = a[1] if n == "hgemm_recon" else a[0]
                GEMM.append((n, w.data_ptr(), w.numel(), a[0].shape[0] if n == "hgemm_recon" else 0, e0, e1))
                return out
            return g
    QX.ext = ExtProxy(QX.ext)

# ---- request loop ------------------------------------------------------------------------------
words = (Path.home() / "bench/ppl/wiki.test.raw").read_text(encoding="utf-8").split()
rng = random.Random(SEED)
SIZES = [int(v) for v in os.environ.get("SP_WORDS", "800,3000,6000,12000").split(",")]
MAXTOK = int(os.environ.get("SP_MAXTOK", "1"))
th = threading.Thread(target=sampler, daemon=True); th.start()

def request(content):
    prompt = S.render_prompt(template, [{"role": "user", "content": content}], None)
    ntok = eng.count_tokens(prompt)
    async def go():
        t = time.perf_counter(); first = None; text = ""
        async for d in eng.generate(prompt, max_tokens=MAXTOK, temperature=0.0, top_p=1.0, stop=[]):
            first = first or time.perf_counter(); text += d
        return t, first, text
    t, first, text = loop.run_until_complete(go())
    torch.cuda.synchronize()
    return ntok, t, first, time.perf_counter(), text

for i in range(NREQ):
    n = rng.choice(SIZES)
    s0 = rng.randrange(0, len(words) - n)
    txt = f"[{i} {rng.random()}] " + " ".join(words[s0:s0 + n]) + "\nSummarize in one word."
    v0 = sysstate(); SAMP.clear(); tw = time.time()
    ntok, t, first, end, text = request(txt)
    v1 = sysstate(); smp = list(SAMP)
    fam = collections.defaultdict(float)
    for cn, e0, e1 in EVT: fam[cn] += e0.elapsed_time(e1)
    EVT.clear()
    gk = collections.defaultdict(lambda: [0, 0.0]); gw = collections.defaultdict(lambda: [0, 0.0, 0, 0])
    for nm, ptr, numel, rows, e0, e1 in GEMM:
        ms = e0.elapsed_time(e1); gk[nm][0] += 1; gk[nm][1] += ms
        if nm == "hgemm_recon":
            c = gw[ptr]; c[0] += 1; c[1] += ms; c[2] = numel; c[3] += rows
    GEMM.clear()
    wall = end - t
    r = {"i": i, "t_wall0": round(tw, 2), "ptok": ntok, "wall": round(wall, 3), "tps": round(ntok / wall, 1), "hash": hashlib.sha1(text.encode()).hexdigest()[:10],
         "d": {k: v1[k] - v0.get(k, 0) for k in v1 if k not in ABS and v1[k] != v0.get(k, 0)},
         "abs0": {k: v0[k] for k in ABS if k in v0}, "abs1": {k: v1[k] for k in ABS if k in v1},
         "fam_ms": {k: round(v, 1) for k, v in sorted(fam.items(), key=lambda kv: -kv[1])},
         "kern_ms": {k: [v[0], round(v[1], 1)] for k, v in gk.items()},
         "gemm_by_w": sorted([[hex(p), c[0], round(c[1], 1), c[2], c[3]] for p, c in gw.items()], key=lambda x: -x[2])[:12],
         "samp": smp}
    cols = list(zip(*smp)) if smp else [[]] * 8
    def st(c):
        c = [v for v in c if v is not None]
        return [min(c), round(statistics.mean(c)), max(c)] if c else None
    r["clk"] = {nm: st(cols[j]) for j, nm in enumerate(["t", "sclk", "mclk", "fclk", "socclk", "power", "temp", "busy"]) if j}
    R["req"].append(r); save()
    print(f"[{i:3d}] ptok {ntok:6d} wall {wall:7.2f} {r['tps']:6.0f} t/s  sclk {r['clk']['sclk']} mclk {r['clk']['mclk']} "
          f"fclk {r['clk']['fclk']} P {r['clk']['power']} T {r['clk']['temp']} stime {r['d'].get('stime_ms', 0)} "
          f"resv+ {(r['abs1'].get('t_reserved_bytes.all.current', 0) - r['abs0'].get('t_reserved_bytes.all.current', 0)) >> 20}M "
          f"dalloc {r['d'].get('t_num_device_alloc', 0)} kswapd {r['d'].get('pgsteal_kswapd', 0)} mig {r['d'].get('pgmigrate_success', 0)} "
          f"o9+ {r['abs0'].get('free_o9plus')} free {r['abs0'].get('MemFree')}M avail {r['abs0'].get('MemAvailable')}M rss {r['abs1'].get('rss_mb')}M {time.strftime('%T')}", flush=True)
stop[0] = True
save(); print("DONE", flush=True)
