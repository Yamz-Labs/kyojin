"""hostidle2 census: per-round ATen-op counts by python call site, plus real launch counts.

Served config (serve.py SPEED_ENV + MTP n1f2 + UNION_V2), one load, then:
  phase 1  TorchDispatchMode over a 96-token greedy decode -> ops/round grouped by the first
           exllamav3/our frame (file:lineno) that issued them. Finds the per-call torch.tensor /
           zeros / copy_ / add sites without guessing.
  phase 2  torch.profiler over a 32-token decode -> hipLaunchKernel / hipModuleLaunchKernel /
           aten::copy_ etc. counts per round (the launch metric the brief asks for).

Usage: python host2_census.py <model> <out.txt> [PHASES]
"""
import os, sys, json, collections, time, ast, traceback

M, OUT = sys.argv[1], sys.argv[2]
PHASES = (sys.argv[3] if len(sys.argv) > 3 else "1,2")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools/glm"))
_src = open(os.path.join(ROOT, "tools/glm/serve.py")).read()
SPEED_ENV = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SPEED_ENV")
for k, v in SPEED_ENV.items(): os.environ.setdefault(k, os.path.expanduser(v))
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
os.environ.setdefault("EXL3_HOST_LEAN", os.environ.get("H2_LEAN", "1"))
os.environ.setdefault("EXL3_HOST_CUTS", os.environ.get("H2_CUTS", "0"))
import torch
torch.set_grad_enabled(False)
import exllamav3
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator import generator as GM
from exllamav3.generator.sampler import GreedySampler
import serve as SV
from mtp_bench_prompts import PROMPTS

cfg = Config.from_directory(M)
model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(model, max_num_tokens=16384, max_history=2)
model.load(device="cuda:0", progressbar=False)
draft = Model.from_config(cfg, component="mtp")
dcache = Cache(draft, max_num_tokens=16384, max_history=2)
draft.load(device="cuda:0", progressbar=False)
GM.MTP_FUSE_CATCHUP = 2
template = open(os.path.join(M, "chat_template.jinja")).read()
def enc(text):
    p = SV.render_prompt(template, [{"role": "user", "content": text}], None)
    return tok.encode(p, encode_special_tokens=True)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache,
                num_draft_tokens=1, record_draft_stats=True)
PROMPTS_FLAT = [PROMPTS[c][i] for c in ("code", "chat", "prose") for i in range(4)]
IDS = [enc(p) for p in PROMPTS_FLAT]
print("loaded", flush=True)

def run(ids, n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), stop_conditions=[])
    gen.enqueue(job)
    toks = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            tid = r.get("token_ids")
            if tid is not None and tid.numel(): toks.append(tid.cpu())
    torch.cuda.synchronize()
    t = torch.cat(toks, -1).flatten()
    return {"ids": t.tolist(), "ntok": int(t.numel()),
            "rounds": max(1, len(getattr(job, "draft_stats", []) or []))}

# warm up every site we will count
for i in range(3): run(IDS[i][:, :16], 4)
run(IDS[0], 32)

out = open(OUT, "w")
def emit(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()

# ---------------- phase 1: op calls per round, by call site ----------------
if "1" in PHASES.split(","):
    from torch.utils._python_dispatch import TorchDispatchMode
    SKIP = ("aten::", "prims::")
    def site():
        f = sys._getframe(1)
        while f is not None:
            fn = f.f_code.co_filename
            if (fn != __file__ and "/torch/" not in fn and "torch/_" not in fn
                    and not fn.startswith("<")):
                return f.f_code.co_filename, f.f_lineno
            f = f.f_back
        return "?", 0
    counts = collections.Counter()
    opsites = collections.defaultdict(collections.Counter)
    class Census(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            s = site()
            counts[s] += 1
            opsites[s][str(func)] += 1
            return func(*args, **(kwargs or {}))
    rounds = 0
    with Census():
        for i in range(4):
            r = run(IDS[i], 96)
            rounds += r["rounds"]
    emit(f"\n== phase1: {rounds} rounds, {sum(counts.values())} op calls "
         f"({sum(counts.values())/rounds:.1f}/round) ==")
    tot = collections.Counter()
    for s, n in counts.items():
        for o, k in opsites[s].items(): tot[o] += k
    emit("-- ops/round (all sites) --")
    for o, k in tot.most_common(40): emit(f"  {k/rounds:8.2f}/rd  {o}")
    emit("-- sites/round (top 60) --")
    for s, n in counts.most_common(60):
        f = s[0].replace(ROOT + "/", "")
        emit(f"  {n/rounds:8.2f}/rd  {f}:{s[1]}  " + ", ".join(f"{o}x{c}" for o, c in opsites[s].most_common(3)))

# ---------------- phase 2: real launch counts per round ----------------
if "2" in PHASES.split(","):
    from torch.profiler import profile, ProfilerActivity
    r = run(IDS[4], 8)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        r = run(IDS[4], 48)
    rounds = max(1, r["rounds"])
    ka = prof.key_averages()
    emit(f"\n== phase2: {rounds} rounds, lean={os.environ.get('EXL3_HOST_LEAN')} "
         f"cuts={os.environ.get('EXL3_HOST_CUTS')} ==")
    rows = []
    for e in ka:
        n = e.count / rounds
        scpu = (getattr(e, "self_cpu_time_total", 0) or 0) / 1000.0 / rounds
        dev = (getattr(e, "self_device_time_total", 0) or 0) / 1000.0 / rounds
        rows.append((n, e.key, scpu, dev))
    tot_launch = 0
    for n, k, scpu, dev in rows:
        if "hipLaunchKernel" in k or "hipModuleLaunchKernel" in k: tot_launch += n
    emit(f"  launches/round (hipLaunch+hipModuleLaunch) = {tot_launch:.0f}")
    emit(f"  {'calls/rd':>9} {'msCPU/rd':>9} {'msGPU/rd':>9}  name")
    for n, k, scpu, dev in sorted(rows, key=lambda t: -t[0])[:45]:
        emit(f"  {n:9.2f} {scpu:9.3f} {dev:9.3f}  {k[:78]}")
    cpu_tot = sum(t[2] for t in rows)
    emit(f"  total self CPU {cpu_tot:.2f} ms/round over profiled ops")
    for e in sorted(rows, key=lambda t: -t[0]):
        if e[1].startswith("Memcpy"):
            emit(f"  memcpy {e[0]:8.2f}/rd  cpu {e[2]:.3f} ms/rd  gpu {e[3]:.3f} ms/rd  {e[1]}")
    # aggregate launches
    for want in ("hipLaunchKernel", "hipModuleLaunchKernel", "aten::copy_", "aten::fill_",
                 "aten::zeros", "aten::_to_copy", "aten::empty", "cudaMemcpy", "aten::item",
                 "aten::_local_scalar_dense", "aten::add", "aten::mul", "aten::add_", "aten::mm",
                 "aten::cat", "aten::index", "aten::view", "aten::reshape", "aten::transpose",
                 "aten::slice", "aten::expand", "aten::where", "aten::sum", "aten::softmax"):
        s = sum(n for n, k, _, _ in rows if want in k)
        if s: emit(f"  agg {want:26s} {s:9.2f}/rd")
out.close()
print("DONE", flush=True)
