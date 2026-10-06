# glm-mtp D1-1 rounddecomp: decompose ONE served MTP round (ndt 1, verify R=2) into
#   draft head / verify per family / catch-up / host gaps / D2H syncs,
# and attribute the GPU idle inside the round. Run under rocprofv3 --kernel-trace.
#
# Method
#  * served config: ndt=1, MTP_FUSE_CATCHUP=2, EXL3_MOE_UNION_V2=1 (serve.py:150-158).
#  * 4K prefill, then --tokens decode steps of the same Generator, greedy.
#  * GPU-timeline phase boundaries: a marker memset of a unique byte size is enqueued at
#    every phase edge (generator.iterate entry, draft fwd entry/exit, verify fwd entry/exit,
#    catch-up prefill entry/exit, sample/accept entry/exit). Markers are 1-2 us, appear in the
#    kernel trace as Memset ops, and give an exact common clock for host phases and GPU kernels.
#  * host side: wall clock per phase, and counters of device->host sync points
#    (Tensor.item / .cpu() / .tolist() / blocking D2H copy) attributed to the running phase.
#  * every phase window is written to the json so the analyzer can slice the trace.
import argparse, collections, json, os, sys, time, torch

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="~/models/glm53-exl3-td205")
ap.add_argument("--corpus", default="~/bench/ppl/wiki.test.raw")
ap.add_argument("--out", default="scratch/rounddecomp/round_decomp.json")
ap.add_argument("--ctx", type=int, default=4096)
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--prompt", default="Say hello.")
ap.add_argument("--warm", type=int, default=8)
args = ap.parse_args()
torch.set_grad_enabled(False)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
if os.environ.get("RD_SPEED_ENV", "0") != "0":   # served SPEED_ENV (serve.py), as the lane runs
    import ast
    _src = open(os.path.join(ROOT, "tools/glm/serve.py")).read()
    for k, v in next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SPEED_ENV").items():
        os.environ.setdefault(k, v)

import exllamav3
from exllamav3.ext import exllamav3_ext
assert exllamav3.__file__.startswith(ROOT), exllamav3.__file__
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator import generator as GM
from exllamav3.generator.sampler import GreedySampler

t0 = time.perf_counter()
config = Config.from_directory(args.model)
model = Model.from_config(config); tok = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=args.ctx + 4096, max_history=1)
model.load(device="cuda:0", progressbar=False)
draft_model = Model.from_config(config, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=args.ctx + 4096, max_history=1)
draft_model.load(device="cuda:0", progressbar=False)
GM.MTP_FUSE_CATCHUP = 2
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft_model,
                draft_cache=draft_cache, num_draft_tokens=1, record_draft_stats=True)
RES = {"env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
       "args": vars(args), "load_s": time.perf_counter() - t0, "rounds": [], "errors": []}
print("loaded", round(RES["load_s"], 1), flush=True)

# ---- markers: unique-size zero-fills, the only clock shared by host phases and GPU kernels ----
MID = 0
# Marker: zero-fill of MID complex64 elements, with MID a multiple of 1024 so the traced grid
# size IS 256 * the marker id: complex64 is 8 B, 4-wide vectorized, 256 threads/block, so
# 1024*MID elements = 256*MID workgroups. Verified on the test box and locally: ids 1/2/7/96 traced as
# grid 256/512/1792/24576, plus one leading init fill of the whole buffer (also id 96) that the
# analyzer skips when it aligns trace markers to the json. The kernel is
#   vectorized_elementwise_kernel<..., at::native::FillFunctor<c10::complex<float> >, ...>
# which no EXL3 kernel uses. One allocation; ids cycle 1..NID.
NID = 96
_mb = torch.zeros(1024 * NID, dtype=torch.complex64, device="cuda:0")
CUR = [None]                     # current phase label
EV = []                          # (marker_id, label, "in"/"out") in mark() call order == GPU order
def mark(label, tag):
    """enqueue the phase marker; the returned id must equal the grid size in the trace"""
    global MID
    MID = MID % NID + 1
    _mb[:1024 * MID].zero_()
    EV.append((MID, label, tag))
    return MID

# ---- phase instrumentation -------------------------------------------------------------------
ARM = [False]                    # instrumenting (decode only)
HT = collections.defaultdict(float)   # host wall ms per phase
HN = collections.defaultdict(int)
SY = collections.defaultdict(int)     # d2h sync points per phase

def ph(label):
    """context manager: host wall + entry/exit markers, in enqueue order"""
    class _P:
        def __enter__(s):
            if not ARM[0]: return s
            s.prev = CUR[0]; CUR[0] = label
            s.m0 = mark(label, "in"); s.h = time.perf_counter()
            HN[label] += 1
            return s
        def __exit__(s, *a):
            if not ARM[0]: return
            HT[label] += 1000 * (time.perf_counter() - s.h)
            s.m1 = mark(label, "out")
            CUR[0] = s.prev
    return _P()

def _rows(a, k):
    x = a[0] if a else k.get("input_ids")
    return int(x.shape[-1])

def wrap(obj, meth, label_fn):
    f = getattr(obj, meth, None)
    if f is None: return
    def w(*a, **k):
        if not ARM[0] or torch.cuda.is_current_stream_capturing():
            return f(*a, **k)
        with ph(label_fn(a, k)):
            return f(*a, **k)
    setattr(obj, meth, w)

wrap(Generator, "iterate", lambda a, k: "round")
wrap(Generator, "iterate_draftmodel_mtp_gen", lambda a, k: "mtp_gen")
wrap(Generator, "iterate_gen", lambda a, k: "sample_accept")
wrap(draft_model, "forward", lambda a, k: f"draft_fwd_R{_rows(a,k)}")
wrap(draft_model, "prefill", lambda a, k: f"draft_prefill_R{_rows(a,k)}")
wrap(model, "forward", lambda a, k: f"verify_fwd_R{_rows(a,k)}")
wrap(model, "prefill", lambda a, k: f"catchup_prefill_R{_rows(a,k)}")
wrap(Job, "receive_logits", lambda a, k: "receive_logits")
wrap(Job, "receive_sample", lambda a, k: "receive_sample")

# ---- d2h sync counters -----------------------------------------------------------------------
_item, _cpu, _tolist, _copy = torch.Tensor.item, torch.Tensor.cpu, torch.Tensor.tolist, torch.Tensor.copy_
def _cnt(name, f, test):
    def w(*a, **k):
        if ARM[0] and test(*a, **k): SY[f"{CUR[0]}:{name}"] += 1
        return f(*a, **k)
    return w
torch.Tensor.item = _cnt("item", _item, lambda s: s.is_cuda)
torch.Tensor.cpu = _cnt("cpu", _cpu, lambda s, *a, **k: s.is_cuda)
torch.Tensor.tolist = _cnt("tolist", _tolist, lambda s: s.is_cuda)
torch.Tensor.copy_ = _cnt("d2h", _copy,
                          lambda s, src, *a, **k: (not s.is_cuda) and src.is_cuda
                          and not k.get("non_blocking", a[0] if a else False))

# ---- run --------------------------------------------------------------------------------------
WIKI = tok.decode(tok.encode(open(args.corpus, encoding="utf-8").read()[:200000], add_bos=False)[0, :args.ctx])
prompt = f"Reference notes (ignore unless relevant):\n{WIKI}\n\nTask: {args.prompt}"
ids = tok.encode(f"[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n", encode_special_tokens=True)
ntok = args.tokens + args.warm
job = Job(input_ids=ids, max_new_tokens=ntok, sampler=GreedySampler(), stop_conditions=[])
gen.enqueue(job)
torch.cuda.synchronize()
w0 = time.perf_counter(); t0m = time.perf_counter()
first = None
while gen.num_remaining_jobs():
    for r in gen.iterate():
        tid = r.get("token_ids")
        if tid is not None and tid.numel():
            if first is None:
                torch.cuda.synchronize()
                first = time.perf_counter() - w0
                w1 = time.perf_counter()
                ARM[0] = True
torch.cuda.synchronize()
ARM[0] = False
w2 = time.perf_counter()
nt = len(job.output_ids) if hasattr(job, "output_ids") else 0
st = list(job.draft_stats)
nr = len(st) if st else 0
acc = sum(s[2] for s in st) / max(len(st), 1) if st else 0.0
RES.update(prefill_s=first, decode_s=w2 - w1, tokens=nt, rounds=nr, accept=acc,
           round_ms=1000 * (w2 - w1) / max(nr, 1), tok_per_round=1 + acc,
           mono_start=t0m, mono_end=w2,
           host_ms_per_round={k: round(v / max(nr, 1), 4) for k, v in sorted(HT.items())},
           host_calls_per_round={k: round(v / max(nr, 1), 3) for k, v in sorted(HN.items())},
           syncs_per_round={k: round(v / max(nr, 1), 3) for k, v in sorted(SY.items())},
           events=EV)
json.dump(RES, open(args.out, "w"), indent=1)
print("RESULT", json.dumps({k: RES[k] for k in
      ("rounds", "accept", "round_ms", "tok_per_round", "host_ms_per_round", "host_calls_per_round",
       "syncs_per_round")}), flush=True)
print("DONE", flush=True)
