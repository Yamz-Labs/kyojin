# verifyrow1: marginal verify-row cost by op. One load, several MTP arms (ndt -> verify R=ndt+1),
# interleaved (RS_NDTS, default 1,2,3,3,2,1), each arm wrapped in an arm_<i>_R<r> marker span so
# row_split_floor.py --arm slices the one rocprofv3 trace per arm. Derived from round_decomp.py:
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
if os.environ.get("RD_SPEED_ENV", "0") != "0":   # verifyfuse1: served SPEED_ENV (serve.py), as the lane runs
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
cache = Cache(model, max_num_tokens=args.ctx + 4096, max_history=max(int(x) for x in os.environ.get("RS_NDTS", "1,2,3,3,2,1").split(",")))
model.load(device="cuda:0", progressbar=False)
draft_model = Model.from_config(config, component="mtp")
draft_cache = Cache(draft_model, max_num_tokens=args.ctx + 4096, max_history=max(int(x) for x in os.environ.get("RS_NDTS", "1,2,3,3,2,1").split(",")))
draft_model.load(device="cuda:0", progressbar=False)
GM.MTP_FUSE_CATCHUP = 2
NDTS = [int(x) for x in os.environ.get("RS_NDTS", "1,2,3,3,2,1").split(",")]
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft_model,
                draft_cache=draft_cache, num_draft_tokens=max(NDTS), record_draft_stats=True)
RES = {"env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
       "args": vars(args), "load_s": time.perf_counter() - t0, "rounds": [], "errors": []}

# ---- byte inventory (roundfloor1): exact in-memory bytes of every weight tensor, by key ------
# The byte floor needs the bytes the engine actually streams, not the HF shapes: EXL3 trellis
# packs differ per matrix. element_size()*numel() over get_tensors() is what the kernels read.
# The routed experts are not registered modules (BlockSparseMLP holds gates/ups/downs lists), so
# one expert slot is measured off gates[0]/ups[0]/downs[0] and multiplied by num_experts.
def _bytes(ts):
    return int(sum(t.element_size() * t.numel() for t in ts.values()))

def inventory(m):
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    inv = {}
    for mod in m:
        if mod.device is None: continue
        ts = mod.get_tensors()
        b = _bytes(ts) if ts else 0
        e = {"b": b}
        for a in ("in_features", "out_features", "num_experts", "num_experts_per_tok",
                  "intermediate_size", "hidden_size", "layer_idx"):
            v = getattr(mod, a, None)
            if v is not None: e[a] = int(v)
        if isinstance(mod, BlockSparseMLP):
            e["kind"] = "moe"
            # one routed expert slot: gate + up + down (the lists are per-expert, so take index 0
            # of each; `gates` is empty when gate and up share one weight set)
            e["expert_slot_b"] = int(sum(_bytes(l.get_tensors()) for l in
                                         (mod.gates[:1] + mod.ups[:1] + mod.downs[:1])))
            e["nexp_lists"] = [len(mod.gates), len(mod.ups), len(mod.downs)]
        if b or e.get("expert_slot_b"): inv[mod.key] = e
    return inv

RES["inv"] = inventory(model)
RES["inv_mtp"] = inventory(draft_model)
# routed experts held outside the module tree
def _slots(m, pfx):
    return {pfx + mod.key: int(sum(_bytes(l.get_tensors())
                                  for l in (mod.gates[:1] + mod.ups[:1] + mod.downs[:1])))
            for mod in m if type(mod).__name__ == "BlockSparseMLP"}
RES["moe_expert_slot_b"] = {**_slots(model, ""), **_slots(draft_model, "mtp:")}
print(f"inventory: {len(RES['inv'])} trunk keys, {len(RES['inv_mtp'])} mtp keys, "
      f"{len(RES['moe_expert_slot_b'])} moe layers", flush=True)

# ---- cache/geometry bytes (roundfloor1) ------------------------------------------------------
# The KDA recurrent state is fp32 (num_v_heads, k_head_dim, v_head_dim) = 4 MiB/layer/slot, which
# is what the round writes once per verified row (gdn.cu:722-736). The MLA cache stores the
# absorbed latent (kv_lora_rank, fp16) plus the indexer keys (index_head_dim, fp16, pooled by
# kpool), so a decode row touches topk latents for the absorb and every context key for the
# indexer. Both are per-row and scale with R and with context, unlike weights.
_nh, _hd = config.linear_num_heads, config.linear_head_dim
RES["kda_state_b"] = int(_nh * _hd * _hd * 4)
_n_mla = sum(1 for t in config.layer_types if t != "linear_attention")
RES["mla_latent_b"] = {
    "absorb": int(config.index_topk * config.kv_lora_rank * 2),
    "indexer": int(args.ctx * (config.index_head_dim + config.index_head_dim // config.index_kpool) * 2),
    "n_layers": _n_mla}
RES["ctx_tokens"] = int(args.ctx)
print(f"geometry: kda_state {RES['kda_state_b']/2**20:.2f} MiB/layer, mla absorb "
      f"{RES['mla_latent_b']['absorb']/2**20:.2f} MiB + indexer {RES['mla_latent_b']['indexer']/2**20:.2f} MiB"
      f" per layer per row, {_n_mla} mla layers", flush=True)

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

# ---- routed-expert picks (roundfloor1): how many DISTINCT experts a round's union touches ----
# The union table is built on device (no host sync), so the unique count is not observable from
# the trace: the moe_gu/moe_down grids are sized for the worst case (rows x topk) and mask the
# duplicates. Clone the pick tensor (device-side, no sync) and count uniques AFTER the decode
# loop, bucketed by (phase, rows) so the trunk verify and the MTP draft layer stay apart.
SELS = []
if hasattr(exllamav3_ext, "exl3_dec_moe_union"):
    _union = exllamav3_ext.exl3_dec_moe_union
    def _union_wrap(*a, **k):
        if ARM[0] and not torch.cuda.is_current_stream_capturing():
            SELS.append((CUR[0] or "?", int(a[2].shape[0]), a[2].detach().clone(), a[3].detach().clone()))
        return _union(*a, **k)
    exllamav3_ext.exl3_dec_moe_union = _union_wrap

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

# ---- run ----
TASKS = ["Summarize the notes in three sentences.", "Write a short poem about the sea.",
         "Explain what a hash table is.", "List five facts about the text.",
         "Write a Python function that reverses a list.", "Describe the weather in Paris in spring."]
CORPUS = open(args.corpus, encoding="utf-8").read()
ARMS = []
for i, ndt in enumerate(NDTS):
    off = 200000 * (i + 1)
    WIKI = tok.decode(tok.encode(CORPUS[off:off + 200000], add_bos=False)[0, :args.ctx])
    prompt = f"Reference notes (ignore unless relevant):\n{WIKI}\n\nTask: {TASKS[i % len(TASKS)]}"
    ids = tok.encode(f"[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n", encode_special_tokens=True)
    gen.num_draft_tokens = ndt
    job = Job(input_ids=ids, max_new_tokens=args.tokens + args.warm, sampler=GreedySampler(),
              stop_conditions=[])
    gen.enqueue(job)
    SELS.clear(); HT.clear(); HN.clear(); SY.clear()
    label = f"arm_{i}_R{ndt + 1}"
    torch.cuda.synchronize()
    ntok = 0; w1 = None; IDS = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            tid = r.get("token_ids")
            if tid is not None and tid.numel():
                ntok += int(tid.numel()); IDS += [int(x) for x in tid.flatten().tolist()]
                if w1 is None and ntok >= args.warm:
                    torch.cuda.synchronize()
                    mark(label, "in")
                    st0 = len(job.draft_stats)
                    w1 = time.perf_counter(); ARM[0] = True
    torch.cuda.synchronize()
    ARM[0] = False
    mark(label, "out")
    w2 = time.perf_counter()
    st = list(job.draft_stats)[st0:]
    nr = len(st)
    acc = sum(s[2] for s in st) / max(nr, 1)
    UNIQ = collections.defaultdict(list)
    for ph_, rows_, sel, _w in SELS:
        UNIQ[f"{ph_}|R{rows_}"].append(int(torch.unique(sel).numel()))
    if os.environ.get("RS_SAVE_PICKS"):   # verifyrow1: union picks + routing weights, for the top-k trim study
        torch.save([(ph_, r_, s_.cpu(), w_.cpu()) for ph_, r_, s_, w_ in SELS if ph_.startswith("verify_fwd")],
                   f"{os.environ['RS_SAVE_PICKS']}_{label}.pt")
    A_ = dict(label=label, ndt=ndt, R=ndt + 1, rounds=nr, accept=acc,
              round_ms=1000 * (w2 - w1) / max(nr, 1), tok_per_round=1 + acc,
              ids=IDS,
              uniq_experts={k: {"n": len(v), "mean": sum(v) / len(v),
                                "hist": dict(sorted(collections.Counter(v).items()))}
                            for k, v in sorted(UNIQ.items())},
              host_ms_per_round={k: round(v / max(nr, 1), 4) for k, v in sorted(HT.items())},
              syncs_per_round={k: round(v / max(nr, 1), 3) for k, v in sorted(SY.items())})
    ARMS.append(A_)
    print("ARM", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in A_.items()
                             if k in ("label", "rounds", "accept", "round_ms", "tok_per_round")}),
          "UNIQ", json.dumps({k: round(v["mean"], 2) for k, v in A_["uniq_experts"].items()}), flush=True)
RES["arms"] = ARMS
RES["events"] = EV
# row_split_floor.py defaults (first arm) so the old analyzer paths still find their keys
RES.update({k: ARMS[0][k] for k in ("uniq_experts", "rounds", "accept", "round_ms", "tok_per_round")})
json.dump(RES, open(args.out, "w"), indent=1)
print("DONE", flush=True)
