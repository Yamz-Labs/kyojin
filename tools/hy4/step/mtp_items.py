"""mtp_items.py: itemise ONE draft step (Hy4MTP.step, shipped HY4_MTP_FAST=1, no head) by torch.profiler (record_shapes) on the real blk.<n_trunk> tensors (64 of 512 experts loaded, as opbench draft_mtp_layer).
   mtp_items.py [--ctx 1024,4096]  (run under both GPU locks: runmicro_mtp.sh)"""
import os, sys, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import ast
src = open(os.path.join(TREE, "tools", "hy4", "serve.py")).read()
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SHIPPED_ENV" for t in n.targets):
        for k, v in ast.literal_eval(n.value).items():
            os.environ.setdefault(k, v)
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))
os.environ["HY4_MTP_HEAD"] = ""      # no output head here (draft_head op)
import numpy as np, torch
import engine as _eng
_eng._stub_package(TREE)
from types import SimpleNamespace
from exllamav3.loader import hy4_forward as HF, mtp_head as MT, attn_eager as AE
from exllamav3.loader.gguf_collection import GGUFCollection
ap = argparse.ArgumentParser(); ap.add_argument("--ctx", default="1024,4096"); ap.add_argument("--reps", type=int, default=20); a = ap.parse_known_args()[0]
dev = torch.device("cuda")
g = GGUFCollection(os.environ["GGUF"]); m = g.meta; p = m["general.architecture"] + "."
il = m[p + "block_count"] - m.get(p + "nextn_predict_layers", 0); b = f"blk.{il}."
ctxs = [int(x) for x in a.ctx.split(",")]; cap = (max(ctxs) + 1024 + 511) // 512 * 512
NE = 64
ixc = dict(n_head=m[p + "attention.indexer.head_count"], head_dim=m[p + "attention.indexer.key_length"], top_k=m[p + "attention.indexer.top_k"], kpool=m[p + "attention.indexer.kpool"],
           select_tail=bool(m[p + "attention.indexer.kpool_select_tail"]), rope_dim=m[p + "rope.dimension_count"], rope_base=float(m[p + "rope.freq_base"]), eps=float(m[p + "attention.layer_norm_rms_epsilon"]))
ty, dims, off = g.infos["token_embd.weight"]; rb = dims[0] // 32 * 34
table = torch.from_numpy(np.array(g._pread(off, 8 * rb))).reshape(8, rb).to(dev)
fw = SimpleNamespace(g=g, dev=dev, n_trunk=il, eps=ixc["eps"], clamp=list(m[p + "swiglu_clamp_exp"]), tm=HF._Tm(), ixc=ixc, topk=m[p + "expert_used_count"], w_scale=float(m[p + "expert_weights_scale"]),
                     D=int(m[p + "embedding_length"]), _ns=SimpleNamespace(cap=cap, ar=torch.arange(cap, device=dev), table=table))
saved = {}
for n_ in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
    t_, d_, o_ = g.infos[b + n_]; saved[b + n_] = g.infos[b + n_]; g.infos[b + n_] = (t_, (d_[0], d_[1], NE), o_)
try:
    mt = MT.Hy4MTP(fw, cap=cap)
finally:
    g.infos.update(saved)
mt.rbias[NE:] = -1e9; mt.logits = lambda y: y[0, :1]; AE.ensure(cap)
D = fw.D
import statistics
ap2 = argparse.ArgumentParser(); ap2.add_argument("--ctx", default="1024,4096"); ap2.add_argument("--reps", type=int, default=20); ap2.add_argument("--prof", action="store_true"); ap2.add_argument("--save", default=""); ap2.add_argument("--ref", default="")
a = ap2.parse_args()
FL = {k: os.environ.get(k) for k in os.environ if k.startswith("HY4_DRAFT_")}
for ctx in ctxs:
    torch.manual_seed(0)
    mt.kbuf.copy_(torch.randn_like(mt.kbuf) * 1.5)
    j = ctx - 8; N = min(cap, ((j + 3) // 512 + 1) * 512); k = 3
    tok = torch.tensor([1], device=dev); h0 = torch.randn(1, D, device=dev); pos0 = mt.ar[j:j + 1]
    def rnd():
        y, lg = mt.step(tok, h0, pos0, N)
        t = torch.zeros(1, dtype=torch.long, device=dev); h = y[-1:]
        for i in range(1, k):
            y, lg = mt.step(t, h, mt.ar[j + i:j + i + 1], N); h = y
        return y
    y1 = mt.step(tok, h0, pos0, N)[0].clone()
    for _ in range(3): rnd()
    torch.cuda.synchronize()
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_): rnd()
    torch.cuda.current_stream().wait_stream(s_); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): rnd()
    for _ in range(5): gr.replay()
    torch.cuda.synchronize()
    ts = []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(100):
        e0.record(); gr.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
    err = ""
    if a.ref:
        r = torch.load(a.ref)[ctx]; err = f" step1 rel err vs ref {((y1 - r).norm() / r.norm()).item():.2e}"
    if a.save:
        d = torch.load(a.save) if os.path.exists(a.save) else {}
        d[ctx] = y1.cpu() if False else y1; torch.save(d, a.save)
    print(f"MTP_ROUND ctx {ctx} N {N}: {statistics.median(ts):.3f} ms per round (3 steps, no head) p10 {sorted(ts)[10]:.3f} p90 {sorted(ts)[90]:.3f} flags {FL}{err}", flush=True)
