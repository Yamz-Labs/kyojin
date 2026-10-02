"""Grouped WMMA MoE prefill vs the per-expert path, on one real MiMo layer.

Loads only the layer's expert tensors (~2 GB), routes random fp16 activations with the layer's
real router (sigmoid + correction bias, top-8, normalized), then compares:
  ref : the BlockSparseMLP per-expert loop (LinearEXL3.forward per expert and projection,
        silu_mul, weighted index_add_ into fp32) = what prefill runs today
  new : ext.exl3_moe_prefill_wmma
Prints rel-L2 / max-abs and event-timed medians.

  gpu-run.sh --bench tools/prefill/env-run.sh python tools/prefill/test_moe_layer.py --layer 1 --rows 2048
"""
import argparse, json, os, statistics, sys
import torch
from safetensors import safe_open

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, W)
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.ext import exllamav3_ext as ext
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--model", default = "~/models/mimo26-exl3")
p.add_argument("--layer", type = int, default = 1)
p.add_argument("--rows", type = int, nargs = "+", default = [2048])
p.add_argument("--iters", type = int, default = 5)
p.add_argument("--no-ref-time", action = "store_true")
p.add_argument("--seed", type = int, default = 0)
p.add_argument("--act-limit", type = float, default = None, help = "SwiGLU clamp (default: config swiglu_limit, else 0)")
args = p.parse_args()
torch.set_grad_enabled(False)
DEV = "cuda:0"
cfg = json.load(open(os.path.join(args.model, "config.json")))
cfg = cfg.get("text_config", cfg)
H, I, E, TOPK = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"], cfg["num_experts_per_tok"]
ACT_LIMIT = args.act_limit if args.act_limit is not None else float(cfg.get("swiglu_limit") or 0.0)

idx = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]
pre = next(p_ for p_ in (f"model.layers.{args.layer}.mlp.", f"model.language_model.layers.{args.layer}.mlp.")
           if f"{p_}gate.weight" in idx)
by_file = {}
for k, f in idx.items():
    if k.startswith(pre):
        by_file.setdefault(f, []).append(k)
T = {}
for f, ks in by_file.items():
    with safe_open(os.path.join(args.model, f), framework = "pt", device = DEV) as h:
        for k in ks:
            T[k[len(pre):]] = h.get_tensor(k)

def lin(e, proj, out_dtype):
    k = f"experts.{e}.{proj}."
    tr = T[k + "trellis"]
    in_f, out_f = tr.shape[0] * 16, tr.shape[1] * 16
    return LinearEXL3(None, in_f, out_f, suh = T[k + "suh"], svh = T[k + "svh"], trellis = tr,
                      mul1 = T.get(k + "mul1"), mcg = T.get(k + "mcg"), out_dtype = out_dtype, key = k)

gates = [lin(e, "gate_proj", torch.half) for e in range(E)]
ups = [lin(e, "up_proj", torch.half) for e in range(E)]
downs = [lin(e, "down_proj", torch.float) for e in range(E)]
Kg, Kd = gates[0].K, downs[0].K
assert all(l.K == Kg for l in gates + ups) and all(l.K == Kd for l in downs)
MCG = bool(gates[0].mcg)
assert all(bool(l.mul1) != bool(l.mcg) and bool(l.mcg) == MCG for l in gates + ups + downs)
print(f"layer {args.layer}: H {H} I {I} E {E} topk {TOPK}  K gate/up {Kg}, K down {Kd}  mcg {MCG}  act_limit {ACT_LIMIT}  MPW_LDM={os.environ.get('MPW_LDM', 'unset')}", flush = True)

def table(ls, attr):
    return torch.tensor([getattr(l, attr).data_ptr() for l in ls], dtype = torch.long, device = DEV)
tabs = {n: {a: table(ls, a) for a in ("trellis", "suh", "svh")} for n, ls in
        (("g", gates), ("u", ups), ("d", downs))}
router_w = T["gate.weight"].float()
router_b = T["gate.e_score_correction_bias"].float()

def route(x):
    s = torch.sigmoid(x.float() @ router_w.t())
    _, sel = torch.topk(s + router_b, TOPK, dim = -1)
    w = s.gather(1, sel)
    w = w / w.sum(-1, keepdim = True)
    # fp32 routing weights, matching the grouped path's routing_weights.float() (step-2 (a))
    return sel.contiguous(), w.contiguous()

def ref(x, sel, w):
    rows = x.shape[0]
    out = torch.zeros((rows, H), dtype = torch.float, device = DEV)
    flat_e = sel.reshape(-1)
    flat_w = w.reshape(-1)
    flat_t = torch.arange(rows, device = DEV).repeat_interleave(TOPK)
    order = flat_e.argsort(stable = True)
    ts, ws = flat_t[order], flat_w[order]
    counts = torch.bincount(flat_e, minlength = E).tolist()
    start = 0
    for e in range(E):
        c = counts[e]
        if c == 0: continue
        tx = ts[start:start + c]
        xc = x.index_select(0, tx)
        g = gates[e].forward(xc, {})
        u = ups[e].forward(xc, {})
        ext.silu_mul(g, u, u, ACT_LIMIT)
        d = downs[e].forward(u, {})
        d.mul_(ws[start:start + c].unsqueeze(1))
        out.index_add_(0, tx, d)
        start += c
    return out

class WS: pass
def new(x, sel, w, ws):
    rows = x.shape[0]
    A = rows * TOPK
    flat_e = sel.reshape(-1)
    order = flat_e.argsort(stable = True)
    count = torch.bincount(flat_e, minlength = E + 1)
    out = torch.empty((rows, H), dtype = torch.float, device = DEV)
    ext.exl3_moe_prefill_wmma(
        x, out, sel, w, order, count,
        tabs["g"]["trellis"], tabs["g"]["suh"], tabs["g"]["svh"],
        tabs["u"]["trellis"], tabs["u"]["suh"], tabs["u"]["svh"],
        tabs["d"]["trellis"], tabs["d"]["suh"], tabs["d"]["svh"],
        float(Kg), float(Kd),
        ws.gu_had[:2 * A], ws.gu_out[:2 * A], ws.down_out[:A],
        ws.offsets, ws.inverse[:A], ws.tiles, ws.tile_count, ACT_LIMIT, MCG)
    return out

def timed(fn, iters):
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
        e0.record(); fn(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)

g = torch.Generator(device = DEV); g.manual_seed(args.seed)
for rows in args.rows:
    A = rows * TOPK
    ws = WS()
    ws.gu_had = torch.empty((2 * A, H), dtype = torch.half, device = DEV)
    ws.gu_out = torch.empty((2 * A, I), dtype = torch.half, device = DEV)
    ws.down_out = torch.empty((A, H), dtype = torch.float, device = DEV)
    ws.offsets = torch.empty((E + 1,), dtype = torch.long, device = DEV)
    ws.inverse = torch.empty((A,), dtype = torch.long, device = DEV)
    ws.tiles = torch.empty((A // 64 + E + 1,), dtype = torch.int, device = DEV)
    ws.tile_count = torch.empty((1,), dtype = torch.int, device = DEV)

    x = torch.randn((rows, H), generator = g, device = DEV).half()
    sel, w = route(x)
    counts = torch.bincount(sel.reshape(-1), minlength = E)
    y_ref = ref(x, sel, w)
    y_new = new(x, sel, w, ws)
    torch.cuda.synchronize()
    d = (y_new - y_ref).float()
    rel = (d.norm() / y_ref.norm()).item()
    mx = d.abs().max().item()
    ok = torch.isfinite(y_new).all().item()
    # determinism
    y_new2 = new(x, sel, w, ws)
    det = torch.equal(y_new, y_new2)
    t_new = timed(lambda: new(x, sel, w, ws), args.iters)
    t_ref = float("nan") if args.no_ref_time else timed(lambda: ref(x, sel, w), max(2, args.iters // 2))
    flops = A * 3 * 2 * H * I
    print(f"rows {rows:6d}  counts max {counts.max().item():5d} min {counts.min().item():4d} "
          f"active {(counts > 0).sum().item():3d}  rel-L2 {rel:.3e}  max-abs {mx:.3e}  finite {ok}  "
          f"deterministic {det}  | new {t_new:8.2f} ms ({flops / t_new / 1e9:5.1f} TFLOP/s)  "
          f"ref {t_ref:8.2f} ms  speedup {t_ref / t_new:5.2f}x", flush = True)

if os.environ.get("MPW_PROF"):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        for _ in range(3): new(x, sel, w, ws)
        torch.cuda.synchronize()
    rows_ = [(e.key, e.self_device_time_total / 3, e.count // 3) for e in prof.key_averages() if e.self_device_time_total > 0]
    rows_.sort(key = lambda r: -r[1])
    for k, t, c in rows_[:12]:
        print(f"{t/1e3:9.3f} ms {c:4d}  {k[:120]}")
