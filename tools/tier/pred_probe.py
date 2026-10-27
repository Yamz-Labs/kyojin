"""Prefetch-predictor probe: real DS4.1 forward on CPU; at every layer l >= 3 record the true top-6 and the top-16 ids
that layer l's router gives on EARLIER residual states (what a prefetcher could know d layers ahead).
Variants (target layer l, weights of layer l):
  A<d>: state entering layer l-d (d=0..4), attention delta skipped:  x = rmsnorm(hc_pre(hc_post(0, h, post, comb), pre), ffn_norm)
        with (pre, post, comb) = hc_mixes(h, hc_attn_*_l)
  B<d>: state after attention of layer l-d (d=1..3), same recipe
  X<d>: MoE input (normed) of layer l-d fed straight to gate_l (d=1..3)
Exact forward is untouched (hooks only read)."""
import argparse, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router_probe as RP
import fetch as F
from router_probe import LOG

ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True); ap.add_argument("--config", required=True); ap.add_argument("--calib", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--engram-subset"); ap.add_argument("--tokenizer")
ap.add_argument("--row-ids", required=True); ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--layers", default="0-39"); ap.add_argument("--maxd", type=int, default=4)
a = ap.parse_args()

from exllamav3.dsv41.config import DSV41Config
from exllamav3.dsv41.stream_calib import StreamCalibrator, SourceLayerWeights, EngramSupport
from exllamav3.dsv41 import calib as CAL
from exllamav3.dsv41 import ops
import torch.nn.functional as Fn

cfg = DSV41Config.from_file(a.config)
tmap = json.load(open(a.map))
os.makedirs(a.out, exist_ok=True)
dev = "cpu"
fetcher = F.TensorFetcher(tmap, cache=None)
src = RP.make_source(fetcher, cfg.num_hidden_layers, dev)
lo, _, hi = a.layers.partition("-"); lo, hi = int(lo), int(hi or lo)
ids = [int(x) for x in a.row_ids.split(",")]
arr = np.load(a.calib).astype(np.int64)[ids][:, :a.max_len]
rows = [torch.from_numpy(r).view(1, -1) for r in arr]
from engram_subset import token_map_from_file
tm, _ = token_map_from_file(a.tokenizer)
engram = EngramSupport(cfg, tm, RP.SubsetEngramStore(a.engram_subset, len(cfg.engram_layer_ids)))
embed = src.tensor("embed", torch.float32, "cpu")
cal = StreamCalibrator(cfg, dev, embed, engram, state_device="cpu")
del embed
cal.start(rows)

cur = {}
orig_route = CAL.moe_route
def route(cfg_, x, gw, gb, topk):
    wts, idx = orig_route(cfg_, x, gw, gb, topk)
    cur["idx"] = idx.detach().cpu().to(torch.int16).numpy()
    cur["x"] = x.detach()
    return wts, idx
CAL.moe_route = route

# B snapshot hook (reads only)
orig_pre = cal.g.block_pre
def block_pre(l, h, ho, taps=None):
    x, ctx = orig_pre(l, h, ho, taps)
    cur.setdefault("B", []).append(ctx[0][0].detach())
    return x, ctx
cal.g.block_pre = block_pre
orig_bundle = cal._bundle
def bundle(*aa, **kk):
    w = orig_bundle(*aa, **kk); cur["w"] = w; return w
cal._bundle = bundle

hc, it, he, ne = cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.rms_norm_eps
Asn, Bsn, Xsn = {}, {}, {}
lens = [r.shape[1] for r in rows]

def gate_top16(x, w):
    sc = Fn.linear(x.float(), w.gate_w.float()) / cfg.gate_temp
    if cfg.scoring_func == "softmax": sc = sc.softmax(-1)
    elif cfg.scoring_func == "sigmoid": sc = sc.sigmoid()
    else: sc = Fn.softplus(sc).sqrt()
    return (sc + w.gate_b.float()).topk(16, dim=-1)[1].to(torch.int16).numpy()

def from_state(states, w):
    outs = []
    for h in states:
        h = h.float()
        pre, post, comb = ops.hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, hc, it, he, ne)
        hm = ops.hc_post(torch.zeros_like(h[:, :, 0]), h, post, comb)
        outs.append(ops.rmsnorm(ops.hc_pre(hm, pre), w.ffn_norm, ne).reshape(-1, cfg.hidden_size))
    return gate_top16(torch.cat(outs, 0), w)

t00 = time.time()
for l in range(lo, hi + 1):
    t0 = time.time()
    cur["B"] = []
    Asn[l] = list(cal.h)
    W = SourceLayerWeights(src, l, dev)
    cal.run_layer(l, W, None, commit=True)
    w = cur["w"]
    Bsn[l] = cur["B"]; Xsn[l] = cur["x"]
    out = {"idx": cur["idx"], "lens": np.array(lens), "row_ids": np.array(ids)}
    if l >= 3:
        with torch.no_grad():
            for d in range(0, a.maxd + 1):
                if l - d in Asn: out[f"A{d}"] = from_state(Asn[l - d], w)
            if True: out["X0"] = gate_top16(Xsn[l], w)
            for d in range(1, min(a.maxd, 3) + 1):
                if l - d in Bsn: out[f"B{d}"] = from_state(Bsn[l - d], w)
                if l - d in Xsn: out[f"X{d}"] = gate_top16(Xsn[l - d], w)
    np.savez_compressed(os.path.join(a.out, f"layer{l}.npz"), **out)
    for dct in (Asn, Bsn, Xsn):
        for k in [k for k in dct if k < l - a.maxd]: del dct[k]
    cur.pop("w", None); del W, w
    LOG(f"layer {l} done {time.time() - t0:.0f}s total {time.time() - t00:.0f}s")
