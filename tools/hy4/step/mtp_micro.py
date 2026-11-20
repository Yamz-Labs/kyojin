"""mtp_micro.py: model-free micro of the draft MoE experts (int8 per-row Triton kernels of moe_i8.py) on SYNTHETIC int8 stacks at the real shapes.
Reads only the GGUF header (topk, dims). Cycles >= 12 random expert selections over 512 experts (far above the 32 MB MALL), HIP-graph replay.
  mtp_micro.py [--sweep] [--out F]      run under both GPU locks (runmicro_mtp.sh), <= 180 s
Reports ms / GB/s for _k_gu, _k_down (shipped tiles) and a tile sweep, plus the split-S down and fused variants of moe_i8 when present."""
import os, sys, json, time, argparse, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))
import torch
import triton
import engine as _eng
_eng._stub_package(TREE)
from exllamav3.loader import moe_i8 as MI
from exllamav3.loader.gguf_collection import GGUFCollection

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", action="store_true")
ap.add_argument("--out", default="")
ap.add_argument("--variants", type=int, default=12)
a = ap.parse_args()
dev = torch.device("cuda")
g = GGUFCollection(os.environ["GGUF"])
m = g.meta
p = m["general.architecture"] + "."
S = int(m[p + "expert_used_count"])
il = m[p + "block_count"] - m.get(p + "nextn_predict_layers", 0)
(K, N, E) = g.infos[f"blk.{il}.ffn_gate_exps.weight"][1]
lim = float(m[p + "swiglu_clamp_exp"][il])
print(f"MTP experts: E {E} N {N} K {K} topk {S} lim {lim}", flush=True)


class St:
    pass


def stack():
    s = St()
    s.E, s.N, s.K = E, N, K
    s.q = torch.randint(-127, 128, (E, N, K), dtype=torch.int8, device=dev)
    s.s = torch.rand(E, N, device=dev) * 1e-3 + 1e-4
    return s


gate, up, down = stack(), stack(), stack()
z = (torch.randn(K, device=dev) * 0.5).float()
sels = [torch.randperm(E, device=dev)[:S].contiguous() for _ in range(a.variants)]
ws = [torch.rand(S, device=dev) + 0.1 for _ in sels]
h = torch.randn(S, N, device=dev)


def bench(fns, nrep=150, budget=4.0):
    for f in fns:
        f()
    torch.cuda.synchronize()
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s_)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for f in fns:
            f()
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts, t0 = [], time.time()
    while len(ts) < nrep and (time.time() - t0 < budget or len(ts) < 20):
        ev0.record(); gr.replay(); ev1.record(); ev1.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    return statistics.median(ts) / len(fns)


res = []


def rep(name, ms, nbytes, ref=None):
    gbs = nbytes / (ms * 1e-3) / 1e9
    r = dict(name=name, us=round(ms * 1e3, 1), gbs=round(gbs, 1), pct240=round(100 * gbs / 240, 0))
    if ref is not None:
        r["err_rel"] = float(((ref[0] - ref[1]).norm() / ref[1].norm()).item())
    res.append(r)
    print(json.dumps(r), flush=True)


b_gu = S * 2 * N * K + S * 2 * N * 4
b_dn = S * N * K + S * N * 4
out_gu = torch.empty(S, N, device=dev)
out_dn = torch.empty(N, device=dev)


def gu(sel, BN=16, BK=128, nw=4):
    MI._k_gu[(S, triton.cdiv(N, BN))](gate.q, up.q, gate.s, up.s, sel, z, out_gu, N, K, lim, BN=BN, BK=BK, num_warps=nw)


def dn(sel, w, BN=8, BK=128, nw=4):
    MI._k_down[(triton.cdiv(N, BN),)](down.q, down.s, sel, h, w, out_dn, N, K, S=S, BN=BN, BK=BK, num_warps=nw)


rep("shipped_gu BN16 BK128 w4", bench([lambda s=s: gu(s) for s in sels]), b_gu)
rep("shipped_down BN8 BK128 w4", bench([lambda s=s, w=w: dn(s, w) for s, w in zip(sels, ws)]), b_dn)
rep("shipped_total (experts_forward)", bench([lambda s=s, w=w: MI.experts_forward(gate, up, down, z, s, w, lim) for s, w in zip(sels, ws)]), b_gu + b_dn)
if a.sweep:
    for BN in (4, 8, 16, 32):
        for BK in (64, 128, 256, 512):
            for nw in (1, 2, 4, 8):
                try:
                    rep(f"gu BN{BN} BK{BK} w{nw}", bench([lambda s=s: gu(s, BN, BK, nw) for s in sels], 40, 0.6), b_gu)
                except Exception as e:      # noqa: BLE001
                    print("gu fail", BN, BK, nw, str(e)[:60], flush=True)
    for BN in (2, 4, 8, 16):
        for BK in (64, 128, 256, 512):
            for nw in (1, 2, 4, 8):
                try:
                    rep(f"down BN{BN} BK{BK} w{nw}", bench([lambda s=s, w=w: dn(s, w, BN, BK, nw) for s, w in zip(sels, ws)], 40, 0.6), b_dn)
                except Exception as e:      # noqa: BLE001
                    print("down fail", BN, BK, nw, str(e)[:60], flush=True)
# new variants live in moe_i8 (experts_forward_v2 and friends), added by the fix
if hasattr(MI, "experts_forward_v2"):
    ref = MI.experts_forward(gate, up, down, z, sels[0], ws[0], lim)
    new = MI.experts_forward_v2(gate, up, down, z, sels[0], ws[0], lim)
    rep("v2_total", bench([lambda s=s, w=w: MI.experts_forward_v2(gate, up, down, z, s, w, lim) for s, w in zip(sels, ws)]), b_gu + b_dn, (new, ref))
if a.out:
    json.dump(res, open(a.out, "w"))
print("MTP_MICRO_DONE", flush=True)
