"""Per-layer KV attention microbench: quant k6v6 / kw6 (k6v6 + sink/window ring) / tq6 through attn_dispatch, like the attention module drives it.
usage: python tools/bench_kv_attn.py [--hd 256 --kvh 2 --qh 24] [--ctx 8192,131072] [--rows 1,2,3,4,5,8] [--modes k6v6,kw6,tq6]
Prints ms per call (median) for decode / verify rows at each context, plus the prefill-chunk KV write cost (rotate + quantize/write)."""
import os, sys, types, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.tq import CacheLayer_tq
from exllamav3.cache.kvwin import KVWinLayerState, KVWinCtx, KVWinCfg
from exllamav3.modules.attention_fn.dispatch import attn_dispatch

ap = argparse.ArgumentParser()
ap.add_argument("--hd", type=int, default=256); ap.add_argument("--kvh", type=int, default=2); ap.add_argument("--qh", type=int, default=24)
ap.add_argument("--bits", type=int, default=6)
ap.add_argument("--ctx", default="8192,131072"); ap.add_argument("--rows", default="1,2,3,4,5,8"); ap.add_argument("--modes", default="k6v6,kw6,tq6")
ap.add_argument("--prof", type=int, default=0, help="rows T to profile (kernel breakdown)")
ap.add_argument("--reps", type=int, default=40); ap.add_argument("--S", type=int, default=4); ap.add_argument("--W", type=int, default=128)
a = ap.parse_args()
DEV = "cuda:0"
PAGE = 256

def med(f, reps):
    for _ in range(3): f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); f(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    ts.sort(); return ts[len(ts) // 2]

def build(mode, ctx):
    attn = types.SimpleNamespace(num_kv_heads=a.kvh, head_dim=a.hd, window_arg=lambda: -1, logit_softcapping=0.0)
    pages = (ctx + 64) // PAGE + 2
    cls = CacheLayer_tq if mode.startswith("tq") else CacheLayer_quant
    cl = cls(None, attn, 0, pages * PAGE, a.bits, a.bits); cl.alloc(torch.device(DEV))
    ring = None
    if mode != "k6v6":
        st = KVWinLayerState(attn, 2, 0, 0); st.cfg = KVWinCfg(a.S, a.W, a.W + 16, a.S + a.W + 16); st.cache_layer = cl; st.alloc(torch.device(DEV))
        ring = KVWinCtx(st, torch.tensor([1], dtype=torch.int32, device=DEV))
    bt = torch.arange(pages, dtype=torch.int32, device=DEV).view(1, -1)
    return cl, ring, bt

g = torch.Generator(device=DEV).manual_seed(0)
def rnd(*s): return (torch.randn(*s, device=DEV, generator=g) * (0.5 + torch.rand(*s[:-1], 1, device=DEV, generator=g) * 2)).half()

print(f"hd {a.hd} kvh {a.kvh} qh {a.qh} bits {a.bits} ring {a.S}:{a.W}  (ms per layer call)")
for ctx in map(int, a.ctx.split(",")):
    for mode in a.modes.split(","):
        cl, ring, bt = build(mode, ctx)
        L = 0
        while L < ctx:                                  # fill the pool by staged prefill chunks
            n = min(2048, ctx - L)
            sl = torch.tensor([L], dtype=torch.int32, device=DEV)
            attn_dispatch(rnd(1, n, a.qh, a.hd), rnd(1, n, a.kvh, a.hd), rnd(1, n, a.kvh, a.hd), cache=cl, causal=True, block_table=bt,
                          cache_seqlens=sl, max_kv_len=L, ov=ring)
            L += n
        out = []
        for T in map(int, a.rows.split(",")):
            q, k, v = rnd(1, T, a.qh, a.hd), rnd(1, T, a.kvh, a.hd), rnd(1, T, a.kvh, a.hd)
            sl = torch.tensor([ctx - T], dtype=torch.int32, device=DEV)
            f = lambda: attn_dispatch(q, k, v, cache=cl, causal=True, block_table=bt, cache_seqlens=sl, max_kv_len=ctx - T, ov=ring)
            out.append(f"T{T} {med(f, a.reps):.3f}")
        if a.prof:
            from torch.profiler import profile, ProfilerActivity
            T = a.prof
            q, k, v = rnd(1, T, a.qh, a.hd), rnd(1, T, a.kvh, a.hd), rnd(1, T, a.kvh, a.hd)
            sl = torch.tensor([ctx - T], dtype=torch.int32, device=DEV)
            f = lambda: attn_dispatch(q, k, v, cache=cl, causal=True, block_table=bt, cache_seqlens=sl, max_kv_len=ctx - T, ov=ring)
            for _ in range(5): f()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as pr:
                for _ in range(20): f()
                torch.cuda.synchronize()
            tot = 0
            for e in pr.key_averages():
                dt = getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0) or getattr(e, "self_device_time_total", 0)
                if dt > 0:
                    print(f"    {mode} T{T}: {e.key[:60]:60s} {dt / 20:8.1f} us/call x{e.count // 20}"); tot += dt / 20
            print(f"    {mode} T{T}: GPU total {tot:.1f} us")
        # prefill KV write: rotate (tq) + quantize into the pool, 2048-row chunk
        k2, v2 = rnd(1, 2048, a.kvh, a.hd), rnd(1, 2048, a.kvh, a.hd)
        sl2 = torch.tensor([ctx - 2048], dtype=torch.int32, device=DEV)
        def w():
            kk, vv = k2, v2
            if getattr(cl, "tq", False):
                _, kk, vv = cl.get_tq().rotate(k2, k2, v2)
            cl.update_kv_direct(sl2, bt, kk, vv, 2048)
        out.append(f"write2048 {med(w, 20):.3f}")
        print(f"ctx {ctx:>7} {mode:>5}: " + "  ".join(out), flush=True)
        del cl, ring; torch.cuda.empty_cache()
