#!/usr/bin/env python
"""DS4.1 synthetic-depth profiler: decode step and one prefill chunk at depth D without a real prefill.

  ds41_depth_profile.py PACK -o out.json [--depths 8192,32768,...] [--chunk 2048] [--steps 8] [--kernels] [--no-prefill]

PACK = a depth pack (ds41_make_depth_pack.py): 4 layers at real dims that carry the attention roles of the real stack
(ratio-2 index source, the candidate source, a candidate-reading index source, a shared top-k consumer).
Every pool (compressed KV, indexer keys) is filled once for the deepest depth with synthetic values of the real shapes
(indexer keys are fp4-valued like the real ones); the step then runs at start_pos = D. Writes the JSON after every depth.
With --kernels the windows are stamped for tools/depth/kernels.py (run under rocprofv3 --kernel-trace).
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "ds41"))


def stamp():
    return {"mono": time.monotonic_ns(), "boot": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack"); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--depths", default="8192,32768,131072,524288,1000000")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--no-prefill", action="store_true"); ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import torch
    from ds41_serving import build_model
    from exllamav3.dsv41 import quant
    torch.set_grad_enabled(False)
    depths = [int(x) for x in a.depths.split(",")]
    dev = "cuda:0"
    m = build_model(a.pack, device=dev, dtype="float16", mtp=False, moe="auto", prefill_chunk=a.chunk)
    cfg = m.cfg
    top = max(depths) + a.chunk + a.steps + 64
    st = m.new_state(1, top)
    st.ensure(top)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    for l in st._kv:
        for pool, fp4 in ((st._kv[l], False), (st._ik[l], True)):
            t = pool.t
            for s in range(0, t.size(1), 1 << 16):
                blk = torch.randn(t[:, s:s + (1 << 16)].shape, generator=g, device=dev, dtype=torch.float16)
                if fp4:
                    quant.fp4_act_quant_(blk, 32)
                t[:, s:s + blk.size(1)] = blk
    st.repack_idx()
    torch.cuda.synchronize()
    ids = torch.randint(0, cfg.vocab_size - 1, (1, a.chunk), device=dev, generator=g)
    tok = ids[:, :1]
    R = {"pack": a.pack, "args": vars(a), "points": {}}

    def save():
        json.dump(R, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)

    def timed(fn, n):
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize(); return (time.time() - t) / n * 1e3

    dec = lambda d: (lambda: m.forward(tok, d, st))
    pre = lambda d: (lambda: m.prefill(ids, st, d, chunk=a.chunk))
    dec(depths[0])(); pre(depths[0])()                                    # warm (kernel compile, autotune)
    torch.cuda.synchronize()
    for d in depths:
        P = R["points"].setdefault(str(d), {})
        if not a.no_decode:
            dec(d)()
            P["decode"] = {"ms": round(timed(dec(d), a.steps), 3), "rows": 1}
            if a.kernels:
                w0 = stamp(); dec(d)(); dec(d)(); torch.cuda.synchronize(); w1 = stamp()
                P["decode"]["window"] = {"start": w0, "end": w1, "steps": 2}
        if not a.no_prefill:
            ms = [timed(pre(d), 1) for _ in range(a.reps)]
            P["prefill"] = {"ms": round(min(ms), 2), "chunk": a.chunk}
            if a.kernels:
                w0 = stamp(); pre(d)(); torch.cuda.synchronize(); w1 = stamp()
                P["prefill"]["window"] = {"start": w0, "end": w1, "steps": 1}
        print(d, {k: v["ms"] for k, v in P.items()}, flush=True)
        save()


if __name__ == "__main__":
    main()
