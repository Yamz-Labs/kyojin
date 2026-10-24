"""Real Engram cost on the official FP8 tables with real token streams (recipe/bench/data short.npy / long.npy).

  python engram_cost_bench.py io  [--dir D]   host side: wall time, unique rows, NVMe reads/bytes, row-cache hit rate, cold vs warm
  python engram_cost_bench.py gpu [--dir D]   GPU: critical-path stall of the Engram path inside a decode step / prefill chunk with a
                                              concurrent synthetic GPU load; sync vs async prefetch vs ideal; bitwise exactness
The tables are read with O_DIRECT (the kernel page cache is bypassed): "cache" below is the engram row cache of EngramPrefetcher.
"""
import argparse, os, statistics, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from engram_prefetch_bench import open_store, LAYERS, PAD, med
from exllamav3.dsv41.engram import EngramLayout, compute_multipliers
from exllamav3.modules.dsv41_fn.engram_prefetch import EngramPrefetcher

DATA = os.path.expanduser("~/ds41_work/recipe/bench/data")
TMAP = os.path.expanduser("~/ds41_work/ds41eng-scratch/tmap.npy")
SHORT_ROWS = {0: "chat", 2: "en", 4: "code", 15: "zh", 16: "tool"}


def setup():
    tmap = np.load(TMAP).tolist()
    lay = EngramLayout.build(LAYERS, None, 4, 16_000_000, 8, 256)
    mult = compute_multipliers(LAYERS, 4, max(tmap) + 1)
    short, long_ = np.load(f"{DATA}/short.npy").astype(np.int64), np.load(f"{DATA}/long.npy").astype(np.int64)
    assert short.max() < len(tmap) and long_.max() < len(tmap)
    return lay, tmap, mult, short, long_


DBG = {}
SPEC = ("spec_top1", "spec_top4", "spec_miss")


def pct(x, p):
    return sorted(x)[min(len(x) - 1, int(p * len(x)))]


def phys(store):
    return sum(r.reads for r in store.readers) * 4096


def cmd_io(a):
    store, files = open_store(a.dir)
    lay, tmap, mult, short, long_ = setup()
    cache_b = int(a.cache_gib * (1 << 30))
    pf = EngramPrefetcher(lay, tmap, PAD, mult, store, max_seq=20000, cache_bytes=cache_b, device=None, max_rows=4096, slots=3)
    print(f"tables {a.dir}; row cache {a.cache_gib} GiB ({pf.cache.cap} rows); reads are O_DIRECT (no kernel page cache)")

    def run(ids, start, R):
        t0 = time.perf_counter()
        job = pf.submit(ids[start:start + R], start)
        for lp in range(2):
            pf.take(job, lp)
        pf.release(job)
        return time.perf_counter() - t0

    def snap():
        s = pf.stats()
        return s["lookups"], s["hits"], s["rows_unique"], s["rows_read"], phys(store)

    def delta(b, a_, n):
        lk, h, u, rr, ph = (a_[i] - b[i] for i in range(5))
        return (f"hit {h / max(1, lk):6.1%} | unique rows/step {u / n:8.1f} | rows from NVMe/step {rr / n:8.1f} | "
                f"NVMe bytes/step {ph / n / 1e3:9.1f} KB ({ph / n / 4096:7.1f} sectors)")

    print("\n== prefill chunks on long.npy[0] (16384 tokens), cold cache then warm (same stream again) ==")
    for C in (2048, 4096):
        pf.cache.__init__(cache_b, pf.rb); pf.reset(); pf.stat.clear()
        for tag in ("cold", "warm"):
            pf.reset()
            ts, b = [], snap()
            for s in range(0, long_.shape[1], C):
                ts.append(run(long_[0], s, C))
            n = len(ts)
            print(f"chunk {C:5d} {tag}: {med(ts)*1e3:9.1f} ms/chunk (min {min(ts)*1e3:.1f}, max {max(ts)*1e3:.1f}) = {med(ts)*1e6/C:7.1f} us/token | {delta(b, snap(), n)}")

    print("\n== decode 1 token/step: prefill 1024 tokens of a short row, then 1000 steps, cold then warm ==")
    for r, name in SHORT_ROWS.items():
        ids = short[r]
        pf.cache.__init__(cache_b, pf.rb); pf.stat.clear()
        for tag in ("cold", "warm"):
            pf.reset()
            for s in range(0, 1024, 1024):
                run(ids, s, 1024)
            b, ts = snap(), []
            for s in range(1024, 2048):
                ts.append(run(ids, s, 1))
            n = len(ts)
            print(f"short[{r:2d}] {name:5s} {tag}: step {med(ts)*1e3:7.2f} ms median, p90 {pct(ts, .9)*1e3:7.2f}, mean {np.mean(ts)*1e3:7.2f} | {delta(b, snap(), n)}")
    pf.close(); store.close()


def cmd_gpu(a):
    from exllamav3.modules.dsv41_fn.engram_dev import EngramDevice
    D = "cuda"
    store, files = open_store(a.dir)
    lay, tmap, mult, short, long_ = setup()
    dev = EngramDevice(lay, tmap, PAD, mult, 20000, store, D)
    gen = torch.Generator().manual_seed(3)
    W = (torch.randn(6144, 25600, generator=gen) * 0.01).half().to(D)
    qk = torch.randn(4, 5120, generator=gen).to(D)
    wkv = lambda e: e @ W
    a_ = torch.randn(2048, 4096, device=D, dtype=torch.half)
    one = [None]

    def spin(ms):                                                 # synthetic GPU load standing in for trunk layers
        if one[0] is None:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(50):
                a_ @ a_.T
            torch.cuda.synchronize(); one[0] = (time.perf_counter() - t0) / 50
        for _ in range(max(1, round(ms / 1e3 / one[0]))):
            a_ @ a_.T

    pf = dev.attach_prefetcher(cache_bytes=int(a.cache_gib * (1 << 30)), token_map=tmap, pad_token_id=PAD, multipliers=mult,
                               max_rows=4096, slots=3, max_seq=20000)
    pre = [None, None]

    def step(mode, ids, s, R, l0, l12, h0):
        h = h0.clone()
        ci = torch.from_numpy(ids[s:s + R]).to(D)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        tp = time.perf_counter()
        if mode.startswith("spec"):
            tk = dev.prefetch(ids[s:s + R], s)
        tq = time.perf_counter() - tp
        if mode == "async":
            tk = dev.prefetch(ids[s:s + R], s)
        spin(l0)
        if mode == "sync":
            rows = dev.hash(ci, s); dev.inject(h, rows[:, 0, :], 0, wkv, qk, 1e-6)
        elif mode == "async" or mode.startswith("spec"):
            dev.inject(h, None, 0, wkv, qk, 1e-6, ticket=tk)
        else:
            dev.inject(h, None, 0, wkv, qk, 1e-6, fetched=pre[0])
        spin(l12)
        if mode == "sync":
            dev.inject(h, rows[:, 1, :], 1, wkv, qk, 1e-6)
        elif mode == "async" or mode.startswith("spec"):
            dev.inject(h, None, 1, wkv, qk, 1e-6, ticket=tk)
        else:
            dev.inject(h, None, 1, wkv, qk, 1e-6, fetched=pre[1])
        t1 = time.perf_counter(); torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if os.environ.get("ENGDBG"):
            DBG.setdefault(mode, []).append((tq * 1e3, (t1 - t0) * 1e3, dt * 1e3))
        if mode == "async" or mode.startswith("spec"):
            dev.prefetcher.release(tk)
        return dt, h

    def trial(label, ids, R, starts, l0, l12, warm):
        h0 = torch.randn(R, 4, 5120, generator=gen).to(D)
        res = {m: [] for m in ("ideal", "sync", "async") + (SPEC if R == 1 else ())}
        exact = True
        for s in starts:
            p3 = max(0, s - 3)
            dev.prefetcher.hash(ids[p3:s], p3)
            rows_np = dev.prefetcher.hash(ids[s:s + R], s)
            for lp in range(2):
                u, inv = np.unique(rows_np[:, lp, :].reshape(-1), return_inverse=True)
                pre[lp] = (store.fetch(lp, torch.from_numpy(u)).to(D), torch.from_numpy(inv.astype(np.int32)).to(D))
            outs = {}
            for m in ("ideal", "sync", "async") + (SPEC if R == 1 else ()):
                if m.startswith("spec"):
                    dev.prefetcher.hash(ids[p3:s], p3)
                    if not warm:
                        dev.prefetcher.cache.__init__(int(a.cache_gib * (1 << 30)), dev.prefetcher.rb)
                    true = int(ids[s]); oth = [int(x) for x in (true + 1 + np.arange(3) * 7919) % 129000]
                    dev.speculate({"spec_top1": [true] + oth[:3], "spec_top4": oth[:2] + [true] + oth[2:], "spec_miss": oth}[m], s)
                    time.sleep(a.spec_lead_ms / 1e3)
                if m == "sync":
                    dev.hash(torch.from_numpy(ids[p3:s]).to(D), p3)
                if m == "async":
                    dev.prefetcher.hash(ids[p3:s], p3)
                    if not warm:
                        dev.prefetcher.cache.__init__(int(a.cache_gib * (1 << 30)), dev.prefetcher.rb)
                dt, h = step(m, ids, s, R, l0, l12, h0)
                res[m].append(dt); outs[m] = h
            exact &= all(bool(torch.equal(outs["ideal"], o)) for o in outs.values())
        b = med(res["ideal"])
        for m, v in DBG.items():
            print("   dbg", m, "submit/host-enqueue/total ms medians", [round(med([x[i] for x in v]), 2) for i in range(3)])
        DBG.clear()
        sy, asy = med(res["sync"]) - b, med(res["async"]) - b
        for m in SPEC if R == 1 else ():
            sp = med(res[m]) - b
            print(f"   {m}: +{sp*1e6:9.0f} us (p90 +{(pct(res[m], .9)-b)*1e6:.0f})  [speculation issued {a.spec_lead_ms} ms before the step]")
        print(f"{label}: ideal {b*1e3:8.2f} ms | sync +{sy*1e6:9.0f} us | async +{asy*1e6:9.0f} us (p90 +{(pct(res['async'], .9)-b)*1e6:.0f}, "
              f"{asy*1e6/R:7.1f} us/token) | bitwise sync==async==ideal {exact}")

    print("== decode, 1 token/step (short[0]), synthetic trunk load: layer0 l0 ms, layers 1..13 l12 ms; async cache cold per step vs warm ==")
    ids = short[0]
    for l0, l12 in [tuple(float(v) for v in x.split(":")) for x in a.decode_loads.split(",")]:
        trial(f"R=1 l0={l0} l12={l12} cold-cache", ids, 1, range(1100, 1100 + 30 * 7, 7), l0, l12, False)
    # warm: fill the row cache by running the stream's tokens once
    for s in range(0, 2048, 512):
        dev.prefetcher.fetch_sync(ids[s:s + 512], s)
    for l0, l12 in [tuple(float(v) for v in x.split(":")) for x in a.decode_loads.split(",")]:
        trial(f"R=1 l0={l0} l12={l12} warm-cache", ids, 1, range(1100, 1100 + 30 * 7, 7), l0, l12, True)
    if a.skip_prefill:
        pf.close(); store.close(); return
    print("== prefill chunks (long[0]) with the next-chunk ticket not overlapped (worst case: same-step submit), trunk load scaled ==")
    for C, l0, l12, reps in ((2048, 140.0, 1700.0, 2), (4096, 280.0, 3400.0, 2)):
        trial(f"R={C} l0={l0} l12={l12} cold-cache", long_[0], C, [C * (i + 1) for i in range(reps)], l0, l12, False)
    pf.close(); store.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["io", "gpu"])
    ap.add_argument("--dir", default=os.path.expanduser("~/models/ds41-1017/engram_fp8"))
    ap.add_argument("--cache-gib", type=float, default=1.0)
    ap.add_argument("--decode-loads", default="2.2:27,1:12", help="gpu mode: comma list of l0:l12 trunk loads in ms for the decode rows")
    ap.add_argument("--spec-lead-ms", type=float, default=0.5, help="head start of the speculative reads over the step (logits ready -> next step)")
    ap.add_argument("--skip-prefill", action="store_true")
    a = ap.parse_args()
    {"io": cmd_io, "gpu": cmd_gpu}[a.mode](a)
