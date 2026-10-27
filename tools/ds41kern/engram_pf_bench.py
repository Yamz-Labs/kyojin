"""Cold Engram prefill reads on the official FP8 tables (O_DIRECT, no page cache).

  python engram_pf_bench.py gather [--dir D]   rows/s of the native gather vs queue depth (`threads`), one layer and both layers at once
  python engram_pf_bench.py cold   [--dir D]   cold chunk cost (2048 / 4096) and exposed read time on top of simulated compute
                                               (--layer-ms per layer x 40 layers, scaled by chunk/2048), baseline vs whole-prompt lookahead
"""
import argparse, os, sys, threading, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from engram_prefetch_bench import open_store, LAYERS, PAD
from engram_cost_bench import setup
from exllamav3.modules.dsv41_fn.engram_prefetch import EngramPrefetcher


def cmd_gather(a):
    rng = np.random.default_rng(1)
    for th in [int(x) for x in a.threads.split(",")]:
        store, _ = open_store(a.dir, threads=th)
        rows = [np.unique(rng.integers(0, store.n[lp], a.rows)) for lp in range(2)]

        def one(lp, out):
            t0 = time.perf_counter()
            store.readers[lp].read_rows(rows[lp].tolist())
            out[lp] = time.perf_counter() - t0
        res = {}
        for mode in ("1 layer", "2 layers concurrent"):
            best = []
            for _ in range(a.reps):
                out = {}
                t0 = time.perf_counter()
                ts = [threading.Thread(target=one, args=(lp, out)) for lp in range(1 if mode == "1 layer" else 2)]
                [t.start() for t in ts]; [t.join() for t in ts]
                best.append((time.perf_counter() - t0, len(ts)))
            dt, nl = min(best)
            n = sum(rows[lp].size for lp in range(nl))
            res[mode] = (n / dt, 2 * n / dt)
            print(f"threads {th:4d} {mode:20s}: {n / dt:9.0f} rows/s ({2 * n / dt:9.0f} 4K reads/s = {2 * n / dt * 4096 / 1e6:6.0f} MB/s) {dt * 1e3 / n * 1e3:6.1f} us/row", flush=True)
        store.close()


def run_cold(a, store, C, lookahead, depth, ids):
    lay, tmap, mult = a._setup
    nl = 2
    pf = EngramPrefetcher(lay, tmap, PAD, mult, store, max_seq=ids.size + C, cache_bytes=int(a.cache_gib * (1 << 30)), device=None,
                          max_rows=C, slots=3)
    pf.cache.__init__(int(a.cache_gib * (1 << 30)), pf.rb)
    scale = C / 2048
    lm = a.layer_ms / 1e3 * scale
    n = ids.size // C
    t0 = time.perf_counter()
    if lookahead:
        pf.lookahead(ids[:n * C], 0, C, depth=depth)
    pending = [pf.submit(ids[i * C:(i + 1) * C], i * C) for i in range(min(2, n))]       # 1 chunk of ticket lookahead (as today)
    exposed, per_chunk, first = [], [], 0.0
    for k in range(n):
        tc = time.perf_counter()
        job = pending.pop(0)
        if k + 2 < n:
            pending.append(pf.submit(ids[(k + 2) * C:(k + 3) * C], (k + 2) * C))
        ex = 0.0
        for lp, layers_before in enumerate((1, 14)):
            time.sleep(lm * (layers_before if lp == 0 else 13))
            t1 = time.perf_counter()
            pf.take(job, lp)
            ex += time.perf_counter() - t1
        time.sleep(lm * 26)
        pf.release(job)
        exposed.append(ex)
        per_chunk.append(time.perf_counter() - tc)
    total = time.perf_counter() - t0
    st = pf.stats()
    pf.close()
    return exposed, per_chunk, total, st, lm * 40 * n


def cmd_cold(a):
    a._setup = setup()[:3]
    long_ = setup()[4][0]
    for C in [int(x) for x in a.chunks.split(",")]:
        for lookahead, depth in ((False, 0),) + tuple((True, d) for d in [int(x) for x in a.depths.split(",")]):
            store, _ = open_store(a.dir, threads=a.io_threads)
            ex, pc, total, st, comp = run_cold(a, store, C, lookahead, depth, long_)
            name = f"lookahead depth {depth}" if lookahead else "baseline (1 chunk ahead)"
            print(f"chunk {C} {name:26s}: exposed/chunk {[round(x, 3) for x in ex]} s | total exposed {sum(ex):6.2f} s "
                  f"(without chunk 0: {sum(ex[1:]):5.2f}) | wall {total:6.1f} s vs compute {comp:6.1f} s | rows read {st['rows_read']}+{st['la_rows_read']} "
                  f"| read s/chunk {(st['la_s'] if lookahead else st['worker_s']) / (len(ex)):.2f}",
                  flush=True)
            store.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gather", "cold"])
    ap.add_argument("--dir", default=os.path.expanduser("~/models/ds41-1017/engram_fp8"))
    ap.add_argument("--threads", default="8,16,32,64,128,256")
    ap.add_argument("--rows", type=int, default=32768)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--chunks", default="2048,4096")
    ap.add_argument("--depths", default="2,4")
    ap.add_argument("--io-threads", type=int, default=256)
    ap.add_argument("--layer-ms", type=float, default=100.0)
    ap.add_argument("--cache-gib", type=float, default=2.0)
    a = ap.parse_args()
    {"gather": cmd_gather, "cold": cmd_cold}[a.mode](a)
