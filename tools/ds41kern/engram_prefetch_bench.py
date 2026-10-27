"""Engram critical-path benchmark on the real MXFP4 tables (O_DIRECT, own files dropped from the page cache with posix_fadvise only).

  python engram_prefetch_bench.py io   [--dir D] [--tok T]   host side: rows/bytes/latency per step, cache hit rate on real text
  python engram_prefetch_bench.py gpu  [--dir D] [--tok T]   GPU: critical-path stall, synchronous EngramDevice vs async prefetch, exactness

Real text = the repo's markdown files tokenized with the DS4.1 tokenizer; the n-gram rows come from the real hash of those token ids.
"""
import argparse, glob, os, statistics, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exllamav3.dsv41.engram import EngramLayout, FileEngramStore, NgramHasher, build_compressed_token_map, compute_multipliers
from exllamav3.modules.dsv41_fn.engram_io import NativeRowReader
from exllamav3.modules.dsv41_fn.engram_prefetch import EngramPrefetcher, HostEngramHash

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRATCH = os.path.expanduser("~/ds41_work/ds41eng-scratch")
LAYERS, PAD, CVOC = [1, 14], 2, 99092


def drop_pages(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def open_store(d, threads=32):
    """Engram tables of `d`: official FP8 shards (read in place) or the MXFP4 .bin files; format auto-detected."""
    from exllamav3.dsv41.engram import detect_engram_format
    readers, n, files = [], [], []
    if detect_engram_format(d) == "safetensors":
        from exllamav3.dsv41.safetensors_io import SafetensorsIndex
        ix = SafetensorsIndex(d)
        for lid in LAYERS:
            fw, _, sw, ow, _ = ix.info(f"layers.{lid}.engram.embed.weight")
            fs, _, ss, os_, _ = ix.info(f"layers.{lid}.engram.embed.scale")
            files += sorted({fw, fs})
            readers.append(NativeRowReader([(fw, ow, sw[1], sw[1]), (fs, os_, ss[1], ss[1])], threads=threads))
            n.append(sw[0])
        return FileEngramStore(readers, n, "fp8" if sw[1] == ss[1] * 32 else "mxfp4"), files
    for lid in LAYERS:
        w, s = f"{d}/layers.{lid}.engram.embed.weight.bin", f"{d}/layers.{lid}.engram.embed.scale.bin"
        files += [w, s]
        readers.append(NativeRowReader([(w, 0, 128, 128), (s, 0, 8, 8)], threads=threads))
        n.append(os.path.getsize(s) // 8)
    return FileEngramStore(readers, n, "mxfp4"), files


def real_setup(tok_path, max_tokens=400_000):
    from tokenizers import Tokenizer
    os.makedirs(SCRATCH, exist_ok=True)
    tk = Tokenizer.from_file(tok_path)
    mp = os.path.join(SCRATCH, "tmap.npy")
    if os.path.isfile(mp):
        tmap = np.load(mp).tolist()
    else:
        class Shim:
            backend_tokenizer, _n = tk, tk.get_vocab_size(True)
            def __len__(self): return self._n
        tmap, nv = build_compressed_token_map(Shim())
        np.save(mp, np.asarray(tmap, dtype=np.int64))
    ids = []
    for f in sorted(glob.glob(f"{ROOT}/**/*.md", recursive=True)):
        if os.path.getsize(f) < 200_000_000:
            ids += tk.encode(open(f, errors="ignore").read()).ids
        if len(ids) >= max_tokens:
            break
    ids = np.asarray(ids[:max_tokens], dtype=np.int64)
    lay = EngramLayout.build(LAYERS, None, 4, 16_000_000, 8, 256)
    return lay, tmap, compute_multipliers(LAYERS, 4, max(tmap) + 1), ids


def med(x):
    return statistics.median(x)


# ---------------------------------------------------------------------------------------------------- io
def cmd_io(a):
    store, files = open_store(a.dir)
    lay, tmap, mult, ids = real_setup(a.tok)
    print(f"real text: {ids.size} tokens, distinct {len(set(ids.tolist()))}")
    drop_pages(files)
    print("\n== 1. current synchronous path (host side: hash, unique, O_DIRECT gather of the unique rows, both layers) ==")
    print("R = rows (tokens) per step; rows = n-gram rows requested (R*24 per layer); phys = 4 KiB sector reads (2 per row: values + scales)")
    for R, reps in ((1, 200), (6, 100), (11, 100), (2048, 3)):
        hh = HostEngramHash(lay, tmap, PAD, mult, ids.size + 1)
        th, tu, tf, nrows, nuniq, rd0 = [], [], [], 0, 0, sum(r.reads for r in store.readers)
        pos = 0
        hh(ids[:3000], 0); pos = 3000
        for _ in range(reps):
            drop_pages(files) if R == 2048 else None
            s = pos
            t0 = time.perf_counter(); rows = hh(ids[s:s + R], s); t1 = time.perf_counter()
            fl = [rows[:, lp, :].reshape(-1) for lp in range(2)]
            un = [np.unique(f) for f in fl]; t2 = time.perf_counter()
            for lp in range(2):
                store.fetch(lp, torch.from_numpy(un[lp]))
            t3 = time.perf_counter()
            th.append(t1 - t0); tu.append(t2 - t1); tf.append(t3 - t2)
            nrows += sum(f.size for f in fl); nuniq += sum(u.size for u in un); pos += R + 100
        rd = sum(r.reads for r in store.readers) - rd0
        print(f"R={R:5d}: rows/step {nrows//reps:6d} unique {nuniq//reps:6d} | useful {nuniq//reps*136/1e3:8.1f} KB, "
              f"phys {rd//reps*4096/1e3:9.1f} KB ({rd//reps} reads) | hash {med(th)*1e6:8.0f} us, unique {med(tu)*1e6:8.0f} us, "
              f"fetch {med(tf)*1e6:9.0f} us (p90 {sorted(tf)[int(.9*len(tf))]*1e6:.0f}), total {(med(th)+med(tu)+med(tf))*1e6:9.0f} us "
              f"= {(med(th)+med(tu)+med(tf))*1e6/R:7.1f} us/token | {rd//reps*4096/med(tf)/1e9:5.2f} GB/s phys")
    print("\n== 2. hot-row cache hit rate on real text (prefill in 2048 chunks, then decode 1 token / verify blocks of 6) ==")
    npre = min(ids.size // 2, 60_000)
    for gib in (0.0, 0.125, 0.5, 1.0, 2.0):
        pf = EngramPrefetcher(lay, tmap, PAD, mult, store, max_seq=ids.size + 16, cache_bytes=int(gib * (1 << 30)), max_rows=2048, slots=3)
        t_pre = 0.0
        for s in range(0, npre, 2048):
            t0 = time.perf_counter()
            job = pf.submit(ids[s:min(s + 2048, npre)], s)
            for lp in range(2):
                pf.take(job, lp)
            pf.release(job); t_pre += time.perf_counter() - t0
        st0 = pf.stats(); h0, l0 = st0["hits"], st0["lookups"]
        r0 = st0["rows_read"]
        n_dec, lat = 0, []
        s = npre
        while s + 6 <= min(ids.size, npre + 30_000):
            R = 1 if (s // 7) % 3 else 6                  # mix of plain decode and verify blocks
            t0 = time.perf_counter()
            job = pf.submit(ids[s:s + R], s)
            for lp in range(2):
                pf.take(job, lp)
            pf.release(job)
            lat.append(time.perf_counter() - t0)
            s += R; n_dec += R
        st = pf.stats()
        print(f"cache {gib:5.3f} GiB ({st['cache_rows']:>9d} rows): prefill hit {h0/max(1,l0):6.1%} ({t_pre/((npre+2047)//2048)*1e3:7.1f} ms/chunk worker+wait) | "
              f"decode hit {(st['hits']-h0)/max(1,st['lookups']-l0):6.1%}, rows read from NVMe per decode token "
              f"{(st['rows_read']-r0)/n_dec:5.1f} of {(st['rows_unique']-st0['rows_unique'])/n_dec:5.1f} unique | step latency (submit..take both layers) "
              f"median {med(lat)*1e6:7.0f} us p90 {sorted(lat)[int(.9*len(lat))]*1e6:7.0f} us")
        pf.close()
    store.close()


# ---------------------------------------------------------------------------------------------------- gpu
def cmd_gpu(a):
    from exllamav3.modules.dsv41_fn.engram_dev import EngramDevice
    D = "cuda"
    store, files = open_store(a.dir)
    lay, tmap, mult, ids = real_setup(a.tok, 200_000)
    drop_pages(files)
    dev = EngramDevice(lay, tmap, PAD, mult, ids.size + 16, store, D)
    gen = torch.Generator().manual_seed(3)
    W = (torch.randn(6144, 25600, generator=gen) * 0.01).half().to(D)               # wkv stand-in (same for every mode)
    qk = torch.randn(4, 5120, generator=gen).to(D)
    wkv = lambda e: e @ W
    a_ = torch.randn(2048, 4096, device=D, dtype=torch.half)

    def busy(ms):                                                                    # GPU work standing in for trunk layers
        t0 = time.perf_counter(); n = 0
        while True:
            for _ in range(8):
                a_ @ a_.T
            torch.cuda.synchronize(); n += 8
            if time.perf_counter() - t0 > 0.05:
                break
        per = (time.perf_counter() - t0) / n
        return max(1, round(ms / 1e3 / per))
    one = [None]

    def spin(ms):
        if one[0] is None:
            t0 = time.perf_counter(); torch.cuda.synchronize()
            for _ in range(50):
                a_ @ a_.T
            torch.cuda.synchronize(); one[0] = (time.perf_counter() - t0) / 50
        for _ in range(max(1, round(ms / 1e3 / one[0]))):
            a_ @ a_.T

    pf = dev.attach_prefetcher(cache_bytes=int(a.cache_gib * (1 << 30)), token_map=tmap, pad_token_id=PAD, multipliers=mult,
                               max_rows=2048, slots=3, max_seq=ids.size + 16)
    print(f"real text {ids.size} tokens; busy stand-in: layer0 / 12 layers per mode below; wkv = random fp16 [6144, 25600]")

    pre = [None, None]

    def step(mode, s, R, l0_ms, l12_ms, h0):
        h = h0.clone()
        ci = torch.from_numpy(ids[s:s + R]).to(D)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if mode == "async":
            tk = dev.prefetch(ids[s:s + R], s)
        spin(l0_ms)                                                                  # layer 0
        if mode == "sync":
            rows = dev.hash(ci, s)
            dev.inject(h, rows[:, 0, :], 0, wkv, qk, 1e-6)
        elif mode == "async":
            dev.inject(h, None, 0, wkv, qk, 1e-6, ticket=tk)
        elif mode == "ideal":
            dev.inject(h, None, 0, wkv, qk, 1e-6, fetched=pre[0])
        spin(l12_ms)                                                                 # layers 1..13
        if mode == "sync":
            dev.inject(h, rows[:, 1, :], 1, wkv, qk, 1e-6)
        elif mode == "async":
            dev.inject(h, None, 1, wkv, qk, 1e-6, ticket=tk)
        elif mode == "ideal":
            dev.inject(h, None, 1, wkv, qk, 1e-6, fetched=pre[1])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if mode == "async":
            dev.prefetcher.release(tk)
        return dt, h

    print("\n== critical path: step wall time minus ideal (rows already on the GPU), median of reps, cold NVMe ==")
    for R, reps, l0, l12 in ((1, 30, 2.2, 27.0), (6, 30, 2.5, 30.0), (11, 30, 2.5, 30.0), (2048, 3, 140.0, 1700.0)):
        h0 = torch.randn(R, 4, 5120, generator=gen).to(D)
        res = {m: [] for m in ("ideal", "sync", "async")}
        exact, warned = True, False
        s = 5000
        for rep in range(reps):
            s += R + 37
            # ideal: fetch on the host first (same bytes), outside the timed region
            dev.prefetcher.hash(ids[max(0, s - 3):s], max(0, s - 3))
            rows_np = dev.prefetcher.hash(ids[s:s + R], s)
            for lp in range(2):
                u, inv = np.unique(rows_np[:, lp, :].reshape(-1), return_inverse=True)
                pre[lp] = (store.fetch(lp, torch.from_numpy(u)).to(D), torch.from_numpy(inv.astype(np.int32)).to(D))
            outs = {}
            for m in ("ideal", "sync", "async"):
                p3 = max(0, s - 3)
                if m == "sync":                                                      # n-gram history of the previous tokens
                    dev.hash(torch.from_numpy(ids[p3:s]).to(D), p3)
                if m == "async":
                    dev.prefetcher.hash(ids[p3:s], p3)
                if m != "ideal":
                    drop_pages(files)
                dt, h = step(m, s, R, l0, l12, h0)
                res[m].append(dt); outs[m] = h
            exact &= bool(torch.equal(outs["sync"], outs["async"]) and torch.equal(outs["ideal"], outs["async"]))
            if not exact and not warned:
                warned = True
                print("  mismatch: sync/async", (outs["sync"] - outs["async"]).abs().max().item(), "ideal/async",
                      (outs["ideal"] - outs["async"]).abs().max().item(), "ideal/sync", (outs["ideal"] - outs["sync"]).abs().max().item())
        base = med(res["ideal"])
        print(f"R={R:5d}: ideal {base*1e3:8.2f} ms | sync {med(res['sync'])*1e3:8.2f} ms (+{(med(res['sync'])-base)*1e6:9.0f} us, "
              f"{(med(res['sync'])-base)*1e6/R:8.1f} us/token) | async {med(res['async'])*1e3:8.2f} ms (+{(med(res['async'])-base)*1e6:9.0f} us, "
              f"{(med(res['async'])-base)*1e6/R:8.1f} us/token) | outputs bitwise equal sync == async == ideal: {exact}")
    pf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["io", "gpu"])
    ap.add_argument("--dir", default=os.path.expanduser("~/models/ds41-1017/engram_mxfp4"))
    ap.add_argument("--tok", default=os.path.expanduser("~/ds41_work/ds41qual/.cache_tok/tokenizer.json"))
    ap.add_argument("--cache-gib", type=float, default=1.0)
    a = ap.parse_args()
    {"io": cmd_io, "gpu": cmd_gpu}[a.mode](a)
