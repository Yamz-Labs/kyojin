#!/usr/bin/env python
"""O_DIRECT read throughput of a file on this box's NVMe (CPU only, no page cache, no mmap).

  python tools/tier/ssd_bench.py --file scratch/ssd10g.bin --size-gib 10 --make
  python tools/tier/ssd_bench.py --file scratch/ssd10g.bin --size-gib 10 --out scratch/ssd_bench.json

Measures random aligned reads of S bytes (S = an expert's size) with T reader threads (os.preadv releases the
GIL, queue depth = T), plus 4 KiB random reads (Engram rows, latency and IOPS). Every arm is repeated and
the arms are interleaved in rounds; the report is mean +- SE over the rounds.
"""
import argparse, json, os, random, statistics as st, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import mmap

ALIGN = 4096


def make(path, gib):
    rnd = os.urandom(1 << 20)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    buf = mmap.mmap(-1, 16 << 20)
    for i in range(16):
        buf[i << 20:(i + 1) << 20] = rnd[::-1] if i & 1 else rnd
    n = int(gib * (1 << 30)) // (16 << 20)
    t0 = time.time()
    for i in range(n):
        # vary the content per block so a compressing controller cannot cheat
        buf[0:8] = i.to_bytes(8, "little")
        os.write(fd, buf)
    os.fsync(fd)
    os.close(fd)
    print(f"wrote {n * 16 / 1024:.1f} GiB in {time.time() - t0:.1f}s", flush=True)


class Reader:
    def __init__(self, path, threads, bufmax):
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.size = os.fstat(self.fd).st_size
        self.bufs = [mmap.mmap(-1, bufmax) for _ in range(threads)]
        self.threads = threads

    def run(self, rsize, nbytes_target, seq=False, seed=0):
        rsize = (rsize + ALIGN - 1) // ALIGN * ALIGN
        nreads = max(self.threads * 4, int(nbytes_target // rsize))
        rng = random.Random(seed)
        nblk = (self.size - rsize) // ALIGN
        offs = [rng.randrange(nblk) * ALIGN for _ in range(nreads)]
        if seq:
            offs = [(i * rsize) % (self.size - rsize) for i in range(nreads)]
        lat = []
        lock = threading.Lock()
        idx = [0]

        def work(ti):
            buf = self.bufs[ti]
            mv = memoryview(buf)[:rsize]
            loc = []
            while True:
                with lock:
                    i = idx[0]
                    idx[0] += 1
                if i >= nreads:
                    break
                t0 = time.perf_counter()
                got = 0
                while got < rsize:
                    r = os.preadv(self.fd, [mv[got:]], offs[i] + got)
                    if r <= 0:
                        break
                    got += r
                loc.append(time.perf_counter() - t0)
            with lock:
                lat.extend(loc)

        ths = [threading.Thread(target=work, args=(i,)) for i in range(self.threads)]
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        dt = time.perf_counter() - t0
        lat.sort()
        return dict(gbps=nreads * rsize / dt / 1e9, iops=nreads / dt, p50_ms=lat[len(lat) // 2] * 1e3,
                    p99_ms=lat[int(len(lat) * 0.99)] * 1e3, n=nreads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--size-gib", type=float, default=10)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-arm-gib", type=float, default=2.0)
    ap.add_argument("--sizes-mib", default="0.0039,1,4,8.85,13.3,16")
    ap.add_argument("--threads", default="1,2,4,8,16,32")
    a = ap.parse_args()
    if a.make:
        make(a.file, a.size_gib)
    sizes = [float(x) for x in a.sizes_mib.split(",")]
    thr = [int(x) for x in a.threads.split(",")]
    arms = []
    for s in sizes:
        for t in thr:
            if s < 0.01 and t < 8:
                continue
            arms.append((s, t))
    res = {arm: [] for arm in arms}
    for rd in range(a.rounds):
        for (s, t) in arms:
            r = Reader(a.file, t, 32 << 20)
            nb = a.per_arm_gib * (1 << 30) if s >= 0.01 else 64e6
            x = r.run(int(s * (1 << 20)), nb, seed=rd * 1000 + t)
            os.close(r.fd)
            res[(s, t)].append(x)
        print(f"round {rd} done", flush=True)
    out = []
    print(f"{'size MiB':>9} {'thr':>4} {'GB/s':>7} {'+-SE':>6} {'IOPS':>9} {'p50 ms':>8} {'p99 ms':>8}")
    for (s, t), xs in res.items():
        g = [x["gbps"] for x in xs]
        m = st.mean(g)
        se = st.stdev(g) / len(g) ** 0.5 if len(g) > 1 else 0
        row = dict(size_mib=s, threads=t, gbps=m, se=se, iops=st.mean(x["iops"] for x in xs),
                   p50_ms=st.mean(x["p50_ms"] for x in xs), p99_ms=st.mean(x["p99_ms"] for x in xs), rounds=len(xs))
        out.append(row)
        print(f"{s:9.3f} {t:4d} {m:7.2f} {se:6.2f} {row['iops']:9.0f} {row['p50_ms']:8.2f} {row['p99_ms']:8.2f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
