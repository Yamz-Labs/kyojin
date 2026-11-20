"""CPU-only: cold random Engram row reads, strategies (readahead off, O_DIRECT, RWF_NOWAIT hybrid). Tree root, engine venv, no GPU."""
import os, sys, time, types, statistics, mmap
import numpy as np
tree = os.getcwd()
for n, sub in (("exllamav3", ""), ("exllamav3.loader", "loader")):
    m = types.ModuleType(n); m.__path__ = [os.path.join(tree, "exllamav3", sub)]; sys.modules[n] = m
from concurrent.futures import ThreadPoolExecutor
from exllamav3.loader.gguf_collection import GGUFCollection
from exllamav3.loader.engram_rows import F8Rows
g = GGUFCollection("/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf")
tab = F8Rows(g)
fi = tab.off >> 48
o0 = g.bases[fi] + (tab.off & ((1 << 48) - 1)); rb = tab.row_bytes; rows = tab.rows
path = os.readlink(f"/proc/self/fd/{g.fds[fi]}")
print("path", path, flush=True)
N = int(os.environ.get("N", "256")); R = int(os.environ.get("ROUNDS", "150"))
rng = np.random.default_rng(11)
out = np.zeros((N, rb), dtype=np.uint8)
mvs = [memoryview(out[i]) for i in range(N)]
def q(v, p): s = sorted(v); return s[min(len(s) - 1, int(p * len(s)))]
fd_buf = os.open(path, os.O_RDONLY)
fd_rand = os.open(path, os.O_RDONLY); os.posix_fadvise(fd_rand, 0, 0, os.POSIX_FADV_RANDOM)
fd_dir = os.open(path, os.O_RDONLY | os.O_DIRECT)
class A:                      # aligned 4 KB buffers per thread slot
    pass
bufs = [mmap.mmap(-1, 4096) for _ in range(N)]
def rd_buf(fd, idxs, pos):
    for i in idxs:
        os.preadv(fd, [mvs[i]], int(pos[i]))
def rd_dir(idxs, pos):
    for i in idxs:
        p = int(pos[i]); a = p & ~4095
        os.preadv(fd_dir, [bufs[i]], a)
        out[i] = np.frombuffer(bufs[i], dtype=np.uint8, count=rb, offset=p - a) if (p - a) + rb <= 4096 else 0
def rd_nowait(idxs, pos, miss):
    for i in idxs:
        try:
            os.preadv(fd_buf, [mvs[i]], int(pos[i]), os.RWF_NOWAIT)
        except BlockingIOError:
            miss.append(i)
def run(name, fn, rounds=R):
    ts = []
    for r in range(rounds):
        pos = o0 + rng.integers(0, rows, size=N) * rb
        t = time.perf_counter(); fn(pos); ts.append((time.perf_counter() - t) * 1000)
    print(f"{name:40s} cold: med {statistics.median(ts):.2f} p90 {q(ts, .9):.2f} max {max(ts):.2f} ms", flush=True)
def pooled(fd, W, pool):
    def f(pos):
        per = -(-N // W)
        fs = [pool.submit(rd_buf, fd, range(a, min(N, a + per)), pos) for a in range(per, N, per)]
        rd_buf(fd, range(0, min(N, per)), pos)
        for x in fs: x.result()
    return f
for W in (64, 128, 256):
    pool = ThreadPoolExecutor(max_workers=W)
    run(f"buffered {W} workers", pooled(fd_buf, W, pool))
    run(f"fadvise RANDOM {W} workers", pooled(fd_rand, W, pool))
    def fdir(pos, W=W, pool=pool):
        per = -(-N // W)
        fs = [pool.submit(rd_dir, range(a, min(N, a + per)), pos) for a in range(per, N, per)]
        rd_dir(range(0, min(N, per)), pos)
        for x in fs: x.result()
    run(f"O_DIRECT 4K {W} workers", fdir)
pool = ThreadPoolExecutor(max_workers=128)
def hyb(pos):
    miss = []
    rd_nowait(range(N), pos, miss)
    if miss:
        W = 128; per = max(1, -(-len(miss) // W))
        fs = [pool.submit(rd_buf, fd_rand, miss[a:a + per], pos) for a in range(per, len(miss), per)]
        rd_buf(fd_rand, miss[:per], pos)
        for x in fs: x.result()
run("NOWAIT + RANDOM 128 (cold)", hyb)
pos = o0 + rng.integers(0, rows, size=N) * rb
hyb(pos)
ts = []
for r in range(100):
    t = time.perf_counter(); hyb(pos); ts.append((time.perf_counter() - t) * 1000)
print(f"NOWAIT hybrid warm same rows: med {statistics.median(ts):.2f} p90 {q(ts, .9):.2f} max {max(ts):.2f} ms")
