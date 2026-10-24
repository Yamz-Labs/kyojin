# Parallel O_DIRECT row gather for the Engram tables (dsv41/engram.py::DirectRowReader issues one preadv per row at queue depth 1:
# ~80 us per random 4 KiB read, i.e. ~4 ms for the 48 rows of one decode token and ~8 s for the ~98K rows of a 4096-token prefill chunk).
# Here the unique (file, 4 KiB sector) windows of a batch are spread over a thread pool; os.preadv releases the GIL, so the reads
# overlap at queue depth `threads`. Same interface and bytes as DirectRowReader (drop-in for FileEngramStore readers).
import mmap, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

_SECTOR = 4096


class ParallelRowReader:
    def __init__(self, segments, threads: int = 16, allow_buffered: bool = False):
        self.segments = [(str(s[0]),) + tuple(int(v) for v in s[1:]) for s in segments]    # (path, base, stride, nbytes)
        self.row_bytes = sum(s[3] for s in self.segments)
        self.direct = True
        self.fds = {}
        for path in dict.fromkeys(s[0] for s in self.segments):
            try:
                self.fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
            except OSError:
                if not allow_buffered:
                    raise
                self.fds[path] = os.open(path, os.O_RDONLY)
                self.direct = False
        self.threads = threads
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self._bufs = [mmap.mmap(-1, 8 * _SECTOR) for _ in range(threads)]          # one aligned buffer per worker
        self.reads = 0
        self.bytes_read = 0

    def close(self):
        self.pool.shutdown(wait=True)
        for b in self._bufs:
            b.close()
        for fd in self.fds.values():
            os.close(fd)
        self.fds = {}

    def _plan(self, rows):
        """-> list of windows (path, start, length, [(out_row, col, off_in_window, nbytes), ...]) sorted by (path, start)."""
        wins = {}
        col = 0
        for path, base, stride, nb in self.segments:
            for i, r in enumerate(rows):
                off = base + r * stride
                a = off // _SECTOR * _SECTOR
                e = -(-(off + nb) // _SECTOR) * _SECTOR
                w = wins.setdefault((path, a), [e, []])
                w[0] = max(w[0], e)
                w[1].append((i, col, off - a, nb))
            col += nb
        return [(p, a, e - a, items) for (p, a), (e, items) in sorted(wins.items())]

    def _run(self, k, work, out):
        buf = self._bufs[k]
        mv = memoryview(buf)
        nread = 0
        for path, a, n, items in work:
            if n > len(buf):                                   # window wider than the per-worker buffer (never for 136 B rows)
                tmp = mmap.mmap(-1, n)
                got = os.preadv(self.fds[path], [memoryview(tmp)], a)
                src = memoryview(tmp)
            else:
                got = os.preadv(self.fds[path], [mv[:n]], a)
                src = mv
            nread += n
            for i, c, o, nb in items:
                if got < o + nb:
                    raise IOError(f"short read in {path} at {a + o}: got {got - o}, need {nb}")
                out[i, c:c + nb] = np.frombuffer(src, dtype=np.uint8, count=nb, offset=o)
        return nread

    def read_rows(self, rows) -> torch.Tensor:
        rows = [int(r) for r in rows]
        out = np.empty((len(rows), self.row_bytes), dtype=np.uint8)
        plan = self._plan(rows)
        k = min(self.threads, max(1, len(plan)))
        shards = [plan[i::k] for i in range(k)]                  # interleaved: neighbouring sectors go to different workers
        futs = [self.pool.submit(self._run, i, shards[i], out) for i in range(k)]
        for f in futs:
            self.bytes_read += f.result()
        self.reads += len(plan)
        if not self.direct:
            for fd in self.fds.values():
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
        return torch.from_numpy(out)


# ---------------------------------------------------------------------------------------------------- C++ pool (engram_gather.cpp)
import ctypes, hashlib, subprocess

_lib = None


def _gather_lib():
    """Compile engram_gather.cpp once (g++ -O2 -pthread, cached by source hash) and load it with ctypes."""
    global _lib
    if _lib is None:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engram_gather.cpp")
        tag = hashlib.sha1(open(src, "rb").read()).hexdigest()[:12]
        so = os.path.join(os.path.expanduser("~/.cache/exllamav3"), f"ds41_engram_gather_{tag}.so")
        if not os.path.isfile(so):
            os.makedirs(os.path.dirname(so), exist_ok=True)
            tmp = so + f".{os.getpid()}.tmp"
            subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-pthread", "-o", tmp, src], check=True)
            os.replace(tmp, so)
        _lib = ctypes.CDLL(so)
        _lib.ds_engram_gather.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                                                  ctypes.c_int64, ctypes.c_int]
        _lib.ds_engram_pool_new.argtypes = [ctypes.c_int]
        _lib.ds_engram_pool_new.restype = ctypes.c_void_p
        _lib.ds_engram_pool_gather.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                                                       ctypes.c_int64]
        _lib.ds_engram_pool_free.argtypes = [ctypes.c_void_p]
    return _lib


class NativeRowReader:
    """ParallelRowReader's interface with the reads done by a persistent C++ thread pool (no GIL, no per-row Python)."""

    def __init__(self, segments, threads: int = 256, allow_buffered: bool = False, private_pool: bool = True):
        """private_pool: own thread pool (queue depth `threads`) so that several readers can gather concurrently."""
        self.segments = [(str(s[0]),) + tuple(int(v) for v in s[1:]) for s in segments]
        self.row_bytes = sum(s[3] for s in self.segments)
        self.direct = True
        paths = list(dict.fromkeys(s[0] for s in self.segments))
        fds = {}
        for path in paths:
            try:
                fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
            except OSError:
                if not allow_buffered:
                    raise
                fds[path] = os.open(path, os.O_RDONLY)
                self.direct = False
        self.fds = fds
        self.threads = threads
        self._fd = (ctypes.c_int * len(self.segments))(*[fds[s[0]] for s in self.segments])
        self._base = (ctypes.c_int64 * len(self.segments))(*[s[1] for s in self.segments])
        self._stride = (ctypes.c_int64 * len(self.segments))(*[s[2] for s in self.segments])
        self._nb = (ctypes.c_int * len(self.segments))(*[s[3] for s in self.segments])
        self.reads = 0
        self.bytes_read = 0
        lib = _gather_lib()
        # two private pools: `threads` deep for prefill batches, a small one for decode batches (waking 256 threads for 24 rows is slow)
        self._pool = lib.ds_engram_pool_new(threads) if private_pool else None
        self._small = lib.ds_engram_pool_new(min(16, threads)) if private_pool else None

    def close(self):
        for name in ("_pool", "_small"):
            if getattr(self, name, None):
                _gather_lib().ds_engram_pool_free(getattr(self, name))
                setattr(self, name, None)
        for fd in self.fds.values():
            os.close(fd)
        self.fds = {}

    def read_rows(self, rows) -> torch.Tensor:
        r = torch.as_tensor(rows, dtype=torch.int64).contiguous()
        out = torch.empty(r.numel(), self.row_bytes, dtype=torch.uint8)
        if self._pool:
            err = _gather_lib().ds_engram_pool_gather(self._small if r.numel() < 1024 else self._pool, self._fd, self._base, self._stride, self._nb, len(self.segments),
                                                      r.data_ptr(), r.numel(), out.data_ptr(), self.row_bytes)
        else:
            err = _gather_lib().ds_engram_gather(self._fd, self._base, self._stride, self._nb, len(self.segments),
                                                 r.data_ptr(), r.numel(), out.data_ptr(), self.row_bytes, self.threads)
        if err:
            raise IOError(f"ds_engram_gather failed ({err})")
        self.reads += r.numel() * len(self.segments)
        self.bytes_read += self.reads * 4096
        return out
