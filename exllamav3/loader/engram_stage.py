"""Host staging of the Engram rows of a graph step (HY4_GRAPH), vectorised.

The graph step reads its Engram rows from static device buffers (eng_raw / eng_valid) that the host fills before every replay: for every Engram layer e, every
verify row t and every column c the id of the row in the (65 GB, file resident) table is a hash of the last n_gram token ids, the bytes are pread into pinned
staging, one H2D each. step_graph.GraphDecoder used a Python hash per (layer, row) and one future per ~4 reads; here

  * ONE numpy pass hashes every (layer, row, column) of the step (uint64 arithmetic wraps exactly like the `& MASK64` of engram_rows.engram_hash),
  * the id bounds are checked in one vectorised test (a bad id would read the next tensor of the file),
  * reads go to cached memoryviews of the pinned staging, a few big chunks (the calling thread takes one itself: no hand-off for small steps).

Generic: nothing is keyed on a model name; every parameter comes from the Hy4EngramGPU modules.
"""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MASK64 = (1 << 64) - 1


class EngramStager:
    """mods: Hy4EngramGPU modules (index e = mod.e), table: F8Rows, rpin [nE, TM, nc, rb] uint8 pinned, vpin [nE, TM, nc] float32 pinned (host views)"""

    def __init__(self, mods, table, rnp: np.ndarray, vnp: np.ndarray, pool: ThreadPoolExecutor | None = None, workers: int | None = None):
        self.mods = sorted(mods, key=lambda m: m.e)
        m0 = self.mods[0]
        self.nE = len(self.mods)
        assert [m.e for m in self.mods] == list(range(self.nE))
        self.ng, self.nh, self.nc = m0.n_gram, m0.n_head, m0.n_cols
        self.table = table
        self.rnp, self.vnp = rnp, vnp
        self.TM = rnp.shape[1]
        assert rnp.shape == (self.nE, self.TM, self.nc, table.row_bytes) and vnp.shape == (self.nE, self.TM, self.nc)
        tm = np.asarray(list(m0.token_map), dtype=np.int64)
        self.tm, self.pad = tm, int(tm[m0.pad_id])
        for m in self.mods:                                   # one token map / pad for every module (the hash below shares them)
            assert m.pad_id == m0.pad_id and m.n_gram == self.ng and m.n_head == self.nh
        self.mult = np.array([[int(x) & MASK64 for x in m.mult] for m in self.mods], dtype=np.uint64)       # [nE, ng]
        self.primes = np.array([[int(x) for x in m.primes] for m in self.mods], dtype=np.uint64)           # [nE, nc]
        self.offs = np.array([[int(x) for x in m.offs] for m in self.mods], dtype=np.int64)                # [nE, nc]
        tab = table
        fi = tab.off >> 48
        self.fd, self.o0, self.rb, self.nrows = tab.g.fds[fi], tab.g.bases[fi] + (tab.off & ((1 << 48) - 1)), tab.row_bytes, tab.rows
        self.views = {}                                       # (e, t, c) -> memoryview of the staging row
        self.pool = pool
        self.workers = workers if workers is not None else int(os.environ.get("HY4_STAGE_WORKERS", "4"))
        self.inline_max = int(os.environ.get("HY4_STAGE_INLINE", "0"))     # <= this many reads: no pool at all

    # -- hash ----------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def window(tokens, lo: int, hi: int, extra=()):
        """token ids at positions lo .. hi - 1 (-1 before the sequence start); positions past len(tokens) come from `extra`"""
        n = len(tokens)
        seg = [-1] * max(0, -lo) + list(tokens[max(lo, 0):min(hi, n)])
        if hi > n:
            seg += list(extra)[max(0, max(lo, n) - n):hi - n]
        if len(seg) != hi - lo:
            raise IndexError(f"token history too short: need positions {lo}..{hi - 1}, have {n} + {len(extra)}")
        return np.asarray(seg, dtype=np.int64)

    def hash(self, tokens, start: int, t0: int, t1: int, extra=()):
        """positions start + t for t in [t0, t1): -> (ids int64 [nE, T, nc], valid float32 [T]) with T = t1 - t0. `tokens` (+ `extra`) holds the history through start + t1 - 1."""
        T, ng = t1 - t0, self.ng
        a = self.window(tokens, start + t0 - (ng - 1), start + t1, extra)
        n_tm = len(self.tm)
        ok = (a >= 0) & (a < n_tm)
        c = np.where(ok, self.tm[np.where(ok, a, 0)], -1)                 # class id, -1 = not in the map / before the sequence start
        # window[t, s] = class of the token at position start + t0 + t - s
        idx = np.arange(T)[:, None] + (ng - 1) - np.arange(ng)[None, :]
        w = c[idx]
        dead = w[:, 0] == -1
        w = np.where(w == -1, self.pad, w).astype(np.uint64)
        prod = w[None] * self.mult[:, None, :]                            # [nE, T, ng], wraps mod 2^64
        ids = np.empty((self.nE, T, self.nc), dtype=np.int64)
        r = prod[:, :, 0].copy()
        for o in range(1, ng):
            r ^= prod[:, :, o]
            sl = slice((o - 1) * self.nh, o * self.nh)
            ids[:, :, sl] = (r[:, :, None] % self.primes[:, None, sl]).astype(np.int64) + self.offs[:, None, sl]
        return ids, np.where(dead, 0.0, 1.0).astype(np.float32)

    # -- reads ---------------------------------------------------------------------------------------------------------------------------
    def _view(self, e, t, c):
        v = self.views.get((e, t, c))
        if v is None:
            v = self.views[(e, t, c)] = memoryview(self.rnp[e, t, c])
        return v

    def _read(self, items):
        fd, o0, rb = self.fd, self.o0, self.rb
        for e, t, c, r in items:
            mv = self._view(e, t, c)
            if os.preadv(fd, [mv], o0 + r * rb) != rb:
                raise IOError(f"{self.table.name}: short read of row {r}")

    def submit(self, ids, valid, t0: int):
        """ids int64 [nE, T, nc] (from hash), valid [T]: fill the staging rows t0 .. t0 + T - 1 -> futures to wait on (the calling thread's own share is already done)"""
        T = ids.shape[1]
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.nrows):
            raise IndexError(f"{self.table.name}: row id outside 0..{self.nrows - 1}")
        if t0 + T > self.TM:
            raise IndexError(f"rows {t0}..{t0 + T - 1} past the staging buffer ({self.TM})")
        self.vnp[:, t0:t0 + T, :] = valid[None, :, None]
        items = [(e, t0 + t, c, int(ids[e, t, c])) for e in range(self.nE) for t in range(T) for c in range(self.nc)]
        n = len(items)
        if self.pool is None or self.workers <= 1 or n <= self.inline_max:
            self._read(items)
            return []
        per = -(-n // self.workers)
        chunks = [items[a:a + per] for a in range(0, n, per)]
        futs = [self.pool.submit(self._read, ch) for ch in chunks[1:]]
        self._read(chunks[0])
        return futs

    def stage(self, tokens, start: int, t0: int, t1: int, extra=()):
        ids, valid = self.hash(tokens, start, t0, t1, extra)
        return self.submit(ids, valid, t0)
