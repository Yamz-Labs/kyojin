"""
Plumbing for the persistent prompt cache on NVMe (every local model): a namespace directory, atomic writes from one
background thread, an LRU byte cap, a write-rate limit and a live free-space floor. The payload formats live in the
adapters (exllamav3/generator/slot_store.py for the paged GLM/Qwen caches, exllamav3/dsv41/prefix_cache.py for DS4.1).

An entry is two files: `<name>.pt` (payload) then `<name>.idx` (small index record, written last: an entry without its
index is torn and ignored/removed). Eviction removes whole entries, least recently used first (mtime, touched on a hit).

Knobs (env):
  EXL3_PREFIX_DISK_DIR      root directory; empty = off
  EXL3_PREFIX_DISK_GB       byte cap of the whole root, all models and builds together (default 30)
  EXL3_PREFIX_DISK_FLOOR_GB free space that must remain on the volume after a write (default 60)
  EXL3_PREFIX_DISK_MB_MIN   write budget per minute (default 8000: the QLC write cliff starts near 10 GB/min)
"""
from __future__ import annotations

import atexit
import hashlib
import os
import queue
import shutil
import threading
import time

import torch

GIB = 1 << 30


def env_dir() -> str:
    return os.path.expanduser(os.environ.get("EXL3_PREFIX_DISK_DIR", "") or "")


def namespace(*parts) -> str:
    """Directory-safe fingerprint of everything that makes a saved state unusable by another model/build."""
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def ids_hash(ids) -> str:
    t = ids if isinstance(ids, torch.Tensor) else torch.tensor(list(ids))
    return hashlib.sha1(t.view(-1).to("cpu", torch.int64).numpy().tobytes()).hexdigest()


class DiskStore:

    def __init__(self, root: str, ns: str, cap_gb: float | None = None, floor_gb: float | None = None,
                 mb_per_min: float | None = None):
        self.dir = os.path.join(os.path.expanduser(root), ns)
        os.makedirs(self.dir, exist_ok=True)
        e = os.environ.get
        self.cap = int(float(e("EXL3_PREFIX_DISK_GB", "30") if cap_gb is None else cap_gb) * GIB)
        self.floor = int(float(e("EXL3_PREFIX_DISK_FLOOR_GB", "60") if floor_gb is None else floor_gb) * GIB)
        self.rate = float(e("EXL3_PREFIX_DISK_MB_MIN", "8000") if mb_per_min is None else mb_per_min) * (1 << 20) / 60.0
        self.stats = {"written": 0, "bytes": 0, "skipped_floor": 0, "skipped_cap": 0, "evicted": 0, "errors": 0, "hits": 0}
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="prefix-disk", daemon=True)
        self._closed = False
        self._worker.start()
        atexit.register(self.close)
        self.sweep()

    # ---- entries

    def _p(self, name: str, ext: str) -> str:
        return os.path.join(self.dir, name + ext)

    def entries(self) -> list[str]:
        """Names of the committed entries (index present)."""
        return sorted(f[:-4] for f in os.listdir(self.dir) if f.endswith(".idx"))

    def size(self, name: str) -> int:
        try:
            return os.path.getsize(self._p(name, ".pt")) + os.path.getsize(self._p(name, ".idx"))
        except OSError:
            return 0

    def total(self) -> int:
        return sum(self.size(n) for n in self.entries())

    def read_idx(self, name: str):
        try:
            return torch.load(self._p(name, ".idx"), map_location="cpu", weights_only=False)
        except Exception:                                                        # noqa: BLE001 torn / foreign
            return None

    def read(self, name: str):
        d = torch.load(self._p(name, ".pt"), map_location="cpu", weights_only=False)
        self.touch(name)
        self.stats["hits"] += 1
        return d

    def touch(self, name: str) -> None:
        try:
            os.utime(self._p(name, ".idx"))
        except OSError:
            pass

    def remove(self, name: str) -> None:
        for ext in (".idx", ".pt"):                                              # index first: no entry without payload
            try:
                os.remove(self._p(name, ext))
            except OSError:
                pass

    def sweep(self) -> None:
        """Drop torn files (payload without index, leftover tmp) from an interrupted run."""
        have = set(self.entries())
        for f in os.listdir(self.dir):
            if f.endswith(".tmp") or (f.endswith(".pt") and f[:-3] not in have):
                try:
                    os.remove(os.path.join(self.dir, f))
                except OSError:
                    pass

    def _all(self) -> list[tuple[float, str, str, int]]:
        """(mtime, namespace dir, name, bytes) of every committed entry under the root: the cap is shared by all models and builds."""
        root = os.path.dirname(self.dir)
        out = []
        for ns in os.listdir(root):
            d = os.path.join(root, ns)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".idx"):
                    try:
                        sz = os.path.getsize(os.path.join(d, f)) + os.path.getsize(os.path.join(d, f[:-4] + ".pt"))
                        out.append((os.path.getmtime(os.path.join(d, f)), d, f[:-4], sz))
                    except OSError:
                        pass
        return out

    def evict(self, need: int = 0) -> None:
        """Make room for `need` more bytes under the cap, least recently used first, across all namespaces of the root
        (a namespace left behind by an old build or an old pack has the oldest mtimes and goes first)."""
        items = sorted(self._all())
        tot = sum(i[3] for i in items)
        for _, d, n, sz in items:
            if tot + need <= self.cap:
                break
            for ext in (".idx", ".pt"):
                try:
                    os.remove(os.path.join(d, n + ext))
                except OSError:
                    pass
            tot -= sz
            self.stats["evicted"] += 1

    # ---- async writes

    def put(self, name: str, payload, idx) -> None:
        """Queue an entry: payload and idx are torch.save-able and must not change afterwards (host copies)."""
        if not self._closed:
            self._q.put((name, payload, idx))

    def flush(self) -> None:
        self._q.join()

    def close(self) -> None:
        if self._closed:
            return
        self._q.join()
        self._closed = True
        self._q.put(None)
        self._worker.join(timeout=30)

    def _run(self) -> None:
        window_start, window_bytes = time.monotonic(), 0
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                name, payload, idx = item
                if os.path.exists(self._p(name, ".idx")):
                    self.touch(name)
                    continue
                tmp = self._p(name, ".pt.tmp")
                try:
                    torch.save(payload, tmp)
                    n = os.path.getsize(tmp)
                    free = shutil.disk_usage(self.dir).free
                    if free - n < self.floor:                                    # live check: other writers share the disk
                        os.remove(tmp)
                        self.stats["skipped_floor"] += 1
                        continue
                    if n > self.cap:
                        os.remove(tmp)
                        self.stats["skipped_cap"] += 1
                        continue
                    with self._lock:
                        self.evict(n)
                        os.replace(tmp, self._p(name, ".pt"))
                        itmp = self._p(name, ".idx.tmp")
                        torch.save(idx, itmp)
                        os.replace(itmp, self._p(name, ".idx"))
                    self.stats["written"] += 1
                    self.stats["bytes"] += n
                    window_bytes += n
                    spent = time.monotonic() - window_start
                    ahead = window_bytes / self.rate - spent                    # stay under the per-minute write budget
                    if ahead > 0:
                        time.sleep(min(ahead, 120.0))
                    if spent > 60:
                        window_start, window_bytes = time.monotonic(), 0
                except Exception:                                                # noqa: BLE001 a failed write never fails a request
                    self.stats["errors"] += 1
                    for ext in (".pt.tmp", ".idx.tmp"):
                        try:
                            os.remove(self._p(name, ext))
                        except OSError:
                            pass
            finally:
                self._q.task_done()
