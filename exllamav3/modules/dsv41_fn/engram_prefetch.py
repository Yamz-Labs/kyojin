# Asynchronous Engram row prefetch (DeepSeek-V4.1 layers 1 and 14).
#
# The n-gram hash only depends on token ids, which the host knows when a step starts (prompt chunk, or the last accepted token plus the
# draft block of a verify pass). So the whole Engram fetch can run in a background thread while the GPU computes layer 0 (layer 1's
# rows) and layers 0..13 (layer 14's rows):
#
#   submit(ids, start)          host hash (numpy, bit-exact with dsv41/engram.py::NgramHasher) -> unique rows per layer
#   worker thread               hot-row RAM cache lookup -> misses read by the store (O_DIRECT, threaded C++ gather, GIL released)
#                               -> rows copied into a pinned staging slot -> async H2D copy on a side stream, one event per layer
#   take(ticket, layer_pos)     (raw uint8 [U, 136 or 264] on the device, inverse int32 [R * nc]), the compute stream waits on the copy event
#   release(ticket)            frees the slot once the compute stream is done with the device buffers
#
# For prefill the caller submits chunk k + 1 before computing chunk k (`depth` tickets in flight). The rows are the table bytes, so the
# result is identical to a synchronous `store.fetch`; the cache only changes where the bytes come from (tables are immutable).
import collections
import queue
import threading
import time

import numpy as np
import torch

ROW_BYTES = 136                       # MXFP4 rows; FP8 rows are 264 bytes (see row_bytes_of)


def row_bytes_of(store, head_dim: int = 256) -> int:
    """Bytes per fetched row: head_dim/2 + head_dim/32 (MXFP4) or head_dim + head_dim/32 (FP8)."""
    return (head_dim if getattr(store, "fmt", "mxfp4") == "fp8" else head_dim // 2) + head_dim // 32
DEAD = -1


class HostEngramHash:
    """numpy twin of NgramHasher for a single sequence: ids int64 [R] at positions start .. start + R - 1 -> rows int64 [R, nl, nc]."""

    def __init__(self, layout, token_map, pad_token_id: int, multipliers, max_seq: int):
        self.layout = layout
        self.nl, self.ng, self.nh = len(layout.layer_ids), layout.max_ngram_size, layout.n_heads
        self.tmap = np.asarray(token_map, dtype=np.int64)
        self.pad = int(self.tmap[pad_token_id])
        self.mult = np.asarray(multipliers, dtype=np.int64)                                        # [nl, ng]
        self.primes = np.asarray(layout.primes, dtype=np.int64).reshape(self.nl, self.ng - 1, self.nh)
        self.offsets = np.stack([layout.offsets(i) for i in range(self.nl)]).astype(np.int64)      # [nl, nc]
        self.cache = np.full(max_seq, DEAD, dtype=np.int64)

    def reset(self):
        self.cache.fill(DEAD)

    def __call__(self, ids: np.ndarray, start: int) -> np.ndarray:
        R = ids.size
        if start + R > self.cache.size:
            grown = np.full(max(2 * self.cache.size, start + R), DEAD, dtype=np.int64)
            grown[:self.cache.size] = self.cache
            self.cache = grown
        self.cache[start:start + R] = self.tmap[ids]
        pos = np.arange(start, start + R)
        blocked = np.zeros(R, dtype=bool)
        toks = []
        for sh in range(self.ng):
            src = self.cache[np.maximum(pos - sh, 0)]
            blocked = blocked | (pos < sh) | (src == DEAD)
            toks.append(np.where(blocked, self.pad, src))
        toks = np.stack(toks, axis=-1)                                                              # [R, ng]
        with np.errstate(over="ignore"):
            prod = toks[:, None, :] * self.mult[None]                                               # [R, nl, ng]
        rolling, hashes = prod[..., 0], []
        for i in range(1, self.ng):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            hashes.append(rolling[..., None] % self.primes[:, i - 1][None])                         # [R, nl, nh]
        return np.concatenate(hashes, axis=-1) + self.offsets


class RowCache:
    """Fixed-budget LRU of Engram rows (all layers share the pool). Single owner thread, no lock."""

    def __init__(self, budget_bytes: int, row_bytes: int = ROW_BYTES):
        self.cap = max(0, budget_bytes // row_bytes)
        self.data = np.empty((self.cap, row_bytes), dtype=np.uint8) if self.cap else None
        self.map = collections.OrderedDict()
        self.free = list(range(self.cap - 1, -1, -1))
        self.hits = self.lookups = 0

    def lookup(self, lp: int, rows: np.ndarray):
        """-> (slots int64 [n], -1 where missing)."""
        base = lp << 32
        m, slots = self.map, np.full(rows.size, -1, dtype=np.int64)
        for i, r in enumerate(rows.tolist()):
            s = m.get(base | r)
            if s is not None:
                m.move_to_end(base | r)
                slots[i] = s
        self.lookups += rows.size
        self.hits += int((slots >= 0).sum())
        return slots

    def insert(self, lp: int, rows: np.ndarray, raw: np.ndarray):
        if not self.cap:
            return
        base, m = lp << 32, self.map
        for i, r in enumerate(rows.tolist()):
            k = base | r
            if k in m:
                continue
            if self.free:
                s = self.free.pop()
            else:
                _, s = m.popitem(last=False)
            m[k] = s
            self.data[s] = raw[i]


class _Slot:
    def __init__(self, nl, max_rows, max_inv, device, row_bytes=ROW_BYTES):
        pin = device is not None
        mk = lambda *shape, dt: torch.empty(*shape, dtype=dt, pin_memory=pin)
        self.raw = [mk(max_rows, row_bytes, dt=torch.uint8) for _ in range(nl)]
        self.inv = [mk(max_inv, dt=torch.int32) for _ in range(nl)]
        if pin:
            self.draw = [torch.empty(max_rows, row_bytes, dtype=torch.uint8, device=device) for _ in range(nl)]
            self.dinv = [torch.empty(max_inv, dtype=torch.int32, device=device) for _ in range(nl)]
            self.copied = [torch.cuda.Event() for _ in range(nl)]
            self.consumed = None                    # event recorded on the compute stream by release()


class EngramPrefetcher:
    """store: an EngramRowStore (fetch(layer_pos, int64 rows) -> uint8 [n, 136 or 264]). device: "cuda" device or None (host tensors)."""

    def __init__(self, layout, token_map, pad_token_id: int, multipliers, store, max_seq: int = 4096, cache_bytes: int = 1 << 30,
                 device=None, max_rows: int = 2048, slots: int = 3):
        self.nl, self.nc = len(layout.layer_ids), (layout.max_ngram_size - 1) * layout.n_heads
        self.hash = HostEngramHash(layout, token_map, pad_token_id, multipliers, max_seq)
        self._hash_args = (layout, token_map, pad_token_id, multipliers)
        self.cache_lock = threading.Lock()              # the row cache is shared by the ticket worker and the lookahead thread
        self._la = None
        self.rb = row_bytes_of(store)
        self.store, self.cache, self.device = store, RowCache(cache_bytes, self.rb), device
        self.max_rows = max_rows
        self.slots = [_Slot(self.nl, max_rows * self.nc, max_rows * self.nc, device, self.rb) for _ in range(slots)]
        self.free = queue.Queue()
        for s in self.slots:
            self.free.put(s)
        self.q = queue.Queue()
        self.side = torch.cuda.Stream(device=device) if device is not None else None
        self.stat = collections.Counter()
        self.t_worker = 0.0
        self._abort_warm = False
        self._spec = None                           # ticket built ahead by speculate() for the most likely next token
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True, name="engram-prefetch")
        self.thread.start()

    # ------------------------------------------------------------------------------------------------ worker
    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                self._run(job)
            except BaseException as e:               # noqa: BLE001 - surfaced by take()
                job["err"] = e
                for ev in job.get("ready", ()):
                    ev.set()

    def _gather(self, lp: int, uniq: np.ndarray) -> np.ndarray:
        c = self.cache
        out = np.empty((uniq.size, self.rb), dtype=np.uint8)
        with self.cache_lock:
            slots = c.lookup(lp, uniq) if c.cap else np.full(uniq.size, -1, dtype=np.int64)
            hit = slots >= 0
            if hit.any():
                out[hit] = c.data[slots[hit]]
        miss = np.flatnonzero(~hit)
        if miss.size:
            mrows = uniq[miss]
            raw = self.store.fetch(lp, torch.from_numpy(mrows)).numpy()
            out[miss] = raw
            with self.cache_lock:
                c.insert(lp, mrows, raw)
            self.stat["rows_read"] += miss.size
        self.stat["rows_unique"] += uniq.size
        return out

    def _warm(self, job):
        """Speculative reads: pull the rows of candidate next tokens into the row cache (no ticket, no device copy)."""
        c = self.cache
        keep = (c.lookups, c.hits, self.stat["rows_unique"], self.stat["rows_read"], self.stat["rows_read"])
        st = job["start"]
        old = int(self.hash.cache[st]) if st < self.hash.cache.size else DEAD
        for tok in job["warm"]:
            if self._abort_warm:                                 # the real step arrived: its own reads come first
                break
            rows = self.hash(np.asarray([tok], dtype=np.int64), st)
            for lp in range(self.nl):
                self._gather(lp, np.unique(rows[:, lp, :].reshape(-1)))
        if st < self.hash.cache.size:
            self.hash.cache[st] = old                            # hash history keeps the best guess (or the real token)
        self.stat["spec_rows_read"] += self.stat["rows_read"] - keep[3]
        c.lookups, c.hits, self.stat["rows_unique"], self.stat["rows_read"] = keep[:4]     # speculation does not count as use

    def _run(self, job):
        if "warm" in job:
            return self._warm(job)
        t0 = time.perf_counter()
        slot = job["slot"]
        ids = job["ids"]
        self._la_wait(job["start"], ids.size)
        rows = self.hash(ids, job["start"])                                             # [R, nl, nc]
        job["rows"] = rows
        if self.device is not None and slot.consumed is not None:
            self.side.wait_event(slot.consumed)
        for lp in range(self.nl):
            flat = rows[:, lp, :].reshape(-1)
            uniq, inv = np.unique(flat, return_inverse=True)
            raw = self._gather(lp, uniq)
            n = uniq.size
            slot.raw[lp][:n].numpy()[:] = raw
            slot.inv[lp][:flat.size].numpy()[:] = inv.reshape(-1).astype(np.int32)
            job["n"][lp] = n
            if self.device is not None:
                with torch.cuda.stream(self.side):
                    slot.draw[lp][:n].copy_(slot.raw[lp][:n], non_blocking=True)
                    slot.dinv[lp][:flat.size].copy_(slot.inv[lp][:flat.size], non_blocking=True)
                    slot.copied[lp].record(self.side)
            job["ready"][lp].set()
        self.t_worker += time.perf_counter() - t0

    # ------------------------------------------------------------------------------------------------ whole-prompt lookahead
    def lookahead(self, ids, start: int, chunk: int, depth: int = 2, ram_fraction: float = 0.8):
        """Prefill: the token ids of the whole prompt are known, so the rows of every chunk can be hashed at once. A background thread
        hashes the chunks of ids (positions start ..) in order and reads their missing rows into the row cache, both layers at the
        same time (one reader pool per layer), at most `depth` chunks beyond the chunk the model is on and never more unconsumed rows
        than `ram_fraction` of the cache. The tickets (submit/take) then hit the cache: same bytes, the NVMe latency is hidden behind
        the compute of the earlier chunks. The first chunk waits for its own reads (nothing to hide behind). Needs a row cache."""
        self.lookahead_stop()
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        n = -(-ids.size // chunk)
        la = {"ids": ids, "start": int(start), "chunk": int(chunk), "depth": int(depth), "n": n, "cur": 0, "stop": False,
              "ev": [threading.Event() for _ in range(n)], "rows": [0] * n, "err": None, "cv": threading.Condition(),
              "budget": int(ram_fraction * self.cache.cap), "read": 0, "t": 0.0}
        if not self.cache.cap:
            for e in la["ev"]:
                e.set()
        la["thread"] = threading.Thread(target=self._la_loop, args=(la,), daemon=True, name="engram-lookahead")
        self._la = la
        if self.cache.cap:
            la["hash"] = HostEngramHash(*self._hash_args, max_seq=max(64, start + ids.size))
            la["hash"].cache[:min(start, self.hash.cache.size)] = self.hash.cache[:start]      # history before the prompt chunk
            la["thread"].start()

    def lookahead_stop(self):
        la, self._la = self._la, None
        if la is not None:
            with la["cv"]:
                la["stop"] = True
                la["cv"].notify_all()
            for e in la["ev"]:
                e.set()
            if la["thread"].is_alive():
                la["thread"].join()

    def _la_wait(self, start: int, R: int):
        la = self._la
        if la is None or R <= 0:
            return
        a, b = (start - la["start"]) // la["chunk"], (start + R - 1 - la["start"]) // la["chunk"]
        if b < 0 or a >= la["n"]:
            return
        a = max(a, 0)
        with la["cv"]:
            la["cur"] = max(la["cur"], a)
            la["cv"].notify_all()
        for k in range(a, min(b, la["n"] - 1) + 1):
            la["ev"][k].wait()
        if la["err"] is not None:
            raise la["err"]

    def _la_loop(self, la):
        import concurrent.futures as cf
        try:
            pool = cf.ThreadPoolExecutor(max_workers=self.nl, thread_name_prefix="engram-la")
            ids, C = la["ids"], la["chunk"]
            for k in range(la["n"]):
                with la["cv"]:
                    while not la["stop"] and (k > la["cur"] + la["depth"] or sum(la["rows"][la["cur"] + 1:k]) > la["budget"]):
                        la["cv"].wait()
                    if la["stop"]:
                        return
                t0 = time.perf_counter()
                rows = la["hash"](ids[k * C:(k + 1) * C], la["start"] + k * C)
                uniq = [np.unique(rows[:, lp, :].reshape(-1)) for lp in range(self.nl)]
                la["rows"][k] = sum(u.size for u in uniq)
                futs = [pool.submit(self._la_fill, lp, uniq[lp], la) for lp in range(self.nl)]
                for f in futs:
                    f.result()
                la["t"] += time.perf_counter() - t0
                la["ev"][k].set()
            pool.shutdown()
        except BaseException as e:                       # noqa: BLE001 - surfaced by the next ticket
            la["err"] = e
            for ev in la["ev"]:
                ev.set()

    def _la_fill(self, lp: int, uniq: np.ndarray, la):
        c = self.cache
        with self.cache_lock:
            keep = (c.lookups, c.hits)
            slots = c.lookup(lp, uniq)
            c.lookups, c.hits = keep                             # warming does not count as use
        miss = uniq[slots < 0]
        if miss.size:
            raw = self.store.fetch(lp, torch.from_numpy(miss)).numpy()
            with self.cache_lock:
                c.insert(lp, miss, raw)
            la["read"] += miss.size

    # ------------------------------------------------------------------------------------------------ API
    def reset(self):
        """New sequence: forget the token history (queued tickets must be taken first)."""
        self.lookahead_stop()
        self.hash.reset()

    def speculate(self, cands, start: int):
        """Call as soon as the logits of the step before position `start` are known: `cands` = likely tokens for that position, best
        first. The best one gets a complete ticket (hash, rows, H2D) that submit() hands back when the sampled token matches; the others
        only warm the row cache. A wrong guess costs reads, never correctness: the rows are table bytes keyed by the real hash."""
        cands = [int(t) for t in (cands.tolist() if hasattr(cands, "tolist") else cands)]
        self.drop_speculation()
        self._abort_warm = False
        self._spec = self._new_job(np.asarray(cands[:1], dtype=np.int64), start)        # first: the likely ticket is the critical one
        if len(cands) > 1:
            self.q.put({"warm": cands[1:], "start": int(start)})

    def drop_speculation(self):
        sp, self._spec = self._spec, None
        if sp is not None:
            self.release(sp)

    def submit(self, ids, start: int) -> dict:
        """ids: int tensor / array [R] (host) at positions start ..; returns a ticket. Blocks when every slot is in flight."""
        ids = np.asarray(ids.cpu().numpy() if torch.is_tensor(ids) else ids, dtype=np.int64).reshape(-1)
        assert ids.size <= self.max_rows, "chunk larger than the staging slots"
        sp, self._spec = self._spec, None
        self._abort_warm = True
        if sp is not None:
            if sp["start"] == int(start) and np.array_equal(sp["ids"], ids):
                self.stat["spec_hit"] += 1
                return sp
            self.stat["spec_miss"] += 1
            self.release(sp)
        return self._new_job(ids, start)

    def _new_job(self, ids, start: int) -> dict:
        slot = self.free.get()
        if self.device is not None:
            for ev in slot.copied:                                   # pinned buffers are free once the previous H2D finished
                ev.synchronize()
        job = {"ids": ids, "start": int(start), "slot": slot, "n": [0] * self.nl, "R": ids.size,
               "ready": [threading.Event() for _ in range(self.nl)], "err": None}
        self.q.put(job)
        return job

    def take(self, job: dict, layer_pos: int):
        """-> (raw uint8 [U, 136 or 264], inv int32 [R * nc]) on the device (the current stream waits on the copy) or on the host."""
        job["ready"][layer_pos].wait()
        if job["err"] is not None:
            raise job["err"]
        s, n, m = job["slot"], job["n"][layer_pos], job["R"] * self.nc
        if self.device is None:
            return s.raw[layer_pos][:n], s.inv[layer_pos][:m]
        torch.cuda.current_stream().wait_event(s.copied[layer_pos])
        return s.draw[layer_pos][:n], s.dinv[layer_pos][:m]

    def release(self, job: dict):
        """Call after the last use of the ticket's buffers was enqueued on the compute stream."""
        for ev in job["ready"]:
            ev.wait()
        s = job["slot"]
        if self.device is not None:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            s.consumed = ev
        self.free.put(s)

    def fetch_sync(self, ids, start: int):
        """Submit + take every layer + release (the synchronous reference through the same code path)."""
        job = self.submit(ids, start)
        out = [tuple(t.clone() for t in self.take(job, lp)) for lp in range(self.nl)]
        self.release(job)
        return out

    def close(self):
        self.lookahead_stop()
        self.q.put(None)
        self.thread.join()

    def stats(self) -> dict:
        c = self.cache
        return {"cache_rows": c.cap, "lookups": c.lookups, "hits": c.hits, "hit_rate": c.hits / max(1, c.lookups),
                "rows_unique": self.stat["rows_unique"], "rows_read": self.stat["rows_read"], "worker_s": self.t_worker,
                "spec_hit": self.stat["spec_hit"], "spec_miss": self.stat["spec_miss"], "spec_rows_read": self.stat["spec_rows_read"],
                "la_rows_read": self._la["read"] if self._la else 0, "la_s": self._la["t"] if self._la else 0.0}
