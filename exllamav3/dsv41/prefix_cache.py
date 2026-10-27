"""
Prompt-state checkpoints of the DS4.1 serving engine: a follow-up chat turn prefills only its new tokens.

Why checkpoints and not `truncate(n)`: the state of a position is NOT recoverable from a later one. The sliding-window ring
(slot = position % 128) of every layer is overwritten by the next 128 positions, and the open compressor group is reset when
it completes. Only the block journal of a speculative verify can undo that, and only for one block. So the engine keeps
*snapshots* of the state at chosen positions and resumes from the longest one whose tokens are a prefix of the new prompt.

A snapshot holds exactly what a position needs, compactly and on the host (a few MiB plus ~0.3 KiB per token):
  win[l]                        sliding-window rings (all layers, the DSpark stage rings included)
  comp_kv / idx_k (/ packed)    the first ceil(n / ratio) rows of each pool
  comp_kv_state / score_state   the open compressor groups
  hasher.cache[:, :n]           the Engram token history
  hidden                        the last `win` target rows (DSpark): the drafter rings are rebuilt from them on resume

Where snapshots are taken (`marks`): after the system/tools block (the first user turn), right after the last <Assistant> token
(the generation prompt `<Assistant><think>` flips to `<Assistant></think>` in the next render when earlier reasoning is dropped,
so the prompt end itself is then NOT a prefix), at the end of the prompt, at every prefill chunk end of a long prompt, and after
the reply (agent tool loops resend the reply verbatim). A budget in bytes and a count bound the store (LRU).

The resumed forward is the same chunked forward the engine always ran at `start_pos > 0` ("any chunk length at any start
position"), so reuse is exact up to the numerical noise of a different chunking.
"""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

PERSIST_MIN = 512


def token_hash(keys: Sequence) -> str:
    h = hashlib.sha1()
    h.update(torch.tensor(list(keys), dtype=torch.int64).numpy().tobytes())
    return h.hexdigest()


def fingerprint(cfg, tag: str = "") -> str:
    """Identity of the model state layout: config scalars + DS41_PFX_TAG (set it to the code revision: a numerics change
    makes old snapshots stale)."""
    items = sorted((k, repr(v)) for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool, tuple)))
    return hashlib.sha1((repr(items) + "|" + tag + "|" + os.environ.get("DS41_PFX_TAG", "")).encode()).hexdigest()[:16]


POOLS = ("comp_kv", "idx_k", "idx_kq", "idx_ks")


def mem_available_gib() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return float("inf")


def lcp(a: Sequence, b: Sequence) -> int:
    """Length of the common prefix (equality of prefixes is monotone: binary search over slice compares)."""
    lo, hi = 0, min(len(a), len(b))
    if a[:hi] == b[:hi]:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


@dataclass
class Snapshot:
    keys: list
    n: int
    kind: str                               # "mark" | "turn" | "end" | "grid" | "reply"
    tensors: dict
    hidden: Any = None                      # cpu [rows, D] target rows ending at position n-1 (DSpark), or None
    nbytes: int = 0
    tick: int = 0


def snapshot(cfg, st, n: int, keys: Sequence, hidden: Any, kind: str) -> Snapshot:
    """Copy the state of positions 0..n-1 to the host. `st` must hold exactly those positions (any extra rows are ignored)."""
    cpu = lambda t: t.detach().to("cpu", copy=True)                                  # noqa: E731
    T: dict[str, Any] = {"win": {l: cpu(t) for l, t in st.win.items()}}
    for name in POOLS:
        d = getattr(st, name, None)
        if not d:
            continue
        T[name] = {}
        for l, t in d.items():
            rows = min(-(-n // max(int(cfg.ratio(l)), 1)), t.size(1))
            T[name][l] = cpu(t[:, :rows])
    T["comp_kv_state"] = {l: cpu(t) for l, t in st.comp_kv_state.items()}
    T["comp_score_state"] = {l: cpu(t) for l, t in st.comp_score_state.items()}
    if getattr(st, "hasher", None) is not None:
        T["hasher"] = cpu(st.hasher.cache[:, :n])
    h = None if hidden is None else cpu(hidden)
    size = sum(t.numel() * t.element_size() for v in T.values() for t in (v.values() if isinstance(v, dict) else [v]))
    if h is not None:
        size += h.numel() * h.element_size()
    return Snapshot(list(keys[:n]), n, kind, T, h, size)


def restore(model, snap: Snapshot, max_seq: int):
    """A fresh state of `model` holding the positions of `snap`."""
    st = model.new_state(1, max_seq)
    if hasattr(st, "ensure"):
        st.ensure(snap.n)
    T = snap.tensors
    for l, t in T["win"].items():
        st.win[l].copy_(t)
    for name in POOLS:
        for l, t in T.get(name, {}).items():
            getattr(st, name)[l][:, :t.size(1)].copy_(t)
    for l, t in T["comp_kv_state"].items():
        st.comp_kv_state[l].copy_(t)
    for l, t in T["comp_score_state"].items():
        st.comp_score_state[l].copy_(t)
    if "hasher" in T:
        st.hasher.cache[:, :snap.n].copy_(T["hasher"])
    if hasattr(st, "length"):
        st.length = snap.n
    return st


class PrefixStore:
    """Snapshots keyed by their token ids; LRU under a byte budget and an entry count."""

    def __init__(self, budget_bytes: int, max_entries: int = 48, min_free_gib: float = 3.0,
                 persist_dir: str | None = None, fingerprint: str = ""):
        self.budget, self.max_entries, self.min_free = int(budget_bytes), int(max_entries), float(min_free_gib)
        self.persist_dir, self.fingerprint = persist_dir or None, fingerprint
        self.disk = None
        self.disk_index: dict[str, dict] = {}
        self.entries: list[Snapshot] = []
        self.tick = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "PrefixStore | None":
        gib = float(os.environ.get("DS41_PFX_GIB", "4"))
        if gib <= 0:
            return None
        return cls(int(gib * (1 << 30)), int(os.environ.get("DS41_PFX_MAX", "48")), float(os.environ.get("DS41_PFX_MIN_FREE_GIB", "3")),
                   persist_dir=os.environ.get("DS41_PFX_DIR") or None)

    def __len__(self) -> int:
        return len(self.entries)

    def nbytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()
        if self.disk is not None:                                                  # same namespace rule on the disk (cache_salt change)
            self.disk.flush()
            for name in self.disk.entries():
                self.disk.remove(name)
        self.disk_index = {}

    def lookup(self, keys: Sequence) -> Snapshot | None:
        """Longest snapshot whose tokens are a prefix of `keys` and that leaves at least one token to prefill."""
        best = None
        with self._lock:
            for e in sorted(self.entries, key=lambda e: -e.n):
                if e.n < len(keys) and keys[:e.n] == e.keys:
                    self.tick += 1
                    e.tick = self.tick
                    best = e
                    break
        if self.disk_index:                                                        # a deeper entry on the disk (earlier run, evicted from RAM)
            try:
                snap = self._disk_lookup(keys, best.n if best is not None else 0)
            except Exception:                                                      # noqa: BLE001 keys that cannot be hashed (image spans)
                snap = None
            if snap is not None:
                self.put(snap, persist=False)
                return snap
        return best

    def best_lcp(self, keys: Sequence) -> tuple[int, int]:
        """(common prefix with the most recently used entry, its length): why a lookup missed. Diagnostics only."""
        with self._lock:
            if not self.entries:
                return 0, 0
            e = max(self.entries, key=lambda e: e.tick)
            return lcp(e.keys, keys), e.n

    # -- disk persistence (exllamav3.util.disk_store): "mark" / "end" / "reply" snapshots of >= PERSIST_MIN tokens are written by a
    #    background thread (tensors are host copies that nothing mutates), keyed by the token hash; a lookup that misses in RAM
    #    reads the longest matching entry back lazily. Namespace = fingerprint (config + pack + DS41_PFX_TAG).
    PERSIST_KINDS = ("mark", "end", "reply")

    def _disk_open(self):
        if self.disk is None and self.persist_dir:
            from exllamav3.util.disk_store import DiskStore
            self.disk = DiskStore(self.persist_dir, self.fingerprint or "default")
        return self.disk

    def _name(self, snap: Snapshot) -> str:
        return f"{snap.kind}_{snap.n}_{token_hash(snap.keys)[:20]}"

    def save(self, snap: Snapshot) -> bool:
        """Queue `snap` for the disk (once per token hash). True when it was queued."""
        d = self._disk_open()
        if d is None or snap.kind not in self.PERSIST_KINDS or snap.n < PERSIST_MIN:
            return False
        name = self._name(snap)
        if name in self.disk_index:
            d.touch(name)
            return False
        h = token_hash(snap.keys)
        idx = {"fp": self.fingerprint, "n": snap.n, "kind": snap.kind, "hash": h, "nbytes": snap.nbytes}
        d.put(name, {"fp": self.fingerprint, "n": snap.n, "kind": snap.kind, "keys": list(snap.keys), "tensors": snap.tensors,
                     "hidden": snap.hidden, "nbytes": snap.nbytes, "hash": h}, idx)
        self.disk_index[name] = idx
        # an older reply of the same conversation is a strict prefix of this one: superseded (the next turn extends the newest;
        # same rule as the RAM store). Marks and prompt ends stay: a render that drops earlier reasoning resumes from them.
        if snap.kind == "reply":
            for other, oi in list(self.disk_index.items()):
                if oi["kind"] == "reply" and oi["n"] < snap.n and other != name and \
                        token_hash(snap.keys[:oi["n"]]) == oi["hash"]:
                    d.remove(other)
                    self.disk_index.pop(other, None)
        return True

    def load_dir(self) -> int:
        """Index the committed disk entries of this namespace (a few KiB each; payloads are read lazily). Returns the count."""
        d = self._disk_open()
        if d is None:
            return 0
        self.disk_index = {}
        for name in d.entries():
            ix = d.read_idx(name)
            if ix and ix.get("fp") == self.fingerprint:
                self.disk_index[name] = ix
            else:
                d.remove(name)
        return len(self.disk_index)

    def _disk_lookup(self, keys: Sequence, above: int) -> Snapshot | None:
        d = self.disk
        if d is None or not self.disk_index:
            return None
        by_n: dict[int, list] = {}
        for name, ix in self.disk_index.items():
            if above < ix["n"] < len(keys):
                by_n.setdefault(ix["n"], []).append((name, ix))
        for n in sorted(by_n, reverse=True):
            h = token_hash(keys[:n])
            for name, ix in by_n[n]:
                if ix["hash"] != h:
                    continue
                try:
                    dd = d.read(name)
                    if dd["hash"] != h or len(dd["keys"]) != dd["n"]:
                        raise ValueError("torn")
                except Exception:                                                  # noqa: BLE001 unreadable: forget it
                    d.remove(name)
                    self.disk_index.pop(name, None)
                    continue
                return Snapshot(dd["keys"], dd["n"], dd["kind"], dd["tensors"], dd["hidden"], dd["nbytes"])
        return None

    def put(self, snap: Snapshot, persist: bool = True) -> None:
        if persist:
            try:
                self.save(snap)
            except Exception:                                                      # noqa: BLE001 the disk never fails a request
                pass
        if mem_available_gib() < self.min_free or snap.nbytes > self.budget:
            return
        with self._lock:
            self.tick += 1
            snap.tick = self.tick
            keep = []
            for e in self.entries:
                if e.n == snap.n and e.keys == snap.keys:
                    continue                                          # same position again: replaced
                if snap.kind == "reply" and e.kind == "reply" and e.n < snap.n and snap.keys[:e.n] == e.keys:
                    continue                                          # an older reply of the same conversation: superseded
                keep.append(e)
            keep.append(snap)
            keep.sort(key=lambda e: e.tick)
            while len(keep) > 1 and (len(keep) > self.max_entries or sum(e.nbytes for e in keep) > self.budget):
                keep.pop(0)                                           # least recently used
            self.entries = keep


def concat_rows(pre: Any, new: Any, seg_len: int, win: int) -> Any:
    """Target rows ending at the end of a segment of `seg_len` tokens: `new` are the rows the forward returned (all of the
    segment, or its last `win` when it ran chunked), `pre` the rows that ended where the segment started."""
    if new is None:
        return None
    if pre is not None and new.size(0) >= seg_len:
        new = torch.cat([pre.to(new.device), new], dim=0)
    return new[-win:]


def marks_for(ids: Sequence[int], cached: int, *, user_id: int | None, assistant_id: int | None, min_sys: int = 512) -> list[int]:
    """Split points of a prefill (positions p: the state after ids[:p] is snapshotted), all in (cached, len(ids))."""
    L = len(ids)
    out = set()
    if user_id is not None:
        try:
            p = list(ids).index(user_id)
        except ValueError:
            p = -1
        if p >= min_sys:
            out.add(p)                                                # the system / tools block, before the first user turn
    if assistant_id is not None:
        for p in range(L - 1, max(L - 64, -1), -1):                   # the generation prompt: right after the last <Assistant>
            if ids[p] == assistant_id:
                out.add(p + 1)
                break
    return sorted(m for m in out if cached < m < L)


def _forward(model, seg: Sequence[int], a: int, st, spans, on_chunk):
    t = torch.tensor([list(seg)], dtype=torch.int64)
    kw: dict[str, Any] = {}
    if spans:
        kw["spans"] = spans
    if hasattr(model, "prefill"):                                      # serving model: chunked, any start position
        if on_chunk is not None:
            kw["on_chunk"] = on_chunk
        return model.forward(t, a, st, **kw)
    if a == 0 or len(seg) == 1:
        return model.forward(t, a, st, **kw)
    mains, out = [], None                                              # reference model: a prefill resumes token at a time
    for i in range(len(seg)):
        out, m = model.forward(t[:, i:i + 1], a + i, st, **kw)
        mains.append(m)
    return out, (torch.cat(mains, dim=1) if mains and mains[0] is not None else None)


@dataclass
class Prefilled:
    logits: Any
    st: Any
    cached: int
    hidden: Any                              # [rows, D] target rows ending at the last prompt position (device), or None
    lcp: int = 0                             # common prefix with the last used snapshot when nothing was reused


def prefill(model, store: PrefixStore | None, ids: Sequence[int], keys: Sequence | None = None, *, max_seq: int,
            spans_fn: Callable[[int], Any] | None = None, user_id: int | None = None, assistant_id: int | None = None,
            win: int | None = None, reuse: bool = True) -> Prefilled:
    """Prefill `ids`, resuming from the longest snapshot of `store` that is a prefix of `keys` (default: ids), and leave
    snapshots behind (marks, chunk ends of a long prompt, the end of the prompt). Returns the logits of the last position,
    the state (positions 0..len(ids)-1) and the last `win` target rows."""
    keys = ids if keys is None else keys
    cfg = model.cfg
    win = win or cfg.sliding_window
    L = len(ids)
    snap = store.lookup(keys) if (store is not None and reuse) else None
    miss = (0, 0) if (snap is not None or store is None or not reuse) else store.best_lcp(keys)
    if snap is not None:
        st, cached, hid = restore(model, snap, max_seq), snap.n, snap.hidden
    else:
        st, cached, hid = model.new_state(1, max_seq), 0, None
    spans = spans_fn(cached) if spans_fn is not None else None
    marks = marks_for(ids, cached, user_id=user_id, assistant_id=assistant_id) if store is not None else []
    a, logits = cached, None
    for b in marks + [L]:
        pre, seg = hid, ids[a:b]
        cb = None
        if store is not None and b - a > 1:
            acc = {"h": pre, "pos": a}

            def cb(end, hc, acc=acc):                                  # one prefill chunk of a long segment is done
                rows = None if hc is None else hc[0]
                acc["h"] = concat_rows(acc["h"], rows, end - acc["pos"], win)
                acc["pos"] = end
                store.put(snapshot(cfg, st, end, keys, acc["h"], "grid"))
        logits, h = _forward(model, seg, a, st, spans, cb)
        hid = concat_rows(pre, None if h is None else h[0], len(seg), win)
        if store is not None:
            store.put(snapshot(cfg, st, b, keys, hid, "mark" if b < L else "end"))
        a = b
    return Prefilled(logits, st, cached, hid, miss[0])
