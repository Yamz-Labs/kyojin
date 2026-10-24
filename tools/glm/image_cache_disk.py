#!/usr/bin/env python3
"""Second-level picture cache on disk: the rows of the pictures the memory cache had to evict.

The memory cache (the shared `image_cache.py`) gives up the pictures its eviction policy weighs least, and with
`KYOJIN_IMAGE_CACHE_POLICY=tail` those are the *last* ones of a prompt, so the pictures a client re-sends a moment
later are exactly the ones it dropped — which today cost a vision pass plus the re-prefill of everything from
their position on. (Under the default `head` policy the first pictures go instead, and then this directory can
still save the vision pass for them, but not the prefix.) Writing a picture out **when it is evicted** and
reading it back later replaces that with a file read: the server also keeps the picture's identity (its pseudo
token ids) for the life of the process, so a restored picture reproduces the prompt token for token and the
paged KV prefix still matches.

Why only evicted pictures: that is the set that ever needs to come back *within* a process, which is the only
case where the re-prefill can be saved at all (after a restart the KV pool is empty). A session that fits in
`--image-cache-mb` therefore writes nothing, and a session that does not writes each evicted picture once.

Files hold the rows only (`--image-cache-format` decides their width) plus what the model needs to use them
again (`grid_thw`); no identity, no image file, no path. Everything here is fail-soft: a missing, corrupt,
unwritable or full directory means "no second-level cache", never a failed request.

    --image-cache-dir DIR          off by default
    --image-cache-dir-max-gb 4     cap on the directory: a conversation this server no longer remembers goes
                                   first (oldest file first), then the end of the one it does. A directory left
                                   over the cap by an earlier run is trimmed at start-up the same way.
"""
from __future__ import annotations

import os
import pathlib
import threading
import time
import torch

from exllamav3.tokenizer.mm_embedding import MMEmbedding, QuantizedMMRows

# The log's module holds the owner-only helpers this cache needs (servers put tools/glm on sys.path).
from serve_reqlog import restrict, secure_dir, secure_open


# A `.tmp` file is what a half-finished write leaves behind (a killed process, a full disk that was not
# reported). It is only removed once it is this old: while it is being written it is milliseconds old,
# and another server of the same directory cannot be ruled out by this module, so a fresh one is left in
# peace. Five minutes is far beyond one write of one picture (a few milliseconds for a few MiB) and far
# below the time it takes anyone to notice a crash and start again.
STALE_TMP_S = 300.0


def content_key(path: pathlib.Path) -> str:
    """The picture key out of a file name: `<env_fingerprint>.<key>.<mode>.pt`, or `<key>.<mode>.pt`.

    The parts after the fingerprint are fields of their own, so the key is the part before the mode and this
    cannot be "the first part": with a fingerprint in front that would return the fingerprint itself and every
    file would look like the same picture. Files written before the fingerprint existed have no prefix, and
    then the first part is the key."""
    parts = path.name.split(".")
    return parts[1] if len(parts) > 3 else parts[0]


def dtype_name(dtype: torch.dtype) -> str:
    """`torch.float32` -> "float32": what a stored payload records about the rows it holds."""
    return str(dtype).rsplit(".", 1)[-1]


def _rows_payload(emb: Any) -> dict | None:
    """The rows of one picture, in the form the memory cache holds them (plain tensor or packed holder)."""
    rows = emb.embeddings
    if isinstance(rows, QuantizedMMRows):
        return {"rows_q": rows.q, "rows_scale": rows.scale, "rows_dtype": dtype_name(rows.dtype)}
    if rows is not None and hasattr(rows, "element_size"):
        return {"rows": rows.detach().clone()}
    return None


def rows_nbytes(emb: Any) -> int:
    """Bytes the rows of one picture take, without building the payload."""
    rows = getattr(emb, "embeddings", None)
    if isinstance(rows, QuantizedMMRows):
        return int(rows.nbytes)
    if rows is not None and hasattr(rows, "element_size"):
        return int(rows.numel()) * int(rows.element_size())
    return 0


def as_live(live) -> list | None:
    """`live` as a list of conversations (each a set of content keys), newest first; None means "not told"."""
    if live is None:
        return None
    if isinstance(live, (set, frozenset)):
        return [set(live)]
    return [set(keys) for keys in live]


class ImageEmbeddingDisk:
    """Rows of evicted pictures, keyed by content hash, with a process-local identity map.

    `remember()` is called for every picture the tower encodes: it keeps the identity (pseudo ids, alias, token
    list, MRoPE metadata) that has to be put back for the prompt to be reproduced. `store()` writes the rows of
    pictures the memory cache just evicted; `fetch()` reads a picture back, restoring the identity when this
    process knows it (otherwise the picture comes back with fresh ids and only the vision pass is saved). When
    the directory is over its cap, `collect()` gives up whole conversations rather than single files — see there
    for why, and for the order it uses.
    """

    def __init__(self, directory: str, max_bytes: int = 0, start_token: int = 0, end_token: int = 0,
                 lock_s: float = 0.0, env_fingerprint: str = ""):
        self.path = pathlib.Path(directory).expanduser()
        # What the rows in this directory were computed with: the model and the settings that decide how a
        # picture is resized and encoded (`make_env_fingerprint` in the server). It names the files, so a
        # restart with another model or other pixel settings cannot reach rows computed with the old ones,
        # and a file that is somehow in the way is refused rather than injected (`fetch`). Empty means "not
        # told", which is what the tests that only exercise the store/collect logic want.
        self.env_fingerprint = str(env_fingerprint or "")
        self.max_bytes = max(0, int(max_bytes))
        self.start_token, self.end_token = int(start_token), int(end_token)
        # A conversation keeps the room its cache took for this long (0 = no tenure). The clock starts when the
        # file is written and is *not* refreshed by later requests: refreshing it would let whichever
        # conversation grabbed the cache first keep it for ever. Tenure is what stops a cyclic workload
        # (four conversations taking turns) from evicting each one just before it is served again.
        self.lock_s = max(0.0, float(lock_s))
        self.enabled = False
        self.error: str | None = None
        self._identity: dict[str, dict] = {}
        self._position: dict[str, int] = {}
        self._live: list | None = None               # conversations the memory cache last said it refers to
        self._bytes = 0
        self._files = 0
        self._locked: dict[str, float] = {}          # content key -> time its tenure ends
        # written_mb / skipped_mb are the numbers to watch when what matters is how much the server writes:
        # a batch that would not fit is not written at all, so bytes are never written just to be deleted.
        self.stats = {"loaded": 0, "stored": 0, "dropped": 0, "skipped": 0,
                      "written_bytes": 0, "skipped_bytes": 0, "errors": 0, "stale_tmp": 0}
        try:
            # Owner-only: these files are rows derived from the pictures clients sent. See
            # tools/strix_halo/serve_reqlog.py for what the modes are and why they are set explicitly.
            secure_dir(self.path)
            self.enabled = True
            self.cleanup_stale_tmp()
            for entry in self.path.glob("*.pt"):
                try:
                    self._bytes += entry.stat().st_size
                    self._files += 1
                    # A file written before the owner-only rule (or by a process with a looser umask) is
                    # tightened here: files are only ever written once, so nothing else would revisit it.
                    restrict(entry, 0o600)
                except OSError:
                    pass
            # A directory left over the cap by an earlier run (or a smaller cap now) is trimmed here, oldest
            # first: at start-up no conversation is remembered yet, so nothing is preferred over anything.
            self.collect(live=set())
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    # -- reporting -------------------------------------------------------------------------------------------

    def describe(self) -> str:
        if not self.enabled:
            return f"picture cache on disk off ({self.error})"
        cap = f"{self.max_bytes / 2 ** 30:.1f} GiB cap" if self.max_bytes else "no cap"
        tenure = f", {self.lock_s / 60:.0f} min tenure" if self.lock_s else ""
        return f"picture cache on disk {self.path} ({self._files} files, {self._bytes / 2 ** 20:.1f} MiB, {cap}{tenure})"

    def report(self) -> dict:
        return {"files": self._files, "mb": round(self._bytes / 2 ** 20, 1),
                "env": self.env_fingerprint or None,
                "written_mb": round(self.stats["written_bytes"] / 2 ** 20, 3),
                "skipped_mb": round(self.stats["skipped_bytes"] / 2 ** 20, 3),
                "cap_mb": round(self.max_bytes / 2 ** 20, 1),
                "locked": sum(1 for key, until in self._locked.items()
                              if until > time.time() and key in self._present()),
                **self.stats}

    def _present(self) -> set:
        try:
            return {content_key(entry) for entry in self.path.glob("*.pt")}
        except OSError:
            return set()

    def _locked_now(self, key: str) -> bool:
        return self.lock_s > 0 and self._locked.get(key, 0.0) > time.time()

    # -- identity --------------------------------------------------------------------------------------------

    def remember(self, key: str, emb: Any) -> None:
        """Keep what a later restore needs to reproduce the prompt: the ids and the text that carries them."""
        if not self.enabled:
            return
        self._identity[key] = {
            "first_index": int(emb.first_index), "last_index": int(emb.last_index),
            "text_alias": emb.text_alias, "token_string": emb.token_string,
            "token_list": list(emb.token_list) if emb.token_list is not None else None,
            "grid_thw": emb.grid_thw, "mrope_merge_size": emb.mrope_merge_size, "metadata": emb.metadata,
        }

    def position(self, key: str, position: int | None) -> None:
        """Where the picture sat in the prompt that last used it: what the eviction order is built from."""
        if position is not None:
            self._position[key] = int(position)

    # -- files -----------------------------------------------------------------------------------------------

    def _file(self, key: str, mode: str) -> pathlib.Path:
        prefix = f"{self.env_fingerprint}." if self.env_fingerprint else ""
        return self.path / f"{prefix}{key}.{mode}.pt"

    def fetch(self, key: str, mode: str):
        """The picture back, or None. Rows from the file; identity from this process when it is known."""
        if not self.enabled:
            return None
        path = self._file(key, mode)
        try:
            blob = torch.load(path, weights_only=True)
        except FileNotFoundError:
            return None
        except Exception:                                   # partial or foreign file: drop it and re-encode
            self.stats["errors"] += 1
            self._unlink(path)
            return None
        if self.env_fingerprint and blob.get("env_fingerprint") not in (None, self.env_fingerprint):
            # reachable but computed with something else: drop it and let the caller encode the picture
            self.stats["errors"] += 1
            self._unlink(path)
            return None
        try:
            rows = blob["rows"] if "rows" in blob else QuantizedMMRows(
                blob["rows_q"], blob["rows_scale"], getattr(torch, blob["rows_dtype"], torch.float32))
            rows_count = int(rows.shape[-2])
            token_string = torch.tensor([[self.start_token] + [-1] * rows_count + [self.end_token]], dtype=torch.long)
            emb = MMEmbedding(embeddings=rows, token_string=token_string, text_alias=None,
                              grid_thw=blob.get("grid_thw"), mrope_merge_size=blob.get("mrope_merge_size"))
            emb.metadata = dict(blob.get("metadata") or {})
        except Exception:                                   # unreadable content: same treatment
            self.stats["errors"] += 1
            self._unlink(path)
            return None
        known = self._identity.get(key)
        if known is not None:                               # put the identity back: the prompt is reproduced
            emb.first_index, emb.last_index = known["first_index"], known["last_index"]
            emb.text_alias, emb.token_string = known["text_alias"], known["token_string"]
            emb.token_list = known["token_list"]
            emb.grid_thw, emb.mrope_merge_size = known["grid_thw"], known["mrope_merge_size"]
            emb.metadata = known["metadata"]
        try:
            os.utime(path, None)                            # used now: the last thing the collector takes
        except OSError:
            pass
        restrict(path, 0o600)                               # belt and braces for a file from before the rule
        self.stats["loaded"] += 1
        return emb

    def cleanup_stale_tmp(self, grace_s: float = STALE_TMP_S) -> int:
        """Remove the `.tmp` files a write left behind, older than `grace_s`.

        Every picture is written to its own `<name>.<pid>.<thread>.tmp` and moved onto its final name with
        `os.replace`, so a `.pt` is always complete and a leftover `.tmp` can only come from a process that
        died mid-write (`kill -9`, the OOM killer, a power cut). Nothing else looks at those files - not the cap, not the
        set of pictures present, not the eviction order - so without this they would stay for ever, taking
        room the cap does not know about.

        A file younger than the grace period is left alone: this module cannot know whether another process
        is writing it (the servers are one per GPU, but a stale one that was not stopped is a real way to
        get two). A write of one picture takes milliseconds, so the grace period never covers a live one,
        and a crash leaves a file that is already minutes old by the time anyone restarts.

        Returns how many were removed; they are counted in `report()` as `stale_tmp`.
        """
        try:
            entries = list(self.path.glob("*.tmp"))
        except OSError:
            return 0
        now, removed = time.time(), 0
        for entry in entries:
            try:
                if now - entry.stat().st_mtime < grace_s:
                    continue                              # young enough to be somebody's live write
                entry.unlink()
                removed += 1
            except OSError:
                pass                                      # a file that vanished or cannot be read: not ours
        self.stats["stale_tmp"] = self.stats.get("stale_tmp", 0) + removed
        return removed

    def store(self, evicted: list, mode: str, live: list | set | None = None,
              protect: set | None = None) -> None:
        """Write the pictures that just left the memory cache. `evicted` is [(key, embedding, position), ...].

        `live` is `ImageEmbeddingCache.live_orders` (remembered conversations, newest first) and `protect` is the
        conversation this request is serving, so that concurrent requests cannot release each other's cache.

        The batch is judged **before** anything is written: room is made first (whole conversations, per
        `collect`), and if the batch still does not fit it is not written at all. A conversation that is cached
        only in part has no reuse anyway — it would hold room a complete one needs — so there is no point writing
        it, and no point writing bytes that would have to be deleted again a moment later.
        """
        if not self.enabled or not evicted:
            return
        if live is not None:
            self._live = as_live(live)      # the conversations the cache still refers to (see collect)
        pending = []
        for key, emb, position in evicted:
            self.position(key, position)
            if getattr(emb, "deepstack_embeddings", None) is not None:
                continue                                    # not stored: only the merger rows are (GLM has none)
            if self._file(key, mode).exists():
                continue                                    # same picture evicted again: already on disk
            pending.append((key, emb))
        if not pending:
            self.collect(live, protect)
            return
        # 4 KiB covers what the container adds around the rows (measured ~3 KiB for a 12 MiB payload)
        estimate = sum(rows_nbytes(emb) + 4096 for _, emb in pending)
        if self.max_bytes:
            # Make room *before* deciding: give up what is free to give up (unclaimed files first, then whole
            # conversations whose tenure is over) so that the batch can be written, not written and deleted.
            self.collect(live, protect, reserve=estimate)
            if self._bytes + estimate > self.max_bytes:
                self.stats["skipped"] += len(pending)       # does not fit: nothing is written for this batch
                self.stats["skipped_bytes"] += estimate
                self.collect(live, protect)
                return
        for key, emb in pending:
            payload = _rows_payload(emb)
            if payload is None:
                continue
            payload |= {"grid_thw": getattr(emb, "grid_thw", None),
                        "mrope_merge_size": getattr(emb, "mrope_merge_size", None),
                        "metadata": dict(getattr(emb, "metadata", None) or {})}
            if self.env_fingerprint:
                # a second copy of the identity, in the file itself: the name already keeps a different
                # environment out, this makes a file that got in some other way visible instead of silent
                payload["env_fingerprint"] = self.env_fingerprint
            path = self._file(key, mode)
            # one temp file per writer: two requests storing the same picture must not interleave into one file
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                # torch.save() would create the temp file with the process umask (0644 on a 0022 host) and
                # os.replace() moves that mode onto the final name, so the file is made owner-only here.
                fd = secure_open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
                with os.fdopen(fd, "wb") as fh:
                    torch.save(payload, fh)
                restrict(tmp, 0o600)
                os.replace(tmp, path)
                size = path.stat().st_size
                self._bytes += size
                self._files += 1
                self.stats["stored"] += 1
                self.stats["written_bytes"] += size
                if self.lock_s:
                    self._locked[key] = time.time() + self.lock_s
            except Exception:                               # disk full, permissions, ...: keep serving
                self.stats["errors"] += 1
                try:
                    tmp.unlink()
                except OSError:
                    pass
        self.collect(live, protect)

    def _unlink(self, path: pathlib.Path) -> None:
        self._locked.pop(content_key(path), None)
        try:
            size = path.stat().st_size
            path.unlink()
            self._bytes = max(0, self._bytes - size)
            self._files = max(0, self._files - 1)
        except OSError:
            pass

    def collect(self, live: list | set | None = None, protect: set | None = None, reserve: int = 0) -> int:
        """Enforce the cap. The unit of eviction is a *conversation*, not a file.

        A conversation's pictures are only worth keeping together: the engine resumes at a checkpoint only when a
        whole prompt is reproduced, so a conversation that loses one file has already lost its reuse — what is
        left of it saves a vision pass per picture but nothing else, and it occupies the room a complete
        conversation would need. So, in order:

        1. pictures no remembered conversation claims, oldest file first, one at a time (the leftovers of a
           conversation the user has moved on from and the server has since forgotten);
        2. a conversation the server remembers but that is not the one being served, released **whole**, oldest
           conversation first — with `--sessions N` the caches of N conversations coexist, and this is what keeps
           as many of them complete as the cap allows;
        3. the conversation being served, last of all, and only in part if its own files alone exceed the cap:
           the end of its prompt goes first, because the head is what a resume point needs.

        `live` is the remembered conversations, newest first (`ImageEmbeddingCache.live_orders`); `protect` is the
        conversation the calling request is serving (`mine` from `get_many`), which must never be released whole
        just because some other request happened to be served more recently. Without `protect` the newest
        remembered conversation plays that part.
        """
        if not self.enabled or not self.max_bytes or self._bytes + reserve <= self.max_bytes:
            return 0
        live = self._live if live is None else as_live(live)
        protected = set(protect) if protect is not None else (set(live[0]) if live else set())
        try:
            files = list(self.path.glob("*.pt"))
        except OSError:
            return 0

        def age(entry: pathlib.Path) -> float:
            try:
                return entry.stat().st_mtime
            except OSError:
                return 0.0

        def within(entry: pathlib.Path) -> float:
            position = self._position.get(content_key(entry))
            return age(entry) if position is None else -position

        dropped = 0
        if live is None:                              # nobody said which conversations are live: position only
            for entry in sorted(files, key=lambda e: (0, age(e)) if self._position.get(content_key(e)) is None
                                else (1, -self._position[content_key(e)])):
                if self._bytes + reserve <= self.max_bytes:
                    break
                if self._locked_now(content_key(entry)):
                    continue
                self._unlink(entry)
                dropped += 1
            dropped += self._break_tenure(files, reserve)
            self.stats["dropped"] += dropped
            return dropped

        def group_of(key: str) -> int:
            """0 = the conversation being served, 1..len(live) = remembered ones (older = larger),
            len(live)+1 = claimed by none of them."""
            if key in protected:
                return 0
            for rank, keys in enumerate(live):
                if key in keys:
                    return rank + 1
            return len(live) + 1

        unclaimed = len(live) + 1
        groups: dict[int, list] = {}
        for entry in files:
            groups.setdefault(group_of(content_key(entry)), []).append(entry)

        for group in sorted(groups, reverse=True):    # least wanted first
            if self._bytes + reserve <= self.max_bytes:
                break
            entries = groups[group]
            if group == unclaimed:                    # nobody claims these: the oldest goes first
                ordered, whole = sorted(entries, key=age), False
            elif group == 0:                          # the one being served: only as a last resort, tail first
                ordered, whole = sorted(entries, key=within), False
            else:                                     # a remembered conversation: it goes as a unit
                ordered, whole = sorted(entries, key=age), True
            if whole and any(self._locked_now(content_key(e)) for e in entries):
                continue                              # still within its tenure: leave it alone for now
            for entry in ordered:
                if not whole and self._bytes + reserve <= self.max_bytes:
                    break
                if not whole and self._locked_now(content_key(entry)):
                    continue
                self._unlink(entry)
                dropped += 1
        # Tenure is only broken when the directory is already over the cap (a cap lowered under it, say): it is
        # not broken to make room for a new batch, which is exactly what it exists to prevent. That batch is
        # simply not written and waits for the tenure to run out.
        dropped += self._break_tenure(files, reserve if self._bytes > self.max_bytes else 0)
        self.stats["dropped"] += dropped
        return dropped

    def _break_tenure(self, files: list, reserve: int = 0) -> int:
        """Last resort: the cap has to hold, so the tenures closest to expiring go first."""
        dropped = 0
        for entry in sorted(files, key=lambda e: self._locked.get(content_key(e), 0.0)):
            if self._bytes + reserve <= self.max_bytes:
                break
            if not entry.exists():
                continue
            self._unlink(entry)
            dropped += 1
        return dropped
