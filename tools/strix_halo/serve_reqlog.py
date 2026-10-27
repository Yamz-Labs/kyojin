#!/usr/bin/env python3
"""Per-request state log for the Kyojin servers: timings and counters only, never text.

A request that is refused, queued or prefilling produces **no client-visible output**, so a report of
"the server went silent" cannot be lined up with what the server was doing. This writes one line per
lifecycle event, plus a heartbeat while a job is queued or running, so the two timelines can be
compared afterwards. The heartbeat and the per-request summary carry the request's **output
accounting** (`out_events`, `out_chars`, `last_out_s`), which is what separates "the server was
writing to a client that did not show it" from "the server had nothing to write".

**Off unless asked for.** With KYOJIN_REQLOG_DIR unset (or 0/off/none/no/false/empty) nothing is opened and
every call is a no-op, so a default deployment behaves exactly as before. The server prints one line at
start-up either way, so the switch stays discoverable when a request does go silent.

**One file per run.** The switch names a directory; a run writes `requests-<run>.log` in it (the default
directory is `~/.cache/kyojin`) and every line carries that run id, so the lines of one run can be read
without filtering the other runs out. A path that ends in a file name is read as that file's directory
(`/var/log/kyojin/requests.log` means `/var/log/kyojin`); that file itself is never written to - every run
writes `requests-<run>.log` in the directory. At start-up the log files of the oldest runs are removed,
keeping the newest `KYOJIN_REQLOG_KEEP` (20 by default) of them: the directory is checked once, at
start-up, and nothing is checked while a run is going on. A long-running server writes continuously
(one `alive` line per heartbeat interval per request plus the events), so a deployment that keeps a log
open for weeks should turn it off or shorten the retention rather than expect the file to stop growing.

**Owner-only files.** The log, and the picture cache the GLM server keeps on disk, hold material derived
from what clients sent, so the files are created `0600` and an older one is tightened on the next start
(`secure_open` / `restrict` below); every directory this server creates is `0700`. Without that the mode
would follow the process umask, `0022` on most systems, and the log would be `0644`.

Two directories are deliberately not this server's to set: one that already exists is **not** touched at
all, and the shared folder the three servers keep state in — `KYOJIN_HEALTH_DIR`, `~/.cache/kyojin` by
default (`shared_cache_dir()`) — is created with the process umask like any other shared directory, the
folders above it included: more than one process writes there, and whichever makes it first must not decide
how it looks. A file inside is still unreadable to others at `0600`.

    KYOJIN_REQLOG_DIR=/var/log/kyojin           # a directory, or 1/on/yes for the default below
    KYOJIN_REQLOG_HEARTBEAT_S=10                # seconds between 'alive' lines while a job runs

Privacy: nothing derived from the request is written except numbers — token counts, message/image
counts, durations, byte counts, HTTP statuses, an opaque request id, a server tag and event names.
Prompt text, completion text, tool arguments and image paths from the body are never logged. Of the
server's own error text, only the exception type is logged plus a truncated message; JSON parse errors
are reduced to line/column because their message can quote the request body.

Reading it back: `grep req_refused`, `grep img_cache`, `grep -E 'alive|job_first_token'`, or follow one
request with `grep <req id>`. Every line starts with a local ISO timestamp with milliseconds, then the
event name, then `key=value` fields; every line of one request carries the same `req=` id.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
import pathlib
import stat
import sys
import threading
import time
import uuid
from typing import Any

# The switch names a directory; every run writes its own file in it, `<BASE>-<run>.log`, so that the lines of
# one run can be read without filtering the lines of every other run out of one growing file.
DEFAULT_DIR = "~/.cache/kyojin"
BASE_NAME = "requests"            # fixed: the directory says where, this says what it is
RUN_STAMP = "%Y%m%dT%H%M%S"
DEFAULT_KEEP = 20                 # runs kept on disk; older files are removed when a server starts
FALSE = {"0", "off", "none", "no", "false", ""}
TRUE = {"1", "on", "yes", "true"}


def shared_cache_dir() -> pathlib.Path:
    """The folder the servers share for the state they keep between starts.

    `KYOJIN_HEALTH_DIR` names it; `~/.cache/kyojin` is the default. More than one module and more than one
    server write there, so it is created with the process umask like any other shared directory (see
    `secure_dir`) and never tightened."""
    return pathlib.Path(os.environ.get("KYOJIN_HEALTH_DIR") or "~/.cache/kyojin").expanduser()


def secure_dir(path: "pathlib.Path", mode: int = 0o700) -> None:
    """Create `path`, and the directories above it that are missing, owner-only. An existing one is left
    alone, whatever its mode.

    `mode` applies to every directory this call creates, applied with the umask cleared so the requested
    mode is the mode that lands. The directories in between are created one level at a time, because
    `mkdir(parents=True, mode=...)` does not pass the mode to those (its C implementation calls
    `mkdir(path)` without one), which would leave them at `0777` with the umask cleared.

    `shared_cache_dir()` is the exception, and it is recognised here rather than passed in by the callers:
    that folder is shared, other processes write there, and whichever of them creates it first must not
    decide how it looks, so it is created with the process umask, and so are the folders above it when they
    are missing. Everything else this server creates is its own and gets `mode`: a directory *below* the
    shared folder (`image-embeddings`), and any level of a path that merely passes through it. Files are
    `0600` either way."""
    path = pathlib.Path(path)
    if path.exists() and not path.is_dir():
        # A file where a directory has to go is a broken setup, and the callers report it as such: the
        # disk cache answers "off" with the reason, the log switches itself off with a line on stderr.
        raise NotADirectoryError(f"{path} exists and is not a directory")
    shared = path == shared_cache_dir()
    # every level that is not there yet, deepest first, down to the first one that exists; decided before
    # anything is created, because a directory this call makes must not change the answer
    missing: list[pathlib.Path] = []
    probe = path
    while not probe.exists():
        missing.append(probe)
        if probe.parent == probe:                            # reached the filesystem root
            break
        probe = probe.parent
    # the shared folder and the levels above it are created the umask way; a directory that merely passes
    # *through* the shared folder, `<shared>/image-embeddings`, keeps `mode`
    previous = os.umask(0)
    try:
        for entry in reversed(missing):                      # shallowest first, so each parent exists
            if shared and (entry == path or entry in path.parents):
                os.umask(previous)                           # a shared folder: the umask decides, as for any
                try:
                    entry.mkdir(exist_ok=True)
                except FileExistsError:                      # a race: somebody created it just now
                    pass
                os.umask(0)
                continue
            try:
                entry.mkdir(mode=mode)
            except FileExistsError:                          # a race: somebody else created it just now
                pass
    finally:
        os.umask(previous)


def secure_open(path: "pathlib.Path", flags: int, mode: int = 0o600) -> int:
    """`os.open` for a file only its owner may read, whatever the process umask is.

    Same trick as `secure_dir`: umask 0 while creating, so the mode lands as asked, and the group and
    other bits are dropped even if the caller asks for more. Opening an existing file does not change its
    mode, so a file that predates the rule is fixed by `restrict` instead."""
    previous = os.umask(0)
    try:
        return os.open(path, flags, mode & ~0o077)
    finally:
        os.umask(previous)


def restrict(path: "pathlib.Path", mode: int) -> None:
    """Bring an existing file to `mode`, owner-only, if it is not already."""
    try:
        if stat.S_IMODE(os.stat(path).st_mode) & 0o077:
            os.chmod(path, mode)
    except OSError:
        pass


def sanitize(exc: BaseException) -> str:
    """Exception type and a short message, safe to log (no request-body snippets)."""
    if isinstance(exc, json.JSONDecodeError):
        return f"JSONDecodeError(line={exc.lineno} col={exc.colno})"
    msg = " ".join(str(exc).split())[:160]
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        # Whole numbers read as integers (cache_pct=40, budget_mb=1024), the rest keep milliseconds.
        return str(int(value)) if value.is_integer() else f"{value:.3f}"
    text = str(value)
    return f'"{text}"' if (" " in text or "=" in text) else text


class StreamStats:
    """What one streaming request actually wrote to its client: bytes, data events, silence.

    A client that reports "the server went silent" is believed or not depending on whether the bytes
    were on the wire, and the heartbeat alone cannot tell the two apart. Every write of a streaming
    reply goes through this, so at any moment `out_events` / `out_chars` say what the client has been
    sent and `last_out_s` how long ago; a reply whose engine is decoding into a held-back tool call
    shows up as a flat line, and one that is streaming shows a rate.
    """

    def __init__(self) -> None:
        self.bytes = 0
        self.events = 0
        self.comments = 0        # SSE keep-alive comments (not data events)
        self.comment_bytes = 0
        self.errors = 0
        self.last_error: str | None = None
        self.t0 = time.perf_counter()
        self.last_data = self.t0
        self.last_comment = self.t0

    def note(self, n: int) -> None:
        self.bytes += n
        self.events += 1
        self.last_data = time.perf_counter()

    def note_comment(self, n: int) -> None:
        self.comments += 1
        self.comment_bytes += n
        self.last_comment = time.perf_counter()

    def fail(self, exc: BaseException) -> None:
        self.errors += 1
        self.last_error = sanitize(exc)

    def since_data_s(self) -> float:
        """Seconds since the last **data event**: comments do not count as content.

        This is the number that matters when a client reports silence — a reply that is decoding into a
        held-back tool call sends comments (which keep the socket alive) and no events at all, and this
        says how long that has been going on.
        """
        return time.perf_counter() - self.last_data

    def lines(self) -> list[tuple[str, dict[str, Any]]]:
        """The summary event and its fields, in the order they read best."""
        return [("sse_summary", {
            "out_events": self.events, "out_chars": self.bytes, "last_out_s": round(self.since_data_s(), 2),
            "comments": self.comments or None, "comment_chars": self.comment_bytes or None,
            "last_comment_s": round(time.perf_counter() - self.last_comment, 2) if self.comments else None,
            "errors": self.errors or None, "error": self.last_error})]


class StreamWriter:
    """A StreamResponse that records what it wrote (see `StreamStats`). One per streaming reply."""

    def __init__(self, response: Any, stats: StreamStats | None = None) -> None:
        self.response = response
        self.stats = stats if stats is not None else StreamStats()

    async def write(self, data: bytes) -> None:
        try:
            await self.response.write(data)
        except BaseException as exc:                              # noqa: BLE001  counted, then re-raised
            self.stats.fail(exc)
            raise
        self.stats.note(len(data))

    async def write_eof(self) -> None:
        await self.response.write_eof()


class Lifecycle:
    """The request lifecycle a server logs, in one place for all of them.

    Every server used to carry its own copy of this: the request table, the context variable that names
    the request, the 'alive' heartbeat, and the route wrapper that stamps one id on every line. The
    copies drifted (one server forgot an argument, another measured a different span, a third spelled an
    event differently), so the machinery lives here and a server only says **when** something happened.

    A server keeps its own event *call sites* — where in its worker and its routes the phases begin is
    the one thing that genuinely differs — and calls the method for the event it has:

        log = Lifecycle(ReqLog.from_env("glm"), ping=ServePing.from_env(reqlog),
                        counters=lambda: {"processing": processing, "running": running})
        app.router.add_post("/v1/chat/completions", log.wrapped(completions))
        log.queued(rid, "/v1/chat/completions", prompt_tokens=n, images=len(imgs), stream=True)
        ...
        await log.heartbeat_forever()      # started on app startup, cancelled on cleanup

    **An event or field a server does not have is simply not passed**, which is what keeps one
    implementation usable by three servers: `ReqLog.log` skips `None` fields and accepts any event name,
    so a missing counter or an event a server never reaches leaves no line and no branch anywhere. What a
    server does report is up to its call sites, never to a per-server flag in here. The one event with
    server-specific fields is the raw `log()` escape hatch (`img_cache` on GLM, `serve_start`), which
    carries whatever that server measured.

    Counters are pulled per heartbeat through `counters()`, so each server reports the ones it has
    (GLM tracks requests in flight and running separately; Qwen and MiMo have one number).
    """

    def __init__(self, reqlog: "ReqLog", ping: Any = None, counters: Any = None,
                 server: str | None = None) -> None:
        self.reqlog = reqlog
        self.ping = ping
        self.counters = counters
        self.server = server or getattr(reqlog, "server", None)
        self.inflight: dict[str, dict[str, Any]] = {}
        self.writers: dict[str, StreamWriter] = {}
        self.current: contextvars.ContextVar[str | None] = contextvars.ContextVar("reqid", default=None)
        self._counters: dict[str, Any] = {}

    # ---- request identity and per-request state ---------------------------------------------------

    def reqid(self) -> str | None:
        """The log id of the request being handled (None outside a wrapped route)."""
        return self.current.get()

    def track(self, rid: str | None, phase: str, **fields: Any) -> None:
        """Record the phase a request is in, and anything the heartbeat should repeat."""
        if rid is None:
            return
        st = self.inflight.setdefault(rid, {"t0": time.perf_counter()})
        st["phase"] = phase
        st.update(fields)

    def writer(self, rid: str | None, response: Any, path: str | None = None) -> StreamWriter:
        """The request's stream writer: what it writes is counted, and the heartbeat reports it."""
        writer = StreamWriter(response)
        writer.path = path                     # so the summary line can name the route, as it always has
        if rid is not None:
            self.writers[rid] = writer
        return writer

    def log(self, event: str, **fields: Any) -> None:
        """A line only this server writes (a start-up record, a cache report): pass it through."""
        self.reqlog.log(event, **fields)

    # ---- request level ---------------------------------------------------------------------------

    def request_started(self, rid: str, request: Any) -> float:
        """Open a request: id, arrival time, and the `req_start` line. Returns the start time."""
        self.current.set(rid)
        started = time.perf_counter()
        self.track(rid, "parse")
        self.reqlog.log("req_start", req=rid, path=request.path, method=request.method,
                        bytes=request.content_length)
        return started

    def refused(self, rid: str, path: str, exc: Any, started: float) -> None:
        self.reqlog.log("req_refused", req=rid, path=path, status=exc.status,
                        reason=refusal_text(exc), elapsed_s=round(time.perf_counter() - started, 3))

    def handler_left(self, rid: str, path: str) -> None:
        self.reqlog.log("client_left", req=rid, path=path, stage="handler")

    def request_error(self, rid: str, path: str, exc: BaseException, started: float) -> None:
        self.reqlog.log("req_error", req=rid, path=path, error=sanitize(exc),
                        elapsed_s=round(time.perf_counter() - started, 3))

    def request_done(self, rid: str, path: str, response: Any, started: float) -> None:
        self.reqlog.log("req_done", req=rid, path=path, status=response.status,
                        elapsed_s=round(time.perf_counter() - started, 3))

    def client_left(self, path: str, stage: str = "waiting", rid: str | None = None) -> None:
        """The client hung up first: the server's side of a cancelled turn, on record."""
        self.reqlog.log("client_left", req=rid if rid is not None else self.reqid(), path=path, stage=stage)

    def wrapped(self, handler: Any) -> Any:
        """Wrap a route so its lifecycle lands in the log: one id per request, `grep <id>` follows it."""
        @functools.wraps(handler)
        async def wrapper(request: Any) -> Any:
            rid = f"r{uuid.uuid4().hex[:12]}"
            token = self.current.set(rid)
            started = time.perf_counter()
            self.track(rid, "parse")
            self.reqlog.log("req_start", req=rid, path=request.path, method=request.method,
                            bytes=request.content_length)
            try:
                response = await handler(request)
            except asyncio.CancelledError:
                self.handler_left(rid, request.path)
                raise
            except Exception as exc:                                  # noqa: BLE001  classified here
                if hasattr(exc, "status"):                            # aiohttp's HTTPException
                    self.refused(rid, request.path, exc, started)
                else:
                    self.request_error(rid, request.path, exc, started)
                raise
            else:
                self.request_done(rid, request.path, response, started)
                return response
            finally:
                self.finish(rid)
                self.current.reset(token)
        return wrapper

    def finish(self, rid: str) -> None:
        """Close a request: its summary line (what the client was sent) and its bookkeeping."""
        writer = self.writers.pop(rid, None)
        if writer is not None:
            for event, fields in writer.stats.lines():
                self.reqlog.log(event, req=rid, path=getattr(writer, "path", None), **fields)
        self.inflight.pop(rid, None)

    # ---- job level (the worker side) -------------------------------------------------------------

    def queued(self, rid: str | None, path: str, **fields: Any) -> None:
        """The request is in the worker queue: what it is, and how many are ahead of it."""
        self.track(rid, "queued", prompt_tokens=fields.get("prompt_tokens"))
        self.reqlog.log("req_queued", req=rid, path=path, **fields)

    def job_started(self, rid: str | None, wait: float, **fields: Any) -> None:
        """The worker picked the job up: `wait` is the queue time, the rest is what this server knows."""
        self.track(rid, "prefill", prompt_tokens=fields.get("prompt_tokens"))
        self.reqlog.log("job_start", req=rid, wait_s=round(wait, 3), **fields)

    def job_skipped(self, rid: str | None, queued_s: float) -> None:
        """A job cancelled while it was still queued: no prefill, no decode, one line."""
        self.reqlog.log("job_skipped", req=rid, queued_s=round(queued_s, 3))

    def first_token(self, rid: str | None, started: float, **fields: Any) -> None:
        """The model produced its first token, and the request moves to decode."""
        self.track(rid, "decode", prompt_tokens=fields.get("prompt_tokens"))
        self.reqlog.log("job_first_token", req=rid, elapsed_s=round(time.perf_counter() - started, 3))

    def job_done(self, rid: str | None, stats: Any, started: float, prompt_tokens: int = 0) -> None:
        """A finished job: what the prompt cache did (cache_n vs prompt tokens) and how long each phase
        took. `prompt_tokens` is the fallback for an engine (or a fake one in tests) that reports none."""
        st = stats or {}
        prompt = int(st.get("prompt_tokens") or prompt_tokens or 0)
        cache_n = int(st.get("cached_tokens") or 0)
        self.reqlog.log("job_done", req=rid, elapsed_s=round(time.perf_counter() - started, 3),
                        prompt_tokens=prompt, cache_n=cache_n,
                        cache_pct=round(100.0 * cache_n / prompt, 1) if prompt else 0.0,
                        new_tokens=int(st.get("new_tokens") or 0),
                        prefill_s=round(float(st.get("time_prefill") or 0.0), 3),
                        decode_s=round(float(st.get("time_generate") or 0.0), 3))

    def job_error(self, rid: str | None, exc: BaseException, started: float | None = None) -> None:
        self.reqlog.log("job_error", req=rid, error=sanitize(exc),
                        elapsed_s=round(time.perf_counter() - started, 3) if started else None)

    def stream_error(self, rid: str | None, exc: BaseException) -> None:
        self.reqlog.log("stream_error", req=rid, error=sanitize(exc))
        print(f"qserve: request failed mid-stream, error event sent: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr, flush=True)

    def request_failed(self, exc: BaseException) -> None:
        """A non-streaming request ended in HTTP 500 (the request log is off by default: stderr is the only trace)."""
        print(f"qserve: request failed, HTTP 500: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr, flush=True)

    # ---- stream level ----------------------------------------------------------------------------

    def stream_started(self, rid: str | None, path: str, response: Any, prompt_tokens: int = 0) -> StreamWriter:
        """The SSE headers are out: register the writer the heartbeat reads and log the headers."""
        writer = self.writer(rid, response, path)
        self.track(rid, "stream", prompt_tokens=prompt_tokens)
        self.reqlog.log("sse_headers", req=rid, path=path)
        return writer

    def first_content(self, rid: str | None, since: float) -> None:
        """The first frame with text in it: the gap the client experiences as silence, in seconds."""
        self.reqlog.log("sse_first_content", req=rid,
                        first_s=round(time.perf_counter() - since, 3))

    def stream_done(self, rid: str | None, out_tokens: int = 0) -> None:
        self.reqlog.log("sse_done", req=rid, out_tokens=out_tokens)

    # ---- heartbeat -------------------------------------------------------------------------------

    def heartbeat_fields(self, st: dict[str, Any], rid: str) -> dict[str, Any]:
        """One `alive` line's fields: the phase, the counters, and what the client has been sent."""
        writer = self.writers.get(rid)
        stats = writer.stats if writer is not None else None
        return dict(phase=st.get("phase"), elapsed_s=round(time.perf_counter() - st["t0"], 2),
                    prompt_tokens=st.get("prompt_tokens"),
                    out_events=stats.events if stats else None,
                    out_chars=stats.bytes if stats else None,
                    last_out_s=round(stats.since_data_s(), 1) if stats else None)

    async def heartbeat_forever(self, interval_s: float | None = None) -> None:
        """One `alive` line per in-flight request every interval, until cancelled.

        Started by a server on app startup and cancelled on cleanup; a no-op loop when the log is off
        (nothing is in flight then either, since `track` is what fills the table)."""
        every = interval_s if interval_s is not None else self.reqlog.heartbeat_s
        while True:
            await asyncio.sleep(every)
            for rid, st in list(self.inflight.items()):
                self._counters = dict(self.counters() if self.counters else {})
                self.reqlog.log("alive", req=rid, **self.heartbeat_fields(st, rid), **self._counters)


def refusal_text(exc: Any) -> str:
    """The server's own refusal text, without the exception type (the line already names the request).

    A JSON parse error is reduced to line/column: its message quotes the request body."""
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, json.JSONDecodeError):
        return sanitize(cause)
    text = " ".join(str(getattr(exc, "reason", "") or "").split())[:160]
    return text or type(exc).__name__


def log_dir_from_env(environ: Any = None) -> pathlib.Path | None:
    """The directory to write logs in, or None when logging is off.

    `KYOJIN_REQLOG_DIR` names a *directory*. A path that ends in a file is read as that file's directory
    (`/var/log/kyojin/requests.log` means `/var/log/kyojin`), because that is what an operator has in mind
    when they point the switch at a log, and that file is never written to: every run writes its own
    `<BASE_NAME>-<run>.log` in the directory.
    """
    env = os.environ if environ is None else environ
    raw = str(env.get("KYOJIN_REQLOG_DIR", "") or "").strip()
    if not raw or raw.lower() in FALSE:
        return None
    if raw.lower() in TRUE:
        return pathlib.Path(DEFAULT_DIR).expanduser()
    target = pathlib.Path(raw).expanduser()
    directory = target if (target.is_dir() or target.name in ("", ".", "..")) else target.parent
    return directory if str(directory) else pathlib.Path(".")


class ReqLog:
    """Append-only event log, one file per run. Safe when logging is disabled or broken."""

    def __init__(self, path: str | None = None, heartbeat_s: float | None = None,
                 server: str | None = None) -> None:
        self.server = server
        if path is None:
            self.directory = log_dir_from_env()
            self.enabled = self.directory is not None
        else:
            # an explicit path is a directory too, with the same rule (a file name means its directory)
            raw = str(path).strip()
            self.enabled = raw.lower() not in FALSE
            target = pathlib.Path(raw).expanduser() if self.enabled else None
            if target is None or raw.lower() in TRUE:
                self.directory = pathlib.Path(DEFAULT_DIR).expanduser() if self.enabled else None
            else:
                self.directory = target if (target.is_dir() or target.name in ("", ".", "..")) else target.parent
        self.base_path = self.directory / BASE_NAME if self.directory is not None else None
        self.run = time.strftime(RUN_STAMP) if self.enabled else None
        # one file per start, so a file is one run; the fixed base name is what the retention matches on
        self.path = (self.base_path.with_name(f"{self.base_path.name}-{self.run}.log")
                     if self.base_path is not None else None)
        if heartbeat_s is None:
            try:
                heartbeat_s = float(os.environ.get("KYOJIN_REQLOG_HEARTBEAT_S", "10") or 10)
            except ValueError:
                heartbeat_s = 10.0
        self.heartbeat_s = max(0.0, heartbeat_s)
        try:
            self.keep = int(os.environ.get("KYOJIN_REQLOG_KEEP", DEFAULT_KEEP) or DEFAULT_KEEP)
        except ValueError:
            self.keep = DEFAULT_KEEP
        self._lock = threading.Lock()
        self.pruned = 0
        self._fh = None
        self.open()

    def open(self) -> bool:
        """Open this run's file (the constructor calls it). A failure leaves the log off, never raises."""
        if not self.enabled or self.path is None or self._fh is not None:
            return self._fh is not None
        try:
            # Owner-only: the log names every request's size, timing and outcome, and it is derived
            # from what clients sent (see the module docstring). The file mode applies when the file
            # is created; an older one is tightened here too. A shared folder (the deployment's
            # ~/.cache/kyojin) keeps the mode the deployment gave it.
            secure_dir(self.path.parent)
            # This run's file is created before pruning, so that the directory holds exactly `keep` of them
            # at every start (and not `keep` plus the one that just appeared)
            fd = secure_open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
            self._fh = os.fdopen(fd, "a", encoding="utf-8")
            restrict(self.path, 0o600)
            self.pruned = self.prune_old_runs()
            return True
        except OSError as exc:
            print(f"serve: request log off ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
            self.enabled, self._fh = False, None
            return False

    @classmethod
    def from_env(cls, server: str | None = None) -> "ReqLog":
        return cls(server=server)

    def describe(self) -> str:
        """One start-up line, so an operator can tell whether the log is on without looking for a file."""
        if self._fh is None or self.path is None:
            return "request log off (set KYOJIN_REQLOG_DIR=<directory> to enable, numbers and event names only)"
        pruned = f", {self.pruned} older removed" if self.pruned else ""
        return (f"request log {self.path} (run {self.run}, heartbeat {self.heartbeat_s:g} s, "
                f"keeping {self.keep} runs{pruned}, numbers and event names only, no prompt or reply text)")

    def prune_old_runs(self) -> int:
        """Remove the log files of the oldest runs, keeping the newest `keep` beside the base path.

        Runs are ordered by the stamp in their name, not by their dates, so a file that was copied around
        keeps its place in the order. Only files this module writes are candidates - `<stem>-<stamp><suffix>`
        - so an archive (a renamed or compressed one, say) is never deleted, and the base file of an earlier
        layout (a plain `requests.log` someone appended to) is left where it is. Called once, at start-up:
        nothing is checked while a run is going on, so the request path pays nothing for this.
        """
        if self.directory is None or self.keep <= 0:
            return 0
        try:
            candidates = sorted(self.directory.glob(f"{BASE_NAME}-*.log"))
        except OSError:
            return 0
        doomed = candidates[:max(0, len(candidates) - self.keep)]         # this run is one of them
        removed = 0
        for entry in doomed:
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def log(self, event: str, **fields: Any) -> None:
        if self._fh is None:
            return
        try:
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            ms = int((time.time() % 1) * 1000)
            parts = [f"{stamp}.{ms:03d}", event]
            if self.server:
                parts.append(f"srv={self.server}")
            if self.run:
                parts.append(f"run={self.run}")
            for key, value in fields.items():
                if value is None:
                    continue
                parts.append(f"{key}={_fmt(value)}")
            line = " ".join(parts) + "\n"
            with self._lock:
                self._fh.write(line)
                self._fh.flush()
        except Exception:  # noqa: BLE001  a logging failure must never touch serving
            pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None

    def lines(self) -> list[str]:
        """The lines written so far (for tests and for a live tail; never opens a second handle)."""
        if self.path is None or not self.path.exists():
            return []
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
        return self.path.read_text(encoding="utf-8").splitlines()
