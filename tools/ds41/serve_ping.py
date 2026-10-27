#!/usr/bin/env python3
"""SSE keep-alive for the kyojin servers: a comment line whenever a stream has nothing to send.

A streaming request is answered with its SSE headers and one empty role chunk, and after that the
client hears nothing until text is ready. Two silences are long enough to matter:

* the prefill of a long prompt — a large image conversation is hundreds of thousands of tokens, and a
  client that aborts a stream it has not heard from for a couple of minutes aborts a server that is
  working; an aborted request leaves no reusable state behind, so the retry pays the whole prefill again;
* the decode of a reply that writes a file — a tool call is parsed only from complete XML, so the whole
  `<tool_call>` is held back and a 7 KB `content` argument is minutes of generation that the client
  never sees (`probe` runs measured 99 s of decode with nothing on the wire).

This writes an SSE **comment** — a line starting with ':', which every event-stream client skips — after
N seconds without a data event, whenever that happens, so a client that only watches its socket sees
that the server is alive and a client that never sees one can report a genuinely dead server.

**Off unless asked for.** With KYOJIN_SSE_PING_S unset (or 0) nothing is ever written:

    KYOJIN_SSE_PING_S=15     # seconds of silence before one ': waiting <n> s' comment line

The interval is read when a request starts streaming, and the server prints one line at start-up
either way, so the switch stays discoverable when a request does go quiet. The class holds timers
only: the caller writes the comment to its own StreamResponse, and every request makes its own
pinger, which keeps this module free of aiohttp and gives the three servers exactly one
implementation of the timing rules.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any


class ServePing:
    """Times the keep-alive comments of one streaming request. Cheap enough to call on every poll.

    A comment is due when the client has heard nothing for the interval, whatever the server is doing:
    prefilling a long prompt, waiting in the queue, or decoding. The last case matters because a tool
    call is parsed only from complete XML, so a reply that writes a large file holds its whole
    `<tool_call>` back and a client watching only its socket sees nothing while the model types it.
    `note_output` is called with every data event the request writes, which is what ends a silence.
    """

    def __init__(self, interval_s: float = 0.0, reqlog: Any = None) -> None:
        self.interval_s = max(0.0, float(interval_s))
        self.reqlog = reqlog
        self.enabled = self.interval_s > 0.0
        self.sent = 0                # comment lines written; for the log and for tests
        self._last_sent = 0.0        # perf_counter of the last comment (one per interval, not per poll)
        self._last_output = 0.0      # perf_counter of the last data event this request wrote

    @classmethod
    def from_env(cls, reqlog: Any = None) -> "ServePing":
        """Read KYOJIN_SSE_PING_S. A value that is not a positive number means off, with one line to
        stderr: this is a client-visible protocol switch, so a server that was asked for it and did
        not get it should say so rather than look like a client that ignores the comments."""
        raw = os.environ.get("KYOJIN_SSE_PING_S", "").strip()
        try:
            interval = float(raw) if raw else 0.0
        except ValueError:
            print(f"serve: SSE keep-alive off (KYOJIN_SSE_PING_S={raw!r} is not a number)",
                  file=sys.stderr, flush=True)
            return cls(0.0, reqlog)
        if raw and interval <= 0:  # "0" is the documented way of asking for off, so it stays quiet
            if interval < 0:
                print(f"serve: SSE keep-alive off (KYOJIN_SSE_PING_S={raw!r} is negative)",
                      file=sys.stderr, flush=True)
            return cls(0.0, reqlog)
        return cls(interval, reqlog)

    def per_request(self) -> "ServePing":
        """This request's own timers, from the application's settings.

        The interval is fixed for the life of a server, so it is parsed once (from_env) and copied
        into one pinger per streaming request: its timers belong to a request, and two
        sessions pinging into the same object would time each other's comments."""
        return ServePing(self.interval_s, self.reqlog) if self.enabled else self

    def describe(self) -> str:
        """One start-up line, so an operator can tell whether the comments are on and how often."""
        if not self.enabled:
            return "SSE keep-alive off (set KYOJIN_SSE_PING_S=<seconds> to hold a silent stream open)"
        return (f"SSE keep-alive: a comment line after {self.interval_s:g} s without output "
                f"(KYOJIN_SSE_PING_S)")

    def due(self, started: float) -> float | None:
        """Whether a comment is due, and the seconds to name in it (None = not now).

        `started` is the perf_counter the request's wait began at, so the comment counts what the
        client has actually been waiting for. Due means the interval has passed since that start,
        since the request's last data event, and since the last comment itself — so a stream that is
        writing never gets comments, and one that has gone quiet gets them even while the engine keeps
        producing (a held-back tool call is decoded the whole time and sent at the end).

        Checking writes nothing: the caller writes the comment and then calls `sent_comment`, which is
        what makes this usable from a loop that must also poll for output.
        """
        if not self.enabled:
            return None
        now = time.perf_counter()
        for mark in (started, self._last_output, self._last_sent):
            if now - mark < self.interval_s:
                return None
        return now - started

    def sent_comment(self, waited: float) -> None:
        """Record a comment the caller just wrote, and log the number of the request's comments."""
        self._last_sent = time.perf_counter()
        self.sent += 1
        if self.reqlog is not None:
            self.reqlog.log("sse_ping", n=self.sent, waited_s=round(waited, 2))

    async def comment(self, writer: Any, started: float) -> None:
        """Write one comment if the client has waited for it. Never returns with a dead connection.

        This is the whole keep-alive, one call, so the three servers' polling loops grow by one line:
        they call it before every poll, because the moment a stall begins is not a moment a poll
        happens to time out — a reply that holds a large tool call back keeps producing deltas (and
        writing nothing to the client) the whole time it is decoded.

        `writer` is the request's StreamWriter (or None for a reply with no stream). Comments are
        counted as comments, not as data events, so a flat `out_events` in the log still means the
        model wrote nothing the client could render.
        """
        if writer is None:
            return
        waited = self.due(started)
        if waited is None:
            return
        data = f": waiting {waited:.0f} s\n\n".encode()
        stats = getattr(writer, "stats", None)
        try:
            if stats is None:                     # a bare StreamResponse: nothing to record into
                await writer.write(data)
            else:                                 # counted as a comment, not as a data event
                await writer.response.write(data)
        except ConnectionError as exc:
            self.limit()
            raise ConnectionResetError("client left") from exc
        if stats is not None:
            stats.note_comment(len(data))
        self.sent_comment(waited)

    def note_output(self) -> None:
        """The request wrote a data event: the client has something, so the silence clock restarts.

        A comment comes again only after the interval has passed with nothing written (which is what
        a decode that is holding a tool call back looks like from the socket)."""
        self._last_output = time.perf_counter()

    def limit(self) -> None:
        """Stop commenting: the request is over, or nobody is left to write to."""
        self.enabled = False
