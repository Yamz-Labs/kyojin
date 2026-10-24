#!/usr/bin/env python3
"""The three servers must log the same thing: one vocabulary, one set of fields per event.

The request log is shared code (tools/strix_halo/serve_reqlog.py) and the point of sharing it is that a
server cannot invent its own spelling of an event or forget a field the others report. This test runs the
same requests against all three servers - model-free, each serve.py loaded from its own path so the three
modules cannot shadow each other - and compares what they wrote.

    kyojin/.venv/bin/python -m unittest tools.test_serve_logging

What is asserted:

* every event a server writes is one this log defines (a typo like `job_started` fails here);
* the events a streamed chat request produces are the same on all three, field for field, once the
  counters each server actually keeps are taken out (`running`, `processing`) - a server that reports
  fewer counters is not a difference in vocabulary;
* `req=` is on every line, and every request opens and closes (`req_start` .. `req_done`), so
  `grep <id>` works on all three;
* the heartbeat carries the output accounting on all three.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent

# Every event name the shared log defines. A server writing anything else has invented a spelling.
KNOWN_EVENTS = {
    "req_start", "req_queued", "req_refused", "req_error", "req_done",
    "job_start", "job_skipped", "job_first_token", "job_done", "job_error",
    "alive", "sse_headers", "sse_first_content", "sse_ping", "sse_done", "stream_error",
    "sse_summary", "client_left",
    # server-specific lines that go through the raw log() escape hatch
    "serve_start", "img_cache", "img_disk", "rss",
}
# Counters each server keeps; keeping one and not the other is not a vocabulary difference.
COUNTER_FIELDS = {"running", "processing"}
# The keep-alive fields only exist for a request that produced a comment, and whether one is due at all
# depends on how often a server polls its queue (Qwen polls once a second, GLM and MiMo twice), so a
# server with no comment is not missing a field. `test_a_comment_fills_the_summary` covers them.
COMMENT_FIELDS = {"comments", "comment_chars", "last_comment_s"}
# Fields that only exist where the feature does: MiMo has no --sessions (one request at a time), so
# there is no session count to report. A server may have them or not; it may not have anything else.
OPTIONAL_FIELDS = {"sessions", "messages"}


class FakeGenerator:
    num_draft_tokens = 2


class FakeEngine:
    """What all three servers need from an engine, and nothing more."""

    ctx = 4096
    supports_vision = True
    spec_on = False
    sessions = 1

    def __init__(self) -> None:
        self.greedy_generator = FakeGenerator()

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def prompt_ids(self, prompt: str):
        """The ids the completion route counts tokens with; the real engines have an encode cache here.

        This double counts words, so the ids are placeholders of that length - the route only asks the
        result for `numel()`, and that number is what it puts in the log."""
        import torch
        return torch.zeros(int(self.count_tokens(prompt)), dtype=torch.long)

    def spec_stats(self) -> dict:
        return {}

    def gate_stats(self) -> dict:
        return {}

    def reset_gate(self) -> None:
        pass

    async def generate(self, prompt, **kw):
        await asyncio.sleep(0.5)                     # a prefill long enough to produce 'alive' lines
        for piece in ("he", "llo"):
            yield piece


def load_server(name: str):
    """Load tools/<name>/serve.py under a private module name, with its own folder on sys.path."""
    here = TOOLS / name
    sys.path.insert(0, str(here))
    try:
        spec = importlib.util.spec_from_file_location(f"{name}_serve_under_test", here / "serve.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(here))


def run_log_file(directory: Path) -> Path:
    """The file the last run wrote in a directory: `requests-<run>.log`, one file per run."""
    if directory.suffix:                      # a file name handed in: its directory is what is used
        directory = directory.parent
    written = sorted(directory.glob("requests-*.log"))
    return written[-1] if written else directory / "requests-none.log"


def parse_log(path: Path) -> dict[str, dict[str, set[str]]]:
    """{request id: {event: fields}}; the heartbeat's own lines land under the key '-' if untagged."""
    path = run_log_file(path)                       # one file per run: read the one this battery wrote
    out: dict[str, dict[str, set[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(" ")
        name = parts[1]
        fields = {p.split("=", 1)[0] for p in parts[2:] if "=" in p}
        rid = next((p[4:] for p in parts[2:] if p.startswith("req=")), "-")
        out.setdefault(rid, {}).setdefault(name, set()).update(fields - {"req"})
    return out


def run_battery(name: str, log_path: Path) -> dict[str, dict[str, set[str]]]:
    """One streamed chat request and one refused request against one server."""
    serve = load_server(name)

    async def go() -> None:
        from aiohttp.test_utils import TestClient, TestServer
        app = serve.create_app(FakeEngine(), "m", "{{ messages[0].content }}")
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            await r.read()
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": []})
            await r.read()
        finally:
            await client.close()

    os.environ["KYOJIN_REQLOG_DIR"] = str(log_path.parent)      # the switch names a directory
    os.environ["KYOJIN_REQLOG_HEARTBEAT_S"] = "0.2"
    os.environ["KYOJIN_SSE_PING_S"] = "0.2"
    try:
        asyncio.run(go())
    finally:
        for key in ("KYOJIN_REQLOG_DIR", "KYOJIN_REQLOG_HEARTBEAT_S", "KYOJIN_SSE_PING_S"):
            os.environ.pop(key, None)
    return parse_log(log_path)


class LoggingVocabularyTests(unittest.TestCase):
    """All three servers, one log format."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.events = {name: run_battery(name, Path(cls.tmp.name) / f"{name}.log")
                      for name in ("glm", "qwen", "mimo")}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def streamed(self, name: str) -> dict[str, set[str]]:
        """The streamed request's events (the one that writes sse_headers and a summary)."""
        for events in self.events[name].values():
            if "sse_headers" in events and "sse_summary" in events:
                return events
        self.fail(f"{name}: no streamed request in the log ({sorted(self.events[name])})")

    def test_every_event_name_is_one_the_log_defines(self):
        for name, requests in self.events.items():
            for rid, events in requests.items():
                for event in events:
                    self.assertIn(event, KNOWN_EVENTS,
                                  f"{name} wrote an unknown event {event!r} (req={rid})")

    def test_the_three_servers_write_the_same_events_for_a_streamed_request(self):
        seen = {}
        for name in ("glm", "qwen", "mimo"):
            events = self.streamed(name)
            # A server whose run produced no keep-alive comment has no comment fields: that is the
            # polling interval (Qwen polls its queue once a second), not a different vocabulary.
            commented = "sse_ping" in events
            seen[name] = {e: f - COUNTER_FIELDS - (set() if commented else COMMENT_FIELDS)
                          for e, f in events.items()}
        reference = seen["glm"]
        for name, events in seen.items():
            self.assertEqual(set(events), set(reference),
                             f"{name} writes {sorted(set(events) ^ set(reference))} that GLM does not")
            for event, fields in events.items():
                theirs, ours = seen["glm"][event], fields
                self.assertEqual(theirs - OPTIONAL_FIELDS, ours - OPTIONAL_FIELDS,
                                 f"{name}: fields of {event!r} differ from GLM")
                self.assertTrue(ours - theirs <= OPTIONAL_FIELDS,
                                f"{name}: {event!r} lacks {sorted((theirs - ours) - OPTIONAL_FIELDS)}")

    def test_every_request_is_traceable_by_id_on_all_three(self):
        for name, requests in self.events.items():
            traced = [rid for rid in requests if rid != "-"]
            self.assertTrue(traced, f"{name}: no line carries a req= id")
            for rid in traced:
                self.assertIn("req_start", requests[rid], f"{name}: {rid} has no req_start")
                self.assertTrue({"req_done", "req_refused"} & set(requests[rid]),
                                f"{name}: {rid} never closed")

    def test_the_heartbeat_reports_output_accounting_on_all_three(self):
        for name, requests in self.events.items():
            alive = [ev["alive"] for ev in requests.values() if "alive" in ev]
            self.assertTrue(alive, f"{name}: no alive line")
            for fields in alive:
                self.assertTrue({"phase", "elapsed_s"} <= fields, f"{name}: alive without phase/elapsed_s")
                self.assertTrue({"out_events", "out_chars", "last_out_s"} <= fields,
                                f"{name}: alive without the output accounting: {sorted(fields)}")

    def test_a_comment_fills_the_summary(self):
        """The keep-alive's own accounting: a stream that wrote a comment reports it in its summary.

        Whether a comment is due at all is timing (Qwen polls its queue once a second, GLM and MiMo
        twice), so a server without one is skipped rather than failed; `sse_ping` itself is checked on
        the server that produced it."""
        checked = 0
        for name in ("glm", "qwen", "mimo"):
            pings = [ev["sse_ping"] for ev in self.events[name].values() if "sse_ping" in ev]
            if not pings:
                continue                                          # this run's timing produced none
            checked += 1
            for fields in pings:
                self.assertTrue({"n", "waited_s"} <= fields,
                                f"{name}: sse_ping lacks n/waited_s: {sorted(fields)}")
            summary = self.streamed(name).get("sse_summary")
            self.assertIsNotNone(summary, f"{name}: comments without a summary")
            self.assertTrue(COMMENT_FIELDS <= summary,
                            f"{name}: a commented stream's summary lacks {sorted(COMMENT_FIELDS - summary)}")
        if not checked:
            self.skipTest("no server produced a comment in this run")

    def test_the_reference_server_reports_both_counters(self):
        # GLM is the complete implementation: the others report the one counter they keep, and that is
        # the only field difference this test tolerates.
        self.assertTrue({"running", "processing"} <= self.streamed("glm")["alive"])
        for name in ("qwen", "mimo"):
            self.assertTrue(COUNTER_FIELDS & self.streamed(name)["alive"],
                            f"{name}: the heartbeat names no in-flight counter")


if __name__ == "__main__":
    unittest.main()
