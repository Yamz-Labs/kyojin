#!/usr/bin/env python3
"""Model-free tests for the shared SSE keep-alive (tools/strix_halo/serve_ping.py).

The timing rules live in one place because the three servers share them, so they are tested here
once: when a comment is due, that it is one per interval and not one per poll, that a retired
pinger stays quiet, and that the switch is off unless KYOJIN_SSE_PING_S says otherwise.

    kyojin/.venv/bin/python -m unittest tools.strix_halo.test_serve_ping     # 8 tests, no GPU, no model
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import time
import unittest
from pathlib import Path

# Load the shared module by its own path: `serve_ping` is also a symlink in tools/glm, tools/qwen and
# tools/mimo, and a test run that imported one of those first would otherwise hand this file a second
# copy of the class (tests/ and the servers must see the same one).
_spec = importlib.util.spec_from_file_location("strix_halo_serve_ping", Path(__file__).with_name("serve_ping.py"))
_serve_ping = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_serve_ping)
ServePing = _serve_ping.ServePing


class SwitchTests(unittest.TestCase):
    """KYOJIN_SSE_PING_S: unset or 0 means off, a positive number is the interval, junk says so once."""

    def setUp(self):
        os.environ.pop("KYOJIN_SSE_PING_S", None)
        self.addCleanup(os.environ.pop, "KYOJIN_SSE_PING_S", None)

    def stderr(self) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            return buf.getvalue()

    def test_off_by_default(self):
        ping = ServePing.from_env()
        self.assertFalse(ping.enabled)
        self.assertEqual(ping.interval_s, 0.0)
        self.assertIn("off", ping.describe())
        self.assertIsNone(ping.due(time.perf_counter()))      # and never writes anything

    def test_zero_is_off_and_says_nothing(self):
        os.environ["KYOJIN_SSE_PING_S"] = "0"
        self.assertFalse(ServePing.from_env().enabled)
        self.assertEqual(self.stderr(), "")                    # asking for off is not a mistake

    def test_an_interval_is_read_and_described(self):
        os.environ["KYOJIN_SSE_PING_S"] = "15"
        ping = ServePing.from_env()
        self.assertTrue(ping.enabled)
        self.assertEqual(ping.interval_s, 15.0)
        self.assertIn("15 s", ping.describe())

    def test_junk_and_negative_values_are_off_with_one_line(self):
        for raw in ("soon", "-3"):
            os.environ["KYOJIN_SSE_PING_S"] = raw
            with self.subTest(raw=raw):
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    ping = ServePing.from_env()
                self.assertFalse(ping.enabled)
                self.assertIn("KYOJIN_SSE_PING_S", buf.getvalue())   # a switch that was asked for is reported


class TimingTests(unittest.TestCase):
    """A comment covers every silence (queue, prefill, decode), once per interval, and never during output."""

    def test_a_comment_is_due_after_the_interval_and_not_before(self):
        ping = ServePing(interval_s=0.2)
        started = time.perf_counter()
        self.assertIsNone(ping.due(started))                      # nothing due while the client just arrived
        time.sleep(0.25)
        waited = ping.due(started)
        self.assertIsNotNone(waited)
        self.assertGreaterEqual(waited, 0.2)
        self.assertEqual(ping.sent, 0)                       # asking is not sending
        ping.sent_comment(waited)
        self.assertEqual(ping.sent, 1)

    def test_only_one_comment_per_interval_even_though_polling_is_faster(self):
        ping = ServePing(interval_s=0.25)
        started = time.perf_counter()
        while time.perf_counter() - started < 0.15:          # the servers poll twice a second
            time.sleep(0.01)
            ping.due(started)
        self.assertEqual(ping.sent, 0)                       # polling is not what makes a comment due
        time.sleep(0.15)
        ping.sent_comment(ping.due(started))
        self.assertIsNone(ping.due(started))                     # and not a second one right after
        self.assertEqual(ping.sent, 1)

    def test_a_retired_pinger_stays_quiet(self):
        ping = ServePing(interval_s=0.1)
        started = time.perf_counter()
        time.sleep(0.15)
        ping.sent_comment(ping.due(started))
        ping.limit()                                          # the stream is over / nobody is listening
        time.sleep(0.15)
        self.assertIsNone(ping.due(started))
        self.assertEqual(ping.sent, 1)

    def test_output_silences_the_comments_and_silence_brings_them_back(self):
        # The case that matters for a held-back tool call: the request writes nothing for minutes in
        # the middle of a generation, and the client must still see that the connection is alive.
        ping = ServePing(interval_s=0.15)
        started = time.perf_counter()
        ping.note_output()                                    # the role chunk went out
        for _ in range(10):                                   # a stream that keeps writing: no comments
            time.sleep(0.02)
            ping.note_output()
            self.assertIsNone(ping.due(started))
        self.assertEqual(ping.sent, 0)
        time.sleep(0.2)                                       # ...and then the model goes quiet
        ping.sent_comment(ping.due(started))
        ping.note_output()                                    # the next data event ends the silence
        self.assertIsNone(ping.due(started))
        time.sleep(0.2)
        ping.sent_comment(ping.due(started))
        self.assertEqual(ping.sent, 2)

    def test_the_end_of_the_stream_ends_the_comments(self):
        ping = ServePing(interval_s=0.1)
        started = time.perf_counter()
        time.sleep(0.12)
        ping.sent_comment(ping.due(started))
        ping.limit()                                          # [DONE] went out: the request is over
        time.sleep(0.15)
        self.assertIsNone(ping.due(started))
        self.assertEqual(ping.sent, 1)

    def test_each_request_gets_its_own_timers(self):
        os.environ["KYOJIN_SSE_PING_S"] = "0.1"
        self.addCleanup(os.environ.pop, "KYOJIN_SSE_PING_S", None)
        app = ServePing.from_env()
        first, second = app.per_request(), app.per_request()
        self.assertIsNot(first, second)
        started = time.perf_counter()
        time.sleep(0.15)
        self.assertIsNotNone(first.due(started))
        self.assertIsNotNone(second.due(started))                 # one session's wait does not mute another's
        first.limit()                                         # the first session is over
        self.assertIsNone(first.due(started))
        second._last_sent = 0.0                                # what this really checks is `limit`'s scope
        self.assertIsNotNone(second.due(started))
        self.assertTrue(app.enabled)                          # the application's own settings are untouched

    def test_the_interval_is_fixed_when_the_switch_is_read(self):
        app = ServePing(interval_s=0.1)
        os.environ["KYOJIN_SSE_PING_S"] = "3600"              # a later change does not reach a running server
        self.addCleanup(os.environ.pop, "KYOJIN_SSE_PING_S", None)
        self.assertEqual(app.per_request().interval_s, 0.1)


if __name__ == "__main__":
    unittest.main()
