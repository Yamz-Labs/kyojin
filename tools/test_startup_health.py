"""CPU tests of tools/startup_health.py (GET /health with load progress) and of its wiring in the three servers.

Run: cd tools && python3 -m unittest test_startup_health
"""
from __future__ import annotations

import ast
import asyncio
import json
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import startup_health as sh  # noqa: E402

TOOLS = Path(__file__).parent


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def tracker(stages, history=None, clock=None):
    return sh.StartupTracker("t", stages, clock=clock or Clock(), history=history or Path("/nonexistent/none.json"))


def get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read()), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), e.headers


class SnapshotTest(unittest.TestCase):
    def test_first_stage_at_zero(self):
        t = tracker(["a", "b"])
        code, body = t.snapshot()
        self.assertEqual(code, 503)
        self.assertEqual((body["status"], body["stage"], body["stage_index"], body["stage_count"]), ("loading", "a", 1, 2))
        self.assertEqual(body["progress"], 0.0)
        self.assertIsNone(body["stage_progress"])
        self.assertIsNone(body["eta_s"])
        self.assertEqual(body["progress_basis"], "stages")

    def test_progress_counts_finished_stages_and_reported_fraction(self):
        c = Clock()
        t = tracker(["a", "b", "c", "d"], clock=c)
        t.progress(1, 2)
        self.assertEqual(t.snapshot()[1]["progress"], 0.125)
        c.t += 30
        t.begin("b")
        body = t.snapshot()[1]
        self.assertEqual((body["stage"], body["stage_index"], body["progress"], body["stage_progress"]), ("b", 2, 0.25, None))
        self.assertEqual(body["elapsed_s"], 30.0)
        t.progress(3, 4)
        self.assertEqual(t.snapshot()[1]["progress"], 0.438)       # (1 + 0.75) / 4, rounded

    def test_unknown_stage_progress_stays_unknown_not_guessed(self):
        c = Clock()
        t = tracker(["a", "b"], clock=c)
        c.t += 500                                                  # a long stage with no counter
        body = t.snapshot()[1]
        self.assertEqual(body["progress"], 0.0)
        self.assertIsNone(body["stage_progress"])
        self.assertIsNone(body["eta_s"])

    def test_never_reports_one_while_loading(self):
        t = tracker(["a"])
        t.progress(10, 10)
        self.assertLess(t.snapshot()[1]["progress"], 1.0)

    def test_stage_never_goes_backwards_and_repeat_keeps_fraction(self):
        t = tracker(["a", "b"])
        t.begin("b")
        t.progress(1, 2)
        t.begin("a")
        t.begin("b")
        body = t.snapshot()[1]
        self.assertEqual((body["stage"], body["stage_progress"]), ("b", 0.5))

    def test_ready_error(self):
        t = tracker(["a"])
        t.ready()
        self.assertEqual(t.snapshot(), (200, {"status": "ok"}))
        t = tracker(["a"])
        t.fail("boom")
        code, body = t.snapshot()
        self.assertEqual((code, body["status"], body["message"]), (500, "error", "boom"))

    def test_progress_ignores_zero_total(self):
        t = tracker(["a"])
        t.progress(0, 0)
        self.assertIsNone(t.snapshot()[1]["stage_progress"])


class HistoryTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "h.json"

    def tearDown(self):
        self.dir.cleanup()

    def run_start(self, stages, durations, skip=()):
        c = Clock()
        t = tracker(stages, history=self.path, clock=c)
        for s in stages:
            if s in skip:
                continue
            t.begin(s)
            c.t += durations[s]
        t.ready()

    def test_history_saved_and_used_for_eta_and_weights(self):
        self.run_start(["a", "b", "c"], {"a": 60, "b": 30, "c": 10})
        self.assertEqual(json.loads(self.path.read_text()), {"a": 60.0, "b": 30.0, "c": 10.0})
        c = Clock()
        t = tracker(["a", "b", "c"], history=self.path, clock=c)
        c.t += 20
        body = t.snapshot()[1]
        self.assertEqual(body["progress_basis"], "history")
        self.assertEqual(body["progress"], 0.0)                     # no fraction reported: no guess
        self.assertEqual(body["eta_s"], 80.0)                       # (60 - 20) + 30 + 10
        t.progress(1, 2)
        t.begin("b")
        self.assertEqual(t.snapshot()[1]["progress"], 0.6)          # 60 of 100 s done

    def test_eta_uses_current_pace_when_fraction_known(self):
        self.run_start(["a", "b"], {"a": 600, "b": 10})
        c = Clock()
        t = tracker(["a", "b"], history=self.path, clock=c)
        c.t += 10
        t.progress(50, 100)                                         # 10 s for half: 10 s left, not 590
        self.assertEqual(t.snapshot()[1]["eta_s"], 20.0)            # 10 + the next stage's 10

    def test_overrun_stage_without_counter_has_no_eta(self):
        self.run_start(["a", "b"], {"a": 10, "b": 10})
        c = Clock()
        t = tracker(["a", "b"], history=self.path, clock=c)
        c.t += 50
        self.assertIsNone(t.snapshot()[1]["eta_s"])

    def test_run_that_skipped_a_stage_saves_nothing(self):
        self.run_start(["a", "b"], {"a": 5, "b": 5}, skip=("b",))
        self.assertFalse(self.path.exists())

    def test_history_for_other_stage_list_is_ignored(self):
        self.path.write_text(json.dumps({"a": 5, "z": 5}))
        t = tracker(["a", "b"], history=self.path)
        self.assertEqual(t.snapshot()[1]["progress_basis"], "stages")

    def test_corrupt_history_is_ignored(self):
        self.path.write_text("{nope")
        self.assertEqual(tracker(["a"], history=self.path).snapshot()[1]["progress_basis"], "stages")


class LoadCallbackTest(unittest.TestCase):
    def setUp(self):
        self.t = tracker(["drafter", "target", "setup"])
        sh.ACTIVE = self.t

    def tearDown(self):
        sh.ACTIVE = None

    def test_single_model_reports_module_fraction(self):
        cb = sh.load_callback()
        cb(0, 10)
        cb(5, 10)
        self.assertEqual(self.t.snapshot()[1]["stage_progress"], 0.5)

    def test_two_models_one_call_roll_to_the_next_stage(self):
        cb = sh.load_callback(["drafter", "target"])
        for i in range(4):
            cb(i, 4)
        self.assertEqual(self.t.snapshot()[1]["stage"], "drafter")
        cb(0, 40)                                                   # the counter dropped: the target started
        body = self.t.snapshot()[1]
        self.assertEqual((body["stage"], body["stage_progress"]), ("target", 0.0))
        cb(20, 40)
        self.assertEqual(self.t.snapshot()[1]["stage_progress"], 0.5)

    def test_helpers_are_noops_without_tracker_or_for_unknown_stage(self):
        sh.stage("nope")                                            # unknown name: ignored
        self.assertEqual(self.t.snapshot()[1]["stage"], "drafter")
        sh.ACTIVE = None
        sh.stage("target")
        sh.progress(1, 2)
        sh.load_callback()(1, 2)
        sh.finish()
        sh.abort("x")


class ListenerTest(unittest.TestCase):
    def test_health_and_other_paths_while_loading_then_handoff(self):
        c = Clock()
        t = tracker(["weights", "warm-up"], clock=c)
        t.serve("127.0.0.1", 0)
        port = t.server.port
        try:
            c.t += 12
            t.progress(1, 4)
            code, body, headers = get(f"http://127.0.0.1:{port}/health")
            self.assertEqual(code, 503)
            self.assertEqual((body["status"], body["stage"], body["stage_progress"], body["elapsed_s"]),
                             ("loading", "weights", 0.25, 12.0))
            self.assertEqual(headers["Retry-After"], "5")
            code, body, _ = get(f"http://127.0.0.1:{port}/v1/models")        # llama.cpp style: 503 "Loading model"
            self.assertEqual((code, body["error"]["message"]), (503, "Loading model"))
            code, body, _ = get(f"http://127.0.0.1:{port}/health?x=1")
            self.assertEqual(code, 503)
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=b"{}", method="POST")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(cm.exception.code, 503)
            t.fail("kernel build failed")
            code, body, _ = get(f"http://127.0.0.1:{port}/health")
            self.assertEqual((code, body["status"], body["message"]), (500, "error", "kernel build failed"))
        finally:
            t.release_port()
        with self.assertRaises(OSError):                                    # the port is closed after the handoff
            socket.create_connection(("127.0.0.1", port), timeout=1)

    def test_port_is_answering_before_the_load_and_real_server_binds_after(self):
        """The early listener and an aiohttp TCPSite take the same port one after the other."""
        from aiohttp import web
        t = tracker(["weights"])
        t.serve("127.0.0.1", 0)
        port = t.server.port
        self.assertEqual(get(f"http://127.0.0.1:{port}/health")[0], 503)    # before any model code runs
        result = {}

        async def run():
            app = web.Application()

            async def health(_):
                return web.json_response({"status": "ok", "model": "m"})
            app.router.add_get("/health", health)
            runner = web.AppRunner(app)
            await runner.setup()
            t.release_port()
            t.ready()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            result["r"] = await asyncio.get_running_loop().run_in_executor(None, get, f"http://127.0.0.1:{port}/health")
            await runner.cleanup()
        asyncio.run(run())
        self.assertEqual(result["r"][0], 200)
        self.assertEqual(result["r"][1]["model"], "m")

    def test_start_fails_fast_when_the_port_is_taken(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        try:
            with self.assertRaises(OSError):
                sh.start("t", ["a"], "127.0.0.1", s.getsockname()[1])
        finally:
            s.close()
            sh.ACTIVE = None

    def test_concurrent_polls(self):
        t = tracker(["a"])
        t.serve("127.0.0.1", 0)
        codes = []
        threads = [threading.Thread(target=lambda: codes.append(get(f"http://127.0.0.1:{t.server.port}/health")[0]))
                   for _ in range(8)]
        [x.start() for x in threads]
        [x.join() for x in threads]
        t.release_port()
        self.assertEqual(codes, [503] * 8)


class WiringTest(unittest.TestCase):
    """The three servers call the same helper, in the right order, with stage names the lists declare."""

    EXPECT = {
        "qwen": {"target weights", "drafter weights", "vision tower", "engine setup", "warm-up"},
        "glm": {"target weights", "drafter weights", "warm-up", "dense GEMM tuning"},
        "mimo": {"target weights", "drafter weights", "engine setup"},
    }

    def calls(self, name):
        src = (TOOLS / name / "serve.py").read_text()
        tree = ast.parse(src)
        out = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                    and n.func.value.id == "startup_health":
                out.append((n.func.attr, n.lineno, [a.value for a in n.args if isinstance(a, ast.Constant)]))
        return src, out

    def test_each_server_starts_listener_and_releases_before_bind(self):
        for name in self.EXPECT:
            src, calls = self.calls(name)
            attrs = {a for a, _, _ in calls}
            self.assertTrue({"start", "stage", "finish", "abort", "load_callback"} <= attrs, name)
            finish = min(l for a, l, _ in calls if a == "finish")
            start = min(l for a, l, _ in calls if a == "start")
            bind = src[: src.index("TCPSite(runner") if "TCPSite(runner" in src else src.index("web.run_app(app")].count("\n") + 1
            self.assertLess(start, finish, name)
            self.assertLess(finish, bind, name)

    def test_stage_names_are_declared(self):
        for name, expect in self.EXPECT.items():
            _, calls = self.calls(name)
            used = {args[0] for a, _, args in calls if a == "stage" and args}
            self.assertTrue(used <= expect, (name, used - expect))
            src = (TOOLS / name / "serve.py").read_text()
            for stage in expect:                                           # every declared stage is entered somewhere
                self.assertIn(f'"{stage}"', src, (name, stage))

    def test_mimo_loads_drafter_first_like_model_init(self):
        src = (TOOLS.parent / "exllamav3" / "model_init.py").read_text()
        self.assertLess(src.index("draft_model.load("), src.index("    model.load("))
        self.assertIn('["drafter weights"] if drafter_path else []) + ["target weights"]', (TOOLS / "mimo" / "serve.py").read_text())

    def test_default_ready_line_unchanged(self):
        self.assertIn("qserve: READY on http://", (TOOLS / "qwen" / "serve.py").read_text())


if __name__ == "__main__":
    unittest.main()
