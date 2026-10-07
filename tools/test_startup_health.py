"""CPU tests of tools/startup_health.py (GET /health with load progress) and of its wiring in the three servers.

Run: cd tools && python3 -m unittest test_startup_health
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import socket
import sys
import tempfile
import threading
import time
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
        self.assertEqual(t.snapshot(), (200, {"status": "ok", "source": "kyojin"}))
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

    def test_variants_keep_their_histories_apart(self):
        # a cold start (slow stage) must not time a later warm start, and the other way round
        base = Path(self.dir.name)
        old = sh.history_dir
        sh.history_dir = lambda: base
        try:
            for variant, dur in (("untuned", {"a": 5, "b": 600}), ("tuned", {"a": 5, "b": 20})):
                c = Clock()
                t = sh.StartupTracker("g", ["a", "b"], clock=c, variant=variant)
                for stage in ("a", "b"):
                    t.begin(stage)
                    c.t += dur[stage]
                t.ready()
            self.assertEqual(json.loads((base / "startup-g-untuned.json").read_text())["b"], 600.0)
            self.assertEqual(json.loads((base / "startup-g-tuned.json").read_text())["b"], 20.0)
            t = sh.StartupTracker("g", ["a", "b"], clock=Clock(), variant="tuned")
            self.assertEqual(t.snapshot()[1]["eta_s"], 25.0)              # 5 + 20, not 605
            t = sh.StartupTracker("g", ["a", "b"], clock=Clock(), variant="other")
            self.assertEqual(t.snapshot()[1]["progress_basis"], "stages")  # no history for it: no made-up eta
        finally:
            sh.history_dir = old

    def test_stage_eta_from_own_pace_without_history(self):
        c = Clock()
        t = tracker(["a", "b"], clock=c)
        body = t.snapshot()[1]
        self.assertIn("stage_eta_s", body)
        self.assertIsNone(body["stage_eta_s"])
        c.t += 30
        t.progress(1, 4)                                                # 30 s for a quarter: 90 s left in this stage
        body = t.snapshot()[1]
        self.assertEqual((body["eta_s"], body["stage_eta_s"], body["progress_basis"]), (None, 90.0, "stages"))
        self.assertIn("about 90 s left in this stage", body["message"])

    def test_stage_eta_is_there_with_history_too(self):
        self.run_start(["a", "b"], {"a": 600, "b": 10})
        c = Clock()
        t = tracker(["a", "b"], history=self.path, clock=c)
        c.t += 10
        t.progress(50, 100)
        body = t.snapshot()[1]
        self.assertEqual((body["eta_s"], body["stage_eta_s"]), (20.0, 10.0))


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


class SourceMessageTest(unittest.TestCase):
    def test_source_in_every_state_and_loading_message(self):
        c = Clock()
        t = tracker(["target weights", "warm-up"], clock=c)
        t.begin("target weights")
        t.progress(2, 5)
        code, body = t.snapshot()
        self.assertEqual((code, body["source"]), (503, "kyojin"))
        self.assertEqual(body["message"], "Target weights, 40 % (stage 1 of 2)")
        t.begin("warm-up")                                                   # no fraction: overall percent, no ETA
        self.assertEqual(t.snapshot()[1]["message"], "Warm-up (stage 2 of 2), 50 % overall")
        t.fail("boom")
        self.assertEqual(t.snapshot()[1]["source"], "kyojin")
        t2 = tracker(["a"])
        t2.ready()
        self.assertEqual(t2.snapshot(), (200, {"status": "ok", "source": "kyojin"}))

    def test_message_has_eta_when_known(self):
        self.assertEqual(sh._loading_message("target weights", 3, 5, 0.42, 0.5, 35.4),
                         "Target weights, 42 % (stage 3 of 5), about 35 s left")

    def test_servers_own_health_carries_source(self):
        for name in ("qwen", "glm", "mimo"):
            src = (TOOLS / name / "serve.py").read_text()
            i = src.index("async def health")
            self.assertIn('"source": "kyojin"', src[i:i + 500], name)
            if name == "glm":
                self.assertIn('add_get("/health", health)', src)


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

    def _hammer(self, port, stop, refused, ok):
        while not stop.is_set():
            try:
                socket.create_connection(("127.0.0.1", port), timeout=2).close()
                ok.append(1)
            except OSError as exc:
                refused.append(exc)

    def _handoff_under_load(self, handoff):
        """Hammer the port with connections while `handoff(t, port, runner)` swaps the listener for aiohttp."""
        from aiohttp import web
        t = tracker(["weights"])
        t.serve("127.0.0.1", 0)
        port = t.server.port
        stop, refused, ok = threading.Event(), [], []
        threads = [threading.Thread(target=self._hammer, args=(port, stop, refused, ok), daemon=True) for _ in range(4)]
        for th in threads:
            th.start()
        time.sleep(0.2)
        result = {}

        async def run():
            app = web.Application()

            async def health(_):
                return web.json_response({"status": "ok"})
            app.router.add_get("/health", health)
            runner = web.AppRunner(app)
            await runner.setup()
            await handoff(t, port, runner)
            await asyncio.sleep(0.3)
            result["r"] = await asyncio.get_running_loop().run_in_executor(None, get, f"http://127.0.0.1:{port}/health")
            stop.set()                                                       # stop hammering before the server closes
            await asyncio.get_running_loop().run_in_executor(None, lambda: [th.join() for th in threads])
            await runner.cleanup()
        try:
            asyncio.run(run())
        finally:
            stop.set()
            for th in threads:
                th.join()
        return refused, ok, result["r"]

    def test_handoff_refuses_no_connection(self):
        from aiohttp import web

        async def handoff(t, port, runner):
            sock = t.handoff_socket()                                        # what startup_health.finish() does
            t.ready()
            await web.SockSite(runner, sock).start()
        refused, ok, r = self._handoff_under_load(handoff)
        self.assertEqual(refused, [])
        self.assertGreater(len(ok), 50)
        self.assertEqual(r[0], 200)

    def test_close_then_bind_does_refuse_connections(self):
        """Control: the old hand-off (close the listener, then bind) is caught by the hammer."""
        from aiohttp import web

        async def handoff(t, port, runner):
            t.release_port()
            t.ready()
            await asyncio.sleep(1.5)                                         # longer than a client's SYN retry
            await web.TCPSite(runner, "127.0.0.1", port).start()
        refused, ok, r = self._handoff_under_load(handoff)
        self.assertGreater(len(refused), 0)

    def test_finish_returns_the_live_socket_and_none_without_listener(self):
        sh.ACTIVE = None
        self.assertIsNone(sh.finish())
        sh.start("x", ["a"], "127.0.0.1", 0)
        port = sh.ACTIVE.server.port
        sock = sh.finish()
        try:
            self.assertEqual(sock.getsockname()[1], port)
            socket.create_connection(("127.0.0.1", port), timeout=1).close()   # still listening
        finally:
            sock.close()
            sh.ACTIVE = None

    def test_start_fails_fast_when_the_port_is_taken(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        try:
            with self.assertRaises(SystemExit) as cm:                       # one plain line, no traceback
                sh.start("t", ["a"], "127.0.0.1", s.getsockname()[1])
            msg = str(cm.exception)
            self.assertIn("cannot listen on 127.0.0.1:", msg)
            self.assertIn("choose another --port", msg)
            self.assertIsNone(sh.ACTIVE)
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


class RouteErrorTest(unittest.TestCase):
    ROUTES = {"/health": ("GET",), "/v1/models": ("GET",), "/v1/chat/completions": ("POST",), "/slots/{id}": ("POST",)}

    def test_early_listener_answers_404_and_405_in_json_and_keeps_503_for_real_endpoints(self):
        t = tracker(["weights"])
        t.routes = self.ROUTES
        t.serve("127.0.0.1", 0)
        base = f"http://127.0.0.1:{t.server.port}"
        try:
            code, body, headers = get(base + "/nope")
            self.assertEqual((code, body["error"]["code"], body["error"]["type"]), (404, 404, "invalid_request_error"))
            self.assertIn("/nope", body["error"]["message"])
            self.assertEqual(headers["Content-Type"], "application/json")
            code, body, _ = get(base + "/v1/chat/completions")             # GET on a POST route
            self.assertEqual((code, body["error"]["code"]), (405, 405))
            self.assertEqual(_["Allow"], "POST")
            req = urllib.request.Request(base + "/health", data=b"{}", method="POST")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(cm.exception.code, 405)
            self.assertEqual(get(base + "/v1/models")[0], 503)             # a real endpoint: still "loading"
            self.assertEqual(get(base + "/health")[0], 503)
            self.assertEqual(get(base + "/slots/3")[0], 405)               # wildcard segment matched, wrong method
            self.assertEqual(get(base + "/slots")[0], 404)
        finally:
            t.release_port()

    def test_route_methods_matching(self):
        self.assertEqual(sh.route_methods(self.ROUTES, "/slots/7"), ("POST",))
        self.assertEqual(sh.route_methods(self.ROUTES, "/v1/models/"), ("GET",))
        self.assertIsNone(sh.route_methods(self.ROUTES, "/slots/"))
        self.assertIsNone(sh.route_methods(self.ROUTES, "/x/y/z"))

    def test_middleware_gives_json_404_405_and_leaves_handler_answers_alone(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        async def ok(_):
            return web.json_response({"ok": 1})

        async def teapot(_):
            return web.json_response({"error": {"message": "mine"}}, status=418)

        async def go():
            app = web.Application(middlewares=[sh.json_errors_middleware()])
            app.router.add_get("/ok", ok)
            app.router.add_post("/tea", teapot)
            client = TestClient(TestServer(app))
            await client.start_server()
            out = {}
            for key, method, path in (("404", "GET", "/nope"), ("405", "POST", "/ok"), ("ok", "GET", "/ok"), ("418", "POST", "/tea")):
                r = await client.request(method, path)
                out[key] = (r.status, r.headers["Content-Type"], await r.json(), r.headers.get("Allow"))
            await client.close()
            return out
        out = asyncio.run(go())
        self.assertEqual(out["404"][0], 404)
        self.assertTrue(out["404"][1].startswith("application/json"))
        self.assertEqual(out["404"][2]["error"], {"message": "not found: GET /nope", "type": "invalid_request_error", "code": 404})
        self.assertEqual(out["405"][0], 405)
        self.assertEqual(out["405"][3], "GET,HEAD")
        self.assertIn("(allowed: GET, HEAD)", out["405"][2]["error"]["message"])
        self.assertIn("method POST not allowed on /ok", out["405"][2]["error"]["message"])
        self.assertEqual(out["ok"][2], {"ok": 1})
        self.assertEqual((out["418"][0], out["418"][2]["error"]["message"]), (418, "mine"))


class ModelDirTest(unittest.TestCase):
    def test_missing_folder_and_missing_files_give_one_plain_line(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                sh.check_model_dir("qserve", Path(d) / "nope")
            self.assertIn("model folder not found", str(cm.exception))
            self.assertIn("--model", str(cm.exception))
            (Path(d) / "config.json").write_text("{}")
            with self.assertRaises(SystemExit) as cm:
                sh.check_model_dir("qserve", d)
            self.assertIn("missing: chat_template.jinja", str(cm.exception))
            sh.check_model_dir("qserve", d, ("config.json",))                # needed list is the caller's
            (Path(d) / "chat_template.jinja").write_text("x")
            sh.check_model_dir("qserve", d)

    def test_servers_check_the_pack_before_the_port_opens(self):
        for name in ("qwen", "glm", "mimo"):
            src = (TOOLS / name / "serve.py").read_text()
            self.assertLess(src.index("startup_health.check_model_dir("), src.index("startup_health.start("), name)


class QuietDisconnectTest(unittest.TestCase):
    def test_client_that_hangs_up_leaves_no_traceback(self):
        import contextlib
        import io
        t = tracker(["weights"], clock=Clock())
        t.serve("127.0.0.1", 0)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                for _ in range(20):
                    s = socket.create_connection(("127.0.0.1", t.server.port))
                    s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
                    s.close()                                                # reset before the reply is read
                time.sleep(0.3)
                code, _, _ = get(f"http://127.0.0.1:{t.server.port}/health")
            self.assertEqual(code, 503)
        finally:
            t.release_port()
        self.assertNotIn("Traceback", buf.getvalue())


class ForkTest(unittest.TestCase):
    def test_forked_child_does_not_keep_the_port(self):
        import os
        import time
        t = tracker(["a"])
        t.serve("127.0.0.1", 0)
        port = t.server.port
        pid = os.fork()
        if pid == 0:                                                # child: lives on while the parent hands the port over
            time.sleep(3)
            os._exit(0)
        try:
            t.release_port()
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            s.listen(1)
            s.close()
        finally:
            os.kill(pid, 9)
            os.waitpid(pid, 0)


class LoadSignatureTest(unittest.TestCase):
    """The keyword the servers pass must exist on every load path they call (read from source: no torch here)."""

    def test_callback_reaches_load_paths(self):
        root = TOOLS.parent / "exllamav3"
        model = (root / "model" / "model.py").read_text()
        self.assertIn("callback: Callable[[int, int], None] | None = None", model)
        self.assertIn("self._load_single(progressbar, device, self.config, self.modules, verbose, callback)", model)
        self.assertIn("def load(self, *args, **kwargs):", model)                  # load() forwards to load_gen
        ls = (root / "model" / "model_ls.py").read_text()
        self.assertIn("verbose: bool,\n        callback = None", ls)
        init = (root / "model_init.py").read_text()
        self.assertEqual(init.count("**kwargs\n"), 3)                             # signature + the two loads


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

    def test_servers_pass_routes_and_json_middleware_and_routes_match_the_app(self):
        for name in self.EXPECT:
            src = (TOOLS / name / "serve.py").read_text()
            self.assertIn("routes=ROUTES", src, name)
            self.assertIn("middlewares=[startup_health.json_errors_middleware()]", src, name)
            registered = set(re.findall(r'router\.add_(?:get|post)\("([^"]+)"', src))
            declared = set(re.search(r"^ROUTES = (\{.*\})$", src, re.M).group(1).replace("(", "[").replace(")", "]") and
                           json.loads(re.search(r"^ROUTES = (\{.*\})$", src, re.M).group(1).replace("(", "[").replace(")", "]").replace(",]", "]")))
            self.assertEqual(registered, declared, name)

    def test_glm_keeps_tuned_and_untuned_history_apart(self):
        self.assertIn('variant="tuned" if tune and dense_tune_path().exists() else "untuned"', (TOOLS / "glm" / "serve.py").read_text())

    def test_compiler_report_does_not_import_the_engine_package(self):
        # The Qwen server sets its serving environment after startup_health.report_compiler(): importing exllamav3 there
        # latches the engine defaults and check_latched_env() refuses to start (found on the first GPU start of the merge).
        import subprocess
        code = ("import sys; sys.path.insert(0, %r); import startup_health as s; s.report_compiler(); h = s.compiler_health(); "
                "assert 'hip_compiler' in h, h; assert 'exllamav3' not in sys.modules and 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith(('exllamav3', 'torch'))); "
                "assert s._hip_compiler() is sys.modules['exllamav3.util.hip_compiler']") % str(TOOLS)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("HIP kernel compiler", r.stdout)

    def test_default_ready_line_unchanged(self):
        self.assertIn("qserve: READY on http://", (TOOLS / "qwen" / "serve.py").read_text())


if __name__ == "__main__":
    unittest.main()
