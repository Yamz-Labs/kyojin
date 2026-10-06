"""Start-up progress for the Kyojin servers: GET /health answers while the model is still loading.

The servers open their HTTP port only after the model is loaded, which can take minutes (GLM tunes kernels for
8-20 minutes on a first start). This module opens a small stdlib listener on the same host and port before the load,
answers every request with 503, and answers GET /health with the real load progress:

    200 {"status": "ok"}                                         ready (the server's own /health takes over)
    503 {"status": "loading", "progress": 0.31, "stage": "target weights", "stage_index": 1, "stage_count": 4,
         "stage_progress": 0.4, "elapsed_s": 52.1, "eta_s": null, "progress_basis": "stages"}
    500 {"status": "error", "message": "..."}

Honest values only: `stage_progress` is null when the stage cannot count its work; `progress` counts finished stages
plus the reported fraction of the current one (equal weights, or the durations of the previous start when it is
recorded: progress_basis "history"); `eta_s` is null unless a previous start recorded the stage durations
(then it adds the old durations of the stages to come, so a stage that is much faster the second time, e.g. cached
kernel tuning, makes it an upper bound).
Nothing here imports torch. The servers call `stage()`, `progress()` and `load_callback()`; all are no-ops when no
tracker is active (tests, library use).
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOADING_BODY = {"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}}
ACTIVE: "StartupTracker | None" = None


def history_dir() -> Path:
    return Path(os.environ.get("KYOJIN_HEALTH_DIR") or "~/.cache/kyojin").expanduser()


class StartupTracker:
    """Thread-safe record of the start-up stages. Stage indices are 1-based."""

    def __init__(self, name: str, stages: list[str], clock=time.monotonic, history: Path | None = None):
        if not stages:
            raise ValueError("at least one stage")
        self.name, self.stages, self.clock = name, list(stages), clock
        self.history_path = (history if history is not None else history_dir() / f"startup-{name}.json")
        self.lock = threading.Lock()
        self.t0 = clock()
        self.state = "loading"
        self.message = ""
        self.current = 0
        self.stage_t0 = self.t0
        self.fraction: float | None = None
        self.durations: dict[str, float] = {}
        self.expected = self._load_history()
        self.server: _EarlyServer | None = None

    def _load_history(self) -> dict[str, float] | None:
        try:
            data = json.loads(Path(self.history_path).read_text())
            exp = {s: float(data[s]) for s in self.stages}
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return exp if all(v > 0 for v in exp.values()) else None

    # ---- updates (called by the loader)

    def begin(self, stage: str) -> None:
        """Enter `stage`. Earlier stages count as finished; a stage that is skipped is simply finished."""
        with self.lock:
            idx = self.stages.index(stage)
            if idx <= self.current:
                return                                    # same stage again, or backwards: ignore
            now = self.clock()
            self.durations[self.stages[self.current]] = now - self.stage_t0
            self.current, self.stage_t0, self.fraction = idx, now, None

    def progress(self, done: float, total: float) -> None:
        if total and total > 0:
            with self.lock:
                self.fraction = min(max(done / total, 0.0), 1.0)

    def ready(self) -> None:
        with self.lock:
            now = self.clock()
            self.durations[self.stages[self.current]] = now - self.stage_t0
            self.state = "ready"
        self._save_history()

    def fail(self, message: str) -> None:
        with self.lock:
            self.state, self.message = "error", message

    def _save_history(self) -> None:
        if set(self.durations) != set(self.stages):
            return                                        # a run that skipped stages teaches nothing
        try:
            Path(self.history_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.history_path).write_text(json.dumps({k: round(v, 3) for k, v in self.durations.items()}))
        except OSError:
            pass

    # ---- reading

    def snapshot(self) -> tuple[int, dict]:
        with self.lock:
            now = self.clock()
            elapsed = round(now - self.t0, 1)
            if self.state == "ready":
                return 200, {"status": "ok"}
            if self.state == "error":
                return 500, {"status": "error", "message": self.message, "elapsed_s": elapsed}
            n, cur, frac = len(self.stages), self.current, self.fraction
            stage_elapsed = now - self.stage_t0
            exp = self.expected
            if exp:
                total = sum(exp.values())
                done = sum(exp[s] for s in self.stages[:cur]) + exp[self.stages[cur]] * (frac or 0.0)
                progress, basis = done / total, "history"
                if frac is not None and frac >= 0.02:     # this run's own pace beats the old duration
                    left = stage_elapsed * (1 - frac) / frac
                elif frac is not None:
                    left = exp[self.stages[cur]]
                else:
                    left = exp[self.stages[cur]] - stage_elapsed
                eta = sum(exp[s] for s in self.stages[cur + 1:]) + left if left >= 0 else None
            else:
                progress, basis, eta = (cur + (frac or 0.0)) / n, "stages", None
            return 503, {"status": "loading", "progress": round(min(progress, 0.999), 3),
                         "stage": self.stages[cur], "stage_index": cur + 1, "stage_count": n,
                         "stage_progress": None if frac is None else round(frac, 3),
                         "elapsed_s": elapsed, "eta_s": None if eta is None else round(eta, 1),
                         "progress_basis": basis}

    # ---- early listener

    def serve(self, host: str, port: int) -> None:
        self.server = _EarlyServer(self, host, port)

    def release_port(self) -> None:
        """Close the early listener; call right before the real server binds the same port."""
        if self.server is not None:
            self.server.close()
            self.server = None


class _EarlyServer:
    def __init__(self, tracker: StartupTracker, host: str, port: int):
        class Handler(BaseHTTPRequestHandler):
            def _reply(self):
                if self.path.split("?", 1)[0].rstrip("/") == "/health":
                    code, body = tracker.snapshot()
                else:
                    code, body = 503, LOADING_BODY
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Retry-After", "5")
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = _reply

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.httpd = Server((host, port), Handler)
        self.port = self.httpd.server_address[1]
        # A fork-without-exec child would keep the listening socket open after close() and block the real bind.
        sock = self.httpd.socket
        os.register_at_fork(after_in_child=lambda: sock.close())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="startup-health")
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


# ---- module-level helpers used by the servers (no-ops without an active tracker)

def start(name: str, stages: list[str], host: str, port: int) -> StartupTracker:
    """Create the tracker and open the early listener. Fails at once when the port is taken."""
    global ACTIVE
    ACTIVE = StartupTracker(name, stages)
    ACTIVE.serve(host, port)
    return ACTIVE


def stage(name: str) -> None:
    if ACTIVE is not None and name in ACTIVE.stages:
        ACTIVE.begin(name)


def progress(done: float, total: float) -> None:
    if ACTIVE is not None:
        ACTIVE.progress(done, total)


def load_callback(stages: list[str] | None = None):
    """Callback for Model.load(callback=...), called as (modules_done, modules_total) before each module.
    With `stages`, a drop of the counter means the next model started, and the next stage begins (several
    models loaded through one call, e.g. drafter then target in model_init.init)."""
    state = {"last": -1, "i": 0}

    def cb(done: int, total: int) -> None:
        if stages and done < state["last"]:
            state["i"] = min(state["i"] + 1, len(stages) - 1)
        state["last"] = done
        if stages:
            stage(stages[state["i"]])
        progress(done, total)
    if stages:
        stage(stages[0])
    return cb


def finish() -> None:
    """Models loaded and the real server about to bind: free the port, then mark ready."""
    if ACTIVE is not None:
        ACTIVE.release_port()
        ACTIVE.ready()


def abort(message: str, grace_s: float = 3.0) -> None:
    """Load failed: report it on /health for a moment before the process exits."""
    if ACTIVE is not None:
        ACTIVE.fail(message)
        time.sleep(grace_s)
