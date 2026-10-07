"""Start-up progress for the Kyojin servers: GET /health answers while the model is still loading.

The servers open their HTTP port only after the model is loaded, which can take minutes (GLM tunes kernels for
8-20 minutes on a first start). This module opens a small stdlib listener on the same host and port before the load,
answers every request with 503, and answers GET /health with the real load progress:

    200 {"status": "ok", "source": "kyojin"}                      ready (the server's own /health takes over)
    503 {"status": "loading", "source": "kyojin", "message": "Target weights, 40 % (stage 1 of 4)", "progress": 0.31, "stage": "target weights", "stage_index": 1, "stage_count": 4,
         "stage_progress": 0.4, "elapsed_s": 52.1, "eta_s": null, "stage_eta_s": 78.0, "progress_basis": "stages"}
    500 {"status": "error", "source": "kyojin", "message": "..."}

Every reply carries `"source": "kyojin"` (a hint for parsers); while loading, `message` is one plain line for a UI
built from the other fields (stage name, percent, stage x of y, about N s left when an ETA exists).
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
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOADING_BODY = {"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}}
ACTIVE: "StartupTracker | None" = None


def error_body(code: int, message: str) -> dict:
    """The JSON error shape of the servers (OpenAI style)."""
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


def route_methods(routes: dict[str, tuple[str, ...]], path: str) -> tuple[str, ...] | None:
    """Allowed methods for `path` in a route table (`{x}` matches one segment), or None when the path is unknown."""
    parts = path.rstrip("/").split("/")
    for pattern, methods in routes.items():
        pp = pattern.rstrip("/").split("/")
        if len(pp) == len(parts) and all(a == b or (a.startswith("{") and a.endswith("}") and b) for a, b in zip(pp, parts)):
            return methods
    return None


def json_errors_middleware():
    """aiohttp middleware: 404, 405 and the other HTTP errors raised by the router answer in the JSON error shape
    (aiohttp's own reply is plain text). Imported lazily: the rest of this module needs no aiohttp."""
    from aiohttp import web

    @web.middleware
    async def middleware(request, handler):
        try:
            return await handler(request)
        except web.HTTPException as exc:
            if exc.status < 400:
                raise
            if exc.status == 404:
                message = f"not found: {request.method} {request.path}"
            elif exc.status == 405:
                message = f"method {request.method} not allowed on {request.path} (allowed: {', '.join(a.strip() for a in exc.headers.get('Allow', '').split(','))})"
            else:
                # aiohttp's default body is "<status>: <reason>"; the JSON message is the reason alone
                message = exc.reason if exc.text in (None, "", f"{exc.status}: {exc.reason}") else exc.text
            headers = {"Allow": exc.headers["Allow"]} if "Allow" in exc.headers else None
            return web.json_response(error_body(exc.status, message), status=exc.status, headers=headers)
    return middleware


def history_dir() -> Path:
    return Path(os.environ.get("KYOJIN_HEALTH_DIR") or "~/.cache/kyojin").expanduser()


def _loading_message(stage: str, index: int, count: int, frac, progress: float, eta, stage_eta=None) -> str:
    """One plain line for a UI, built only from the fields of the same reply."""
    what = stage[0].upper() + stage[1:]
    pct = f"{round((frac if frac is not None else progress) * 100)} %"
    text = f"{what}, {pct} (stage {index} of {count})" if frac is not None else f"{what} (stage {index} of {count}), {pct} overall"
    if eta is not None:
        text += f", about {round(eta)} s left"
    elif stage_eta is not None:
        text += f", about {round(stage_eta)} s left in this stage"
    return text


class StartupTracker:
    """Thread-safe record of the start-up stages. Stage indices are 1-based."""

    def __init__(self, name: str, stages: list[str], clock=time.monotonic, history: Path | None = None,
                 variant: str | None = None):
        if not stages:
            raise ValueError("at least one stage")
        self.name, self.stages, self.clock = name, list(stages), clock
        # `variant` keeps the durations of different kinds of start apart (e.g. cached kernel tuning or not), so a
        # fast start is never timed with the history of a slow one.
        suffix = f"-{variant}" if variant else ""
        self.history_path = (history if history is not None else history_dir() / f"startup-{name}{suffix}.json")
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
        self.routes: dict[str, tuple[str, ...]] | None = None

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
                return 200, {"status": "ok", "source": "kyojin"}
            if self.state == "error":
                return 500, {"status": "error", "source": "kyojin", "message": self.message, "elapsed_s": elapsed}
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
            # Pace of this run in the current stage: an estimate for that stage alone, whatever the history says.
            stage_eta = stage_elapsed * (1 - frac) / frac if frac is not None and frac >= 0.02 and stage_elapsed > 0 else None
            progress = round(min(progress, 0.999), 3)
            return 503, {"status": "loading", "source": "kyojin",
                         "message": _loading_message(self.stages[cur], cur + 1, n, frac, progress, eta, stage_eta),
                         "progress": progress,
                         "stage": self.stages[cur], "stage_index": cur + 1, "stage_count": n,
                         "stage_progress": None if frac is None else round(frac, 3),
                         "elapsed_s": elapsed, "eta_s": None if eta is None else round(eta, 1),
                         "stage_eta_s": None if stage_eta is None else round(stage_eta, 1),
                         "progress_basis": basis}

    # ---- early listener

    def serve(self, host: str, port: int) -> None:
        self.server = _EarlyServer(self, host, port)

    def release_port(self) -> None:
        """Close the early listener (tests and the failure path). The servers use `handoff_socket` instead."""
        if self.server is not None:
            self.server.close()
            self.server = None

    def handoff_socket(self) -> socket.socket | None:
        """Stop the early accept loop and give the still-listening socket to the real server.
        The port never closes, so no connection is refused: clients that connect during the hand-off wait in the
        listen backlog until the real server accepts them."""
        if self.server is None:
            return None
        sock = self.server.detach()
        self.server = None
        return sock


class _EarlyServer:
    def __init__(self, tracker: StartupTracker, host: str, port: int):
        class Handler(BaseHTTPRequestHandler):
            def _reply(self):
                path = self.path.split("?", 1)[0]
                routes = getattr(tracker, "routes", None)
                allowed = route_methods(routes, path) if routes is not None else None
                extra = None
                if path.rstrip("/") == "/health" and self.command in ("GET", "HEAD"):
                    code, body = tracker.snapshot()
                elif allowed is None and routes is not None:
                    code, body = 404, error_body(404, f"not found: {self.command} {path}")
                elif allowed is not None and self.command not in allowed and not (self.command == "HEAD" and "GET" in allowed):
                    code, body = 405, error_body(405, f"method {self.command} not allowed on {path} (allowed: {', '.join(allowed)})")
                    extra = ("Allow", ", ".join(allowed))
                else:
                    code, body = 503, LOADING_BODY
                data = json.dumps(body).encode()
                try:
                    self.send_response(code)
                    if extra:
                        self.send_header(*extra)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Retry-After", "5")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass                                # the client gave up (poll timeout): nothing to report

            do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = _reply

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 128   # clients arriving during the hand-off wait in this backlog

        self.httpd = Server((host, port), Handler)
        self.port = self.httpd.server_address[1]
        # A fork-without-exec child would keep the listening socket open after close() and block the real bind.
        sock = self.httpd.socket
        self._owned = True
        os.register_at_fork(after_in_child=lambda: sock.close() if self._owned else None)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="startup-health")
        self.thread.start()

    def detach(self) -> socket.socket:
        """Stop answering but keep the listening socket open and return it (the caller now owns it)."""
        self._owned = False
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        return self.httpd.socket

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


# ---- module-level helpers used by the servers (no-ops without an active tracker)

def check_model_dir(name: str, model_dir, needed: tuple[str, ...] = ("config.json", "chat_template.jinja")) -> None:
    """Exit with one plain line (no traceback) when the pack folder is missing or lacks a file every start needs."""
    d = Path(model_dir).expanduser()
    if not d.is_dir():
        raise SystemExit(f"{name}: model folder not found: {d} (point --model at the downloaded pack folder)")
    missing = [f for f in needed if not (d / f).is_file()]
    if missing:
        raise SystemExit(f"{name}: {d} is not a complete pack, missing: {', '.join(missing)} (download it again)")


def start(name: str, stages: list[str], host: str, port: int, variant: str | None = None,
          routes: dict[str, tuple[str, ...]] | None = None) -> StartupTracker:
    """Create the tracker and open the early listener. Fails at once when the port is taken.
    `routes` maps each path of the server (`{x}` is a wildcard segment) to its allowed methods: while loading, a path
    outside it answers 404 and a wrong method 405 (JSON), the real endpoints answer 503."""
    global ACTIVE
    ACTIVE = StartupTracker(name, stages, variant=variant)
    ACTIVE.routes = routes
    try:
        ACTIVE.serve(host, port)
    except OSError as exc:
        ACTIVE = None
        raise SystemExit(f"{name}: cannot listen on {host}:{port}: {exc.strerror or exc} "
                         "(another server on this port? choose another --port)") from None
    return ACTIVE


def compiler_health() -> dict:
    """Fields for /health: the HIP kernel compiler line, plus the warning line when it is not the rocm-sdk compiler.
    Lazy import: nothing here needs torch at import time."""
    try:
        from exllamav3.util import hip_compiler
        out = {"hip_compiler": hip_compiler.describe().removeprefix("HIP kernel compiler: ")}
        if hip_compiler.warning():
            out["hip_compiler_warning"] = hip_compiler.warning()
        return out
    except Exception as e:  # noqa: BLE001
        return {"hip_compiler": f"unknown ({type(e).__name__})"}


def report_compiler() -> None:
    """Start-up log: which compiler builds the JIT kernels, plus one warning line when it is not the rocm-sdk one."""
    try:
        from exllamav3.util import hip_compiler
        hip_compiler.report(lambda line: print(line, flush=True))
    except Exception as e:  # noqa: BLE001
        print(f"[kyojin] HIP kernel compiler: unknown ({type(e).__name__}: {e})", flush=True)


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


def finish() -> socket.socket | None:
    """Models loaded: stop the early answers, mark ready, and return the listening socket for the real server
    (aiohttp `SockSite` / `run_app(sock=...)`). None without an early listener: the server then binds host/port."""
    if ACTIVE is None:
        return None
    sock = ACTIVE.handoff_socket()
    ACTIVE.ready()
    return sock


def abort(message: str, grace_s: float = 3.0) -> None:
    """Load failed: report it on /health for a moment before the process exits."""
    if ACTIVE is not None:
        ACTIVE.fail(message)
        time.sleep(grace_s)
