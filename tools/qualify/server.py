"""Server lifecycle: start under the caller's locks, wait for GET /health == 200 (never a port check: the port opens ~120 s before the
load ends), scan the log for OOM, kill by PID only (no pkill)."""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def health(port: int) -> int:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status
    except urllib.error.HTTPError as e:      # 503 while loading
        return e.code
    except Exception:
        return 0


def pick_port(preferred: int, span: int = 200) -> int:
    """First port >= preferred that nothing listens on (other workers' servers run on 859x too: a fixed port made a soak 'not start')."""
    import socket
    for p in range(preferred, preferred + span):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free port in {preferred}..{preferred + span}")


def own_ready(log: Path, port: int) -> bool:
    """The server we spawned printed 'READY on http://127.0.0.1:<port>' (tools/qwen/serve.py); a foreign server answering /health does not count."""
    try:
        return f"READY on http://127.0.0.1:{port}" in log.read_text(errors="replace")
    except OSError:
        return False


def oom_in(log: Path) -> bool:
    try:
        t = log.read_text(errors="replace").lower()
    except OSError:
        return False
    return "out of memory" in t or "hip out of memory" in t


@contextlib.contextmanager
def running_server(prof: dict, repo: Path, argv: list[str], port: int, log: Path, ready_s: int = 1500, logf=print, args: dict | None = None):
    """Yield (proc, started) when /health is 200 AND the log says this very process is READY on its port; started=False (proc exited or timeout)
    lets the caller treat it as 'does not fit'.
    The port is re-picked HERE, right before the spawn (after the GPU lock wait: another worker's server may have taken the one chosen earlier,
    and a /health 200 from THEIR server must never pass for ours: 27/10 an autotune sweep went to a foreign server). `args['port']` follows."""
    new = pick_port(port)
    if new != port and "--port" in argv:
        argv = list(argv); argv[argv.index("--port") + 1] = str(new)
    port = new
    if args is not None:
        args["port"] = port
    env = {**os.environ, **prof["env"]}
    fh = open(log, "wb")
    p = subprocess.Popen([prof["python"], "-u", *argv], cwd=repo, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    ok = False
    t0 = time.time()
    try:
        while time.time() - t0 < ready_s:
            if p.poll() is not None:
                break
            if health(port) == 200 and own_ready(log, port):
                ok = True; break
            time.sleep(3)
        logf(f"server pid {p.pid} {'ready' if ok else 'NOT ready'} after {time.time()-t0:.0f} s")
        yield p, ok
    finally:
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGTERM)       # own process group, scoped to our PID
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait(30)
        fh.close()
