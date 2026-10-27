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


def oom_in(log: Path) -> bool:
    try:
        t = log.read_text(errors="replace").lower()
    except OSError:
        return False
    return "out of memory" in t or "hip out of memory" in t


@contextlib.contextmanager
def running_server(prof: dict, repo: Path, argv: list[str], port: int, log: Path, ready_s: int = 1500, logf=print):
    """Yield (proc, started) when /health is 200; started=False (proc exited or timeout) lets the caller treat it as 'does not fit'."""
    env = {**os.environ, **prof["env"]}
    fh = open(log, "wb")
    p = subprocess.Popen([prof["python"], "-u", *argv], cwd=repo, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    ok = False
    t0 = time.time()
    try:
        while time.time() - t0 < ready_s:
            if p.poll() is not None:
                break
            if health(port) == 200:
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
