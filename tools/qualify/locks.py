"""GPU locks: one holder per job, flock waits, released between jobs. Never blocks another worker longer than one job."""
from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import time

MIN_MEM_GIB = 4.0
LIGHT_GIB = 4.0     # an untimed step whose server needs less device memory than this takes no exclusive GPU lock


def _blocking_flock(fd: int, wait_s: float, path: str) -> None:
    def on_alarm(*_):
        raise TimeoutError(f"lock {path} not free after {wait_s:.0f} s")
    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(int(wait_s))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except BaseException:
        os.close(fd)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@contextlib.contextmanager
def gpu_locks(paths: list[str], log=print, wait_s: float = 6 * 3600):
    """Take every lock in order (flock; a busy lock is waited on in the kernel queue); release in reverse order."""
    fds = []
    try:
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
            t0 = time.time()
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Blocking wait (kernel queue), not a poll: a poller loses every release to the `flock -w` waiters queued
                # beside it (autotune0 sat 4 h behind blocking waiters with a 5 s poll)
                log(f"waiting for lock {p}")
                _blocking_flock(fd, wait_s, p)
            fds.append(fd)
            log(f"lock held {p} (waited {time.time()-t0:.0f} s)")
        yield
    finally:
        for fd in reversed(fds):
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
        log("locks released")
