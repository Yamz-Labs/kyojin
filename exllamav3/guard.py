#!/usr/bin/env python3
"""
guard.py -- an external supervisor that SIGKILLs its child before the box dies.

Why this exists
---------------
On unified-memory boxes the GPU and host share one pool.  When that pool is exhausted
the kernel may starve rather than cleanly OOM-kill, costing a physical power cycle.
`torch.cuda.mem_get_info()` reports device/free numbers, not the host's ability to
service the kernel and background processes.  This supervisor watches
/proc/meminfo from the OUTSIDE, in a process that allocates nothing, and kills the
child while the host still has its configured floor.

The device side is handled by the in-process pre-flight guard: it caps each device at
that device's own free/total memory.  The supervisor owns only the host floor, because
that is the resource whose exhaustion hangs a unified-memory box.

Usage
-----
    python exllamav3/guard.py [options] -- <command> [args...]

Options
-------
    --floor-gib F          kill when MemAvailable < F GiB       (default 8 or
                           MEMGUARD_FLOOR)
    --interval S           poll period in seconds               (default 0.5)
    --warn-gib F           log a warning below this             (default floor + 8)
    --log PATH             also append the trace here
    --log-max-mb F         rotate --log to PATH.1 at F MiB        (default 16; 0 off)
    --grace S              seconds to wait after SIGKILL       (default 30)
    --sample-every S       append a host-memory sample to the log every S seconds
    --fadvise-dir DIR      release DIR/*.safetensors page cache periodically
    --fadvise-every S      page-cache release period            (default 5)
    --selftest             print what it would do

Exit code
---------
    the child's exit code, or 137 if guard killed it.

`util/memguard.py` remains a compatibility wrapper that execs this file without
importing the package (so it does not initialize torch or the allocator).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

# Import the stdlib-only shared floor policy without importing the package/torch.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from guard_common import host_floor_from_env

GIB = 1024.0 * 1024.0 * 1024.0


def meminfo() -> dict[str, int]:
    """Return /proc/meminfo in bytes. Cheap: one read of a ~1.5 kB pseudo-file."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding = "ascii") as f:
            for line in f:
                key, _, rest = line.partition(":")
                rest = rest.strip()
                if rest.endswith(" kB"):
                    out[key] = int(rest[:-3]) * 1024
                else:
                    try:
                        out[key] = int(rest)
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def fmt_gib(x: float) -> str:
    return f"{x / GIB:6.1f}"


def proc_rss(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding = "ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _open_log(path: str):
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "a", buffering = 1)


class Logger:
    def __init__(self, path: str | None, max_bytes: int = 16 * 1024 * 1024):
        self.path = path
        self.max_bytes = max(0, int(max_bytes))
        self.size = 0
        self.fh = None
        if path:
            try:
                self.size = os.path.getsize(path)
            except OSError:
                self.size = 0
            try:
                self.fh = _open_log(path)
            except OSError as exc:
                print(f"[guard] cannot open log {path}: {exc}", file = sys.stderr, flush = True)

    def _rotate(self) -> None:
        if not self.fh or not self.path:
            return
        self.fh.close()
        try:
            os.replace(self.path, self.path + ".1")
        except OSError as exc:
            # A failed rename (PATH.1 is a directory, a pinned file, ...) must not
            # defeat the cap: truncate the log instead of growing without bound.
            print(f"[guard] could not rotate {self.path} ({exc}); truncating log",
                  file = sys.stderr, flush = True)
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(self.path, flags, 0o600)
                self.fh = os.fdopen(fd, "a", buffering = 1)
                self.size = 0
            except OSError as reopen_exc:
                self.fh = None
                print(f"[guard] cannot reopen log {self.path}: {reopen_exc}",
                      file = sys.stderr, flush = True)
            return
        try:
            self.fh = _open_log(self.path)
            self.size = 0
        except OSError as exc:
            self.fh = None
            print(f"[guard] cannot reopen log {self.path}: {exc}", file = sys.stderr, flush = True)

    def __call__(self, msg: str) -> None:
        line = f"[guard {time.strftime('%H:%M:%S')}] {msg}"
        print(line, file = sys.stderr, flush = True)
        if self.fh:
            data = line + "\n"
            if self.max_bytes and self.size + len(data) > self.max_bytes:
                self._rotate()
            if self.fh:
                self.fh.write(data)
                self.size += len(data)


def kill_tree(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def main() -> int:
    default_floor = host_floor_from_env()
    ap = argparse.ArgumentParser(add_help = True)
    ap.add_argument("--floor-gib", type = float, default = default_floor)
    ap.add_argument("--warn-gib", type = float, default = None)
    ap.add_argument("--interval", type = float, default = 0.5)
    ap.add_argument("--log", type = str, default = None)
    ap.add_argument("--log-max-mb", type = float, default = 16.0,
                    help = "rotate --log to PATH.1 when it reaches this size (0 disables)")
    ap.add_argument("--grace", type = float, default = 30.0)
    ap.add_argument("--label", type = str, default = None)
    ap.add_argument("--sample-every", type = float, default = 0.0,
                    help = "append a MemAvailable/MemFree host sample to --log this often")
    ap.add_argument(
        "--fadvise-dir",
        action = "append",
        default = [],
        help = "model dir whose *.safetensors should be dropped from page cache periodically "
               "while the child runs. exllamav3 reads shards with buffered pread, so on "
               "unified memory the shards end up resident twice -- once on the device, once "
               "in page cache -- and the page cache half grows DURING the load, i.e. before "
               "any in-process hook could run.",
    )
    ap.add_argument("--fadvise-every", type = float, default = 5.0)
    ap.add_argument("--selftest", action = "store_true")
    ap.add_argument("cmd", nargs = argparse.REMAINDER)
    args = ap.parse_args()

    log = Logger(args.log, max_bytes = int(max(0.0, args.log_max_mb) * 1024 ** 2))

    if args.selftest:
        mi = meminfo()
        log(
            f"selftest: MemAvailable {fmt_gib(mi.get('MemAvailable', 0))} GiB  "
            f"MemFree {fmt_gib(mi.get('MemFree', 0))} GiB  Cached {fmt_gib(mi.get('Cached', 0))} GiB  "
            f"floor {args.floor_gib} GiB -> would "
            f"{'KILL' if mi.get('MemAvailable', 0) < args.floor_gib * GIB else 'run'}"
        )
        return 0

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given (use: guard.py [opts] -- cmd args...)")

    floor = args.floor_gib * GIB
    warn = (args.warn_gib if args.warn_gib is not None else args.floor_gib + 8.0) * GIB

    mi = meminfo()
    label = args.label or os.path.basename(cmd[-1] if len(cmd) > 1 else cmd[0])
    log(
        f"start [{label}] host floor={args.floor_gib:.1f} GiB every {args.interval}s; "
        f"MemAvailable {fmt_gib(mi.get('MemAvailable', 0))} GiB, "
        f"Cached {fmt_gib(mi.get('Cached', 0))} GiB"
    )
    log("cmd: " + " ".join(cmd))

    if mi.get("MemAvailable", 0) < floor:
        log(
            f"REFUSING TO START: MemAvailable {fmt_gib(mi.get('MemAvailable', 0))} GiB is already "
            f"below the {args.floor_gib:.1f} GiB floor"
        )
        return 3

    t0 = time.time()
    child = subprocess.Popen(cmd, start_new_session = True)

    lowest = mi.get("MemAvailable", 0)
    peak_rss = 0
    last_warn = 0.0
    last_beat = 0.0
    last_fadv = 0.0
    last_sample = 0.0
    killed = False

    fadvise_files: list[str] = []
    for d in args.fadvise_dir:
        if os.path.isdir(d):
            fadvise_files += sorted(
                os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors")
            )
        elif os.path.isfile(d):
            fadvise_files.append(d)
    if fadvise_files:
        log(
            f"page-cache release armed for {len(fadvise_files)} shard(s) every "
            f"{args.fadvise_every}s"
        )

    def fadvise_now() -> None:
        for path in fadvise_files:
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            finally:
                os.close(fd)

    try:
        while True:
            rc = child.poll()
            if rc is not None:
                break
            mi = meminfo()
            avail = mi.get("MemAvailable", 0)
            lowest = min(lowest, avail)
            rss = proc_rss(child.pid)
            peak_rss = max(peak_rss, rss)
            now = time.time()

            if args.sample_every > 0 and now - last_sample >= args.sample_every:
                last_sample = now
                log(f"sample t+{now - t0:.1f}s MemAvailable={avail / GIB:.2f} "
                    f"MemFree={mi.get('MemFree', 0) / GIB:.2f} "
                    f"Cached={mi.get('Cached', 0) / GIB:.2f} "
                    f"Dirty={mi.get('Dirty', 0) / GIB:.2f} "
                    f"Unevictable={mi.get('Unevictable', 0) / GIB:.2f}")

            if avail < floor:
                log(
                    "*** MEMORY FLOOR BREACHED -- KILLING CHILD ***\n"
                    f"    MemAvailable {fmt_gib(avail)} GiB < floor {args.floor_gib:.1f} GiB\n"
                    f"    MemFree {fmt_gib(mi.get('MemFree', 0))} GiB  "
                    f"Cached {fmt_gib(mi.get('Cached', 0))} GiB  "
                    f"Dirty {fmt_gib(mi.get('Dirty', 0))} GiB  "
                    f"Unevictable {fmt_gib(mi.get('Unevictable', 0))} GiB\n"
                    f"    child pid {child.pid} RSS {fmt_gib(rss)} GiB, alive {now - t0:.1f}s\n"
                    "    Killing now, before the unified pool starves the kernel."
                )
                killed = True
                kill_tree(child.pid, signal.SIGKILL)
                try:
                    with open("/proc/sys/vm/drop_caches", "w", encoding = "ascii") as f:
                        f.write("1\n")
                    log("    dropped clean page cache (vm.drop_caches=1)")
                except OSError:
                    pass
                break

            if fadvise_files and now - last_fadv >= args.fadvise_every:
                last_fadv = now
                cached_before = mi.get("Cached", 0)
                fadvise_now()
                after = meminfo()
                freed = cached_before - after.get("Cached", 0)
                if freed > 2 * GIB:
                    log(
                        f"page-cache release: Cached {fmt_gib(cached_before)} -> "
                        f"{fmt_gib(after.get('Cached', 0))} GiB, "
                        f"MemAvailable {fmt_gib(after.get('MemAvailable', 0))} GiB"
                    )

            if avail < warn and now - last_warn > 10.0:
                last_warn = now
                log(
                    f"WARNING MemAvailable {fmt_gib(avail)} GiB (floor {args.floor_gib:.1f}), "
                    f"Cached {fmt_gib(mi.get('Cached', 0))}, child RSS {fmt_gib(rss)}"
                )
            elif now - last_beat > 60.0:
                last_beat = now
                log(
                    f"ok  MemAvailable {fmt_gib(avail)} GiB  Cached {fmt_gib(mi.get('Cached', 0))} GiB  "
                    f"child RSS {fmt_gib(rss)} GiB  t+{now - t0:.0f}s"
                )

            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("interrupted -- terminating child")
        kill_tree(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout = 10)
        except subprocess.TimeoutExpired:
            kill_tree(child.pid, signal.SIGKILL)
        return 130

    if killed:
        deadline = time.time() + args.grace
        while time.time() < deadline:
            if child.poll() is not None:
                break
            time.sleep(0.2)
        mi = meminfo()
        log(
            f"child killed. MemAvailable now {fmt_gib(mi.get('MemAvailable', 0))} GiB. "
            f"lowest seen {fmt_gib(lowest)} GiB, peak child RSS {fmt_gib(peak_rss)} GiB, "
            f"wall {time.time() - t0:.1f}s"
        )
        return 137

    rc = child.wait()
    mi = meminfo()
    log(
        f"child exited rc={rc} after {time.time() - t0:.1f}s; "
        f"lowest MemAvailable seen {fmt_gib(lowest)} GiB, peak child RSS {fmt_gib(peak_rss)} GiB; "
        f"MemAvailable now {fmt_gib(mi.get('MemAvailable', 0))} GiB"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
