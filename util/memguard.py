#!/usr/bin/env python3
"""
memguard.py -- a tiny supervisor that SIGKILLs its child before the box dies.

Why this exists
---------------
On the DGX Spark (GB10) the GPU and the host share ONE 121.6 GiB pool. There is no
separate VRAM. When that pool is exhausted the kernel does not OOM-kill anything --
it starves, the NVIDIA driver logs NV_ERR_NO_MEMORY, and recovery is a physical
power cycle. On 2026-09-22 03:1x exactly that happened while loading the 87 GiB
EXL3 quant: the shards' pages stayed in page cache while the same bytes were also
pinned on the device.

`torch.cuda.mem_get_info()` reports MemFree, not MemAvailable, so nothing inside
the process notices. This supervisor watches /proc/meminfo from the OUTSIDE, in a
process that allocates nothing, and kills the child while there is still room to
recover.

Usage
-----
    python util/memguard.py [options] -- <command> [args...]

Options
-------
    --floor-gib F     kill when MemAvailable < F GiB          (default 8)
    --interval S      poll period in seconds                  (default 0.5)
    --warn-gib F      log a warning below this                (default floor + 8)
    --log PATH        also append the sampler trace here
    --grace S         seconds to wait after SIGKILL before giving up (default 30)
    --cgroup-limit    (informational) also print the child's peak RSS

Exit code
---------
    the child's exit code, or 137 if memguard killed it (and a loud explanation on
    stderr / in --log).

Self-test
---------
    python util/memguard.py --selftest
prints what it would do, and

    python util/memguard.py --floor-gib 999 -- sleep 60
must kill immediately (999 GiB floor can never be satisfied) -- that is the
"prove it kills" check.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

GIB = 1024.0 * 1024.0 * 1024.0


def meminfo() -> dict[str, int]:
    """Return /proc/meminfo in bytes. Cheap: one read of a ~1.5 kB pseudo-file."""
    out = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            k, _, rest = line.partition(":")
            rest = rest.strip()
            if rest.endswith(" kB"):
                out[k] = int(rest[:-3]) * 1024
            else:
                try:
                    out[k] = int(rest)
                except ValueError:
                    pass
    return out


def fmt_gib(x: int | float) -> str:
    return f"{x / GIB:6.1f}"


def proc_rss(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


class Logger:
    def __init__(self, path: str | None):
        self.fh = open(path, "a", buffering=1) if path else None

    def __call__(self, msg: str) -> None:
        line = f"[memguard {time.strftime('%H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        if self.fh:
            self.fh.write(line + "\n")


def kill_tree(pid: int, sig: int) -> None:
    """Signal the child's whole process group (it is a session leader, see start_new_session)."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--floor-gib", type=float, default=8.0)
    ap.add_argument("--warn-gib", type=float, default=None)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--grace", type=float, default=30.0)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument(
        "--fadvise-dir",
        action="append",
        default=[],
        help="model dir whose *.safetensors should be dropped from page cache periodically "
        "while the child runs. exllamav3 reads shards with buffered pread, so on unified "
        "memory the shards end up resident twice -- once on the device, once in page cache "
        "-- and the page cache half grows DURING the load, i.e. before any in-process hook "
        "could run. Doing it from the supervisor covers every loader, TabbyAPI included.",
    )
    ap.add_argument("--fadvise-every", type=float, default=5.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    log = Logger(args.log)

    if args.selftest:
        mi = meminfo()
        log(
            f"selftest: MemAvailable {fmt_gib(mi['MemAvailable'])} GiB  "
            f"MemFree {fmt_gib(mi['MemFree'])} GiB  Cached {fmt_gib(mi['Cached'])} GiB  "
            f"floor {args.floor_gib} GiB -> would "
            f"{'KILL' if mi['MemAvailable'] < args.floor_gib * GIB else 'run'}"
        )
        return 0

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given (use: memguard.py [opts] -- cmd args...)")

    floor = args.floor_gib * GIB
    warn = (args.warn_gib if args.warn_gib is not None else args.floor_gib + 8.0) * GIB

    mi = meminfo()
    label = args.label or os.path.basename(cmd[-1] if len(cmd) > 1 else cmd[0])
    log(
        f"start [{label}] floor={args.floor_gib:.1f} GiB every {args.interval}s; "
        f"MemAvailable {fmt_gib(mi['MemAvailable'])} GiB, Cached {fmt_gib(mi['Cached'])} GiB"
    )
    log("cmd: " + " ".join(cmd))

    if mi["MemAvailable"] < floor:
        log(
            f"REFUSING TO START: MemAvailable {fmt_gib(mi['MemAvailable'])} GiB is already "
            f"below the {args.floor_gib:.1f} GiB floor"
        )
        return 3

    t0 = time.time()
    # start_new_session so the child gets its own process group; a model load spawns
    # worker threads/processes and we want to kill all of them at once.
    child = subprocess.Popen(cmd, start_new_session=True)

    lowest = mi["MemAvailable"]
    peak_rss = 0
    last_warn = 0.0
    last_beat = 0.0
    last_fadv = 0.0
    fadv_total = 0
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

    def fadvise_now() -> int:
        n = 0
        for p in fadvise_files:
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            except OSError:
                pass
            finally:
                os.close(fd)
        return n

    try:
        while True:
            rc = child.poll()
            if rc is not None:
                break
            mi = meminfo()
            avail = mi["MemAvailable"]
            lowest = min(lowest, avail)
            rss = proc_rss(child.pid)
            peak_rss = max(peak_rss, rss)
            now = time.time()

            if avail < floor:
                log(
                    "*** MEMORY FLOOR BREACHED -- KILLING CHILD ***\n"
                    f"    MemAvailable {fmt_gib(avail)} GiB < floor {args.floor_gib:.1f} GiB\n"
                    f"    MemFree {fmt_gib(mi['MemFree'])} GiB  Cached {fmt_gib(mi['Cached'])} GiB  "
                    f"Dirty {fmt_gib(mi.get('Dirty', 0))} GiB\n"
                    f"    child pid {child.pid} RSS {fmt_gib(rss)} GiB, alive {now - t0:.1f}s\n"
                    "    This box has unified memory and does NOT OOM-kill; starving it costs a "
                    "physical power cycle. Killing now."
                )
                killed = True
                kill_tree(child.pid, signal.SIGKILL)
                # Drop clean page cache immediately -- it is the cheapest memory to get back
                # and it is usually what is squeezing us.
                try:
                    with open("/proc/sys/vm/drop_caches", "w") as f:
                        f.write("1\n")
                    log("    dropped clean page cache (vm.drop_caches=1)")
                except OSError:
                    pass
                break

            if fadvise_files and now - last_fadv >= args.fadvise_every:
                last_fadv = now
                cached_before = mi["Cached"]
                fadvise_now()
                fadv_total += 1
                after = meminfo()
                freed = cached_before - after["Cached"]
                if freed > 2 * GIB:
                    log(
                        f"page-cache release: Cached {fmt_gib(cached_before)} -> "
                        f"{fmt_gib(after['Cached'])} GiB, MemAvailable {fmt_gib(after['MemAvailable'])} GiB"
                    )

            if avail < warn and now - last_warn > 10.0:
                last_warn = now
                log(
                    f"WARNING MemAvailable {fmt_gib(avail)} GiB (floor {args.floor_gib:.1f}), "
                    f"Cached {fmt_gib(mi['Cached'])}, child RSS {fmt_gib(rss)}"
                )
            elif now - last_beat > 60.0:
                last_beat = now
                log(
                    f"ok  MemAvailable {fmt_gib(avail)} GiB  Cached {fmt_gib(mi['Cached'])} GiB  "
                    f"child RSS {fmt_gib(rss)} GiB  t+{now - t0:.0f}s"
                )

            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("interrupted -- terminating child")
        kill_tree(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
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
            f"child killed. MemAvailable now {fmt_gib(mi['MemAvailable'])} GiB. "
            f"lowest seen {fmt_gib(lowest)} GiB, peak child RSS {fmt_gib(peak_rss)} GiB, "
            f"wall {time.time() - t0:.1f}s"
        )
        return 137

    rc = child.wait()
    mi = meminfo()
    log(
        f"child exited rc={rc} after {time.time() - t0:.1f}s; "
        f"lowest MemAvailable seen {fmt_gib(lowest)} GiB, peak child RSS {fmt_gib(peak_rss)} GiB; "
        f"MemAvailable now {fmt_gib(mi['MemAvailable'])} GiB"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
