#!/usr/bin/env python3
"""
guard.py -- an external supervisor that SIGKILLs its child before the box dies.

Why this exists
---------------
On unified-memory boxes (DGX Spark/GB10, Strix Halo) the GPU and the host share one
pool. There is no separate VRAM, and when the pool is exhausted the kernel can starve
rather than OOM-kill, costing a physical power cycle. `torch.cuda.mem_get_info()`
reports MemFree, not MemAvailable, so nothing inside the process notices. This
supervisor watches /proc/meminfo and the per-device free memory from the OUTSIDE, in a
process that allocates nothing, and kills the child while there is still room to
recover.

It is also the page-cache release: exllamav3 reads shards with buffered pread, so on
unified memory the shards end up resident twice -- once on the device, once in page
cache -- and the page cache half grows DURING the load, before any in-process hook
could run.

Usage
-----
    python exllamav3/guard.py [options] -- <command> [args...]

Options
-------
    --floor-gib F          kill when host MemAvailable < F GiB       (default 6)
    --device-floor-gib F   kill when any watched device free < F GiB (default 0 = off)
    --device-floor SPEC    per-device floors, e.g. card1=4.5,nvidia:0=2
    --interval S           poll period in seconds                    (default 0.5)
    --warn-gib F           log a warning below this                  (default floor + 8)
    --log PATH             also append the sampler trace here
    --grace S              seconds to wait after SIGKILL            (default 30)
    --fadvise-dir DIR      release DIR/*.safetensors page cache periodically
    --fadvise-every S      page-cache release period                 (default 5)
    --selftest             print what it would do

Exit code
---------
    the child's exit code, or 137 if guard killed it (and a loud explanation on
    stderr / in --log).

`util/memguard.py` remains as a compatibility wrapper that execs this file without
importing the package (so it does not initialize torch or the allocator).
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
import sys
import time

GIB = 1024.0 * 1024.0 * 1024.0


def meminfo() -> dict[str, int]:
    """Return /proc/meminfo in bytes. Cheap: one read of a ~1.5 kB pseudo-file."""
    out = {}
    try:
        with open("/proc/meminfo", "r", encoding = "ascii") as f:
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


class Logger:
    def __init__(self, path: str | None):
        self.fh = open(path, "a", buffering = 1) if path else None  # noqa: SIM115

    def __call__(self, msg: str) -> None:
        line = f"[guard {time.strftime('%H:%M:%S')}] {msg}"
        print(line, file = sys.stderr, flush = True)
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


def _read_int(path: str) -> int | None:
    try:
        with open(path, "r", encoding = "ascii") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def amd_device_free() -> dict[str, int]:
    """Free memory of every AMD DRM card, in bytes.

    An APU's unified pool is the GTT aperture, whose total is close to host RAM.  A
    discrete card also exposes ``mem_info_gtt_*`` (a host-RAM aperture), but its real
    budget is VRAM, so choose by pool size rather than by file presence.
    """
    out: dict[str, int] = {}
    host_total = meminfo().get("MemTotal", 0)
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        device = os.path.join(card, "device")
        name = os.path.basename(card)
        gtt_total = _read_int(os.path.join(device, "mem_info_gtt_total"))
        gtt_used = _read_int(os.path.join(device, "mem_info_gtt_used"))
        vram_total = _read_int(os.path.join(device, "mem_info_vram_total"))
        vram_used = _read_int(os.path.join(device, "mem_info_vram_used"))
        total = used = None
        if gtt_total is not None and host_total and gtt_total >= 0.9 * host_total:
            total, used = gtt_total, gtt_used
        elif vram_total is not None:
            total, used = vram_total, vram_used
        elif gtt_total is not None:
            total, used = gtt_total, gtt_used
        if total is not None and used is not None:
            out[name] = max(0, total - used)
    return out


_NVIDIA_CACHE: dict[str, object] = {"t": -1e9, "data": {}, "unavailable": False,
                                    "last_try": -1e9}
NVIDIA_QUERY_INTERVAL_S = 5.0
NVIDIA_RETRY_S = 60.0


def nvidia_device_free(now: float | None = None, force: bool = False) -> dict[str, int]:
    """Cached NVIDIA query; ``nvidia-smi`` is not spawned on every supervisor poll."""
    now = time.monotonic() if now is None else now
    if not force:
        if _NVIDIA_CACHE["unavailable"] and now - float(_NVIDIA_CACHE["last_try"]) < NVIDIA_RETRY_S:
            return {}
        if now - float(_NVIDIA_CACHE["t"]) < NVIDIA_QUERY_INTERVAL_S:
            return dict(_NVIDIA_CACHE["data"])
    _NVIDIA_CACHE["last_try"] = now
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output = True, text = True, timeout = 5, check = False)
    except (OSError, subprocess.TimeoutExpired):
        _NVIDIA_CACHE.update(unavailable = True, t = now, data = {})
        return {}
    if result.returncode != 0:
        _NVIDIA_CACHE.update(unavailable = True, t = now, data = {})
        return {}
    out: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            index, total_mib, used_mib = int(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        out[f"nvidia:{index}"] = int(max(0.0, total_mib - used_mib) * 1024 * 1024)
    _NVIDIA_CACHE.update(unavailable = False, t = now, data = out)
    return dict(out)


def device_free(floors: dict[str, float] | None = None, now: float | None = None,
                force_nvidia: bool = False) -> dict[str, int]:
    out = amd_device_free()
    wanted = floors or {}
    want_nvidia = force_nvidia or not wanted or any(
        key.startswith("nvidia") or key in ("default", "nvidia") for key in wanted)
    if want_nvidia:
        out.update(nvidia_device_free(now = now, force = force_nvidia))
    return out


def parse_device_floors(specs: list[str] | None, global_gib: float) -> dict[str, float]:
    """Parse ``card1=4.5,nvidia:0=2`` plus a global fallback, in GiB."""
    floors: dict[str, float] = {}
    for spec in specs or []:
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            key, sep, raw = item.partition("=")
            if not sep:
                raise SystemExit(f"--device-floor expected selector=GiB, got {item!r}")
            try:
                floors[key.strip().lower()] = float(raw)
            except ValueError as exc:
                raise SystemExit(f"--device-floor bad value for {key!r}: {raw!r}") from exc
    if global_gib > 0:
        floors.setdefault("default", float(global_gib))
    return floors


def floor_for(name: str, floors: dict[str, float]) -> float:
    key = name.lower()
    if key in floors:
        return floors[key]
    if key.startswith("nvidia:") and "nvidia" in floors:
        return floors["nvidia"]
    if key.startswith("card") and "amd" in floors:
        return floors["amd"]
    return floors.get("default", 0.0)


def effective_floor(name: str, floors: dict[str, float], multiple: bool) -> float:
    """A global floor is not safe on a multi-device host: use explicit selectors."""
    key = name.lower()
    if key in floors:
        return floors[key]
    if key.startswith("nvidia:") and "nvidia" in floors:
        return floors["nvidia"]
    if key.startswith("card") and "amd" in floors:
        return floors["amd"]
    return 0.0 if multiple else floors.get("default", 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(add_help = True)
    ap.add_argument("--floor-gib", type = float, default = 6.0)
    ap.add_argument("--warn-gib", type = float, default = None)
    ap.add_argument("--interval", type = float, default = 0.5)
    ap.add_argument("--log", type = str, default = None)
    ap.add_argument("--grace", type = float, default = 30.0)
    ap.add_argument("--label", type = str, default = None)
    ap.add_argument("--device-floor-gib", type = float, default = 0.0,
                    help = "free-memory floor for every watched device, in GiB (0 = off)")
    ap.add_argument("--device-floor", action = "append", default = [],
                    help = "per-device floors, e.g. card1=4.5,nvidia:0=2")
    ap.add_argument(
        "--fadvise-dir",
        action = "append",
        default = [],
        help = "model dir whose *.safetensors should be dropped from page cache periodically "
               "while the child runs. exllamav3 reads shards with buffered pread, so on unified "
               "memory the shards end up resident twice -- once on the device, once in page "
               "cache -- and the page cache half grows DURING the load, i.e. before any "
               "in-process hook could run. Doing it from the supervisor covers every loader.",
    )
    ap.add_argument("--fadvise-every", type = float, default = 5.0)
    ap.add_argument("--selftest", action = "store_true")
    ap.add_argument("cmd", nargs = argparse.REMAINDER)
    args = ap.parse_args()

    log = Logger(args.log)
    floors = parse_device_floors(args.device_floor, args.device_floor_gib)

    if args.selftest:
        mi = meminfo()
        log(
            f"selftest: MemAvailable {fmt_gib(mi.get('MemAvailable', 0))} GiB  "
            f"MemFree {fmt_gib(mi.get('MemFree', 0))} GiB  Cached {fmt_gib(mi.get('Cached', 0))} GiB  "
            f"floor {args.floor_gib} GiB -> would "
            f"{'KILL' if mi.get('MemAvailable', 0) < args.floor_gib * GIB else 'run'}"
        )
        dev = device_free(floors, force_nvidia = True)
        for name, free in sorted(dev.items()):
            floor = effective_floor(name, floors, len(dev) > 1)
            log(f"selftest: {name} free {fmt_gib(free)} GiB, device floor {floor:.1f} GiB")
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
        f"MemAvailable {fmt_gib(mi.get('MemAvailable', 0))} GiB, Cached {fmt_gib(mi.get('Cached', 0))} GiB"
    )
    if floors:
        dev = device_free(floors)
        if dev:
            multiple = len(dev) > 1
            if multiple and floors.get("default", 0) > 0:
                log("WARNING: a global device floor is ignored on a multi-device host; "
                    "use --device-floor <cardN|nvidia:N>=GiB to select devices")
            log("device floors: " + ", ".join(
                f"{name}={effective_floor(name, floors, multiple):.1f} GiB "
                f"(free {fmt_gib(free)})"
                for name, free in sorted(dev.items())))
        else:
            log("device floors requested but no per-device query is available; "
                "using the host floor only")
    log("cmd: " + " ".join(cmd))

    if mi.get("MemAvailable", 0) < floor:
        log(
            f"REFUSING TO START: MemAvailable {fmt_gib(mi.get('MemAvailable', 0))} GiB is already "
            f"below the {args.floor_gib:.1f} GiB floor"
        )
        return 3

    t0 = time.time()
    # start_new_session so the child gets its own process group; a model load spawns
    # worker threads/processes and we want to kill all of them at once.
    child = subprocess.Popen(cmd, start_new_session = True)

    lowest = mi.get("MemAvailable", 0)
    peak_rss = 0
    last_warn = 0.0
    last_beat = 0.0
    last_fadv = 0.0
    killed = False
    device_warned = False

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
            avail = mi.get("MemAvailable", 0)
            lowest = min(lowest, avail)
            rss = proc_rss(child.pid)
            peak_rss = max(peak_rss, rss)
            now = time.time()

            if avail < floor:
                log(
                    "*** MEMORY FLOOR BREACHED -- KILLING CHILD ***\n"
                    f"    MemAvailable {fmt_gib(avail)} GiB < floor {args.floor_gib:.1f} GiB\n"
                    f"    MemFree {fmt_gib(mi.get('MemFree', 0))} GiB  "
                    f"Cached {fmt_gib(mi.get('Cached', 0))} GiB  "
                    f"Dirty {fmt_gib(mi.get('Dirty', 0))} GiB\n"
                    f"    child pid {child.pid} RSS {fmt_gib(rss)} GiB, alive {now - t0:.1f}s\n"
                    "    This box may have unified memory and does NOT OOM-kill; starving it "
                    "costs a physical power cycle. Killing now."
                )
                killed = True
                kill_tree(child.pid, signal.SIGKILL)
                # Drop clean page cache immediately -- it is the cheapest memory to get back
                # and it is usually what is squeezing us.
                try:
                    with open("/proc/sys/vm/drop_caches", "w", encoding = "ascii") as f:
                        f.write("1\n")
                    log("    dropped clean page cache (vm.drop_caches=1)")
                except OSError:
                    pass
                break

            if floors:
                dev = device_free(floors, now = now)
                if not dev and not device_warned:
                    device_warned = True
                    log("device-floor query unavailable; falling back to the host floor")
                multiple = len(dev) > 1
                for name, free in sorted(dev.items()):
                    dev_floor = effective_floor(name, floors, multiple)
                    if dev_floor > 0 and free < dev_floor * GIB:
                        log(
                            "*** DEVICE MEMORY FLOOR BREACHED -- KILLING CHILD ***\n"
                            f"    {name} free {fmt_gib(free)} GiB < floor {dev_floor:.1f} GiB\n"
                            f"    host MemAvailable {fmt_gib(avail)} GiB\n"
                            f"    child pid {child.pid} RSS {fmt_gib(rss)} GiB, "
                            f"alive {now - t0:.1f}s\n"
                            "    Killing before the unified pool starves the box."
                        )
                        killed = True
                        kill_tree(child.pid, signal.SIGKILL)
                        break
                if killed:
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
