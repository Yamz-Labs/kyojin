#!/usr/bin/env python3
"""Sample GPU busy% from sysfs while a benchmark runs, with no profiler attached.

profile_mtp.py reports wall-minus-sum-of-kernel-time as "unaccounted host time",
but the torch profiler itself instruments every launch, so that figure is
inflated. This measures the same quantity from outside the process using the
amdgpu driver's own counter, which costs the benched process nothing.

    /sys/class/drm/card*/device/gpu_busy_percent

Usage:  gpu_busy.py -- <command to run>
Example:
    gpu_busy.py -- .venv/bin/python tools/strix_halo/bench_mtp.py -n 256 -ndt 2 -dds -g

Prints the child's stdout, then the busy-percent distribution. A mean well below
100 means real GPU idle (host-bound); near 100 means the GPU is saturated and
only fewer-bytes-per-token or faster kernels can help.
"""
import glob
import os
import statistics
import subprocess
import sys
import threading
import time


def find_counter():
    cands = sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"))
    if not cands:
        return None
    # Prefer a card that actually moves; on a single-GPU box there is one.
    return cands[0]


class Sampler(threading.Thread):
    def __init__(self, path, interval=0.02):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                with open(self.path) as f:
                    self.samples.append((time.perf_counter(), int(f.read().strip())))
            except (OSError, ValueError):
                pass
            time.sleep(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)


def main():
    if "--" not in sys.argv:
        print(__doc__)
        return 2
    cmd = sys.argv[sys.argv.index("--") + 1:]
    if not cmd:
        print("no command given after --")
        return 2

    path = find_counter()
    if path is None:
        print("FAIL: no gpu_busy_percent under /sys/class/drm/card*/device/")
        return 1
    print(f"counter: {path}", flush=True)

    # Baseline: what does an idle GPU read?
    idle = []
    for _ in range(25):
        with open(path) as f:
            idle.append(int(f.read().strip()))
        time.sleep(0.02)
    print(f"idle baseline: mean {statistics.mean(idle):.1f}%  max {max(idle)}%",
          flush=True)

    s = Sampler(path)
    s.start()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    wall = time.perf_counter() - t0
    s.stop()

    out = proc.stdout.decode(errors="replace")
    for line in out.splitlines():
        if any(k in line for k in ("tok/s", "acceptance", "family=", "ttft")):
            print(f"  child: {line.strip()}")

    raw = s.samples
    if not raw:
        print("no samples collected")
        return 1

    # The wall clock includes ~20s of model load, which is mostly GPU-idle and
    # would drag the mean down. Window to the decode phase: parse the child's
    # reported decode seconds and keep only that trailing slice.
    dec_s = None
    for line in out.splitlines():
        if "decode:" in line and "tok/s" in line and "(" in line:
            try:
                dec_s = float(line.split("(")[1].split("chunks,")[1].split("s)")[0])
            except (IndexError, ValueError):
                pass
    t_end = raw[-1][0]
    if dec_s:
        cutoff = t_end - dec_s
        windowed = [v for (t, v) in raw if t >= cutoff]
        print(f"\n  windowed to the {dec_s:.1f}s decode phase "
              f"({len(windowed)} of {len(raw)} samples); "
              f"full run was {wall:.1f}s incl. model load")
    else:
        windowed = [v for (_, v) in raw]
        print("\n  WARNING: could not parse decode seconds; reporting whole run")
    d = windowed
    if not d:
        print("no samples in the decode window")
        return 1

    d_sorted = sorted(d)
    n = len(d_sorted)
    print(f"\n=== GPU busy during decode, {n} samples @20ms ===")
    print(f"  mean        {statistics.mean(d):.1f}%")
    print(f"  median      {d_sorted[n // 2]}%")
    print(f"  p10 / p90   {d_sorted[n // 10]}% / {d_sorted[9 * n // 10]}%")
    print(f"  min / max   {d_sorted[0]}% / {d_sorted[-1]}%")
    below = sum(1 for v in d if v < 90)
    print(f"  samples <90%: {below} ({100.0 * below / n:.1f}%)")
    print(f"\n  => implied idle: {100.0 - statistics.mean(d):.1f}% of wall")
    print(f"  => if idle were fully closed, throughput scales by "
          f"{100.0 / max(statistics.mean(d), 1.0):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
