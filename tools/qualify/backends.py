"""Backend profiles: where the python, the engine, the locks and the GPU counters live. One entry point per backend, same stages.
MLX: add a profile with its own server and lock (stages and report are backend-neutral)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("QUALIFY_PYTHON", sys.executable)
ENGINE = os.environ.get("QUALIFY_ENGINE", str(REPO))
# Optional GPU lock files (os.pathsep-separated), taken in this order before each server start
LOCKS = [p for p in os.environ.get("QUALIFY_LOCKS", "").split(os.pathsep) if p]

PROFILES = {
    "cuda": dict(
        python=PYTHON, engine=ENGINE, locks=LOCKS, env={"EXL3_CUDA_GRAPH": "1"},
        gpu_used_mib="nvidia-smi", host_peak_note="NVIDIA device memory",
    ),
    "rocm": dict(
        python=PYTHON, engine=ENGINE, locks=LOCKS, env={},
        gpu_used_mib="/sys/class/drm/card1/device/mem_info_gtt_used", host_peak_note="Strix unified memory",
    ),
}


def profile(name: str, repo: str | None = None) -> dict:
    """repo = the engine checkout with the built extension (PYTHONPATH and cwd of every server, as the lanes run them)."""
    if name not in PROFILES:
        raise SystemExit(f"unknown backend {name!r}: {sorted(PROFILES)}")
    p = dict(PROFILES[name])
    p["env"] = {**p["env"], "PYTHONPATH": repo or p["engine"]}
    return p


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return 0.0


def gpu_free_mib(backend: str) -> float | None:
    """Free device memory in MiB, None when it cannot be read (used by the max-ctx upper bound)."""
    import subprocess
    try:
        if backend == "cuda":
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            tot, used = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
            return tot - used
        # rocm: GTT is the unified-memory pool, invisible to `free`
        d = Path("/sys/class/drm/card1/device")
        return (int((d / "mem_info_gtt_total").read_text()) - int((d / "mem_info_gtt_used").read_text())) / 2**20
    except Exception:
        return None
