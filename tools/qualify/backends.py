"""Backend profiles: where the python, the engine, the locks and the GPU counters live. One entry point per backend, same stages.
MLX: add a profile with its own server and lock (stages and report are backend-neutral)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("QUALIFY_PYTHON")      # None: found per backend by find_python()
ENGINE = os.environ.get("QUALIFY_ENGINE", str(REPO))
# Optional GPU lock files (os.pathsep-separated), taken in this order before each server start
LOCKS = [p for p in os.environ.get("QUALIFY_LOCKS", "").split(os.pathsep) if p]

PROFILES = {
    "cuda": dict(
        python=None, engine=ENGINE, locks=LOCKS, env={"EXL3_CUDA_GRAPH": "1"},
        gpu_used_mib="nvidia-smi", host_peak_note="NVIDIA device memory",
    ),
    "rocm": dict(
        python=None, engine=ENGINE, locks=LOCKS, env={},
        gpu_used_mib="/sys/class/drm/card1/device/mem_info_gtt_used", host_peak_note="Strix unified memory",
        ctx_cap_default=65536,      # ~120 GB of unified memory: the free-memory bound is ~1M tokens and a near-full probe at 500K would run for hours; --ctx-cap raises it
    ),
}


_TORCH_PROBE = "import torch,sys; sys.exit(0 if (torch.version.{attr} and torch.__version__) else 3)"


def find_python(backend: str, repo: str | None = None, required: bool = True) -> str | None:
    """The interpreter whose torch matches the backend (cuda: torch.version.cuda, rocm: torch.version.hip): QUALIFY_PYTHON, then the
    engine checkout's own venv, $VIRTUAL_ENV, then every $QUALIFY_WORK_ROOT/*/{venv,.venv} (if set) and the current interpreter. No hand flag:
    drill 0 tried the system python first (no torch) and had to be told which venv the lanes use. Not required (--dry-run): None when none is found."""
    import glob
    import subprocess
    if PYTHON:
        return PYTHON
    attr = "cuda" if backend == "cuda" else "hip"
    cands = []
    if repo:
        cands += [f"{repo}/.venv/bin/python", f"{repo}/venv/bin/python"]
    if os.environ.get("VIRTUAL_ENV"):
        cands.append(os.environ["VIRTUAL_ENV"] + "/bin/python")
    root = os.environ.get("QUALIFY_WORK_ROOT")
    if root:
        root = os.path.expanduser(root)
        cands += sorted(glob.glob(f"{root}/*/venv/bin/python") + glob.glob(f"{root}/*/.venv/bin/python"))
    cands.append(sys.executable)
    for c in cands:
        if not os.path.exists(c):
            continue
        try:
            if subprocess.run([c, "-I", "-c", _TORCH_PROBE.format(attr=attr)], capture_output=True, timeout=120).returncode == 0:
                return c
        except Exception:      # noqa: BLE001
            continue
    if not required:
        return None
    raise SystemExit(f"no python with a {attr} torch found (set QUALIFY_PYTHON)")


def profile(name: str, repo: str | None = None) -> dict:
    """repo = the engine checkout with the built extension (PYTHONPATH and cwd of every server, as the lanes run them)."""
    if name not in PROFILES:
        raise SystemExit(f"unknown backend {name!r}: {sorted(PROFILES)}")
    p = dict(PROFILES[name])
    p["env"] = {**p["env"], "PYTHONPATH": repo or p["engine"]}
    p["python"] = p["python"] or find_python(name, repo)
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
