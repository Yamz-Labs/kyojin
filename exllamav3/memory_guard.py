from __future__ import annotations

"""Read-only pre-flight memory guard.

The device cap is the device itself: ``min(mem_get_info().free, total)``.  There is no
GTT/VRAM reserve on this branch.  The only memory budget the guard owns is the host
floor: the process must leave enough host memory for the OS, the driver and background
processes.  A supervisor enforces the same floor from outside the process; the preflight
adds a small planning margin so it does not deliberately plan at the kill threshold.

Models are loaded from actual measured bytes: module weights, cache/recurrent bytes, the
part0 load slab and a prefill workspace.  A dedicated-VRAM OOM is left to the process
(and is normally a recoverable allocation error); the host floor is what prevents the
unified-memory box from starving.
"""

import glob
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

try:
    from .guard_common import (
        DEFAULT_HOST_FLOOR_GIB,
        host_floor_from_env,
    )
except ImportError:                                      # loaded as a standalone file in CPU tests
    from guard_common import (
        DEFAULT_HOST_FLOOR_GIB,
        host_floor_from_env,
    )

GIB = 1024 ** 3

MODES = ("auto", "off")
DEFAULT_PREFLIGHT_MARGIN_GIB = 1.0
DEFAULT_WORKSPACE_GIB = 2.0


class OOMGuardError(RuntimeError):
    """The pre-flight estimate does not fit; raised before allocation."""


@dataclass
class GuardPolicy:
    """The only tunables: host reserve, planner margin and the prefill workspace."""

    mode: str = "auto"
    host_floor_gib: float = DEFAULT_HOST_FLOOR_GIB
    preflight_margin_gib: float = DEFAULT_PREFLIGHT_MARGIN_GIB
    workspace_gib: float = DEFAULT_WORKSPACE_GIB

    def __post_init__(self) -> None:
        self.mode = (self.mode or "auto").strip().lower()
        if self.mode not in MODES:
            raise ValueError(f"oom guard mode must be one of {MODES}, not {self.mode!r}")
        self.host_floor_gib = max(0.0, float(self.host_floor_gib))
        self.preflight_margin_gib = max(0.0, float(self.preflight_margin_gib))
        self.workspace_gib = max(0.0, float(self.workspace_gib))

    @classmethod
    def from_env(cls, default_mode: str = "auto",
                 explicit_mode: str | None = None) -> GuardPolicy:
        def value(name: str, fallback: float) -> float:
            try:
                return float(os.environ.get(name, fallback))
            except ValueError:
                return float(fallback)

        mode = explicit_mode if explicit_mode is not None else os.environ.get(
            "KYOJIN_OOM_GUARD", default_mode)
        return cls(
            mode=mode,
            host_floor_gib=host_floor_from_env(),
            preflight_margin_gib=value("KYOJIN_OOM_GUARD_PREFLIGHT_MARGIN_GIB",
                                       DEFAULT_PREFLIGHT_MARGIN_GIB),
            workspace_gib=value("KYOJIN_OOM_GUARD_WORKSPACE_GIB", DEFAULT_WORKSPACE_GIB),
        )

    @property
    def preflight_host_floor_gib(self) -> float:
        return self.host_floor_gib + self.preflight_margin_gib


# ---------------------------------------------------------------------------------------------------------
# Probes (monkeypatchable from CPU tests)
# ---------------------------------------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def meminfo() -> dict[str, int]:
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


def host_available_bytes() -> int | None:
    value = meminfo().get("MemAvailable")
    return int(value) if value is not None else None


def host_total_bytes() -> int | None:
    value = meminfo().get("MemTotal")
    return int(value) if value is not None else None


def device_total_bytes(index: int) -> int | None:
    try:
        return int(_torch().cuda.get_device_properties(int(index)).total_memory)
    except Exception:                                    # noqa: BLE001 - unavailable probe
        return None


def device_free_bytes(index: int) -> int | None:
    try:
        free, _total = _torch().cuda.mem_get_info(int(index))
        return int(free)
    except Exception:                                    # noqa: BLE001 - unavailable probe
        return None


def is_unified_device(index: int, arch: str | None = None, total: int | None = None) -> bool:
    """Positive unified/GTT detection, independent of a hardcoded device index."""
    arch_key = (arch or "").strip().lower()
    if arch_key.startswith("gfx115"):
        return True
    props = None
    try:
        props = _torch().cuda.get_device_properties(int(index))
    except Exception:                                    # noqa: BLE001, S110 - unavailable probe
        pass
    if props is not None:
        if getattr(props, "is_integrated", False) or getattr(props, "integrated", False):
            return True
        name = (getattr(props, "name", "") or "").lower()
        if "gb10" in name or "dgx spark" in name:
            return True
    total = device_total_bytes(index) if total is None else total
    host = host_total_bytes()
    # A dedicated GPU can be more than half of host RAM, so require a near-host total
    # together with an AMD GTT pool before treating it as unified.
    return (total is not None and host is not None and total >= 0.9 * host
            and bool(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total")))


# ---------------------------------------------------------------------------------------------------------
# Footprints and comparison
# ---------------------------------------------------------------------------------------------------------

@dataclass
class DeviceFootprint:
    index: int
    arch: str = ""
    name: str = ""
    unified: bool = False
    total_bytes: int | None = None
    free_bytes: int | None = None
    weights_bytes: int = 0
    cache_bytes: int = 0
    slab_bytes: int = 0
    workspace_bytes: int = 0

    @property
    def need_bytes(self) -> int:
        return (int(self.weights_bytes) + int(self.cache_bytes)
                + int(self.slab_bytes) + int(self.workspace_bytes))

    def capacity_bytes(self) -> int | None:
        """The device cap is its own free/total memory; no device reserve."""
        if self.total_bytes is None and self.free_bytes is None:
            return None
        if self.free_bytes is None:
            return int(self.total_bytes)
        if self.total_bytes is None:
            return int(self.free_bytes)
        return min(int(self.total_bytes), int(self.free_bytes))

    def describe(self) -> str:
        parts = [f"cuda:{self.index}"]
        if self.arch:
            parts.append(self.arch)
        if self.name:
            parts.append(self.name)
        if self.unified:
            parts.append("unified")
        return " ".join(parts)


@dataclass
class DeviceResult:
    footprint: DeviceFootprint
    capacity_bytes: int | None

    @property
    def deficit_bytes(self) -> int:
        if self.capacity_bytes is None:
            return 0
        return max(0, self.footprint.need_bytes - self.capacity_bytes)


@dataclass
class HostResult:
    available_bytes: int | None
    need_bytes: int
    floor_bytes: int

    @property
    def capacity_bytes(self) -> int | None:
        if self.available_bytes is None:
            return None
        return max(0, int(self.available_bytes) - int(self.floor_bytes))

    @property
    def deficit_bytes(self) -> int:
        if self.capacity_bytes is None:
            return 0
        return max(0, int(self.need_bytes) - int(self.capacity_bytes))


@dataclass
class GuardReport:
    device_results: list[DeviceResult] = field(default_factory = list)
    host_result: HostResult | None = None

    @property
    def issues(self) -> list[str]:
        out = []
        for result in self.device_results:
            if result.capacity_bytes is not None and result.deficit_bytes > 0:
                out.append(f"cuda:{result.footprint.index} needs "
                           f"{result.footprint.need_bytes / GIB:.2f} GiB, capacity "
                           f"{result.capacity_bytes / GIB:.2f} GiB "
                           f"(deficit {result.deficit_bytes / GIB:.2f} GiB)")
        host = self.host_result
        if host is not None and host.capacity_bytes is not None and host.deficit_bytes > 0:
            out.append(f"host needs {host.need_bytes / GIB:.2f} GiB, capacity "
                       f"{host.capacity_bytes / GIB:.2f} GiB "
                       f"(deficit {host.deficit_bytes / GIB:.2f} GiB)")
        return out

    @property
    def ok(self) -> bool:
        return not self.issues


def _format_report(report: GuardReport, *, context: int | None,
                   chosen_n: int | None, largest_n: int | None,
                   largest_ctx: int | None) -> str:
    lines = ["device      arch       need    capacity     free     total  slab  workspace  deficit"]
    for result in report.device_results:
        fp = result.footprint
        free = "?" if fp.free_bytes is None else f"{fp.free_bytes / GIB:8.2f}"
        total = "?" if fp.total_bytes is None else f"{fp.total_bytes / GIB:8.2f}"
        cap = "?" if result.capacity_bytes is None else f"{result.capacity_bytes / GIB:8.2f}"
        lines.append(
            f"{fp.describe():18s} {fp.need_bytes / GIB:8.2f} {cap:>10s} {free:>8s} {total:>8s} "
            f"{fp.slab_bytes / GIB:5.2f} {fp.workspace_bytes / GIB:10.2f} "
            f"{result.deficit_bytes / GIB:8.2f}")
    host = report.host_result
    if host is not None and host.capacity_bytes is not None:
        lines.append(f"host MemAvailable {host.available_bytes / GIB:.2f} GiB, GTT need "
                     f"{host.need_bytes / GIB:.2f} GiB, planning floor "
                     f"{host.floor_bytes / GIB:.2f} GiB, capacity "
                     f"{host.capacity_bytes / GIB:.2f} GiB, deficit "
                     f"{host.deficit_bytes / GIB:.2f} GiB")
    if context is not None:
        lines.append(f"chosen: part0 N={chosen_n} at -c {context}")
    if largest_n is not None:
        shown = "none" if largest_n < 0 else str(largest_n)
        lines.append(f"largest part0 N that fits at this context: {shown}")
    if largest_ctx is not None:
        lines.append(f"largest context that fits at this N: {largest_ctx}")
    return "\n".join(lines)


class OOMGuard:
    def __init__(self, policy: GuardPolicy, logger: Callable[[str], None] = print):
        self.policy = policy
        self.logger = logger

    def footprint(self, index: int, arch: str = "", name: str = "", *,
                  total_bytes: int | None = None, free_bytes: int | None = None,
                  weights_bytes: int = 0, cache_bytes: int = 0,
                  reserve_bytes: int = 0) -> DeviceFootprint:
        index = int(index)
        if total_bytes is None:
            total_bytes = device_total_bytes(index)
        if free_bytes is None:
            free_bytes = device_free_bytes(index)
        return DeviceFootprint(
            index = index, arch = arch, name = name,
            unified = is_unified_device(index, arch, total_bytes),
            total_bytes = total_bytes, free_bytes = free_bytes,
            weights_bytes = int(weights_bytes), cache_bytes = int(cache_bytes),
            slab_bytes = int(reserve_bytes),
            workspace_bytes = int(self.policy.workspace_gib * GIB),
        )

    def evaluate(self, footprints: Iterable[DeviceFootprint],
                 host_available: int | None = None) -> GuardReport:
        footprints = list(footprints)
        results = [DeviceResult(fp, fp.capacity_bytes()) for fp in footprints]
        host = None
        if any(fp.unified for fp in footprints):
            if host_available is None:
                host_available = host_available_bytes()
            host = HostResult(available_bytes = host_available,
                              need_bytes = sum(fp.need_bytes for fp in footprints if fp.unified),
                              floor_bytes = int(self.policy.preflight_host_floor_gib * GIB))
        return GuardReport(device_results = results, host_result = host)

    def check(self, footprints: Iterable[DeviceFootprint], *,
              host_available: int | None = None,
              context: int | None = None,
              chosen_n: int | None = None,
              largest_n: int | None = None,
              largest_ctx: int | None = None) -> GuardReport:
        report = self.evaluate(footprints, host_available = host_available)
        if self.policy.mode == "off":
            return report
        for result in report.device_results:
            fp = result.footprint
            if result.capacity_bytes is None:
                self.logger(f"[oom-guard] {fp.describe()}: capacity unavailable; skipped")
                continue
            self.logger(f"[oom-guard] {fp.describe()}: need {fp.need_bytes / GIB:.2f} GiB "
                        f"(weights {fp.weights_bytes / GIB:.2f}, cache {fp.cache_bytes / GIB:.2f}, "
                        f"slab {fp.slab_bytes / GIB:.2f}, workspace {fp.workspace_bytes / GIB:.2f}) "
                        f"of {result.capacity_bytes / GIB:.2f} GiB")
        host = report.host_result
        if host is not None and host.capacity_bytes is not None:
            self.logger(f"[oom-guard] host: GTT need {host.need_bytes / GIB:.2f} GiB of "
                        f"{host.capacity_bytes / GIB:.2f} GiB (MemAvailable "
                        f"{host.available_bytes / GIB:.2f} - planning floor "
                        f"{host.floor_bytes / GIB:.2f})")
        if report.ok:
            return report
        message = _format_report(report, context = context, chosen_n = chosen_n,
                                 largest_n = largest_n, largest_ctx = largest_ctx)
        raise OOMGuardError(
            "pre-flight OOM guard refused the load before allocating anything "
            f"(mode={self.policy.mode}):\n{message}\n"
            "Lower -c/--max-ctx or --part0-layers, raise MEMGUARD_FLOOR/"
            "KYOJIN_OOM_GUARD_PREFLIGHT_MARGIN_GIB if they are wrong, or set "
            "KYOJIN_OOM_GUARD=off to disable the preflight.")


def check_single_device(
    weights_bytes: int,
    cache_bytes: int,
    device,
    *,
    policy: GuardPolicy | None = None,
    arch: str = "",
    context: int | None = None,
    logger: Callable[[str], None] = print,
) -> GuardReport:
    try:
        index = int(device)
    except (TypeError, ValueError):
        text = str(device)
        index = int(text.split(":", 1)[1]) if text.startswith("cuda:") else 0
    guard = OOMGuard(policy or GuardPolicy.from_env(), logger = logger)
    fp = guard.footprint(index, arch = arch, weights_bytes = weights_bytes,
                         cache_bytes = cache_bytes)
    return guard.check([fp], context = context)
