from __future__ import annotations

"""Read-only pre-flight memory guard for unified and dedicated devices.

The guard is deliberately independent of the split placement code: ``device_map``
estimates what each device will hold (weights, cache, reserve slabs and the prefill
workspace) and hands the numbers to :class:`OOMGuard`.  The guard only compares them
with the device and host budgets and either refuses the load, logs a warning, or does
nothing.  It never allocates, changes an allocator setting or mutates the plan.

The arithmetic is kept in one place so the split adapters cannot drift:

* dedicated VRAM: ``capacity = min(total, free) - reserve - safety - floor``
* unified/GTT: the observed usable ceiling (default 113.3 GiB of a reported 118 GiB,
  i.e. the ~4.5 GiB reserve) is the hard cap; the current free memory is a second cap:
  ``capacity = min(usable_ceiling, free - reserve) - safety - floor``.  The reserve is
  not subtracted twice: the ceiling already accounts for the unified overhead.
* unified devices additionally consume the host pool, so the summed GTT footprint is
  compared with ``MemAvailable - host_floor``.

Per-device settings accept the split's device-selector syntax:
``gfx1151=4.5,gfx1201=0.8``, ``cuda:0=4`` or a bare number for every device.
"""

import dataclasses
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

GIB = 1024 ** 3

# Calibrated, overridable tuning values from the measured Strix Halo + RX 9070 pair.
DEFAULT_UNIFIED_RESERVE_GIB = 4.5
DEFAULT_DEDICATED_RESERVE_GIB = 0.8
DEFAULT_BULK_WORKSPACE_GIB = 2.0
DEFAULT_OTHER_WORKSPACE_GIB = 1.5
DEFAULT_SAFETY_GIB = 0.0
DEFAULT_DEVICE_FLOOR_GIB = 0.0
DEFAULT_HOST_FLOOR_GIB = 6.0
DEFAULT_UNIFIED_CEILING_GIB = 113.3

# Do not refuse a load that misses the estimate by a hair: the estimator is byte
# accounting, not a measurement.  The band is small enough that the measured 640K
# failure is still refused while a measured-fits 500K point is not.
DEFAULT_TOLERANCE_GIB = 0.25

MODES = ("auto", "warn", "off")


class OOMGuardError(RuntimeError):
    """The estimated footprint cannot fit; raised before any allocation."""


# ---------------------------------------------------------------------------------------------------------
# Device settings
# ---------------------------------------------------------------------------------------------------------

def parse_device_spec(value) -> tuple[float | None, dict[str, float]]:
    """Parse a scalar or ``selector=value,selector=value`` device setting.

    Returns ``(default, overrides)``.  ``default`` applies to every device that does
    not match an override.  ``None`` means the caller's per-device default is used.
    """
    if value is None or value == "":
        return None, {}
    if isinstance(value, (int, float)):
        return float(value), {}
    text = str(value).strip()
    if not text:
        return None, {}
    if "=" not in text:
        try:
            return float(text), {}
        except ValueError as exc:
            raise ValueError(
                f"expected a number or selector=value list, got {value!r}") from exc
    default = None
    overrides: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"expected selector=value, got {item!r}")
        try:
            number = float(raw)
        except ValueError as exc:
            raise ValueError(f"expected a number on the right of {key!r}, got {raw!r}") from exc
        key = key.strip().lower()
        if key in ("default", "all", "*"):
            default = number
        else:
            overrides[key] = number
    return default, overrides


def resolve_device_spec(
    value,
    *,
    infos: Iterable,
    select: Callable[[str], int],
    default: Callable[[int], float],
) -> dict[int, float]:
    """Resolve a device spec to ``{torch_index: value}`` using the split selector.

    ``infos`` is the visible inventory (objects with ``.index``); ``select`` is the
    adapter's selector callback so ``gfxXXXX``/``cuda:N``/name/UUID semantics are
    exactly the split's.
    """
    scalar, overrides = parse_device_spec(value)
    out = {int(info.index): (float(scalar) if scalar is not None else float(default(info.index)))
           for info in infos}
    for key, number in overrides.items():
        index = int(select(key))
        out[index] = float(number)
    return out


@dataclass
class DeviceBudget:
    """Per-device pre-flight policy.

    ``reserve``, ``workspace``, ``safety`` and ``device_floor`` may be a number, a
    selector list or ``None`` (use the mode-dependent defaults).
    """

    mode: str = "auto"
    reserve: str | float | None = None
    workspace: str | float | None = None
    safety: str | float | None = None
    device_floor: str | float | None = None
    unified_reserve_gib: float = DEFAULT_UNIFIED_RESERVE_GIB
    dedicated_reserve_gib: float = DEFAULT_DEDICATED_RESERVE_GIB
    bulk_workspace_gib: float = DEFAULT_BULK_WORKSPACE_GIB
    other_workspace_gib: float = DEFAULT_OTHER_WORKSPACE_GIB
    safety_gib: float = DEFAULT_SAFETY_GIB
    device_floor_gib: float = DEFAULT_DEVICE_FLOOR_GIB
    host_floor_gib: float = DEFAULT_HOST_FLOOR_GIB
    unified_ceiling_gib: float = DEFAULT_UNIFIED_CEILING_GIB
    tolerance_gib: float = DEFAULT_TOLERANCE_GIB

    def __post_init__(self) -> None:
        self.mode = (self.mode or "auto").strip().lower()
        if self.mode not in MODES:
            raise ValueError(f"oom guard mode must be one of {MODES}, not {self.mode!r}")

    @classmethod
    def from_env(cls, default_mode: str = "auto") -> DeviceBudget:
        """Build from ``KYOJIN_OOM_GUARD*`` environment fallbacks."""
        def f(name, fallback):
            try:
                return float(os.environ.get(name, fallback))
            except ValueError:
                return float(fallback)

        return cls(
            mode=os.environ.get("KYOJIN_OOM_GUARD", default_mode),
            reserve=os.environ.get("KYOJIN_OOM_GUARD_RESERVE_GIB"),
            workspace=os.environ.get("KYOJIN_OOM_GUARD_WORKSPACE_GIB"),
            safety=os.environ.get("KYOJIN_OOM_GUARD_SAFETY_GIB"),
            device_floor=os.environ.get("KYOJIN_OOM_GUARD_DEVICE_FLOOR"),
            unified_reserve_gib=f("KYOJIN_OOM_GUARD_UNIFIED_RESERVE_GIB", DEFAULT_UNIFIED_RESERVE_GIB),
            dedicated_reserve_gib=f("KYOJIN_OOM_GUARD_DEDICATED_RESERVE_GIB", DEFAULT_DEDICATED_RESERVE_GIB),
            bulk_workspace_gib=f("KYOJIN_OOM_GUARD_BULK_WORKSPACE_GIB", DEFAULT_BULK_WORKSPACE_GIB),
            other_workspace_gib=f("KYOJIN_OOM_GUARD_OTHER_WORKSPACE_GIB", DEFAULT_OTHER_WORKSPACE_GIB),
            safety_gib=f("KYOJIN_OOM_GUARD_DEFAULT_SAFETY_GIB", DEFAULT_SAFETY_GIB),
            device_floor_gib=f("KYOJIN_OOM_GUARD_DEFAULT_DEVICE_FLOOR_GIB", DEFAULT_DEVICE_FLOOR_GIB),
            host_floor_gib=f("KYOJIN_OOM_GUARD_HOST_FLOOR_GIB", DEFAULT_HOST_FLOOR_GIB),
            unified_ceiling_gib=f("KYOJIN_OOM_GUARD_USABLE_CEILING_GIB", DEFAULT_UNIFIED_CEILING_GIB),
            tolerance_gib=f("KYOJIN_OOM_GUARD_TOLERANCE_GIB", DEFAULT_TOLERANCE_GIB),
        )

    def resolve(
        self,
        infos: Iterable,
        select: Callable[[str], int],
        *,
        unified: dict[int, bool],
        bulk_index: int | None,
    ) -> dict[str, dict[int, float]]:
        """Resolve every setting to a per-device byte map."""
        def is_unified(index: int) -> bool:
            return bool(unified.get(int(index), False))

        def reserve_default(index: int) -> float:
            return self.unified_reserve_gib if is_unified(index) else self.dedicated_reserve_gib

        def workspace_default(index: int) -> float:
            return self.bulk_workspace_gib if bulk_index is not None and int(index) == int(bulk_index) \
                else self.other_workspace_gib

        def safety_default(_index: int) -> float:
            return self.safety_gib

        def floor_default(_index: int) -> float:
            return self.device_floor_gib

        return {
            "reserve": resolve_device_spec(self.reserve, infos = infos, select = select,
                                           default = reserve_default),
            "workspace": resolve_device_spec(self.workspace, infos = infos, select = select,
                                             default = workspace_default),
            "safety": resolve_device_spec(self.safety, infos = infos, select = select,
                                          default = safety_default),
            "device_floor": resolve_device_spec(self.device_floor, infos = infos, select = select,
                                                default = floor_default),
        }


# ---------------------------------------------------------------------------------------------------------
# Device/host probes (monkeypatchable in CPU tests)
# ---------------------------------------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def meminfo() -> dict[str, int]:
    """Return ``/proc/meminfo`` in bytes (empty when unavailable)."""
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
        torch = _torch()
        return int(torch.cuda.get_device_properties(int(index)).total_memory)
    except Exception:                                        # noqa: BLE001 - CPU tests / unavailable probe
        return None


def device_free_bytes(index: int) -> int | None:
    try:
        torch = _torch()
        free, _total = torch.cuda.mem_get_info(int(index))
        return int(free)
    except Exception:                                        # noqa: BLE001 - CPU tests / unavailable probe
        return None


def is_unified_device(index: int, arch: str | None = None, total: int | None = None) -> bool:
    """Positive unified/GTT detection, independent of a hardcoded device index.

    Large dedicated GPUs can have more than half of host RAM in VRAM, so a bare
    total/host ratio is not enough.  Require an APU arch/property, or a near-host
    total together with a GTT sysfs pool, so a discrete card is never put on the
    host-pool budget.
    """
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
    if total is not None and host is not None and total >= 0.9 * host:
        import glob
        if glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total"):
            return True
    return False


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
    slab_bytes: int = 0        # actually held during the load (the part0 reserve slab)
    reserve_bytes: int = 0     # headroom policy: reduces capacity, not footprint
    workspace_bytes: int = 0
    safety_bytes: int = 0
    floor_bytes: int = 0

    @property
    def need_bytes(self) -> int:
        """Estimated resident footprint (headroom is a capacity reduction, not a byte)."""
        return (int(self.weights_bytes) + int(self.cache_bytes) + int(self.slab_bytes)
                + int(self.workspace_bytes))

    def capacity_bytes(self, unified_ceiling_gib: float) -> int | None:
        """Usable bytes for this device, or None when nothing can be probed."""
        reserve = int(self.reserve_bytes)
        if self.unified:
            hard = int(unified_ceiling_gib * GIB)
            if self.total_bytes is not None:
                hard = min(hard, int(self.total_bytes))
            if self.free_bytes is not None:
                hard = min(hard, int(self.free_bytes) - reserve)
            else:
                hard -= reserve
        else:
            if self.total_bytes is None and self.free_bytes is None:
                return None
            if self.free_bytes is None:
                base = int(self.total_bytes)
            elif self.total_bytes is None:
                base = int(self.free_bytes)
            else:
                base = min(int(self.total_bytes), int(self.free_bytes))
            hard = base - reserve
        hard -= int(self.safety_bytes) + int(self.floor_bytes)
        return max(0, hard)

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
    tolerance_gib: float = DEFAULT_TOLERANCE_GIB

    @property
    def issues(self) -> list[str]:
        out = []
        band = int(self.tolerance_gib * GIB)
        for result in self.device_results:
            if result.capacity_bytes is not None and result.deficit_bytes > band:
                out.append(f"cuda:{result.footprint.index} needs "
                           f"{result.footprint.need_bytes / GIB:.2f} GiB, capacity "
                           f"{result.capacity_bytes / GIB:.2f} GiB "
                           f"(deficit {result.deficit_bytes / GIB:.2f} GiB)")
        if self.host_result is not None and self.host_result.capacity_bytes is not None \
                and self.host_result.deficit_bytes > band:
            out.append(f"host needs {self.host_result.need_bytes / GIB:.2f} GiB GTT, "
                       f"capacity {self.host_result.capacity_bytes / GIB:.2f} GiB "
                       f"(deficit {self.host_result.deficit_bytes / GIB:.2f} GiB)")
        return out

    @property
    def ok(self) -> bool:
        return not self.issues


def _format_report(report: GuardReport, *, context: int | None,
                   chosen_n: int | None, largest_n: int | None,
                   largest_ctx: int | None) -> str:
    lines = ["device      arch       need    capacity     free     total  reserve  workspace  slab  deficit"]
    for result in report.device_results:
        fp = result.footprint
        free = "?" if fp.free_bytes is None else f"{fp.free_bytes / GIB:8.2f}"
        total = "?" if fp.total_bytes is None else f"{fp.total_bytes / GIB:8.2f}"
        cap = "?" if result.capacity_bytes is None else f"{result.capacity_bytes / GIB:8.2f}"
        lines.append(
            f"{fp.describe():18s} {fp.need_bytes / GIB:8.2f} {cap:>10s} {free:>8s} {total:>8s} "
            f"{fp.reserve_bytes / GIB:7.2f} {fp.workspace_bytes / GIB:10.2f} "
            f"{fp.slab_bytes / GIB:5.2f} {result.deficit_bytes / GIB:8.2f}")
    if report.host_result is not None and report.host_result.capacity_bytes is not None:
        h = report.host_result
        lines.append(f"host MemAvailable {h.available_bytes / GIB:.2f} GiB, GTT need "
                     f"{h.need_bytes / GIB:.2f} GiB, floor {h.floor_bytes / GIB:.2f} GiB, "
                     f"capacity {h.capacity_bytes / GIB:.2f} GiB, deficit "
                     f"{h.deficit_bytes / GIB:.2f} GiB")
    if context is not None:
        lines.append(f"chosen: part0 N={chosen_n} at -c {context}")
    if largest_n is not None:
        shown = "none" if largest_n < 0 else str(largest_n)
        lines.append(f"largest part0 N that fits at this context: {shown}")
    if largest_ctx is not None:
        lines.append(f"largest context that fits at this N: {largest_ctx}")
    return "\n".join(lines)


class OOMGuard:
    """Compare prepared footprints with the per-device and host budgets."""

    def __init__(self, budget: DeviceBudget, resolved: dict[str, dict[int, float]],
                 logger: Callable[[str], None] = print):
        self.budget = budget
        self.resolved = resolved
        self.logger = logger

    @classmethod
    def from_budget(cls, budget: DeviceBudget, infos, select, *,
                    unified: dict[int, bool], bulk_index: int | None,
                    logger: Callable[[str], None] = print) -> OOMGuard:
        return cls(budget, budget.resolve(infos, select, unified = unified,
                                          bulk_index = bulk_index), logger = logger)

    @property
    def mode(self) -> str:
        return self.budget.mode

    def footprint(self, index: int, arch: str = "", name: str = "", *,
                  total_bytes: int | None = None,
                  free_bytes: int | None = None, weights_bytes: int = 0,
                  cache_bytes: int = 0, reserve_bytes: int = 0) -> DeviceFootprint:
        index = int(index)
        unified = is_unified_device(index, arch, total_bytes)
        if total_bytes is None:
            total_bytes = device_total_bytes(index)
        if free_bytes is None:
            free_bytes = device_free_bytes(index)
        reserve = self.resolved["reserve"].get(index, 0.0)
        workspace = self.resolved["workspace"].get(index, 0.0)
        safety = self.resolved["safety"].get(index, 0.0)
        floor = self.resolved["device_floor"].get(index, 0.0)
        return DeviceFootprint(
            index = index, arch = arch, name = name, unified = bool(unified),
            total_bytes = total_bytes, free_bytes = free_bytes,
            weights_bytes = int(weights_bytes), cache_bytes = int(cache_bytes),
            slab_bytes = int(reserve_bytes),
            reserve_bytes = int(reserve * GIB),
            workspace_bytes = int(workspace * GIB),
            safety_bytes = int(safety * GIB),
            floor_bytes = int(floor * GIB),
        )

    def evaluate(self, footprints: Iterable[DeviceFootprint],
                 host_available: int | None = None) -> GuardReport:
        footprints = list(footprints)
        results = [DeviceResult(fp, fp.capacity_bytes(self.budget.unified_ceiling_gib))
                   for fp in footprints]
        host_need = sum(fp.need_bytes for fp in footprints if fp.unified)
        host = None
        if any(fp.unified for fp in footprints):
            if host_available is None:
                host_available = host_available_bytes()
            host = HostResult(available_bytes = host_available, need_bytes = host_need,
                              floor_bytes = int(self.budget.host_floor_gib * GIB))
        return GuardReport(device_results = results, host_result = host,
                           tolerance_gib = self.budget.tolerance_gib)

    def check(self, footprints: Iterable[DeviceFootprint], *,
              host_available: int | None = None,
              context: int | None = None,
              chosen_n: int | None = None,
              largest_n: int | None = None,
              largest_ctx: int | None = None) -> GuardReport:
        report = self.evaluate(footprints, host_available = host_available)
        if self.mode == "off":
            return report
        for result in report.device_results:
            fp = result.footprint
            if result.capacity_bytes is None:
                self.logger(f"[oom-guard] {fp.describe()}: capacity unavailable; skipped")
                continue
            self.logger(f"[oom-guard] {fp.describe()}: need {fp.need_bytes / GIB:.2f} GiB "
                        f"(weights {fp.weights_bytes / GIB:.2f}, cache {fp.cache_bytes / GIB:.2f}, "
                        f"reserve {fp.reserve_bytes / GIB:.2f}, workspace {fp.workspace_bytes / GIB:.2f}, "
                        f"slab {fp.slab_bytes / GIB:.2f}) "
                        f"of {result.capacity_bytes / GIB:.2f} GiB")
        if report.host_result is not None and report.host_result.capacity_bytes is not None:
            h = report.host_result
            self.logger(f"[oom-guard] host: GTT need {h.need_bytes / GIB:.2f} GiB of "
                        f"{h.capacity_bytes / GIB:.2f} GiB (MemAvailable "
                        f"{h.available_bytes / GIB:.2f} - floor {h.floor_bytes / GIB:.2f})")
        if report.ok:
            return report
        message = _format_report(report, context = context, chosen_n = chosen_n,
                                 largest_n = largest_n, largest_ctx = largest_ctx)
        if self.mode == "warn":
            self.logger("[oom-guard] WARNING: estimated footprint does not fit "
                        f"(mode=warn, continuing):\n{message}")
            return report
        raise OOMGuardError(
            "pre-flight OOM guard refused the load before allocating anything "
            f"(mode={self.mode}):\n{message}\n"
            "Lower -c/--max-ctx or --part0-layers, raise the reserve/workspace only if "
            "they are wrong, or set --oom-guard warn/off (KYOJIN_OOM_GUARD=warn|off).")


def check_single_device(
    weights_bytes: int,
    cache_bytes: int,
    device,
    *,
    budget: DeviceBudget | None = None,
    arch: str = "",
    name: str = "",
    context: int | None = None,
    logger: Callable[[str], None] = print,
) -> GuardReport:
    """Best-effort guard for a single-device load (default mode ``warn``).

    Single-device defaults are not calibrated on every host, so an unset
    ``KYOJIN_OOM_GUARD`` mode means ``warn`` here.  Split loads use ``auto``.
    """
    try:
        index = int(device)
    except (TypeError, ValueError):
        text = str(device)
        index = int(text.split(":", 1)[1]) if text.startswith("cuda:") else 0
    budget = budget or DeviceBudget.from_env(default_mode = "warn")
    if budget.mode == "off":
        return GuardReport()
    infos = [dataclasses.make_dataclass("_Info", [("index", int)])(index = index)]
    unified = {index: is_unified_device(index, arch)}
    guard = OOMGuard.from_budget(budget, infos, lambda _key: index,
                                 unified = unified, bulk_index = index,
                                 logger = logger)
    fp = guard.footprint(index, arch = arch, name = name, weights_bytes = weights_bytes,
                         cache_bytes = cache_bytes)
    return guard.check([fp], context = context)


def main(argv: list[str] | None = None) -> int:
    """Allow ``python -m exllamav3.memory_guard --selftest`` style probing."""
    import argparse
    ap = argparse.ArgumentParser(description = "pre-flight OOM guard self-test")
    ap.add_argument("--selftest", action = "store_true")
    ap.add_argument("--gib", type = float, default = 0.0,
                    help = "simulate a single unified device needing this many GiB")
    args = ap.parse_args(argv)
    if not args.selftest:
        ap.error("nothing to do (use --selftest)")
    budget = DeviceBudget.from_env()
    guard = OOMGuard(budget, {"reserve": {}, "workspace": {}, "safety": {}, "device_floor": {}})
    fp = DeviceFootprint(index = 0, arch = "selftest", unified = True,
                         total_bytes = 118 * GIB, free_bytes = 118 * GIB,
                         weights_bytes = int(args.gib * GIB))
    report = guard.check([fp], context = 131072, chosen_n = 6)
    print("fits" if report.ok else "does not fit")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
