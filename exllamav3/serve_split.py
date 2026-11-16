from __future__ import annotations

"""Shared serve-side helper for the contiguous layer split.

Adapters (``tools/glm`` first, then Qwen/DSV4) parse the same flags, resolve the
same devices and hand their components to ``SplitPlan``.  No placement logic or
hardware-specific default lives here: a component is declared with a name and,
optionally, a policy.  Devices are chosen by position (``cuda:N``, ``gfxXXXX``,
UUID or name); either device may serve ``part0`` or ``part1``.

This module stays pure stdlib at import: adapters must be able to apply their
serving environment before ``exllamav3`` (and therefore torch) is imported, so the
exllamav3 imports are deliberately deferred to call time.
"""

import argparse
import glob
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exllamav3.device_map import ComponentMap, ComponentPolicy, SplitPlan

# Load-time slab held on the part0 device.  A tuning value, not a hardware rule:
# size it to the part0 card (Stage 5 derives it from the device total).
DEFAULT_PART0_RESERVE_GB = 1.2

# Split mode runs eager until per-segment capture lands (Stage 6).  A whole-forward
# or per-block graph must never span the host-staged bridge between segments.
# Deferred prefill residual fusions keep a pending state on the producing stream;
    # the next block may be on the other device, beyond the bridge. Split mode runs
    # the unfused path (same numerics, no cross-stream pending tensor).
SPLIT_UNSAFE_ENV = {
    "EXL3_PF_HC_FUSE": "0",
    "EXL3_PF_GR_FUSE": "0",
    "EXL3_GR_PF_FUSE": "0",
}

GRAPH_ENV_OFF = {
    "EXL3_BLOCK_GRAPH": "0",
    "EXL3_BLOCK_GRAPH_MLA": "0",
    "EXL3_BLOCK_GRAPH_VERIFY": "0",
    "EXL3_BLOCK_GRAPH_DRAFT": "0",
    "EXL3_BLOCK_GRAPH_ATTN": "0",
    "EXL3_BLOCK_GRAPH_DEC_MOE": "0",
    "EXL3_CUDA_GRAPH": "0",
    "EXL3_HIP_GRAPH": "0",
}

# ---------------------------------------------------------------------------------------------------------
# Allocator mode (applied before `import exllamav3`, process-global)
# ---------------------------------------------------------------------------------------------------------

EXPANDABLE_SEGMENTS_MODES = ("auto", "on", "off")

# `auto` turns expandable segments off only for large-context unified loads, the
# measured case where the default allocator's fragmentation costs the last ~0.5 GiB
# (512K loaded with expandable_segments:False and failed with the default).  Both the
# context gate and the estimate are deliberately conservative and overridable:
# KYOJIN_EXPANDABLE_SEGMENTS_AUTO_CTX / _AUTO_OFF / --expandable-segments on|off.
AUTO_EXPANDABLE_MIN_CTX = 512 * 1024
AUTO_EXPANDABLE_CACHE_BYTES_PER_TOKEN = 16 * 1024   # upper band for MLA/MoE caches
AUTO_EXPANDABLE_BAND_GIB = 8.0
UNIFIED_USABLE_CEILING_GIB = 113.3


def _meminfo_bytes(key: str) -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding = "ascii") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _has_gtt_device() -> bool:
    return bool(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total"))


def _model_dir_bytes(path: str | None) -> int:
    if not path or not os.path.isdir(path):
        return 0
    total = 0
    try:
        for name in os.listdir(path):
            if name.endswith(".safetensors"):
                try:
                    total += os.path.getsize(os.path.join(path, name))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _auto_expandable_off(model_dir: str | None, ctx: int | None,
                         reserve_gib: float = 4.5) -> tuple[bool, str]:
    """Documented `auto` heuristic; returns ``(off, reason)``.

    Only large-context unified loads are considered.  The estimate is the model's
    safetensors bytes plus a deliberately high cache-per-token band, so a model that
    clearly sits below the usable ceiling keeps Kyojin's normal default (expandable on).
    """
    try:
        min_ctx = int(os.environ.get("KYOJIN_EXPANDABLE_SEGMENTS_AUTO_CTX",
                                     str(AUTO_EXPANDABLE_MIN_CTX)))
    except ValueError:
        min_ctx = AUTO_EXPANDABLE_MIN_CTX
    if not ctx or ctx < min_ctx:
        return False, f"context below the {min_ctx} token auto-off gate"
    if not _has_gtt_device():
        return False, "no unified/GTT device detected"
    host_total = _meminfo_bytes("MemTotal")
    model_bytes = _model_dir_bytes(model_dir)
    if not host_total or not model_bytes:
        return False, "model size or host total unavailable"
    usable = min(host_total, int(UNIFIED_USABLE_CEILING_GIB * 1024 ** 3))
    estimate = model_bytes + int(ctx * AUTO_EXPANDABLE_CACHE_BYTES_PER_TOKEN) + int(reserve_gib * 1024 ** 3)
    band = int(AUTO_EXPANDABLE_BAND_GIB * 1024 ** 3)
    off = estimate >= usable - band
    return off, (f"estimate {estimate / 1024 ** 3:.1f} GiB vs usable "
                 f"{usable / 1024 ** 3:.1f} GiB at -c {ctx}")


def _merge_allocator_conf(conf: str | None, value: bool) -> str:
    """Return PYTORCH_CUDA_ALLOC_CONF with expandable_segments forced to ``value``."""
    parts = [p.strip() for p in (conf or "").split(",") if p.strip()]
    parts = [p for p in parts if not p.lower().startswith("expandable_segments:")]
    parts.append(f"expandable_segments:{'True' if value else 'False'}")
    return ",".join(parts)


def apply_allocator_env(settings: "SplitSettings", *, model_dir: str | None = None,
                        ctx: int | None = None, logger = print) -> str:
    """Apply ``--expandable-segments`` before the first ``import exllamav3``.

    Returns the chosen mode (``on`` or ``off``).  ``auto`` never changes anything
    unless the documented large-context unified heuristic fires; an explicit
    ``EXL3_EXPANDABLE_SEGMENTS=0`` is always honoured.
    """
    mode = (getattr(settings, "expandable_segments", None) or "auto").strip().lower()
    if mode not in EXPANDABLE_SEGMENTS_MODES:
        raise ValueError(f"--expandable-segments must be one of {EXPANDABLE_SEGMENTS_MODES}, "
                         f"not {mode!r}")
    if os.environ.get("EXL3_EXPANDABLE_SEGMENTS", "").strip() == "0":
        logger("[allocator] EXL3_EXPANDABLE_SEGMENTS=0: expandable segments off")
        return "off"
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").replace(" ", "").lower()
    if mode == "auto" and "expandable_segments:false" in conf:
        logger("[allocator] expandable segments off (PYTORCH_CUDA_ALLOC_CONF)")
        return "off"
    reason = ""
    if mode == "auto":
        forced = os.environ.get("KYOJIN_EXPANDABLE_SEGMENTS_AUTO_OFF", "").strip().lower()
        if forced in ("1", "on", "true", "yes"):
            off = True
            reason = "KYOJIN_EXPANDABLE_SEGMENTS_AUTO_OFF is set"
        else:
            off, reason = _auto_expandable_off(
                model_dir, ctx,
                reserve_gib=float(getattr(settings, "part0_reserve_gb", 1.2) or 1.2))
        if not off:
            logger(f"[allocator] expandable segments on (auto: {reason})")
            return "on"
    elif mode == "on":
        off = False
        reason = "explicit --expandable-segments on"
    else:
        off = True
        reason = "explicit --expandable-segments off"

    if off:
        os.environ["EXL3_EXPANDABLE_SEGMENTS"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _merge_allocator_conf(
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), False)
        logger(f"[allocator] expandable segments off ({reason}); this recovers about "
               "0.5 GiB of fragmentation, not additional capacity")
        return "off"

    os.environ["EXL3_EXPANDABLE_SEGMENTS"] = "1"
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _merge_allocator_conf(
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), True)
    logger(f"[allocator] expandable segments on ({reason})")
    return "on"

# Per-model unsafe-env presets.  Stage 1 only needs the GLM adapter; the empty
# entries are the declared slots for the later adapters.
SPLIT_PRESETS: dict[str, dict[str, str]] = {
    "none": {},
    "auto": {},
    "glm": {},
    "qwen": {},
    "ds41": {},
}


def parse_part0_layers(value) -> int | str:
    if value is None:
        return 0
    if isinstance(value, int):
        return "auto" if value == -1 else max(0, value)
    v = str(value).strip().lower()
    if v in ("", "0", "off", "none"):
        return 0
    if v == "auto":
        return "auto"
    try:
        return max(0, int(v))
    except ValueError as exc:
        raise ValueError(f"part0 layers must be 0, N or auto, not {value!r}") from exc


@dataclass
class SplitSettings:
    part0_layers: int | str = 0
    part0_device: str = "auto"
    part1_device: str = "auto"
    endcaps: bool = True
    part0_reserve_gb: float = DEFAULT_PART0_RESERVE_GB
    preset: str = "auto"
    split_graphs: str = "auto"
    expandable_segments: str = "auto"
    oom_guard: str | None = None
    oom_guard_reserve_gib: str | None = None
    oom_guard_workspace_gib: str | None = None
    oom_guard_safety_gib: str | None = None
    oom_guard_device_floor: str | None = None

    @property
    def enabled(self) -> bool:
        return self.part0_layers == "auto" or (isinstance(self.part0_layers, int) and self.part0_layers > 0)

    @property
    def auto(self) -> bool:
        return self.part0_layers == "auto"

    @property
    def guard_mode(self) -> str:
        return self.oom_guard or ("auto" if self.enabled else "warn")

    def guard_budget(self):
        """Per-device pre-flight policy, with ``KYOJIN_OOM_GUARD*`` fallbacks."""
        from exllamav3.memory_guard import DeviceBudget
        budget = DeviceBudget.from_env(default_mode = self.guard_mode)
        if self.oom_guard:
            budget.mode = self.oom_guard.strip().lower()
        if self.oom_guard_reserve_gib is not None:
            budget.reserve = self.oom_guard_reserve_gib
        if self.oom_guard_workspace_gib is not None:
            budget.workspace = self.oom_guard_workspace_gib
        if self.oom_guard_safety_gib is not None:
            budget.safety = self.oom_guard_safety_gib
        if self.oom_guard_device_floor is not None:
            budget.device_floor = self.oom_guard_device_floor
        return budget

    def describe(self) -> str:
        return (f"part0_layers={self.part0_layers} part1_device={self.part1_device} "
                f"part0_device={self.part0_device} "
                f"endcaps={'on' if self.endcaps else 'off'} "
                f"reserve={self.part0_reserve_gb:.2f}GiB preset={self.preset} "
                f"graphs={self.split_graphs} allocator={self.expandable_segments} "
                f"oom_guard={self.guard_mode}")


def add_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--part0-layers", default = None,
                        help = "contiguous layer split: first N transformer layers on the part0 device; "
                               "auto = fit from actual bytes; 0 = off (default: KYOJIN_PART0_LAYERS, else 0)")
    parser.add_argument("--part1-device", default = None,
                        help = "part1 device (the rest of the layers): auto, gfxXXXX, cuda:N or a "
                               "UUID/name substring (default: KYOJIN_PART1_DEVICE, else auto = cuda:0)")
    parser.add_argument("--part0-device", default = None,
                        help = "part0 device (the first N layers): auto, gfxXXXX, cuda:N or a UUID/name "
                               "substring (default: KYOJIN_PART0_DEVICE, else auto = the only other device). "
                               "cuda:N is the torch index after HIP_VISIBLE_DEVICES; either device may "
                               "serve part0 or part1")
    parser.add_argument("--endcaps", choices = ("on", "off"), default = None,
                        help = "with a layer split, keep allowlisted embedding/norm/lm_head/MTP/vision "
                               "modules on the part0 device (default: KYOJIN_ENDCAPS, else on)")
    parser.add_argument("--part0-reserve-gb", type = float, default = None,
                        help = "VRAM held as a load-time slab on the part0 device for workspace; a "
                               "tuning value, not a hardware rule (default: KYOJIN_PART0_RESERVE_GB, else 1.2)")
    parser.add_argument("--split-preset", choices = tuple(SPLIT_PRESETS), default = None,
                        help = "per-model split preset (default: KYOJIN_SPLIT_PRESET, else auto)")
    parser.add_argument("--split-graphs", choices = ("auto", "on", "off"), default = None,
                        help = "per-segment graphs (Stage 6); auto forces graphs off for now")
    parser.add_argument("--expandable-segments", choices = EXPANDABLE_SEGMENTS_MODES, default = None,
                        help = "allocator mode, applied before exllamav3/torch import: auto leaves "
                               "expandable segments on except for large-context unified loads, on/off "
                               "override it (default: KYOJIN_EXPANDABLE_SEGMENTS, else auto). Disabling "
                               "recovers about 0.5 GiB of fragmentation, not capacity")
    parser.add_argument("--oom-guard", choices = ("auto", "warn", "off"), default = None,
                        help = "pre-flight per-device memory guard: auto refuses an impossible load "
                               "before allocating, warn only logs, off disables (default: "
                               "KYOJIN_OOM_GUARD, else auto for split and warn for single-device)")
    parser.add_argument("--oom-guard-reserve-gib", type = str, default = None,
                        help = "per-device headroom, e.g. 4.5 or cuda:0=4.5,cuda:1=0.8 "
                               "(default: KYOJIN_OOM_GUARD_RESERVE_GIB, else 8.0 unified / 0.8 dedicated)")
    parser.add_argument("--oom-guard-workspace-gib", type = str, default = None,
                        help = "per-device prefill workspace margin, e.g. cuda:0=4,cuda:1=3.5 "
                               "(default: KYOJIN_OOM_GUARD_WORKSPACE_GIB, else 4.0 bulk / 3.5 other)")
    parser.add_argument("--oom-guard-safety-gib", type = str, default = None,
                        help = "extra per-device safety margin (default: KYOJIN_OOM_GUARD_SAFETY_GIB, else 0)")
    parser.add_argument("--oom-guard-device-floor", type = str, default = None,
                        help = "free-memory floor per device, for the preflight and the supervisor "
                               "(default: KYOJIN_OOM_GUARD_DEVICE_FLOOR, else 0)")


def settings_from_namespace(args) -> SplitSettings:
    raw = getattr(args, "part0_layers", None)
    if raw is None:
        raw = os.environ.get("KYOJIN_PART0_LAYERS", "0")
    endcaps = getattr(args, "endcaps", None)
    if endcaps is None:
        endcaps = os.environ.get("KYOJIN_ENDCAPS", "1") != "0"
    else:
        endcaps = endcaps == "on"
    preset = getattr(args, "split_preset", None) or os.environ.get("KYOJIN_SPLIT_PRESET", "auto")
    graphs = getattr(args, "split_graphs", None) or "auto"
    reserve = getattr(args, "part0_reserve_gb", None)
    if reserve is None:
        reserve = float(os.environ.get("KYOJIN_PART0_RESERVE_GB", str(DEFAULT_PART0_RESERVE_GB)))
    allocator = getattr(args, "expandable_segments", None) \
        or os.environ.get("KYOJIN_EXPANDABLE_SEGMENTS") or "auto"
    return SplitSettings(
        part0_layers = parse_part0_layers(raw),
        part1_device = getattr(args, "part1_device", None) or os.environ.get("KYOJIN_PART1_DEVICE") or "auto",
        part0_device = getattr(args, "part0_device", None) or os.environ.get("KYOJIN_PART0_DEVICE") or "auto",
        endcaps = bool(endcaps),
        part0_reserve_gb = float(reserve),
        preset = (preset or "auto").strip().lower(),
        split_graphs = graphs,
        expandable_segments = (allocator or "auto").strip().lower(),
        oom_guard = getattr(args, "oom_guard", None) or os.environ.get("KYOJIN_OOM_GUARD"),
        oom_guard_reserve_gib = getattr(args, "oom_guard_reserve_gib", None)
            or os.environ.get("KYOJIN_OOM_GUARD_RESERVE_GIB"),
        oom_guard_workspace_gib = getattr(args, "oom_guard_workspace_gib", None)
            or os.environ.get("KYOJIN_OOM_GUARD_WORKSPACE_GIB"),
        oom_guard_safety_gib = getattr(args, "oom_guard_safety_gib", None)
            or os.environ.get("KYOJIN_OOM_GUARD_SAFETY_GIB"),
        oom_guard_device_floor = getattr(args, "oom_guard_device_floor", None)
            or os.environ.get("KYOJIN_OOM_GUARD_DEVICE_FLOOR"),
    )


def apply_split_env(settings: SplitSettings, logger = print) -> list[str]:
    """Force the split-safe environment and return the overridden ``KEY=value`` list."""
    if not settings.enabled:
        return []
    preset = SPLIT_PRESETS.get(settings.preset)
    if preset is None:
        raise ValueError(f"unknown split preset {settings.preset!r}; choose one of {sorted(SPLIT_PRESETS)}")
    forced = dict(SPLIT_UNSAFE_ENV)
    forced.update(GRAPH_ENV_OFF)
    forced.update(preset)
    if settings.split_graphs == "on":
        # Stage 6 has no per-segment capture yet: honouring "on" would let a graph
        # span the bridge and silently break the split.
        logger("[split] --split-graphs on is not implemented yet (Stage 6); forcing graphs off")
    overridden = [f"{k}={os.environ[k]}" for k, v in forced.items()
                  if os.environ.get(k) not in (None, "", v)]
    os.environ.update(forced)
    logger("[split] graphs and deferred residual fusions forced off for split mode"
           + (f"; overrode {', '.join(overridden)}" if overridden else ""))
    return overridden


class SplitController:
    """The adapter-facing wrapper around ``SplitPlan``.

    The controller resolves devices when split is enabled, owns the plan, and
    times/audits each component load.  Split off is a no-op: no device lookup,
    no env change, no map, so the split-off path stays byte-identical.
    """

    def __init__(self, settings: SplitSettings, logger = print):
        self.settings = settings
        self.logger = logger
        self.devices = None
        self.plan = None
        if settings.enabled:
            from exllamav3.device_map import resolve_devices
            self.devices = resolve_devices(settings.part0_device, settings.part1_device)

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def build_plan(self):
        if not self.enabled:
            return None
        if self.plan is None:
            from exllamav3.device_map import SplitPlan
            self.plan = SplitPlan(
                self.devices,
                endcaps = self.settings.endcaps,
                part0_layers = self.settings.part0_layers,
                part0_reserve_gb = self.settings.part0_reserve_gb,
                preset = self.settings.preset,
                guard_budget = self.settings.guard_budget(),
                logger = self.logger,
            )
        return self.plan

    def add(self, component: str, model, config = None, cache = None,
            policy: "ComponentPolicy | None" = None, load_order: int | None = None):
        plan = self.build_plan()
        if plan is None:
            return None
        if policy is None:
            from exllamav3.device_map import component_policy
            policy = component_policy(component)
        cmap = plan.add(component, model, config = config, cache = cache,
                        policy = policy, load_order = load_order)
        plan.apply_policies()
        plan.install_bridges()
        return cmap

    def build(self) -> None:
        if self.plan is not None:
            self.plan.build()
            self.plan.install_bridges()

    def map(self, component: str):
        return self.plan.map(component) if self.plan is not None else None

    def engage_budget(self) -> None:
        if self.plan is not None:
            self.plan.engage_budget()

    def load(self, component: str, model, **load_kwargs) -> bool:
        """Load one component through its map; returns False when split is off."""
        if not self.enabled:
            return False
        if self.plan is not None and self.plan.auto and not self.plan._built and not self.plan.records:
            self.build()
        cmap = self.map(component)
        if cmap is None:
            raise RuntimeError(f"split: component {component!r} was not declared")
        # The owner map holds the part0 reserve for the whole load sequence; repeated
        # calls are idempotent (ComponentMap.engage_budget).
        self.plan.engage_budget()
        t0 = time.perf_counter()
        model.load(device_map = cmap, **load_kwargs)
        self.plan.record_load(component, time.perf_counter() - t0)
        return True

    def finalize(self, model, cache, extra: list[tuple[str, object, object]] | None = None,
                 ctx: int | None = None) -> None:
        if self.plan is None:
            return
        self.plan.report(model, cache, extra = extra, ctx = ctx)
        self.plan.release_budget()
