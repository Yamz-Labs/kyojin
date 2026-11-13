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

    @property
    def enabled(self) -> bool:
        return self.part0_layers == "auto" or (isinstance(self.part0_layers, int) and self.part0_layers > 0)

    @property
    def auto(self) -> bool:
        return self.part0_layers == "auto"

    def describe(self) -> str:
        return (f"part0_layers={self.part0_layers} part1_device={self.part1_device} "
                f"part0_device={self.part0_device} "
                f"endcaps={'on' if self.endcaps else 'off'} "
                f"reserve={self.part0_reserve_gb:.2f}GiB preset={self.preset} "
                f"graphs={self.split_graphs}")


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
    return SplitSettings(
        part0_layers = parse_part0_layers(raw),
        part1_device = getattr(args, "part1_device", None) or os.environ.get("KYOJIN_PART1_DEVICE") or "auto",
        part0_device = getattr(args, "part0_device", None) or os.environ.get("KYOJIN_PART0_DEVICE") or "auto",
        endcaps = bool(endcaps),
        part0_reserve_gb = float(reserve),
        preset = (preset or "auto").strip().lower(),
        split_graphs = graphs,
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
