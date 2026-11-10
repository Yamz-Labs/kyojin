from __future__ import annotations

"""
Contiguous layer split: resolve the two devices and map model modules to them.

The map is deliberately independent of the architecture: modules carrying a
``layer_idx`` are transformer layers and are assigned to the front device for
indices ``0 .. front_layers - 1`` and to the main device for the rest.  Modules
without a ``layer_idx`` are the endcaps (embedding, hyper-connection drivers,
final norm, lm_head, and the separate MTP/vision components).

Endcaps ON puts the non-CPU endcaps on the front device; ENDCAPS OFF lets them
follow their natural neighbour (prologue with layer 0, epilogue with the last
layer).  ``prefer_cpu`` modules, such as the GLM token embedding, always stay on
the CPU: their upload path is part of the embedding design, not a tile to move.
"""

import os
from dataclasses import dataclass, field

import torch

from .util.arch_list import device_arch

GIB = 1024 ** 3


def _arch_of(index: int) -> str:
    # The shared helper owns the ROCm arch name and its card-name fallback.
    return device_arch(index)


def _device_uuid(index: int) -> str:
    p = torch.cuda.get_device_properties(index)
    u = getattr(p, "uuid", None) or getattr(p, "device_uuid", None) or ""
    if isinstance(u, bytes):
        return u.hex().lower()
    return str(u).lower().replace("-", "")


@dataclass(frozen = True)
class Devices:
    main: torch.device
    front: torch.device | None
    main_arch: str
    front_arch: str | None


def resolve_devices(front: str | None = None) -> Devices:
    """Resolve main/front by architecture.

    ``front`` (or KYOJIN_FRONT_DEVICE) accepts ``auto``, ``gfx1201``, ``cuda:N``
    or a device UUID / name substring.  Auto selects the first gfx120/gfx12
    device that is not the gfx1151 main device.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("layer split requires a HIP/CUDA device")
    n = torch.cuda.device_count()
    arch = {i: _arch_of(i) for i in range(n)}
    main_i = next((i for i, a in arch.items() if a == "gfx1151"), 0)
    req = (front if front is not None else os.environ.get("KYOJIN_FRONT_DEVICE", "auto") or "auto").strip().lower()

    if req in ("", "auto"):
        front_i = next((i for i, a in arch.items() if a.startswith("gfx12") and i != main_i), None)
    elif req.startswith("cuda:"):
        front_i = int(req.split(":", 1)[1])
        if front_i < 0 or front_i >= n:
            raise RuntimeError(f"KYOJIN_FRONT_DEVICE={req}: no such device")
    elif req.startswith("gfx"):
        front_i = next((i for i, a in arch.items() if a == req), None)
    else:
        key = req.replace("-", "")
        front_i = next((i for i in range(n)
                        if key and (key in _device_uuid(i) or req in torch.cuda.get_device_properties(i).name.lower())), None)

    if front_i is None:
        raise RuntimeError(
            f"layer split requested but no front device matched {req!r}; "
            f"devices: {[(i, a) for i, a in arch.items()]}. "
            "Set KYOJIN_FRONT_DEVICE=gfx1201 (or cuda:N)."
        )
    if front_i == main_i:
        raise RuntimeError("layer split front device resolves to the main device")
    return Devices(
        main = torch.device("cuda", main_i),
        front = torch.device("cuda", front_i),
        main_arch = arch[main_i],
        front_arch = arch[front_i],
    )


@dataclass
class LayerDeviceMap:
    main: torch.device
    front: torch.device | None
    front_layers: int
    endcaps: bool
    component: str = "text"
    layer_count: int = 0
    main_arch: str = ""
    front_arch: str | None = None
    reserve_bytes: dict[int, int] = field(default_factory = dict)
    endcap_keys: tuple[str, ...] = ()
    _positions: dict[int, int] = field(default_factory = dict, repr = False)
    _first_layer_pos: int = 0
    _last_layer_pos: int = -1

    @classmethod
    def from_model(
        cls,
        model,
        front_layers: int,
        endcaps: bool,
        devices: Devices,
        component: str = "text",
        front_reserve_gb: float = 1.2,
        main_reserve_gb: float = 0.0,
    ) -> "LayerDeviceMap":
        modules = list(model.modules)
        layers = [m for m in modules if getattr(m, "layer_idx", None) is not None]
        n_layers = len(layers)
        if component in ("mtp", "vision"):
            # These components are endcaps in their entirety: they have no trunk
            # boundary of their own, so ENDCAPS OFF leaves them on the main device.
            front_layers = n_layers if endcaps else 0
        front_layers = max(0, min(int(front_layers), n_layers))
        if devices.front is None:
            front_layers = 0
        positions = {id(m): i for i, m in enumerate(modules)}
        first = positions[id(layers[0])] if layers else 0
        last = positions[id(layers[-1])] if layers else -1

        # The front reserve caps the small card while its weights load.  It is released before
        # the first main module (see Model._load_mapped): on ROCm the memory-fraction cap
        # changes the allocator's behavior process-wide and perturbs the main device's kernel
        # choices, which would make split outputs differ from a single-device baseline even
        # with zero front layers.  The main device has ample headroom and is never capped.
        reserve: dict[int, int] = {}
        if devices.front is not None and front_layers > 0 and front_reserve_gb > 0:
            reserve[devices.front.index] = max(0, int(front_reserve_gb * GIB))
        if devices.main is not None and main_reserve_gb > 0:
            reserve[devices.main.index] = max(0, int(main_reserve_gb * GIB))

        return cls(
            main = devices.main,
            front = devices.front,
            front_layers = front_layers,
            endcaps = bool(endcaps) and front_layers > 0,
            component = component,
            layer_count = n_layers,
            main_arch = devices.main_arch,
            front_arch = devices.front_arch,
            reserve_bytes = reserve,
            endcap_keys = tuple(m.key for m in modules if getattr(m, "layer_idx", None) is None),
            _positions = positions,
            _first_layer_pos = first,
            _last_layer_pos = last,
        )

    @property
    def enabled(self) -> bool:
        return self.front is not None and self.front_layers > 0

    @property
    def boundary(self) -> int | None:
        if not self.enabled or self.layer_count == 0:
            return None
        return self.front_layers

    @property
    def used_devices(self) -> list[torch.device]:
        out = [self.main]
        if self.enabled and self.front is not None:
            out.append(self.front)
        return out

    def device_for(self, module) -> torch.device:
        if getattr(module, "caps", {}).get("prefer_cpu"):
            return torch.device("cpu")
        idx = getattr(module, "layer_idx", None)
        if idx is not None:
            return self.front if idx < self.front_layers else self.main
        if not self.enabled:
            return self.main
        if self.endcaps:
            return self.front
        # ENDCAPS OFF: prologue follows layer 0, epilogue follows the last layer.
        pos = self._positions.get(id(module), 0)
        if pos < self._first_layer_pos:
            return self.front
        return self.front if self.front_layers >= self.layer_count else self.main

    def engage_budget(self) -> None:
        """Apply the per-device reserve once, before the first component loads."""
        if getattr(self, "_budget_engaged", False):
            return
        from .util.memory import set_memory_fraction_reserve
        for i, reserve in self.reserve_bytes.items():
            if reserve > 0:
                set_memory_fraction_reserve(reserve, i)
        self._budget_engaged = True

    def release_front_budget(self) -> None:
        """Release the front load cap before the first main module loads.

        A process-wide allocator cap on ROCm changes the main device's kernel/allocator
        behavior; the split must not carry that into the main segment (it would perturb
        outputs relative to a single-device run).
        """
        if not getattr(self, "_budget_engaged", False):
            return
        if self.front is None or self.front.index not in self.reserve_bytes:
            return
        from .util.memory import unset_memory_fraction
        unset_memory_fraction([self.front.index])
        self._budget_engaged = any(
            i != self.front.index and r > 0 for i, r in self.reserve_bytes.items())

    def release_budget(self) -> None:
        if not getattr(self, "_budget_engaged", False):
            return
        from .util.memory import unset_memory_fraction
        unset_memory_fraction(list(self.reserve_bytes.keys()))
        self._budget_engaged = False

    def describe(self) -> str:
        parts = [f"{self.component}: layers 0..{self.front_layers - 1} front, "
                 f"{self.front_layers}..{self.layer_count - 1} main"]
        if self.endcaps:
            parts.append(f"endcaps front ({len(self.endcap_keys)} modules)")
        else:
            parts.append("endcaps follow neighbours")
        return "; ".join(parts)


def _cache_bytes(cache) -> dict[torch.device, int]:
    table: dict[torch.device, int] = {}
    if cache is None:
        return table
    for cl in list(getattr(cache, "layers", {}).values()):
        dev = getattr(getattr(cl, "attention", None), "device", None)
        if dev is None:
            continue
        table[dev] = table.get(dev, 0) + int(cl.storage_size())
    for rl in list(getattr(cache, "recurrent_layers", {}).values()):
        dev = getattr(getattr(rl, "attention", None), "device", None)
        if dev is None:
            continue
        try:
            table[dev] = table.get(dev, 0) + int(rl.storage_size())
        except Exception:                                    # noqa: BLE001 - a state type without a byte count
            pass
    return table


def _gib(n: int) -> str:
    return f"{n / GIB:7.2f} GiB"


def print_split_table(
    model,
    cache,
    device_map: LayerDeviceMap,
    extra: list[tuple[str, object, object]] | None = None,
    ctx: int | None = None,
) -> None:
    """Print the per-device memory table that Gate 3 uses as ground truth.

    Weights are reported as the device allocator sees them (which is what the
    budget actually has to hold); the cache column is the sum of the allocated
    cache layer storage on that device.
    """
    cache_b: dict[torch.device, int] = {}
    for c in [cache] + [c for _, _, c in (extra or []) if c is not None]:
        for dev, n in _cache_bytes(c).items():
            cache_b[dev] = cache_b.get(dev, 0) + n

    print(f"[split] {device_map.describe()}" + (f"; -c {ctx}" if ctx else ""), flush=True)
    for dev in sorted(device_map.used_devices, key=lambda d: (d.type, d.index or 0)):
        cache_bytes = cache_b.get(dev, 0)
        allocated = torch.cuda.memory_allocated(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        weights = allocated - cache_bytes
        print(f"[split] {str(dev):9s} weights {_gib(weights)}  cache {_gib(cache_bytes)}  "
              f"allocated {_gib(allocated)}  peak {_gib(peak)}", flush=True)
    if device_map.front is not None:
        p = torch.cuda.get_device_properties(device_map.front)
        reserve = device_map.reserve_bytes.get(device_map.front.index, 0)
        print(f"[split] front {str(device_map.front)} {device_map.front_arch}: "
              f"{p.total_memory / GIB:.2f} GiB total, reserve {reserve / GIB:.2f} GiB, "
              f"budget {(p.total_memory - reserve) / GIB:.2f} GiB", flush=True)
    cpu = [m.key for m in model.modules if getattr(m, "caps", {}).get("prefer_cpu")]
    if cpu:
        print(f"[split] cpu-resident modules: {', '.join(cpu)}", flush=True)
