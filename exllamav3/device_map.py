from __future__ import annotations

"""
Contiguous layer split: resolve the two devices, declare per-component policies and
map model modules to them.

``part0`` is the first part of the layer stack (the first ``part0_layers``
transformer layers plus its endcaps); ``part1`` is the rest of the component.
Devices are selected by position, never by architecture: ``part1`` defaults to
``cuda:0`` and ``part0`` to the only other visible device when exactly two are
present.  A part0-only selector keeps working when it lands on ``cuda:0``: part1
auto then takes the other device.  Either position accepts ``cuda:N``,
``gfxXXXX``, a device UUID or a device-name substring, and either device may
serve either part.  ``cuda:N`` is the torch index after ``HIP_VISIBLE_DEVICES``,
not the ``rocm-smi`` index.  More than two visible devices require an explicit
selector.

The map itself is deliberately independent of the architecture: modules carrying a
``layer_idx`` are transformer layers and are assigned to ``part0`` for indices
``0 .. part0_layers - 1`` and to ``part1`` for the rest.  Modules without a
``layer_idx`` are endcaps (embedding, hyper-connection drivers, final norm,
lm_head, and the separate MTP/vision components).

The universal core separates *policy* from *placement*:

- ``ComponentPolicy`` says whether a component may use ``part0`` and which
  non-layer modules are safe to move there.  The text prologue/epilogue allowlist
  is norm/hc/lm_head-like modules only; anything the policy does not recognise is
  an unknown non-layer module and stays on ``part1``.
- ``ComponentMap`` resolves the contiguous boundary for one component.
- ``SplitPlan`` owns every component of a process (target, MTP, vision, extra),
  the part0 reserve, auto-N and the load/placement audit.  Adapters declare their
  components; the core contains no model-specific placement logic.

Endcaps ON puts the allowlisted text endcaps on ``part0``.  ENDCAPS OFF lets the
allowlisted prologue follow layer 0 and the epilogue follow the last layer.
``prefer_cpu`` modules, such as the GLM token embedding, always stay on the CPU:
their upload path is part of the embedding design, not a tile to move.
"""

import os
from dataclasses import dataclass, field

import torch

from .util.arch_list import device_arch

GIB = 1024 ** 3

# Default load-time slab on the part0 device.  A tuning value, not an architecture
# assumption: size it to the part0 card.  Stage 5 derives it from the device total.
DEFAULT_PART0_RESERVE_GB = 1.2
DEFAULT_PART1_DEVICE = "auto"
DEFAULT_PART0_DEVICE = "auto"

# Text prologue/epilogue modules that are safe to place on part0.  The list is by
# class name and key suffix, not by model: an adapter can extend it in its
# ComponentPolicy.  Modules outside this list default to part1.
TEXT_ENDCAP_TYPES = frozenset((
    "RMSNorm", "GatedRMSNorm", "LayerNorm", "HyperHead", "ExpandStreams",
))
TEXT_ENDCAP_SUFFIXES = (
    "norm", "ln_f", "final_norm", "output_norm", "lm_head", "hc_head", "hc_expand",
)

# Components that are endcaps in their entirety (no trunk boundary of their own).
WHOLE_COMPONENT_KINDS = frozenset(("mtp", "vision", "draft"))


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
class DeviceInfo:
    """One visible device as the selector sees it.

    ``arch`` is a ROCm arch name when the runtime exposes one; an unknown or
    non-ROCm device keeps an empty string so selection by ``cuda:N``/name still
    works.  The inventory is injectable so selection tests need no GPU.
    """

    index: int
    name: str = ""
    arch: str = ""
    uuid: str = ""

    def describe(self) -> str:
        parts = [f"cuda:{self.index}"]
        if self.arch:
            parts.append(self.arch)
        if self.name:
            parts.append(self.name)
        return " ".join(parts)


@dataclass(frozen = True)
class Devices:
    part1: torch.device
    part0: torch.device | None
    part1_arch: str
    part0_arch: str | None
    inventory: tuple[DeviceInfo, ...] = ()


def _device_info(index: int) -> DeviceInfo:
    name = ""
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:                                        # noqa: BLE001 - inventory is best effort
        pass
    try:
        arch = _arch_of(index)
    except Exception:                                        # noqa: BLE001 - selection does not need an arch
        arch = ""
    try:
        uuid = _device_uuid(index)
    except Exception:                                        # noqa: BLE001
        uuid = ""
    return DeviceInfo(index = index, name = name, arch = arch, uuid = uuid)


def probe_devices() -> list[DeviceInfo]:
    """Inventory the visible devices; the only GPU-dependent step in selection."""
    if not torch.cuda.is_available():
        raise RuntimeError("layer split requires two HIP/CUDA devices")
    return [_device_info(i) for i in range(torch.cuda.device_count())]


def _format_inventory(infos: list[DeviceInfo]) -> str:
    return ", ".join(d.describe() for d in infos) or "none"


def _select_device(req: str, role: str, infos: list[DeviceInfo], env_name: str) -> int:
    req = (req or "").strip().lower()
    indexes = {d.index for d in infos}
    if req.startswith("cuda:"):
        try:
            index = int(req.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"{role} device selector {req!r} is not a valid cuda:N") from exc
        if index not in indexes:
            raise RuntimeError(f"{role} device {req!r}: no such device; available: "
                               f"{_format_inventory(infos)}")
        return index
    if req.startswith("gfx"):
        matches = [d.index for d in infos if (d.arch or "").lower() == req]
    else:
        key = req.replace("-", "")
        matches = [
            d.index for d in infos
            if (key and key in (d.uuid or "").lower().replace("-", ""))
            or (req and req in (d.name or "").lower())
        ]
    if len(matches) > 1:
        raise RuntimeError(f"{role} device {req!r} matched several devices; use cuda:N or "
                           f"a more specific selector; available: {_format_inventory(infos)}")
    if not matches:
        raise RuntimeError(f"{role} device {req!r} did not match any device; available: "
                           f"{_format_inventory(infos)}. Set {env_name}=cuda:N "
                           f"(or gfxXXXX/uuid/name).")
    return matches[0]


def resolve_devices(
    part0: str | None = None,
    part1: str | None = None,
    devices: list[DeviceInfo] | None = None,
) -> Devices:
    """Resolve the part0/part1 pair by position, never by architecture.

    ``part1`` (or KYOJIN_PART1_DEVICE) defaults to ``auto`` = ``cuda:0``.  ``part0``
    (or KYOJIN_PART0_DEVICE) defaults to the only other visible device when exactly
    two are visible; with one device the split cannot run, and with more than two an
    explicit selector is required.  If part0 is explicit and occupies ``cuda:0``
    while part1 is ``auto`` with exactly two devices visible, part1 takes the other
    device so a part0-only selector keeps working.  Both selectors accept
    ``cuda:N``, ``gfxXXXX``, a device UUID or a device-name substring; either
    position may be either device.

    ``devices`` injects a synthetic inventory; production callers omit it and probe
    the runtime.  Selection itself does not depend on an architecture.
    """
    infos = list(devices) if devices is not None else probe_devices()
    if len(infos) < 2:
        raise RuntimeError(
            f"layer split needs two devices; visible: {_format_inventory(infos)}. "
            "Unset KYOJIN_PART0_LAYERS / --part0-layers to run on one device.")

    part1_req = part1 if part1 is not None else os.environ.get("KYOJIN_PART1_DEVICE", DEFAULT_PART1_DEVICE)
    part0_req = part0 if part0 is not None else os.environ.get("KYOJIN_PART0_DEVICE", DEFAULT_PART0_DEVICE)
    part1_req = (part1_req or DEFAULT_PART1_DEVICE).strip().lower()
    part0_req = (part0_req or DEFAULT_PART0_DEVICE).strip().lower()
    part1_auto = part1_req in ("", "auto")

    if part1_auto:
        part1_i = min(d.index for d in infos)
    else:
        part1_i = _select_device(part1_req, "part1", infos, "KYOJIN_PART1_DEVICE")

    if part0_req in ("", "auto"):
        if len(infos) != 2:
            raise RuntimeError(
                f"part0 device is 'auto' but {len(infos)} devices are visible; set "
                f"KYOJIN_PART0_DEVICE (or --part0-device) to one of: {_format_inventory(infos)}")
        part0_i = next(d.index for d in infos if d.index != part1_i)
    else:
        part0_i = _select_device(part0_req, "part0", infos, "KYOJIN_PART0_DEVICE")
        if part1_auto and part0_i == part1_i:
            # An explicit part0 selector must keep working when it lands on cuda:0:
            # part1 auto takes the only other device.  With more devices the choice
            # is ambiguous, so ask for the part1 selector instead of guessing.
            if len(infos) == 2:
                part1_i = next(d.index for d in infos if d.index != part0_i)
            else:
                raise RuntimeError(
                    f"part1 device is 'auto' (cuda:{part1_i}) and equals the explicit part0; "
                    f"with {len(infos)} devices visible set KYOJIN_PART1_DEVICE (or --part1-device)")

    if part0_i == part1_i:
        raise RuntimeError(f"layer split part0 and part1 resolve to the same device "
                           f"(cuda:{part1_i}); choose a different --part0-device/--part1-device")
    by_index = {d.index: d for d in infos}
    part1_info, part0_info = by_index[part1_i], by_index[part0_i]
    return Devices(
        part1 = torch.device("cuda", part1_i),
        part0 = torch.device("cuda", part0_i),
        part1_arch = part1_info.arch or "",
        part0_arch = part0_info.arch or "",
        inventory = tuple(infos),
    )


# ---------------------------------------------------------------------------------------------------------
# Component policy
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen = True)
class ComponentPolicy:
    """Placement policy for one component of a model process.

    ``kind`` is one of ``text``, ``mtp``, ``vision``, ``draft``, ``aux`` or
    ``unknown``.  Whole-component kinds (MTP/vision/draft) put every module on
    part0 when endcaps are on, because they have no trunk boundary of their own.
    Auxiliary components never visit part0 implicitly.
    """

    name: str
    kind: str = "aux"
    part0_eligible: bool = False
    endcaps_part0: bool = True
    whole_component: bool = False
    endcap_types: frozenset[str] = frozenset()
    endcap_key_suffixes: tuple[str, ...] = ()
    arch_gate: frozenset[str] | None = None
    unsafe_env: tuple[tuple[str, str], ...] = ()

    def allows_part0(self, arch: str | None) -> bool:
        if not self.part0_eligible or arch is None:
            return False
        return self.arch_gate is None or arch in self.arch_gate


def component_policy(name: str, **overrides) -> ComponentPolicy:
    """Default policy for a component name; adapters may override any field."""
    key = (name or "unknown").strip().lower()
    if key == "text":
        defaults = dict(
            name = name, kind = "text", part0_eligible = True, endcaps_part0 = True,
            endcap_types = TEXT_ENDCAP_TYPES, endcap_key_suffixes = TEXT_ENDCAP_SUFFIXES)
    elif key in WHOLE_COMPONENT_KINDS:
        defaults = dict(
            name = name, kind = key, part0_eligible = True, endcaps_part0 = True,
            whole_component = True)
    else:
        defaults = dict(name = name, kind = "aux", part0_eligible = False,
                        endcaps_part0 = False)
    defaults.update(overrides)
    return ComponentPolicy(**defaults)


def _type_name(module) -> str:
    return type(module).__name__


def _key_matches(key: str, suffixes) -> bool:
    for s in suffixes:
        if key == s or key.endswith("." + s):
            return True
    return False


def is_part0_endcap(module, policy: ComponentPolicy) -> bool:
    """True for a non-layer module the text policy allows on part0."""
    if getattr(module, "caps", {}).get("prefer_cpu"):
        return False
    if getattr(module, "caps", {}).get("logits_output"):
        return True
    if _type_name(module) in policy.endcap_types:
        return True
    return _key_matches(getattr(module, "key", ""), policy.endcap_key_suffixes)


def _module_bytes(module, stc = None) -> int:
    """Actual stored bytes of a top-level module, before it is loaded.

    The safetensors collection knows the file span of every tensor, which is the
    number auto-N must use.  A synthetic model without an stc falls back to its
    element count at fp16.
    """
    if stc is not None and hasattr(stc, "get_tensor_sizes"):
        try:
            sizes = stc.get_tensor_sizes(module.key)
            if sum(sizes):
                return int(sum(sizes))
            # HyperHead's tensors are underscore-joined (hc_head_fn) and a tied
            # lm_head has no lm_head.* span; both need the module's own names.
            compile_sizes = getattr(module, "get_compile_sizes", None)
            if compile_sizes is not None:
                sizes = compile_sizes(stc)
                if sum(sizes):
                    return int(sum(sizes))
            alt = getattr(module, "alt_key", None)
            if alt:
                keys = alt if isinstance(alt, (list, tuple)) else (alt,)
                total = sum(sum(stc.get_tensor_sizes(k)) for k in keys)
                if total:
                    return int(total)
        except Exception:                                    # noqa: BLE001 - fall back to a numel estimate
            pass
    try:
        return int(module.weights_numel()) * 2
    except Exception:                                        # noqa: BLE001
        return 0


def _layer_bytes(model, stc = None) -> dict[int, int]:
    out: dict[int, int] = {}
    for module in model.modules:
        idx = getattr(module, "layer_idx", None)
        if idx is not None:
            out[idx] = out.get(idx, 0) + _module_bytes(module, stc)
    return out


def _cache_bytes_for(cache, n_layers: int, upto: int | None = None) -> int:
    """Cache/recurrent bytes whose layer index is below ``upto`` (all when None)."""
    if cache is None:
        return 0
    total = 0
    for key, cl in list(getattr(cache, "layers", {}).items()):
        idx = key[0] if isinstance(key, tuple) else key
        if upto is not None and idx >= upto:
            continue
        try:
            total += int(cl.storage_size())
        except Exception:                                    # noqa: BLE001
            pass
    for key, rl in list(getattr(cache, "recurrent_layers", {}).items()):
        idx = key[0] if isinstance(key, tuple) else key
        if upto is not None and idx >= upto:
            continue
        try:
            total += int(rl.storage_size())
        except Exception:                                    # noqa: BLE001
            pass
    return total


def _cache_bytes_by_layer(cache) -> dict[int, int]:
    out: dict[int, int] = {}
    if cache is None:
        return out
    for key, cl in list(getattr(cache, "layers", {}).items()):
        idx = key[0] if isinstance(key, tuple) else key
        try:
            out[idx] = out.get(idx, 0) + int(cl.storage_size())
        except Exception:                                    # noqa: BLE001
            pass
    for key, rl in list(getattr(cache, "recurrent_layers", {}).items()):
        idx = key[0] if isinstance(key, tuple) else key
        try:
            out[idx] = out.get(idx, 0) + int(rl.storage_size())
        except Exception:                                    # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------------------------------------
# Component map
# ---------------------------------------------------------------------------------------------------------

@dataclass
class ComponentMap:
    """Contiguous layer placement for one component.

    ``reserve_bytes`` is a per-device *slab* held for the duration of the load,
    not a process-wide allocator cap: a cap changes the ROCm allocator's layout
    and perturbs the part1 device's kernel choices (measured at 128K), while a
    slab only removes bytes from the pool and releases them without changing the
    allocator's policy.
    """

    part1: torch.device
    part0: torch.device | None
    part0_layers: int
    endcaps: bool
    component: str = "text"
    layer_count: int = 0
    policy: ComponentPolicy | None = None
    part1_arch: str = ""
    part0_arch: str | None = None
    reserve_bytes: dict[int, int] = field(default_factory = dict)
    part0_endcap_keys: tuple[str, ...] = ()
    unknown_keys: tuple[str, ...] = ()
    size_bytes: int = 0
    load_order: int = 0
    config: object | None = field(default = None, repr = False)
    cache: object | None = field(default = None, repr = False)
    _positions: dict[int, int] = field(default_factory = dict, repr = False)
    _first_layer_pos: int = 0
    _last_layer_pos: int = -1
    _layer_bytes: dict[int, int] = field(default_factory = dict, repr = False)
    _cache_by_layer: dict[int, int] = field(default_factory = dict, repr = False)
    _module_refs: tuple = field(default_factory = tuple, repr = False)
    _reserve_slabs: dict[int, torch.Tensor] = field(default_factory = dict, repr = False)

    @classmethod
    def from_model(
        cls,
        model,
        part0_layers: int,
        endcaps: bool,
        devices: Devices,
        component: str = "text",
        part0_reserve_gb: float = DEFAULT_PART0_RESERVE_GB,
        part1_reserve_gb: float = 0.0,
        config = None,
        cache = None,
        policy: ComponentPolicy | None = None,
        load_order: int = 0,
    ) -> "ComponentMap":
        policy = policy or component_policy(component)
        return cls.from_policy(model, part0_layers, endcaps, devices, policy,
                               part0_reserve_gb = part0_reserve_gb,
                               part1_reserve_gb = part1_reserve_gb,
                               config = config, cache = cache, load_order = load_order)

    @classmethod
    def from_policy(
        cls,
        model,
        part0_layers: int | str,
        endcaps: bool,
        devices: Devices,
        policy: ComponentPolicy,
        part0_reserve_gb: float = DEFAULT_PART0_RESERVE_GB,
        part1_reserve_gb: float = 0.0,
        config = None,
        cache = None,
        load_order: int = 0,
    ) -> "ComponentMap":
        modules = list(model.modules)
        layers = [m for m in modules if getattr(m, "layer_idx", None) is not None]
        n_layers = len(layers)
        auto = part0_layers in ("auto", -1) or (isinstance(part0_layers, str) and part0_layers.strip().lower() == "auto")
        part0_layers = 0 if auto else int(part0_layers)
        endcaps = bool(endcaps and policy.endcaps_part0)
        if policy.whole_component:
            # These components are endcaps in their entirety: they have no trunk
            # boundary of their own, so ENDCAPS OFF leaves them on part1.
            part0_layers = n_layers if endcaps else 0
        part0_layers = max(0, min(int(part0_layers), n_layers))
        if devices.part0 is None or not policy.allows_part0(devices.part0_arch):
            part0_layers = 0
        positions = {id(m): i for i, m in enumerate(modules)}
        first = positions[id(layers[0])] if layers else 0
        last = positions[id(layers[-1])] if layers else -1
        stc = getattr(config, "stc", None) if config is not None else None

        # The part0 reserve caps the small card while its weights load.  It is released
        # before the first part1 module (see Model._load_mapped): the part1 device has
        # ample headroom and is never capped.  A component other than the budget owner
        # clears reserve_bytes after construction (SplitPlan does this).
        reserve: dict[int, int] = {}
        if devices.part0 is not None and part0_layers > 0 and part0_reserve_gb > 0:
            reserve[devices.part0.index] = max(0, int(part0_reserve_gb * GIB))
        if devices.part1 is not None and part1_reserve_gb > 0:
            reserve[devices.part1.index] = max(0, int(part1_reserve_gb * GIB))

        layer_bytes = _layer_bytes(model, stc)
        cache_by_layer = _cache_bytes_by_layer(cache)
        part0_endcaps = tuple(
            m.key for m in modules
            if getattr(m, "layer_idx", None) is None and is_part0_endcap(m, policy)
        )
        unknown = tuple(
            m.key for m in modules
            if getattr(m, "layer_idx", None) is None
            and not getattr(m, "caps", {}).get("prefer_cpu")
            and not is_part0_endcap(m, policy)
        )
        size = (sum(layer_bytes.values()) +
                sum(_module_bytes(m, stc) for m in modules if getattr(m, "layer_idx", None) is None))

        return cls(
            part1 = devices.part1,
            part0 = devices.part0,
            part0_layers = part0_layers,
            endcaps = bool(endcaps) and part0_layers > 0,
            component = policy.name,
            layer_count = n_layers,
            policy = policy,
            part1_arch = devices.part1_arch,
            part0_arch = devices.part0_arch,
            reserve_bytes = reserve,
            part0_endcap_keys = part0_endcaps,
            unknown_keys = unknown,
            size_bytes = int(size),
            load_order = load_order,
            config = config,
            cache = cache,
            _positions = positions,
            _first_layer_pos = first,
            _last_layer_pos = last,
            _layer_bytes = layer_bytes,
            _cache_by_layer = cache_by_layer,
            _module_refs = tuple(modules),
        )

    # -- placement ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.part0 is not None and self.part0_layers > 0

    @property
    def used_devices(self) -> list[torch.device]:
        out = [self.part1]
        if self.enabled and self.part0 is not None:
            out.append(self.part0)
        return out

    @property
    def placed_devices(self) -> list[torch.device]:
        """Devices that actually hold tensors for this component.

        ``used_devices`` is the configured part1/part0 pair; a whole component with
        endcaps on has no tensors on part1 even though that pair still contains it.
        CPU-resident modules (the shared embedding) are reported separately.
        """
        out: list[torch.device] = []
        for module in self._module_refs:
            if getattr(module, "caps", {}).get("prefer_cpu"):
                continue
            device = self.device_for(module)
            if device.type == "cuda" and device not in out:
                out.append(device)
        return sorted(out, key = lambda d: d.index or 0)

    @property
    def policy_or_default(self) -> ComponentPolicy:
        return self.policy or component_policy(self.component)

    def _endcap_allowed(self, module) -> bool:
        return is_part0_endcap(module, self.policy_or_default)

    def device_for(self, module) -> torch.device:
        if getattr(module, "caps", {}).get("prefer_cpu"):
            return torch.device("cpu")
        idx = getattr(module, "layer_idx", None)
        if idx is not None:
            if not self.enabled:
                return self.part1
            if self.policy_or_default.whole_component:
                return self.part0
            return self.part0 if idx < self.part0_layers else self.part1
        if not self.enabled:
            return self.part1
        if self.policy_or_default.whole_component:
            # MTP/vision/draft have no trunk boundary: all their modules share the
            # component's endcaps verdict.
            return self.part0 if self.endcaps else self.part1
        if not self._endcap_allowed(module):
            # Unknown / auxiliary non-layer module: always safe, never moved implicitly.
            return self.part1
        if not self.policy_or_default.endcaps_part0:
            return self.part1
        if self.endcaps:
            return self.part0
        # ENDCAPS OFF: allowlisted prologue follows layer 0, epilogue follows the last layer.
        pos = self._positions.get(id(module), 0)
        if pos < self._first_layer_pos:
            return self.part0
        if pos > self._last_layer_pos:
            return self.part0 if self.part0_layers >= self.layer_count else self.part1
        return self.part1

    # -- load-time budget --------------------------------------------------

    def engage_budget(self) -> None:
        """Hold the part0 reserve as a slab (never a process-wide allocator cap)."""
        if getattr(self, "_budget_engaged", False):
            return
        for i, reserve in self.reserve_bytes.items():
            if reserve <= 0:
                continue
            try:
                # An inert slab: it removes bytes from the pool without changing the
                # allocator's configuration or the numerics of any later kernel.
                self._reserve_slabs[i] = torch.empty(reserve, dtype = torch.uint8,
                                                     device = torch.device("cuda", i))
            except Exception as exc:                         # noqa: BLE001
                print(f"[split] warning: could not hold the {reserve / GIB:.2f} GiB reserve "
                      f"on cuda:{i} ({type(exc).__name__}: {exc}); continuing without it",
                      flush = True)
        self._budget_engaged = True

    def release_part0_budget(self) -> None:
        """Release the part0 slab before the first part1 module loads.

        The slab is dropped; the caching allocator keeps the freed block for the
        forward workspace, so the headroom is preserved without a cap.
        """
        if not getattr(self, "_budget_engaged", False):
            return
        if self.part0 is None:
            return
        self._reserve_slabs.pop(self.part0.index, None)
        self._budget_engaged = any(i != self.part0.index and r > 0 for i, r in self.reserve_bytes.items())

    def release_budget(self) -> None:
        self._reserve_slabs.clear()
        self._budget_engaged = False

    # -- sizing / reporting ------------------------------------------------

    def layer_weight_bytes(self, idx: int) -> int:
        return int(self._layer_bytes.get(idx, 0))

    def layer_cache_bytes(self, idx: int) -> int:
        return int(self._cache_by_layer.get(idx, 0))

    def part0_cost(self, n: int, endcaps: bool | None = None) -> int:
        """Estimated part0 bytes for n text layers plus the policy's part0 endcaps."""
        n = max(0, min(n, self.layer_count))
        endcaps = self.endcaps if endcaps is None else bool(endcaps)
        endcaps = endcaps and self.policy_or_default.endcaps_part0
        cost = sum(self.layer_weight_bytes(i) for i in range(n))
        cost += sum(self.layer_cache_bytes(i) for i in range(n))
        if not endcaps:
            return cost
        stc = getattr(self.config, "stc", None)
        if self.policy_or_default.whole_component:
            return cost + sum(_module_bytes(m, stc)
                              for m in self._module_refs if getattr(m, "layer_idx", None) is None
                              and not getattr(m, "caps", {}).get("prefer_cpu"))
        allowed = set(self.part0_endcap_keys)
        return cost + sum(_module_bytes(m, stc) for m in self._module_refs if m.key in allowed)

    def describe(self) -> str:
        parts = [f"{self.component}: layers 0..{self.part0_layers - 1} part0, "
                 f"{self.part0_layers}..{self.layer_count - 1} part1"]
        if self.endcaps:
            parts.append(f"endcaps part0 ({len(self.part0_endcap_keys)} modules)")
        else:
            parts.append("endcaps follow neighbours")
        return "; ".join(parts)


# Backwards-compatible name for the Stage 0 callers and the strix_halo harnesses.
LayerDeviceMap = ComponentMap


# ---------------------------------------------------------------------------------------------------------
# Split plan
# ---------------------------------------------------------------------------------------------------------

@dataclass
class ComponentLoad:
    component: str
    seconds: float
    part0_layers: int
    layer_count: int
    device_map: ComponentMap | None
    weight_bytes: int = 0
    cache_bytes: int = 0
    warnings: tuple[str, ...] = ()


class SplitPlan:
    """Every component the process places, the reserve and the placement audit."""

    def __init__(
        self,
        devices: Devices,
        endcaps: bool = True,
        part0_layers: int | str = 0,
        part0_reserve_gb: float = DEFAULT_PART0_RESERVE_GB,
        part1_reserve_gb: float = 0.0,
        preset: str = "auto",
        auto_cap: int | None = None,
        logger = print,
    ):
        self.devices = devices
        self.endcaps = bool(endcaps)
        self.requested_part0_layers = part0_layers
        self.auto = isinstance(part0_layers, str) and part0_layers.strip().lower() == "auto" or part0_layers == -1
        self.part0_layers = 0 if self.auto else max(0, int(part0_layers or 0))
        self.part0_reserve_gb = float(part0_reserve_gb)
        self.part1_reserve_gb = float(part1_reserve_gb)
        self.preset = (preset or "auto").strip().lower()
        self.auto_cap = None if auto_cap is None else max(0, int(auto_cap))
        self.logger = logger
        self.maps: dict[str, ComponentMap] = {}
        self._decls: dict[str, tuple] = {}
        self.configs: list[object] = []
        self.records: list[ComponentLoad] = []
        self.audit: dict[str, object] = {}
        self._owner: str | None = None
        self._budget_engaged_once = False
        self._built = False

    # -- declarations ------------------------------------------------------

    def add(
        self,
        component: str,
        model,
        config = None,
        cache = None,
        policy: ComponentPolicy | None = None,
        part0_layers: int | str | None = None,
        load_order: int | None = None,
    ) -> ComponentMap:
        was_built = self._built
        policy = policy or component_policy(component)
        layer = self.requested_part0_layers if part0_layers is None else part0_layers
        self._decls[component] = (model, config, cache, policy, layer, load_order)
        if config is not None:
            self.register_config(config)
        cmap = ComponentMap.from_policy(
            model, layer, self.endcaps, self.devices, policy,
            part0_reserve_gb = self.part0_reserve_gb,
            part1_reserve_gb = self.part1_reserve_gb,
            config = config, cache = cache,
            load_order = load_order if load_order is not None else len(self._decls),
        )
        self.maps[component] = cmap
        if self._owner is None or (policy.kind == "text" and self._owner != "text"):
            self._owner = component
        if component != self._owner:
            # Only the target/text map owns the load reserve.
            cmap.reserve_bytes.clear()
        # A late component (vision/MTP after the target load) must not reset the
        # already-chosen auto-N plan.
        self._built = was_built
        return cmap

    def register_config(self, config) -> None:
        if config is not None and all(c is not config for c in self.configs):
            self.configs.append(config)

    # -- sizing ------------------------------------------------------------

    def build(self) -> "SplitPlan":
        if not self._decls:
            self._built = True
            return self
        text = next((k for k, d in self._decls.items() if d[3].kind == "text"), None)
        if self.auto:
            n = self._auto_n(text)
            if self.auto_cap is not None:
                n = min(n, self.auto_cap)
            self.part0_layers = n
            self.logger(f"[split] auto-N: {n} part0 layer(s) "
                        f"(budget {self._auto_budget() / GIB:.2f} GiB, "
                        f"reserve {self.part0_reserve_gb:.2f} GiB tuning slab)")
        for name, (model, config, cache, policy, layer, load_order) in self._decls.items():
            if self.auto and policy.kind == "text":
                layer = self.part0_layers
            self.maps[name] = ComponentMap.from_policy(
                model, layer, self.endcaps, self.devices, policy,
                part0_reserve_gb = self.part0_reserve_gb,
                part1_reserve_gb = self.part1_reserve_gb,
                config = config, cache = cache,
                load_order = load_order if load_order is not None else 0,
            )
        self._assign_owner()
        self.apply_policies()
        self._built = True
        return self

    def apply_policies(self) -> None:
        """Apply (and log) the named preset and every adapter-declared unsafe env lever."""
        if self.preset not in ("auto", "none"):
            from .serve_split import SPLIT_PRESETS
            preset = SPLIT_PRESETS.get(self.preset)
            if preset is None:
                raise ValueError(f"unknown split preset {self.preset!r}; choose one of {sorted(SPLIT_PRESETS)}")
            for key, value in preset.items():
                old = os.environ.get(key)
                if old != value:
                    self.logger(f"[split] preset {self.preset}: {key}={value}"
                                + (f" (was {old})" if old is not None else ""))
                os.environ[key] = value
        for name, cmap in self.maps.items():
            for key, value in cmap.policy_or_default.unsafe_env:
                old = os.environ.get(key)
                if old != value:
                    self.logger(f"[split] {name}: unsafe env {key}={value}"
                                + (f" (was {old})" if old is not None else ""))
                os.environ[key] = value

    def _assign_owner(self) -> None:
        owner = "text" if "text" in self.maps else (self._owner or next(iter(self.maps), None))
        self._owner = owner
        for name, cmap in self.maps.items():
            if name != owner:
                # Only the target/text map owns the load reserve.
                cmap.reserve_bytes.clear()

    def _auto_budget(self) -> int:
        if self.devices.part0 is None:
            return 0
        total = int(torch.cuda.get_device_properties(self.devices.part0).total_memory)
        # Measured stopgap, not an architecture rule: 2 GiB covers the 32K MTP+vision
        # warm-up on a 16 GiB part0 (1.0 GiB still chose N=6 and OOMed).  Stage 5 replaces
        # it with a context-aware workspace/headroom budget.
        safety = int(float(os.environ.get("KYOJIN_AUTON_SAFETY_GB", "2.0")) * GIB)
        return max(0, total - int(self.part0_reserve_gb * GIB) - safety)

    def _auto_n(self, text: str | None) -> int:
        if text is None or text not in self.maps:
            return 0
        cmap = self.maps[text]
        if cmap.layer_count == 0:
            return 0
        budget = self._auto_budget()
        # Components that are endcaps in their entirety and already declared.
        other = sum(
            m.size_bytes + _cache_bytes_for(m.cache, m.layer_count)
            for name, m in self.maps.items()
            if name != text and m.enabled and m.policy_or_default.whole_component and self.endcaps)
        for n in range(cmap.layer_count, -1, -1):
            if self.endcaps:
                cost = other + cmap.part0_cost(n, self.endcaps)
            else:
                # Prologue follows layer 0; epilogue only when every layer is part0.
                cost = other + cmap.part0_cost(n, self.endcaps)
                if n > 0:
                    cost += self._prologue_bytes(cmap)
                if n >= cmap.layer_count:
                    cost += self._epilogue_bytes(cmap)
            if cost <= budget:
                return n
        return 0

    def _prologue_bytes(self, cmap: ComponentMap) -> int:
        stc = getattr(cmap.config, "stc", None)
        out = 0
        for m in cmap._module_refs:
            if getattr(m, "layer_idx", None) is not None or not is_part0_endcap(m, cmap.policy_or_default):
                continue
            if cmap._positions.get(id(m), 0) < cmap._first_layer_pos:
                out += _module_bytes(m, stc)
        return out

    def _epilogue_bytes(self, cmap: ComponentMap) -> int:
        stc = getattr(cmap.config, "stc", None)
        out = 0
        for m in cmap._module_refs:
            if getattr(m, "layer_idx", None) is not None or not is_part0_endcap(m, cmap.policy_or_default):
                continue
            if cmap._positions.get(id(m), 0) > cmap._last_layer_pos:
                out += _module_bytes(m, stc)
        return out

    # -- runtime -----------------------------------------------------------

    def map(self, component: str) -> ComponentMap | None:
        return self.maps.get(component)

    @property
    def enabled(self) -> bool:
        return any(m.enabled for m in self.maps.values())

    def engage_budget(self) -> None:
        # One reserve for the whole load sequence (target + drafter + vision).  After
        # the owner releases the slab at its first part1 module, a later component must
        # not allocate a second one.
        if self._budget_engaged_once:
            return
        if self._owner is not None and self._owner in self.maps:
            self.maps[self._owner].engage_budget()
        self._budget_engaged_once = True

    def install_bridges(self) -> None:
        """Install one bridge per config (target, separate draft config, vision, extra)."""
        if not self.enabled:
            return
        from .bridge import install_bridge
        for config in self.configs:
            install_bridge(config, part1 = self.devices.part1, part0 = self.devices.part0)

    def record_load(self, component: str, seconds: float, warnings = ()) -> None:
        cmap = self.maps.get(component)
        if cmap is None:
            return
        cache_bytes = _cache_bytes_for(cmap.cache, cmap.layer_count)
        self.records.append(ComponentLoad(
            component = component, seconds = float(seconds),
            part0_layers = cmap.part0_layers, layer_count = cmap.layer_count,
            device_map = cmap, weight_bytes = int(cmap.size_bytes),
            cache_bytes = cache_bytes, warnings = tuple(warnings)))

    def audit_caches(self) -> dict[str, list[str]]:
        """Check every cache/recurrent tensor against the component that owns its layer."""
        issues: dict[str, list[str]] = {}
        for name, cmap in self.maps.items():
            cache = cmap.cache
            if cache is None:
                continue
            bad: list[str] = []
            for key, cl in list(getattr(cache, "layers", {}).items()):
                idx = key[0] if isinstance(key, tuple) else key
                expected = self._expected_layer_device(cmap, idx)
                actual = getattr(cl, "device", None) or getattr(getattr(cl, "attention", None), "device", None)
                if actual is not None and expected is not None and torch.device(actual) != torch.device(expected):
                    bad.append(f"cache layer {idx}: {actual} != {expected}")
            for key, rl in list(getattr(cache, "recurrent_layers", {}).items()):
                idx = key[0] if isinstance(key, tuple) else key
                expected = self._expected_layer_device(cmap, idx)
                actual = (getattr(rl, "device", None) or
                          getattr(getattr(rl, "module", None), "device", None))
                if actual is not None and expected is not None and torch.device(actual) != torch.device(expected):
                    bad.append(f"recurrent layer {idx}: {actual} != {expected}")
            if bad:
                issues[name] = bad
        self.audit = issues
        return issues

    @staticmethod
    def _expected_layer_device(cmap: ComponentMap, idx: int):
        if not cmap.enabled:
            return cmap.part1
        if cmap.policy_or_default.whole_component:
            return cmap.part0 if cmap.endcaps else cmap.part1
        return cmap.part0 if idx < cmap.part0_layers else cmap.part1

    def report(self, model, cache, extra: list[tuple[str, object, object]] | None = None,
               ctx: int | None = None) -> None:
        if not self.enabled:
            return
        self.install_bridges()
        if self._owner in self.maps:
            print_split_table(model, cache, self.maps[self._owner], extra = extra, ctx = ctx)
        self.print_component_table()
        for name, cmap in self.maps.items():
            if cmap.unknown_keys and cmap.policy_or_default.kind == "text":
                self.logger(f"[split] {name}: unknown non-layer modules kept on the part1 device: "
                            f"{', '.join(cmap.unknown_keys)}")
        issues = self.audit_caches()
        for name, bad in issues.items():
            self.logger(f"[split] {name}: cache placement mismatch: " + "; ".join(bad[:8]))
        if not issues:
            self.logger("[split] cache/recurrent placement: all tensors follow their layer")
        bridge = next((getattr(c, "device_bridge", None) for c in self.configs
                       if getattr(c, "device_bridge", None) is not None), None)
        if bridge is not None:
            self.logger(f"[split] {bridge.describe()}")
            self.logger(bridge.summary())
        else:
            self.logger("[split] bridge disabled (KYOJIN_BRIDGE=0); crossings use to_device()")

    def print_component_table(self) -> None:
        if not self.records:
            return
        self.logger("[split] component load table:")
        for r in sorted(self.records, key = lambda x: (x.device_map.load_order if x.device_map else 0, x.component)):
            d = r.device_map
            if d is None:
                continue
            part0 = f"0..{r.part0_layers - 1}" if r.part0_layers else "-"
            part1 = (f"{r.part0_layers}..{r.layer_count - 1}"
                     if r.part0_layers < r.layer_count else "-")
            placed = ", ".join(str(x) for x in d.placed_devices) or "-"
            self.logger(f"[split]   {r.component:7s} {r.seconds:7.1f}s  part0 {part0:>10s}  "
                        f"part1 {part1:>10s}  weights {r.weight_bytes / GIB:7.2f} GiB  "
                        f"cache {r.cache_bytes / GIB:6.2f} GiB  devices {placed}")
            for w in r.warnings:
                self.logger(f"[split]     warning: {w}")

    def release_budget(self) -> None:
        for cmap in self.maps.values():
            cmap.release_budget()


# ---------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------

def _cache_bytes(cache) -> dict[torch.device, int]:
    table: dict[torch.device, int] = {}
    if cache is None:
        return table
    for cl in list(getattr(cache, "layers", {}).values()):
        dev = getattr(cl, "device", None) or getattr(getattr(cl, "attention", None), "device", None)
        if dev is None:
            continue
        try:
            table[dev] = table.get(dev, 0) + int(cl.storage_size())
        except Exception:                                    # noqa: BLE001 - a state type without a byte count
            pass
    for rl in list(getattr(cache, "recurrent_layers", {}).values()):
        dev = (getattr(rl, "device", None) or
               getattr(getattr(rl, "module", None), "device", None))
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
    device_map: ComponentMap,
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

    print(f"[split] {device_map.describe()}" + (f"; -c {ctx}" if ctx else ""), flush = True)
    for dev in sorted(device_map.used_devices, key = lambda d: (d.type, d.index or 0)):
        cache_bytes = cache_b.get(dev, 0)
        try:
            allocated = torch.cuda.memory_allocated(dev)
            peak = torch.cuda.max_memory_allocated(dev)
        except Exception:                                    # noqa: BLE001 - CPU-only test/table
            allocated = peak = 0
        weights = allocated - cache_bytes
        print(f"[split] {str(dev):9s} weights {_gib(weights)}  cache {_gib(cache_bytes)}  "
              f"allocated {_gib(allocated)}  peak {_gib(peak)}", flush = True)
    if device_map.part0 is not None:
        try:
            p = torch.cuda.get_device_properties(device_map.part0)
            total = p.total_memory
        except Exception:                                    # noqa: BLE001
            total = 0
        reserve = device_map.reserve_bytes.get(device_map.part0.index, 0)
        print(f"[split] part0 {str(device_map.part0)} {device_map.part0_arch}: "
              f"{total / GIB:.2f} GiB total, reserve {reserve / GIB:.2f} GiB, "
              f"budget {(total - reserve) / GIB:.2f} GiB", flush = True)
    cpu = [m.key for m in model.modules if getattr(m, "caps", {}).get("prefer_cpu")]
    if cpu:
        print(f"[split] cpu-resident modules: {', '.join(cpu)}", flush = True)
