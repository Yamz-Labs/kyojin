from __future__ import annotations

"""
Contiguous layer split: resolve the two devices, declare per-component policies and
map model modules to them.

The map is deliberately independent of the architecture: modules carrying a
``layer_idx`` are transformer layers and are assigned to the front device for
indices ``0 .. front_layers - 1`` and to the main device for the rest.  Modules
without a ``layer_idx`` are endcaps (embedding, hyper-connection drivers, final
norm, lm_head, and the separate MTP/vision components).

The universal core separates *policy* from *placement*:

- ``ComponentPolicy`` says whether a component may use the front device and which
  non-layer modules are safe to move there.  The text prologue/epilogue allowlist
  is norm/hc/lm_head-like modules only; anything the policy does not recognise is
  an unknown non-layer module and stays on the main device.
- ``ComponentMap`` resolves the contiguous boundary for one component.
- ``SplitPlan`` owns every component of a process (target, MTP, vision, extra),
  the front reserve, auto-N and the load/placement audit.  Adapters declare their
  components; the core contains no model-specific placement logic.

Endcaps ON puts the allowlisted text endcaps on the front device.  ENDCAPS OFF
lets the allowlisted prologue follow layer 0 and the epilogue follow the last
layer.  ``prefer_cpu`` modules, such as the GLM token embedding, always stay on
the CPU: their upload path is part of the embedding design, not a tile to move.
"""

import os
from dataclasses import dataclass, field

import torch

from .util.arch_list import device_arch

GIB = 1024 ** 3

# Text prologue/epilogue modules that are safe to place on the front device.  The
# list is by class name and key suffix, not by model: an adapter can extend it in
# its ComponentPolicy.  Modules outside this list default to the main device.
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


# ---------------------------------------------------------------------------------------------------------
# Component policy
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen = True)
class ComponentPolicy:
    """Placement policy for one component of a model process.

    ``kind`` is one of ``text``, ``mtp``, ``vision``, ``draft``, ``aux`` or
    ``unknown``.  Whole-component kinds (MTP/vision/draft) put every module on
    the front when endcaps are on, because they have no trunk boundary of their
    own.  Auxiliary components never visit the front implicitly.
    """

    name: str
    kind: str = "aux"
    front_eligible: bool = False
    endcaps_front: bool = True
    whole_component: bool = False
    endcap_types: frozenset[str] = frozenset()
    endcap_key_suffixes: tuple[str, ...] = ()
    arch_gate: frozenset[str] | None = None
    unsafe_env: tuple[tuple[str, str], ...] = ()

    def allows_front(self, arch: str | None) -> bool:
        if not self.front_eligible or arch is None:
            return False
        return self.arch_gate is None or arch in self.arch_gate


def component_policy(name: str, **overrides) -> ComponentPolicy:
    """Default policy for a component name; adapters may override any field."""
    key = (name or "unknown").strip().lower()
    if key == "text":
        defaults = dict(
            name = name, kind = "text", front_eligible = True, endcaps_front = True,
            endcap_types = TEXT_ENDCAP_TYPES, endcap_key_suffixes = TEXT_ENDCAP_SUFFIXES)
    elif key in WHOLE_COMPONENT_KINDS:
        defaults = dict(
            name = name, kind = key, front_eligible = True, endcaps_front = True,
            whole_component = True)
    else:
        defaults = dict(name = name, kind = "aux", front_eligible = False,
                        endcaps_front = False)
    defaults.update(overrides)
    return ComponentPolicy(**defaults)


def _type_name(module) -> str:
    return type(module).__name__


def _key_matches(key: str, suffixes) -> bool:
    for s in suffixes:
        if key == s or key.endswith("." + s):
            return True
    return False


def is_front_endcap(module, policy: ComponentPolicy) -> bool:
    """True for a non-layer module the text policy allows on the front device."""
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
    and perturbs the main device's kernel choices (measured at 128K), while a
    slab only removes bytes from the pool and releases them without changing the
    allocator's policy.
    """

    main: torch.device
    front: torch.device | None
    front_layers: int
    endcaps: bool
    component: str = "text"
    layer_count: int = 0
    policy: ComponentPolicy | None = None
    main_arch: str = ""
    front_arch: str | None = None
    reserve_bytes: dict[int, int] = field(default_factory = dict)
    front_endcap_keys: tuple[str, ...] = ()
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
        front_layers: int,
        endcaps: bool,
        devices: Devices,
        component: str = "text",
        front_reserve_gb: float = 1.2,
        main_reserve_gb: float = 0.0,
        config = None,
        cache = None,
        policy: ComponentPolicy | None = None,
        load_order: int = 0,
    ) -> "ComponentMap":
        policy = policy or component_policy(component)
        return cls.from_policy(model, front_layers, endcaps, devices, policy,
                               front_reserve_gb = front_reserve_gb,
                               main_reserve_gb = main_reserve_gb,
                               config = config, cache = cache, load_order = load_order)

    @classmethod
    def from_policy(
        cls,
        model,
        front_layers: int | str,
        endcaps: bool,
        devices: Devices,
        policy: ComponentPolicy,
        front_reserve_gb: float = 1.2,
        main_reserve_gb: float = 0.0,
        config = None,
        cache = None,
        load_order: int = 0,
    ) -> "ComponentMap":
        modules = list(model.modules)
        layers = [m for m in modules if getattr(m, "layer_idx", None) is not None]
        n_layers = len(layers)
        auto = front_layers in ("auto", -1) or (isinstance(front_layers, str) and front_layers.strip().lower() == "auto")
        front_layers = 0 if auto else int(front_layers)
        endcaps = bool(endcaps and policy.endcaps_front)
        if policy.whole_component:
            # These components are endcaps in their entirety: they have no trunk
            # boundary of their own, so ENDCAPS OFF leaves them on the main device.
            front_layers = n_layers if endcaps else 0
        front_layers = max(0, min(int(front_layers), n_layers))
        if devices.front is None or not policy.allows_front(devices.front_arch):
            front_layers = 0
        positions = {id(m): i for i, m in enumerate(modules)}
        first = positions[id(layers[0])] if layers else 0
        last = positions[id(layers[-1])] if layers else -1
        stc = getattr(config, "stc", None) if config is not None else None

        # The front reserve caps the small card while its weights load.  It is released
        # before the first main module (see Model._load_mapped): the main device has ample
        # headroom and is never capped.  A component other than the budget owner clears
        # reserve_bytes after construction (SplitPlan does this).
        reserve: dict[int, int] = {}
        if devices.front is not None and front_layers > 0 and front_reserve_gb > 0:
            reserve[devices.front.index] = max(0, int(front_reserve_gb * GIB))
        if devices.main is not None and main_reserve_gb > 0:
            reserve[devices.main.index] = max(0, int(main_reserve_gb * GIB))

        layer_bytes = _layer_bytes(model, stc)
        cache_by_layer = _cache_bytes_by_layer(cache)
        front_endcaps = tuple(
            m.key for m in modules
            if getattr(m, "layer_idx", None) is None and is_front_endcap(m, policy)
        )
        unknown = tuple(
            m.key for m in modules
            if getattr(m, "layer_idx", None) is None
            and not getattr(m, "caps", {}).get("prefer_cpu")
            and not is_front_endcap(m, policy)
        )
        size = (sum(layer_bytes.values()) +
                sum(_module_bytes(m, stc) for m in modules if getattr(m, "layer_idx", None) is None))

        return cls(
            main = devices.main,
            front = devices.front,
            front_layers = front_layers,
            endcaps = bool(endcaps) and front_layers > 0,
            component = policy.name,
            layer_count = n_layers,
            policy = policy,
            main_arch = devices.main_arch,
            front_arch = devices.front_arch,
            reserve_bytes = reserve,
            front_endcap_keys = front_endcaps,
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
        return self.front is not None and self.front_layers > 0

    @property
    def used_devices(self) -> list[torch.device]:
        out = [self.main]
        if self.enabled and self.front is not None:
            out.append(self.front)
        return out

    @property
    def policy_or_default(self) -> ComponentPolicy:
        return self.policy or component_policy(self.component)

    def _endcap_allowed(self, module) -> bool:
        return is_front_endcap(module, self.policy_or_default)

    def device_for(self, module) -> torch.device:
        if getattr(module, "caps", {}).get("prefer_cpu"):
            return torch.device("cpu")
        idx = getattr(module, "layer_idx", None)
        if idx is not None:
            if not self.enabled:
                return self.main
            if self.policy_or_default.whole_component:
                return self.front
            return self.front if idx < self.front_layers else self.main
        if not self.enabled:
            return self.main
        if self.policy_or_default.whole_component:
            # MTP/vision/draft have no trunk boundary: all their modules share the
            # component's endcaps verdict.
            return self.front if self.endcaps else self.main
        if not self._endcap_allowed(module):
            # Unknown / auxiliary non-layer module: always safe, never moved implicitly.
            return self.main
        if not self.policy_or_default.endcaps_front:
            return self.main
        if self.endcaps:
            return self.front
        # ENDCAPS OFF: allowlisted prologue follows layer 0, epilogue follows the last layer.
        pos = self._positions.get(id(module), 0)
        if pos < self._first_layer_pos:
            return self.front
        if pos > self._last_layer_pos:
            return self.front if self.front_layers >= self.layer_count else self.main
        return self.main

    # -- load-time budget --------------------------------------------------

    def engage_budget(self) -> None:
        """Hold the front reserve as a slab (never a process-wide allocator cap)."""
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

    def release_front_budget(self) -> None:
        """Release the front slab before the first main module loads.

        The slab is dropped; the caching allocator keeps the freed block for the
        forward workspace, so the headroom is preserved without a cap.
        """
        if not getattr(self, "_budget_engaged", False):
            return
        if self.front is None:
            return
        self._reserve_slabs.pop(self.front.index, None)
        self._budget_engaged = any(i != self.front.index and r > 0 for i, r in self.reserve_bytes.items())

    def release_budget(self) -> None:
        self._reserve_slabs.clear()
        self._budget_engaged = False

    # -- sizing / reporting ------------------------------------------------

    def layer_weight_bytes(self, idx: int) -> int:
        return int(self._layer_bytes.get(idx, 0))

    def layer_cache_bytes(self, idx: int) -> int:
        return int(self._cache_by_layer.get(idx, 0))

    def front_cost(self, n: int, endcaps: bool | None = None) -> int:
        """Estimated front bytes for n text layers plus the policy's front endcaps."""
        n = max(0, min(n, self.layer_count))
        endcaps = self.endcaps if endcaps is None else bool(endcaps)
        endcaps = endcaps and self.policy_or_default.endcaps_front
        cost = sum(self.layer_weight_bytes(i) for i in range(n))
        cost += sum(self.layer_cache_bytes(i) for i in range(n))
        if not endcaps:
            return cost
        stc = getattr(self.config, "stc", None)
        if self.policy_or_default.whole_component:
            return cost + sum(_module_bytes(m, stc)
                              for m in self._module_refs if getattr(m, "layer_idx", None) is None
                              and not getattr(m, "caps", {}).get("prefer_cpu"))
        allowed = set(self.front_endcap_keys)
        return cost + sum(_module_bytes(m, stc) for m in self._module_refs if m.key in allowed)

    def describe(self) -> str:
        parts = [f"{self.component}: layers 0..{self.front_layers - 1} front, "
                 f"{self.front_layers}..{self.layer_count - 1} main"]
        if self.endcaps:
            parts.append(f"endcaps front ({len(self.front_endcap_keys)} modules)")
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
    front_layers: int
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
        front_layers: int | str = 0,
        front_reserve_gb: float = 1.2,
        main_reserve_gb: float = 0.0,
        preset: str = "auto",
        auto_cap: int | None = None,
        logger = print,
    ):
        self.devices = devices
        self.endcaps = bool(endcaps)
        self.requested_front_layers = front_layers
        self.auto = isinstance(front_layers, str) and front_layers.strip().lower() == "auto" or front_layers == -1
        self.front_layers = 0 if self.auto else max(0, int(front_layers or 0))
        self.front_reserve_gb = float(front_reserve_gb)
        self.main_reserve_gb = float(main_reserve_gb)
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
        front_layers: int | str | None = None,
        load_order: int | None = None,
    ) -> ComponentMap:
        was_built = self._built
        policy = policy or component_policy(component)
        layer = self.requested_front_layers if front_layers is None else front_layers
        self._decls[component] = (model, config, cache, policy, layer, load_order)
        if config is not None:
            self.register_config(config)
        cmap = ComponentMap.from_policy(
            model, layer, self.endcaps, self.devices, policy,
            front_reserve_gb = self.front_reserve_gb,
            main_reserve_gb = self.main_reserve_gb,
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
            self.front_layers = n
            self.logger(f"[split] auto-N: {n} front layer(s) "
                        f"(budget {self._auto_budget() / GIB:.2f} GiB, "
                        f"reserve {self.front_reserve_gb:.2f} GiB)")
        for name, (model, config, cache, policy, layer, load_order) in self._decls.items():
            if self.auto and policy.kind == "text":
                layer = self.front_layers
            self.maps[name] = ComponentMap.from_policy(
                model, layer, self.endcaps, self.devices, policy,
                front_reserve_gb = self.front_reserve_gb,
                main_reserve_gb = self.main_reserve_gb,
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
        if self.devices.front is None:
            return 0
        total = int(torch.cuda.get_device_properties(self.devices.front).total_memory)
        safety = int(float(os.environ.get("KYOJIN_AUTON_SAFETY_GB", "0.25")) * GIB)
        return max(0, total - int(self.front_reserve_gb * GIB) - safety)

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
                cost = other + cmap.front_cost(n, self.endcaps)
            else:
                # Prologue follows layer 0; epilogue only when every layer is front.
                cost = other + cmap.front_cost(n, self.endcaps)
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
            if getattr(m, "layer_idx", None) is not None or not is_front_endcap(m, cmap.policy_or_default):
                continue
            if cmap._positions.get(id(m), 0) < cmap._first_layer_pos:
                out += _module_bytes(m, stc)
        return out

    def _epilogue_bytes(self, cmap: ComponentMap) -> int:
        stc = getattr(cmap.config, "stc", None)
        out = 0
        for m in cmap._module_refs:
            if getattr(m, "layer_idx", None) is not None or not is_front_endcap(m, cmap.policy_or_default):
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
        # the owner releases the slab at its first main module, a later component must
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
            install_bridge(config, self.devices.main, self.devices.front)

    def record_load(self, component: str, seconds: float, warnings = ()) -> None:
        cmap = self.maps.get(component)
        if cmap is None:
            return
        cache_bytes = _cache_bytes_for(cmap.cache, cmap.layer_count)
        self.records.append(ComponentLoad(
            component = component, seconds = float(seconds),
            front_layers = cmap.front_layers, layer_count = cmap.layer_count,
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
            return cmap.main
        if cmap.policy_or_default.whole_component:
            return cmap.front if cmap.endcaps else cmap.main
        return cmap.front if idx < cmap.front_layers else cmap.main

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
                self.logger(f"[split] {name}: unknown non-layer modules kept on the main device: "
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
            front = f"0..{r.front_layers - 1}" if r.front_layers else "-"
            main = f"{r.front_layers}..{r.layer_count - 1}" if r.layer_count else "-"
            self.logger(f"[split]   {r.component:7s} {r.seconds:7.1f}s  front {front:>10s}  "
                        f"main {main:>10s}  weights {r.weight_bytes / GIB:7.2f} GiB  "
                        f"cache {r.cache_bytes / GIB:6.2f} GiB  devices "
                        f"{', '.join(str(x) for x in d.used_devices)}")
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
    if device_map.front is not None:
        try:
            p = torch.cuda.get_device_properties(device_map.front)
            total = p.total_memory
        except Exception:                                    # noqa: BLE001
            total = 0
        reserve = device_map.reserve_bytes.get(device_map.front.index, 0)
        print(f"[split] front {str(device_map.front)} {device_map.front_arch}: "
              f"{total / GIB:.2f} GiB total, reserve {reserve / GIB:.2f} GiB, "
              f"budget {(total - reserve) / GIB:.2f} GiB", flush = True)
    cpu = [m.key for m in model.modules if getattr(m, "caps", {}).get("prefer_cpu")]
    if cpu:
        print(f"[split] cpu-resident modules: {', '.join(cpu)}", flush = True)
