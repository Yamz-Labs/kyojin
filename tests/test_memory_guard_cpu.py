"""CPU tests for the pre-flight memory guard and the allocator-mode entry point.

The guard must be testable without a GPU: capacity arithmetic, per-device parsing,
refusal/warn/off behavior and the allocator env merge are pure Python.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def mg():
    module = _load("kyojin_memory_guard", ROOT / "exllamav3" / "memory_guard.py")
    yield module
    sys.modules.pop("kyojin_memory_guard", None)


@pytest.fixture(scope = "module")
def guard_mod():
    module = _load("kyojin_guard", ROOT / "exllamav3" / "guard.py")
    yield module
    sys.modules.pop("kyojin_guard", None)


@pytest.fixture(scope = "module")
def split_mod():
    module = _load("kyojin_serve_split_cpu", ROOT / "exllamav3" / "serve_split.py")
    yield module
    sys.modules.pop("kyojin_serve_split_cpu", None)


def test_parse_device_spec_scalar_and_list(mg):
    assert mg.parse_device_spec(None) == (None, {})
    assert mg.parse_device_spec("4.5") == (4.5, {})
    assert mg.parse_device_spec("gfx1151=4.5,gfx1201=0.8") == (None, {"gfx1151": 4.5, "gfx1201": 0.8})
    assert mg.parse_device_spec("default=1,cuda:1=2") == (1.0, {"cuda:1": 2.0})
    with pytest.raises(ValueError):
        mg.parse_device_spec("gfx1151")
    with pytest.raises(ValueError):
        mg.parse_device_spec("gfx1151=big")


def test_resolve_device_spec_uses_the_split_selector(mg):
    infos = [SimpleNamespace(index = 0), SimpleNamespace(index = 1)]
    seen = []

    def select(key):
        seen.append(key)
        return {"gfx1151": 1, "gfx1201": 0}[key]

    out = mg.resolve_device_spec("gfx1151=4.5,gfx1201=0.8", infos = infos, select = select,
                                 default = lambda _i: 1.0)
    assert out == {0: 0.8, 1: 4.5}
    assert seen == ["gfx1151", "gfx1201"]


def test_budget_defaults_follow_unified_and_bulk(mg):
    infos = [SimpleNamespace(index = 0), SimpleNamespace(index = 1)]
    budget = mg.DeviceBudget(mode = "auto")
    resolved = budget.resolve(infos, lambda key: int(key.split(":")[1]),
                              unified = {0: False, 1: True}, bulk_index = 1)
    assert resolved["reserve"] == {0: 0.8, 1: 4.5}
    assert resolved["workspace"] == {0: 1.5, 1: 2.0}
    assert resolved["safety"] == {0: 0.0, 1: 0.0}
    assert resolved["device_floor"] == {0: 0.0, 1: 0.0}


def test_explicit_per_device_values_override_defaults(mg):
    infos = [SimpleNamespace(index = 0), SimpleNamespace(index = 1)]
    budget = mg.DeviceBudget(mode = "warn", reserve = "gfx1151=6",
                             workspace = "cuda:0=1.5", safety = "2",
                             device_floor = "gfx1151=1")
    resolved = budget.resolve(infos, lambda key: {"gfx1151": 1, "cuda:0": 0}[key],
                              unified = {0: False, 1: True}, bulk_index = 1)
    assert resolved["reserve"] == {0: 0.8, 1: 6.0}
    assert resolved["workspace"] == {0: 1.5, 1: 2.0}
    assert resolved["safety"] == {0: 2.0, 1: 2.0}
    assert resolved["device_floor"] == {0: 0.0, 1: 1.0}


def _footprint(mg, **overrides):
    values = {"index": 0, "arch": "gfx1151", "unified": True,
              "total_bytes": 118 * GIB, "free_bytes": 118 * GIB,
              "weights_bytes": 0, "cache_bytes": 0, "slab_bytes": 0,
              "reserve_bytes": 0, "workspace_bytes": 0, "safety_bytes": 0,
              "floor_bytes": 0}
    values.update(overrides)
    return mg.DeviceFootprint(**values)


def test_unified_capacity_uses_the_usable_ceiling_without_double_counting(mg):
    fp = _footprint(mg, reserve_bytes = int(4.5 * GIB), weights_bytes = 100 * GIB,
                    workspace_bytes = int(4 * GIB))
    capacity = fp.capacity_bytes(113.3)
    assert capacity == int(113.3 * GIB)
    assert fp.need_bytes == 104 * GIB
    assert fp.need_bytes < capacity

    # The same ceiling still caps a box that reports more raw memory.
    fp2 = _footprint(mg, total_bytes = 124 * GIB, free_bytes = 124 * GIB,
                     reserve_bytes = int(4.5 * GIB))
    assert fp2.capacity_bytes(113.3) == int(113.3 * GIB)

    # Current free memory is a second cap; the reserve is subtracted from free only.
    fp3 = _footprint(mg, free_bytes = 100 * GIB, reserve_bytes = int(4.5 * GIB))
    assert fp3.capacity_bytes(113.3) == int(95.5 * GIB)


def test_dedicated_capacity_subtracts_reserve_and_floor(mg):
    fp = _footprint(mg, arch = "gfx1201", unified = False, total_bytes = int(15.92 * GIB),
                    free_bytes = int(15.92 * GIB), reserve_bytes = int(0.8 * GIB),
                    floor_bytes = int(0.5 * GIB))
    assert fp.capacity_bytes(113.3) == int(15.92 * GIB) - int(0.8 * GIB) - int(0.5 * GIB)


def _guard(mg, mode, reserve = "4.5", host_floor = 6.0):
    budget = mg.DeviceBudget(mode = mode, reserve = reserve, host_floor_gib = host_floor)
    resolved = {"reserve": {0: 4.5}, "workspace": {0: 4.0},
                "safety": {0: 0.0}, "device_floor": {0: 0.0}}
    return mg.OOMGuard(budget, resolved, logger = lambda *_a: None)


def test_auto_refuses_and_warn_only_logs(mg):
    fp = _footprint(mg, weights_bytes = 112 * GIB, workspace_bytes = int(4 * GIB),
                    reserve_bytes = int(4.5 * GIB))
    with pytest.raises(mg.OOMGuardError, match = "refused the load"):
        _guard(mg, "auto").check([fp], host_available = 120 * GIB,
                                 context = 786432, chosen_n = 5)
    report = _guard(mg, "warn").check([fp], host_available = 120 * GIB,
                                      context = 786432, chosen_n = 5)
    assert not report.ok
    off_report = _guard(mg, "off").check([fp], host_available = 1)
    assert not off_report.ok  # off reports the estimate but never raises


def test_host_pool_is_checked_even_when_the_device_fits(mg):
    fp = _footprint(mg, weights_bytes = 100 * GIB, workspace_bytes = int(4 * GIB))
    # Device capacity 113.3 fits, but only 104 GiB is available on the host.
    with pytest.raises(mg.OOMGuardError, match = "host MemAvailable"):
        _guard(mg, "auto").check([fp], host_available = 104 * GIB)


def test_refusal_message_contains_table_and_largest_fit(mg):
    fp = _footprint(mg, weights_bytes = 112 * GIB, workspace_bytes = int(4 * GIB))
    guard = _guard(mg, "auto")
    with pytest.raises(mg.OOMGuardError) as exc:
        guard.check([fp], host_available = 120 * GIB, context = 786432,
                    chosen_n = 5, largest_n = 3, largest_ctx = 524288)
    text = str(exc.value)
    assert "device" in text and "capacity" in text
    assert "largest part0 N that fits" in text and "3" in text
    assert "largest context that fits" in text and "524288" in text


def test_single_device_default_mode_is_warn(mg, monkeypatch):
    monkeypatch.delenv("KYOJIN_OOM_GUARD", raising = False)
    monkeypatch.setattr(mg, "device_total_bytes", lambda _i: 118 * GIB)
    monkeypatch.setattr(mg, "device_free_bytes", lambda _i: 118 * GIB)
    monkeypatch.setattr(mg, "host_available_bytes", lambda: 120 * GIB)
    report = mg.check_single_device(116 * GIB, 0, 0, logger = lambda *_a: None)
    assert report is not None and not report.ok


def test_allocator_off_sets_the_existing_opt_out_without_clobbering(mg, split_mod, monkeypatch):
    for key in ("EXL3_EXPANDABLE_SEGMENTS", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    settings = split_mod.SplitSettings(part0_layers = 1, expandable_segments = "off")
    assert split_mod.apply_allocator_env(settings, logger = lambda *_a: None) == "off"
    assert os.environ["EXL3_EXPANDABLE_SEGMENTS"] == "0"
    conf = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    assert "max_split_size_mb:128" in conf and conf.endswith("expandable_segments:False")
    assert conf.count("expandable_segments:") == 1


def test_allocator_on_overrides_a_false_allocator_conf(mg, split_mod, monkeypatch):
    monkeypatch.delenv("EXL3_EXPANDABLE_SEGMENTS", raising = False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False,garbage_collection_threshold:0.8")
    settings = split_mod.SplitSettings(part0_layers = 1, expandable_segments = "on")
    assert split_mod.apply_allocator_env(settings, logger = lambda *_a: None) == "on"
    conf = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    assert conf.count("expandable_segments:") == 1
    assert "expandable_segments:True" in conf
    assert "garbage_collection_threshold:0.8" in conf


def test_allocator_auto_stays_on_below_the_large_context_gate(split_mod, monkeypatch):
    monkeypatch.delenv("EXL3_EXPANDABLE_SEGMENTS", raising = False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    settings = split_mod.SplitSettings(part0_layers = 1, expandable_segments = "auto")
    assert split_mod.apply_allocator_env(settings, model_dir = "/nonexistent", ctx = 131072,
                                         logger = lambda *_a: None) == "on"
    assert "expandable_segments:" not in os.environ["PYTORCH_CUDA_ALLOC_CONF"]


def test_allocator_auto_off_for_large_unified_load(split_mod, monkeypatch):
    monkeypatch.delenv("EXL3_EXPANDABLE_SEGMENTS", raising = False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising = False)
    monkeypatch.setattr(split_mod, "_has_gtt_device", lambda: True)
    monkeypatch.setattr(split_mod, "_model_dir_bytes", lambda _p: 119 * GIB)
    monkeypatch.setattr(split_mod, "_meminfo_bytes", lambda _k: 124 * GIB)
    settings = split_mod.SplitSettings(part0_layers = 1, expandable_segments = "auto")
    assert split_mod.apply_allocator_env(settings, model_dir = "/model", ctx = 512 * 1024,
                                         logger = lambda *_a: None) == "off"
    assert os.environ["EXL3_EXPANDABLE_SEGMENTS"] == "0"
    assert "expandable_segments:False" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]


def test_guard_device_floor_parsing(guard_mod):
    floors = guard_mod.parse_device_floors(["card1=4.5,nvidia:0=2"], global_gib = 1.0)
    assert floors["card1"] == 4.5 and floors["nvidia:0"] == 2.0
    assert guard_mod.floor_for("card2", floors) == 1.0
    assert guard_mod.floor_for("nvidia:1", floors) == 1.0
    floors["amd"] = 3.0
    assert guard_mod.floor_for("card9", floors) == 3.0


def test_unified_detection_does_not_flag_a_large_dedicated_gpu(mg, monkeypatch):
    monkeypatch.setattr(mg, "device_total_bytes", lambda _i: 80 * GIB)
    monkeypatch.setattr(mg, "host_total_bytes", lambda: 128 * GIB)
    monkeypatch.setattr(mg, "_torch", lambda: SimpleNamespace(cuda = SimpleNamespace(
        get_device_properties = lambda _i: (_ for _ in ()).throw(RuntimeError("no gpu")))))
    monkeypatch.setattr(mg, "glob", SimpleNamespace(glob = lambda _p: []), raising = False)
    assert mg.is_unified_device(0, "gfx1201", 80 * GIB) is False
    assert mg.is_unified_device(0, "gfx1151", 80 * GIB) is True


def test_effective_floor_ignores_global_floor_on_multiple_devices(guard_mod):
    floors = guard_mod.parse_device_floors([], global_gib = 4.0)
    assert guard_mod.effective_floor("card0", floors, multiple = False) == 4.0
    assert guard_mod.effective_floor("card0", floors, multiple = True) == 0.0
    floors["card1"] = 2.5
    assert guard_mod.effective_floor("card1", floors, multiple = True) == 2.5
