"""CPU tests for the pre-flight memory guard and the host-only supervisor.

The guard must be testable without a GPU: device caps, host floor, strict refusal and
off mode are pure Python.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "exllamav3") not in sys.path:
    sys.path.insert(0, str(ROOT / "exllamav3"))
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


def _footprint(mg, **overrides):
    values = {"index": 0, "arch": "gfx1151", "unified": True,
              "total_bytes": 118 * GIB, "free_bytes": 118 * GIB,
              "weights_bytes": 0, "cache_bytes": 0, "slab_bytes": 0,
              "workspace_bytes": 0}
    values.update(overrides)
    return mg.DeviceFootprint(**values)


def _guard(mg, mode="auto", host_floor=7.0, margin=1.0, workspace=2.0):
    policy = mg.GuardPolicy(mode=mode, host_floor_gib=host_floor,
                            preflight_margin_gib=margin, workspace_gib=workspace)
    return mg.OOMGuard(policy, logger=lambda *_a: None)


def test_policy_env_defaults_and_overrides(mg, monkeypatch):
    for key in ("KYOJIN_OOM_GUARD", "MEMGUARD_FLOOR",
                "KYOJIN_OOM_GUARD_PREFLIGHT_MARGIN_GIB", "KYOJIN_OOM_GUARD_WORKSPACE_GIB"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("MEMGUARD_FLOOR", raising=False)
    policy = mg.GuardPolicy.from_env()
    assert policy.host_floor_gib == 8.0
    monkeypatch.setenv("MEMGUARD_FLOOR", "7")
    policy = mg.GuardPolicy.from_env()
    assert policy.mode == "auto"
    assert policy.host_floor_gib == 7.0
    assert policy.preflight_margin_gib == 1.0
    assert policy.workspace_gib == 2.0

    monkeypatch.setenv("KYOJIN_OOM_GUARD", "OFF")
    monkeypatch.setenv("MEMGUARD_FLOOR", "9.5")
    monkeypatch.setenv("KYOJIN_OOM_GUARD_PREFLIGHT_MARGIN_GIB", "0.5")
    monkeypatch.setenv("KYOJIN_OOM_GUARD_WORKSPACE_GIB", "3")
    policy = mg.GuardPolicy.from_env()
    assert policy.mode == "off"
    assert policy.host_floor_gib == 9.5
    assert policy.preflight_host_floor_gib == 10.0
    assert policy.workspace_gib == 3.0

    # An explicit flag wins over the environment.
    monkeypatch.setenv("KYOJIN_OOM_GUARD", "auto")
    policy = mg.GuardPolicy.from_env(explicit_mode="off")
    assert policy.mode == "off"


def test_device_capacity_is_free_or_total_not_a_reserve(mg):
    fp = _footprint(mg, total_bytes=118 * GIB, free_bytes=110 * GIB,
                    weights_bytes=100 * GIB, cache_bytes=8 * GIB,
                    slab_bytes=GIB, workspace_bytes=2 * GIB)
    assert fp.capacity_bytes() == 110 * GIB
    assert fp.need_bytes == 111 * GIB

    # Another process using the card lowers free memory; the cap follows it.
    fp = _footprint(mg, total_bytes=16 * GIB, free_bytes=12 * GIB)
    assert fp.capacity_bytes() == 12 * GIB

    # No probe data means no opinion, not a hard-coded size.
    fp = _footprint(mg, total_bytes=None, free_bytes=None)
    assert fp.capacity_bytes() is None


def test_preflight_refuses_strictly_when_need_exceeds_host_capacity(mg):
    # Device fits (need 108 < 118), host planning capacity is 120 - (7 + 1) = 112.
    fp = _footprint(mg, weights_bytes=106 * GIB, workspace_bytes=2 * GIB)
    guard = _guard(mg)
    assert guard.check([fp], host_available=120 * GIB).ok

    # 0.1 GiB over is refused; there is no tolerance band.
    fp = _footprint(mg, weights_bytes=110.1 * GIB, workspace_bytes=2 * GIB)
    with pytest.raises(mg.OOMGuardError, match="refused the load"):
        guard.check([fp], host_available=120 * GIB)


def test_device_cap_refuses_even_when_the_host_is_fine(mg):
    fp = _footprint(mg, total_bytes=16 * GIB, free_bytes=15 * GIB,
                    weights_bytes=14.5 * GIB, workspace_bytes=2 * GIB)
    with pytest.raises(mg.OOMGuardError, match="deficit"):
        _guard(mg).check([fp], host_available=120 * GIB)


def test_mode_off_returns_a_report_without_raising(mg):
    fp = _footprint(mg, total_bytes=16 * GIB, free_bytes=15 * GIB,
                    weights_bytes=20 * GIB)
    report = _guard(mg, mode="off").check([fp], host_available=1)
    assert not report.ok


def test_refusal_message_contains_table_and_largest_fit(mg):
    fp = _footprint(mg, weights_bytes=116 * GIB, workspace_bytes=2 * GIB)
    with pytest.raises(mg.OOMGuardError) as exc:
        _guard(mg).check([fp], host_available=120 * GIB, context=786432,
                         chosen_n=5, largest_n=3, largest_ctx=524288)
    text = str(exc.value)
    assert "device" in text and "capacity" in text
    assert "largest part0 N that fits" in text and "3" in text
    assert "largest context that fits" in text and "524288" in text


def test_single_device_uses_the_automatic_policy(mg, monkeypatch):
    monkeypatch.delenv("KYOJIN_OOM_GUARD", raising=False)
    monkeypatch.setattr(mg, "device_total_bytes", lambda _i: 118 * GIB)
    monkeypatch.setattr(mg, "device_free_bytes", lambda _i: 118 * GIB)
    monkeypatch.setattr(mg, "host_available_bytes", lambda: 120 * GIB)
    monkeypatch.setattr(mg, "is_unified_device", lambda _i: True)
    # Device fits, host planning capacity (112) does not.
    with pytest.raises(mg.OOMGuardError, match="refused the load"):
        mg.check_single_device(116 * GIB, 0, 0, logger=lambda *_a: None)


def test_unified_detection_uses_only_the_integrated_property(mg, monkeypatch):
    # fake props -> expected classification.  A runtime without the property is
    # dedicated, including a large AMD dGPU whose total is close to host RAM.
    cases = [
        (SimpleNamespace(name = "AMD Radeon 8060S", total_memory = 114 * GIB,
                         is_integrated = 1), True),
        (SimpleNamespace(name = "AMD Radeon Graphics", total_memory = 112 * GIB,
                         integrated = True), True),
        (SimpleNamespace(name = "NVIDIA GeForce RTX 3080", total_memory = 10 * GIB), False),
        (SimpleNamespace(name = "NVIDIA GeForce RTX 3080 Ti Laptop GPU",
                         total_memory = 16 * GIB), False),
        (SimpleNamespace(name = "NVIDIA GeForce RTX 3090", total_memory = 24 * GIB), False),
        (SimpleNamespace(name = "AMD Radeon RX 9070 XT", total_memory = 118 * GIB), False),
    ]
    for props, expected in cases:
        monkeypatch.setattr(mg, "_torch", lambda props = props: SimpleNamespace(cuda = SimpleNamespace(
            get_device_properties = lambda _i, props = props: props)))
        assert mg.is_unified_device(0) is expected

    # A failed probe means dedicated, not a guess.
    monkeypatch.setattr(mg, "_torch", lambda: SimpleNamespace(cuda = SimpleNamespace(
        get_device_properties = lambda _i: (_ for _ in ()).throw(RuntimeError("no gpu")))))
    assert mg.is_unified_device(0) is False


def test_dedicated_footprint_has_no_workspace_and_only_the_hard_fit_cap(mg, monkeypatch):
    monkeypatch.setattr(mg, "is_unified_device", lambda _i: False)
    guard = _guard(mg)

    fp = guard.footprint(0, total_bytes = 16 * GIB, free_bytes = 16 * GIB,
                         weights_bytes = int(15.5 * GIB))
    assert fp.unified is False and fp.workspace_bytes == 0
    # Fits by 0.5 GiB: the old unconditional 2 GiB workspace would have refused it.
    assert fp.need_bytes <= fp.capacity_bytes() < fp.need_bytes + 2 * GIB
    report = guard.check([fp], host_available = 1)
    assert report.ok and report.host_result is None

    # Hard overcommit is still refused under auto.
    over = guard.footprint(0, total_bytes = 16 * GIB, free_bytes = 16 * GIB,
                           weights_bytes = 17 * GIB)
    with pytest.raises(mg.OOMGuardError, match = "deficit"):
        guard.check([over], host_available = 1)

    # A model resident on the card lowers free; the cap follows it.
    resident = guard.footprint(0, total_bytes = 16 * GIB, free_bytes = 14 * GIB,
                               weights_bytes = 15 * GIB)
    with pytest.raises(mg.OOMGuardError, match = "deficit"):
        guard.check([resident], host_available = 1)


def test_unified_footprint_adds_the_workspace_and_plans_against_host(mg, monkeypatch):
    monkeypatch.setattr(mg, "is_unified_device", lambda _i: True)
    guard = _guard(mg)

    fp = guard.footprint(0, total_bytes = 118 * GIB, free_bytes = 118 * GIB,
                         weights_bytes = 106 * GIB)
    assert fp.unified is True and fp.workspace_bytes == 2 * GIB
    # Device fits (108/118); host capacity is 120 - (7 + 1) = 112.
    report = guard.check([fp], host_available = 120 * GIB)
    assert report.ok and report.host_result is not None

    # A resident host process lowers MemAvailable below the plan.
    with pytest.raises(mg.OOMGuardError, match = "deficit"):
        guard.check([fp], host_available = 109 * GIB)

    # The workspace estimate is env-tunable for unified devices.
    for key in ("KYOJIN_OOM_GUARD", "MEMGUARD_FLOOR",
                "KYOJIN_OOM_GUARD_PREFLIGHT_MARGIN_GIB", "KYOJIN_OOM_GUARD_WORKSPACE_GIB"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setenv("KYOJIN_OOM_GUARD_WORKSPACE_GIB", "5")
    fp = mg.OOMGuard(mg.GuardPolicy.from_env(), logger = lambda *_a: None).footprint(
        0, total_bytes = 118 * GIB, free_bytes = 118 * GIB, weights_bytes = 100 * GIB)
    assert fp.workspace_bytes == 5 * GIB


def test_unified_capacity_uses_reclaimable_host_memory(mg, monkeypatch):
    monkeypatch.setattr(mg, "is_unified_device", lambda _i: True)
    guard = _guard(mg)

    # A runtime that reports MemFree makes page cache look like used device memory.
    fp = guard.footprint(0, total_bytes = 118 * GIB, free_bytes = 10 * GIB,
                         weights_bytes = 108 * GIB)
    report = guard.check([fp], host_available = 120 * GIB)
    assert report.ok
    # MemAvailable is the alternative bound, clamped by the visible aperture.
    assert report.device_results[0].capacity_bytes == 118 * GIB

    # A genuine host deficit is still refused.
    with pytest.raises(mg.OOMGuardError, match = "deficit"):
        guard.check([fp], host_available = 110 * GIB)


def test_guard_mode_accepts_boolean_like_values(mg, monkeypatch):
    for value in ("0", "false", "no", "off", "disabled", " FALSE ", 0, False):
        assert mg.GuardPolicy(mode = value).mode == "off"
    for value in ("1", "true", "yes", "on", "auto", 1, True):
        assert mg.GuardPolicy(mode = value).mode == "auto"
    with pytest.raises(ValueError, match = "oom guard mode"):
        mg.GuardPolicy(mode = "maybe")

    monkeypatch.setenv("KYOJIN_OOM_GUARD", "0")
    assert mg.GuardPolicy.from_env().mode == "off"


def test_reusable_device_bytes_adds_cached_allocator_blocks(mg, monkeypatch):
    monkeypatch.setattr(mg, "_torch", lambda: SimpleNamespace(cuda = SimpleNamespace(
        mem_get_info = lambda _i: (5 * GIB, 16 * GIB),
        memory_reserved = lambda _i: 3 * GIB,
        memory_allocated = lambda _i: GIB)))
    assert mg.device_reusable_bytes(0) == 7 * GIB

    # Missing counters fall back to raw free; a failed probe means no opinion.
    def boom(_i):
        raise RuntimeError("no counters")

    monkeypatch.setattr(mg, "_torch", lambda: SimpleNamespace(cuda = SimpleNamespace(
        mem_get_info = lambda _i: (5 * GIB, 16 * GIB),
        memory_reserved = boom,
        memory_allocated = boom)))
    assert mg.device_reusable_bytes(0) == 5 * GIB

    monkeypatch.setattr(mg, "_torch", lambda: SimpleNamespace(cuda = SimpleNamespace(
        mem_get_info = boom)))
    assert mg.device_reusable_bytes(0) is None


def test_supervisor_host_floor_is_env_tunable(guard_mod, monkeypatch, capsys):
    monkeypatch.delenv("MEMGUARD_FLOOR", raising=False)
    assert guard_mod.host_floor_from_env() == 8.0
    monkeypatch.setenv("MEMGUARD_FLOOR", "9.5")
    assert guard_mod.host_floor_from_env() == 9.5

    # A safety setting must not silently fall back after a typo, and non-finite
    # values must not disable the floor (nan) or crash the lane (inf).
    for value in ("eight", "nan", "inf", "-inf", "1e999"):
        monkeypatch.setenv("MEMGUARD_FLOOR", value)
        assert guard_mod.host_floor_from_env() == 8.0
        assert "MEMGUARD_FLOOR" in capsys.readouterr().err


def test_supervisor_log_rotates_at_the_size_cap(guard_mod, tmp_path):
    path = tmp_path / "memguard.log"
    log = guard_mod.Logger(str(path), max_bytes = 200)
    for i in range(20):
        log(f"line {i} " + "x" * 20)
    assert (tmp_path / "memguard.log.1").exists()
    assert path.exists() and path.stat().st_size <= 200


def test_supervisor_log_rotation_failure_still_caps_the_file(guard_mod, tmp_path):
    path = tmp_path / "memguard.log"
    (tmp_path / "memguard.log.1").mkdir()               # makes os.replace fail
    log = guard_mod.Logger(str(path), max_bytes = 200)
    for i in range(20):
        log(f"line {i} " + "x" * 20)
    assert path.exists() and path.stat().st_size <= 200
