"""CPU tests of the gdn_hf_hip device gate and the hipcc error tail.
No GPU, no ROCm, no built extension (the package __init__ files are bypassed with empty namespace
stubs while the module is imported, then removed)."""
import importlib, sys, types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
_PKGS = ["exllamav3", "exllamav3.util", "exllamav3.vendor", "exllamav3.vendor.fla", "exllamav3.vendor.fla.hip"]
_added = []
for _n in _PKGS:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); _m.__path__ = [str(ROOT.joinpath(*_n.split(".")))]
        sys.modules[_n] = _m; _added.append(_n)
try:
    gdn_hf = importlib.import_module("exllamav3.vendor.fla.hip.gdn_hf_hip")
finally:
    for _n in _added:
        sys.modules.pop(_n, None)


@pytest.fixture(autouse=True)
def _clean_state():
    gdn_hf._supported.clear()
    yield
    gdn_hf._supported.clear()


def _dev(arch):
    return SimpleNamespace(gcnArchName = f"{arch}:sramecc+:xnack-")


@pytest.mark.parametrize("arch", ["gfx1150", "gfx1151", "gfx1152"])
def test_supported_accepts_the_gfx115_family(monkeypatch, arch):
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: _dev(arch))
    assert gdn_hf.supported(0) is True


@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201", "gfx1100", "sm_90", "", "unknown"])
def test_supported_rejects_other_archs(monkeypatch, arch):
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: _dev(arch))
    assert gdn_hf.supported(0) is False


def test_supported_requires_hip(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _index: (_ for _ in ()).throw(AssertionError("probed without HIP")))
    assert gdn_hf.supported(0) is False


def test_supported_probes_once_per_device(monkeypatch):
    probes = []
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: (probes.append(index), _dev("gfx1151"))[1])
    assert gdn_hf.supported(0) is True
    assert gdn_hf.supported(torch.device("cuda", 0)) is True
    assert gdn_hf.supported("0") is True
    assert probes == [0]


def test_supported_probe_failure_is_false(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _index: (_ for _ in ()).throw(RuntimeError("no device")))
    assert gdn_hf.supported(0) is False


def test_compile_tail_short_output_is_unchanged():
    lines = [f"line {i}" for i in range(5)]
    assert gdn_hf._compile_tail(lines) == lines


def test_compile_tail_keeps_an_error_before_the_tail():
    lines = ["warning: noise"] * 30 + ["fatal error: 'cstdlib' file not found"] + ["note: context"] * 20
    tail = gdn_hf._compile_tail(lines)
    assert "fatal error: 'cstdlib' file not found" in tail
    assert tail[-1] == lines[-1]
    assert "..." in tail
