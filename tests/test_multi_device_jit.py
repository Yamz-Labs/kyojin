"""JIT HIP loaders must key their module/function handles by the calling device.

The engine can run the same kernel on two GPUs from one process. A loader that asks for
device 0's arch or keeps one handle per defines string runs the wrong code object on the
other card. The CDC tests below fake the compiler and the HIP runtime, so the state-key
and arch-selection invariant is checked without a GPU; the hardware tests load every
loader on both devices. gdn_hf, gdn_fused, kda_fused and kda_hf are gfx11.5-only: on
gfx1201 their supported() gate must refuse before any hipcc attempt.
"""
import ctypes
import importlib
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
_PKGS = ["exllamav3", "exllamav3.util", "exllamav3.vendor", "exllamav3.vendor.fla", "exllamav3.vendor.fla.hip",
         "exllamav3.modules", "exllamav3.modules.ple_fn", "exllamav3.modules.gr_fn",
         "exllamav3.modules.dsv41_fn", "exllamav3.modules.attention_fn"]
_added = []
for _n in _PKGS:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        _m.__path__ = [str(ROOT.joinpath(*_n.split(".")))]
        sys.modules[_n] = _m
        _added.append(_n)
try:
    hip_compiler = importlib.import_module("exllamav3.util.hip_compiler")
    arch_list = importlib.import_module("exllamav3.util.arch_list")
    gr_mix = importlib.import_module("exllamav3.modules.gr_fn.gr_mix_hip")
    ple = importlib.import_module("exllamav3.modules.ple_fn.ple_hip")
    qsa = importlib.import_module("exllamav3.modules.attention_fn.qsa_prefill_hip")
    ds41_pf = importlib.import_module("exllamav3.modules.dsv41_fn.ds41_pf")
    dsv41_hip = importlib.import_module("exllamav3.modules.dsv41_fn.dsv41_hip")
    dsv41_idx = importlib.import_module("exllamav3.modules.dsv41_fn.dsv41_idx")
    dsv41_fuse = importlib.import_module("exllamav3.modules.dsv41_fn.dsv41_fuse")
    dsv41_moe = importlib.import_module("exllamav3.modules.dsv41_fn.dsv41_moe")
    gdn_hf = importlib.import_module("exllamav3.vendor.fla.hip.gdn_hf_hip")
    gdn_fused = importlib.import_module("exllamav3.vendor.fla.hip.gdn_fused_h_hip")
    kda_fused = importlib.import_module("exllamav3.vendor.fla.hip.kda_fused_h_hip")
    try:
        kda_hf = importlib.import_module("exllamav3.vendor.fla.hip.kda_hf_hip")
    except ImportError:
        kda_hf = None
finally:
    for _n in _added:
        sys.modules.pop(_n, None)

GFX115 = arch_list.GFX115_ARCHS

# name -> (module, load(device), arch gate)
CASES = {
    "gr_mix_hip": (gr_mix, lambda m, d: m._load("", d), GFX115),
    "ple_hip": (ple, lambda m, d: m._load(d), None),
    "qsa_prefill_hip": (qsa, lambda m, d: m._load("", d), GFX115),
    "ds41_pf": (ds41_pf, lambda m, d: m._fn("ds41_gather_sum", d), GFX115),
    "dsv41_hip": (dsv41_hip, lambda m, d: m._load(d), None),
    "dsv41_idx": (dsv41_idx, lambda m, d: m._load(d), GFX115),
    "dsv41_fuse": (dsv41_fuse, lambda m, d: m._load(d), None),
    "dsv41_moe": (dsv41_moe, lambda m, d: m._load(d), None),
    "gdn_hf": (gdn_hf, lambda m, d: m._load(d), GFX115),
    "gdn_fused_h_hip": (gdn_fused, lambda m, d: m._load("", d), GFX115),
    "kda_fused_h_hip": (kda_fused, lambda m, d: m._load("", d), GFX115),
    "kda_hf_hip": (kda_hf, lambda m, d: m._load("", d), GFX115),
}
SUPPORTED = {"gdn_hf", "gdn_fused_h_hip", "kda_fused_h_hip", "kda_hf_hip"}


class FakeCuda:
    def __init__(self, archs):
        self.archs = list(archs)
        self.current = 0
        self.set_calls = []

    def current_device(self):
        return self.current

    def set_device(self, index):
        self.set_calls.append(int(index))
        self.current = int(index)

    def get_device_properties(self, index):
        return SimpleNamespace(gcnArchName = f"{self.archs[int(index)]}:sramecc+:xnack-")

    def init(self):
        pass

    def synchronize(self, *_a, **_k):
        pass


class FakeLaunch:
    argtypes = None

    def __call__(self, *_a, **_k):
        return 0


class FakeLib:
    def __init__(self):
        self.hipModuleLaunchKernel = FakeLaunch()
        self.loaded = []

    def hipModuleLoad(self, ref, path):
        self.loaded.append(path)
        ctypes.cast(ref, ctypes.POINTER(ctypes.c_void_p))[0] = 0x1000 + len(self.loaded)
        return 0

    def hipModuleGetFunction(self, ref, _mod, name):
        ctypes.cast(ref, ctypes.POINTER(ctypes.c_void_p))[0] = 0x2000 + sum(name) % 0x1000
        return 0

    def hipModuleGetGlobal(self, _ptr, _size, _mod, _name):
        return 1


@pytest.fixture
def fake_jit(monkeypatch, tmp_path):
    for k in ("EXL3_HIPCC", "EXL3_ROCM_SDK"):
        monkeypatch.delenv(k, raising = False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(hip_compiler, "_info",
                        {"path": "/bin/true", "version": "hipcc fake", "key": "deadbeef",
                         "sdk": None, "sdk_ok": True, "source": "test"})
    monkeypatch.setattr(hip_compiler, "gcc_flags", list)
    compiles = []

    def fake_run(cmd, **_kwargs):
        compiles.append(list(cmd))
        if "-o" in cmd:
            path = cmd[cmd.index("-o") + 1]
            os.makedirs(os.path.dirname(path), exist_ok = True)
            with open(path, "wb") as f:
                f.write(b"built")
        return SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(qsa, "require_proof", lambda *_a, **_k: None)
    monkeypatch.setattr(kda_fused, "_precompile_siblings", lambda *_a, **_k: None)
    cuda = FakeCuda(["gfx1151", "gfx1201"])
    monkeypatch.setattr(torch.cuda, "current_device", cuda.current_device)
    monkeypatch.setattr(torch.cuda, "set_device", cuda.set_device)
    monkeypatch.setattr(torch.cuda, "get_device_properties", cuda.get_device_properties)
    monkeypatch.setattr(torch.cuda, "init", cuda.init)
    monkeypatch.setattr(torch.cuda, "synchronize", cuda.synchronize)
    arch_list.reset_support_cache()
    libs = {}
    for module, _call, _gate in CASES.values():
        if module is not None:
            module._state.clear()
            lib = FakeLib()
            libs[module] = lib
            monkeypatch.setattr(module, "load_hip_runtime", lambda lib = lib: lib)
    yield compiles, cuda
    for module in libs:
        module._state.clear()
    arch_list.reset_support_cache()


def _call(case, module, device):
    return case[1](module, device)


def _handles(fn):
    if isinstance(fn, dict):
        return list(fn.values())
    return [fn]


@pytest.mark.parametrize("name", list(CASES))
def test_state_keys_are_per_device(fake_jit, name):
    module, _call_fn, _gate = CASES[name]
    if module is None:
        pytest.skip("module is not importable without Triton")
    compiles, cuda = fake_jit
    compiles.clear()

    fn0 = _call(CASES[name], module, 0)
    fn1 = _call(CASES[name], module, 1)

    assert (0, "") in module._state, name
    assert (1, "") in module._state, name
    assert module._state[(0, "")] is not module._state[(1, "")], name
    assert ("mod", (0, "")) in module._state and ("mod", (1, "")) in module._state, name
    assert all(h.value for h in _handles(fn0)), name
    assert all(h.value for h in _handles(fn1)), name
    assert module._state["lib"] is not None, name

    got = {x.split("=", 1)[1] for c in compiles for x in c if x.startswith("--offload-arch=")}
    assert {"gfx1151", "gfx1201"} <= got, (name, got)
    assert cuda.set_calls == [1], name


def test_dsv41_emu_hook_survives_the_state_rewrite(fake_jit):
    sentinel = object()
    dsv41_hip._state["emu"] = sentinel
    dsv41_hip._load(0)
    assert dsv41_hip._state["emu"] is sentinel


@pytest.mark.parametrize("name", sorted(SUPPORTED))
def test_gfx12_gate_refuses_before_compiling(fake_jit, name):
    module = CASES[name][0]
    if module is None:
        pytest.skip("module is not importable without Triton")
    assert module.supported(1) is False
    assert module.supported(0) is True
    # No compile for the refused device: the state stays empty
    assert not any(isinstance(k, tuple) and k[0] == 1 for k in module._state)


@pytest.mark.skipif(not (torch.version.hip and torch.cuda.device_count() >= 2),
                    reason = "requires two ROCm GPUs")
def test_each_loader_loads_on_both_devices():
    prev = torch.cuda.current_device()
    try:
        archs = [arch_list.device_arch(i) for i in range(torch.cuda.device_count())]
        for name, (module, call, gate) in CASES.items():
            if module is None:
                continue
            module._state.clear()
            arch_list.reset_support_cache()
            for dev in range(2):
                arch = archs[dev]
                if gate is not None and arch not in gate:
                    if name in SUPPORTED:
                        assert module.supported(dev) is False
                    continue
                torch.cuda.set_device(dev)
                fn = call(module, dev)
                assert all(h.value for h in _handles(fn)), (name, dev)
                assert (dev, "") in module._state, (name, dev)
            loaded = [d for d in range(2) if (d, "") in module._state]
            if len(loaded) == 2:
                assert module._state[(0, "")] is not module._state[(1, "")], name
    finally:
        torch.cuda.set_device(prev)
