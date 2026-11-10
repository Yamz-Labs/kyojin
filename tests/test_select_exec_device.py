"""select_exec_device: select the module's device at a transition so Triton and unguarded
extension launches see the device the tensors live on. The cache is thread-local and the
helper compares against the real current device, so another library changing it is noticed."""
import importlib
import sys
import threading
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
_PKGS = ["exllamav3", "exllamav3.modules", "exllamav3.util", "exllamav3.model"]
_added = []
for _n in _PKGS:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        _m.__path__ = [str(ROOT.joinpath(*_n.split(".")))]
        sys.modules[_n] = _m
        _added.append(_n)
try:
    module_mod = importlib.import_module("exllamav3.modules.module")
finally:
    for _n in _added:
        sys.modules.pop(_n, None)


class FakeCuda:
    """Per-thread current device, like CUDA's, plus a log of set_device calls."""

    def __init__(self):
        self._current = threading.local()
        self.set_calls = []
        self._lock = threading.Lock()

    def current_device(self):
        return getattr(self._current, "index", 0)

    def set_device(self, index):
        with self._lock:
            self.set_calls.append(int(index))
        self._current.index = int(index)


@pytest.fixture
def fake_cuda(monkeypatch):
    fake = FakeCuda()
    monkeypatch.setattr(module_mod.torch.cuda, "current_device", fake.current_device)
    monkeypatch.setattr(module_mod.torch.cuda, "set_device", fake.set_device)
    module_mod._exec_device.index = None
    yield fake


def test_sets_the_current_device_and_the_cache(fake_cuda):
    module_mod.select_exec_device(torch.device("cuda:1"))
    assert fake_cuda.current_device() == 1
    assert fake_cuda.set_calls == [1]
    assert module_mod._exec_device.index == 1


def test_accepts_a_device_string(fake_cuda):
    module_mod.select_exec_device("cuda:1")
    assert fake_cuda.current_device() == 1
    assert module_mod._exec_device.index == 1


def test_ignores_none_cpu_and_indexless_devices(fake_cuda):
    for device in (None, "cpu", torch.device("cpu"), torch.device("cuda")):
        module_mod.select_exec_device(device)
    assert fake_cuda.set_calls == []
    assert module_mod._exec_device.index is None


def test_notices_an_external_change_behind_the_cache(fake_cuda):
    module_mod._exec_device.index = 1
    fake_cuda._current.index = 0
    module_mod.select_exec_device(torch.device("cuda:1"))
    assert fake_cuda.set_calls == [1]
    assert fake_cuda.current_device() == 1


def test_does_not_repeat_set_device_when_already_selected(fake_cuda):
    module_mod.select_exec_device(torch.device("cuda:1"))
    module_mod.select_exec_device(torch.device("cuda:1"))
    assert fake_cuda.set_calls == [1]


def test_another_thread_keeps_its_own_device(fake_cuda):
    module_mod.select_exec_device(torch.device("cuda:1"))
    seen = []

    def worker():
        seen.append(fake_cuda.current_device())
        module_mod.select_exec_device(torch.device("cuda:0"))
        seen.append(fake_cuda.current_device())

    thread = threading.Thread(target = worker)
    thread.start()
    thread.join()

    assert seen == [0, 0]
    assert fake_cuda.current_device() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a CUDA/ROCm device")
def test_real_current_device_is_set():
    prev = torch.cuda.current_device()
    try:
        target = 1 if torch.cuda.device_count() > 1 else 0
        module_mod._exec_device.index = None
        module_mod.select_exec_device(torch.device("cuda", target))
        assert torch.cuda.current_device() == target
    finally:
        torch.cuda.set_device(prev)
