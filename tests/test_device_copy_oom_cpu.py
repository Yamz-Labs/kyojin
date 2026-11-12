"""CPU contract tests for util/device_copy.to_device's out-of-memory handling.

A device OOM must be re-raised (and cache no bounce verdict), not mistaken for a
broken P2P fabric: bouncing through host memory would not help an exhausted device,
so a cached verdict would be wrong and would stick for the process lifetime.
Runs without a GPU or the built extension.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope = "module")
def dc():
    added = []
    if "exllamav3" not in sys.modules:
        pkg = types.ModuleType("exllamav3")
        pkg.__path__ = [str(ROOT / "exllamav3")]
        sys.modules["exllamav3"] = pkg
        added.append("exllamav3")
    if "exllamav3.util" not in sys.modules:
        util = types.ModuleType("exllamav3.util")
        util.__path__ = [str(ROOT / "exllamav3" / "util")]
        sys.modules["exllamav3.util"] = util
        added.append("exllamav3.util")
    from exllamav3.util import device_copy as dc

    yield dc

    sys.modules.pop("exllamav3.util.device_copy", None)
    for name in added:
        sys.modules.pop(name, None)


@pytest.fixture
def clean(dc):
    old_forced = dc.FORCED
    dc.reset_verdicts()
    for k in dc.stats:
        dc.stats[k] = 0
    yield
    dc.FORCED = old_forced
    dc.reset_verdicts()


def test_direct_copy_oom_re_raises_no_verdict(dc, clean):
    dc.FORCED = False

    class _OOMTensor:
        device = torch.device("cuda", 0)

        def to(self, _device, non_blocking = False):
            raise torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.OutOfMemoryError):
        dc.to_device(_OOMTensor(), torch.device("cuda", 1))

    assert dc._verdicts == {}
    assert dc.stats["probes"] == 0
    assert dc.stats["bounced"] == 0
    assert dc.stats["direct"] == 0


def test_probe_oom_re_raises_no_verdict(dc, clean, monkeypatch):
    dc.FORCED = None

    class _Src:
        device = torch.device("cuda", 0)

    def _oom_rand(*_args, **_kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(dc.torch, "rand", _oom_rand)
    with pytest.raises(torch.OutOfMemoryError):
        dc.to_device(_Src(), torch.device("cuda", 1))

    # needs_bounce bumps `probes` before the probe runs; the raise happens inside it,
    # before any verdict is stored.
    assert dc._verdicts == {}
    assert dc.stats["probes"] == 1