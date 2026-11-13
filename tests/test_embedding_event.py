"""CPU regression test for the shared CPU-embedding upload event.

A split places the target's first layer and the MTP block on different devices while
both gather from the same CPU-resident embedding table.  `_narrow_ev` must belong to
the device it is recorded on, so it has to be recreated when the upload device changes.
The test fakes `torch.cuda.Event`/`torch.cuda.device`, so it runs without a GPU: the
bug was a torch-side device mismatch, not a kernel issue.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _package(name: str, path: Path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    return module


def _registered(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _StubModule:
    def __init__(self, config, key, layer_idx = None):
        self.config = config
        self.key = key
        self.layer_idx = layer_idx
        self.caps = {}
        self.device = None


@pytest.fixture(scope = "module")
def embedding_module():
    """Load `embedding.py` with stub exllamav3 packages (the real package pulls in the extension)."""
    saved = dict(sys.modules)
    try:
        sys.modules["exllamav3"] = _package("exllamav3", ROOT / "exllamav3")
        modules = _package("exllamav3.modules", ROOT / "exllamav3" / "modules")
        modules.Module = _StubModule
        sys.modules["exllamav3.modules"] = modules
        sys.modules["exllamav3.model"] = _package("exllamav3.model", ROOT / "exllamav3" / "model")
        sys.modules["exllamav3.util"] = _package("exllamav3.util", ROOT / "exllamav3" / "util")
        sys.modules["exllamav3.tokenizer"] = _package(
            "exllamav3.tokenizer", ROOT / "exllamav3" / "tokenizer")
        _registered("exllamav3.model.config", Config = object)
        _registered("exllamav3.model.model_tp_alloc", TPAllocation = object)
        _registered("exllamav3.util.tensor", to2 = lambda x, *_a, **_k: x)
        _registered("exllamav3.util.device_copy", to_device = lambda x, device: x.to(device))
        _registered("exllamav3.tokenizer.mm_embedding",
                    FIRST_MM_EMBEDDING_INDEX = 1, mm_rows = None)

        spec = importlib.util.spec_from_file_location(
            "exllamav3.modules.embedding", ROOT / "exllamav3" / "modules" / "embedding.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


def _embedding(module):
    return module.Embedding(config = None, key = "model.embed_tokens",
                            vocab_size = 16, hidden_size = 4)


class _FakeTransfer:
    """The only tensor API `_upload_narrow` uses."""
    dtype = torch.float16

    def to(self, *_args, **_kwargs):
        return self


def _fake_cuda(monkeypatch, created, contexts):
    class Event:
        def __init__(self):
            self.records = 0

        def record(self):
            self.records += 1

    class DeviceContext:
        def __init__(self, device):
            self.device = torch.device(device)

        def __enter__(self):
            contexts.append(self.device)

        def __exit__(self, *_exc):
            return False

    def make_event(*_args, **_kwargs):
        created.append(Event())
        return created[-1]

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "device", DeviceContext)


def test_upload_event_is_recreated_when_device_changes(embedding_module, monkeypatch):
    emb = _embedding(embedding_module)
    created, contexts = [], []
    _fake_cuda(monkeypatch, created, contexts)
    ids = torch.zeros(4, dtype = torch.long)
    y = _FakeTransfer()

    emb._upload_narrow(y, ids, torch.float16, torch.device("cuda:0"))
    first = emb._narrow_ev
    assert len(created) == 1 and first.records == 1
    assert emb._narrow_ev_device == torch.device("cuda:0")

    # Same device: reuse the event (the steady-state upload path).
    emb._upload_narrow(y, ids, torch.float16, torch.device("cuda:0"))
    assert len(created) == 1 and first.records == 2

    # The MTP block's device differs: the old event would fail to record on it.
    emb._upload_narrow(y, ids, torch.float16, torch.device("cuda:1"))
    second = emb._narrow_ev
    assert len(created) == 2 and second is not first
    assert first.records == 2 and second.records == 1
    assert emb._narrow_ev_device == torch.device("cuda:1")

    emb._upload_narrow(y, ids, torch.float16, torch.device("cuda:1"))
    assert len(created) == 2 and second.records == 2
    assert contexts == [torch.device("cuda:0")] * 3 + [torch.device("cuda:1")] * 3


def test_gather_reallocation_forgets_the_event_device(embedding_module, monkeypatch):
    emb = _embedding(embedding_module)
    emb.embedding = SimpleNamespace(weight = torch.zeros(16, 4, dtype = torch.float16))
    emb._narrow_buf = None
    emb._narrow_ev = object()
    emb._narrow_ev_device = torch.device("cuda:0")

    buf = torch.empty((2, 4), dtype = torch.float16)
    monkeypatch.setattr(torch, "empty", lambda *_a, **_k: buf)
    monkeypatch.setattr(torch, "index_select", lambda _w, _dim, _idx, out = None: out)

    emb._gather_narrow(torch.zeros(2, dtype = torch.long))
    assert emb._narrow_ev is None
    assert emb._narrow_ev_device is None
