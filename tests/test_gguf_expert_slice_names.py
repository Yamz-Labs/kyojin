"""CPU test: GGUFCollection resolves per-expert slice names (blk.N.ffn_{gate,up,down}_exps.eK.weight),
both when passed directly and when produced by a name map."""
import os
import struct
import tempfile

import numpy as np
import pytest
import torch

import importlib.util

# Load the module by path: importing the exllamav3 package would build/load the CUDA extension (not needed here)
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3", "loader", "gguf_collection.py")
_spec = importlib.util.spec_from_file_location("gguf_collection_under_test", _p)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GGUFCollection = _mod.GGUFCollection

K, N, E = 64, 6, 4          # K = row length (2 Q8_0 blocks of 32), N rows, E experts
NB, BB = K // 32, 34        # Q8_0: 32 elements in 34 bytes
ROW = NB * BB
NAMES = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]


def _write_gguf(path, payloads):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, len(payloads), 0)
    infos, off = b"", 0
    for name, data in payloads.items():
        infos += s(name) + struct.pack("<I", 3) + struct.pack("<QQQ", K, N, E) + struct.pack("<I", 8) + struct.pack("<Q", off)
        off += (len(data) + 31) // 32 * 32
    blob = head + infos
    blob += b"\0" * (-len(blob) % 32)
    for data in payloads.values():
        blob += data + b"\0" * (-len(data) % 32)
    with open(path, "wb") as f:
        f.write(blob)


@pytest.fixture(scope="module")
def gguf():
    rng = np.random.default_rng(0)
    raw = {f"blk.0.{n}.weight": rng.integers(0, 256, E * N * ROW, dtype=np.uint8) for n in NAMES}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.gguf")
        _write_gguf(p, {k: v.tobytes() for k, v in raw.items()})
        yield GGUFCollection(p), raw


def _expect(raw, name, e):
    return raw[f"blk.0.{name}.weight"].reshape(E, N, ROW)[e]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("e", [0, 3])
def test_direct_slice_name(gguf, name, e):
    c, raw = gguf
    key = f"blk.0.{name}.e{e}.weight"
    assert c.has_tensor(key)
    t = c.get_tensor(key)
    assert t.shape == (N, ROW)
    assert np.array_equal(t.numpy(), _expect(raw, name, e))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("e", [0, 3])
def test_slice_name_after_name_map(gguf, name, e):
    c, raw = gguf
    ek = f"model.layers.0.mlp.experts.{e}.{name}.weight"
    c.name_map = {ek: f"blk.0.{name}.e{e}.weight",
                  ek + ".nosfx": f"blk.0.{name}.e{e}"}
    try:
        assert c.has_tensor(ek)
        assert np.array_equal(c.get_tensor(ek).numpy(), _expect(raw, name, e))
        assert c.has_tensor(ek + ".nosfx")
        assert np.array_equal(c.get_tensor(ek + ".nosfx").numpy(), _expect(raw, name, e))
    finally:
        c.name_map = {}


def test_missing_slice_not_found(gguf):
    c, _ = gguf
    c.name_map = {"x": "blk.0.ffn_gate_exps.eX.weight", "y": "blk.0.nope_exps.e1.weight"}
    try:
        assert not c.has_tensor("x")
        assert not c.has_tensor("y")
    finally:
        c.name_map = {}
