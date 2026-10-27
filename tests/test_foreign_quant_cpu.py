"""CPU-only tests for exllamav3.foreign_quant (ggml Q8_0 / IQ2_XXS, NVFP4).

Modules are loaded by file path (stdlib+numpy only, no torch). Three
independent oracles are used:

* a C program (tests/foreign_quant_oracle/dequant_oracle.c) carrying the
  verbatim llama.cpp dequant algorithm + tables (MIT) - bit-exact checks;
* ml_dtypes fp8/fp4 codebooks (skipped if not installed);
* real Bolt checkpoint tensors fetched with HTTP Range requests (no full
  download; skipped if the network is unavailable).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request

import numpy as np
import pytest

FQ_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "exllamav3", "foreign_quant"))
ORACLE_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "foreign_quant_oracle", "dequant_oracle.c"))
HF_REPO = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main/"


def _load(name):
    modname = f"foreign_quant_{name}_under_test"
    if modname in sys.modules:
        return sys.modules[modname]
    path = os.path.join(FQ_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


q8_0 = _load("q8_0")
iq2_xxs = _load("iq2_xxs")
nvfp4 = _load("nvfp4")


# --------------------------------------------------------------------------
# C oracle (verbatim llama.cpp algorithm)
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def oracle(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")
    exe = str(tmp_path_factory.mktemp("oracle") / "dequant_oracle")
    try:
        subprocess.run(["gcc", "-O2", "-o", exe, ORACLE_SRC],
                       check=True, capture_output=True, timeout=120)
    except Exception as e:
        pytest.skip(f"oracle build failed: {e}")
    return exe


def _oracle_run(oracle_exe, mode, payload: bytes, tmp_path):
    f = tmp_path / f"in_{mode}.bin"
    f.write_bytes(bytes(payload))
    r = subprocess.run([oracle_exe, mode, str(f)], check=True,
                       capture_output=True, timeout=120)
    return np.array([int(x, 16) for x in r.stdout.decode().split()], dtype=np.uint32)


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


# --------------------------------------------------------------------------
# Q8_0
# --------------------------------------------------------------------------

def test_q8_0_closed_form():
    blk = bytes([0x00, 0x3C]) + bytes(range(32))  # d = 1.0, qs = 0..31
    out = q8_0.dequant_q8_0_block(blk)
    assert out.shape == (32,)
    np.testing.assert_array_equal(out, np.arange(32, dtype=np.float32))


def test_q8_0_negative_and_scale():
    blk = struct.pack("<H", 0xBC00) + bytes([0x80, 0xFF, 0x7F] + [0] * 29)
    out = q8_0.dequant_q8_0_block(blk)  # d = -1.0
    assert out[0] == np.float32(128.0) and out[1] == np.float32(1.0)
    assert out[2] == np.float32(-127.0)


def test_q8_0_vs_oracle(oracle, tmp_path):
    rng = np.random.default_rng(0)
    n = 5
    payload = bytearray()
    for _ in range(n):
        payload += rng.integers(0, 256, size=2).astype(np.uint8).tobytes()
        payload += rng.integers(0, 256, size=32).astype(np.uint8).tobytes()
    ref = _oracle_run(oracle, "q8_0", bytes(payload), tmp_path)
    got = q8_0.dequant_q8_0(np.frombuffer(bytes(payload), dtype=np.uint8).reshape(n, 34))
    assert got.shape == (n, 32)
    np.testing.assert_array_equal(_bits(got).ravel(), ref)


def test_fp16_converter_vs_numpy(oracle, tmp_path):
    edge = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00,
            0x7BFF, 0x7C00, 0xFC00, 0x7C01, 0x3555, 0xC000]
    rng = np.random.default_rng(1)
    vals = edge + [int(v) for v in rng.integers(0, 65536, size=200)]
    payload = struct.pack(f"<{len(vals)}H", *vals)
    ref = _oracle_run(oracle, "fp16", payload.encode() if isinstance(payload, str) else payload,
                      tmp_path)
    got = np.array(vals, dtype=np.uint16).view("<f2").astype(np.float32)
    np.testing.assert_array_equal(_bits(got), ref)


# --------------------------------------------------------------------------
# IQ2_XXS
# --------------------------------------------------------------------------

def test_iq2_xxs_grid_values():
    assert set(int(v) for v in np.unique(iq2_xxs._GRID_BYTES)) == {8, 25, 43}


def test_iq2_xxs_closed_form():
    blk = bytes([0x00, 0x3C]) + bytes(64)  # d = 1.0, all indices 0
    out = iq2_xxs.dequant_iq2_xxs_block(blk)
    assert out.shape == (256,)
    # db = 1*(0.5+0)*0.25 = 0.125; grid[0] = all 8s; signs = ksigns[0] = 0.
    np.testing.assert_array_equal(out, np.full(256, np.float32(1.0)))


def test_iq2_xxs_vs_oracle(oracle, tmp_path):
    rng = np.random.default_rng(2)
    n = 4
    payload = rng.integers(0, 256, size=(n, 66)).astype(np.uint8)
    ref = _oracle_run(oracle, "iq2_xxs", payload.tobytes(), tmp_path)
    got = iq2_xxs.dequant_iq2_xxs(payload)
    assert got.shape == (n, 256)
    np.testing.assert_array_equal(_bits(got).ravel(), ref)


# --------------------------------------------------------------------------
# NVFP4
# --------------------------------------------------------------------------

def test_nvfp4_e2m1_lut():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    ref = np.arange(16, dtype=np.uint8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
    np.testing.assert_array_equal(np.array(nvfp4.E2M1_LUT, dtype=np.float32), ref)


def test_nvfp4_e4m3_all_codes():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    codes = np.arange(256, dtype=np.uint8)
    ref = codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    got = nvfp4.e4m3_to_float32(codes)
    # Exact for all finite codes (powers of two times (1+m/8)); NaNs compared
    # by payload-insensitive equality.
    np.testing.assert_allclose(got, ref, rtol=0, atol=0, equal_nan=True)


def test_nvfp4_closed_form():
    # One full group of 16: nibbles (1,1),(0,7),(0,0) x1 + 13 more zeros.
    packed = np.array([[0x11, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]],
                      dtype=np.uint8)
    scales = np.array([[0x38]], dtype=np.uint8)  # 1.0
    out = nvfp4.dequant_nvfp4(packed, scales, 2.0)
    assert out.shape == (1, 16)
    np.testing.assert_array_equal(out[0, :4], np.array([1.0, 1.0, 0.0, 12.0],
                                                       dtype=np.float32))
    np.testing.assert_array_equal(out[0, 4:], np.zeros(12, dtype=np.float32))


def test_nvfp4_shape_mismatch():
    with pytest.raises(ValueError):
        nvfp4.dequant_nvfp4(np.zeros((2, 8), np.uint8), np.zeros((2, 2), np.uint8), 1.0)


# --------------------------------------------------------------------------
# Real Bolt tensors via HTTP Range (no full download)
# --------------------------------------------------------------------------

def _http_range(url, start, n, timeout=120):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + n - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        assert r.status in (200, 206), f"range request failed: {r.status}"
        data = r.read(n)
    assert len(data) == n, f"short read: {len(data)} != {n}"
    return data


@pytest.fixture(scope="session")
def hf_tensors():
    try:
        with urllib.request.urlopen(HF_BASE + "model.safetensors.index.json",
                                    timeout=120) as r:
            index = json.load(r)
    except Exception as e:
        pytest.skip(f"HF unreachable: {e}")
    weight_map = index["weight_map"]
    headers = {}

    def shard_header(fname):
        if fname not in headers:
            raw = _http_range(HF_BASE + fname, 0, 8)
            (hlen,) = struct.unpack("<Q", raw)
            headers[fname] = (8 + hlen, json.loads(_http_range(HF_BASE + fname, 8, hlen)))
        return headers[fname]

    def fetch(name, rows):
        fname = weight_map[name]
        base, header = shard_header(fname)
        meta = header[name]
        shape, (start, end) = meta["shape"], meta["data_offsets"]
        esize = {"F32": 4, "F16": 2, "BF16": 2, "U8": 1, "F8_E4M3": 1}[meta["dtype"]]
        nrows = shape[0] if shape else 1
        row_n = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        assert end - start == nrows * row_n * esize
        lo = base + start + rows.start * row_n * esize
        hi = base + start + rows.stop * row_n * esize
        return _http_range(HF_BASE + fname, lo, hi - lo), meta

    out = {}
    try:
        out["lm_head"], out["lm_head_meta"] = fetch("language_model.lm_head.weight",
                                                    range(0, 8))
        out["fc1"], out["fc1_meta"] = fetch(
            "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight", range(0, 2))
        out["exp0up"], out["exp0up_meta"] = fetch(
            "language_model.backbone.layers.1.mixer.experts.0.up_proj.weight", range(0, 2))
        w, wmeta = fetch("language_model.backbone.layers.0.mixer.in_proj.weight",
                         range(0, 2))
        s, smeta = fetch("language_model.backbone.layers.0.mixer.in_proj.weight_scale",
                         range(0, 2))
        g, gmeta = fetch("language_model.backbone.layers.0.mixer.in_proj.weight_scale_2",
                         range(0, 1))
        out.update(in_w=w, in_wmeta=wmeta, in_s=s, in_smeta=smeta, in_g=g)
    except Exception as e:
        pytest.skip(f"HF range fetch failed: {e}")
    return out


def test_real_q8_0_lm_head(hf_tensors):
    meta = hf_tensors["lm_head_meta"]
    assert meta["dtype"] == "U8" and meta["shape"][1:] == [128, 34]
    raw = np.frombuffer(hf_tensors["lm_head"], dtype=np.uint8).reshape(8, 128, 34)
    out = q8_0.dequant_q8_0(raw)
    assert out.shape == (8, 128, 32) and np.all(np.isfinite(out))
    assert 1e-4 < float(out.std()) < 1.0  # d-first layout sanity


def test_real_q8_0_latent_proj(hf_tensors, oracle, tmp_path):
    meta = hf_tensors["fc1_meta"]
    assert meta["dtype"] == "U8" and meta["shape"][1:] == [128, 34]
    raw = np.frombuffer(hf_tensors["fc1"], dtype=np.uint8).reshape(2, 128, 34)
    out = q8_0.dequant_q8_0(raw)
    assert np.all(np.isfinite(out)) and 1e-4 < float(out.std()) < 1.0
    ref = _oracle_run(oracle, "q8_0", hf_tensors["fc1"], tmp_path)
    np.testing.assert_array_equal(_bits(out).ravel(), ref)


def test_real_iq2_xxs_expert(hf_tensors, oracle, tmp_path):
    meta = hf_tensors["exp0up_meta"]
    assert meta["dtype"] == "U8" and meta["shape"][1:] == [4, 66]
    raw = np.frombuffer(hf_tensors["exp0up"], dtype=np.uint8).reshape(2, 4, 66)
    out = iq2_xxs.dequant_iq2_xxs(raw)
    assert out.shape == (2, 4, 256) and np.all(np.isfinite(out))
    assert 1e-4 < float(out.std()) < 5.0
    ref = _oracle_run(oracle, "iq2_xxs", hf_tensors["exp0up"], tmp_path)
    np.testing.assert_array_equal(_bits(out).ravel(), ref)


def test_real_nvfp4_in_proj(hf_tensors):
    assert hf_tensors["in_wmeta"]["shape"][1:] == [2048]
    assert hf_tensors["in_smeta"]["shape"][1:] == [256]
    packed = np.frombuffer(hf_tensors["in_w"], dtype=np.uint8).reshape(2, 2048)
    scales = np.frombuffer(hf_tensors["in_s"], dtype=np.uint8).reshape(2, 256)
    (g,) = struct.unpack("<f", hf_tensors["in_g"])
    assert np.isfinite(g) and g > 0
    svals = nvfp4.e4m3_to_float32(scales)
    assert np.all(np.isfinite(svals)) and np.all(svals > 0) and np.all(svals <= 600)
    out = nvfp4.dequant_nvfp4(packed, scales, g)
    assert out.shape == (2, 4096) and np.all(np.isfinite(out))
    assert 1e-4 < float(out.std()) < 5.0
