"""Inkling2 whole-model forward (loader/inkling2_forward.py), CPU, torch only (no extension, no GPU).

  - dequant: NVFP4 (ggml type 40) vs a literal port of dequantize_row_nvfp4, Q8_0, BF16, F32 round trips
  - twin (tools/inkling2/make_twin.py, F32 tensors, 8 physical layers, 3 loop passes = 16 applications; needs gguf-py of the llama.cpp
    clone and the real GGUF tokenizer KVs, else skipped):
      * one prefill == token by token decode (KV + conv state of every application), and == chunked prefill (chunk 3 / 256)
      * 16 separate KV slots (the loop passes do not share), SWA slots trimmed to the window
      * l_out of every application + result_output vs the llama.cpp CPU oracle dumps (tools/inkling2/compare_oracle.py), bar 1e-3,
        when ~/scratch/inkling2/oracle/twin3f_dump exists (llama-inkling-dump on the same twin, KV f32, fa off)
Bars: self-consistency 1e-4 (fp32 reordering), oracle 1e-3 max abs / max |ref| (measured 2.4e-5)."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "inkling2"))
import cpu_env  # noqa: E402

F = cpu_env.install(ROOT)
H = os.environ.get("INK_H", os.path.expanduser("~"))      # set INK_H when HOME differs (runner env)
SRC = f"{H}/models/inkling2-mini/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf"
GGUFPY = os.environ.get("INK_GGUFPY", f"{H}/scratch/inkling_llama/gguf-py")
ORACLE = f"{H}/scratch/inkling2/oracle"
IDS = [1, 5, 9, 100, 2000, 77, 5, 6, 300, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def _nvfp4_c(blk):
    """literal port of ggml dequantize_row_nvfp4 + ggml_ue4m3_to_fp32 for one 36-byte block (64 values)"""
    kv = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
    y = np.zeros(64, np.float32)
    for s in range(4):
        x = int(blk[s])
        if x == 0 or x == 0x7F:
            d = 0.0
        else:
            e, m = (x >> 3) & 15, x & 7
            d = (m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)) * 0.5
        for j in range(8):
            q = int(blk[4 + s * 8 + j])
            y[s * 16 + j] = kv[q & 15] * d
            y[s * 16 + j + 8] = kv[q >> 4] * d
    return y


def test_dq_nvfp4_matches_ggml():
    rng = np.random.default_rng(1)
    rows, K = 3, 128
    raw = rng.integers(0, 256, size=(rows, K // 64 * 36), dtype=np.uint8)
    raw[..., ::36][:] = rng.integers(0x10, 0x40, size=raw[..., ::36].shape, dtype=np.uint8)
    got = F.dq_bytes(torch.from_numpy(raw.reshape(-1).copy()), 40, rows, K).numpy()
    for r in range(rows):
        for b in range(K // 64):
            ref = _nvfp4_c(raw[r, b * 36:(b + 1) * 36])
            assert np.array_equal(got[r, b * 64:(b + 1) * 64], ref)


def test_nvfp4_repack_modelopt():
    """GGUF type 40 bytes -> ModelOpt layout: the numpy ModelOpt reference (foreign_quant.nvfp4) decodes to the same values as the ggml dequant"""
    from exllamav3.foreign_quant import nvfp4 as MO
    rng = np.random.default_rng(3)
    rows, K = 5, 192
    raw = rng.integers(0, 256, size=(rows, K // 64 * 36), dtype=np.uint8)
    raw.reshape(rows, -1, 36)[..., :4] = rng.integers(0x08, 0x70, size=(rows, K // 64, 4), dtype=np.uint8)      # finite non-zero scales
    t = torch.from_numpy(raw)
    packed, scales = F.nvfp4_gguf_to_modelopt(t)
    assert packed.shape == (rows, K // 2) and scales.shape == (rows, K // 16)
    ref = F.dq_bytes(t.reshape(-1), 40, rows, K).numpy()
    got = MO.dequant_nvfp4(packed.numpy(), scales.numpy(), 1.0)
    assert np.array_equal(got, ref)


def test_dq_q8_bf16_f32():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 64)).astype(np.float32)
    d = (np.abs(x.reshape(2, 2, 32)).max(-1) / 127).astype(np.float16)
    q = np.round(x.reshape(2, 2, 32) / np.where(d == 0, 1, d).astype(np.float32)[..., None]).astype(np.int8)
    blk = np.concatenate([d.view(np.uint8).reshape(2, 2, 2), q.view(np.uint8)], -1).reshape(-1)
    ref = (d.astype(np.float32)[..., None] * q.astype(np.float32)).reshape(2, 64)
    assert np.array_equal(F.dq_bytes(torch.from_numpy(blk.copy()), 8, 2, 64).numpy(), ref)
    bf = torch.from_numpy(x).bfloat16()
    assert torch.equal(F.dq_bytes(bf.view(torch.uint8).reshape(-1), 30, 2, 64), bf.float())
    assert np.array_equal(F.dq_bytes(torch.from_numpy(x.view(np.uint8).reshape(-1).copy()), 0, 2, 64).numpy(), x)


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    if not (os.path.exists(SRC) and os.path.isdir(GGUFPY)):
        pytest.skip("real GGUF (tokenizer KVs) or gguf-py absent")
    out = str(tmp_path_factory.mktemp("twin") / "twin3f.gguf")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "inkling2", "make_twin.py"), out, "--src", SRC, "--f32", "--loops", "3"],
                       capture_output=True, text=True, env={**os.environ, "INK_GGUFPY": GGUFPY})
    assert r.returncode == 0, r.stderr[-800:]
    return out


def _model(path, chunk=256):
    from exllamav3.loader.gguf_collection import GGUFCollection
    return F.Inkling2Forward(GGUFCollection(path), chunk=chunk)


def test_twin_loop_structure(twin):
    m = _model(twin)
    assert (m.n_app, m.n_phys, m.n_loops, m.core_start, m.core_end) == (16, 8, 3, 2, 6)
    assert [m.loop_layer(i) for i in range(16)] == [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]


def test_prefill_equals_decode_and_chunks(twin):
    full = _model(twin).step(IDS, all_pos=True)
    dec = _model(twin)
    rows = [dec.step([t]) for t in IDS]
    assert float((torch.stack(rows) - full).abs().max() / full.abs().max()) < 1e-4
    ch = _model(twin, chunk=3).step(IDS, all_pos=True)
    assert float((ch - full).abs().max() / full.abs().max()) < 1e-4
    # separate KV slots per application; SWA slots hold at most the window
    assert all(s.k is not None for s in dec.slots) and len(dec.slots) == 16
    assert not torch.equal(dec.slots[2].k, dec.slots[6].k)
    for il, s in enumerate(dec.slots):
        assert s.k.shape[1] == (min(len(IDS), dec.window) if dec.pattern[il] else len(IDS))


def test_oracle_dump(twin):
    dump = os.path.join(ORACLE, "twin3f_dump")
    if not os.path.isdir(dump):
        pytest.skip("no llama.cpp oracle dump")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "inkling2", "compare_oracle.py"), os.path.join(ORACLE, "twin3f.gguf"), dump,
                        ",".join(map(str, IDS)) + ";4,7,8", "--tol", "1e-3", "--threads", "4"], capture_output=True, text=True)
    assert "PASS" in r.stdout.splitlines()[-1], r.stdout[-600:]
