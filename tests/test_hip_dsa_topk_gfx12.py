"""Contracts for the gfx12 QSA dsa_topk path (k <= 512)."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm dsa_topk tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext
from exllamav3 import ext_fallbacks
from exllamav3.modules.qsa_indexer import QSAIndexer

K = 512
INT_MAX = 2**31 - 1


def _gfx12_devices() -> list[int]:
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        arch = getattr(props, "gcnArchName", "").split(":", 1)[0]
        if arch in ("gfx1200", "gfx1201", "gfx1150", "gfx1151", "gfx1152") and getattr(props, "warp_size", 0) == 32:
            devices.append(index)
    return devices


GFX12_DEVICES = _gfx12_devices()
if not GFX12_DEVICES:
    pytest.skip("requires a wave32 gfx11.5/gfx12 device", allow_module_level = True)
if not hasattr(ext, "dsa_topk_gfx12"):
    pytest.skip("dsa_topk_gfx12 binding is unavailable", allow_module_level = True)
# The full-width selection route exists only while the tiled selection bindings are absent
# (they are CUDA-only); the select-rows test below covers that route.
assert not hasattr(ext, "dsa_topk_tile")


@pytest.fixture(params = GFX12_DEVICES, ids = lambda index: f"cuda:{index}")
def device(request) -> torch.device:
    return torch.device("cuda", request.param)


def _scores(device: torch.device, rows: int, width: int, *, seed: int = 1201) -> torch.Tensor:
    generator = torch.Generator(device = device).manual_seed(seed)
    stride = -(-width // 128) * 128
    storage = torch.randn((rows, stride), generator = generator, device = device, dtype = torch.half)
    return storage[:, :width]


def _fp16_bit_key_stable_oracle(
    scores: torch.Tensor, k: int = K, t_ptr = None, t_seq: int = 0,
) -> torch.Tensor:
    """Select by the fp16 bit key, breaking the threshold tie by ascending index."""
    rows, width = scores.shape
    raw_rows = scores.detach().view(torch.int16).cpu().tolist()
    bounds = None if t_ptr is None else t_ptr.detach().reshape(-1).cpu().tolist()
    result = []
    for row, raw in enumerate(raw_rows):
        bound = width if bounds is None else bounds[row // t_seq] if t_seq > 0 else bounds[0]
        candidates = []
        for index, signed_bits in enumerate(raw[:max(0, min(bound, width))]):
            bits = signed_bits & 0xffff
            key = (~bits & 0xffff) if bits & 0x8000 else bits | 0x8000
            if key > 0x03ff:
                candidates.append((key, index))
        chosen = sorted(candidates, key = lambda item: (-item[0], item[1]))[:k]
        result.append(sorted(index for _, index in chosen) + [-1] * (k - len(chosen)))
    return torch.tensor(result, device = scores.device, dtype = torch.int32)


@contextmanager
def _force_fallback(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(ext_fallbacks, "dsa_topk_gfx12_supported", lambda *_args: False)
        yield


@pytest.mark.parametrize("rows,width,cutoff", [
    (1, 513, 511), (32, 3072, 1023), (64, 32768, 1023),
    (1024, 512, 511), (1024, 513, 511), (1025, 512, 511), (1025, 513, 511),
])
@torch.inference_mode()
def test_gfx12_qsa_topk_matches_fp16_bit_key_oracle(device, rows, width, cutoff):
    scores = torch.full((rows, width), -float("inf"), device = device, dtype = torch.half)
    for row in range(rows):
        scores[row, :max(0, cutoff - 8)] = 1.0
        start = max(0, cutoff - 8)
        end = min(width, cutoff + 9)
        if end > start:
            scores[row, start:end] = 0.0
        if cutoff + 9 < width:
            scores[row, cutoff + 9:] = -1.0
    expected = _fp16_bit_key_stable_oracle(scores)
    actual = torch.empty_like(expected)
    ext.dsa_topk(scores, actual, K)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_gfx12_qsa_topk_route_uses_native_without_fallback(device, monkeypatch):
    scores = torch.zeros((4, 1024), device = device, dtype = torch.half)
    scores[:, 500:] = 1
    expected = _fp16_bit_key_stable_oracle(scores)
    actual = torch.empty_like(expected)
    calls = {"native": 0}
    native = ext.dsa_topk_gfx12

    def spy(*args):
        calls["native"] += 1
        return native(*args)

    def unexpected_fallback(*_args, **_kwargs):
        pytest.fail("eligible gfx12 dsa_topk unexpectedly used the fallback")

    monkeypatch.setattr(ext, "dsa_topk_gfx12", spy)
    monkeypatch.setattr(ext_fallbacks, "dsa_topk", unexpected_fallback)
    ext.dsa_topk(scores, actual, K)
    torch.cuda.synchronize(device)
    assert calls["native"] == 1
    assert torch.equal(actual, expected)
    assert torch.equal(actual[0], torch.arange(500, 1012, device = device, dtype = torch.int32))


@torch.inference_mode()
def test_gfx12_qsa_topk_ties_and_special_values_match_fp16_bit_key_oracle(device):
    raw = torch.full((3, 1024), -1024, device = device, dtype = torch.int16)
    raw[0, :511] = 0
    raw[0, 511] = 1
    raw[0, 512] = 1
    raw[1, :512] = 0x7c00  # +inf
    raw[1, 512:] = -1024   # -inf
    raw[2, :512] = 0x7e00  # positive NaN ranks above +inf in CUDA key order
    raw[2, 512:] = -512    # 0xfe00: negative quiet NaN, excluded by the fp16 key
    scores = raw.view(torch.float16)
    expected = _fp16_bit_key_stable_oracle(scores)
    actual = torch.empty_like(expected)
    ext.dsa_topk(scores, actual, K)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[0], torch.cat((
        torch.arange(510, device = device, dtype = torch.int32),
        torch.tensor([511, 512], device = device, dtype = torch.int32),
    )))
    assert torch.equal(actual[2], torch.arange(K, device = device, dtype = torch.int32))
    assert not torch.isin(actual[2], torch.arange(K, 1024, device = device, dtype = torch.int32)).any()


@pytest.mark.parametrize("valid_count", [0, 1, 511, 512])
@torch.inference_mode()
def test_gfx12_qsa_topk_low_valid_counts_have_exact_indices_and_padding(device, valid_count):
    raw = torch.full((1, 1024), -1024, device = device, dtype = torch.int16)  # -inf
    raw[0, :valid_count] = 0  # +0, a valid fp16 bit key
    scores = raw.view(torch.float16)
    expected = _fp16_bit_key_stable_oracle(scores)
    actual = torch.full_like(expected, -2)

    ext.dsa_topk_gfx12(scores, actual)
    torch.cuda.synchronize(device)

    exact = torch.full((K,), -1, device = device, dtype = torch.int32)
    exact[:valid_count] = torch.arange(valid_count, device = device, dtype = torch.int32)
    assert torch.equal(expected[0], exact)
    assert torch.equal(actual, expected)
    if valid_count == 512:
        assert (actual[0] != -1).all()
        assert torch.equal(actual[0], torch.arange(K, device = device, dtype = torch.int32))


@pytest.mark.parametrize("bad", [
    "scores-dtype", "output-dtype", "output-shape", "score-stride", "wide-width", "wide-stride",
])
@torch.inference_mode()
def test_gfx12_binding_rejects_malformed_fixed_k_inputs(device, bad):
    scores = _scores(device, 2, 1024)
    out = torch.empty((2, K), device = device, dtype = torch.int32)
    if bad == "scores-dtype":
        scores = scores.float()
    elif bad == "output-dtype":
        out = out.long()
    elif bad == "output-shape":
        out = out[:, :-1]
    elif bad == "score-stride":
        scores = scores[:, ::2]
    elif bad == "wide-width":
        scores = torch.empty((0, INT_MAX + 1), device = device, dtype = torch.half)
        out = torch.empty((0, K), device = device, dtype = torch.int32)
    else:
        scores = torch.empty_strided((0, K), (INT_MAX + 128, 1), device = device, dtype = torch.half)
        out = torch.empty((0, K), device = device, dtype = torch.int32)
    if bad.startswith("wide-"):
        assert not ext_fallbacks.dsa_topk_gfx12_supported(scores, out, K)
    with pytest.raises(RuntimeError, match = "dsa_topk_gfx12"):
        ext.dsa_topk_gfx12(scores, out)


@pytest.mark.parametrize("case", ["wide-out", "t-ptr", "stride"])
@torch.inference_mode()
def test_gfx12_qsa_topk_unsupported_inputs_keep_the_fallback_contract(device, case, monkeypatch):
    scores = _scores(device, 2, 1024)
    out = torch.empty((2, K), device = device, dtype = torch.int32)
    kwargs = {}
    if case == "wide-out":
        k = K
        out = torch.empty((2, K + 32), device = device, dtype = torch.int32)
    elif case == "t-ptr":
        k = K
        kwargs = {"t_ptr": torch.tensor([400], device = device, dtype = torch.int32)}
    else:
        k = K
        scores = scores[:, ::2]

    native_calls = {"count": 0}
    native = ext.dsa_topk_gfx12

    def spy(*args):
        native_calls["count"] += 1
        return native(*args)

    monkeypatch.setattr(ext, "dsa_topk_gfx12", spy)
    if case == "stride":
        with pytest.raises(ValueError, match = "layout"):
            ext.dsa_topk(scores, out, k, **kwargs)
        with pytest.raises(ValueError, match = "layout"):
            ext_fallbacks.dsa_topk(scores, torch.empty_like(out), k, **kwargs)
    else:
        selected = _fp16_bit_key_stable_oracle(scores, k, **kwargs)
        expected = torch.full_like(out, -1)
        expected[:, :k] = selected
        fallback = torch.full_like(out, 12345)
        ext_fallbacks.dsa_topk(scores, fallback, k, **kwargs)
        actual = torch.full_like(out, 12345)
        ext.dsa_topk(scores, actual, k, **kwargs)
        torch.cuda.synchronize(device)
        assert torch.equal(fallback, expected)
        assert torch.equal(actual, expected)
        assert (actual[:, k:] == -1).all()
        if case == "t-ptr":
            selected = actual[actual >= 0]
            assert (selected < 400).all()
            assert (actual[:, 400:] == -1).all()
    assert native_calls["count"] == 0


@pytest.mark.parametrize("width,k,out_w", [
    (1, 1, 32), (64, 64, 64), (100, 33, 64), (320, 320, 320), (448, 448, 448),
    (511, 511, 512), (511, 200, 224), (1024, 1, 32), (1024, 511, 512), (3000, 257, 288),
])
@torch.inference_mode()
def test_gfx12_qsa_topk_small_k_native_matches_fallback(device, width, k, out_w, monkeypatch):
    """moeA5: k < 512, T < 512 and narrow outputs run natively, bitwise with the fallback."""
    scores = _scores(device, 300, width)  # 128-aligned row stride
    scores[:, ::7] = scores[:, :1]  # ties
    causal = torch.arange(width, device = device)[None] >= (torch.arange(300, device = device)[:, None] * width // 300 + 1)
    scores.masked_fill_(causal, -float("inf"))
    scores[5, 0] = float("nan")
    scores[7, :] = -float("inf")
    fallback = torch.full((300, out_w), 12345, device = device, dtype = torch.int32)
    ext_fallbacks.dsa_topk(scores, fallback, k)
    calls = {"native": 0}
    native = ext.dsa_topk_gfx12

    def spy(*args):
        calls["native"] += 1
        return native(*args)

    monkeypatch.setattr(ext, "dsa_topk_gfx12", spy)
    actual = torch.full_like(fallback, 12345)
    ext.dsa_topk(scores, actual, k)
    torch.cuda.synchronize(device)
    assert calls["native"] == 1
    assert torch.equal(actual, fallback)
    assert torch.equal(fallback[:3, :k], _fp16_bit_key_stable_oracle(scores[:3], k))


@torch.inference_mode()
def test_gfx12_qsa_topk_missing_binding_uses_python_fallback(device, monkeypatch):
    scores = _scores(device, 2, 1024)
    expected = _fp16_bit_key_stable_oracle(scores)
    actual = torch.empty_like(expected)
    monkeypatch.delattr(ext, "dsa_topk_gfx12")
    ext.dsa_topk(scores, actual, K)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_gfx12_qsa_topk_non_gfx12_uses_python_fallback(device, monkeypatch):
    scores = _scores(device, 2, 1024)
    expected = _fp16_bit_key_stable_oracle(scores)
    actual = torch.empty_like(expected)
    native_calls = {"count": 0}
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _index: type("Props", (), {"gcnArchName": "gfx1100", "warp_size": 32})(),
    )
    monkeypatch.setattr(
        ext, "dsa_topk_gfx12",
        lambda *_args: native_calls.__setitem__("count", native_calls["count"] + 1),
    )
    ext.dsa_topk(scores, actual, K)
    torch.cuda.synchronize(device)
    assert native_calls["count"] == 0
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_gfx12_qsa_topk_uses_the_nondefault_current_stream(device):
    source = _scores(device, 4, 1024)
    expected = _fp16_bit_key_stable_oracle(source)
    scores = torch.full_like(source, -float("inf"))
    actual = torch.full_like(expected, -2)
    consumed = torch.empty_like(actual)
    torch.cuda.synchronize(device)
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device = device)
    with torch.cuda.stream(stream):
        scores.copy_(source)  # Same-stream producer for the native kernel's input.
        ext.dsa_topk_gfx12(scores, actual)
        consumed.copy_(actual)  # Same-stream consumer must observe the native result.
        done = torch.cuda.Event()
        done.record()
    stream.synchronize()
    assert done.query()
    assert torch.equal(consumed, expected)


@torch.inference_mode()
def test_gfx12_qsa_topk_native_is_byte_identical_across_repeated_launches(device):
    source = _scores(device, 4, 1024)
    expected = _fp16_bit_key_stable_oracle(source)

    default_outputs = []
    for _ in range(4):
        actual = torch.empty_like(expected)
        ext.dsa_topk_gfx12(source, actual)
        default_outputs.append(actual.clone())
    torch.cuda.synchronize(device)
    for output in default_outputs:
        assert torch.equal(output, expected)
        assert torch.equal(output, default_outputs[0])

    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device = device)
    stream_outputs = []
    with torch.cuda.stream(stream):
        for _ in range(4):
            actual = torch.empty_like(expected)
            ext.dsa_topk_gfx12(source, actual)
            stream_outputs.append(actual.clone())
        done = torch.cuda.Event()
        done.record()
    stream.synchronize()
    assert done.query()
    for output in stream_outputs:
        assert torch.equal(output, expected)
        assert torch.equal(output, stream_outputs[0])
    assert torch.equal(default_outputs[0], stream_outputs[0])


@pytest.mark.parametrize("pools", [513, 8193])
@torch.inference_mode()
def test_gfx12_qsa_select_rows_matches_forced_fallback_and_expands_tail(device, monkeypatch, pools):
    indexer = QSAIndexer(
        config = None, key = "test.qsa", hidden_size = 16, n_heads = 2, kv_heads = 1,
        head_dim = 8, token_budget = K * 4, compress_ratio = 4, rms_norm_eps = 1e-6,
    )
    rows, pos0 = 5, 4 * pools - 5
    generator = torch.Generator(device = device).manual_seed(1201)
    q_rows = torch.randn((rows, 2, 8), generator = generator, device = device, dtype = torch.half)
    pool_flat = torch.randn((pools, 8), generator = generator, device = device, dtype = torch.half)
    fallback = torch.empty((rows, indexer.k_pad()), device = device, dtype = torch.int32)
    native = torch.empty_like(fallback)

    with _force_fallback(monkeypatch):
        indexer._select_rows(q_rows, pool_flat, pos0, pools, fallback)
    torch.cuda.synchronize(device)

    calls = {"native": 0}
    native_topk = ext.dsa_topk_gfx12

    def spy(*args):
        calls["native"] += 1
        return native_topk(*args)

    def unexpected_fallback(*_args, **_kwargs):
        pytest.fail("eligible QSA _select_rows unexpectedly used the fallback")

    monkeypatch.setattr(ext, "dsa_topk_gfx12", spy)
    monkeypatch.setattr(ext_fallbacks, "dsa_topk", unexpected_fallback)
    indexer._select_rows(q_rows, pool_flat, pos0, pools, native)
    torch.cuda.synchronize(device)

    assert calls["native"] == 1
    assert torch.equal(native, fallback)
    expanded = native[:, :K * indexer.compress_ratio].view(rows, K, indexer.compress_ratio)
    assert (expanded >= 0).all()
    assert torch.all(expanded[:, :, 1:] - expanded[:, :, :1] == torch.arange(
        1, indexer.compress_ratio, device = device, dtype = torch.int32).view(1, 1, -1),
    )
    for row in range(rows):
        tail = native[row, K * indexer.compress_ratio:]
        qpos = pos0 + row
        tail_start = ((qpos + 1) // indexer.compress_ratio) * indexer.compress_ratio
        tail_count = max(0, qpos - tail_start + 1)
        assert torch.equal(tail[:tail_count], torch.arange(
            tail_start, qpos + 1, device = device, dtype = torch.int32,
        ))
        assert (tail[tail_count:] == -1).all()
