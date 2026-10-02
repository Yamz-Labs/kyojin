"""Optional real-model K5 coverage for Qwen3.8-Flash-Next shared experts."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from hip_flash_quant import exl3_expert_quant
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext

_DEFAULT_MODEL = Path("~/Models/Qwen3.8-Flash-Next-exl3").expanduser()
_MODEL = Path(os.environ.get("EXL3_FLASH_TEST_MODEL", _DEFAULT_MODEL))
_DEVICE_INDICES = list(range(torch.cuda.device_count())) or [0]
_SHARED_EXPERT_KEYS = [
    "model.language_model.layers.0.mlp.shared_expert.gate_proj",
    "model.language_model.layers.0.mlp.shared_expert.up_proj",
    "model.language_model.layers.0.mlp.shared_expert.down_proj",
]


def _require_gfx12(device_index):
    if not (torch.version.hip and torch.cuda.is_available()):
        pytest.skip("ROCm build / device not available")
    arch = getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "")
    if arch.split(":", 1)[0] not in ("gfx1200", "gfx1201"):
        pytest.skip(f"K5 real-model oracle requires gfx1200/gfx1201, got {arch or 'unknown'}")
    assert hasattr(ext, "exl3_gemv"), \
        "gfx12 target build is missing the required ext.exl3_gemv binding"
    assert hasattr(ext, "exl3_gemv_supported"), \
        "gfx12 target build is missing the required ext.exl3_gemv_supported binding"
    assert ext.exl3_gemv_supported(device_index), \
        f"ext.exl3_gemv_supported rejected gfx12 device {device_index} ({arch})"


@pytest.fixture(scope="module")
def flash_model():
    if not (torch.version.hip and torch.cuda.is_available()):
        pytest.skip("ROCm build / device not available")
    if not _MODEL.is_dir():
        pytest.skip(f"Test model not found: {_MODEL} (set EXL3_FLASH_TEST_MODEL)")
    _codebook, bits = exl3_expert_quant(_MODEL)
    if bits["shared"] != 5:
        pytest.skip(
            f"K5 route needs K5 shared experts; {_MODEL} has K{bits['shared']} "
            "(set EXL3_FLASH_TEST_MODEL)"
        )
    return Model.from_config(Config.from_directory(str(_MODEL)))


def _spy_k5_route():
    calls = {"gemv": 0, "k5_gemv": 0, "reconstruct": 0,
             "reconstruct_slice": 0, "reconstruct_had_slice": 0, "hgemm": 0}
    originals = {
        "gemv": ext.exl3_gemv,
        "reconstruct": ext.reconstruct,
        "reconstruct_slice": getattr(ext, "reconstruct_slice", None),
        "reconstruct_had_slice": getattr(ext, "reconstruct_had_slice", None),
        "hgemm": ext.hgemm,
    }

    def gemv_spy(*args, **kwargs):
        calls["gemv"] += 1
        if args[1].shape[-1] // 16 == 5:
            calls["k5_gemv"] += 1
        return originals["gemv"](*args, **kwargs)

    def wrap(name):
        def spy(*args, **kwargs):
            calls[name] += 1
            return originals[name](*args, **kwargs)
        return spy

    ext.exl3_gemv = gemv_spy
    ext.reconstruct = wrap("reconstruct")
    if originals["reconstruct_slice"] is not None:
        ext.reconstruct_slice = wrap("reconstruct_slice")
    if originals["reconstruct_had_slice"] is not None:
        ext.reconstruct_had_slice = wrap("reconstruct_had_slice")
    ext.hgemm = wrap("hgemm")

    def restore():
        ext.exl3_gemv = originals["gemv"]
        ext.reconstruct = originals["reconstruct"]
        if originals["reconstruct_slice"] is not None:
            ext.reconstruct_slice = originals["reconstruct_slice"]
        if originals["reconstruct_had_slice"] is not None:
            ext.reconstruct_had_slice = originals["reconstruct_had_slice"]
        ext.hgemm = originals["hgemm"]

    return calls, restore


@pytest.mark.parametrize("device_index", _DEVICE_INDICES)
@torch.inference_mode()
def test_flash_shared_expert_k5_modules_match_reference_on_each_gfx12(flash_model, device_index):
    """Real gate/up/down K5 modules take the direct route and match reconstruction."""
    _require_gfx12(device_index)
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device_index):
        for key in _SHARED_EXPERT_KEYS:
            linear = flash_model.find_module(key)
            try:
                linear.load(device=device)
                assert linear.inner.K == 5, f"{key} is K{linear.inner.K}, expected K5"
                assert linear.inner.mul1 and not linear.inner.mcg
                torch.manual_seed(1201)
                x = torch.randn((1, 1, linear.in_features), dtype=torch.float16,
                                device=device) * 1e-3
                expected = linear.forward(x, {"reconstruct": True})

                calls, restore = _spy_k5_route()
                try:
                    actual = linear.forward(x, {"reconstruct": False})
                finally:
                    restore()

                assert calls["gemv"] == calls["k5_gemv"] == 1, \
                    f"{key} on device {device_index} did not make exactly one K5 GEMV call: {calls}"
                assert calls["reconstruct"] == 0
                assert calls["reconstruct_slice"] == 0
                assert calls["reconstruct_had_slice"] == 0
                assert calls["hgemm"] == 0
                assert torch.isfinite(actual).all()
                assert torch.isfinite(expected).all()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)
            finally:
                linear.unload()
        torch.cuda.empty_cache()
