"""CPU tests of Linear.load_fp16 on synthetic vendor checkpoints (FP8 grids, MXFP4, NVFP4, BF16): the weight the
converter sees must equal the numpy reference dequant (foreign_quant.vendor_input) after the fp16 cast."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ml_dtypes")
HERE = os.path.dirname(os.path.abspath(__file__))


def _load_helpers():
    spec = importlib.util.spec_from_file_location("tvi_under_test", os.path.join(HERE, "test_vendor_input_cpu.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["tvi_under_test"] = m
    spec.loader.exec_module(m)
    return m


H = _load_helpers()
vi = H.vi

try:
    from exllamav3.loader.safetensors import SafetensorsCollection
    from exllamav3.modules.linear import Linear, apply_scale_grid
except Exception as e:                                           # extension not built on this host
    pytest.skip(f"exllamav3 not importable: {e}", allow_module_level=True)


def make_linear(tmp_path, tensors, cfg, key, out, in_):
    H.write_st(tmp_path / "model.safetensors", tensors)
    json.dump(cfg, open(tmp_path / "config.json", "w"))
    conf = types.SimpleNamespace(stc=SafetensorsCollection(str(tmp_path), load_method="python"), config_dict=cfg)
    lin = Linear(conf, key, in_, out, pad_to=1)
    lin.load(torch.device("cpu"))
    return lin.inner.weight                                          # (in, out) fp16


def check(w, ref):                                                   # fp16 cast of the float32 reference, bit equal
    exp = torch.from_numpy(ref).half().T
    assert w.shape == exp.shape and torch.equal(w.view(torch.int16), exp.contiguous().view(torch.int16))


@pytest.mark.parametrize("shape,grid,block", [((256, 384), (2, 3), [128, 128]), ((200, 300), (2, 3), [128, 128]), ((96, 128), (3, 4), None)])
@pytest.mark.parametrize("sname", ["weight_scale_inv", "weight_scale"])
def test_fp8_block_grids(tmp_path, shape, grid, block, sname):
    wb, s = H.e4m3_bytes(shape, 0, 0x60), H.f32_scale(grid, -3, 1)
    cfg = {"quantization_config": {"quant_method": "fp8"}}
    if block:
        cfg["quantization_config"]["weight_block_size"] = block
    T = {"l.weight": H.t("F8_E4M3", wb), f"l.{sname}": H.t("F32", s)}
    w = make_linear(tmp_path, T, cfg, "l", *shape)
    check(w, vi.dequant_fp8(wb, "F8_E4M3", s, tuple(block) if block else None))


def test_fp8_per_tensor_and_channel(tmp_path):
    wb = H.e4m3_bytes((64, 96), 0, 0x60)
    for i, s in enumerate((np.array(0.3, np.float32), H.f32_scale((64, 1), -3, 1))):
        d = tmp_path / str(i)
        d.mkdir()
        w = make_linear(d, {"l.weight": H.t("F8_E4M3", wb), "l.weight_scale": H.t("F32", s)}, {}, "l", 64, 96)
        check(w, vi.dequant_fp8(wb, "F8_E4M3", s))


def test_mxfp4_weight_packed_name(tmp_path):
    """AutoRound / compressed-tensors MXFP4: `weight_packed` + U8 `weight_scale` (found on a real Qwen3-0.6B MXFP4 checkpoint)."""
    nib = H.rng.integers(0, 16, size=(64, 128)).astype(np.uint8)
    sb = H.rng.integers(120, 132, size=(64, 4)).astype(np.uint8)
    w = make_linear(tmp_path, {"l.weight_packed": H.t("U8", H.pack_nibbles(nib)), "l.weight_scale": H.t("U8", sb)}, {}, "l", 64, 128)
    check(w, vi.dequant_mxfp4_linear(H.pack_nibbles(nib), vi.e8m0_to_f32(sb)))


def test_mxfp4_linear_u8_and_e8m0(tmp_path):
    nib = H.rng.integers(0, 16, size=(64, 128)).astype(np.uint8)
    sb = H.rng.integers(120, 132, size=(64, 4)).astype(np.uint8)
    ref = vi.dequant_mxfp4_linear(H.pack_nibbles(nib), vi.e8m0_to_f32(sb))
    for i, (wsn, tag) in enumerate((("weight_scale", "U8"), ("scale", "F8_E8M0"))):
        d = tmp_path / str(i)
        d.mkdir()
        w = make_linear(d, {"l.weight": H.t("I8", H.pack_nibbles(nib).view(np.int8)), f"l.{wsn}": H.t(tag, sb)}, {}, "l", 64, 128)
        check(w, ref)


@pytest.mark.parametrize("flavour", ["modelopt", "compressed-tensors"])
def test_nvfp4_both_conventions(tmp_path, flavour):
    nib, packed, sc, g = H.nvfp4_case(48, 128)
    if flavour == "modelopt":
        T = {"l.weight": H.t("U8", packed), "l.weight_scale": H.t("F8_E4M3", sc), "l.weight_scale_2": H.t("F32", np.array(g, np.float32))}
        ref = vi.dequant_nvfp4(packed, sc, g)
    else:
        gi = np.array([1.0 / g], np.float32)
        T = {"l.weight_packed": H.t("U8", packed), "l.weight_scale": H.t("F8_E4M3", sc), "l.weight_global_scale": H.t("F32", gi)}
        ref = vi.dequant_nvfp4(packed, sc, gi, inverse_global=True)
    T["l.bias"] = H.t("F32", np.arange(48, dtype=np.float32))
    H.write_st(tmp_path / "model.safetensors", T)
    json.dump({}, open(tmp_path / "config.json", "w"))
    conf = types.SimpleNamespace(stc=SafetensorsCollection(str(tmp_path), load_method="python"), config_dict={})
    lin = Linear(conf, "l", 128, 48, pad_to=1)
    lin.load(torch.device("cpu"))
    check(lin.inner.weight, ref)
    assert torch.equal(lin.inner.bias.float(), torch.arange(48, dtype=torch.float))


def test_bf16_dense(tmp_path):
    bf = torch.randn(32, 64).bfloat16()
    w = make_linear(tmp_path, {"l.weight": ("BF16", (32, 64), bf.view(torch.int16).numpy().view(np.uint16))}, {}, "l", 32, 64)
    assert torch.equal(w, bf.half().T)


@pytest.mark.parametrize("shape,grid,block", [((256, 384), (2, 3), None), ((200, 300), (2, 3), (128, 128)), ((7, 5), (), None)])
def test_torch_grid_equals_numpy_grid(shape, grid, block):
    w = torch.randn(*shape)
    s = torch.from_numpy(H.f32_scale(grid, -3, 1)) if grid else torch.tensor(0.7)
    got = apply_scale_grid(w, s, block).numpy()
    assert H.same(got, vi.apply_scales(w.numpy(), s.numpy(), block))


@pytest.mark.parametrize("interleaved", [False, True])
def test_gpt_oss_fused_mxfp4_experts(tmp_path, interleaved):
    E, N, G = 3, 16, 2                                              # gate_up_proj_blocks [E, 2I... N, G, 16] -> [E, N, G*32]
    nib = H.rng.integers(0, 16, size=(E, N, G, 32)).astype(np.uint8)
    blk, sb = H.pack_nibbles(nib), H.rng.integers(120, 132, size=(E, N, G)).astype(np.uint8)
    full = vi.dequant_mxfp4_blocks(blk, sb)                         # float32 [E, N, 64]
    T = {"m.gate_up_proj_blocks": H.t("U8", blk), "m.gate_up_proj_scales": H.t("U8", sb),
         "m.gate_up_proj_bias": H.t("F32", np.arange(E * N, dtype=np.float32).reshape(E, N))}
    H.write_st(tmp_path / "model.safetensors", T)
    json.dump({}, open(tmp_path / "config.json", "w"))
    conf = types.SimpleNamespace(stc=SafetensorsCollection(str(tmp_path), load_method="python"), config_dict={})
    I = N // 2
    for e, (lo, hi) in ((1, (0, I)), (2, (I, N))):
        lin = Linear(conf, "m.x", G * 32, I, fkey="m.gate_up_proj", fidx=e, frange=(lo, hi), finterleaved=interleaved, pad_to=1)
        lin.load(torch.device("cpu"))
        rows = full[e, 0::2] if interleaved and lo == 0 else full[e, 1::2] if interleaved else full[e, lo:hi]
        check(lin.inner.weight, np.ascontiguousarray(rows))


def test_mxfp8_u8_scale_inv(tmp_path):
    wb = H.e4m3_bytes((48, 128), 0, 0x70)
    sb = H.rng.integers(118, 134, size=(48, 4)).astype(np.uint8)
    cfg = {"quantization_config": {"quant_method": "mxfp8", "weight_block_size": [1, 32]}}
    w = make_linear(tmp_path, {"l.weight": H.t("F8_E4M3", wb), "l.weight_scale_inv": H.t("U8", sb)}, cfg, "l", 48, 128)
    check(w, vi.dequant_fp8(wb, "F8_E4M3", vi.e8m0_to_f32(sb), (1, 32)))


def test_fp8_per_channel_1d_scale_square_layer(tmp_path):
    """[out] scale on a square layer: must scale output channels (rows of the checkpoint matrix), not input columns."""
    wb, s = H.e4m3_bytes((64, 64), 0, 0x60), H.f32_scale((64,), -3, 1)
    w = make_linear(tmp_path, {"l.weight": H.t("F8_E4M3", wb), "l.weight_scale": H.t("F32", s)}, {}, "l", 64, 64)
    check(w, vi.dequant_fp8(wb, "F8_E4M3", s))


def test_fp8_with_e8m0_scale_grid_deepseek_v4(tmp_path):
    wb = H.e4m3_bytes((64, 96), 0, 0x70)
    sb = H.rng.integers(118, 134, size=(2, 3)).astype(np.uint8)
    w = make_linear(tmp_path, {"l.weight": H.t("F8_E4M3", wb), "l.scale": H.t("F8_E8M0", sb)}, {}, "l", 64, 96)
    check(w, vi.dequant_fp8(wb, "F8_E4M3", vi.e8m0_to_f32(sb)))
