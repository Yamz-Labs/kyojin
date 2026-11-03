"""CPU tests for exllamav3.foreign_quant.vendor_input (converter INPUT formats: BF16, FP8, MXFP4, NVFP4).

The module is loaded by file path (numpy only). Synthetic checkpoints use the vendor tensor names and
dtypes and are written by a raw safetensors writer (numpy cannot hold F8 / BF16). Oracles are independent
of the module: torch float8 / bfloat16 casts and ml_dtypes codebooks (each skipped when missing), plus a
numpy encoder whose output must round-trip exactly.
"""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys

import numpy as np
import pytest

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exllamav3", "foreign_quant"))


def _load(name):
    mod = f"{name}_under_test_vi"
    if mod in sys.modules:
        return sys.modules[mod]
    spec = importlib.util.spec_from_file_location(mod, os.path.join(PATH, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


vi = _load("vendor_input")
nvfp4_ref = _load("nvfp4")
torch = pytest.importorskip("torch")
ml_dtypes = pytest.importorskip("ml_dtypes")

TORCH8 = {"F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
rng = np.random.default_rng(1234)


def same(a, b):
    """bit-exact float32 equality (signed zero distinguished; every NaN equals every NaN)."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    n = ~np.isnan(a)
    return np.array_equal(a[n].view(np.uint32), b[n].view(np.uint32))


# ---------------------------------------------------------------------------------------------- writer

def write_st(path, tensors):
    """tensors: {name: (tag, shape, bytes-like numpy array of the storage dtype)}"""
    hdr, off, blobs = {}, 0, []
    for k, (tag, shape, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        hdr[k] = {"dtype": tag, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def t(tag, a):
    return (tag, a.shape, a)


def e4m3_bytes(shape, lo=0, hi=0x7E):
    """random finite E4M3 bytes (sign random, magnitude byte in [lo, hi])."""
    return (rng.integers(lo, hi + 1, size=shape) | (rng.integers(0, 2, size=shape) << 7)).astype(np.uint8)


def f32_scale(shape, lo=-6, hi=3):
    return (rng.uniform(1.0, 2.0, size=shape) * 2.0 ** rng.integers(lo, hi, size=shape)).astype(np.float32)


# ---------------------------------------------------------------------------------------------- codebooks

@pytest.mark.parametrize("tag,torch_dt,lut", [("F8_E4M3", torch.float8_e4m3fn, vi.E4M3FN_LUT), ("F8_E5M2", torch.float8_e5m2, vi.E5M2_LUT)])
def test_fp8_codebook_vs_torch_and_ml_dtypes(tag, torch_dt, lut):
    b = np.arange(256, dtype=np.uint8)
    ref = torch.from_numpy(b).view(torch_dt).float().numpy()
    assert same(lut, ref)
    mld = {"F8_E4M3": ml_dtypes.float8_e4m3fn, "F8_E5M2": ml_dtypes.float8_e5m2}[tag]
    assert same(lut, b.view(mld).astype(np.float32))


def test_e8m0_codebook_vs_ml_dtypes():
    b = np.arange(256, dtype=np.uint8)
    assert same(vi.E8M0_LUT, b.view(ml_dtypes.float8_e8m0fnu).astype(np.float32))


def test_e2m1_codebook_vs_ml_dtypes_and_low_nibble_first():
    n = np.arange(16, dtype=np.uint8)
    ref = n.view(ml_dtypes.float4_e2m1fn).astype(np.float32) if hasattr(ml_dtypes, "float4_e2m1fn") else vi.E2M1_LUT
    assert same(vi.E2M1_LUT, ref)
    packed = np.array([[0x21, 0xF7]], dtype=np.uint8)               # low nibble first: 1,2 | 7,15
    assert vi.unpack_e2m1(packed).tolist() == [[0.5, 1.0, 6.0, -6.0]]
    assert vi.unpack_e2m1(packed.view(np.int8)).tolist() == [[0.5, 1.0, 6.0, -6.0]]


def test_bf16_helpers_vs_torch():
    x = torch.randn(4096) * 10
    x[:4] = torch.tensor([0.0, -0.0, float("inf"), 3.4e38])
    assert same(vi.bf16_to_f32(x.bfloat16().view(torch.int16).numpy().view(np.uint16)), x.bfloat16().float().numpy())
    bits = vi.f32_to_bf16_bits(x.numpy())
    assert np.array_equal(bits.view(np.int16), x.bfloat16().view(torch.int16).numpy())


# ---------------------------------------------------------------------------------------------- FP8

def fp8_ref(wb, tag, scale, bo, bi):
    """torch oracle: decode with torch float8 cast, expand the block grid with repeat_interleave, fp32 multiply."""
    w = torch.from_numpy(wb).view(TORCH8[tag]).float()
    s = torch.from_numpy(np.asarray(scale, np.float32))
    if s.numel() == 1:
        return (w * s.reshape(())).numpy()
    if s.ndim == 1:
        return (w * s[:, None]).numpy()
    s = s.repeat_interleave(bo, 0).repeat_interleave(bi, 1)[: w.shape[0], : w.shape[1]]
    return (w * s).numpy()


@pytest.mark.parametrize("out,in_,sr,sc,block", [
    (256, 384, 2, 3, None),           # DeepSeek / Qwen3-FP8 / GLM / MiniMax 128x128, divisible
    (200, 300, 2, 3, (128, 128)),     # ceil-div grid (Qwen3-FP8 with dims not a multiple of 128)
    (96, 128, 3, 4, None),            # 32x32 grid inferred from the shapes
    (128, 128, 128, 1, None),         # [out,1] channel layout as a grid
])
@pytest.mark.parametrize("tag", ["F8_E4M3", "F8_E5M2"])
def test_fp8_block_grid_bit_exact(out, in_, sr, sc, block, tag):
    wb = e4m3_bytes((out, in_), 0, 0x7B) if tag == "F8_E4M3" else e4m3_bytes((out, in_), 0, 0x7B)
    s = f32_scale((sr, sc))
    bo, bi = (block or (-(-out // sr), -(-in_ // sc)))
    got = vi.dequant_fp8(wb, tag, s, block)
    assert same(got, fp8_ref(wb, tag, s, bo, bi))


def test_fp8_per_tensor_and_per_channel_and_noscale():
    wb = e4m3_bytes((64, 96))
    for s, shape in ((np.float32(0.37), ()), (np.array([0.37], np.float32), (1,))):
        assert same(vi.dequant_fp8(wb, "F8_E4M3", s.reshape(shape)), fp8_ref(wb, "F8_E4M3", s, 1, 1))
    ch = f32_scale((64, 1))
    assert same(vi.dequant_fp8(wb, "F8_E4M3", ch), fp8_ref(wb, "F8_E4M3", ch[:, 0], 1, 1))
    assert same(vi.dequant_fp8(wb, "F8_E4M3", ch[:, 0]), fp8_ref(wb, "F8_E4M3", ch[:, 0], 1, 1))
    assert same(vi.dequant_fp8(wb, "F8_E4M3", None), torch.from_numpy(wb).view(torch.float8_e4m3fn).float().numpy())


def test_fp8_e8m0_scale_grid_deepseek_v4_style():
    wb = e4m3_bytes((64, 96), 0, 0x70)
    sb = rng.integers(118, 136, size=(2, 3)).astype(np.uint8)
    got = vi.dequant_fp8(wb, "F8_E4M3", vi.e8m0_to_f32(sb))
    w = torch.from_numpy(wb).view(torch.float8_e4m3fn).float()
    s = torch.from_numpy(sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)).repeat_interleave(32, 0).repeat_interleave(32, 1)
    assert same(got, (w * s).numpy())


def test_scale_grid_errors():
    with pytest.raises(ValueError):
        vi.apply_scales(np.zeros((100, 100), np.float32), np.ones((7, 7), np.float32))


# ---------------------------------------------------------------------------------------------- MXFP4

def encode_e2m1(vals):
    idx = np.abs(vi.E2M1_LUT[None, :8] - np.abs(vals)[..., None]).argmin(-1)
    return (idx | ((vals < 0) << 3)).astype(np.uint8)


def pack_nibbles(n):
    return (n[..., 0::2] | (n[..., 1::2] << 4)).astype(np.uint8)


def test_mxfp4_linear_roundtrip_and_oracle_int8_and_u8():
    out, in_ = 64, 256
    nib = rng.integers(0, 16, size=(out, in_)).astype(np.uint8)
    packed = pack_nibbles(nib)
    sb = rng.integers(117, 134, size=(out, in_ // 32)).astype(np.uint8)
    # round trip against a float64 numpy reference
    ref = vi.E2M1_LUT[nib].astype(np.float64) * np.repeat(2.0 ** (sb.astype(np.float64) - 127), 32, axis=1)
    for pk in (packed, packed.view(np.int8)):
        got = vi.dequant_mxfp4_linear(pk, vi.e8m0_to_f32(sb))
        assert same(got, ref.astype(np.float32))
    # ml_dtypes oracle for the nibbles and the scales
    if hasattr(ml_dtypes, "float4_e2m1fn"):
        q = nib.view(ml_dtypes.float4_e2m1fn).astype(np.float32)
        s = sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
        assert same(vi.dequant_mxfp4_linear(packed, vi.e8m0_to_f32(sb)), q * np.repeat(s, 32, axis=1))


def test_mxfp4_blocks_gpt_oss_layout():
    E, N, G = 3, 40, 4                                    # [E, N, G, 16] + [E, N, G] -> [E, N, G*32]
    nib = rng.integers(0, 16, size=(E, N, G, 32)).astype(np.uint8)
    blocks = pack_nibbles(nib)
    sb = rng.integers(120, 132, size=(E, N, G)).astype(np.uint8)
    got = vi.dequant_mxfp4_blocks(blocks, sb)
    ref = (vi.E2M1_LUT[nib].astype(np.float64) * (2.0 ** (sb.astype(np.float64) - 127))[..., None]).reshape(E, N, G * 32)
    assert got.shape == (E, N, G * 32) and same(got, ref.astype(np.float32))
    # same maths through the linear kernel, expert by expert
    lin = vi.dequant_mxfp4_linear(blocks[1].reshape(N, G * 16), vi.e8m0_to_f32(sb[1]))
    assert same(lin, got[1])


# ---------------------------------------------------------------------------------------------- NVFP4

def nvfp4_case(out=48, in_=128):
    nib = rng.integers(0, 16, size=(out, in_)).astype(np.uint8)
    sc = e4m3_bytes((out, in_ // 16), 0x08, 0x70) & 0x7F          # positive scales
    g = np.float32(rng.uniform(1e-4, 1e-2))
    return nib, pack_nibbles(nib), sc, g


def test_nvfp4_modelopt_vs_exact_reference_and_serving_module():
    nib, packed, sc, g = nvfp4_case()
    got = vi.dequant_nvfp4(packed, sc, g)
    ref = (vi.E2M1_LUT[nib].astype(np.float64) * np.repeat(np.asarray(sc).view(ml_dtypes.float8_e4m3fn).astype(np.float64), 16, axis=1)
           * np.float64(g)).astype(np.float32)
    assert same(got, ref)                                             # one rounding = the exact product rounded
    assert same(got, nvfp4_ref.dequant_nvfp4(packed, sc, float(g)))   # identical to the serving-side reference


def test_nvfp4_compressed_tensors_inverse_global():
    nib, packed, sc, g = nvfp4_case()
    ginv = np.float32(1.0 / g)
    got = vi.dequant_nvfp4(packed, sc, np.array([ginv]), inverse_global=True)
    ref = (vi.E2M1_LUT[nib].astype(np.float64) * np.repeat(sc.view(ml_dtypes.float8_e4m3fn).astype(np.float64), 16, axis=1) / np.float64(ginv)).astype(np.float32)
    assert same(got, ref)


# ---------------------------------------------------------------------------------------------- checkpoints

def make_ckpt(tmp_path, tensors, config=None, shards=1):
    d = tmp_path
    names = list(tensors)
    wm = {}
    for i in range(shards):
        part = {k: tensors[k] for k in names[i::shards]}
        fn = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        write_st(d / fn, part)
        wm.update({k: fn for k in part})
    if shards > 1:
        json.dump({"weight_map": wm}, open(d / "model.safetensors.index.json", "w"))
    if config is not None:
        json.dump(config, open(d / "config.json", "w"))
    return str(d)


def test_checkpoint_fp8_qwen_style_end_to_end(tmp_path):
    out, in_ = 200, 300
    wb, s = e4m3_bytes((out, in_), 0, 0x7B), f32_scale((2, 3))
    bf = torch.randn(8, 16).bfloat16()
    T = {"model.layers.0.self_attn.q_proj.weight": t("F8_E4M3", wb), "model.layers.0.self_attn.q_proj.weight_scale_inv": t("F32", s),
         "model.embed_tokens.weight": ("BF16", (8, 16), bf.view(torch.int16).numpy().view(np.uint16)),
         "model.norm.weight": t("F32", np.ones(16, np.float32))}
    cfg = {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}}
    ck = vi.Checkpoint(make_ckpt(tmp_path, T, cfg, shards=2))
    assert same(vi.load_linear(ck, "model.layers.0.self_attn.q_proj"), fp8_ref(wb, "F8_E4M3", s, 128, 128))
    assert same(vi.load_linear(ck, "model.embed_tokens"), bf.float().numpy())
    rep = vi.detect_checkpoint(ck.info(), ck.config)
    assert rep["primary"] == "fp8_block" and rep["block_size"] == (128, 128) and rep["counts"] == {"fp8_block": 1, "dense_bf16": 1}
    ck.close()


def test_checkpoint_deepseek_v4_mixed_fp8_and_fp4(tmp_path):
    wb, sb = e4m3_bytes((64, 96), 0, 0x70), rng.integers(118, 134, size=(2, 3)).astype(np.uint8)
    nib = rng.integers(0, 16, size=(32, 128)).astype(np.uint8)
    eb = rng.integers(118, 134, size=(32, 4)).astype(np.uint8)
    T = {"layers.0.attn.wq_a.weight": t("F8_E4M3", wb), "layers.0.attn.wq_a.scale": t("F8_E8M0", sb),
         "layers.0.ffn.experts.0.w1.weight": t("I8", pack_nibbles(nib).view(np.int8)), "layers.0.ffn.experts.0.w1.scale": t("F8_E8M0", eb)}
    ck = vi.Checkpoint(make_ckpt(tmp_path, T, {}))
    rep = vi.detect_checkpoint(ck.info(), ck.config)
    assert rep["primary"] == "mixed:fp8_block+mxfp4"
    a = vi.load_linear(ck, "layers.0.attn.wq_a")
    assert a.shape == (64, 96) and not np.isnan(a).any()
    e = vi.load_linear(ck, "layers.0.ffn.experts.0.w1")
    assert same(e, vi.dequant_mxfp4_linear(pack_nibbles(nib), vi.e8m0_to_f32(eb)))
    ck.close()


def test_checkpoint_mimo_u8_weight_scale_and_gpt_oss_fused(tmp_path):
    nib = rng.integers(0, 16, size=(32, 64)).astype(np.uint8)
    sb = rng.integers(118, 134, size=(32, 2)).astype(np.uint8)
    blk = pack_nibbles(rng.integers(0, 16, size=(2, 8, 3, 32)).astype(np.uint8))
    scl = rng.integers(118, 134, size=(2, 8, 3)).astype(np.uint8)
    T = {"m.experts.0.up.weight": t("U8", pack_nibbles(nib)), "m.experts.0.up.weight_scale": t("U8", sb),
         "m.mlp.experts.gate_up_proj_blocks": t("U8", blk), "m.mlp.experts.gate_up_proj_scales": t("U8", scl),
         "m.mlp.experts.gate_up_proj_bias": t("F32", np.zeros((2, 8), np.float32))}
    ck = vi.Checkpoint(make_ckpt(tmp_path, T, {}))
    rep = vi.detect_checkpoint(ck.info())
    assert rep["counts"] == {"mxfp4": 1, "mxfp4_blocks": 1}
    assert same(vi.load_linear(ck, "m.experts.0.up"), vi.dequant_mxfp4_linear(pack_nibbles(nib), vi.e8m0_to_f32(sb)))
    assert vi.load_linear(ck, "m.mlp.experts.gate_up_proj").shape == (2, 8, 96)
    ck.close()


def test_checkpoint_nvfp4_modelopt_and_compressed_tensors(tmp_path):
    nib, packed, sc, g = nvfp4_case(32, 64)
    T = {"a.q.weight": t("U8", packed), "a.q.weight_scale": t("F8_E4M3", sc), "a.q.weight_scale_2": t("F32", np.array(g, np.float32)),
         "a.q.input_scale": t("F32", np.array(1.0, np.float32)),
         "b.q.weight_packed": t("U8", packed), "b.q.weight_scale": t("F8_E4M3", sc),
         "b.q.weight_global_scale": t("F32", np.array([1.0 / g], np.float32))}
    ck = vi.Checkpoint(make_ckpt(tmp_path, T, {}))
    rep = vi.detect_checkpoint(ck.info())
    assert rep["primary"] == "nvfp4" and rep["counts"] == {"nvfp4": 2}
    assert same(vi.load_linear(ck, "a.q"), vi.dequant_nvfp4(packed, sc, g))
    assert same(vi.load_linear(ck, "b.q"), vi.dequant_nvfp4(packed, sc, np.float32(1.0 / g), inverse_global=True))
    assert np.allclose(vi.load_linear(ck, "a.q"), vi.load_linear(ck, "b.q"), rtol=1e-6)
    ck.close()


def test_unsupported_integer_quant_is_reported_not_guessed():
    info = {"l.qweight": ("I32", (8, 8)), "l.weight": ("U8", (8, 8)), "l.scales": ("F16", (1, 8))}
    rep = vi.detect_checkpoint(info)
    assert rep["primary"] == "unsupported-integer-quant" and rep["unsupported_integer_linears"] == 1


def test_truncated_file_is_rejected(tmp_path):
    p = tmp_path / "x.safetensors"
    write_st(p, {"a": t("F32", np.ones(4, np.float32))})
    p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises(ValueError):
        vi.SafetensorsFile(str(p))


def test_mxfp8_u8_weight_scale_inv_is_e8m0(tmp_path):
    """MiniMax-M3-MXFP8 / nvidia MiniMax-M3-NVFP4 attention: F8_E4M3 weight + U8 weight_scale_inv [out, in/32] (block [1,32])."""
    wb = e4m3_bytes((48, 128), 0, 0x70)
    sb = rng.integers(118, 134, size=(48, 4)).astype(np.uint8)
    T = {"a.q.weight": t("F8_E4M3", wb), "a.q.weight_scale_inv": t("U8", sb)}
    ck = vi.Checkpoint(make_ckpt(tmp_path, T, {"quantization_config": {"quant_method": "mxfp8", "weight_block_size": [1, 32]}}))
    assert vi.detect_checkpoint(ck.info(), ck.config)["block_size"] == (1, 32)
    w = torch.from_numpy(wb).view(torch.float8_e4m3fn).float()
    s = torch.from_numpy(sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)).repeat_interleave(32, 1)
    assert same(vi.load_linear(ck, "a.q"), (w * s).numpy())
    ck.close()
