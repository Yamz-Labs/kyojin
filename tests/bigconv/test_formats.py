"""bigconv: source-format detection (formats CLI, preflight check), adapter classification of vendor scale layouts,
and the shared reader, on small synthetic checkpoints with the vendor tensor names."""
import importlib.util
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("tvi_for_bigconv", os.path.join(ROOT, "tests", "test_vendor_input_cpu.py"))
H = importlib.util.module_from_spec(spec)
sys.modules["tvi_for_bigconv"] = H
try:
    spec.loader.exec_module(H)
except BaseException as e:                                     # torch / ml_dtypes missing on this host
    pytest.skip(f"vendor_input tests unavailable: {e}", allow_module_level=True)

from bigconv import formats, preflight                                  # noqa: E402
from bigconv.arch import TableAdapter                                   # noqa: E402

vi = H.vi


def ckpt(tmp_path, fmt):
    r = H.rng
    q = "model.layers.0.self_attn.q_proj"
    T = {"model.embed_tokens.weight": ("BF16", (16, 64), r.integers(0, 0x7F00, size=(16, 64)).astype(np.uint16)),
         "model.norm.weight": H.t("F32", np.ones(64, np.float32))}
    cfg = {}
    if fmt == "bf16":
        T[q + ".weight"] = ("BF16", (64, 128), r.integers(0, 0x7F00, size=(64, 128)).astype(np.uint16))
    elif fmt == "fp8_block":
        T[q + ".weight"] = H.t("F8_E4M3", H.e4m3_bytes((64, 128), 0, 0x60))
        T[q + ".weight_scale_inv"] = H.t("F32", H.f32_scale((2, 2)))
        cfg = {"quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 64]}}
    elif fmt == "fp8_channel":
        T[q + ".weight"] = H.t("F8_E4M3", H.e4m3_bytes((64, 128), 0, 0x60))
        T[q + ".weight_scale"] = H.t("F32", H.f32_scale((64, 1)))
    elif fmt == "mxfp4":
        T[q + ".weight"] = H.t("U8", H.pack_nibbles(r.integers(0, 16, size=(64, 128)).astype(np.uint8)))
        T[q + ".weight_scale"] = H.t("U8", r.integers(120, 132, size=(64, 4)).astype(np.uint8))
    elif fmt == "nvfp4":
        nib, packed, sc, g = H.nvfp4_case(64, 128)
        T[q + ".weight"] = H.t("U8", packed)
        T[q + ".weight_scale"] = H.t("F8_E4M3", sc)
        T[q + ".weight_scale_2"] = H.t("F32", np.array(g, np.float32))
        T[q + ".input_scale"] = H.t("F32", np.array(1.0, np.float32))
        cfg = {"quantization_config": {"quant_algo": "NVFP4"}}
    H.write_st(tmp_path / "model.safetensors", T)
    json.dump(cfg, open(tmp_path / "config.json", "w"))
    return str(tmp_path)


class Gen(TableAdapter):
    name = "gen_test"
    layer_re = re.compile(r"^model\.layers\.(\d+)\.")
    expert_re = re.compile(r"^(model\.layers\.(\d+))\.mlp\.experts\.(\d+)\.(w1)\.weight$")
    ROLES = [(re.compile(r"proj\.weight(_packed)?$"), "attn", "q", None)]


@pytest.mark.parametrize("fmt,primary", [("bf16", "dense_bf16"), ("fp8_block", "fp8_block"), ("fp8_channel", "fp8_channel"),
                                         ("mxfp4", "mxfp4"), ("nvfp4", "nvfp4")])
def test_detect_cli_and_preflight_report_the_format(tmp_path, fmt, primary):
    d = ckpt(tmp_path, fmt)
    info, cfg = formats.load_source(d)
    assert formats.detect(info, cfg)["primary"] == primary
    out = subprocess.run([sys.executable, "-m", "bigconv", "formats", "--src", d], capture_output=True, text=True, cwd=str(tmp_path),
                         env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "tools")})
    assert out.returncode == 0 and primary in out.stdout, out.stderr
    name, status, msg = formats.check_input_format(info, cfg)
    assert (name, status) == ("input-format", "ok") and primary in msg
    res = preflight.run(str(tmp_path / "work"), 1.0, src=d)
    assert [m for n, s, m in res if n == "input-format"][0].startswith(primary)


def test_tensor_map_input(tmp_path):
    tm = {"tensors": {"l.weight": {"dtype": "U8", "shape": [8, 32]}, "l.weight_scale": {"dtype": "F8_E4M3", "shape": [8, 4]},
                      "l.weight_scale_2": {"dtype": "F32", "shape": []}}}
    json.dump(tm, open(tmp_path / "m.json", "w"))
    info, cfg = formats.load_source(map_path=str(tmp_path / "m.json"))
    assert formats.detect(info, cfg)["primary"] == "nvfp4"


def test_unsupported_integer_quant_fails_preflight():
    info = {"l.qweight": ("I32", (8, 8)), "l.qzeros": ("I32", (1, 1)), "l.scales": ("F16", (1, 8)), "l.weight": ("U8", (8, 8))}
    assert formats.check_input_format(info)[1] == "FAIL"


@pytest.mark.parametrize("fmt,scale,in_f", [("fp8_block", ".weight_scale_inv", 128), ("fp8_channel", ".weight_scale", 128),
                                            ("mxfp4", ".weight_scale", 128), ("nvfp4", ".weight_scale", 128)])
def test_adapter_classifies_every_scale_layout(tmp_path, fmt, scale, in_f):
    d = ckpt(tmp_path, fmt)
    info, _ = formats.load_source(d)
    T = {n: {"dtype": t, "shape": list(s)} for n, (t, s) in info.items()}
    ad = Gen({})
    q = "model.layers.0.self_attn.q_proj"
    c = ad.classify(q + ".weight", T)
    assert c.action == "quant" and c.fmt == fmt and c.scale == q + scale and c.out_f == 64
    if fmt in ("nvfp4", "mxfp4"):
        assert c.in_f == in_f                                    # packed bytes x 2
    for n in T:
        if n.startswith(q) and n != q + ".weight":
            assert ad.classify(n, T).action == "skip", n          # scale, global scale and activation scale are consumed with the weight
    assert ad.classify("model.norm.weight", T).action == "copy"


@pytest.mark.parametrize("fmt", ["fp8_block", "mxfp4", "nvfp4", "bf16"])
def test_reader_streams_one_linear(tmp_path, fmt):
    d = ckpt(tmp_path, fmt)
    w = formats.read_linear(d, "model.layers.0.self_attn.q_proj")
    assert w.dtype == np.float32 and w.shape[0] == 64 and np.isfinite(w).all()
