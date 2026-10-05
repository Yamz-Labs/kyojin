"""CPU tests for the Qwen (qwen4_exp) runtime-ablation port: hidden 2560 kernel (non power of two), MTP skip,
arch table, spec export on mock directions, trunk-state capture. No model, no GPU."""
import os
os.environ["TRITON_INTERPRET"] = "1"  # kernel runs in the Triton CPU interpreter (set before triton is imported)
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "ablit"))
H = 2560


def load_hook(triton_interpret=False):
    if triton_interpret:
        os.environ["TRITON_INTERPRET"] = "1"
    s = importlib.util.spec_from_file_location("ablit_runtime_q", os.path.join(ROOT, "exllamav3/modules/ablit_runtime.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def dense(y, w, r):
    r = (r / r.norm()).float()
    return y.float() - w * torch.outer(y.float() @ r, r)


def test_block_size_is_power_of_two():
    hook = load_hook()
    assert hook._block(2560) == 4096 and hook._block(4096) == 4096 and hook._block(2049) == 4096 and hook._block(1) == 1


@pytest.mark.parametrize("dt", [torch.float32, torch.float16])
def test_triton_kernel_hidden_2560_matches_dense(dt):
    pytest.importorskip("triton")
    hook = load_hook(triton_interpret=True)
    if hook.triton is None:
        pytest.skip("no triton")
    g = torch.Generator().manual_seed(0)
    r = torch.randn(H, generator=g)
    r32 = (r / r.norm()).float()
    y = torch.randn(7, H, generator=g).to(dt)
    want = dense(y, 1.3, r)
    got = y.clone()
    hook._USE_TRITON = True
    hook.project(got, 1.3, r32, r32.half())
    assert torch.allclose(got.float(), want, atol=2e-3 if dt == torch.float16 else 1e-4, rtol=1e-3)
    # the masked lanes must not touch memory past the row: neighbouring rows of a bigger tensor stay equal
    big = torch.randn(3, H, generator=g).to(dt)
    ref = big.clone()
    hook.project(big[1:2], 0.0, r32, r32.half())
    assert torch.equal(big, ref)


def test_torch_path_hidden_2560():
    hook = load_hook()
    hook._USE_TRITON = False
    g = torch.Generator().manual_seed(1)
    r = torch.randn(H, generator=g)
    r32 = (r / r.norm()).float()
    y = torch.randn(5, H, generator=g)
    assert torch.allclose(hook.project(y.clone(), 0.9, r32, r32.half()), dense(y, 0.9, r), atol=1e-4)


def test_mtp_blocks_are_not_edited(tmp_path, monkeypatch):
    r = torch.randn(32, dtype=torch.float64)
    save_file({"r": r.contiguous()}, str(tmp_path / "s.safetensors"))
    (tmp_path / "s.json").write_text(json.dumps({"hidden": 32, "n_layers": 48, "attn_w": [0.5] * 48, "mlp_w": [0.5] * 48}))
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", str(tmp_path / "s.json"))
    hook = load_hook()
    hook._USE_TRITON = False
    def blk(key):
        return types.SimpleNamespace(key=key, attn=object(), mlp=object(), ablit=None, config=types.SimpleNamespace(directory=str(tmp_path)))
    for key, expect in (("model.language_model.layers.5", True), ("model.language_model.layers.47", True),
                        ("mtp.layers.0", False), ("model.mtp.layers.0", False)):
        b = blk(key)
        hook.prepare(b, "cpu")
        assert (b.ablit is not None) == expect, key


def test_arch_qwen_detect_and_chat():
    import arch
    d = os.path.expanduser("~/models/qwen38-yamz-v1")
    if os.path.isdir(d):
        assert arch.get("auto", d)["name"] == "qwen4_exp"
    a = arch.get("qwen4_exp")
    assert (a["n_layers"], a["hidden"], a["readout"]) == (48, 2560, "hc_mean")
    assert arch.chat_wrap("Hi", a).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert a["pca_layer"] == 31


def test_export_spec_on_mock_directions(tmp_path):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools/ablit/collect_dirs.py"), "--mock", "--mock-layers", "48",
                          "--mock-hidden", str(H), "--mock-n", "16", "--out", str(tmp_path / "dirs")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    k = {"attn": {"max_w": 2.0, "peak": 30.0, "min_w": 1.0, "width": 20.0}, "mlp": {"max_w": 1.0, "peak": 18.0, "min_w": 0.3, "width": 28.0}}
    (tmp_path / "k.json").write_text(json.dumps(k))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools/ablit/export_spec.py"), "--dirs", str(tmp_path / "dirs"),
                          "--arch", "qwen4_exp", "--kernel", str(tmp_path / "k.json"), "--scale-kernel", "--out", str(tmp_path / "spec")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    spec = json.load(open(tmp_path / "spec" / "edit_spec.json"))
    assert spec["hidden"] == H and spec["n_layers"] == 48 and len(spec["attn_w"]) == 48 and len(spec["mlp_w"]) == 48
    from safetensors.torch import load_file
    r = load_file(str(tmp_path / "spec" / "edit_spec.safetensors"))["r"]
    assert r.shape == (H,) and abs(float(r.norm()) - 1.0) < 1e-4


def test_capture_states_collects_48_blocks_with_hc_mean():
    import qwen_collect_dirs as Q
    class TransformerBlock:
        def __init__(self, k): self.k = k
        def prepare_for_device(self, s, p): return s
        def forward(self, s, p): return s + 1.0
    class Embed:
        def prepare_for_device(self, s, p): return s
        def forward(self, s, p): return s
    x = torch.zeros(1, 5, 4, 8)
    x[..., 0] = torch.tensor([0.0, 2.0, 4.0, 6.0])  # streams differ -> mean 3
    out = Q.capture_states([Embed()] + [TransformerBlock(i) for i in range(48)], x, {"export_states": [1]})
    assert len(out) == 48 and out[0].shape == (5, 8)
    assert torch.allclose(out[0][:, 0], torch.full((5,), 4.0)) and torch.allclose(out[47][:, 0], torch.full((5,), 51.0))
    with pytest.raises(AssertionError):
        Q.capture_states([TransformerBlock(0)], x, {})
