"""CPU test of the runtime direction-projection hook (exllamav3/modules/ablit_runtime.py): no model, no GPU.

The hook must equal the dense edit  y' = (I - w r r^T) y  for a unit direction r, and must leave blocks
past the spec's n_layers (e.g. MTP) untouched.
"""
import importlib.util
import json
import os
import types

import pytest
import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_hook():
    s = importlib.util.spec_from_file_location("ablit_runtime", os.path.join(ROOT, "exllamav3/modules/ablit_runtime.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m._USE_TRITON = False
    return m


@pytest.fixture
def spec_path(tmp_path, monkeypatch):
    h = 64
    g = torch.Generator().manual_seed(0)
    r = torch.randn(h, generator=g, dtype=torch.float64)
    save_file({"r": r.contiguous()}, str(tmp_path / "spec.safetensors"))
    spec = {"hidden": h, "n_layers": 2, "attn_w": [0.7, 0.0], "mlp_w": [0.5, 0.25]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", str(p))
    return str(p), r


def block(layer):
    return types.SimpleNamespace(key=f"model.layers.{layer}", attn=object(), mlp=object(), ablit=None)


def test_project_matches_dense_edit(spec_path):
    _, r = spec_path
    hook = load_hook()
    b = block(0)
    hook.prepare(b, "cpu")
    assert b.ablit is not None
    w_attn, w_mlp, r32, r16 = b.ablit
    assert (w_attn, w_mlp) == (0.7, 0.5)
    ru = (r / r.norm()).float()
    y = torch.randn(5, 64, generator=torch.Generator().manual_seed(1))
    want = y - w_attn * torch.outer(y @ ru, ru)
    got = hook.project(y.clone(), w_attn, r32, r16)
    assert torch.allclose(got, want, atol=1e-5)
    # half precision input goes through the fp16 direction
    yh = y.half()
    got_h = hook.project(yh.clone(), w_mlp, r32, r16)
    want_h = (yh.float() - w_mlp * torch.outer(yh.float() @ ru, ru))
    assert torch.allclose(got_h.float(), want_h, atol=2e-2)


def test_zero_weight_and_out_of_range_layers_are_untouched(spec_path):
    hook = load_hook()
    b = block(1)  # attn weight 0.0, mlp weight 0.25 -> active
    hook.prepare(b, "cpu")
    assert b.ablit[0] == 0.0 and b.ablit[1] == 0.25
    mtp = block(5)  # past n_layers
    hook.prepare(mtp, "cpu")
    assert mtp.ablit is None
    y = torch.randn(3, 64)
    assert torch.equal(hook.project(y.clone(), 0.0, b.ablit[2], b.ablit[3]), y)


def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv("EXL3_ABLIT_RUNTIME", raising=False)
    hook = load_hook()
    assert not hook.enabled()
    b = block(0)
    hook.prepare(b, "cpu")
    assert b.ablit is None
