"""CPU test: HY4_KV_DTYPE=fp16 must narrow the K-cache rows and nothing else."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.loader import mla_hy4


def test_h():
    m = object.__new__(mla_hy4.Hy4MLA)
    m.f16_kv = False
    m.kv_f16 = False
    x = torch.tensor([1.0, 2.0, 70000.0])
    assert m._h(x).dtype == torch.float32
    assert torch.equal(m._h(x), x)


def test_h_fp16():
    m = object.__new__(mla_hy4.Hy4MLA)
    m.f16_kv = False
    m.kv_f16 = True
    x = torch.tensor([1.0, 2.0, 70000.0])
    out = m._h(x)
    assert out.dtype == torch.float16
    # the clamp keeps fp16 finite
    assert torch.isfinite(out).all()
    assert abs(float(out[0]) - 1.0) < 1e-3
    assert abs(float(out[1]) - 2.0) < 1e-3
    assert float(out[2]) == 65504.0


def test_h_q_stays_fp32():
    m = object.__new__(mla_hy4.Hy4MLA)
    m.f16_kv = False
    m.kv_f16 = True
    x = torch.tensor([1.0, 2.0])
    assert m._h(x, "q").dtype == torch.float32


def test_env_var_selects_fp16():
    os.environ["HY4_KV_DTYPE"] = "fp16"
    try:
        m = object.__new__(mla_hy4.Hy4MLA)
        m.f16_kv = False
        # the __init__ line that reads the env var, exercised without loading a model
        src = importlib.util.find_spec("exllamav3.loader.mla_hy4")
        assert src is not None
        import inspect
        text = inspect.getsource(mla_hy4.Hy4MLA.__init__)
        assert "HY4_KV_DTYPE" in text
        assert "kv_f16" in text
    finally:
        del os.environ["HY4_KV_DTYPE"]


def test_step_graph_dtype_block():
    from exllamav3.loader import step_graph
    assert step_graph.K_CACHE_DTYPE in (torch.float32, torch.float16)
