"""CPU test for the fp16 K-cache option.

Two things matter here and nothing else:
  * the "k" part of a cache row is stored narrow, and
  * the "q" part is not - only the cache is narrowed.

The env var is exercised in a subprocess, because `step_graph` reads it at import time and
reloading it in-process would leave a half-initialised module behind.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.loader import mla_hy4
from exllamav3.loader import step_graph


def _mla(kv_f16):
    m = object.__new__(mla_hy4.Hy4MLA)
    m.f16_kv = False
    m.kv_f16 = kv_f16
    return m


def test_cache_row_stays_fp32_by_default():
    x = torch.tensor([1.0, 2.0, 70000.0])
    out = _mla(False)._h(x)
    assert out.dtype == torch.float32
    assert torch.equal(out, x)


def test_cache_row_is_narrowed_and_clamped():
    x = torch.tensor([1.0, 2.0, 70000.0])
    out = _mla(True)._h(x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert abs(float(out[0]) - 1.0) < 1e-3
    assert abs(float(out[1]) - 2.0) < 1e-3
    assert float(out[2]) == 65504.0


def test_q_is_not_narrowed():
    x = torch.tensor([1.0, 2.0])
    assert _mla(True)._h(x, "q").dtype == torch.float32


def test_constructor_takes_the_flag():
    import inspect
    assert "kv_f16" in inspect.signature(mla_hy4.Hy4MLA.__init__).parameters


def _cache_dtype_in_a_child(env_value):
    """Run a child interpreter and report the dtype it picked for the K cache."""
    code = (
        "import torch\n"
        "from exllamav3.loader import step_graph\n"
        "print('fp16' if step_graph.K_CACHE_DTYPE == torch.float16 else 'fp32')\n"
    )
    env = dict(os.environ)
    env["HY4_KV_DTYPE"] = env_value
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-600:]
    return r.stdout.strip().splitlines()[-1]


def test_env_var_selects_the_cache_dtype():
    assert _cache_dtype_in_a_child("fp16") == "fp16"
    assert _cache_dtype_in_a_child("fp32") == "fp32"
    assert _cache_dtype_in_a_child("FP16") == "fp16"


def test_an_unknown_dtype_is_rejected():
    code = "from exllamav3.loader import step_graph\n"
    env = dict(os.environ)
    env["HY4_KV_DTYPE"] = "int8"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode != 0
    assert "HY4_KV_DTYPE must be fp32 or fp16" in r.stderr


def test_the_default_is_fp32():
    assert step_graph.K_CACHE_DTYPE == torch.float32
