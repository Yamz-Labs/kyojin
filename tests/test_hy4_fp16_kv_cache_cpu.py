"""CPU test for the fp16 K-cache option.

Two things matter here and nothing else:
  * the "k" part of a cache row is stored narrow, and
  * the "q" part is not - only the cache is narrowed.

The env var is exercised in a subprocess, because `step_graph` reads it at import time and
reloading it in-process would leave a half-initialised module behind. The child's path is
built from this process's `sys.path`, not from `PYTHONPATH`: in the nix build `PYTHONPATH` is
how torch and the built extension are found, and overwriting it leaves the child with a bare
interpreter.
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


def _child_env(env_value):
    """An environment for a child interpreter that can import the same packages as this one."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = [root]
    for p in sys.path:
        if p and p not in parts and os.path.isdir(p):
            parts.append(p)
    env = dict(os.environ)
    env["HY4_KV_DTYPE"] = env_value
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run_child(env_value, code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=_child_env(env_value))


def test_env_var_selects_the_cache_dtype():
    code = (
        "import torch\n"
        "from exllamav3.loader import step_graph\n"
        "print('fp16' if step_graph.K_CACHE_DTYPE == torch.float16 else 'fp32')\n"
    )
    for value in ("fp16", "fp32", "FP16"):
        r = _run_child(value, code)
        assert r.returncode == 0, r.stderr[-600:]
        assert r.stdout.strip().splitlines()[-1] == ("fp16" if value.lower() == "fp16" else "fp32")


def test_an_unknown_dtype_is_rejected():
    r = _run_child("int8", "from exllamav3.loader import step_graph\n")
    assert r.returncode != 0
    assert "HY4_KV_DTYPE must be fp32 or fp16" in r.stderr


def test_the_default_is_fp32():
    assert step_graph.K_CACHE_DTYPE == torch.float32
