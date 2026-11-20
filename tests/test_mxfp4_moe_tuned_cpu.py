"""CPU checks for EXL3_MXFP4_MOE_TUNED (no GPU, no kernel launch)."""
import importlib.util
import os
import re
import sys

import pytest
import torch

triton = pytest.importorskip("triton")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load the foreign_quant package by path: importing exllamav3 would build/load the CUDA extension.
_dir = os.path.join(ROOT, "exllamav3", "foreign_quant")
_spec = importlib.util.spec_from_file_location("fq_mx", os.path.join(_dir, "__init__.py"), submodule_search_locations=[_dir])
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["fq_mx"] = _pkg
_spec.loader.exec_module(_pkg)
import fq_mx.triton_moe_mxfp4 as MX  # noqa: E402


def test_fused_branch_is_gated_and_else_is_unchanged():
    s = open(os.path.join(ROOT, "exllamav3", "modules", "block_sparse_mlp.py")).read()
    i = s.index("def _forward_mxfp4_grouped")
    body = s[i:s.index("def _forward_fp8_grouped", i)]
    m = re.search(r"        if \(MX\.tuned\(\) and R == 1 and self\.activation_fn == \"swiglu_oai\" and self\.act_limit > 0 and\n"
                  r"                gb is not None and ub is not None\):\n.*?\n        else:\n((?:            [^\n]*\n)+)", body, re.S)
    assert m, "tuned branch not found"
    assert "MX.gu_fused(" in m.group(0).split("        else:")[0]
    else_lines = [l.strip() for l in m.group(1).splitlines()]
    assert else_lines == [
        "g = MX.moe_gemv(h, gw, gb, sel, I, K, S).half()",
        "u = MX.moe_gemv(h, uw, ub, sel, I, K, S).half()",
        "a = torch.empty_like(g)",
        "self.activation_fn_call(g, u, a, self.act_limit)",
    ]


@pytest.fixture
def launches(monkeypatch):
    rec = []

    class L:
        def __getitem__(self, grid):
            def call(*a, **k):
                rec.append((grid, k["BN"], k["num_warps"]))
            return call

    monkeypatch.setattr(MX, "_mxfp4_moe", L())
    return rec


def _run(rec, monkeypatch, env, hip):
    monkeypatch.delenv("EXL3_MXFP4_MOE_TUNED", raising=False)
    if env is not None:
        monkeypatch.setenv("EXL3_MXFP4_MOE_TUNED", env)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    rec.clear()
    sel = torch.zeros(4, dtype=torch.int32)
    MX.moe_gemv(torch.zeros(1, 8), sel, None, sel, 2880, 2944, 4)
    return rec[0]


def test_flag_semantics(launches, monkeypatch):
    assert _run(launches, monkeypatch, None, None)[1:] == (MX.BN, MX.NW)      # default off, read at call time
    assert _run(launches, monkeypatch, "0", None)[1:] == (MX.BN, MX.NW)
    assert _run(launches, monkeypatch, "1", "6.1")[1:] == (MX.BN, MX.NW)      # HIP ignores the flag
    grid, bn, nw = _run(launches, monkeypatch, "1", None)
    assert (bn, nw) == (16, 4) and grid == (180, 4)
