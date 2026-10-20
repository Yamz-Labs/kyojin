"""Check wide-launch eligibility without a GPU."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]


def policy():
    tree = ast.parse((ROOT / "exllamav3/modules/moe_fused.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoEHalf")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_wide_ok")
    state = {"variant": 0x160000, "timing": 0}
    ns = {"os": os, "MF9": state, "torch": NS(ops=NS(mf9=NS(half=True)))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "wide-policy", "exec"), ns)
    return ns["_wide_ok"], state


@pytest.mark.parametrize("shape", [(2560, 4, 320, 512, 10, 640, 6, 8),
                                    (256, 4, 32, 64, 4, 128, 6, 8)])
def test_wide_k6k8_requires_supported_settings(monkeypatch, shape):
    fn, state = policy()
    obj = NS(native=True, RB=6, SB=8, shape=shape)
    monkeypatch.delenv("EXL3_MOE_R8_K6S8", raising=False)
    assert not fn(obj)
    monkeypatch.setenv("EXL3_MOE_R8_K6S8", "1")
    for variant in (0x160000, 0x161000):
        state["variant"] = variant
        assert fn(obj)
    for variant in (0, 0x60000, 0x160100):
        state["variant"] = variant
        assert not fn(obj)
    state.update(variant=0x160000, timing=1)
    assert not fn(obj)
    state["timing"] = 0
    obj.native = False
    assert not fn(obj)
    obj.native = True
    obj.shape = (512, 4, 32, 64, 4, 128, 6, 8)
    assert not fn(obj)


def test_existing_low_bit_path_needs_no_new_setting(monkeypatch):
    fn, _ = policy()
    monkeypatch.delenv("EXL3_MOE_R8_K6S8", raising=False)
    assert fn(NS(native=True, RB=4, SB=5, shape=()))
