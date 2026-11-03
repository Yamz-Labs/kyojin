"""Check wide-launch eligibility without a GPU."""
import ast
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module_tree():
    return ast.parse((ROOT / "exllamav3/modules/moe_fused.py").read_text())


def _shapes_constant():
    """The K6S8_WIDE_SHAPES assignment of moe_fused.py, evaluated without importing torch."""
    node = next(n for n in _module_tree().body
                if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "K6S8_WIDE_SHAPES" for t in n.targets))
    return ast.literal_eval(node.value)


K6S8_WIDE_SHAPES = _shapes_constant()


def policy():
    tree = _module_tree()
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoEHalf")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_wide_ok")
    state = {"variant": 0x160000, "timing": 0}
    ns = {"os": os, "MF9": state, "torch": NS(ops=NS(mf9=NS(half=True))), "K6S8_WIDE_SHAPES": K6S8_WIDE_SHAPES}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "wide-policy", "exec"), ns)
    return ns["_wide_ok"], state


def test_shape_constant_matches_the_cpp_structs():
    """K6S8_WIDE_SHAPES must list exactly the shapes that exl3_moe_fused_idx.h instantiates as K6S8."""
    header = (ROOT / "exllamav3/exllamav3_ext/quant/exl3_moe_fused_idx.h").read_text()

    def fields(name):
        m = re.search(r"struct %s\s*(?::\s*(\w+)\s*)?\{([^}]*)\}" % name, header)
        assert m, name
        vals = dict((k, int(v)) for k, v in re.findall(r"\b(\w+)\s*=\s*(\d+)", m.group(2)))
        if m.group(1):
            vals = {**fields(m.group(1)), **vals}
        return vals

    keys = ("D", "H", "LR", "NEXP", "TOPK", "INTER", "RB", "SB")
    want = {tuple(fields(n)[k] for k in keys) for n in ("QwenK6S8", "SmallK6S8")}
    assert want == set(K6S8_WIDE_SHAPES)


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_index_arithmetic_has_no_out_of_bounds_access(tmp_path):
    """Builds tests/moe_fused/moe8_bounds.cpp with ASan and UBSan in a temporary directory and runs it."""
    exe = tmp_path / "moe8-bounds"
    subprocess.run(["g++", "-O1", "-g", "-std=c++17", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    f"-I{ROOT / 'exllamav3/exllamav3_ext/quant'}", str(ROOT / "tests/moe_fused/moe8_bounds.cpp"),
                    "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


@pytest.mark.parametrize("shape", K6S8_WIDE_SHAPES)
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
