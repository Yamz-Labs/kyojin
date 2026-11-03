"""CPU checks for the experimental K6/K8 extension; these are not GPU validation."""
import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUANT = ROOT / "exllamav3/exllamav3_ext/quant"
CPP = ROOT / "tests/moe_fused/k8_check.cpp"


def build_and_run(headers, output):
    subprocess.run([
        "g++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all", "-I", str(headers), str(CPP), "-o", str(output),
    ], check=True, capture_output=True, text=True)
    # LeakSanitizer cannot inspect this container's /proc; address/UB checks remain enabled.
    return subprocess.run([str(output)], capture_output=True, text=True,
                          env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"})


def test_k8_word_values_and_bounds(tmp_path):
    result = build_and_run(QUANT, tmp_path / "check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "K8 CHECK OK" in result.stdout


@pytest.mark.parametrize("old,new", [
    ("bits + 16; }", "bits + 17; }"),
    ("bits * g8 - 1 : tw(bits) - 1", "bits * g8 - 2 : tw(bits) - 1"),
    ("t + 4 * bits * g + bits", "t + 3 * bits * g + bits"),
])
def test_k8_oracle_rejects_broken_indexing(tmp_path, old, new):
    source = (QUANT / "exl3_moe_fused_idx.h").read_text()
    assert source.count(old) == 1
    (tmp_path / "exl3_moe_fused_idx.h").write_text(source.replace(old, new))
    result = build_and_run(tmp_path, tmp_path / "mutant")
    assert result.returncode != 0, "corrupted extraction escaped the check"


@pytest.fixture
def eligibility():
    # Isolate this pure policy helper without importing torch or compiling the HIP extension.
    tree = ast.parse((ROOT / "exllamav3/modules/moe_fused.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("_first_k", "_k6s8_support_reason")]
    assert len(functions) == 2
    state = {"variant": 0x160000, "timing": 0}
    namespace = {"os": os, "MF9": state}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "eligibility", "exec"), namespace)
    def experts(bits, count):
        return SimpleNamespace(**{
            key: [SimpleNamespace(inner=SimpleNamespace(K=bits, mul1=True, mcg=False))
                  for _ in range(count)] for key in ("gates", "ups", "downs")
        })
    model = experts(6, 3)
    model.shared_experts = experts(8, 1)
    return namespace["_k6s8_support_reason"], state, model


def test_disabled_without_explicit_opt_in(eligibility, monkeypatch):
    fn, _, model = eligibility
    monkeypatch.delenv("EXL3_MOE_FUSED_K6S8", raising=False)
    assert "requires EXL3_MOE_FUSED_K6S8=1" in fn(model)


def test_opt_in_and_unchanged_existing_pack(eligibility, monkeypatch):
    fn, _, model = eligibility
    monkeypatch.setenv("EXL3_MOE_FUSED_K6S8", "1")
    monkeypatch.delenv("EXL3_MOE_FUSED_LAYOUT", raising=False)
    assert fn(model) is None
    model.shared_experts.gates[0].inner.K = 6
    monkeypatch.delenv("EXL3_MOE_FUSED_K6S8")
    assert fn(model) is None


@pytest.mark.parametrize("part,projection,index,attribute,value", [
    ("routed", "ups", 2, "K", 5),
    ("routed", "downs", 1, "mul1", False),
    ("shared", "downs", 0, "K", 6),
    ("shared", "ups", 0, "mcg", True),
])
def test_rejects_mixed_projection_formats(eligibility, monkeypatch, part, projection, index, attribute, value):
    fn, _, model = eligibility
    monkeypatch.setenv("EXL3_MOE_FUSED_K6S8", "1")
    monkeypatch.delenv("EXL3_MOE_FUSED_LAYOUT", raising=False)
    module = model if part == "routed" else model.shared_experts
    setattr(getattr(module, projection)[index].inner, attribute, value)
    assert "uniform K6" in fn(model)


@pytest.mark.parametrize("break_part", ["routed_inner", "routed_k", "shared_inner", "shared_k", "shared_gates", "shared_missing"])
def test_block_without_k_falls_back(eligibility, monkeypatch, break_part):
    # A block whose projections have no `.inner.K` (for example a non-EXL3 linear) must fall back, not raise.
    fn, _, model = eligibility
    monkeypatch.setenv("EXL3_MOE_FUSED_K6S8", "1")
    monkeypatch.delenv("EXL3_MOE_FUSED_LAYOUT", raising=False)
    assert fn(model) is None
    if break_part == "routed_inner":
        model.gates[0] = SimpleNamespace()
    elif break_part == "routed_k":
        del model.gates[0].inner.K
    elif break_part == "shared_inner":
        model.shared_experts.gates[0] = SimpleNamespace()
    elif break_part == "shared_k":
        del model.shared_experts.gates[0].inner.K
    elif break_part == "shared_gates":
        model.shared_experts = SimpleNamespace()
    else:
        model.shared_experts = None
    assert fn(model) is None


def test_missing_attribute_in_a_later_projection_is_a_mismatch_not_an_error(eligibility, monkeypatch):
    fn, _, model = eligibility
    monkeypatch.setenv("EXL3_MOE_FUSED_K6S8", "1")
    monkeypatch.delenv("EXL3_MOE_FUSED_LAYOUT", raising=False)
    del model.downs[1].inner.mul1
    assert "uniform K6" in fn(model)
    model.downs[1].inner.mul1 = True
    model.shared_experts.ups[0] = SimpleNamespace()
    assert "uniform K6" in fn(model)


def test_rejects_repack_and_nondefault_launches(eligibility, monkeypatch):
    fn, state, model = eligibility
    monkeypatch.setenv("EXL3_MOE_FUSED_K6S8", "1")
    monkeypatch.setenv("EXL3_MOE_FUSED_LAYOUT", "repack")
    assert "native layout" in fn(model)
    monkeypatch.delenv("EXL3_MOE_FUSED_LAYOUT")
    for variant in (0, 0x60000, 0xe0000, 0x160100):
        state["variant"] = variant
        assert "six-stage" in fn(model)
    state.update(variant=0x160000, timing=1)
    assert "without timing" in fn(model)


def test_k8_decoder_is_wired_into_both_tile_paths():
    source = (QUANT / "exl3_moe_fused.cuh").read_text()
    body = source.split("void mf_dq8_k8(", 1)[1].split("\n}", 1)[0]
    assert "const uint32_t (&w)[9]" in body
    assert "kn_words(8, w, t, v);" in body
    assert "decode8<2>(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], f0, f1);" in body
    for name in ("exl3_moe_fused.cuh", "exl3_moe_fused7_mv.cuh"):
        text = (QUANT / name).read_text()
        assert "BITS == 8) mf_dq8_k8(wa[u], t, f0[t], f1[t]);" in text
        assert "w[5] = v1.x; w[6] = v1.y; w[7] = v1.z; w[8] = v1.w;" in text
