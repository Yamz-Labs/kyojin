"""CPU proofs for the 5 bpw routes: fused shapes K4S6 / K5S6 (index and size arithmetic), grouped GEMV staging bounds for K5 / K6, and source guards
that every dispatch carries the new bitrates and that unsupported ones fail loudly. No GPU."""
import os, re, subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QD = os.path.join(ROOT, "exllamav3", "exllamav3_ext", "quant")
CPP = os.path.join(ROOT, "tests", "moe_fused", "k56_check.cpp")

def _run(src, out, *flags):
    r = subprocess.run(["g++", "-std=c++17", "-O1", "-I", QD, *flags, src, "-o", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return subprocess.run([out], capture_output=True, text=True)

def _read(name):
    return open(os.path.join(QD, name)).read()

def test_k56_shapes_and_staging(tmp_path):
    r = _run(CPP, str(tmp_path / "chk"))
    assert r.returncode == 0 and "K56 CHECK OK" in r.stdout, r.stdout

def test_k56_check_fails_on_wrong_shared_bits(tmp_path):
    src = open(CPP).read().replace("DM::SHARED_U32 == 3L * (S::D / 16) * (S::INTER / 16) * 8 * S::SB", "DM::SHARED_U32 == 3L * (S::D / 16) * (S::INTER / 16) * 8 * S::RB")
    p = tmp_path / "mut.cpp"; p.write_text(src)
    assert _run(str(p), str(tmp_path / "mut")).returncode != 0

def test_new_shapes_are_instantiated_everywhere():
    for s in ("QwenK4S6", "QwenK5S6", "SmallK4S6", "SmallK5S6"):
        assert f"X(mf::{s})" in _read("exl3_moe_fused.cu")
        assert f"X9D(mf::{s})" in _read("exl3_moe_fused9.cu")
        assert f"XW(mf::{s})" in _read("exl3_moe_fused9.cu")

def test_grouped_decode_dispatch_covers_k5_k6_and_fails_loudly():
    t = _read("exl3_gemv.cu")
    m = re.search(r"static void launch_moe_grouped_k.*?\n\}\n", t, re.S).group(0)
    assert "case 10: launch_moe_grouped<5, false" in m and "case 12: launch_moe_grouped<6, false" in m
    assert "default: TORCH_CHECK(false" in m and "default: launch_moe_grouped<4" not in m
    assert "k2 == 10 || k2 == 12" in t

def test_wmma_prefill_dispatch_covers_k5_k6():
    t = _read("exl3_moe_prefill.cu")
    assert t.count("case 10: MPW2_K(5, false)") == 1 and t.count("case 12: MPW2_K(6, false)") == 1
    assert t.count("case 10: MPW_K(5, false)") == 1 and t.count("case 12: MPW_K(6, false)") == 1

def test_dense_decode_dispatch_covers_k7():
    t = _read("exl3_dec.cu")
    assert "kb2 <= 14" in t and "case 14: { constexpr int KB2 = 14;" in t and "case 14: gemv_kernel<14, CB>" in t
    # the routed-MoE launches must still reject K7
    assert "moe_gu_kernel<14" not in t and "moe_down_kernel<14" not in t

def test_python_gates():
    b = open(os.path.join(ROOT, "exllamav3", "modules", "block_sparse_mlp.py")).read()
    assert b.count("multi.K in (2, 2.5, 3, 4, 5, 6)") == 2
    e = open(os.path.join(ROOT, "exllamav3", "modules", "quant", "exl3.py")).read()
    assert "_DEC_KB2_DENSE = _DEC_KB2 + (14,)" in e
    assert re.search(r"def dec_supported.*?_DEC_KB2_DENSE", e, re.S)
    assert re.search(r"def dec_moe_supported.*?in _DEC_KB2 and", e, re.S)

def test_dense_k7_tile_windows_match_engine(tmp_path):
    r = _run(os.path.join(ROOT, "tests", "moe_fused", "dec_k7_check.cpp"), str(tmp_path / "dk7"))
    assert r.returncode == 0 and "DEC K7 CHECK OK" in r.stdout, r.stdout
