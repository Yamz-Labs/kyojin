"""CPU proof of the K6 instance of the fused MoE half-layer (word extraction vs the engine's dq4<6>, index bounds, alignment), with mutants that must fail,
plus a source guard that the kernels use the proven helpers. No GPU."""
import os, re, shutil, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QD = os.path.join(ROOT, "exllamav3", "exllamav3_ext", "quant")
CPP = os.path.join(ROOT, "tests", "moe_fused", "k6_check.cpp")

MUTANTS = {
    "rel_const": ("bits + 16; }", "bits + 17; }"),
    "s2_offset": ("(kn_rel(bits, t, g) + 3 * bits + 16); }", "(kn_rel(bits, t, g) + 3 * bits + 15); }"),
    "word_order": ("((unsigned long long) w[kn_i0(bits, t, g)] << 32) | (unsigned long long) w[kn_i2(bits, t, g)]", "((unsigned long long) w[kn_i2(bits, t, g)] << 32) | (unsigned long long) w[kn_i0(bits, t, g)]"),
    "i2_end": ("(kn_rel(bits, t, g) + 3 * bits + 15) / 32", "(kn_rel(bits, t, g) + 3 * bits + 16) / 32"),
    "wrap_index": ("bits * g8 - 1 : tw(bits) - 1", "bits * g8 - 2 : tw(bits) - 1"),
    "stride": ("t + 4 * bits * g + bits", "t + 3 * bits * g + bits"),
}

def _build(srcdir, out):
    r = subprocess.run(["g++", "-std=c++17", "-O1", "-I", srcdir, CPP, "-o", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return subprocess.run([out], capture_output=True, text=True)

def test_k6_words_match_engine(tmp_path):
    r = _build(QD, str(tmp_path / "chk"))
    assert r.returncode == 0 and "K6 CHECK OK" in r.stdout, r.stdout

@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_k6_mutants_fail(tmp_path, name):
    d = tmp_path / "src"; d.mkdir()
    txt = open(os.path.join(QD, "exl3_moe_fused_idx.h")).read()
    old, new = MUTANTS[name]
    assert old in txt, "mutant anchor missing: " + name
    open(d / "exl3_moe_fused_idx.h", "w").write(txt.replace(old, new, 1))
    r = _build(str(d), str(tmp_path / "chk"))
    assert r.returncode != 0 and "FAILED" in r.stdout, "mutant survived: " + name

def test_kernels_use_proven_helpers():
    for f in ("exl3_moe_fused.cuh", "exl3_moe_fused7_mv.cuh"):
        t = open(os.path.join(QD, f)).read()
        assert "BITS == 6) mf_dq8_k6(wa[u], t, f0[t], f1[t])" in t
        assert "const uint2 v0 = *(const uint2*) q, v1 = *(const uint2*) (q + 2), v2 = *(const uint2*) (q + 4); w[1] = v0.x; w[2] = v0.y; w[3] = v1.x; w[4] = v1.y; w[5] = v2.x; w[6] = v2.y;" in t
    t = open(os.path.join(QD, "exl3_moe_fused.cuh")).read()
    m = re.search(r"void mf_dq8_k6\(.*?\n\}\n", t, re.S)
    assert m and "kn_words(6, w, t, v);" in m.group(0) and "decode8<2>(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], f0, f1)" in m.group(0)
    assert "X9D(mf::QwenK6S6)" in open(os.path.join(QD, "exl3_moe_fused9.cu")).read()
    assert "X(mf::QwenK6S6)" in open(os.path.join(QD, "exl3_moe_fused.cu")).read()
