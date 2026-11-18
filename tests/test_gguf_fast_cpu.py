"""CPU checks for EXL3_GGUF_FAST (exllamav3/foreign_quant/gguf_fast.py): flag resolution, explicit values win, EXL3_MOE_FOLD_FUSED excluded,
and every reader of the six flags goes through flag() (no direct os.environ read left). No GPU, no extension: gguf_fast is loaded by path."""
import importlib.util
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "exllamav3")
FAST = ("EXL3_FQ_MMVQ_LEGACY", "EXL3_MXFP4_MOE_TUNED", "EXL3_FQ_MMVQ_MXFP4", "EXL3_MXFP4_PREFILL_GROUPED", "EXL3_FQ_MMVQ_EPI", "EXL3_LEGACY_PREFILL_DQ")
READERS = ("foreign_quant/moe_mxfp4_prefill.py", "foreign_quant/triton_moe_mxfp4.py", "modules/foreign_linear.py")


def _gf():
    spec = importlib.util.spec_from_file_location("gguf_fast_t", os.path.join(PKG, "foreign_quant", "gguf_fast.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def clean(monkeypatch):
    for k in FAST + ("EXL3_GGUF_FAST", "EXL3_MOE_FOLD_FUSED", "SOME_OTHER_FLAG"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_default_off(clean):
    gf = _gf()
    assert gf.FAST_FLAGS == set(FAST)
    assert not any(gf.flag(k) for k in FAST)


def test_switch_turns_on_exactly_the_six(clean):
    gf = _gf()
    clean.setenv("EXL3_GGUF_FAST", "1")
    assert all(gf.flag(k) for k in FAST)
    assert not gf.flag("EXL3_MOE_FOLD_FUSED"), "fold stays opt-in"
    assert not gf.flag("SOME_OTHER_FLAG")


@pytest.mark.parametrize("fast", ["0", "", "2", "true"])
def test_switch_only_on_exactly_1(clean, fast):
    gf = _gf()
    clean.setenv("EXL3_GGUF_FAST", fast)
    assert not any(gf.flag(k) for k in FAST)


@pytest.mark.parametrize("name", FAST)
def test_explicit_zero_wins(clean, name):
    gf = _gf()
    clean.setenv("EXL3_GGUF_FAST", "1")
    clean.setenv(name, "0")
    assert not gf.flag(name)
    assert all(gf.flag(k) for k in FAST if k != name), "the other five stay on"


@pytest.mark.parametrize("name", FAST)
def test_explicit_one_without_switch(clean, name):
    gf = _gf()
    clean.setenv(name, "1")
    assert gf.flag(name)
    assert not any(gf.flag(k) for k in FAST if k != name)


def test_readers_use_flag():
    for rel in READERS:
        src = open(os.path.join(PKG, rel)).read()
        for k in FAST:
            if k in src:
                assert not re.search(r'environ\.get\(\s*"%s"' % k, src), (rel, k, "direct env read")
        assert "from .gguf_fast import flag" in src or "from ..foreign_quant.gguf_fast import flag" in src, rel
    n = sum(len(re.findall(r'flag\("(%s)"\)' % "|".join(FAST), open(os.path.join(PKG, r)).read())) for r in READERS)
    assert n >= 6, n
    fl = open(os.path.join(PKG, "modules/foreign_linear.py")).read()
    for k in ("EXL3_FQ_MMVQ_LEGACY", "EXL3_FQ_MMVQ_EPI", "EXL3_LEGACY_PREFILL_DQ"):
        assert 'flag("%s")' % k in fl, k
