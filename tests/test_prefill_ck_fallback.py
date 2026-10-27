import pytest
import torch

from exllamav3.modules.attention_fn import triton_paged as tp


@pytest.fixture
def no_lib(monkeypatch, tmp_path):
    monkeypatch.setenv("EXL3_PF_CK_SO", str(tmp_path / "absent" / "libckpf.so"))
    monkeypatch.setattr(tp, "_pf_ck_fn", None)
    monkeypatch.setattr(tp, "_pf_ck", True)


def _call():
    q = torch.zeros(1, 8, 24, 256, dtype = torch.half)
    kv = torch.zeros(1, 16, 4, 256, dtype = torch.half)
    return tp._prefill_ck_contig(q, kv, kv, 0, 4, 256, 0.0625, torch.zeros_like(q))


def test_auto_missing_library_falls_back_silently(monkeypatch, no_lib, capsys):
    monkeypatch.setattr(tp, "_pf_ck_mode", "auto")
    assert _call() is None          # caller keeps the Triton kernel
    assert tp._pf_ck is False       # not retried on every chunk
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_forced_missing_library_warns_and_falls_back(monkeypatch, no_lib, capsys):
    monkeypatch.setattr(tp, "_pf_ck_mode", "1")
    assert _call() is None
    assert "EXL3_PF_CK=1" in capsys.readouterr().out
