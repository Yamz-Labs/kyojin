"""A missing built kernel library stops hy4_forward with the install step; HY4_ALLOW_SLOW=1 keeps the slow path."""
import pytest

from exllamav3.loader import hy4_forward as hf


@pytest.fixture
def no_ext(monkeypatch, tmp_path):
    monkeypatch.setenv("HY4_EXT_DIR", str(tmp_path))
    monkeypatch.setattr(hf, "_EXT", [])
    return tmp_path


def test_missing_extension_stops(no_ext, monkeypatch):
    monkeypatch.delenv("HY4_ALLOW_SLOW", raising = False)
    with pytest.raises(RuntimeError, match = r"build\.sh.*install_strix\.sh"):
        hf._ext()
    assert hf._EXT == []                          # nothing cached: the next call stops again


def test_allow_slow_continues(no_ext, monkeypatch, capsys):
    monkeypatch.setenv("HY4_ALLOW_SLOW", "1")
    assert hf._ext() is None
    assert hf.mmvq_fn(12, 256) is None
    assert "slow path" in capsys.readouterr().out
