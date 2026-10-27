from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

# Load ck_pagedkv.py on its own: importing the exllamav3 package would pull in the native extension (and its
# toolchain), which these tests do not need.
import importlib.util
import sys
import types

_pkg = types.ModuleType("_ckpkg")
_pkg.__path__ = []
_common = types.ModuleType("_ckpkg.common")
_common.AttnArgs = object
sys.modules.update({"_ckpkg": _pkg, "_ckpkg.common": _common})
_spec = importlib.util.spec_from_file_location(
    "_ckpkg.ck_pagedkv", Path(__file__).resolve().parents[1] / "exllamav3/modules/attention_fn/ck_pagedkv.py")
ck_pagedkv = importlib.util.module_from_spec(_spec)
sys.modules["_ckpkg.ck_pagedkv"] = ck_pagedkv
_spec.loader.exec_module(ck_pagedkv)


def _args(q_len=16):
    return SimpleNamespace(
        q=SimpleNamespace(device=torch.device("cuda")), q_len=q_len, kv_len=q_len,
        is_varlen=lambda: False, is_swa=lambda: False, has_kv_cache=lambda: True,
        cu_seqlens=None, softcap=0.0, sinks=None, causal=True, dim=192,
        num_q_heads=64, num_kv_heads=4, bsz=1, v_dim=128,
    )


def test_ck_library_path_uses_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("EXL3_CK_PAGEDKV_SO", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert ck_pagedkv.ck_library_path() == tmp_path / "kyojin/ck-pagedkv/libck_pagedkv.so"


def test_ck_library_path_accepts_explicit_override(monkeypatch, tmp_path):
    library = tmp_path / "custom.so"
    monkeypatch.setenv("EXL3_CK_PAGEDKV_SO", str(library))
    assert ck_pagedkv.ck_library_path() == library


def test_ownership_declines_decode(monkeypatch):
    monkeypatch.setattr(ck_pagedkv, "_supported_device", lambda _: True)
    assert not ck_pagedkv.owns(_args(q_len=1))
    assert ck_pagedkv.owns(_args())


def _reset(monkeypatch, so):
    monkeypatch.setenv("EXL3_CK_PAGEDKV_SO", str(so))
    monkeypatch.setattr(ck_pagedkv, "_fn", None)
    monkeypatch.setattr(ck_pagedkv, "_unavailable", None)


def test_missing_library_declines_instead_of_raising(monkeypatch, tmp_path, capsys):
    _reset(monkeypatch, tmp_path / "absent.so")
    monkeypatch.setattr(ck_pagedkv, "_supported_device", lambda _: True)
    assert ck_pagedkv.kernel_or_none() is None
    assert "disabled" in capsys.readouterr().out
    assert ck_pagedkv.fn_ck_paged_attn(_args()) is None
    assert ck_pagedkv.kernel_or_none() is None
    assert capsys.readouterr().out == ""   # one warning only


def test_unloadable_library_declines(monkeypatch, tmp_path):
    bad = tmp_path / "bad.so"
    bad.write_bytes(b"not an elf")
    _reset(monkeypatch, bad)
    assert ck_pagedkv.kernel_or_none() is None
