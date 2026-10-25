from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from exllamav3.modules.attention_fn import ck_pagedkv


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
