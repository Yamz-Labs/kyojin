import pytest
import torch
from exllamav3.modules.block_sparse_mlp import (
    _HIP_PREFILL_MAX_EXPERT_ROWS,
    _moe_sync_free_count,
    _scatter_expert_count,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_scatter_matches_bincount_random():
    E = 512
    ids = torch.randint(0, E + 1, (5120,), device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)


def test_scatter_matches_bincount_sentinel_all():
    E = 512
    ids = torch.full((5120,), E, dtype=torch.long, device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)
    assert result[E].item() == 5120


def test_scatter_matches_bincount_empty_experts():
    E = 512
    ids = torch.randint(0, 8, (5120,), device=DEV)
    ids[:100] = torch.randint(200, 500, (100,), device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)
    assert (result > 0).sum().item() == (expected > 0).sum().item()
    assert (result == 0).sum().item() >= 100


def test_scatter_long_index():
    E = 512
    ids = torch.randint(0, E + 1, (_HIP_PREFILL_MAX_EXPERT_ROWS + 1,), device=DEV)
    assert ids.shape[0] > _HIP_PREFILL_MAX_EXPERT_ROWS
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)


def test_scatter_empty_index():
    E = 512
    ids = torch.empty((0,), dtype=torch.long, device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)
    assert result.dtype == torch.long
    assert result.shape == (E + 1,)


def test_scatter_dtype_and_shape():
    E = 512
    ids = torch.randint(0, E + 1, (1024,), device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    assert result.dtype == torch.long
    assert result.shape == (E + 1,)
    assert result.device == ids.device


def test_env_gate_default_on(monkeypatch):
    monkeypatch.delenv("EXL3_MOE_SYNC_FREE_COUNT", raising=False)
    assert _moe_sync_free_count() is True
    monkeypatch.setenv("EXL3_MOE_SYNC_FREE_COUNT", "0")
    assert _moe_sync_free_count() is False
    monkeypatch.setenv("EXL3_MOE_SYNC_FREE_COUNT", "1")
    assert _moe_sync_free_count() is True


def test_helpers_have_expected_signature():
    assert callable(_scatter_expert_count)
    ids = torch.tensor([0, 1, 2], dtype=torch.long, device=DEV)
    result = _scatter_expert_count(ids, 3)
    assert len(result.shape) == 1