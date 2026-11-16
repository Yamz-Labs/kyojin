"""CPU tests for the split_eval quality helpers.

Task scoring and KLD/PPL math must be checked without loading a model: they are the
numbers Stage 2 uses to judge split quality when token-for-token equality is not
expected on a mixed-architecture pair.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope = "module")
def ev():
    path = ROOT / "tools" / "strix_halo" / "split_eval.py"
    spec = importlib.util.spec_from_file_location("split_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["split_eval"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("split_eval", None)


@pytest.mark.parametrize("index,text,passed", [
    (0, "The capital of France is Paris.", True),
    (0, "I am not sure.", False),
    (1, "def reverse(head):\n    nxt = head.next\n    return nxt", True),
    (1, "Use a loop.", False),
    (2, "Rayleigh scattering makes the sky blue.", True),
    (3, "53, 59, 61, 67, 71", True),
    (3, "51, 55, 57", False),
    (3, "153, 253, 353", False),
    (4, "The trip takes 2 hours 45 minutes.", True),
    (4, "2 h 45 min", True),
    (4, "two hours and forty-five minutes", True),
    (4, "The trip takes about 3 hours.", False),
    (4, "The trip takes 3 hours and 45 minutes, arriving at 17:20.", False),
    (5, "Romeo and Juliet are in Verona.", True),
])
def test_score_task(ev, index, text, passed):
    assert ev.score_task(index, text)["passed"] is passed


def test_acceptance_ci95_bounds(ev):
    ci = ev.acceptance_ci95(50, 50)
    assert ci is not None
    assert ci[0] < 50.0 < ci[1]
    assert ev.acceptance_ci95(0, 0) is None


def test_kld_stats_identical_is_zero(ev):
    lp = torch.log_softmax(torch.randn(8, 16), -1)
    st = ev.kld_stats(lp, lp)
    assert st["positions"] == 8
    assert st["kld_base_to_split"] == pytest.approx(0.0, abs = 1e-6)
    assert st["kld_split_to_base"] == pytest.approx(0.0, abs = 1e-6)
    assert st["top1_agreement_pct"] == pytest.approx(100.0)


def test_kld_stats_detects_divergence(ev):
    a = torch.log_softmax(torch.randn(8, 16), -1)
    b = torch.log_softmax(torch.randn(8, 16), -1)
    st = ev.kld_stats(a, b)
    assert st["kld_base_to_split"] > 0.0
    assert st["kld_split_to_base"] > 0.0
    assert 0.0 <= st["top1_agreement_pct"] < 100.0


@pytest.fixture(scope = "module")
def sessions_smoke():
    pytest.importorskip("aiohttp")
    path = ROOT / "tools" / "strix_halo" / "split_sessions_smoke.py"
    spec = importlib.util.spec_from_file_location("split_sessions_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["split_sessions_smoke"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("split_sessions_smoke", None)


def test_parse_metrics(sessions_smoke):
    body = (
        "# HELP llamacpp:tokens_predicted_total Total tokens generated\n"
        "# TYPE llamacpp:tokens_predicted_total counter\n"
        "llamacpp:tokens_predicted_total 456\n"
        "llamacpp:prompt_seconds_total 2.794384\n"
        "llamacpp:tokens_predicted_seconds_total 65.976150\n"
        "llamacpp:kv_cache_usage_ratio 0\n"
        "not a sample line\n"
    )
    m = sessions_smoke.parse_metrics(body)
    assert m["llamacpp:tokens_predicted_total"] == 456.0
    assert m["llamacpp:prompt_seconds_total"] == pytest.approx(2.794384)
    assert m["llamacpp:tokens_predicted_seconds_total"] == pytest.approx(65.976150)
    assert m["llamacpp:kv_cache_usage_ratio"] == 0.0


def test_kld_compare_roundtrip(ev, tmp_path, capsys):
    a = torch.log_softmax(torch.randn(6, 12), -1).half()
    b = torch.log_softmax(torch.randn(6, 12), -1).half()
    common = dict(model = "m", corpus = "c", slices = 1, len = 6, scored = [5], nll = [2.0])
    torch.save({**common, "mode": "baseline", "ids": [[1, 2, 3, 4, 5, 6]],
                "logprobs": [a]}, tmp_path / "a.kld")
    torch.save({**common, "mode": "split", "ids": [[1, 2, 3, 4, 5, 6]],
                "logprobs": [b]}, tmp_path / "b.kld")
    ev.kld_compare(str(tmp_path / "a.kld"), str(tmp_path / "b.kld"))
    out = capsys.readouterr().out
    assert "KLD mean" in out
    assert "top1 agreement" in out
    assert "PPL" in out
