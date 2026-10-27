"""Sanity of tools/tier/model.py (pure python/numpy): sizes, monotonicity, the hotset greedy rule."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "tier"))
import numpy as np
import model as M


def test_expert_bytes():
    # K=2: 3 x 5120 x 2304 x 2/8 B of trellis + the fp16 vectors
    assert abs(M.expert_bytes(2) - 3 * 5120 * 2304 * 0.25) < 1e5
    assert M.expert_bytes(3) > M.expert_bytes(2) * 1.4


def test_zipf_monotone():
    h = [M.zipf_hit(b, 1.0)[0] for b in (60, 80, 100, 118)]
    assert h == sorted(h) and 0.8 < h[0] < h[-1] < 1.0
    assert M.zipf_hit(100, 0.0)[0] < M.zipf_hit(100, 1.2)[0]
    assert abs(M.zipf_q(1.0).sum() - M.TOPK) < 1e-6


def test_decode_model():
    full, res = M.decode_tok_s(1.0)
    assert abs(full - res) < 1e-9
    lo = M.decode_tok_s(0.9)[0]
    hi = M.decode_tok_s(0.99)[0]
    assert lo < hi < res
    # perfect token-level prefetch needs a lower hit rate than demand reads only
    assert M.hit_needed(0.9, "plain", "ds41split", 1.0, 40) < M.hit_needed(0.9, "plain", "ds41split", 0.0)


def test_hotset_rule():
    rng = np.random.default_rng(0)
    cal = {f"layers.{l}.ffn": (rng.random(384) ** 4 * 1000).astype(int).tolist() for l in range(40)}
    picked, hc, he = M.hotset(cal, cal, 60.0)
    n = sum(len(v) for v in picked.values())
    assert 0 < n < 40 * 384 and hc == he and 0.5 < hc < 1.0
    _, hc2, _ = M.hotset(cal, None, 120.0)
    assert hc2 > hc
