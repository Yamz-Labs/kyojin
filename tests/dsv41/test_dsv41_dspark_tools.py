"""Cost model and calibration fit (pure numpy, no model)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "ds41"))
from dspark_cost import Cost
import dspark_calibrate as C


def test_cost_model_matches_the_facts():
    c = Cost()
    assert abs(1 / c.decode_s() - 25.0) < 0.1                      # 7.2 GB at 180 GB/s
    assert abs(c.routed_tok - 2.65) < 0.02
    assert c.verify_s(2) > c.decode_s() and c.round_s(5) > c.round_s(1)
    assert c.tokens_per_round(3, 1.0) == 4.0 and c.tokens_per_round(3, 0.0) == 1.0
    n, v = c.best(0.9)
    assert 3 <= n <= 6 and v > 25
    assert c.best(0.2)[0] == 0                                     # bad drafter: do not draft


def synth(rate_hi, rate_lo, n=3000, depth=10, seed=0):
    """Confidence separates good positions (match prob rate_hi per slot) from bad ones (rate_lo)."""
    rng = np.random.default_rng(seed)
    good = rng.random(n) < 0.5
    a = np.where(good, rate_hi, rate_lo)[:, None]
    match = rng.random((n, depth)) < a
    conf = np.where(good, 2.0, -2.0)[:, None] + rng.normal(0, 1.0, (n, depth))
    return [(np.arange(n), match, conf.astype(np.float32))]


def test_fit_prefers_a_confidence_stop_and_beats_plain():
    tr = synth(0.95, 0.15)
    cost = Cost(overlap=0.5)
    plain = 1 / cost.decode_s()
    no_stop = C.simulate(tr, cost, 5, None, None)
    stop = C.simulate(tr, cost, 5, 0.5, None)
    assert stop["tok_s"] > no_stop["tok_s"] > plain
    auto = C.simulate(tr, cost, 10, None, {"cost_per_position": 0.1, "decay": 0.9})
    assert auto["tok_s"] > plain
    bad = synth(0.05, 0.05)
    assert C.simulate(bad, cost, 5, 0.5, None)["tok_s"] > C.simulate(bad, cost, 5, None, None)["tok_s"]
