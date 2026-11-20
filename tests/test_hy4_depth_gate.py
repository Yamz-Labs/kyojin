"""CPU test of the HY4_SPEC2_GATE online depth gate (S7 arm O): synthetic acceptance stream into DepthGate, no GPU, no torch, no model.
Asserts the depth decisions, the per-round depth histogram and the 'rounds where the gate changed the depth' counter (0 = the arm did not fire)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "hy4"))
from depth_gate import DepthGate, MODE_DEFAULT  # noqa: E402

ROW = {1: 61.8, 2: 76.7, 3: 90.3, 4: 104.0, 5: 117.3, 6: 130.6, 7: 143.8, 8: 157.0}
HI, LO = [0.99, 0.99], [0.3, 0.3]           # base confidences: bin 9 (s = 0.98) and bin 0 (s = 0.09)


def _gate(mode="learn", **kw):
    return DepthGate(ROW, 3.0, 0.3, 2, 1, mode=mode, **kw)


def run(g, stream):
    """stream: list of (confs, ext_accepted). Mimics the engine: decide, then update with the accepted drafts of the round."""
    depths = []
    for confs, ext_ok in stream:
        go, b = g.decide(confs)
        depths.append(2 + go)
        acc = (3 if ext_ok else 2) if go else 2          # base chain always right here; the extension only counts when it ran
        g.update(b, go, acc)
    return depths


def test_arm_o_is_the_default_mode():
    assert MODE_DEFAULT == "learn"
    os.environ.pop("HY4_SPEC2_MODE", None)
    assert DepthGate.from_env(2, 1).mode == "learn"
    os.environ["HY4_SPEC2_MODE"] = "static"
    try:
        assert DepthGate.from_env(2, 1).mode == "static"
    finally:
        del os.environ["HY4_SPEC2_MODE"]


def test_confident_and_unconfident_rounds():
    g = _gate()
    g.reset()
    assert run(g, [(HI, True)] * 5) == [3] * 5                      # prior of bin 9 (0.475) beats the break-even (about 0.41)
    g.reset()
    assert run(g, [(LO, False)] * 15) == [2] * 15                   # prior of bin 0 (0.025) is far below it
    st = g.stats()
    assert st["req_depth_hist"] == {2: 15} and st["req_changed"] == 15
    assert st["depth_hist"] == {2: 15, 3: 5} and st["changed"] == 15 and st["rounds"] == 20


def test_probe_round_extends_whatever_the_bin_says():
    g = _gate()
    g.reset()
    depths = run(g, [(LO, False)] * 33)
    assert [i + 1 for i, d in enumerate(depths) if d == 3] == [16, 32]      # every 16th round (HY4_SPEC2_PROBE)


def test_gate_learns_to_stop_a_confident_but_wrong_bin():
    g = _gate(probe=0)
    g.reset()
    depths = run(g, [(HI, False)] * 40)                              # confidence says yes, the extension is never accepted
    assert depths[0] == 3 and depths[-1] == 2
    first_cut = depths.index(2)
    assert 0 < first_cut < 20 and all(d == 2 for d in depths[first_cut:])
    assert g.stats()["req_changed"] == 40 - first_cut


def test_per_request_reset_and_lifetime_counters():
    g = _gate()
    s1 = [(HI, True), (LO, False), (HI, False)] * 7
    g.reset()
    d1 = run(g, s1)
    st1 = g.stats()
    run(g, [(LO, False)] * 9)                                        # an unrelated request in between
    g.reset()
    d2 = run(g, s1)
    st2 = g.stats()
    assert d1 == d2 and st1["digest"] == st2["digest"]               # same request, same decisions (no carried state)
    assert st2["req_rounds"] == len(s1) and st2["req_depth_hist"] == st1["req_depth_hist"] and st2["req_changed"] == st1["req_changed"]
    assert st2["rounds"] == 2 * len(s1) + 9 and sum(st2["depth_hist"].values()) == st2["rounds"]    # lifetime survives reset


def test_carry_mode_is_the_negative_control():
    g = _gate("carry")
    g.reset()
    s = [(HI, False)] * 25
    d1 = run(g, s)
    g.reset()                                                        # no-op in carry mode
    d2 = run(g, s)
    assert d1 != d2


def test_static_mode_threshold():
    g = _gate("static", theta=0.6)
    assert run(g, [([0.9, 0.9], True), ([0.5, 0.5], True), ([0.8, 0.75], True), ([0.99, 0.2], True)]) == [3, 2, 3, 2]
    assert g.stats()["changed"] == 2


def test_fires_line_zero_means_not_fired():
    g = _gate()
    g.reset()
    run(g, [(HI, True)] * 4)
    line = g.fires_line()
    assert line.startswith("FIRES gate mode=learn") and "changed=0/4" in line and "depth_hist={3: 4}" in line
    run(g, [(LO, False)] * 2)
    assert "changed=2/6" in g.fires_line() and "req_changed=2/6" in g.fires_line()
