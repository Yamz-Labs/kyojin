"""Serve hook of the DSpark early stop and auto draft length: `--draft-auto`, `--draft-conf`, cost fields of `timings`."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds41_engine as eng  # noqa: E402
import test_ds41_serve as base  # noqa: E402

serve = base.serve


def test_draft_timings_cost_fields():
    st = {"accepted_draft_tokens": 9, "rejected_draft_tokens": 3, "draft_steps": 4, "accepted_per_step": 2.25,
          "draft_len_mean": 3.0, "verify_positions": 20, "conf_dropped_tokens": 5, "conf_cut_rounds": 2, "tuner_k": 3}
    assert serve.draft_timings(st) == {
        "accepted_draft_tokens": 9, "rejected_draft_tokens": 3, "draft_steps": 4, "accepted_per_step": 2.25,
        "draft_len_mean": 3.0, "verify_positions": 20, "conf_dropped_tokens": 5, "conf_cut_rounds": 2,
        "draft_tuner_k": 3}
    # fields the engine does not send are absent; a zero tuner_k (no tuner) is left out
    got = serve.draft_timings({"accepted_draft_tokens": 6, "rejected_draft_tokens": 2, "draft_steps": 2, "tuner_k": 0})
    assert "draft_len_mean" not in got and "draft_tuner_k" not in got and "verify_positions" not in got
    # every draft cut by the confidence stop: still reported (the saving is the point)
    got = serve.draft_timings({"accepted_draft_tokens": 0, "rejected_draft_tokens": 0, "conf_dropped_tokens": 7})
    assert got["conf_dropped_tokens"] == 7 and got["draft_steps"] == 0
    assert serve.draft_timings({"conf_dropped_tokens": 0}) == {}


def test_draft_auto_and_conf_flags_reach_the_loader(monkeypatch):
    seen = []

    def fake_load(path, ctx, **opts):
        seen.append({k: v for k, v in opts.items() if k != "loader"})
        raise SystemExit("stop after load")

    monkeypatch.setattr(eng, "load_model", fake_load)
    for name in ("start", "stage", "abort"):
        monkeypatch.setattr(serve.startup_health, name, lambda *a, **k: None)
    cases = (([], {}),
             (["--draft-auto"], {"num_draft": 4, "draft_auto": True}),
             (["--num-draft", "3", "--draft-auto"], {"num_draft": 3, "draft_auto": True}),
             (["--num-draft", "5", "--draft-conf", "0.4"], {"num_draft": 5, "draft_conf_threshold": 0.4}),
             (["--draft-auto", "--draft-conf", "0.25"],
              {"num_draft": 4, "draft_auto": True, "draft_conf_threshold": 0.25}))
    for argv, want in cases:
        monkeypatch.setattr(sys, "argv", ["serve.py", "--loader", "stub", *argv])
        with pytest.raises(SystemExit):
            serve.main()
        assert seen[-1] == want
    n = len(seen)
    for argv in (["--draft-conf", "0.4"], ["--num-draft", "3", "--draft-conf", "1.0"],
                 ["--num-draft", "3", "--draft-conf", "0"]):
        monkeypatch.setattr(sys, "argv", ["serve.py", "--loader", "stub", *argv])
        with pytest.raises(SystemExit):
            serve.main()
    assert len(seen) == n                                           # refused before the loader
