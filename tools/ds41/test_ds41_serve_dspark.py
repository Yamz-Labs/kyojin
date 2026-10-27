"""DSpark hook of the DS4.1 server: `--num-draft` and the draft fields of `timings`. CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds41_engine as eng  # noqa: E402
import test_ds41_serve as base  # noqa: E402

serve = base.serve


def test_draft_timings_only_when_drafting_ran():
    assert serve.draft_timings({}) == {}
    assert serve.draft_timings({"accepted_draft_tokens": 0, "rejected_draft_tokens": 0}) == {}
    got = serve.draft_timings({"accepted_draft_tokens": 9, "rejected_draft_tokens": 6, "draft_steps": 3})
    assert got == {"accepted_draft_tokens": 9, "rejected_draft_tokens": 6, "draft_steps": 3, "accepted_per_step": 3.0}
    got = serve.draft_timings({"accepted_draft_tokens": 9, "rejected_draft_tokens": 6, "draft_steps": 3,
                               "accepted_per_step": 2.5})
    assert got["accepted_per_step"] == 2.5


def test_timings_carry_the_draft_fields():
    async def go():
        async with base.client_for("Hi there") as (c, _):
            _, body = await base.post_chat(c)
            t = body["timings"]
            assert t["accepted_draft_tokens"] == 6 and t["rejected_draft_tokens"] == 6
            assert "accepted_per_step" in t and "cache_n" in t
    base.run(go())


def test_num_draft_flag_reaches_the_loader_only_when_on(monkeypatch):
    seen = []

    def fake_load(path, ctx, **opts):
        seen.append({k: v for k, v in opts.items() if k != "loader"})
        raise SystemExit("stop after load")

    monkeypatch.setattr(eng, "load_model", fake_load)
    for name in ("start", "stage", "abort"):
        monkeypatch.setattr(serve.startup_health, name, lambda *a, **k: None)
    for argv, want in ((["--num-draft", "3"], {"num_draft": 3}), ([], {}), (["--num-draft", "0"], {})):
        monkeypatch.setattr(sys, "argv", ["serve.py", "--loader", "stub", *argv])
        with pytest.raises(SystemExit):
            serve.main()
        assert seen[-1] == want
    monkeypatch.setattr(sys, "argv", ["serve.py", "--loader", "stub", "--num-draft", "-1"])
    with pytest.raises(SystemExit):
        serve.main()
    assert len(seen) == 3
