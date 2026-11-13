#!/usr/bin/env python3
"""EXL3_SPEC_TYPICAL_COOLDOWN / EXL3_SPEC_TYPICAL_REWIND at the server: model-free tests (stub generator, no GPU).   python tools/qwen/test_spec_rewind.py
A looping thinking block is rewound to just before the first copy of the repeated line (at most 3 times a reply), the new job starts in an
exact-acceptance window, the answer channel is never rewound, and with the flags off or no signal the reply is the guard-only reply."""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evalgate"))
import serve
import test_guard as TG
from test_guard import CLOSE, StubJob, StubGenerator, make_engine
from exllamav3.generator import spec_sample as ss

UNIQUE, PERIOD, APERIOD = 30, 70, 5     # a loop of 70 tokens: its 2nd sighting lies past the 64-token debounce of the restart


def text_of(t):
    """One line per token (>= 40 non-space chars, distinct per id); the close tag is not a line."""
    if t == CLOSE:
        return "</think>"
    return f"reasoning step number {t} written out in full so that the line is long\n"


class LineGenerator(StubGenerator):
    def iterate(self):
        evs = super().iterate()
        for ev in evs:
            ev["text"] = "".join(text_of(t) for t in ev["token_ids"].tolist())
        return evs


def make(model, per_iter=3):
    e = make_engine(model, per_iter)
    e.generator = LineGenerator(model, per_iter)
    return e


def loop_in_think(ctx, job):
    gen = ctx[3:]
    if CLOSE in gen:
        return TG.answer_after_close(ctx)
    return 200 + len(gen) if len(gen) < UNIQUE else 500 + (len(gen) - UNIQUE) % PERIOD


def loop_in_answer(ctx, job):
    gen = ctx[3:]
    if CLOSE not in gen:
        return 200 + len(gen) if len(gen) < 20 else CLOSE
    a = len(gen) - gen.index(CLOSE) - 1
    return 700 + a % APERIOD if a >= 3 else 600 + a


def run(engine, temperature=0.7, **kw):
    async def go():
        out = ""
        async for piece in engine.generate("p", sampling={"temperature": temperature}, stop=[], **{"max_tokens": 600, "reasoning_first": True} | kw):
            out = out[:len(out) - int(piece)] if isinstance(piece, serve.Retract) else out + piece
        return out
    return asyncio.run(go()), engine.last_stats


class RewindTests(unittest.TestCase):
    def setUp(self):
        self.old = (ss.TYPICAL, dict(ss.STATS))
        ss.TYPICAL = True
        for k in ("EXL3_SPEC_TYPICAL_COOLDOWN", "EXL3_SPEC_TYPICAL_REWIND"):
            os.environ.pop(k, None)
        self.n0 = dict(ss.STATS)

    def tearDown(self):
        ss.TYPICAL = self.old[0]
        for k in ("EXL3_SPEC_TYPICAL_COOLDOWN", "EXL3_SPEC_TYPICAL_REWIND"):
            os.environ.pop(k, None)

    def d(self, k):
        return ss.STATS[k] - self.n0[k]

    def test_thinking_loop_rewinds_to_before_first_copy_then_cap_then_hard_guard(self):
        text, st = run(make(loop_in_think), seed=1234)
        self.assertEqual(self.d("rewinds"), 3)                                 # the cap
        jobs = StubJob.jobs
        first = jobs[1].ids[3:]                                                # the kept thinking of the first rewound job
        self.assertEqual(first, [200 + i for i in range(UNIQUE)])              # cut just before the first copy of the repeated line
        self.assertGreater(self.d("rewound_tokens"), 0)
        self.assertGreaterEqual(self.d("cooldown_triggers"), 3)                # one per rewind, then the window re-triggers (doubles) before the hard guard
        for j in jobs[1:4]:
            self.assertIsNotNone(getattr(j, "cd", None))                       # exact-acceptance window carried to the new job
            self.assertTrue(j.cd.armed(0)); self.assertGreaterEqual(j.cd.win, 512)
        self.assertEqual([j.seed for j in jobs[:4]], [1234 + 1000003 * k for k in range(4)])   # the sampling key moves on at every rewind
        self.assertIn("loop_guard", st)                                        # after the cap the hard guard closes the think block
        self.assertEqual(st["spec_rewinds"], 3)
        self.assertIn("</think>", text)

    def test_rewound_reply_does_not_double_the_text(self):
        text, _ = run(make(loop_in_think), max_tokens=600)
        for k in range(200, 200 + UNIQUE):
            self.assertEqual(text.count(f"number {k} "), 1)

    def test_rewind_off_cooldown_only(self):
        os.environ["EXL3_SPEC_TYPICAL_REWIND"] = "0"
        text, st = run(make(loop_in_think))
        self.assertEqual(self.d("rewinds"), 0)
        self.assertEqual(self.d("cooldown_triggers"), 2)                       # the signal still arms the window; a re-trigger (64+ tokens later) doubles it
        self.assertEqual(StubJob.jobs[0].cd.win, 1024)
        self.assertEqual(len(StubJob.jobs), 2)                    # only the hard guard's restart
        self.assertNotIn("spec_rewinds", st)

    def test_cooldown_off_nothing_fires(self):
        os.environ["EXL3_SPEC_TYPICAL_COOLDOWN"] = "0"
        run(make(loop_in_think))
        self.assertEqual((self.d("rewinds"), self.d("cooldown_triggers")), (0, 0))

    def test_relaxed_rule_off_or_greedy_nothing_fires(self):
        ss.TYPICAL = False
        run(make(loop_in_think))
        self.assertEqual((self.d("rewinds"), self.d("cooldown_triggers")), (0, 0))
        ss.TYPICAL = True
        run(make(loop_in_think), temperature=0)
        self.assertEqual((self.d("rewinds"), self.d("cooldown_triggers")), (0, 0))

    def test_answer_channel_never_rewound(self):
        text, st = run(make(loop_in_answer))
        self.assertEqual(self.d("rewinds"), 0)
        self.assertEqual(self.d("cooldown_triggers"), 0)
        self.assertEqual(len(StubJob.jobs), 1)                                 # no restart before the hard guard (answer) ends the reply
        self.assertEqual(st.get("eos_reason"), "loop_guard")

    def test_no_think_block_never_rewound(self):
        run(make(loop_in_think), reasoning_first=False)
        self.assertEqual(self.d("rewinds"), 0)

    def test_no_signal_identical_to_flags_off(self):
        a = run(make(TG.normal_model))
        os.environ["EXL3_SPEC_TYPICAL_COOLDOWN"] = "0"; os.environ["EXL3_SPEC_TYPICAL_REWIND"] = "0"
        b = run(make(TG.normal_model))
        self.assertEqual(a[0], b[0]); self.assertEqual(a[1]["token_ids"], b[1]["token_ids"])
        self.assertEqual((self.d("rewinds"), self.d("cooldown_triggers")), (0, 0))
        self.assertNotIn("spec_rewinds", a[1])

    def test_fenced_repeat_does_not_trigger(self):   # a legit repeated code line inside a fence
        w = serve.LineWatch()
        line = "    result = compute_the_value(alpha, beta, gamma, delta)\n"
        self.assertFalse(w.feed("```python\n" + line + line + line + "```\n"))


class LineWatchTests(unittest.TestCase):
    def test_second_sighting_hits_and_marks_first_copy(self):
        w = serve.LineWatch()
        a = "this is a rather long reasoning line with enough characters\n"
        b = "another distinct and rather long reasoning line of text here\n"
        self.assertFalse(w.feed(a + b))
        self.assertTrue(w.feed(a))
        self.assertEqual(w.first_at, 0)
        w2 = serve.LineWatch()
        w2.feed("short\n" + b + a)
        self.assertTrue(w2.feed(b))
        self.assertEqual(w2.first_at, len("short\n"))

    def test_short_line_and_fence_ignored(self):
        w = serve.LineWatch()
        self.assertFalse(w.feed("too short\ntoo short\ntoo short\n"))
        line = "    result = compute_the_value(alpha, beta, gamma, delta)\n"
        self.assertFalse(w.feed("```\n" + line * 3 + "```\n"))

    def test_split_across_feeds(self):
        w = serve.LineWatch()
        a = "this is a rather long reasoning line with enough characters\n"
        w.feed(a[:20]); w.feed(a[20:])
        self.assertTrue(w.feed(a[:30]) or w.feed(a[30:]))


if __name__ == "__main__":
    unittest.main()
