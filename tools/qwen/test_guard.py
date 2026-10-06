#!/usr/bin/env python3
"""Loop guard + seed + thinking budget: model-free tests (stub generator, no GPU, no exllamav3)."""
from __future__ import annotations

import asyncio
import os
import random
import re
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evalgate"))
import serve
from test_serve import FakeEngine, with_client

CLOSE, CLOSE_SEQ = 9, [5, 9, 6]  # stub token ids: </think>, and "\n</think>\n\n"


class StubTok:
    def encode(self, text, add_bos=False, encode_special_tokens=False):
        return torch.tensor([[CLOSE] if text == serve.THINK_CLOSE else CLOSE_SEQ])


class StubJob:
    jobs: list = []

    def __init__(self, input_ids, max_new_tokens, sampler, stop_conditions, embeddings=None, decode_special_tokens=False, seed=None):
        self.ids, self.max_new, self.seed, self.stops = input_ids.flatten().tolist(), max_new_tokens, seed, stop_conditions
        self.rng = random.Random(seed)
        StubJob.jobs.append(self)


class StubGenerator:
    """Emits `per_iter` tokens per iterate() call from model(context) -> token id (0 = eos)."""

    def __init__(self, model, per_iter=3):
        self.model, self.per_iter, self.job, self.out, self.serial = model, per_iter, None, [], 0
        self.active_jobs, self.pending_jobs = [], []

    def enqueue(self, job):
        self.job, self.out = job, []
        self.active_jobs[:] = [job]
        job.serial_number, self.serial = self.serial, self.serial + 1
        return job.serial_number

    def num_remaining_jobs(self):
        return 1 if self.job is not None else 0

    def cancel(self, job):
        self.job = None
        self.active_jobs[:] = []

    def iterate(self):
        j, toks, eos, reason = self.job, [], False, None
        for _ in range(self.per_iter):
            t = self.model(j.ids + self.out, j)
            if t in j.stops:
                eos, reason = True, "stop_token"
                break
            self.out.append(t)
            toks.append(t)
            if len(self.out) >= j.max_new:
                eos, reason = True, "max_new_tokens"
                break
        ev = {"job": j, "serial": j.serial_number, "token_ids": torch.tensor(toks), "text": "".join("</think>" if t == CLOSE else f"<{t}>" for t in toks)}
        if eos:
            ev |= {"eos": True, "eos_reason": reason, "new_tokens": len(self.out), "prompt_tokens": len(j.ids),
                   "cached_tokens": 0}
            self.job = None
            self.active_jobs[:] = []
        return [ev]


def make_engine(model, per_iter=3):
    e = object.__new__(serve.QwenEngine)
    e.torch, e.Job, e.generator = torch, StubJob, StubGenerator(model, per_iter)
    e.tokenizer, e.eos, e.draft_model, e.spec_on, e.slot_store, e.last_stats = StubTok(), [0], None, False, None, {}
    e._encode = lambda prompt, urls: (torch.tensor([[100, 101, 102]]), [])
    e._sampler = lambda s: None
    StubJob.jobs = []
    return e


def collect(engine, **kw):
    async def go():
        out = ""
        async for piece in engine.generate("p", sampling={"temperature": 0}, stop=[], **{"max_tokens": 600} | kw):
            out = out[:len(out) - int(piece)] if isinstance(piece, serve.Retract) else out + piece
        return out
    return asyncio.run(go()), engine.last_stats


def answer_after_close(ctx):
    """Tokens 1000.. after the (natural or forced) close, then eos."""
    k = len(ctx) - 1 - ctx[::-1].index(CLOSE)
    n = len(ctx) - k - 1 - (1 if ctx[k + 1:k + 2] == [6] else 0)
    return 1000 + n if n < 8 else 0


def loop_model(unique=40, period=9, loop_in="think"):
    """Greedy-like model: `unique` fresh tokens, then an endless cycle (in think or in answer)."""
    def m(ctx, job):
        gen = ctx[3:]
        if loop_in == "think":
            if CLOSE in gen:
                return answer_after_close(ctx)
            return 200 + len(gen) if len(gen) < unique else 500 + (len(gen) - unique) % period
        if CLOSE not in gen:
            return 200 + len(gen) if len(gen) < unique else CLOSE
        a = len(gen) - gen.index(CLOSE) - 1
        return 700 + a % period if a >= 3 else 600 + a
    return m


def normal_model(ctx, job):
    gen = ctx[3:]
    if CLOSE in gen:
        return answer_after_close(ctx)
    return 200 + len(gen) if len(gen) < 300 else CLOSE


class GuardEngineTests(unittest.TestCase):
    def test_normal_output_identical_guard_on_off(self):
        on, off = collect(make_engine(normal_model), reasoning_first=True, loop_guard=True), collect(make_engine(normal_model), reasoning_first=True, loop_guard=False)
        self.assertEqual(on, off)
        self.assertNotIn("loop_guard", on[1])
        self.assertEqual(len(on[1]["token_ids"]), 300 + 1 + 8)

    def test_final_check_recovers_answer_that_repeats_the_thinking_drafts(self):
        import loops
        draft = [800 + i for i in range(30)]

        def drafter(ctx, job):          # thinking: 40 fresh tokens, then the same 30-token draft 3 times, then close; the answer copies the draft
            gen = ctx[3:]
            if CLOSE in gen:
                a = len(gen) - gen.index(CLOSE) - 1 - (1 if gen[gen.index(CLOSE) + 1:gen.index(CLOSE) + 2] == [6] else 0)
                k = gen[:gen.index(CLOSE)]
                drafts = sum(1 for i in range(len(k) - 29) if k[i:i + 30] == draft)
                if drafts >= 2:
                    return draft[a] if a < 30 else 0
                return 1000 + a if a < 8 else 0
            n = len(gen)
            if n < 40:
                return 200 + n
            return CLOSE if n >= 40 + 90 else draft[(n - 40) % 30]
        off = collect(make_engine(drafter, per_iter=1), reasoning_first=True, loop_guard=True, loop_final=False)
        self.assertTrue(loops.is_loop(off[0]))
        text, st = collect(make_engine(drafter, per_iter=1), reasoning_first=True, loop_guard=True)
        self.assertEqual(st["loop_guard"]["kind"], "final")
        self.assertEqual(text.count("</think>"), 1)
        self.assertFalse(loops.is_loop(text), text[-200:])
        self.assertIn("<1007>", text)
        self.assertEqual(len(StubJob.jobs), 2)

    def test_final_check_recovers_think_loop_that_ends_at_the_cap(self):
        import loops

        def drift(ctx, job):            # thinking never closes: 40 fresh tokens then a 70-token cycle (above the n-gram window)
            gen = ctx[3:]
            if CLOSE in gen:
                return answer_after_close(ctx)
            n = len(gen)
            return 200 + n if n < 40 else 500 + (n - 40) % 70
        text, st = collect(make_engine(drift, per_iter=1), reasoning_first=True, loop_guard=True, loop_final=False, max_tokens=300)
        self.assertEqual(st["eos_reason"], "max_new_tokens")
        self.assertTrue(loops.is_loop(text))
        text, st = collect(make_engine(drift, per_iter=1), reasoning_first=True, loop_guard=True, max_tokens=300)
        self.assertEqual(st["loop_guard"]["kind"], "final")
        self.assertFalse(loops.is_loop(text), text[-200:])
        self.assertIn("<1007>", text)                         # the answer was generated after the cut
        self.assertLessEqual(len(st["token_ids"]), 300)       # discarded tokens are not charged to max_tokens

    def test_final_cut_unit(self):
        think = "".join(f"line {i} unique words here\n" for i in range(20)) + ("draft: the quiet man never spoke and all laughed loudly\n" * 4)
        ans = "the quiet man never spoke and all laughed loudly"
        full = think + "</think>" + ans
        self.assertTrue(serve.text_loop(full))
        c = serve.final_cut(full)
        self.assertFalse(serve.text_loop(think[:c] + "</think>" + ans))
        self.assertGreater(c, 0)
        self.assertFalse(serve.text_loop("fine text\n" * 3 + "</think>" + ans)) 

    def test_final_cut_leaves_no_span_that_the_answer_could_bring_to_three(self):
        span = "the committee agreed that the budget would be reviewed again next spring"
        think = "".join(f"line {i} unique words here\n" for i in range(20)) + span + "\n" + "middle line, different words entirely\n" + span + "\n" + "final line, nothing repeated\n"
        ans = "a short unrelated reply"
        self.assertEqual(serve.max_span_repeat(think), 2)
        c = serve.final_cut(think + "</think>" + ans)
        self.assertLessEqual(serve.max_span_repeat(think[:c].rstrip()), 1)
        self.assertLessEqual(c, think.index(span, think.index(span) + 1))

    def test_think_loop_closed_and_answer_follows(self):
        e = make_engine(loop_model(), per_iter=1)
        text, st = collect(e, reasoning_first=True, loop_guard=True)
        self.assertEqual(text.count("</think>"), 1)
        self.assertIn("\n</think>\n\n", text)
        self.assertIn("<1007>", text)                       # the answer proceeded (8 answer tokens then eos)
        self.assertEqual(st["loop_guard"]["kind"], "period")
        self.assertEqual(len(StubJob.jobs), 2)
        # restart context = prompt + thinking up to the end of the FIRST cycle + forced close
        self.assertEqual(StubJob.jobs[1].ids, [100, 101, 102] + [200 + i for i in range(40)] + [500 + i for i in range(9)] + CLOSE_SEQ)
        self.assertEqual(st["prompt_tokens"], 3)
        self.assertGreater(st["new_tokens"], len(st["token_ids"]))

    def test_visible_text_matches_kept_tokens_with_speculative_batches(self):
        piece = lambda x: f"<{x}>"
        for per_iter in (1, 2, 3, 4, 5):
            text, st = collect(make_engine(loop_model(), per_iter=per_iter), reasoning_first=True, loop_guard=True)
            toks = st["token_ids"]
            k = next(i for i in range(len(toks)) if toks[i:i + 3] == CLOSE_SEQ)
            expect = "".join(map(piece, toks[:k])) + serve.THINK_CLOSE_TEXT + "".join(map(piece, toks[k + 3:]))
            self.assertEqual(text, expect, per_iter)           # the discarded tail was retracted, nothing else changed
            self.assertGreater(st["new_tokens"], len(toks))    # generated counts the discarded tail too

    def test_guard_off_runs_to_the_cap(self):
        text, st = collect(make_engine(loop_model()), reasoning_first=True, loop_guard=False, max_tokens=300)
        self.assertNotIn("</think>", text)
        self.assertEqual(st["eos_reason"], "max_new_tokens")

    def test_env_switch(self):
        os.environ["EXL3_LOOP_GUARD"] = "0"
        try:
            text, st = collect(make_engine(loop_model()), reasoning_first=True, max_tokens=300)
        finally:
            del os.environ["EXL3_LOOP_GUARD"]
        self.assertNotIn("</think>", text)

    def test_answer_loop_ends_the_turn(self):
        text, st = collect(make_engine(loop_model(loop_in="answer")), reasoning_first=True, loop_guard=True, max_tokens=900)
        self.assertEqual(st["eos_reason"], "loop_guard")
        self.assertEqual(st["loop_guard"]["phase"], "answer")
        self.assertLess(len(st["token_ids"]), 400)

    def test_budget_closes_think(self):
        text, st = collect(make_engine(normal_model, per_iter=1), reasoning_first=True, loop_guard=False, thinking_budget=50)
        self.assertEqual(st["loop_guard"]["kind"], "budget")
        self.assertEqual(StubJob.jobs[1].ids, [100, 101, 102] + [200 + i for i in range(50)] + CLOSE_SEQ)
        self.assertEqual(text.count("</think>"), 1 + 0)

    def test_budget_ignored_without_think_and_after_natural_close(self):
        text, st = collect(make_engine(normal_model), reasoning_first=False, thinking_budget=5, loop_guard=False)
        self.assertEqual(len(StubJob.jobs), 1)
        text, st = collect(make_engine(normal_model), reasoning_first=True, thinking_budget=5000)
        self.assertEqual(len(StubJob.jobs), 1)

    def test_cap_reached_during_restart(self):
        text, st = collect(make_engine(loop_model()), reasoning_first=True, loop_guard=True, loop_final=False, max_tokens=70)
        self.assertEqual(st["eos_reason"], "max_new_tokens")
        self.assertLessEqual(len(st["token_ids"]), 70 + 5)

    def test_seed_forwarded_and_reproducible(self):
        jobs = []
        for sd in (1234, 1234, 99):
            collect(make_engine(normal_model), reasoning_first=True, seed=sd)
            jobs.append(StubJob.jobs[0])
        a, b, c = jobs
        self.assertEqual((a.seed, b.seed, c.seed), (1234, 1234, 99))
        draw = lambda j: [j.rng.randint(0, 2**32 - 1) for _ in range(5)]
        self.assertEqual(draw(a), draw(b))
        self.assertNotEqual(draw(a), draw(c))

    def test_real_job_accepts_seed(self):
        src = (Path(__file__).resolve().parents[2] / "exllamav3" / "generator" / "job.py").read_text()
        self.assertTrue(re.search(r"seed: int = None", src) and "random.Random(seed)" in src)


class MultiStubGenerator(StubGenerator):
    """StubGenerator with several jobs at once: each iterate() steps every job, like a batched decode."""

    def __init__(self, model, per_iter=3, requeue_at=None):
        super().__init__(model, per_iter)
        self.jobs, self.peak, self.requeue_at = {}, 0, requeue_at

    @property
    def active_jobs(self):
        return list(self.jobs)

    @active_jobs.setter
    def active_jobs(self, _):
        pass

    def enqueue(self, job):
        job.serial_number = self.serial
        if job.seed == 0:   # like Job.prepare_for_queue's fit check: the serial is set, then the enqueue fails
            raise AssertionError("job does not fit")
        self.jobs[job] = []
        self.serial += 1
        return job.serial_number

    def num_remaining_jobs(self):
        return len(self.jobs)

    def cancel(self, job):
        self.jobs.pop(job, None)

    def iterate(self):
        self.peak = max(self.peak, len(self.jobs))
        evs = []
        for j in list(self.jobs):
            self.job, self.out = j, self.jobs[j]
            evs += StubGenerator.iterate(self)
            if self.job is None:
                del self.jobs[j]
            elif self.requeue_at and len(self.out) >= self.requeue_at and not getattr(j, "rq", False):
                # a requeue that hands back another Job object with the same serial (Generator.prepare_for_requeue reuses
                # the object; routing by serial must not depend on that)
                rq = object.__new__(StubJob)
                rq.__dict__.update(j.__dict__, ids=j.ids + self.out, max_new=j.max_new - len(self.out), rq=True)
                del self.jobs[j]
                self.jobs[rq] = []
        self.job = None
        return evs


class SessionTests(unittest.TestCase):
    @staticmethod
    def seeded(ctx, job):
        n = len(ctx) - 3
        return 10_000 * job.seed + 200 + n if n < 30 else 0

    def run_two(self, cancel_first_after=None, requeue_at=None):
        e = make_engine(self.seeded, per_iter=2)
        e.generator = MultiStubGenerator(self.seeded, per_iter=2, requeue_at=requeue_at)

        async def one(seed, cancel):
            st, n = {}, 0
            async for _ in e.generate("p", stats=st, sampling={"temperature": 0}, stop=[], max_tokens=600, seed=seed,
                                      loop_guard=False, cancel=cancel):
                n += 1
                if cancel_first_after and seed == 1 and n == cancel_first_after:
                    cancel.set()
            # generate() returns only after its cancels ran: no job of this request may be left at this point
            self.assertFalse([j for j in e.generator.jobs if j.seed == seed])
            return st.get("token_ids")

        async def both():
            # a lost route waits forever: fail instead of hanging
            return await asyncio.wait_for(asyncio.gather(one(1, asyncio.Event()), one(2, asyncio.Event())), 10)
        return asyncio.run(both()), e.generator

    def test_concurrent_requests_get_their_own_tokens(self):
        (a, b), gen = self.run_two()
        self.assertEqual(a, [10_200 + i for i in range(30)])
        self.assertEqual(b, [20_200 + i for i in range(30)])
        self.assertEqual(gen.peak, 2)
        self.assertEqual(gen.jobs, {})

    def test_requeued_job_keeps_its_stream(self):
        (a, b), gen = self.run_two(requeue_at=10)
        self.assertEqual(a, [10_200 + i for i in range(30)])
        self.assertEqual(b, [20_200 + i for i in range(30)])

    def test_cancel_after_requeue_reaches_the_new_job(self):
        (a, b), gen = self.run_two(cancel_first_after=8, requeue_at=4)
        self.assertIsNone(a)
        self.assertEqual(b, [20_200 + i for i in range(30)])
        self.assertEqual(gen.jobs, {})

    def test_failed_enqueue_does_not_cancel_the_job_that_reuses_its_serial(self):
        e = make_engine(self.seeded, per_iter=2)
        e.generator = MultiStubGenerator(self.seeded, per_iter=2)

        async def one(seed):
            st = {}
            try:
                async for _ in e.generate("p", stats=st, sampling={"temperature": 0}, stop=[], max_tokens=600, seed=seed,
                                          loop_guard=False):
                    pass
            except AssertionError:
                return "failed"
            return st.get("token_ids")

        async def both():
            return await asyncio.wait_for(asyncio.gather(one(0), one(2)), 10)
        failed, b = asyncio.run(both())
        self.assertEqual(failed, "failed")
        self.assertEqual(b, [20_200 + i for i in range(30)])

    def test_stopped_driver_fails_the_request_instead_of_hanging(self):
        e = make_engine(self.seeded, per_iter=1)
        e.generator = MultiStubGenerator(self.seeded, per_iter=1)

        async def go():
            n = 0
            async for _ in e.generate("p", stats={}, sampling={"temperature": 0}, stop=[], max_tokens=600, seed=1,
                                      loop_guard=False):
                n += 1
                if n == 2:
                    e._driver.cancel()
        with self.assertRaisesRegex(RuntimeError, "driver has stopped"):
            asyncio.run(asyncio.wait_for(go(), 10))

    def test_cancel_leaves_the_other_session_running(self):
        (a, b), gen = self.run_two(cancel_first_after=3)
        self.assertIsNone(a)
        self.assertEqual(b, [20_200 + i for i in range(30)])
        self.assertEqual(gen.jobs, {})


class HttpFieldTests(unittest.TestCase):
    def post(self, payload):
        eng = FakeEngine("hi")

        async def fn(client):
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]} | payload)
            return r.status
        return with_client(eng, fn), eng

    def test_fields_reach_engine(self):
        st, eng = self.post({"seed": 7, "thinking_budget": 300, "loop_guard": False})
        self.assertEqual(st, 200)
        kw = eng.calls[0][1]
        self.assertEqual((kw["seed"], kw["thinking_budget"], kw["loop_guard"], kw["reasoning_first"]), (7, 300, False, True))

    def test_defaults(self):
        st, eng = self.post({})
        kw = eng.calls[0][1]
        self.assertEqual((kw["seed"], kw["thinking_budget"], kw["loop_guard"]), (None, None, None))

    def test_bad_values_400(self):
        for bad in ({"seed": "x"}, {"seed": True}, {"thinking_budget": 0}, {"thinking_budget": "a"}, {"loop_guard": "yes"}):
            self.assertEqual(self.post(bad)[0], 400, bad)


class RetractEngine(FakeEngine):
    async def generate(self, prompt, **kw):
        self.last_stats = dict(self.stats, loop_guard={"kind": "period"})
        for p in ("thinking ok ", "LOOPLOOP", serve.Retract(8), "\n</think>\n\n", "answer"):
            yield p


class RetractHttpTests(unittest.TestCase):
    def post(self, stream):
        async def fn(client):
            r = await client.post("/v1/chat/completions", json={"model": "m", "stream": stream, "messages": [{"role": "user", "content": "x"}]})
            return await (r.text() if stream else r.json())
        return with_client(RetractEngine(""), fn)

    def test_non_stream_drops_the_discarded_tail(self):
        b = self.post(False)
        m = b["choices"][0]["message"]
        self.assertEqual((m["reasoning_content"], m["content"]), ("thinking ok", "answer"))
        self.assertEqual(b["timings"]["loop_guard"], {"kind": "period"})

    def test_stream_survives_retract(self):
        raw = self.post(True)
        self.assertIn("answer", raw)
        self.assertIn("[DONE]", raw)


# ------------------------------------------------------------------ detector: no false positive on normal text

def tokenize(text):
    """Word-ish tokens -> ids, text pieces per token (stands in for the BPE)."""
    vocab: dict[str, int] = {}
    pieces = re.findall(r"\s+|\w+|[^\w\s]", text)
    return [vocab.setdefault(p, len(vocab) + 1000) for p in pieces], pieces


def run_monitor(text, phase="think", chunk=3):
    ids, pieces = tokenize(text)
    m = serve.LoopMonitor(phase, None)
    for i in range(0, len(ids), chunk):
        hit = m.feed(ids[i:i + chunk], "".join(pieces[i:i + chunk]))
        if hit:
            return hit
    return None


FALSE_POSITIVE_TEXTS = {
    "counting": " ".join(str(i) for i in range(1, 51)) + "\n" + "\n".join(f"{i}. item number {i}" for i in range(1, 51)),
    "bullets": "\n".join(f"- Point {i}: we must check the constraint number {i} again" for i in range(60)),
    "table": "| id | name | score |\n|----|------|-------|\n" + "\n".join(f"| {i} | user{i} | {i * 7 % 100} |" for i in range(80)),
    "code": "def f(x):\n    y = x + 1\n" + "    print('step')\n" * 5 + "    return y\n" + "\n".join(f"    v{i} = compute(a, b, {i})" for i in range(30)),
    "draft_rewrite": ("Need a haiku about dogs. Draft: the dog runs fast / the day is long and bright / warm sun on the grass. "
                      "Check syllables: five seven five. Looks fine. Rewrite the draft: the dog runs fast / the day is long and bright / "
                      "warm sun on the grass. Now verify no commas appear anywhere in the response. Good. Final: the dog runs fast / "
                      "the day is long and bright / warm sun on the grass."),
    "letter_count": " ".join(f"{i}:{w}" for i, w in enumerate("the quick brown fox jumps over the lazy dog and keeps running far away".split() * 4)),
}


class DetectorTests(unittest.TestCase):
    def test_no_false_positive(self):
        for name, text in FALSE_POSITIVE_TEXTS.items():
            for phase in ("think", "answer"):
                self.assertIsNone(run_monitor(text, phase), (name, phase))

    def test_exact_cycle_fires_think_and_reports_first_cycle_cut(self):
        pre = list(range(1000, 1040))
        cyc = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        m = serve.LoopMonitor("think", None)
        hit = None
        for t in pre + cyc * 10:
            hit = hit or m.feed([t])
        self.assertEqual((hit["kind"], hit["period"]), ("period", 9))
        self.assertEqual(hit["cut"], 40 + 9)

    def test_period_one_and_answer_is_stricter(self):
        m = serve.LoopMonitor("think", None)
        hit = next((h for t in [7] * 60 if (h := m.feed([t]))), None)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["at"], 48)
        m = serve.LoopMonitor("answer", None)
        hit = next((h for t in [7] * 60 if (h := m.feed([t]))), None)
        self.assertIsNone(hit)                              # 60 < answer min span 96

    def test_text_rule_fires_on_sustained_drift_only(self):
        cycle = ['"alpha beta" no.\n', '"gamma delta" no.\n', '"alpha beta" maybe.\n', '"epsilon zeta" no.\n']
        vocab = {}
        m, hit, n = serve.LoopMonitor("think", None), None, 0
        for rep in range(200):
            for p in cycle[:3] + [f'"drift{rep} word" no.\n']:   # one drifting piece per cycle: no exact period
                tid = vocab.setdefault(p, 5000 + len(vocab))
                n += 1
                hit = hit or m.feed([tid], p)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["kind"], "text")

    def test_close_token_switches_phase(self):
        m = serve.LoopMonitor("think", 9)
        self.assertIsNone(m.feed([1, 2, 3, 9, 4]))
        self.assertEqual(m.phase, "answer")
        self.assertEqual(m.ids, [4])

    def test_text_rule_equals_release_gate_rule(self):
        import loops
        samples = ["ab" * 800, "the quick brown fox " * 100, "hello world. " * 40 + "unique tail " * 5,
                   "".join(chr(0x0e01 + i % 40) for i in range(1200)), FALSE_POSITIVE_TEXTS["table"] * 3]
        rng = random.Random(1)
        samples += ["".join(rng.choice("abc def\n") for _ in range(2500)) for _ in range(5)]
        for s in samples:
            self.assertEqual(serve.text_loop(s), bool(loops.is_loop(s)))


if __name__ == "__main__":
    unittest.main()
