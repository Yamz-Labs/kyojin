#!/usr/bin/env python3
"""EXL3_SCHED_CHUNK_IL: chunked-prefill interleave. Pure python (no torch, no GPU): the real planner (exllamav3/generator/chunk_il.py)
and the real Generator glue (_cil_views/_cil_parked/_cil_hold/_cil_release, loaded by ast) drive a fake engine on a virtual clock.
Proves: interleave order, hard max deferral (fixed seconds and proportional to the big prefill), big TTFT <= solo + 5 % at a small
budget, per-job chunk ranges identical to the non-interleaved run, one batch-1 decode per step, equal bigs served in arrival order,
single-user path untouched. Slot-limited serving model (serve()): the default budget costs the big request less than a fixed 8 s
cap and gives the shorts the same first-token times."""
from __future__ import annotations

import ast
import importlib.util
import logging
import os
import sys
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("chunk_il", ROOT / "exllamav3" / "generator" / "chunk_il.py")
chunk_il = importlib.util.module_from_spec(spec)
sys.modules["chunk_il"] = chunk_il
spec.loader.exec_module(chunk_il)

PAGE, CHUNK, MERGE = 256, 4096, 1024
ROW_S, DEC_S, DEC_B = 35.8 / 45000, 0.03, 0.004   # 45K prefill = 35.8 s (FIFO reference), batch-1 decode step 30 ms, +4 ms per extra row


def load_glue():
    src = (ROOT / "exllamav3" / "generator" / "generator.py").read_text()
    names = ("_cil_views", "_cil_parked", "_cil_hold", "_cil_release")
    ns = {"os": os, "logging": logging, "chunk_il": chunk_il, "PAGE_SIZE": PAGE}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(textwrap.dedent(ast.get_source_segment(src, n)), ns)
    return {k: ns[k] for k in names}


class Clock:
    t = 0.0
    def __call__(self):
        return self.t


class FakeJob:
    def __init__(self, sim, name, n, out=0, serial=0, resume=0):
        self.sim, self.name, self.n, self.out, self.serial_number = sim, name, n, out, serial
        self.resume, self.time_first_prefill, self.time_first_token = resume, None, None      # resume: tail-checkpoint position the first step jumps to
        self.sequences = [types.SimpleNamespace(sequence_ids=range(n), kv_position=0)]
        self.cached_pages, self._cil_hold = 0, False
        self.ranges, self.tokens, self.t_prefill_done, self.t_arrive = [], 0, None, sim.clk.t

    def is_prefill_done(self):
        return (not self._cil_hold) and self.sequences[0].kv_position == self.n - 1

    def _tail_peek(self, seq):
        return self.resume if self.time_first_prefill is None else 0

    def prefill(self):
        seq = self.sequences[0]
        if self.time_first_prefill is None:                       # Job._tail_restore: jump over the cached conversation
            self.time_first_prefill = self.sim.clk.t
            seq.kv_position = max(seq.kv_position, self.resume)
            if seq.kv_position == self.n - 1:
                self.t_prefill_done = self.sim.clk.t; return
        rem = self.n - 1 - seq.kv_position
        chunk = self.sim.g.max_chunk_size                          # lowered at run time under decode priority
        end = self.n - 1 if rem <= chunk + MERGE else (seq.kv_position + chunk) // PAGE * PAGE   # Job.prefill's chunk rule (shape)
        self.ranges.append((seq.kv_position, end))
        self.sim.clk.t += (end - seq.kv_position) * ROW_S
        seq.kv_position = end
        if end == self.n - 1:
            self.t_prefill_done = self.sim.clk.t


class Sim:
    """Generator.iterate() reduced to its scheduling skeleton, same call order as _iterate_inner."""
    def __init__(self, flag, **planner_kw):
        self.clk = Clock()
        self.flag = flag
        g = types.SimpleNamespace(active_jobs=[], max_chunk_size=CHUNK, base_chunk_size=CHUNK)
        for k, f in load_glue().items():
            setattr(g, k, types.MethodType(f, g))
        g._cil = chunk_il.ChunkPlanner(clock=self.clk, **planner_kw) if flag else None
        self.g, self.serial, self.all, self.log = g, 0, [], []     # log: (t_start, t_end, prefilled names, decoders)
        os.environ["EXL3_SCHED_CHUNK_IL"] = "1" if flag else "0"

    def add(self, name, n, out=0, resume=0):
        j = FakeJob(self, name, n, out, self.serial, resume)
        self.serial += 1
        self.g.active_jobs.append(j); self.all.append(j)
        return j

    def step(self):
        g, t0 = self.g, self.clk.t
        parked = g._cil_parked() if chunk_il.chunk_il_enabled() else []
        pre = []
        for j in list(g.active_jobs):
            if j not in parked and not j.is_prefill_done():
                j.prefill(); pre.append(j.name)
        g._cil_hold()
        dec = [j for j in g.active_jobs if j.is_prefill_done() and j.t_prefill_done is not None]
        if dec:
            self.clk.t += DEC_S + DEC_B * (len(dec) - 1)
            for j in dec:
                j.tokens += 1
                if j.time_first_token is None:
                    j.time_first_token = self.clk.t
        g._cil_release()
        for j in dec:
            if j.tokens >= j.out:
                g.active_jobs.remove(j)
        self.log.append((t0, self.clk.t, pre, [j.name for j in dec]))

    def run(self, until, arrivals=(), refill=None):
        """arrivals: [(t, name, n, out)]; refill(sim): called after every step (closed loop of shorts)."""
        pend = sorted(arrivals)
        while self.g.active_jobs or pend:
            while pend and pend[0][0] <= self.clk.t:
                _, nm, n, out = pend.pop(0); self.add(nm, n, out)
            if not self.g.active_jobs:
                self.clk.t = pend[0][0]; continue
            self.step()
            if refill:
                refill(self)
            if self.clk.t > until:
                break


def serve(flag, shorts_t, bigs=((8.0, 45000),), cap=3, short_rows=900, out=40, **planner_kw):
    """Slot-limited serving: at most `cap` jobs run, the rest wait in arrival order (the --sessions permits); a short of `short_rows`
    rows and `out` tokens arrives at each time of shorts_t, bigs = [(t, rows)]. Returns the sim; job.t_arrive = arrival time."""
    s = Sim(flag, **planner_kw)
    arr = sorted([(t, f"s{k}", short_rows, out) for k, t in enumerate(shorts_t)] + [(t, f"A{i}", n, 1) for i, (t, n) in enumerate(bigs)])
    queue = []
    while arr or queue or s.g.active_jobs:
        while arr and arr[0][0] <= s.clk.t:
            queue.append(arr.pop(0))
        while queue and len(s.g.active_jobs) < cap:
            ta, nm, n, o = queue.pop(0)
            s.add(nm, n, o).t_arrive = ta
        if not s.g.active_jobs:
            s.clk.t = arr[0][0]; continue
        s.step()
    return s


def first_tokens(s, prefix):
    return sorted(j.time_first_token - j.t_arrive for j in s.all if j.name.startswith(prefix))


def solo_big_ttft(n=45000):
    s = Sim(False); a = s.add("A", n, 1); s.run(1e9)
    return a.t_prefill_done


class DecodePriorityTests(unittest.TestCase):
    """EXL3_CIL_DECODE_PRIO=1: a running stream waits one short chunk, not the whole big prefill (3 users on one 3090)."""

    def tearDown(self):
        os.environ.pop("EXL3_SCHED_CHUNK_IL", None)

    def run_case(self, flag, **kw):
        s = Sim(flag, **kw)
        s.add("u1", 900, out=4000); s.add("u2", 900, out=4000)
        big = []

        def refill(sim):
            if not big and sim.clk.t >= 3.0:
                big.append(sim.add("A", 100000, 1))
            if big and big[0].t_prefill_done is not None and sim.clk.t > big[0].t_prefill_done + 5:
                sim.g.active_jobs.clear()
        s.run(600, refill=refill)
        a = big[0]
        ends = [t1 for (t0, t1, pre, dec) in s.log if "u1" in dec and a.t_arrive <= t0 <= a.t_prefill_done]
        gaps = [b - a_ for a_, b in zip(ends, ends[1:])]
        toks = sum(1 for (t0, t1, pre, dec) in s.log if "u1" in dec and a.t_arrive <= t0 <= a.t_prefill_done)
        return max(gaps), a.t_prefill_done - a.t_arrive, toks / (a.t_prefill_done - a.t_arrive), s

    def test_stream_gap_bounded_and_big_prefill_not_starved(self):
        solo = solo_big_ttft(100000)
        gap0, t0_, _, _ = self.run_case(True)                       # decode priority off: leader chunks are exclusive
        self.assertGreater(gap0, 3.0)
        os.environ["EXL3_CIL_DECODE_PRIO"] = "1"
        try:
            gap1, t1_, rate1, s = self.run_case(True, slice_s=0.35)
        finally:
            os.environ.pop("EXL3_CIL_DECODE_PRIO", None)
        self.assertLess(gap1, 1.5)                                  # one 768-row chunk (the last one merges the tail, 1792 rows), never a 4096-row chunk
        self.assertLess(t1_, 1.7 * solo)                            # the big prefill keeps about 60 % of the GPU
        self.assertGreater(rate1, 10)                               # rounds per second of a stream during the big prefill (full speed 29: about 40 %)

    def test_short_prefill_beside_a_stream_takes_short_chunks(self):
        def gap(prio):
            if prio:
                os.environ["EXL3_CIL_DECODE_PRIO"] = "1"
            try:
                s = Sim(True)
                s.add("u1", 900, out=2000)
                arrived = []

                def refill(sim):
                    if not arrived and sim.clk.t >= 3.0:
                        arrived.append(sim.add("u2", 3500, out=30))   # a tool result of 3.5K rows: a short prefill (below the 4096 big threshold)
                    if arrived and arrived[0].time_first_token is not None:
                        sim.g.active_jobs.clear()
                s.run(60, refill=refill)
                ends = [t1 for (t0, t1, pre, dec) in s.log if "u1" in dec and 2.5 <= t0 <= arrived[0].t_prefill_done]
                return max(b - a for a, b in zip(ends, ends[1:]))
            finally:
                os.environ.pop("EXL3_CIL_DECODE_PRIO", None)
        self.assertGreater(gap(False), 2.0)    # one 3.5K-row chunk (4096 cap) stops the stream
        self.assertLess(gap(True), 1.1)        # 768-row chunks with a decode round between them (the last chunk merges the tail: <= 1792 rows)

    def test_chunk_size_restored_when_nobody_decodes(self):
        os.environ["EXL3_CIL_DECODE_PRIO"] = "1"
        try:
            s = Sim(True)
            s.add("A", 30000, out=1)
            s.add("u1", 900, out=3)
            s.run(60)
            s.add("B", 5000, out=1)       # alone: the first step of a lone job restores the generator's chunk size before its prefill
            s.step()
            self.assertEqual(s.g.max_chunk_size, CHUNK)
        finally:
            os.environ.pop("EXL3_CIL_DECODE_PRIO", None)


class InterleaveTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("EXL3_SCHED_CHUNK_IL", None)

    def stream(self, flag, n_users=2, **kw):
        s = Sim(flag, **kw)
        cnt = [0]

        def new_short(sim):
            for u in range(n_users):
                if not any(j.name.startswith(f"s{u}_") for j in sim.g.active_jobs):
                    cnt[0] += 1; sim.add(f"s{u}_{cnt[0]}", 900, 40)
        for u in range(n_users):
            s.add(f"s{u}_0", 900, 40)
        a_added = []
        def refill(sim):
            if not a_added and sim.clk.t >= 8.0:
                a_added.append(sim.add("A", 45000, 1))
            if a_added and a_added[0].t_prefill_done is not None and sim.clk.t > a_added[0].t_prefill_done + 30:
                sim.g.active_jobs.clear(); return
            new_short(sim)
        s.run(400, refill=refill)
        return s, a_added[0]

    def test_order_shorts_between_chunks_and_exclusive_big_chunk(self):
        s, a = self.stream(True)
        a_steps = [i for i, (_, _, pre, _) in enumerate(s.log) if "A" in pre]
        self.assertGreaterEqual(len(a_steps), 11)
        first = a_steps[0]
        for i in a_steps:                                   # a leader chunk never shares its step with other work
            _, _, pre, dec = s.log[i]
            self.assertEqual(pre, ["A"], s.log[i]); self.assertEqual([d for d in dec if d != "A"], [], s.log[i])
        served_between = [i for i in range(first, a_steps[-1]) if i not in a_steps and (s.log[i][2] or s.log[i][3])]
        self.assertTrue(served_between, "no short step between A chunks")
        # at least one short step sits between two A chunks (interleave), and one before the first chunk (shorts first)
        self.assertTrue(any(a_steps[k] + 1 < a_steps[k + 1] for k in range(len(a_steps) - 1)))
        self.assertTrue(all(s.log[i][2] != ["A"] for i in range(0, 1)) or first > 0)
        self.assertGreaterEqual(s.g._cil.finished[0]["slices"], 2)

    def test_max_deferral_bound_and_ttft_within_5pct(self):
        solo = solo_big_ttft()
        self.assertAlmostEqual(solo, 35.8, delta=0.05)
        s, a = self.stream(True, n_users=3, max_defer_s=1.5)
        ttft = a.t_prefill_done - a.t_arrive
        deferral = sum(t1 - t0 for t0, t1, pre, _ in s.log if t0 >= a.t_arrive and t1 <= a.t_prefill_done and "A" not in pre)
        self.assertLessEqual(deferral, 1.5 + 0.2, deferral)             # hard cap + estimate error of one step
        self.assertLessEqual(ttft, solo * 1.05, (ttft, solo))
        st = s.g._cil.finished
        self.assertTrue(st and st[0]["deferred_s"] <= 1.5 + 0.2)

    def test_no_unbounded_starvation_with_endless_shorts_and_tiny_budget(self):
        for budget in (0.0, 0.3):
            s, a = self.stream(True, n_users=4, max_defer_s=budget)
            self.assertIsNotNone(a.t_prefill_done)
            self.assertLessEqual(a.t_prefill_done - a.t_arrive, 35.8 * 1.0 + budget + 0.25)

    def test_budget_zero_is_fifo(self):
        s, a = self.stream(True, n_users=2, max_defer_s=0.0)
        self.assertAlmostEqual(a.t_prefill_done - a.t_arrive, solo_big_ttft(), delta=0.01)

    def test_chunk_ranges_equal_non_interleaved_and_reconstruct_tokens(self):
        ref, aref = self.stream(False, n_users=2)
        il, ail = self.stream(True, n_users=2)
        byname = lambda sim: {j.name: j.ranges for j in sim.all}
        r, i = byname(ref), byname(il)
        self.assertEqual(aref.ranges, ail.ranges)
        for name in set(r) & set(i):                         # every request both runs started: same chunk calls
            n = next(j.n for j in il.all if j.name == name)
            if il.all[[j.name for j in il.all].index(name)].t_prefill_done is None:
                continue
            self.assertEqual(r[name][:len(i[name])], i[name]) if len(i[name]) <= len(r[name]) else self.assertEqual(r[name], i[name][:len(r[name])])
        for j in il.all:
            if j.t_prefill_done is None:
                continue
            flat = [x for a_, b_ in j.ranges for x in range(a_, b_)]
            self.assertEqual(flat, list(range(0, j.n - 1)))  # contiguous cover of the prompt rows, no gap, no overlap
            self.assertTrue(all(a_ % PAGE == 0 for a_, _ in j.ranges))

    def test_single_batch1_decode_while_interleaving(self):
        s, a = self.stream(True, n_users=4)
        for t0, t1, pre, dec in s.log:
            if a.t_arrive <= t0 and t1 <= a.t_prefill_done:
                self.assertLessEqual(len(dec), 1, (t0, pre, dec))

    def test_single_user_path_unchanged(self):
        ref = Sim(False); ra = ref.add("A", 45000, 5); ref.run(1e9)
        il = Sim(True); ia = il.add("A", 45000, 5); il.run(1e9)
        self.assertEqual(ra.ranges, ia.ranges)
        self.assertEqual((ra.t_prefill_done, ref.clk.t), (ia.t_prefill_done, il.clk.t))
        self.assertEqual([x[2:] for x in ref.log], [x[2:] for x in il.log])
        self.assertIsNone(il.g._cil._lead)

    def test_cached_conversation_turns_are_shorts_not_bigs(self):
        # 20.7K-row prompts with 19.2K rows behind a tail checkpoint (~1.5K new rows): classified from the peek, they must run
        # in the slices while the 45K prefill is parked, not wait behind it (the GPU run: C TTFT 36 s, B stalled 27 s)
        def run(peek):
            s = Sim(True, max_defer_s=8.0)
            if not peek:
                for j in (): pass
            a = s.add("A", 45000, 1)
            bs = [s.add(f"B{k}", 20700, 30, resume=19200) for k in range(2)]
            if not peek:
                for b in bs: b._tail_peek = lambda seq: 0       # the old behaviour: cursor before restore
            s.run(1e9)
            return s, a, bs
        s, a, bs = run(True)
        self.assertLess(max(b.t_prefill_done for b in bs), 8.0)          # served inside the budgeted slices, before A's chunks
        self.assertLessEqual(a.t_prefill_done, solo_big_ttft() + 8.0 + 0.3)
        s0, a0, bs0 = run(False)
        self.assertGreater(min(b.t_prefill_done for b in bs0), 30.0)    # without the peek they are parked behind A for good

    def test_budget_scales_with_the_big_prefill(self):
        # four shorts present when an 8K prefill (6.5 s) starts: the fixed 8 s cap would let them defer it by 4 s, 18 % is ~1.2 s,
        # which still takes the oldest short(s) whose rows fit (a slice runs a subset, not all-or-nothing)
        solo = serve(False, [], bigs=((0.0, 8192),)).all[0].time_first_token
        burst = [0.0, 0.0, 0.0, 0.0]
        a = next(j for j in serve(True, burst, bigs=((0.0, 8192),), cap=5).all if j.name == "A0")
        self.assertLessEqual(a.time_first_token, solo * 1.18 + 0.8, (a.time_first_token, solo))
        s8 = serve(True, burst, bigs=((0.0, 8192),), cap=5, defer_frac=0.0, max_defer_s=8.0)
        a8 = next(j for j in s8.all if j.name == "A0")
        self.assertGreater(a8.time_first_token, a.time_first_token + 1.0)        # fixed seconds hurt the small big much more
        sh = next(j for j in serve(True, burst, bigs=((0.0, 8192),), cap=5).all if j.name == "s0")
        self.assertLess(sh.time_first_token, a.t_prefill_done)                   # the oldest short got its token before the big one

    def test_equal_bigs_are_served_in_arrival_order(self):
        off = serve(False, [], bigs=tuple((0.0, 8192) for _ in range(4)), cap=4)
        on = serve(True, [], bigs=tuple((0.0, 8192) for _ in range(4)), cap=4)
        f_off, f_on = first_tokens(off, "A"), first_tokens(on, "A")
        self.assertGreater(f_off[0], 0.95 * f_off[-1])                           # round robin: everyone waits for everyone
        self.assertEqual([j.name for j in sorted(on.all, key=lambda j: j.t_prefill_done)], ["A0", "A1", "A2", "A3"])
        self.assertLess(f_on[1], 0.6 * f_off[1] + 0.01 * f_off[-1] + f_on[0])   # median 37 % earlier
        self.assertLess((f_on[1] + f_on[2]) / 2, 0.7 * (f_off[1] + f_off[2]) / 2)
        self.assertAlmostEqual(f_on[-1], f_off[-1], delta=0.1)                  # the last one pays nothing
        self.assertEqual({j.name: j.ranges for j in off.all}, {j.name: j.ranges for j in on.all})   # same chunk calls per job

    def test_default_budget_costs_the_big_less_with_the_same_short_first_tokens(self):
        solo = solo_big_ttft()
        for shorts_t in ([8.5, 9, 9.5, 10, 25, 30, 34], [8.5 + 6 * k for k in range(6)]):
            fixed = serve(True, shorts_t, defer_frac=0.0, max_defer_s=8.0)
            dflt = serve(True, shorts_t)
            cf = first_tokens(fixed, "A")[0] / solo - 1
            cd = first_tokens(dflt, "A")[0] / solo - 1
            self.assertLess(cd, cf - 0.03, (cd, cf))
            self.assertLessEqual(cd, 0.18)
            ff, fd = first_tokens(fixed, "s"), first_tokens(dflt, "s")
            self.assertLessEqual(sum(fd) / len(fd), sum(ff) / len(ff) * 1.03)
            self.assertLessEqual(fd[-1], ff[-1] * 1.03)

    def test_batched_decode_option_is_off_by_default(self):
        def max_batch(**kw):
            s = serve(True, [8.5, 9, 9.5, 10], **kw)
            a = next(j for j in s.all if j.name == "A0")                         # only while the big prefill is in progress
            return max(len(d) for t0, t1, pre, d in s.log if t1 < a.t_prefill_done and "A0" not in pre and any(x.startswith("s") for x in d))
        self.assertEqual(max_batch(), 1)
        self.assertGreater(max_batch(single_decode=False), 1)

    def test_flag_off_by_default(self):
        os.environ.pop("EXL3_SCHED_CHUNK_IL", None)
        self.assertFalse(chunk_il.chunk_il_enabled())
        s = Sim(False)
        a, b = s.add("A", 45000, 1), s.add("B", 900, 2)
        s.g._cil_hold(); s.g._cil_release()                 # no plan, no effect
        self.assertFalse(a._cil_hold or b._cil_hold)

    def test_two_bigs_fifo(self):
        s = Sim(True)
        a, b = s.add("A", 45000, 1), s.add("B", 40000, 1)
        s.add("c", 500, 3)
        s.run(1e9)
        self.assertLess(a.t_prefill_done, b.ranges and b.t_prefill_done)
        first_b = next(i for i, x in enumerate(s.log) if "B" in x[2])
        last_a = max(i for i, x in enumerate(s.log) if "A" in x[2])
        self.assertGreater(first_b, last_a)                 # B starts only after A's prefill is done

    def test_steps_without_company_are_not_charged_to_the_leader(self):
        def late_short_ttft(patch_solo):
            s = Sim(True)
            if patch_solo:
                s.g._cil.solo = lambda: None
            s.add("A", 45000, 1); s.add("s0", 900, 40)         # s0 leaves while A is the leader; A then runs alone for ~19 s
            s.run(1e9, arrivals=[(20.0, "s1", 900, 40)])
            late = next(j for j in s.all if j.name == "s1")
            return late.time_first_token - 20.0
        self.assertLess(late_short_ttft(False), 8.0)             # a slice is granted: one leader chunk (~3 s) + the short itself
        self.assertGreater(late_short_ttft(True), 12.0)          # without the re-base the solo time reads as deferral: strict FIFO

    def test_planner_error_disables_interleave(self):
        s = Sim(True)
        s.g._cil.plan_prefill = lambda v: 1 / 0
        a, b = s.add("A", 45000, 1), s.add("B", 900, 2)
        import io, contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            s.run(1e9)
        self.assertTrue(s.g._cil_broken)
        self.assertEqual(a.ranges[0], (0, 4096))


if __name__ == "__main__":
    unittest.main(verbosity=2)
