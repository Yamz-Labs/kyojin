#!/usr/bin/env python3
"""Concurrent prefill of one shared prefix (exllamav3/generator/prefill_share.py + Generator._prefill_leader / iterate_start_jobs).
Pure python: the real helper and the real Generator glue (loaded by ast) on fake jobs. Proves: four jobs starting together on one
prefix start one leader and park three followers; a follower starts once the leader has prefilled the shared run (not before); a
job on another prefix or with a short shared run is never held; a finished leader releases everyone; EXL3_PF_SHARE=0 is the old
behaviour (all four start at once)."""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prefill_share", ROOT / "exllamav3" / "generator" / "prefill_share.py")
prefill_share = importlib.util.module_from_spec(spec)
sys.modules["prefill_share"] = prefill_share
spec.loader.exec_module(prefill_share)
PAGE = 256


def glue():
    src = (ROOT / "exllamav3" / "generator" / "generator.py").read_text()
    ns = {"os": os, "prefill_share": prefill_share, "PAGE_SIZE": PAGE, "_PF_PROF": None}   # _PF_PROF: the EXL3_PF_PROF diagnostic is off
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name in ("_prefill_leader", "iterate_start_jobs", "_prefill_round"):
            exec(textwrap.dedent(ast.get_source_segment(src, n)), ns)
    return ns["_prefill_leader"], ns["iterate_start_jobs"], ns["_prefill_round"]


class Seq:
    def __init__(self, hashes, kv=0):
        self.page_hashes, self.kv_position = hashes, kv


class FakeJob:
    skips, max_skips = 0, 4

    def __init__(self, hashes, kv=0):
        self.sequences = [Seq(hashes, kv)]
        self.done = False

    def is_prefill_done(self):
        return self.done

    def current_new_pages_required(self):
        return 1

    def activate(self):
        pass

    def allocate_pages(self):
        pass

    serial_number, identifier = 0, None


class FakeGen:
    max_batch_size = 8

    def __init__(self, pending):
        self.pending_jobs, self.active_jobs = list(pending), []
        self.pagetable = types.SimpleNamespace(num_unreferenced_pages=lambda: 100)

    _prefill_leader, iterate_start_jobs, _prefill_round = glue()

    def reap_failed_job(self, job, e, results):
        self.active_jobs.remove(job)


def chain(n, tag="p"):
    return [f"{tag}{i}" for i in range(n)]


class PJob(FakeJob):
    """prefill() advances `rows` of `need`; done once it got them all."""
    def __init__(self, need, rows=1, fail=False):
        super().__init__([])
        self.need, self.have, self.rows, self.fail, self.ran = need, 0, rows, fail, 0

    def is_prefill_done(self):
        return self.have >= self.need

    def prefill(self, results):
        if self.fail:
            raise RuntimeError("boom")
        self.ran += 1
        self.have += self.rows


class T(unittest.TestCase):
    def setUp(self):
        os.environ.pop("EXL3_PF_SHARE", None)

    def test_four_on_one_prefix_start_one_leader(self):
        pre = chain(90)
        g = FakeGen([FakeJob(pre + [f"t{i}"]) for i in range(4)])
        g.iterate_start_jobs([])
        self.assertEqual(len(g.active_jobs), 1)
        self.assertEqual(len(g.pending_jobs), 3)

    def test_follower_starts_when_leader_passed_the_shared_run(self):
        pre = chain(90)
        g = FakeGen([FakeJob(pre + ["a"] * 3), FakeJob(pre + ["b"] * 3)])
        g.iterate_start_jobs([])
        lead = g.active_jobs[0]
        lead.sequences[0].kv_position = 89 * PAGE
        g.iterate_start_jobs([])
        self.assertEqual(len(g.active_jobs), 1)
        lead.sequences[0].kv_position = 90 * PAGE
        g.iterate_start_jobs([])
        self.assertEqual(len(g.active_jobs), 2)

    def test_unrelated_and_short_runs_are_not_held(self):
        pre = chain(90)
        g = FakeGen([FakeJob(pre + ["a"]), FakeJob(chain(90, "q")), FakeJob(pre[:3] + ["z"] * 5)])
        g.iterate_start_jobs([])
        self.assertEqual(len(g.active_jobs), 3)

    def test_finished_leader_releases(self):
        pre = chain(90)
        g = FakeGen([FakeJob(pre + ["a"]), FakeJob(pre + ["b"])])
        g.iterate_start_jobs([])
        g.active_jobs[0].done = True
        g.iterate_start_jobs([])
        self.assertEqual(len(g.active_jobs), 2)

    def test_off_switch(self):
        os.environ["EXL3_PF_SHARE"] = "0"
        pre = chain(90)
        g = FakeGen([FakeJob(pre + [f"t{i}"]) for i in range(4)])
        g.iterate_start_jobs([])
        self.assertEqual(len(g.active_jobs), 4)

    def test_follower_does_not_block_unrelated_job_behind_it(self):
        pre = chain(90)
        g = FakeGen([FakeJob(pre + ["a"]), FakeJob(pre + ["b"]), FakeJob(chain(10, "q"))])
        g.iterate_start_jobs([])
        self.assertEqual(len(g.active_jobs), 2)
        self.assertEqual(g.active_jobs[1].sequences[0].page_hashes[0], "q0")

    def _round(self, jobs, parked=()):
        g = FakeGen([])
        g.active_jobs = list(jobs)
        g._prefill_round(parked, [])
        return g

    def test_round_ends_at_first_job_that_finishes_its_prefill(self):
        a, b, c = PJob(1), PJob(1), PJob(1)
        self._round([a, b, c])
        self.assertEqual((a.ran, b.ran, c.ran), (1, 0, 0))

    def test_round_runs_every_job_that_does_not_finish(self):
        a, b, c = PJob(5), PJob(5), PJob(5)
        self._round([a, b, c])
        self.assertEqual((a.ran, b.ran, c.ran), (1, 1, 1))

    def test_round_last_job_finishing_ends_nothing_early(self):
        a, b = PJob(5), PJob(1)
        self._round([a, b])
        self.assertEqual((a.ran, b.ran), (1, 1))

    def test_round_skips_parked_and_failed(self):
        a, b, c = PJob(1), PJob(1), PJob(1)
        g = self._round([a, b, c], parked=(b,))
        self.assertEqual((a.ran, b.ran, c.ran), (1, 0, 0))
        f, d = PJob(2, fail=True), PJob(2)
        g = self._round([f, d])
        self.assertEqual((d.ran, len(g.active_jobs)), (1, 1))

    def test_round_off_switch(self):
        os.environ["EXL3_PF_FIRST_OUT"] = "0"
        try:
            a, b, c = PJob(1), PJob(1), PJob(1)
            self._round([a, b, c])
            self.assertEqual((a.ran, b.ran, c.ran), (1, 1, 1))
        finally:
            del os.environ["EXL3_PF_FIRST_OUT"]


if __name__ == "__main__":
    unittest.main()
