"""Cache-write commitment and request accounting without model/GPU imports.

The interval helper is imported directly. The Job and recurrent-cache observer
methods are compiled from their production ASTs, leaving native initialization
and unrelated inference dependencies out of these CPU contracts.
"""
import ast
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_generator_cache_write_test", ROOT / "exllamav3/generator/cache_write.py")
cache_write = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_write)
Tracker = cache_write.CacheWriteTracker


def load_methods(relative, source_class, methods, name, namespace, base="object"):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    original = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == source_class)
    selected = [node for node in original.body
                if isinstance(node, ast.FunctionDef) and node.name in methods]
    if len(selected) != len(methods):
        raise AssertionError("A tested production observer method disappeared")
    cls = ast.ClassDef(name=name, bases=[ast.Name(id=base, ctx=ast.Load())],
                       keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(ROOT / relative), "exec"), namespace)
    return namespace[name]


_namespace = {"weakref": weakref, "CacheWriteTracker": Tracker, "PAGE_SIZE": 256,
              "ThreadPoolExecutor": ThreadPoolExecutor, "_TAIL_RING": False}
_job_tree = ast.parse((ROOT / "exllamav3/generator/job.py").read_text(encoding="utf-8"))
_shared = next(node for node in _job_tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "_draft_pages_shared")
exec(compile(ast.Module(body=[_shared], type_ignores=[]), "job.py", "exec"), _namespace)
ObservedJob = load_methods(
    "exllamav3/generator/job.py", "Job",
    {"cache_write_ranges", "cache_read_ranges", "cache_write_tokens", "cache_read_tokens",
     "_cache_write_kv_commit", "_cache_write_checkpoint"}, "ObservedJob", _namespace)
ObservedCache = load_methods(
    "exllamav3/cache/recurrent.py", "RecurrentCache",
    {"on_stash_ready", "stash_async", "settle", "_finish"}, "ObservedCache", _namespace, "dict")

# Exercise the actual constructor/requeue transfer expressions, without running
# tokenizers, pinned buffers, model setup or the unrelated Job constructor code.
_job_class = next(node for node in _job_tree.body if isinstance(node, ast.ClassDef) and node.name == "Job")
_constructor = next(node for node in _job_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
_ledger_init = next(node for node in _constructor.body if isinstance(node, ast.Assign) and
                    any(isinstance(target, ast.Attribute) and target.attr == "_cache_writes" for target in node.targets))
_requeue = next(node for node in _job_class.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_for_requeue")
_state = next(node.value for node in _requeue.body if isinstance(node, ast.Assign) and
              any(isinstance(target, ast.Name) and target.id == "rq_state" for target in node.targets))
_ledger_transfer = next(value for key, value in zip(_state.keys, _state.values)
                        if isinstance(key, ast.Constant) and key.value == "cache_writes")
_init_function = ast.parse("def initialize_ledger(self, rq_state):\n    pass\n").body[0]
_init_function.body = [_ledger_init]
_transfer_function = ast.parse("def transfer_ledger(self):\n    pass\n").body[0]
_transfer_function.body = [ast.Return(value=_ledger_transfer)]
exec(compile(ast.fix_missing_locations(ast.Module(body=[_init_function, _transfer_function], type_ignores=[])),
             "job.py", "exec"), _namespace)
ObservedJob.initialize_ledger = _namespace["initialize_ledger"]
ObservedJob.transfer_ledger = _namespace["transfer_ledger"]


class Page:
    def __init__(self, index, rows=256):
        self.page_index, self.kv_position = index, rows
        self.phash, self.prev_hash = "page_" + str(index), None if index == 0 else "page_" + str(index - 1)
        self.wgen = 7


class Table:
    def __init__(self, pages):
        self.all_pages = pages
        self.resumable = True

    def is_resumable(self, key):
        return self.resumable


class ModelGenerator:
    pass


def fixture(limit=256, recurrent=True, rows=256):
    pages = [Page(index) for index in range(max(1, (limit + 255) // 256))]
    pages[-1].kv_position = rows
    table = Table(pages)
    rc = ObservedCache() if recurrent else None
    if rc is not None:
        rc._stash_futures, rc.exact_meta, rc._pending, rc._pin_pool = {}, {}, [], {}
    generator = ModelGenerator()
    generator.recurrent_cache = rc
    generator.draft_model = generator.draft_cache = None
    generator.tail_ckpts, generator.sub_ckpts = [], []
    seq = SimpleNamespace(kv_position=limit, allocated_pages=pages,
                          sequence_ids=list(range(limit)))
    job = ObservedJob()
    job.sequences, job.generator, job.pagetable = [seq], generator, table
    job._cache_writes, job._cache_write_seen, job._cache_write_result = Tracker([limit]), set(), None
    job.rq_cached, job.cached_pages, job.cached_tokens = None, 0, 0
    return job, seq, rc, table


def admitted(job, seq, rc, end=256):
    key = seq.allocated_pages[end // 256 - 1].phash
    entry = {"position": end}
    rc[key] = entry
    return key, entry


def tail_fixture():
    job, seq, rc, table = fixture(limit=300, rows=44)
    stashed = {"position": 300}
    tail = {"prev_hash": "page_0", "ids": SimpleNamespace(shape=(1, 44)),
            "page": seq.allocated_pages[1], "stashed": stashed, "carry": None}
    job.generator.tail_ckpts.append(tail)
    job._cache_writes.compute(0, 0, 300)
    return job, seq, rc, table, stashed, tail


class TrackerTests(unittest.TestCase):
    def test_computation_without_retention_and_retention_without_computation(self):
        tracker = Tracker([16])
        tracker.compute(0, 0, 8)
        self.assertEqual(tracker.count(), 0)
        tracker.commit(0, 8, 16)
        self.assertEqual(tracker.count(), 0)
        self.assertTrue(tracker.pending(0))
        tracker.commit(0, 0, 8)
        self.assertEqual(tracker.count(), 8)
        self.assertFalse(tracker.pending(0))

    def test_disjoint_cached_pages_do_not_hide_an_earlier_new_page(self):
        tracker = Tracker([768])
        tracker.read(0, 256, 512)
        tracker.compute(0, 0, 256)
        tracker.compute(0, 512, 768)
        tracker.commit(0, 0, 768)
        self.assertEqual(tracker.read_ranges(), [(256, 512)])
        self.assertEqual(tracker.write_ranges(), [(0, 256), (512, 768)])
        self.assertEqual(tracker.count(), 512)

    def test_restores_and_copies_do_not_create_writes(self):
        tracker = Tracker([20])
        tracker.read(0, 0, 12)
        tracker.commit(0, 0, 12)
        self.assertEqual(tracker.count(), 0)
        tracker.compute(0, 12, 16)
        tracker.commit(0, 0, 16)
        self.assertEqual(tracker.count(), 4)

    def test_overlapping_checkpoints_and_duplicate_replacement_count_once(self):
        tracker = Tracker([20])
        tracker.compute(0, 0, 20)
        tracker.commit(0, 0, 4, new=False)
        tracker.commit(0, 0, 12)
        tracker.commit(0, 0, 20)
        tracker.commit(0, 0, 20)
        self.assertEqual(tracker.write_ranges(), [(4, 20)])
        self.assertEqual(tracker.count(), 16)

    def test_output_and_healed_replacement_are_outside_effective_input(self):
        tracker = Tracker([11])  # a 12-token input with its healed token removed
        tracker.compute(0, 0, 30)
        tracker.commit(0, 0, 30)
        self.assertEqual(tracker.write_ranges(), [(0, 11)])
        self.assertEqual(tracker.count(), 11)

    def test_requeue_restore_does_not_reclassify_own_writes_as_reads(self):
        tracker = Tracker([12])
        tracker.read(0, 0, 4)
        tracker.compute(0, 4, 12)
        tracker.commit(0, 0, 12)
        tracker.read(0, 0, 40)  # next physical prompt includes generated output
        tracker.compute(0, 12, 40)
        tracker.commit(0, 0, 40)
        self.assertEqual(tracker.read_ranges(), [(0, 4)])
        self.assertEqual(tracker.write_ranges(), [(4, 12)])
        self.assertEqual(tracker.count(), 8)

    def test_per_sequence_mean_matches_existing_usage_convention(self):
        tracker = Tracker([12, 10])
        tracker.compute(0, 0, 12)
        tracker.read(1, 0, 3)
        tracker.compute(1, 3, 10)
        tracker.commit(0, 0, 12)
        tracker.commit(1, 0, 10)
        self.assertEqual(tracker.count(), (12 + 7) // 2)

    def test_concurrent_completion_callbacks_are_idempotent(self):
        tracker = Tracker([1024])
        tracker.compute(0, 0, 1024)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda end: tracker.commit(0, 0, end),
                          [256, 512, 1024, 768, 1024, 512] * 8))
        self.assertEqual(tracker.write_ranges(), [(0, 1024)])
        self.assertEqual(tracker.count(), 1024)


class JobCommitTests(unittest.TestCase):
    def test_actual_constructor_captures_effective_healed_input_length(self):
        job, seq, _, _ = fixture(limit=12, recurrent=False)
        seq.input_ids = list(range(12))
        seq.sequence_ids = list(range(11))
        job.initialize_ledger({})
        self.assertEqual(job._cache_writes.limits, (11,))
        job._cache_write_kv_commit(seq, 0, 20)
        self.assertEqual(job.cache_write_tokens(), 11)

    def test_actual_requeue_transfer_preserves_identity_bounds_and_history(self):
        job, seq, _, _ = fixture(limit=256, recurrent=False)
        job._cache_write_kv_commit(seq, 0, 256)
        previous = job.transfer_ledger()
        seq.sequence_ids.extend([55] * 300)
        job.initialize_ledger({"cache_writes": previous})
        self.assertIs(job._cache_writes, previous)
        self.assertEqual(job._cache_writes.limits, (256,))
        job._cache_write_kv_commit(seq, 256, 556)
        self.assertEqual(job.cache_write_tokens(), 256)

    def test_last_original_input_token_commits_in_first_generation_forward(self):
        job, seq, _, _ = fixture(recurrent=False)
        job._cache_write_kv_commit(seq, 0, 255)
        self.assertEqual(job.cache_write_tokens(), 255)
        job._cache_write_kv_commit(seq, 255, 256)
        self.assertEqual(job.cache_write_tokens(), 256)
        job._cache_write_kv_commit(seq, 256, 300)
        self.assertEqual(job.cache_write_tokens(), 256)

    def test_one_partial_row_is_not_yet_reusable(self):
        job, seq, _, _ = fixture(limit=8, recurrent=False, rows=1)
        job._cache_write_kv_commit(seq, 0, 1)
        self.assertEqual(job.cache_write_tokens(), 0)
        job._cache_write_kv_commit(seq, 1, 2)
        self.assertEqual(job.cache_write_tokens(), 2)

    def test_unshared_draft_pages_retain_only_complete_pages(self):
        job, seq, _, _ = fixture(recurrent=False)
        job.generator.draft_model = object()
        job.generator.draft_cache = object()
        job.generator._draft_pages_shared = False
        job._cache_write_kv_commit(seq, 0, 255)
        self.assertEqual(job.cache_write_tokens(), 0)
        job._cache_write_kv_commit(seq, 255, 256)
        self.assertEqual(job.cache_write_tokens(), 256)

    def test_recurrent_kv_without_retained_checkpoint_is_not_a_write(self):
        job, seq, _, _ = fixture()
        job._cache_write_kv_commit(seq, 0, 256)
        self.assertEqual(job.cache_write_tokens(), 0)

    def test_retained_checkpoint_must_describe_the_claimed_position(self):
        job, seq, rc, _ = fixture()
        key, entry = admitted(job, seq, rc)
        job._cache_writes.compute(0, 0, 256)
        entry["position"] = 255
        job._cache_write_checkpoint(seq, 256, entry, key=key)
        self.assertEqual(job.cache_write_tokens(), 0)
        self.assertFalse(job._cache_write_seen)

    def test_newly_completed_kv_can_make_existing_checkpoint_reusable(self):
        job, seq, rc, _ = fixture()
        admitted(job, seq, rc)
        job._cache_write_kv_commit(seq, 0, 256)
        self.assertEqual(job.cache_write_tokens(), 256)

    def test_checkpoint_must_still_be_admitted_when_async_copy_finishes(self):
        job, seq, rc, _ = fixture()
        key, entry = admitted(job, seq, rc)
        future = Future()
        rc._stash_futures[id(entry)] = (entry, future)
        job._cache_write_kv_commit(seq, 0, 256)
        self.assertEqual(job.cache_write_tokens(), 0)
        del rc[key]
        future.set_result(None)
        self.assertEqual(job.cache_write_tokens(), 0)

    def test_orphan_failure_does_not_suppress_retry_after_anchor_repair(self):
        job, seq, rc, table = fixture()
        key, entry = admitted(job, seq, rc)
        table.resumable = False
        job._cache_write_kv_commit(seq, 0, 256)
        self.assertEqual(job.cache_write_tokens(), 0)
        self.assertFalse(job._cache_write_seen)
        table.resumable = True
        job._cache_write_checkpoint(seq, 256, entry, key=key)
        self.assertEqual(job.cache_write_tokens(), 256)

    def test_future_pending_after_event_popped_does_not_credit_early(self):
        job, seq, rc, _ = fixture()
        key, entry = admitted(job, seq, rc)
        future = Future()
        rc._stash_futures[id(entry)] = (entry, future)
        job._cache_write_kv_commit(seq, 0, 256)
        self.assertNotIn("_ev", entry)  # _finish can pop it before its clones finish
        self.assertEqual(job.cache_write_tokens(), 0)
        future.set_result(None)
        self.assertEqual(job.cache_write_tokens(), 256)
        job._cache_write_checkpoint(seq, 256, entry, key=key)
        self.assertEqual(job.cache_write_tokens(), 256)

    def test_failed_async_copy_never_credits(self):
        job, seq, rc, _ = fixture()
        _, entry = admitted(job, seq, rc)
        future = Future()
        rc._stash_futures[id(entry)] = (entry, future)
        job._cache_write_kv_commit(seq, 0, 256)
        future.set_exception(RuntimeError("copy failed"))
        self.assertEqual(job.cache_write_tokens(), 0)

    def test_exact_checkpoint_checks_creation_page_generations(self):
        job, seq, rc, _ = fixture()
        key, entry = admitted(job, seq, rc)
        rc.exact_meta[key] = {"eid": id(entry), "sig": (0, 256), "pw": (99,)}
        job._cache_writes.compute(0, 0, 256)
        job._cache_write_checkpoint(seq, 256, entry, key=key)
        self.assertEqual(job.cache_write_tokens(), 0)
        rc.exact_meta[key]["pw"] = (7,)
        seq.sequence_ids.extend([88] * 20)  # decode growth must not change original proof
        job._cache_write_checkpoint(seq, 256, entry, key=key)
        self.assertEqual(job.cache_write_tokens(), 256)

    def test_exact_checkpoint_rewrite_during_pending_copy_invalidates_credit(self):
        job, seq, rc, _ = fixture()
        key, entry = admitted(job, seq, rc)
        rc.exact_meta[key] = {"eid": id(entry), "sig": (0, 256), "pw": (7,)}
        future = Future()
        rc._stash_futures[id(entry)] = (entry, future)
        job._cache_write_kv_commit(seq, 0, 256)
        seq.allocated_pages[0].wgen += 1
        future.set_result(None)
        self.assertEqual(job.cache_write_tokens(), 0)
        self.assertFalse(job._cache_write_seen)

    def test_async_credit_does_not_keep_job_generator_or_live_state_alive(self):
        job, seq, rc, _ = fixture()
        _, entry = admitted(job, seq, rc)
        future = Future()
        rc._stash_futures[id(entry)] = (entry, future)
        job.recurrent_state = ModelGenerator()
        observed = (weakref.ref(job), weakref.ref(job.generator), weakref.ref(job.recurrent_state))
        job._cache_write_kv_commit(seq, 0, 256)
        del job
        gc.collect()
        self.assertTrue(all(reference() is None for reference in observed))
        future.set_result(None)


class TailCommitTests(unittest.TestCase):
    def test_retained_tail_covers_computed_input_only(self):
        job, seq, _, _, stashed, tail = tail_fixture()
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        self.assertEqual(job.cache_write_tokens(), 300)

    def test_deferred_draft_tail_is_not_credited_until_reusable(self):
        job, seq, _, _, stashed, tail = tail_fixture()
        job.generator.draft_model = job.generator.draft_cache = object()
        job.generator._draft_pages_shared = True
        tail["draft_ok"] = False
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        self.assertEqual(job.cache_write_tokens(), 0)
        tail["draft_ok"] = True
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        self.assertEqual(job.cache_write_tokens(), 300)

    def test_unshared_draft_partial_tail_needs_ring_opt_in_or_full_page(self):
        job, seq, _, _, stashed, tail = tail_fixture()
        job.generator.draft_model = job.generator.draft_cache = object()
        job.generator._draft_pages_shared = False
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        self.assertEqual(job.cache_write_tokens(), 0)
        tail["page"].kv_position = 256
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        self.assertEqual(job.cache_write_tokens(), 300)

    def test_ring_opt_in_allows_target_only_partial_tail_reuse(self):
        job, seq, _, _, stashed, tail = tail_fixture()
        job.generator.draft_model = job.generator.draft_cache = object()
        job.generator._draft_pages_shared = False
        _namespace["_TAIL_RING"] = True
        try:
            job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        finally:
            _namespace["_TAIL_RING"] = False
        self.assertEqual(job.cache_write_tokens(), 300)

    def test_tail_source_defragmentation_does_not_change_valid_proof(self):
        job, seq, rc, table, stashed, tail = tail_fixture()
        future = Future()
        rc._stash_futures[id(stashed)] = (stashed, future)
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        self.assertEqual(job.cache_write_tokens(), 0)
        root, partial = table.all_pages
        table.all_pages[:] = [partial, root]
        partial.page_index, root.page_index = 0, 1
        future.set_result(None)
        self.assertEqual(job.cache_write_tokens(), 300)

    def test_evicted_tail_entry_cannot_credit_pending_copy(self):
        job, seq, rc, _, stashed, tail = tail_fixture()
        future = Future()
        rc._stash_futures[id(stashed)] = (stashed, future)
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        job.generator.tail_ckpts.clear()
        future.set_result(None)
        self.assertEqual(job.cache_write_tokens(), 0)

    def test_tail_page_rewrite_cannot_credit_pending_copy(self):
        job, seq, rc, _, stashed, tail = tail_fixture()
        future = Future()
        rc._stash_futures[id(stashed)] = (stashed, future)
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        tail["page"].wgen += 1
        future.set_result(None)
        self.assertEqual(job.cache_write_tokens(), 0)

    def test_tail_source_must_still_hold_the_required_rows(self):
        job, seq, _, _, stashed, tail = tail_fixture()
        tail["page"].kv_position = 43
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        self.assertEqual(job.cache_write_tokens(), 0)

    def test_tail_orphan_can_be_retried_after_anchor_repair(self):
        job, seq, _, table, stashed, tail = tail_fixture()
        table.resumable = False
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        self.assertEqual(job.cache_write_tokens(), 0)
        table.resumable = True
        job._cache_write_checkpoint(seq, 300, stashed, tail=tail)
        self.assertEqual(job.cache_write_tokens(), 300)


class ReadyCallbackTests(unittest.TestCase):
    def asynchronous(self):
        cache = ObservedCache()
        cache._pending, cache._pin_pool, cache._stash_futures = [], {}, {}
        future = Future()
        cache._worker = SimpleNamespace(submit=lambda *args: future)
        stashed = {"_ev": object()}
        state = SimpleNamespace(stash=lambda **kwargs: stashed)
        self.assertIs(cache.stash_async(state), stashed)
        return cache, future, stashed

    def test_existing_finisher_success_cleans_metadata_and_notifies_once(self):
        cache, future, stashed = self.asynchronous()
        observed = []
        cache.on_stash_ready(stashed, lambda: observed.append("ready"))
        stashed.pop("_ev")
        self.assertEqual(observed, [])
        future.set_result(None)
        self.assertEqual(observed, ["ready"])
        self.assertFalse(cache._stash_futures)
        self.assertNotIn("_cache_write_failed", stashed)

    def test_failure_is_marked_before_metadata_removal_and_stays_uncredited(self):
        cache, future, stashed = self.asynchronous()
        observed = []
        cache.on_stash_ready(stashed, lambda: observed.append("ready"))
        stashed.pop("_ev")
        future.set_exception(RuntimeError("host clone failed"))
        self.assertEqual(observed, [])
        self.assertTrue(stashed["_cache_write_failed"])
        self.assertFalse(cache._stash_futures)
        cache.on_stash_ready(stashed, lambda: observed.append("late"))
        self.assertEqual(observed, [])

    def test_cancelled_finisher_is_not_success(self):
        cache, future, stashed = self.asynchronous()
        observed = []
        cache.on_stash_ready(stashed, lambda: observed.append("ready"))
        future.cancel()
        self.assertEqual(observed, [])
        self.assertTrue(stashed["_cache_write_failed"])
        self.assertFalse(cache._stash_futures)


if __name__ == "__main__":
    unittest.main()
