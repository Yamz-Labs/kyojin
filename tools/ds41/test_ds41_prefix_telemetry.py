"""Measured logical input writes through the native retained RAM prefix store."""
import importlib.util
from pathlib import Path
import sys
from unittest import TestCase
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

# Store tests need real torch for the module contract, but no package initializer
# or native extension. Loading this file directly keeps them CPU-only.
_path = Path(__file__).resolve().parents[2] / "exllamav3" / "dsv41" / "prefix_cache.py"
_spec = importlib.util.spec_from_file_location("_ds41_prefix_cache_telemetry", _path)
pc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pc
_spec.loader.exec_module(pc)


def snap(keys, kind="end", nbytes=32):
    return pc.Snapshot(list(keys), len(keys), kind, {}, None, nbytes)


class PrefixWriteTests(TestCase):
    def setUp(self):
        self.memory = patch.object(pc, "mem_available_gib", return_value=100.0)
        self.memory.start()
        self.addCleanup(self.memory.stop)
        self.store = pc.PrefixStore(1024, min_free_gib=1.0)
        self.keys = list(range(12))
        self.tracked = pc.RequestPrefixStore(self.store, self.keys)

    def test_put_reports_new_replacement_and_refused_without_changing_retention(self):
        first = snap(self.keys)
        self.assertIs(self.store.put(first), True)
        replacement = snap(self.keys, kind="grid")
        self.assertIs(self.store.put(replacement), False)
        self.assertEqual(self.store.entries, [replacement])
        self.assertGreater(replacement.tick, first.tick)
        self.assertIsNone(self.store.put(snap([8], nbytes=1025)))
        self.assertEqual(self.store.entries, [replacement])

    def test_overlapping_checkpoints_count_each_input_position_once(self):
        for n in (4, 8, 12):
            self.assertIs(self.tracked.put(snap(self.keys[:n])), True)
            self.assertEqual(self.tracked.cache_write_tokens, n)
        self.assertEqual(sum(e.n for e in self.store.entries), 24)
        self.assertEqual(len(self.tracked), len(self.store))
        self.assertEqual(self.tracked.nbytes(), self.store.nbytes())

    def test_actual_lookup_is_read_baseline(self):
        self.store.put(snap(self.keys[:8]))
        restored = self.tracked.lookup(self.keys)
        self.assertEqual(restored.n, 8)
        self.assertEqual(self.tracked.cache_write_tokens, 0)
        self.tracked.put(snap(self.keys))
        self.assertEqual(self.tracked.cache_write_tokens, 4)

    def test_failed_lookup_does_not_infer_reuse_from_a_common_prefix(self):
        self.store.put(snap([0, 1, 2, 99]))
        self.assertIsNone(self.tracked.lookup(self.keys))
        self.assertEqual(self.tracked.best_lcp(self.keys), (3, 4))
        self.tracked.put(snap(self.keys))
        self.assertEqual(self.tracked.cache_write_tokens, 12)

    def test_exact_duplicate_is_not_credited_in_a_longer_checkpoint(self):
        self.store.put(snap(self.keys[:8]))
        # Models overlapping requests: another request saved after our lookup.
        self.assertIs(self.tracked.put(snap(self.keys[:8])), False)
        self.assertEqual(self.tracked.cache_write_tokens, 0)
        self.tracked.put(snap(self.keys))
        self.assertEqual(self.tracked.cache_write_tokens, 4)

    def test_same_prompt_end_replacement_has_no_write_credit(self):
        for n in (4, 8, 12):
            self.store.put(snap(self.keys[:n]))
        self.assertEqual(self.tracked.lookup(self.keys).n, 8)
        self.assertIs(self.tracked.put(snap(self.keys)), False)
        self.assertEqual(self.tracked.cache_write_tokens, 0)

    def test_reply_snapshot_counts_only_original_input_positions(self):
        self.tracked.put(snap(self.keys[:8]))
        reply = snap(self.keys + [21, 22, 23], kind="reply")
        self.tracked.put(reply)
        self.assertEqual(self.tracked.cache_write_tokens, len(self.keys))
        self.tracked.put(snap(self.keys + [21, 22, 23, 24], kind="reply"))
        self.assertEqual(self.tracked.cache_write_tokens, len(self.keys))

    def test_reply_can_be_first_successful_retention_without_counting_output(self):
        with patch.object(pc, "mem_available_gib", return_value=0.0):
            self.assertIsNone(self.tracked.put(snap(self.keys)))
        self.assertEqual(self.tracked.cache_write_tokens, 0)
        self.tracked.put(snap(self.keys + [55, 56], kind="reply"))
        self.assertEqual(self.tracked.cache_write_tokens, len(self.keys))

    def test_memory_and_byte_refusal_do_not_credit_attempts(self):
        with patch.object(pc, "mem_available_gib", return_value=0.0):
            self.assertIsNone(self.tracked.put(snap(self.keys)))
        self.assertIsNone(self.tracked.put(snap(self.keys, nbytes=1025)))
        self.assertEqual(self.tracked.cache_write_tokens, 0)
        self.assertFalse(self.store.entries)

    def test_disk_queue_without_ram_admission_is_not_a_measured_write(self):
        with patch.object(self.store, "save", return_value=True) as queued:
            self.assertIsNone(self.tracked.put(snap(self.keys, nbytes=1025)))
        queued.assert_called_once()
        self.assertEqual(self.tracked.cache_write_tokens, 0)

    def test_disk_lookup_readmission_is_a_read_not_a_write(self):
        restored = snap(self.keys[:8])
        self.store.disk_index = {"existing": {}}
        with patch.object(self.store, "_disk_lookup", return_value=restored):
            self.assertIs(self.tracked.lookup(self.keys), restored)
        self.assertIn(restored, self.store.entries)
        self.assertEqual(self.tracked.cache_write_tokens, 0)
        self.tracked.put(snap(self.keys))
        self.assertEqual(self.tracked.cache_write_tokens, 4)

    def test_persistence_failure_preserves_ram_admission_and_accounting(self):
        with patch.object(self.store, "save", side_effect=OSError("disk unavailable")):
            self.assertIs(self.tracked.put(snap(self.keys)), True)
        self.assertEqual(self.tracked.cache_write_tokens, 12)

    def test_failed_put_does_not_credit_an_unfinished_write(self):
        with patch.object(self.store, "put", side_effect=RuntimeError("store failed")):
            with self.assertRaisesRegex(RuntimeError, "store failed"):
                self.tracked.put(snap(self.keys))
        self.assertEqual(self.tracked.cache_write_tokens, 0)

    def test_later_eviction_does_not_erase_completed_write(self):
        self.store.max_entries = 1
        self.tracked.put(snap(self.keys[:8]))
        self.tracked.put(snap(self.keys))
        self.assertEqual(len(self.store), 1)
        self.assertEqual(self.tracked.cache_write_tokens, 12)
        self.store.clear()
        self.assertEqual(self.tracked.cache_write_tokens, 12)

    def test_unrelated_snapshot_does_not_count_as_this_requests_input(self):
        self.tracked.put(snap([99] * 12))
        self.assertEqual(self.tracked.cache_write_tokens, 0)
        self.tracked.put(snap(self.keys))
        self.assertEqual(self.tracked.cache_write_tokens, 12)
