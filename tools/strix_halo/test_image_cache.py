#!/usr/bin/env python3
"""Image embedding cache tests: CPU only, a fake vision tower, no model, no GPU."""
from __future__ import annotations

import itertools
import os
import sys
import threading
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import image_cache
from image_cache import ImageEmbeddingCache


class FakeEmb:
    """Stands in for MMEmbedding: ids come from a counter, like the engine's global allocator."""
    _next = itertools.count(1_000_000_000, 1000)

    def __init__(self, rows: int, hidden: int = 8, deepstack: int = 0):
        self.first_index = next(self._next)
        self.text_alias = f"<$EMB_{self.first_index}$>"
        self.embeddings = torch.zeros(rows, hidden, dtype=torch.half)
        self.deepstack_embeddings = [torch.zeros(rows, hidden, dtype=torch.half) for _ in range(deepstack)] or None


class FakeTower:
    def __init__(self, rows=4, deepstack=0):
        self.calls, self.rows, self.deepstack = 0, rows, deepstack

    def __call__(self, image):
        self.calls += 1
        return FakeEmb(self.rows, deepstack=self.deepstack)


class ImageCacheTests(unittest.TestCase):
    def test_digest_is_stable_and_content_based(self):
        self.assertEqual(image_cache.digest(b"abc"), image_cache.digest(b"abc"))
        self.assertNotEqual(image_cache.digest(b"abc"), image_cache.digest(b"abd"))

    def test_same_image_same_object_across_requests(self):
        cache, tower = ImageEmbeddingCache(), FakeTower()
        a1 = cache.get_many([("img-a", "a")], tower)[0]
        a2 = cache.get_many([("img-a", "a")], tower)[0]
        self.assertIs(a1, a2)
        self.assertEqual((a1.first_index, a1.text_alias), (a2.first_index, a2.text_alias))
        self.assertEqual(tower.calls, 1)

    def test_different_image_different_ids(self):
        cache, tower = ImageEmbeddingCache(), FakeTower()
        a, b = cache.get_many([("x", "a"), ("y", "b")], tower)
        self.assertNotEqual(a.first_index, b.first_index)
        self.assertNotEqual(a.text_alias, b.text_alias)
        self.assertEqual(tower.calls, 2)

    def test_history_turn_keeps_earlier_ids(self):
        cache, tower = ImageEmbeddingCache(), FakeTower()
        t1 = cache.get_many([("s1", "k1")], tower)
        t2 = cache.get_many([("s1", "k1"), ("s2", "k2")], tower)
        t3 = cache.get_many([("s1", "k1"), ("s2", "k2"), ("s3", "k3")], tower)
        self.assertEqual([e.first_index for e in t2[:1]], [e.first_index for e in t1])
        self.assertEqual([e.first_index for e in t3[:2]], [e.first_index for e in t2])
        self.assertEqual(tower.calls, 3)

    def test_same_image_twice_in_one_prompt_is_two_stable_objects(self):
        cache, tower = ImageEmbeddingCache(), FakeTower()
        first = cache.get_many([("p", "k"), ("p", "k")], tower)
        self.assertIsNot(first[0], first[1])
        self.assertNotEqual(first[0].first_index, first[1].first_index)
        again = cache.get_many([("p", "k"), ("p", "k")], tower)
        self.assertIs(first[0], again[0])
        self.assertIs(first[1], again[1])

    def test_eviction_by_count_drops_least_recently_used(self):
        cache, tower = ImageEmbeddingCache(max_items=2), FakeTower()
        a = cache.get_many([("a", "a")], tower)[0]
        cache.get_many([("b", "b")], tower)
        cache.get_many([("a", "a")], tower)          # a is now the most recent
        cache.get_many([("c", "c")], tower)          # evicts b
        self.assertEqual(len(cache), 2)
        self.assertIs(cache.get_many([("a", "a")], tower)[0], a)
        calls = tower.calls
        cache.get_many([("b", "b")], tower)
        self.assertEqual(tower.calls, calls + 1)

    def test_the_default_policy_is_the_lru_a_server_always_had(self):
        self.assertEqual(image_cache.DEFAULT_POLICY, "head")
        self.assertEqual(ImageEmbeddingCache().policy, "head")     # nothing asked for: the policy upstream had
        cache, tower = ImageEmbeddingCache(max_items=2, max_bytes=0), FakeTower()
        cache.get_many([("a", "a"), ("b", "b")], tower)            # a used, then b
        cache.get_many([("b", "b"), ("c", "c")], tower)            # b again, then c: a is the oldest use now
        calls = tower.calls
        cache.get_many([("a", "a")], tower)                        # ... so a was the one given up
        self.assertEqual(tower.calls, calls + 1)
        calls = tower.calls
        cache.get_many([("c", "c")], tower)
        self.assertEqual(tower.calls, calls)                       # c survived: no encode

    def test_the_policy_can_be_asked_for_by_the_environment(self):
        os.environ[image_cache.POLICY_ENV] = "tail"
        self.addCleanup(os.environ.pop, image_cache.POLICY_ENV, None)
        self.assertEqual(ImageEmbeddingCache().policy, "tail")
        os.environ[image_cache.POLICY_ENV] = "nonsense"
        with self.assertRaises(ValueError):
            ImageEmbeddingCache()
        os.environ.pop(image_cache.POLICY_ENV, None)
        self.assertEqual(ImageEmbeddingCache().policy, "head")
        self.assertEqual(ImageEmbeddingCache(policy="tail").policy, "tail")   # explicit beats the environment

    def test_tail_policy_gives_up_the_end_of_a_prompt_first(self):
        cache, tower = ImageEmbeddingCache(max_items=2, max_bytes=0, policy="tail"), FakeTower()
        first = cache.get_many([("a", "a"), ("b", "b")], tower)    # one prompt: a sits before b
        cache.get_many([("a", "a"), ("c", "c")], tower)            # room for two of a, b and c
        calls = tower.calls                                        # a, b, c encoded
        # a is the head of both prompts and stays; b and c both sit second, and of two pictures at the same
        # distance from the start the one from the prompt seen longest ago goes first, so b is the one given up.
        self.assertIs(cache.get_many([("a", "a")], tower)[0], first[0])
        self.assertEqual(tower.calls, calls)                       # the head survived: no encode
        cache.get_many([("b", "b")], tower)
        self.assertEqual(tower.calls, calls + 1)                   # ... and b was the one given up

    def test_eviction_by_bytes(self):
        one = image_cache.embedding_nbytes(FakeEmb(4))
        self.assertEqual(one, 4 * 8 * 2)
        cache, tower = ImageEmbeddingCache(max_items=100, max_bytes=2 * one + 1), FakeTower()
        for k in "abc":
            cache.get_many([(k, k)], tower)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 2 * one)

    def test_bytes_count_deepstack_rows(self):
        self.assertEqual(image_cache.embedding_nbytes(FakeEmb(10, hidden=16, deepstack=3)), 4 * 10 * 16 * 2)

    def test_clear_resets_memory(self):
        cache, tower = ImageEmbeddingCache(), FakeTower()
        cache.get_many([("a", "a")], tower)
        cache.clear()
        self.assertEqual((len(cache), cache.nbytes), (0, 0))

    def test_concurrent_requests_encode_once(self):
        cache, tower, out = ImageEmbeddingCache(), FakeTower(), []
        ts = [threading.Thread(target=lambda: out.append(cache.get_many([("a", "a")], tower)[0])) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(tower.calls, 1)
        self.assertEqual(len({id(e) for e in out}), 1)


if __name__ == "__main__":
    unittest.main()
