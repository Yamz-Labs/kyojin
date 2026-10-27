"""Stdlib-only regressions for the two MiMo sparse scratch-memory corrections.

Run directly, without pytest/package initialization:
    python3 -I -S -B tests/test_ck_pagedkv_sparse_memory_cpu.py -v

Only the production page-mean statements and _stage_buffers function are executed,
with inert tensor/allocator doubles. These tests establish Python call and ownership
contracts, not framework numerics, device memory peaks, or cross-stream safety.
Scratch is fully overwritten per accepted call on the existing single-stream path;
no persistent external views are retained across growth in these tests.
"""

import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref


_SOURCE = Path(__file__).resolve().parents[1] / "exllamav3/modules/attention_fn"


def _function(filename, name):
    path = _SOURCE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError(f"expected exactly one {name} in {path}")
    return path, functions[0]


def _page_mean_code():
    path, function = _function("ck_pagedkv_sparse.py", "select_pages")
    start = next(i for i, node in enumerate(function.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "kp"
                         for target in node.targets))
    end = next(i for i, node in enumerate(function.body[start:], start)
               if isinstance(node, ast.Delete)
               and any(isinstance(target, ast.Name) and target.id == "kp"
                       for target in node.targets))
    return compile(ast.Module(body=function.body[start:end + 1], type_ignores=[]),
                   str(path), "exec")


class PageMeanMemoryTests(unittest.TestCase):
    def test_page_mean_uses_fp32_reduction_without_full_prefix_float_copy(self):
        fp32 = object()
        physical_pages = object()
        reduced_mean = object()
        reductions = []

        class PagePrefix:
            def float(self):
                raise AssertionError("full-prefix .float() copy before page mean")

            def mean(self, *, dim, dtype=None):
                reductions.append((dim, dtype))
                return reduced_mean

        class Cache:
            def __getitem__(self, pages):
                self.pages = pages
                prefix = PagePrefix()
                self.prefix_ref = weakref.ref(prefix)
                return prefix

        cache = Cache()
        namespace = {"k_cache": cache, "phys": physical_pages,
                     "torch": SimpleNamespace(float32=fp32)}
        exec(_page_mean_code(), namespace)

        self.assertIs(cache.pages, physical_pages)
        self.assertEqual(reductions, [(1, fp32)])
        self.assertIs(namespace["mean_k"], reduced_mean)
        self.assertNotIn("kp", namespace)
        gc.collect()
        self.assertIsNone(cache.prefix_ref(), "gathered prefix must be retired")


def _stage_function(allocator):
    path, function = _function("ck_pagedkv_sparse_qc.py", "_stage_buffers")
    namespace = {"torch": allocator, "_STAGE": {}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
         namespace)
    allocator.stages = namespace["_STAGE"]
    return namespace["_stage_buffers"], namespace["_STAGE"]


class StageCapacityTests(unittest.TestCase):
    def test_capacity_steps_up_and_stops_at_the_pool(self):
        path, function = _function("ck_pagedkv_sparse_qc.py", "_stage_capacity")
        namespace = {"min": min, "_STAGE_STEP_PAGES": 128}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        capacity = namespace["_stage_capacity"]
        self.assertEqual([capacity(n, 512) for n in (1, 128, 129, 300, 512)], [128, 128, 256, 384, 512])
        self.assertEqual(capacity(130, 140), 140)


class _InertTensor:
    def __init__(self, shape):
        self.shape = shape


class _InertAllocator:
    half = object()

    def __init__(self):
        self.calls = []
        self.allocated_refs = []
        self.watched_refs: tuple = ()
        self.fail_at: int | None = None
        self.stages = {}

    def empty(self, shape, *, dtype, device):
        # Collect unreachable objects for non-refcounted Python implementations.
        # Snapshots and allocation records must never own the observed tensors.
        gc.collect()
        self.calls.append({"shape": shape, "dtype": dtype, "device": device,
                           "published": device in self.stages,
                           "watched_alive": tuple(ref() is not None
                                                  for ref in self.watched_refs)})
        if len(self.calls) == self.fail_at:
            raise MemoryError("inert second allocation failure")
        tensor = _InertTensor(shape)
        self.allocated_refs.append(weakref.ref(tensor))
        return tensor


class StageBufferMemoryTests(unittest.TestCase):
    def setUp(self):
        self.allocator = _InertAllocator()
        self.stage_buffers, self.stages = _stage_function(self.allocator)
        self.dev = "inert:0"
        self.vals = 768

    def test_initial_allocation_publishes_only_complete_pair(self):
        current = self.stage_buffers(self.dev, 2, self.vals)
        self.assertIs(self.stages[self.dev], current)
        self.assertEqual(set(current), {"k", "v", "ident"})
        self.assertIsNone(current["ident"])
        self.assertEqual(len(self.allocator.calls), 2)
        for name in ("k", "v"):
            self.assertEqual(current[name].shape, (2 * 256, self.vals))
        for call in self.allocator.calls:
            self.assertEqual(call["shape"], (2 * 256, self.vals))
            self.assertIs(call["dtype"], self.allocator.half)
            self.assertEqual(call["device"], self.dev)
            self.assertFalse(call["published"])

    def test_equal_and_smaller_requests_reuse_pair_and_identity(self):
        current = self.stage_buffers(self.dev, 3, self.vals)
        current["ident"] = _InertTensor((3,))
        identity = current["ident"]
        for pages in (3, 2, 0):
            self.assertIs(self.stage_buffers(self.dev, pages, self.vals), current)
            self.assertIs(current["ident"], identity)
        self.assertEqual(len(self.allocator.calls), 2)

    def test_growth_allocates_exact_requested_capacity(self):
        for pages in (2, 3, 5):
            current = self.stage_buffers(self.dev, pages, self.vals)
            for name in ("k", "v"):
                self.assertEqual(current[name].shape, (pages * 256, self.vals))
            del current
        self.assertEqual([call["shape"] for call in self.allocator.calls],
                         [(pages * 256, self.vals) for pages in (2, 3, 5)
                          for _ in range(2)])

    def test_growth_retires_old_pair_before_first_replacement_allocation(self):
        self.stage_buffers(self.dev, 2, self.vals)
        for pages in (3, 5):
            self.stages[self.dev]["ident"] = _InertTensor((pages - 1,))
            self.allocator.watched_refs = tuple(
                weakref.ref(self.stages[self.dev][name])
                for name in ("k", "v", "ident"))
            start = len(self.allocator.calls)
            current = self.stage_buffers(self.dev, pages, self.vals)
            for call in self.allocator.calls[start:]:
                self.assertEqual(call["watched_alive"], (False, False, False),
                                 "old K/V/identity still reachable during replacement")
                self.assertFalse(call["published"])
            self.assertIsNone(current["ident"])
            del current

    def test_failed_second_allocation_leaves_no_stage_and_retry_succeeds(self):
        self.stage_buffers(self.dev, 2, self.vals)
        self.stages[self.dev]["ident"] = _InertTensor((2,))
        old_refs = tuple(weakref.ref(self.stages[self.dev][name])
                         for name in ("k", "v", "ident"))
        self.allocator.watched_refs = old_refs
        start = len(self.allocator.calls)
        self.allocator.fail_at = start + 2
        with self.assertRaisesRegex(MemoryError, "inert second allocation failure"):
            self.stage_buffers(self.dev, 3, self.vals)
        gc.collect()
        self.assertNotIn(self.dev, self.stages,
                         "failed replacement must not publish stale or partial scratch")
        self.assertEqual(tuple(ref() is not None for ref in old_refs),
                         (False, False, False))
        self.assertIsNone(self.allocator.allocated_refs[-1](),
                          "unpublished replacement K must not survive failed V")
        self.assertEqual(len(self.allocator.calls) - start, 2)
        for call in self.allocator.calls[start:]:
            self.assertFalse(call["published"])
            self.assertEqual(call["watched_alive"], (False, False, False))

        self.allocator.fail_at = None
        current = self.stage_buffers(self.dev, 3, self.vals)
        self.assertIs(self.stages[self.dev], current)
        self.assertIsNone(current["ident"])
        self.assertEqual(len(self.allocator.calls) - start, 4)
        for name in ("k", "v"):
            self.assertEqual(current[name].shape, (3 * 256, self.vals))
        for call in self.allocator.calls[start + 2:]:
            self.assertFalse(call["published"])


if __name__ == "__main__":
    unittest.main()
