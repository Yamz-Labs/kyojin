"""Stdlib-only sparse cache isolation regressions; no package initialization.

Run directly: python3 -I -S -B tests/test_ck_sparse_cache_cpu.py -v
The frozen mode helper is loaded by file path. Heavyweight cache modules are
AST-extracted, executing their actual functions with inert tensor/I/O seams.
No torch, exllamav3 package, native library, CUDA, or pytest is imported.
"""

import __future__
import ast
from collections import deque
import hashlib
import importlib.util
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch


_ROOT = Path(__file__).resolve().parents[1]
_MODE_SOURCE = _ROOT / "exllamav3/util/attention_mode.py"


def _mode(**env):
    if not _MODE_SOURCE.is_file():
        raise AssertionError("missing frozen attention mode helper")
    spec = importlib.util.spec_from_file_location("_cache_test_attention_mode", _MODE_SOURCE)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load frozen attention mode helper")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, env, clear=True):
        spec.loader.exec_module(module)
    return module


def _extract(relative, names, mode, **seams):
    """Compile real production definitions, binding only their explicit mode imports."""
    source = _ROOT / relative
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    namespace = dict(seams)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("attention_mode"):
            for alias in node.names:
                namespace[alias.asname or alias.name] = getattr(mode, alias.name)
    body = []
    for name in names:
        parts = name.split(".")
        scope = tree.body
        if len(parts) == 2:
            scope = next(n for n in scope if isinstance(n, ast.ClassDef) and n.name == parts[0]).body
        node = next(n for n in scope if (
            isinstance(n, ast.FunctionDef) and n.name == parts[-1]
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == parts[-1]
                                                for t in n.targets)
        ))
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []  # inert inference_mode/staticmethod boundaries, not method bodies
        body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return SimpleNamespace(**namespace)


class _BytesTensor:
    """Only the CPU byte-view boundary used by the production checksum."""
    def __init__(self, data):
        self.data = data

    def numpy(self):
        return self

    def tobytes(self):
        return self.data


def _checksum(mode):
    source = "exllamav3/generator/pagetable.py"
    code = _extract(source, ["_tensor_blake2b_checksum", "tensor_hash_checksum"], mode,
                    hashlib=hashlib)
    if code.tensor_hash_checksum is not code._tensor_blake2b_checksum:
        raise AssertionError("cache callers no longer use the tested production checksum")
    return code.tensor_hash_checksum


class _Ids:
    def __init__(self, values):
        self.values = tuple(values)
        self.shape = (1, len(self.values))

    def __getitem__(self, index):
        return _Ids(self.values[index])

    def clone(self):
        return _Ids(self.values)

    def cpu(self):
        return self

    def numel(self):
        return len(self.values)

    def copy_(self, other, **kwargs):
        self.values = other.values


class _Rows:
    """Inert page-dimension cache tensor: observe real save/restore copy calls."""
    dtype = "inert-cache-dtype"
    device = "inert-device"

    def __init__(self, rows):
        self.rows = [_Ids(row) for row in rows]
        self.shape = (len(self.rows), 2)

    def index_select(self, dimension, indices):
        if dimension != 0:
            raise AssertionError("unexpected cache dimension")
        return _Rows(self.rows[i].values for i in indices)

    def cpu(self):
        return self

    def __getitem__(self, index):
        return self.rows[index]


class _Page:
    def __init__(self, table, phash, ids, complete):
        self.table = table
        self.phash = phash
        self.page_index = 0
        self.prev_hash = None
        self.ref_count = 0
        self.sequence = _Ids(ids)
        self.kv_position = 256 if complete else 0

    def add_ref_clear(self, serial, phash):
        self.table.events.append("inject-page")
        del self.table.unreferenced_pages[self.phash]
        self.phash = phash
        self.ref_count = 1
        self.table.referenced_pages[phash] = self

    def sub_ref(self):
        self.ref_count = 0
        del self.table.referenced_pages[self.phash]
        self.table.unreferenced_pages[self.phash] = self


class _PageTable:
    def __init__(self, phash, ids, complete):
        self.events = []
        self.access_serial = 0
        self.referenced_pages = {}
        self.page = _Page(self, phash, ids, complete)
        self.unreferenced_pages = {phash: self.page}

    def build_eviction_order(self, protected):
        self.events.append("eviction-order")
        return deque(p for h, p in self.unreferenced_pages.items() if h not in protected)

    def evict(self, page, protected):
        self.events.append("evict")


class _SlotFixture:
    """Actual SlotStore methods; only framework, storage and page boundaries inert."""
    def __init__(self, mode, resident=False, payload=None):
        self.payload: Any = payload
        self.events = []
        self.cache = _Rows([(31, 32)] if resident else [(0, 0)])
        self.phash = _checksum(mode)(_BytesTensor(b"resident token page"), None)
        self.table = _PageTable(self.phash if resident else b"empty-placeholder",
                                range(256) if resident else [0] * 256, resident)
        generator = SimpleNamespace(pagetable=self.table, recurrent_cache=None,
                                    tail_ckpts=[], tail_ckpt_max=1)
        torch = SimpleNamespace(load=self._load, save=self._save,
                                cuda=SimpleNamespace(synchronize=lambda: self.events.append("sync")),
                                tensor=lambda values, **kwargs: values, long="inert-long")
        paths = SimpleNamespace(expanduser=os.path.expanduser, basename=os.path.basename,
                                join=os.path.join, getsize=lambda _: 1)
        io = SimpleNamespace(path=paths, replace=lambda *args: self.events.append("replace"))
        code = _extract("exllamav3/generator/slot_store.py", ["FORMAT_VERSION", "FORMAT_VERSION_TAIL",
                        "FORMAT_VERSION_SPARSE", "FORMAT_VERSION_SPARSE_TAIL",
                        "SlotStore.__init__", "SlotStore._path", "SlotStore.save", "SlotStore.restore"],
                        mode, torch=torch, os=io, time=time, PAGE_SIZE=256)
        cls = type("ExtractedSlotStore", (), {name: getattr(code, name)
                   for name in ("__init__", "_path", "save", "restore")})
        self.slot: Any = cls(generator, [self.cache], "/inert/slots", "inert-model")
        self.slot.ids = _Ids(range(256) if resident else [99])
        self.slot.prompt = "saved prompt" if resident else "old prompt"
        self.slot._resident_chain = lambda: ([self.phash], [self.table.page]) if resident else ([], [])
        self.slot._tensors = self._tensors
        self.slot._restore_tail = self._restore_tail

    def _save(self, payload, path):
        self.events.append("save-payload")
        self.payload = payload

    def _load(self, path, **kwargs):
        self.events.append("load-payload")
        return self.payload

    def _tensors(self):
        self.events.append("tensors")
        return [self.cache]

    def _restore_tail(self, *args):
        raise AssertionError("foreign tail reached restore")


def _sparse(retain="0.5", min_sk="0"):
    return _mode(EXL3_CK_PAGEDKV="sparse", EXL3_CK_SPARSE_RETAIN=retain,
                 EXL3_CK_MIN_SK=min_sk)


class FrozenModeTests(unittest.TestCase):
    def test_mode_captures_sparse_policy_at_import_not_at_use(self):
        cases = [
            ({}, False, 0.5, 0),
            ({"EXL3_CK_PAGEDKV": "dense"}, False, 0.5, 0),
            ({"EXL3_CK_PAGEDKV": "SPARSE"}, True, 0.5, 0),
            ({"EXL3_CK_PAGEDKV": "sparse", "EXL3_CK_SPARSE_RETAIN": "0.25",
              "EXL3_CK_MIN_SK": "4096"}, True, 0.25, 4096),
        ]
        for env, enabled, retain, min_sk in cases:
            with self.subTest(env=env):
                mode = _mode(**env)
                self.assertIs(mode.CK_SPARSE_ENABLED, enabled)
                self.assertEqual(mode.CK_SPARSE_RETAIN, retain)
                self.assertEqual(mode.CK_SPARSE_MIN_SK, min_sk)
                marker = mode.ATTENTION_CACHE_MODE
                self.assertEqual(marker is not None, enabled)
                with patch.dict(os.environ, {
                    "EXL3_CK_PAGEDKV": "normal" if enabled else "sparse",
                    "EXL3_CK_SPARSE_RETAIN": "0.75", "EXL3_CK_MIN_SK": "8192",
                }, clear=True):
                    self.assertIs(mode.CK_SPARSE_ENABLED, enabled)
                    self.assertEqual(mode.CK_SPARSE_RETAIN, retain)
                    self.assertEqual(mode.CK_SPARSE_MIN_SK, min_sk)
                    self.assertEqual(mode.ATTENTION_CACHE_MODE, marker)

    def test_normal_ignores_invalid_sparse_only_settings(self):
        for value in ("", "dense", "0", "sparse "):
            with self.subTest(value=value):
                mode = _mode(EXL3_CK_PAGEDKV=value, EXL3_CK_SPARSE_RETAIN="invalid",
                             EXL3_CK_MIN_SK="invalid")
                self.assertIs(mode.CK_SPARSE_ENABLED, False)
                self.assertIsNone(mode.ATTENTION_CACHE_MODE)

    def test_sparse_marker_identifies_effective_retain_and_min_sk(self):
        normal = _mode().ATTENTION_CACHE_MODE
        sparse = _sparse().ATTENTION_CACHE_MODE
        self.assertNotEqual(sparse, normal)
        self.assertIsInstance(sparse, str)
        self.assertEqual(sparse, _sparse(retain="0.50", min_sk="00").ATTENTION_CACHE_MODE)
        self.assertNotEqual(sparse, _sparse(retain="0.25").ATTENTION_CACHE_MODE)
        self.assertNotEqual(sparse, _sparse(min_sk="4096").ATTENTION_CACHE_MODE)


class PageChecksumTests(unittest.TestCase):
    def test_normal_checksum_is_exact_legacy_root_and_chain(self):
        checksum = _checksum(_mode())
        first = _BytesTensor(b"\x01\x00\x02\x00")
        second = _BytesTensor(b"\x03\x00\x04\x00")
        root = hashlib.blake2b(first.data, digest_size=16).digest()
        self.assertEqual(checksum(first, None), root)
        child = hashlib.blake2b(root + second.data, digest_size=16).digest()
        self.assertEqual(checksum(second, root), child)
        self.assertEqual(checksum(first, b""), root)

    def test_sparse_salts_actual_root_and_inherits_isolation_through_chain(self):
        modes = [_mode(), _sparse(), _sparse(retain="0.25"), _sparse(min_sk="4096")]
        first, second = _BytesTensor(b"same token ids"), _BytesTensor(b"next token ids")
        roots, children = [], []
        for mode in modes:
            checksum = _checksum(mode)
            root = checksum(first, None)
            roots.append(root)
            child = checksum(second, root)
            children.append(child)
            self.assertEqual(child, hashlib.blake2b(root + second.data, digest_size=16).digest())
            self.assertEqual(checksum(first, None), root)
        self.assertEqual(len(set(roots)), len(modes), "normal/sparse/retain/min-sk roots must not alias")
        self.assertEqual(len(set(children)), len(modes), "isolation must propagate along the real hash chain")
        equivalent = _checksum(_sparse(retain="0.50"))
        self.assertEqual(equivalent(first, None), roots[1])
        self.assertEqual(equivalent(second, roots[1]), children[1])

    def test_env_flip_cannot_switch_checksum_mode(self):
        tensor = _BytesTensor(b"same ids after a runtime environment change")
        for mode in (_mode(), _sparse(retain="0.25", min_sk="4096")):
            with self.subTest(marker=mode.ATTENTION_CACHE_MODE):
                checksum = _checksum(mode)
                expected = checksum(tensor, None)
                with patch.dict(os.environ, {"EXL3_CK_PAGEDKV": "sparse" if not mode.CK_SPARSE_ENABLED else "",
                                             "EXL3_CK_SPARSE_RETAIN": "0.75", "EXL3_CK_MIN_SK": "8192"}, clear=True):
                    self.assertEqual(checksum(tensor, None), expected)
                    self.assertEqual(_checksum(mode)(tensor, None), expected)


class SlotPersistenceTests(unittest.TestCase):
    def test_sparse_save_is_unreadable_by_legacy_normal_readers(self):
        source = _SlotFixture(_sparse(), resident=True)
        source.slot.save("sparse.pt")
        # Legacy SlotStore.restore accepts only v1/v2 and ignores unknown fields.
        # A marker alone is insufficient: use an incompatible sparse version too.
        self.assertNotIn(source.payload["version"], (1, 2),
                         "legacy normal readers must reject sparse session files")

    def test_save_persists_frozen_mode_in_the_actual_payload(self):
        for mode in (_mode(), _sparse(), _sparse(retain="0.25", min_sk="4096")):
            with self.subTest(marker=mode.ATTENTION_CACHE_MODE):
                with patch.dict(os.environ, {"EXL3_CK_PAGEDKV": "" if mode.CK_SPARSE_ENABLED else "sparse",
                                             "EXL3_CK_SPARSE_RETAIN": "0.75", "EXL3_CK_MIN_SK": "8192"}, clear=True):
                    source = _SlotFixture(mode, resident=True)
                    result = source.slot.save("session.pt")
                self.assertEqual(result["n_saved"], 256)
                self.assertIn("attention_mode", source.payload)
                self.assertEqual(source.payload["attention_mode"], mode.ATTENTION_CACHE_MODE)
                self.assertEqual(source.payload["hashes"], [source.phash])
                self.assertEqual(source.payload["tensors"][0][0].values, (31, 32))

    def test_same_mode_save_restore_reuses_the_actual_page(self):
        pairs = [(_mode(), _mode()), (_sparse(), _sparse(retain="0.50", min_sk="00")),
                 (_sparse(retain="0.25", min_sk="4096"), _sparse(retain="0.25", min_sk="4096"))]
        for saved_mode, restored_mode in pairs:
            with self.subTest(marker=saved_mode.ATTENTION_CACHE_MODE):
                source = _SlotFixture(saved_mode, resident=True)
                source.slot.save("session.pt")
                with patch.dict(os.environ, {"EXL3_CK_PAGEDKV": "" if restored_mode.CK_SPARSE_ENABLED else "sparse",
                                             "EXL3_CK_SPARSE_RETAIN": "0.75", "EXL3_CK_MIN_SK": "8192"}, clear=True):
                    target = _SlotFixture(restored_mode, payload=source.payload)
                    result = target.slot.restore("session.pt")
                self.assertEqual(result["n_restored"], 256)
                self.assertEqual(result["n_read"], 256)
                self.assertEqual(target.table.page.phash, source.phash)
                self.assertEqual(target.table.page.sequence.values, tuple(range(256)))
                self.assertEqual(target.cache[0].values, (31, 32))
                self.assertEqual(target.table.page.kv_position, 256)
                self.assertEqual(target.slot.prompt, "saved prompt")
                self.assertEqual(target.slot.ids.values, tuple(range(256)))
                self.assertEqual(target.table.events.count("inject-page"), 1)
                result = target.slot.restore("session.pt")
                self.assertEqual(result["n_read"], 0, "resident equal-mode page should be reused, not injected again")
                self.assertEqual(target.table.events.count("inject-page"), 1)

    def test_same_mode_tail_save_restore_and_legacy_tail_acceptance(self):
        for mode, version in ((_mode(), 2), (_sparse(), 4)):
            with self.subTest(mode=mode.ATTENTION_CACHE_MODE):
                source = _SlotFixture(mode, resident=True)
                source.slot.generator.recurrent_cache = SimpleNamespace(settle=lambda: None, get=lambda _: None)
                tail_ids = _Ids(range(17))
                entry = dict(ids=tail_ids, page=source.table.page, stashed={}, carry=None)
                source.slot._find_tail = lambda hashes: (0, entry)
                source.slot.save("tail.pt")
                self.assertEqual(source.payload["version"], version)
                self.assertEqual(source.payload["hashes"], [])
                self.assertEqual(source.payload["tail"]["ids"].values, tuple(range(17)))
                payload = dict(source.payload)
                if version == 2:
                    payload.pop("attention_mode")  # old ordinary tail files had no marker
                target = _SlotFixture(mode, payload=payload)
                injected = []
                target.slot._restore_tail = lambda *args: injected.append(args)
                result = target.slot.restore("tail.pt")
                self.assertEqual(result["n_restored"], 17)
                self.assertEqual(len(injected), 1)
                self.assertIs(injected[0][0], payload["tail"])
                self.assertIsNone(injected[0][1])
                self.assertEqual(target.slot.ids.values, tuple(range(17)))

    def test_mismatched_mode_is_rejected_before_tensors_or_any_injection(self):
        modes = [_mode(), _sparse(), _sparse(retain="0.25"), _sparse(min_sk="4096")]
        for saved_mode in modes:
            source = _SlotFixture(saved_mode, resident=True)
            source.slot.save("session.pt")
            # Supply the intended marker explicitly so a missing save field cannot hide a broken restore guard.
            payload = dict(source.payload, attention_mode=saved_mode.ATTENTION_CACHE_MODE)
            for restored_mode in modes:
                if restored_mode.ATTENTION_CACHE_MODE == saved_mode.ATTENTION_CACHE_MODE:
                    continue
                for version in ((1, 2) if saved_mode.ATTENTION_CACHE_MODE is None else (3, 4)):
                    with self.subTest(saved=saved_mode.ATTENTION_CACHE_MODE,
                                      restored=restored_mode.ATTENTION_CACHE_MODE, version=version):
                        data = dict(payload, version=version)
                        if version in (2, 4):
                            data["tail"] = {"inert": "foreign prompt-end checkpoint"}
                        target = _SlotFixture(restored_mode, payload=data)
                        with self.assertRaisesRegex(ValueError, "attention mode|slot file is for"):
                            target.slot.restore("session.pt")
                        self.assertEqual(target.events, ["load-payload"])
                        self.assertEqual(target.table.events, [])
                        self.assertEqual(target.cache[0].values, (0, 0))
                        self.assertEqual(target.table.page.kv_position, 0)
                        self.assertEqual(target.slot.prompt, "old prompt")
                        self.assertEqual(target.slot.ids.values, (99,))

    def test_missing_marker_is_legacy_normal_and_only_reusable_in_normal_mode(self):
        source = _SlotFixture(_mode(), resident=True)
        source.slot.save("legacy.pt")
        data = dict(source.payload)
        data.pop("attention_mode", None)
        target = _SlotFixture(_mode(), payload=data)
        self.assertEqual(target.slot.restore("legacy.pt")["n_read"], 256)
        self.assertEqual(target.cache[0].values, (31, 32))
        sparse = _SlotFixture(_sparse(), payload=data)
        with self.assertRaisesRegex(ValueError, "attention mode|slot file is for"):
            sparse.slot.restore("legacy.pt")
        self.assertEqual(sparse.events, ["load-payload"])
        self.assertEqual(sparse.table.events, [])

    def test_mode_guard_also_protects_tail_only_payloads_without_a_root_hash(self):
        for saved_mode, restored_mode in ((_mode(), _sparse()), (_sparse(), _mode())):
            with self.subTest(saved=saved_mode.ATTENTION_CACHE_MODE):
                payload = {"version": 2 if saved_mode.ATTENTION_CACHE_MODE is None else 4,
                           "model_id": "inert-model", "hashes": [], "tensors": [],
                           "stash": None, "tail": {"inert": "rootless foreign tail"},
                           "attention_mode": saved_mode.ATTENTION_CACHE_MODE}
                target = _SlotFixture(restored_mode, payload=payload)
                with self.assertRaisesRegex(ValueError, "attention mode|slot file is for"):
                    target.slot.restore("tail.pt")
                self.assertEqual(target.events, ["load-payload"])
                self.assertEqual(target.table.events, [])


class _IndexDisk:
    """In-memory stand-in for namespace-directory selection, not its composition."""
    def __init__(self, root, ns, shared):
        self.root = root
        self.ns = ns
        self.entries_by_name = shared.setdefault(ns, {})
        self.removed = []

    def entries(self):
        return sorted(self.entries_by_name)

    def read_idx(self, name):
        return self.entries_by_name[name]

    def remove(self, name):
        self.removed.append(name)
        del self.entries_by_name[name]


def _prefix(mode, shared=None, extra="") -> tuple[Any, Any]:
    shared = {} if shared is None else shared
    namespace = _extract("exllamav3/util/disk_store.py", ["namespace"], mode,
                         hashlib=hashlib).namespace
    slots = SimpleNamespace(_tensors=lambda: [_Rows([(0, 0)])])
    code = _extract("exllamav3/generator/prefix_disk.py", ["FORMAT", "PagedPrefixDisk.__init__",
                    "PagedPrefixDisk.load_index"], mode, namespace=namespace,
                    DiskStore=lambda root, ns: _IndexDisk(root, ns, shared), env_dir=lambda: "/inert/env-root")
    cls = type("ExtractedPagedPrefixDisk", (), {name: getattr(code, name) for name in ("__init__", "load_index")})
    return cls(slots, root="/inert/cache", model_id="inert-model", extra=extra), namespace


class PrefixNamespaceTests(unittest.TestCase):
    def test_normal_namespace_is_exact_legacy_composition(self):
        mode = _mode(EXL3_CK_PAGEDKV="dense", EXL3_CK_SPARSE_RETAIN="invalid", EXL3_CK_MIN_SK="invalid")
        for extra in ("", "caller-specific-tag"):
            with self.subTest(extra=extra):
                prefix, namespace = _prefix(mode, extra=extra)
                expected = namespace("paged1", "inert-model", [((2,), "inert-cache-dtype")], extra)
                self.assertEqual(prefix.ns, expected)
                self.assertEqual(prefix.disk.ns, expected)
                self.assertEqual(prefix.disk.root, "/inert/cache")

    def test_sparse_namespace_separates_each_state_affecting_policy(self):
        modes = [_mode(), _sparse(), _sparse(retain="0.25"), _sparse(min_sk="4096")]
        namespaces = [_prefix(mode)[0].ns for mode in modes]
        self.assertEqual(len(set(namespaces)), len(modes), "normal/sparse/retain/min-sk disk namespaces must not alias")
        self.assertEqual(_prefix(_sparse(retain="0.50", min_sk="00"))[0].ns, namespaces[1])
        for mode, ns in zip(modes[1:], namespaces[1:]):
            prefix, namespace = _prefix(mode)
            self.assertEqual(prefix.ns, namespace("paged1", "inert-model", [((2,), "inert-cache-dtype")], "",
                                                 mode.ATTENTION_CACHE_MODE))
            self.assertEqual(ns, prefix.disk.ns)

    def test_equal_mode_loads_existing_index_but_other_modes_cannot_see_it(self):
        shared = {}
        modes = [_mode(), _sparse(), _sparse(retain="0.25"), _sparse(min_sk="4096")]
        for i, mode in enumerate(modes):
            prefix, _ = _prefix(mode, shared)
            prefix.disk.entries_by_name[f"segment-{i}"] = {
                "ns": prefix.ns, "k": "seg", "hashes": [bytes([i]) * 16]}
            prefix.disk.entries_by_name[f"stash-{i}"] = {
                "ns": prefix.ns, "k": "stash", "anchor": bytes([i]) * 16, "depth": 1}
            prefix.disk.entries_by_name[f"rootless-tail-{i}"] = {
                "ns": prefix.ns, "k": "tail", "prev": None, "pages": 0}
        for i, mode in enumerate(modes):
            with self.subTest(marker=mode.ATTENTION_CACHE_MODE):
                prefix, _ = _prefix(mode, shared)
                self.assertEqual(prefix.pages, {bytes([i]) * 16: (f"segment-{i}", 0)})
                self.assertEqual(prefix.stashes, {bytes([i]) * 16: (f"stash-{i}", 1)})
                self.assertEqual(list(prefix.tails), [f"rootless-tail-{i}"])
                self.assertEqual(prefix.disk.removed, [])

    def test_foreign_index_namespace_is_rejected_in_both_directions(self):
        modes = [_mode(), _sparse(), _sparse(retain="0.25"), _sparse(min_sk="4096")]
        for source_mode in modes:
            source, _ = _prefix(source_mode)
            for target_mode in modes:
                if source_mode.ATTENTION_CACHE_MODE == target_mode.ATTENTION_CACHE_MODE:
                    continue
                with self.subTest(saved=source_mode.ATTENTION_CACHE_MODE, target=target_mode.ATTENTION_CACHE_MODE):
                    prefix, _ = _prefix(target_mode)
                    prefix.disk.entries_by_name["foreign-page"] = {"ns": source.ns, "k": "seg", "hashes": [b"hash"]}
                    prefix.disk.entries_by_name["foreign-tail"] = {"ns": source.ns, "k": "tail", "prev": None}
                    self.assertEqual(prefix.load_index(), 0)
                    self.assertEqual(prefix.pages, {})
                    self.assertEqual(prefix.tails, {})
                    self.assertEqual(prefix.disk.removed, ["foreign-page", "foreign-tail"])

    def test_env_flip_cannot_create_a_different_mode_disk_session(self):
        for mode in (_mode(), _sparse(retain="0.25", min_sk="4096")):
            with self.subTest(marker=mode.ATTENTION_CACHE_MODE):
                expected, _ = _prefix(mode)
                with patch.dict(os.environ, {"EXL3_CK_PAGEDKV": "" if mode.CK_SPARSE_ENABLED else "sparse",
                                             "EXL3_CK_SPARSE_RETAIN": "0.75", "EXL3_CK_MIN_SK": "8192"}, clear=True):
                    new_session, _ = _prefix(mode)
                    self.assertEqual(new_session.ns, expected.ns)


if __name__ == "__main__":
    unittest.main()
