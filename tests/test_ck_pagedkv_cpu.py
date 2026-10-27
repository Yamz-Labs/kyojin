"""Stdlib-only optional CK loader and admission tests; no package initialization.

Run directly: python3 -I -S -B tests/test_ck_pagedkv_cpu.py -v
Only actual production definitions execute, with inert loader/framework boundaries.
No native library, torch, pytest, CUDA, or exllamav3 package is imported.
"""

import ast
import __future__
import ctypes
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings


_SOURCE = Path(__file__).resolve().parents[1] / "exllamav3/modules/attention_fn/ck_pagedkv.py"


def _load():
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
    names = {"ck_library_path", "_kernel", "owns"}
    body: list[ast.stmt] = [node for node in tree.body
                           if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace: dict = dict(ctypes=ctypes, math=math, os=os, Path=Path, warnings=warnings,
                     _fn=None, _load_failed=False, _supported_device=lambda _: True)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_SOURCE), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return SimpleNamespace(**namespace), namespace


def _args(q_len=16):
    return SimpleNamespace(
        q=SimpleNamespace(device=object()), q_len=q_len, kv_len=q_len,
        is_varlen=lambda: False, is_swa=lambda: False, has_kv_cache=lambda: True,
        cu_seqlens=None, softcap=0.0, sinks=None, causal=True, dim=192,
        num_q_heads=64, num_kv_heads=4, bsz=1, v_dim=128,
    )


class LibraryPathAndOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.ck, self.namespace = _load()

    def test_library_path_uses_cache(self):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "/inert/cache"}, clear=True):
            self.assertEqual(self.ck.ck_library_path(),
                             Path("/inert/cache/kyojin/ck-pagedkv/libck_pagedkv.so"))

    def test_library_path_accepts_explicit_override(self):
        with patch.dict(os.environ, {"EXL3_CK_PAGEDKV_SO": "~/custom.so"}, clear=True):
            self.assertEqual(self.ck.ck_library_path(), Path.home() / "custom.so")

    def test_ownership_declines_decode(self):
        self.assertFalse(self.ck.owns(_args(q_len=1)))
        self.assertTrue(self.ck.owns(_args()))


class OptionalLibraryTests(unittest.TestCase):
    def test_unloadable_library_and_missing_symbol_decline_once(self):
        for failure in (OSError("inert bad ELF"), AttributeError("missing ck_pagedkv_forward")):
            with self.subTest(failure=type(failure).__name__):
                ck, namespace = _load()
                namespace["ck_library_path"] = lambda: Path("/inert/libck.so")
                if isinstance(failure, OSError):
                    loader: dict = dict(side_effect=failure)
                else:
                    class MissingSymbol:
                        def __getattr__(self, name):
                            raise failure
                    loader = dict(return_value=MissingSymbol())
                with patch.object(Path, "is_file", return_value=True), \
                        patch.object(ctypes, "CDLL", **loader) as cdll, \
                        warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    self.assertIsNone(ck._kernel())
                    self.assertIsNone(ck._kernel())
                cdll.assert_called_once_with("/inert/libck.so")
                self.assertEqual(len(seen), 1)
                self.assertIn("fallback", str(seen[0].message).lower())

    def test_unexpected_loader_error_propagates(self):
        ck, namespace = _load()
        namespace["ck_library_path"] = lambda: Path("/inert/libck.so")
        failure = RuntimeError("unexpected loader logic failure")
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(ctypes, "CDLL", side_effect=failure):
            with self.assertRaises(RuntimeError) as caught:
                ck._kernel()
        self.assertIs(caught.exception, failure)

    def test_successful_loader_caches_bound_function_and_float_abi(self):
        ck, namespace = _load()
        namespace["ck_library_path"] = lambda: Path("/inert/libck.so")
        function = SimpleNamespace()
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(ctypes, "CDLL", return_value=SimpleNamespace(
                    ck_pagedkv_forward=function)) as cdll:
            self.assertIs(ck._kernel(), function)
            self.assertIs(ck._kernel(), function)
        cdll.assert_called_once_with("/inert/libck.so")
        self.assertIs(function.restype, ctypes.c_float)
        self.assertEqual(len(function.argtypes), 14)

    def test_missing_library_declines_once_without_loading(self):
        ck, namespace = _load()
        path = SimpleNamespace(is_file=lambda: False)
        namespace["ck_library_path"] = lambda: path
        with patch.object(ctypes, "CDLL") as cdll, warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            self.assertIsNone(ck._kernel())
            self.assertIsNone(ck._kernel())
        cdll.assert_not_called()
        self.assertEqual(len(seen), 1)
        self.assertIn("fallback", str(seen[0].message).lower())


if __name__ == "__main__":
    unittest.main()
