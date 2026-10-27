"""Isolated Python CK fallback contracts, executed from real AST definitions.

Run: python3 -I -S -B tests/test_ck_pagedkv_fallback_cpu.py -v
Tensor, stream and native boundaries are inert; no package/framework is imported.
"""

import ast
import builtins
import __future__
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


_SOURCE = Path(__file__).resolve().parents[1] / "exllamav3/modules/attention_fn"


def _load(filename, names, namespace: dict, dependencies=None) -> dict:
    path = _SOURCE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dependencies = dependencies or {}

    def relative_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 1 or name not in dependencies:
            raise AssertionError(f"unexpected import: {level}:{name}")
        return SimpleNamespace(**dependencies[name])

    namespace["__builtins__"] = dict(vars(builtins), __import__=relative_import)
    body: list[ast.stmt] = [node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in body if isinstance(node, ast.FunctionDef)} != set(names):
        raise ValueError(f"missing production function in {path}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def _forbidden(label):
    def fail(*args, **kwargs):
        raise AssertionError(f"unexpected {label} work")
    return fail


def _args():
    half = object()
    return SimpleNamespace(
        q=SimpleNamespace(device="inert:0", dtype=half, shape=(1, 256, 64, 192)),
        q_len=256, kv_len=256, k=object(), v=object(),
        k_cache=SimpleNamespace(shape=(2, 256, 4, 192)), v_cache=object(),
        block_table=object(), cache_seqlens=[SimpleNamespace(item=lambda: 256)],
        q_cache=(SimpleNamespace(shape=(2, 256, 192)),
                 SimpleNamespace(shape=(2, 256, 24)), object(), object(), 8, 8),
        is_varlen=lambda: False, is_swa=lambda: False, has_kv_cache=lambda: True,
        cu_seqlens=None, softcap=0.0, sinks=None, causal=True, dim=192,
        num_q_heads=64, num_kv_heads=4, bsz=1, v_dim=128, sm_scale=192 ** -0.5,
    )


def _entries(kernel):
    args = _args()
    torch = SimpleNamespace(float16=args.q.dtype,
                            zeros=_forbidden("allocation"),
                            cuda=SimpleNamespace(current_stream=_forbidden("stream lookup")))
    common: dict = dict(torch=torch, math=math, os=os, _PAGE=256, _page=256,
                  CK_SPARSE_RETAIN=0.5, CK_SPARSE_MIN_SK=0,
                  _supported_device=lambda _: True, _kernel=kernel,
                  ext=SimpleNamespace(paged_kv_cache_update=_forbidden("cache update")))
    dense = _load("ck_pagedkv.py", {"owns", "fn_ck_paged_attn"},
                  dict(common, _stats={"accepted": 0, "rejected": 0}))
    sparse = _load("ck_pagedkv_sparse.py",
                   {"_owned", "fn_ck_paged_attn_sparse", "fn_ck_paged_attn_sparse_auto"},
                   dict(common, select_pages=_forbidden("selector"),
                        compact_table=_forbidden("compaction"),
                        _stats={"dense": 0, "sparse": 0, "declined": 0}))
    qc = _load("ck_pagedkv_sparse_qc.py", {"_owned_qc", "fn_ck_paged_attn_sparse_qc"},
               dict(common, _stage_buffers=_forbidden("staging"),
                    _dequant8_pages_pad=_forbidden("dequantization"),
                    _stats={"qc_calls": 0, "qc_staged_pages": 0, "qc_kept_v_pages": 0}),
               {"ck_pagedkv_sparse": sparse})
    return args, dense, sparse, qc


class AvailabilityTests(unittest.TestCase):
    def test_unavailable_library_declines_all_entries_before_tensor_work(self):
        lookups = []

        def unavailable():
            lookups.append("lookup")
            return None

        args, dense, sparse, qc = _entries(unavailable)
        entries = [
            (dense["fn_ck_paged_attn"], (args,), None),
            (sparse["fn_ck_paged_attn_sparse"], (args, object()), None),
            (sparse["fn_ck_paged_attn_sparse_auto"], (args,), (None, None)),
            (qc["fn_ck_paged_attn_sparse_qc"], (args,), None),
        ]
        for fn, params, expected in entries:
            with self.subTest(entry=fn.__name__):
                before = len(lookups)
                self.assertEqual(fn(*params), expected)
                self.assertEqual(len(lookups), before + 1)


class FrozenPolicyTests(unittest.TestCase):
    def test_modules_bind_frozen_policy_instead_of_reading_environment(self):
        for filename, name in (("ck_pagedkv_sparse.py", "fn_ck_paged_attn_sparse_auto"),
                               ("ck_pagedkv_sparse_qc.py", "fn_ck_paged_attn_sparse_qc")):
            with self.subTest(module=filename):
                tree = ast.parse((_SOURCE / filename).read_text(encoding="utf-8"))
                imported = {alias.name for node in tree.body
                            if isinstance(node, ast.ImportFrom) and node.level == 3
                            and node.module == "util.attention_mode" for alias in node.names}
                self.assertTrue({"CK_SPARSE_RETAIN", "CK_SPARSE_MIN_SK"} <= imported)
                function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                and node.name == name)
                self.assertFalse(any(isinstance(node, ast.Constant)
                                     and node.value in ("EXL3_CK_SPARSE_RETAIN", "EXL3_CK_MIN_SK")
                                     for node in ast.walk(function)))

    def test_qc_declines_below_frozen_threshold_before_staging(self):
        args, _, _, qc = _entries(lambda: object())
        qc["CK_SPARSE_MIN_SK"] = 1024
        self.assertIsNone(qc["fn_ck_paged_attn_sparse_qc"](args))

    def test_auto_uses_frozen_threshold_and_retain(self):
        args, _, sparse, _ = _entries(lambda: object())
        sparse["CK_SPARSE_RETAIN"] = 0.25
        sparse["CK_SPARSE_MIN_SK"] = 1024
        self.assertEqual(sparse["fn_ck_paged_attn_sparse_auto"](args), (None, None))
        seen = []
        sparse["CK_SPARSE_MIN_SK"] = 0
        sparse["select_pages"] = lambda *params, **kwargs: (seen.append(kwargs) or object(), {})
        sparse["fn_ck_paged_attn_sparse"] = lambda *params: "output"
        self.assertEqual(sparse["fn_ck_paged_attn_sparse_auto"](args), ("output", {}))
        self.assertEqual(seen[-1]["retain"], 0.25)
        sparse["fn_ck_paged_attn_sparse_auto"](args, retain=0.75)
        self.assertEqual(seen[-1]["retain"], 0.75)


class _Tensor:
    def __init__(self, shape=(), value=None):
        self.shape = shape
        self.value = value
        self.device = "inert:0"

    def __getitem__(self, key):
        return self

    def long(self):
        return self

    def permute(self, *dims):
        return self

    def contiguous(self):
        return self

    def numel(self):
        return self.shape[0]

    def data_ptr(self):
        return id(self)


def _direct_entries(result=1.0, failure=None):
    args, dense, sparse, _ = _entries(lambda: None)
    trace = []
    outputs = []
    args.q = _Tensor((1, 256, 64, 192))
    args.q.device = "inert:0"
    args.k_cache = _Tensor((2, 256, 4, 192))
    args.v_cache = _Tensor((2, 256, 4, 192))
    args.block_table = _Tensor((1, 2))
    args.cache_seqlens = [SimpleNamespace(item=lambda: 256)]

    def cache_update(k, v, kc, vc, bt, cs):
        trace.append(("write", bt, cs, cs[0].item()))

    def zeros(*shape, **kwargs):
        out = _Tensor(shape)
        outputs.append(out)
        return out

    def kernel(*params):
        trace.append(("kernel", params))
        if failure is not None:
            raise failure
        return result

    torch = SimpleNamespace(
        float16=object(), int32=object(), zeros=zeros,
        arange=lambda n, **kwargs: _Tensor((n,)),
        tensor=lambda values, **kwargs: _Tensor((len(values),)),
        cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=123)),
    )
    for namespace in (dense, sparse):
        namespace.update(torch=torch, _kernel=lambda: kernel,
                         ext=SimpleNamespace(paged_kv_cache_update=cache_update))
    sparse["compact_table"] = lambda *params: (_Tensor((2,)), 512)
    return SimpleNamespace(args=args, dense=dense, sparse=sparse,
                           trace=trace, outputs=outputs, update=cache_update)


class DirectResultTests(unittest.TestCase):
    def _call(self, entry, result=1.0, failure=None):
        fixture = _direct_entries(result, failure)
        if entry == "dense":
            function = fixture.dense["fn_ck_paged_attn"]
            params = (fixture.args,)
        else:
            function = fixture.sparse["fn_ck_paged_attn_sparse"]
            params = (fixture.args, object())
        return fixture, lambda: function(*params)

    def test_explicit_support_rejection_declines_to_ordinary_handler(self):
        for entry in ("dense", "sparse"):
            with self.subTest(entry=entry):
                fixture, call = self._call(entry, -1.0)
                fallback_output = object()
                fallback_calls = []

                def fallback():
                    fallback_calls.append(True)
                    return fallback_output

                for handler in (call, fallback):
                    output = handler()
                    if output is not None:
                        break
                self.assertIs(output, fallback_output)
                self.assertEqual(fallback_calls, [True])
                self.assertEqual([event[0] for event in fixture.trace], ["write", "kernel"])
                self.assertIs(fixture.trace[0][1], fixture.args.block_table)
                self.assertIs(fixture.trace[0][2], fixture.args.cache_seqlens)
                self.assertEqual(fixture.args.cache_seqlens[0].item(), 256)
                self.assertEqual(fixture.dense["_stats"]["accepted"], 0)
                self.assertEqual(fixture.sparse["_stats"]["sparse"], 0)
                counter = fixture.dense["_stats"] if entry == "dense" else fixture.sparse["_stats"]
                self.assertEqual(counter["rejected" if entry == "dense" else "declined"], 1)

    def test_success_returns_allocated_output(self):
        for entry in ("dense", "sparse"):
            with self.subTest(entry=entry):
                fixture, call = self._call(entry)
                self.assertIs(call(), fixture.outputs[0])
                self.assertEqual([event[0] for event in fixture.trace], ["write", "kernel"])

    def test_invalid_status_does_not_reach_fallback(self):
        for entry in ("dense", "sparse"):
            for result in (-2.0, -0.5, 0.0, float("nan"), float("inf"), -float("inf")):
                with self.subTest(entry=entry, result=result):
                    fixture, call = self._call(entry, result)
                    with self.assertRaisesRegex(RuntimeError, "rc="):
                        for handler in (call, _forbidden("fallback")):
                            if handler() is not None:
                                break
                    self.assertEqual(fixture.dense["_stats"]["accepted"], 0)
                    self.assertEqual(fixture.sparse["_stats"]["sparse"], 0)

    def test_python_visible_invocation_errors_propagate_unchanged(self):
        # Native uncaught C++ exceptions can terminate the process; no ABI recovery
        # is claimed by a test substituting these Python-visible callables.
        for entry in ("dense", "sparse"):
            for failure in (OSError("inert launch error"), RuntimeError("inert device error")):
                with self.subTest(entry=entry, failure=type(failure).__name__):
                    fixture, call = self._call(entry, failure=failure)
                    with self.assertRaises(type(failure)) as caught:
                        call()
                    self.assertIs(caught.exception, failure)
                    self.assertEqual([event[0] for event in fixture.trace], ["write", "kernel"])


if __name__ == "__main__":
    unittest.main()
