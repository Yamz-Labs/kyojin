"""Stdlib-only CK admission regressions; no package/framework initialization.

Run directly:
    python3 -I -S -B tests/test_ck_pagedkv_admission_cpu.py -v

Extract the actual admission functions, direct sparse guard, and engine installer
with AST. Relative CK imports bind only these inert namespaces. All selector,
cache-update/staging, allocation, and native boundaries fail if reached. These
checks establish Python admission behavior, not device or numerical correctness.
"""

import ast
import builtins
import __future__
from pathlib import Path
from types import SimpleNamespace
import unittest


_SOURCE = Path(__file__).resolve().parents[1] / "exllamav3/modules/attention_fn"


def _tree(filename):
    path = _SOURCE / filename
    return path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _load(filename, names, namespace, dependencies=None):
    path, tree = _tree(filename)
    dependencies = dependencies or {}

    def relative_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 1 or name not in dependencies:
            raise AssertionError(f"unexpected import: {level}:{name}")
        return SimpleNamespace(**dependencies[name])

    namespace["__builtins__"] = dict(vars(builtins), __import__=relative_import)
    body: list[ast.stmt] = [node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in names)
            or (isinstance(node, ast.ImportFrom) and node.module in dependencies)]
    if {node.name for node in body if isinstance(node, ast.FunctionDef)} != set(names):
        raise ValueError(f"missing extracted function in {path}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def _forbidden(label):
    def fail(*args, **kwargs):
        raise AssertionError(f"unexpected {label} work")
    return fail


def _entry_points():
    half = object()
    torch = SimpleNamespace(
        float16=half, version=SimpleNamespace(hip="inert"),
        cuda=SimpleNamespace(is_available=lambda: True,
                             get_device_properties=lambda device:
                             SimpleNamespace(gcnArchName=device.arch)),
        zeros=_forbidden("allocation"), arange=_forbidden("allocation"),
        tensor=_forbidden("allocation"),
    )
    ext = SimpleNamespace(paged_kv_cache_update=_forbidden("cache update"))
    dense = _load("ck_pagedkv.py", ("_supported_device", "owns", "fn_ck_paged_attn"),
                  {"torch": torch, "ext": ext, "_PAGE": 256,
                   "_kernel": _forbidden("optional loader"),
                   "_stats": {"accepted": 0, "rejected": 0}})
    sparse = _load(
        "ck_pagedkv_sparse.py",
        ("_owned", "fn_ck_paged_attn_sparse", "fn_ck_paged_attn_sparse_auto"),
        {"torch": torch, "ext": ext, "os": SimpleNamespace(environ={}),
         "select_pages": _forbidden("selector"),
         "compact_table": _forbidden("compaction"),
         "_stats": {"dense": 0, "sparse": 0, "declined": 0}},
        {"ck_pagedkv": dense},
    )
    qc = _load(
        "ck_pagedkv_sparse_qc.py", ("_owned_qc", "fn_ck_paged_attn_sparse_qc"),
        {"torch": torch, "_page": 256, "os": SimpleNamespace(environ={}),
         "_stage_buffers": _forbidden("cache staging"),
         "_dequant8_pages_pad": _forbidden("dequantization"),
         "_stats": {"qc_calls": 0, "qc_staged_pages": 0, "qc_kept_v_pages": 0}},
        {"ck_pagedkv": dense, "ck_pagedkv_sparse": sparse},
    )
    engine = _load(
        "ck_pagedkv_sparse_engine.py",
        ("install_sparse_engine", "fn_ck_paged_attn_sparse_engine"),
        {"_stats": {"accepted": 0, "declined": 0, "pages_kept": 0, "pages_total": 0}},
        {"ck_pagedkv": dense, "ck_pagedkv_sparse": sparse},
    )
    # Execute the direct sparse entry's actual duplicated predicate separately
    # so its accepted control needs no tensor/cache/native implementation.
    path, tree = _tree("ck_pagedkv_sparse.py")
    direct = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "fn_ck_paged_attn_sparse")
    guard = next(node.value for node in direct.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "owned"
                         for target in node.targets))
    guard_code = compile(ast.Expression(body=guard), str(path), "eval")
    return SimpleNamespace(dense=dense, sparse=sparse, qc=qc, engine=engine,
                           half=half, direct_guard=guard_code)


def _args(half, *, arch="gfx1151:sramecc-", v_dim: int | None = 128):
    qk = SimpleNamespace(shape=(2, 256, 192))
    sk = SimpleNamespace(shape=(2, 256, 24))
    return SimpleNamespace(
        q=SimpleNamespace(device=SimpleNamespace(type="cuda", arch=arch),
                          dtype=half, shape=(1, 256, 64, 192)),
        q_len=256, kv_len=256, k=object(), v=object(),
        k_cache=SimpleNamespace(shape=(2, 256, 4, 192)), v_cache=object(),
        block_table=object(), cache_seqlens=[SimpleNamespace(item=lambda: 256)],
        q_cache=(qk, sk, qk, sk, 8, 8),
        is_varlen=lambda: False, is_swa=lambda: False, has_kv_cache=lambda: True,
        cu_seqlens=None, softcap=0.0, sinks=None, causal=True, dim=192,
        num_q_heads=64, num_kv_heads=4, bsz=1, v_dim=v_dim,
    )


class _AdmissionFixture(unittest.TestCase):
    def setUp(self):
        self.entries = _entry_points()

    def predicates(self, args):
        e = self.entries
        return {
            "dense": e.dense["owns"](args),
            "sparse_auto": e.sparse["_owned"](args),
            "sparse_direct": eval(e.direct_guard, dict(e.sparse, args=args)),
            "qc": e.qc["_owned_qc"](args),
        }

    def installed(self):
        e = self.entries
        fallback = _forbidden("fallback")
        handlers = [e.dense["fn_ck_paged_attn"], fallback]
        self.assertIs(e.engine["install_sparse_engine"](handlers), handlers)
        self.assertIs(handlers[0], e.engine["fn_ck_paged_attn_sparse_engine"])
        self.assertIs(handlers[1], fallback)
        return handlers[0]


class DeviceAdmissionTests(_AdmissionFixture):
    def test_supported_gfx1151_v128_accepts_every_predicate(self):
        for name, owned in self.predicates(_args(self.entries.half)).items():
            with self.subTest(entry=name):
                self.assertTrue(owned)

    def test_optional_cache_formats_decline_every_predicate(self):
        for field in ("ov", "tq"):
            args = _args(self.entries.half)
            setattr(args, field, object())
            for name, owned in self.predicates(args).items():
                with self.subTest(field=field, entry=name):
                    self.assertFalse(owned, "CK does not implement KVWIN or TurboQuant cache semantics")

    def test_unsupported_device_declines_every_predicate(self):
        args = _args(self.entries.half, arch="gfx1100")
        for name, owned in self.predicates(args).items():
            with self.subTest(entry=name):
                self.assertFalse(owned)

    def test_unsupported_device_declines_installed_engine_before_work(self):
        self.assertIsNone(self.installed()(_args(self.entries.half, arch="gfx1100")))
        self.assertEqual(self.entries.engine["_stats"]["declined"], 1)

    def test_supported_installed_engine_reaches_selector(self):
        # Availability is checked before selection; supply an inert available entry
        # rather than turning a native lookup into a launch or hiding that lookup.
        self.entries.sparse["_kernel"] = lambda: object()
        self.entries.sparse["CK_SPARSE_RETAIN"] = 0.5
        self.entries.sparse["CK_SPARSE_MIN_SK"] = 0
        with self.assertRaisesRegex(AssertionError, "unexpected selector work"):
            self.installed()(_args(self.entries.half))

    def test_unsupported_device_declines_direct_sparse_before_work(self):
        self.assertIsNone(self.entries.sparse["fn_ck_paged_attn_sparse"](
            _args(self.entries.half, arch="gfx1100"), object()))

    def test_unsupported_device_declines_direct_dense_before_work(self):
        self.assertIsNone(self.entries.dense["fn_ck_paged_attn"](
            _args(self.entries.half, arch="gfx1100")))

    def test_unsupported_device_declines_direct_qc_before_work(self):
        self.assertIsNone(self.entries.qc["fn_ck_paged_attn_sparse_qc"](
            _args(self.entries.half, arch="gfx1100")))


class SpanMaskAdmissionTests(_AdmissionFixture):
    def args_with_spans(self, spans):
        args = _args(self.entries.half)
        args.q.shape = (1, 64, 64, 192)
        args.q_len = args.kv_len = 64
        args.k_cache.shape = (3, 256, 4, 192)
        args.q_cache[0].shape = (3, 256, 192)
        args.q_cache[1].shape = (3, 256, 24)
        args.cache_seqlens = [SimpleNamespace(item=lambda: 512)]
        args.non_causal_spans = spans
        return args

    def test_nonempty_spans_decline_every_predicate(self):
        args = self.args_with_spans([(0, 64, True)])
        for name, owned in self.predicates(args).items():
            with self.subTest(entry=name):
                self.assertFalse(owned, "CK does not implement non-causal span masks")

    def test_none_or_empty_spans_accept_every_predicate(self):
        for spans in (None, []):
            for name, owned in self.predicates(self.args_with_spans(spans)).items():
                with self.subTest(spans=spans, entry=name):
                    self.assertTrue(owned)

    def test_nonempty_spans_decline_direct_dense_before_work(self):
        self.assertIsNone(self.entries.dense["fn_ck_paged_attn"](
            self.args_with_spans([(0, 64, True)])))

    def test_nonempty_spans_decline_direct_sparse_before_work(self):
        self.assertIsNone(self.entries.sparse["fn_ck_paged_attn_sparse"](
            self.args_with_spans([(0, 64, True)]), object()))

    def test_nonempty_spans_decline_auto_sparse_before_work(self):
        self.assertEqual(self.entries.sparse["fn_ck_paged_attn_sparse_auto"](
            self.args_with_spans([(0, 64, True)])), (None, None))

    def test_nonempty_spans_decline_installed_engine_before_work(self):
        self.assertIsNone(self.installed()(self.args_with_spans([(0, 64, True)])))
        self.assertEqual(self.entries.engine["_stats"]["declined"], 1)

    def test_nonempty_spans_decline_direct_qc_before_work(self):
        self.assertIsNone(self.entries.qc["fn_ck_paged_attn_sparse_qc"](
            self.args_with_spans([(0, 64, True)])))

    def test_none_or_empty_spans_installed_engine_reaches_selector(self):
        self.entries.sparse["_kernel"] = lambda: object()
        self.entries.sparse["CK_SPARSE_RETAIN"] = 0.5
        self.entries.sparse["CK_SPARSE_MIN_SK"] = 0
        handler = self.installed()
        for spans in (None, []):
            with self.subTest(spans=spans):
                with self.assertRaisesRegex(AssertionError, "unexpected selector work"):
                    handler(self.args_with_spans(spans))


class ValueWidthAdmissionTests(_AdmissionFixture):
    def test_explicit_v128_accepts_every_predicate(self):
        for name, owned in self.predicates(_args(self.entries.half, v_dim=128)).items():
            with self.subTest(entry=name):
                self.assertTrue(owned)

    def test_implicit_or_full_v192_declines_every_predicate(self):
        for width in (None, 192):
            for name, owned in self.predicates(_args(self.entries.half, v_dim=width)).items():
                with self.subTest(v_dim=width, entry=name):
                    self.assertFalse(owned)

    def test_implicit_or_full_v192_declines_installed_engine_before_work(self):
        handler = self.installed()
        for width in (None, 192):
            with self.subTest(v_dim=width):
                self.assertIsNone(handler(_args(self.entries.half, v_dim=width)))

    def test_implicit_or_full_v192_declines_direct_sparse_before_work(self):
        for width in (None, 192):
            with self.subTest(v_dim=width):
                self.assertIsNone(self.entries.sparse["fn_ck_paged_attn_sparse"](
                    _args(self.entries.half, v_dim=width), object()))

    def test_implicit_or_full_v192_declines_direct_dense_before_work(self):
        for width in (None, 192):
            with self.subTest(v_dim=width):
                self.assertIsNone(self.entries.dense["fn_ck_paged_attn"](
                    _args(self.entries.half, v_dim=width)))

    def test_implicit_or_full_v192_declines_direct_qc_before_work(self):
        for width in (None, 192):
            with self.subTest(v_dim=width):
                self.assertIsNone(self.entries.qc["fn_ck_paged_attn_sparse_qc"](
                    _args(self.entries.half, v_dim=width)))


if __name__ == "__main__":
    unittest.main()
