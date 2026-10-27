"""Import-time dispatch/cache policy agreement; no framework or toolchain needed."""
import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'exllamav3/modules/attention_fn/dispatch.py'


class DispatchModeTests(unittest.TestCase):
    def test_shared_mode_controls_registration_not_later_environment(self):
        tree = ast.parse(SOURCE.read_text())
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                   and node.module == 'util.attention_mode']
        self.assertEqual(len(imports), 1, 'dispatch must import the frozen shared cache/attention mode')
        branch = tree.body[-1]
        self.assertIsInstance(branch, ast.If)
        for enabled, available in ((False, False), (False, True), (True, False), (True, True)):
            calls = []
            def importer(name, globals=None, locals=None, fromlist=(), level=0):
                if name == 'util.attention_mode' and level == 3:
                    return SimpleNamespace(CK_SPARSE_ENABLED=enabled)
                if name == 'ck_pagedkv' and level == 1:
                    return SimpleNamespace(_kernel=lambda: object() if available else None)
                if name == 'ck_pagedkv_sparse_engine' and level == 1:
                    return SimpleNamespace(install_sparse_engine=lambda handlers: calls.append('fp16'))
                if name == 'ck_pagedkv_sparse_qc' and level == 1:
                    return SimpleNamespace(install_sparse_qc=lambda handlers: calls.append('packed8'))
                raise AssertionError('unexpected import ' + str((name, level)))
            namespace = {'__builtins__': dict(vars(builtins), __import__=importer),
                         'os': SimpleNamespace(environ={'EXL3_CK_PAGEDKV': '' if enabled else 'sparse'}),
                         'attn_fns': [], '_fns_qc': []}
            with self.subTest(enabled=enabled, available=available):
                exec(compile(ast.Module(body=imports + [branch], type_ignores=[]), str(SOURCE), 'exec'), namespace)
                self.assertEqual(calls, ['fp16', 'packed8'] if enabled and available else [])


if __name__ == '__main__':
    unittest.main()
