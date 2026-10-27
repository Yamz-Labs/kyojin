"""Real local Git patch regressions; stdlib only, no commit or build.

Run directly: python3 -I -S -B tests/test_ck_pagedkv_patch_cpu.py -v
Git accepts a bare-empty unified context line, but emits a single-space
marker in its canonical diff. Both spellings must retain exact stage ownership.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import unittest


_BUILDER = Path(__file__).resolve().parents[1] / 'tools/strix_halo/ck_pagedkv_build.py'
_namespace: dict[str, Any] = {'__file__': str(_BUILDER), '__name__': '_ck_patch_test'}
exec(compile(_BUILDER.read_bytes(), str(_BUILDER), 'exec'), _namespace)
apply_patch = _namespace['apply_patch']
BuildError = _namespace['BuildError']


@unittest.skipUnless(shutil.which('git'), 'git required')
class PatchPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ck-patch-', dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.original = b'alpha\n\nbefore\nomega\n'
        self.patched = b'alpha\n\nafter\nomega\n'
        (self.repo / 'owned.txt').write_bytes(self.original)
        (self.repo / 'other.txt').write_bytes(b'unrelated\n')
        self.git('add', '--', 'owned.txt', 'other.txt')
        # A tree object suffices for diff HEAD; no commit object is created.
        tree = self.git('write-tree').stdout.strip()
        (self.repo / '.git/HEAD').write_bytes(tree + b'\n')
        (self.repo / 'owned.txt').write_bytes(self.patched)
        self.canonical = self.git('diff', '--no-ext-diff', '--no-color', '--full-index', 'HEAD', '--').stdout
        self.assertEqual(self.canonical.count(b'\n \n'), 1)
        self.compact = self.canonical.replace(b'\n \n', b'\n\n')
        self.assertNotEqual(self.compact, self.canonical)
        self.patch = self.root / 'change.patch'
        self.patch.write_bytes(self.compact)
        (self.repo / 'owned.txt').write_bytes(self.original)

    def git(self, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
                   GIT_OPTIONAL_LOCKS='0', GIT_NO_REPLACE_OBJECTS='1',
                   GIT_NO_LAZY_FETCH='1', GIT_ALLOW_PROTOCOL='file', LC_ALL='C')
        result = subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(self.repo), *args],
                                env=env, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        return result

    def test_bare_empty_context_applies_and_retries(self):
        self.git('apply', '--check', '--whitespace=error-all', str(self.patch))
        self.assertEqual(apply_patch(self.repo, self.patch), 'applied')
        self.assertEqual((self.repo / 'owned.txt').read_bytes(), self.patched)
        self.assertEqual(apply_patch(self.repo, self.patch), 'already-applied')
        self.assertEqual((self.repo / 'owned.txt').read_bytes(), self.patched)

    def test_canonical_context_applies_and_retries(self):
        self.patch.write_bytes(self.canonical)
        self.assertEqual(apply_patch(self.repo, self.patch), 'applied')
        self.assertEqual(apply_patch(self.repo, self.patch), 'already-applied')

    def test_unrelated_tracked_dirt_refused_before_apply_and_retry(self):
        other = self.repo / 'other.txt'
        for patched in (False, True):
            with self.subTest(patched=patched):
                if patched:
                    self.git('apply', str(self.patch))
                other.write_bytes(b'unrelated dirty\n')
                before = (self.repo / 'owned.txt').read_bytes()
                with self.assertRaisesRegex(BuildError, 'changes beyond the exact pinned patch'):
                    apply_patch(self.repo, self.patch)
                self.assertEqual((self.repo / 'owned.txt').read_bytes(), before)
                self.assertEqual(other.read_bytes(), b'unrelated dirty\n')
                other.write_bytes(b'unrelated\n')

    def test_source_whitespace_dirt_is_not_normalized(self):
        self.git('apply', str(self.patch))
        dirty = self.patched.replace(b'alpha\n', b'alpha \n')
        (self.repo / 'owned.txt').write_bytes(dirty)
        with self.assertRaisesRegex(BuildError, 'changes beyond the exact pinned patch'):
            apply_patch(self.repo, self.patch)
        self.assertEqual((self.repo / 'owned.txt').read_bytes(), dirty)

    def test_untracked_dirt_refused(self):
        extra = self.repo / 'extra.txt'
        extra.write_bytes(b'untracked\n')
        with self.assertRaisesRegex(BuildError, 'untracked source files'):
            apply_patch(self.repo, self.patch)
        self.assertEqual((self.repo / 'owned.txt').read_bytes(), self.original)
        self.assertEqual(extra.read_bytes(), b'untracked\n')


if __name__ == '__main__':
    unittest.main()
