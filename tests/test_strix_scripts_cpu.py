"""CPU tests of the Strix Halo shell scripts that failed silently: tools/strix_halo/env.sh (the HSA runtime that torch needs on gfx1151: found through the
rocm-sdk wheel when EXL3_ROCM_SDK is unset, a clear error instead of a later segfault when there is none) and hipsp/build.sh (a failed hipcc must fail the
script and leave no stale library). Fake rocm-sdk / hipcc / gcc scripts stand in for the toolchain: no GPU, no ROCm.
Run: python -m pytest tests/test_strix_scripts_cpu.py"""
import os
import shutil
import stat
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASH = shutil.which("bash")


def script(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("#!/bin/bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


@unittest.skipIf(BASH is None, "bash not found")
class EnvSh(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.venv = os.path.join(self.tmp, "venv")
        self.sdk = os.path.join(self.tmp, "sdk")

    def fake_rocm_sdk(self, root):
        script(os.path.join(self.venv, "bin", "rocm-sdk"), f'[ "$1 $2" = "path --root" ] && echo "{root}" || exit 1\n' if root else "exit 1\n")

    def source(self, **env):
        base = {k: v for k, v in os.environ.items() if not k.startswith(("EXL3_", "LD_PRELOAD"))}
        base.update({"EXL3_VENV": self.venv, "EXL3_HSA_SYSTEM_DIRS": os.path.join(self.tmp, "no-system-rocm"), "HOME": self.tmp, **env})
        return subprocess.run([BASH, "-c", 'source tools/strix_halo/env.sh >/dev/null; rc=$?; echo "LD_PRELOAD=$LD_PRELOAD"; exit $rc'], cwd=ROOT, env=base,
                              capture_output=True, text=True, timeout=60)

    def make_lib(self, directory):
        os.makedirs(directory, exist_ok=True)
        lib = os.path.join(directory, "libhsa-runtime64.so.1")
        open(lib, "w").close()
        return lib

    def test_the_rocm_sdk_wheel_is_the_fallback_when_exl3_rocm_sdk_is_unset(self):
        lib = self.make_lib(os.path.join(self.sdk, "lib"))
        self.fake_rocm_sdk(self.sdk)
        r = self.source()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"LD_PRELOAD={lib}", r.stdout)

    def test_exl3_rocm_sdk_wins_over_the_wheel(self):
        mine = self.make_lib(os.path.join(self.tmp, "mine", "lib"))
        self.make_lib(os.path.join(self.sdk, "lib"))
        self.fake_rocm_sdk(self.sdk)
        r = self.source(EXL3_ROCM_SDK=os.path.join(self.tmp, "mine"))
        self.assertIn(f"LD_PRELOAD={mine}", r.stdout)

    def test_exl3_hsa_lib_wins_over_everything(self):
        pinned = self.make_lib(os.path.join(self.tmp, "pinned"))
        self.make_lib(os.path.join(self.sdk, "lib"))
        self.fake_rocm_sdk(self.sdk)
        r = self.source(EXL3_HSA_LIB=pinned)
        self.assertIn(f"LD_PRELOAD={pinned}", r.stdout)

    def test_the_system_rocm_is_tried_last(self):
        lib = self.make_lib(os.path.join(self.tmp, "system-rocm"))
        self.fake_rocm_sdk(None)
        r = self.source(EXL3_HSA_SYSTEM_DIRS=os.path.dirname(lib))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"LD_PRELOAD={lib}", r.stdout)

    def test_no_runtime_is_a_clear_error_not_a_segfault_later(self):
        self.fake_rocm_sdk(None)                              # rocm-sdk present but failing, no EXL3_ROCM_SDK, nothing on the system
        r = self.source()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no libhsa-runtime64.so.1", r.stderr)
        self.assertIn("segfault", r.stderr)
        self.assertIn("LD_PRELOAD=\n", r.stdout)

    def test_no_runtime_without_rocm_sdk_at_all_is_the_same_error(self):
        r = self.source(PATH=os.path.dirname(BASH) + os.pathsep + "/usr/bin" + os.pathsep + "/bin")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no libhsa-runtime64.so.1", r.stderr)

    def test_allow_no_hsa_only_warns(self):
        self.fake_rocm_sdk(None)
        r = self.source(EXL3_ALLOW_NO_HSA="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("WARNING", r.stderr)


@unittest.skipIf(BASH is None, "bash not found")
class HipspBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.venv, self.sdk = os.path.join(self.tmp, "venv"), os.path.join(self.tmp, "sdk")
        script(os.path.join(self.venv, "bin", "rocm-sdk"), f'echo "{self.sdk}"\n')
        script(os.path.join(self.tmp, "bin", "gcc"), 'echo /usr/lib/gcc/x86_64-linux-gnu/13/libgcc.a\n')
        self.src, self.out = os.path.join(self.tmp, "k.cu"), os.path.join(self.tmp, "libk.so")
        open(self.src, "w").close()

    def hipcc(self, body):
        script(os.path.join(self.sdk, "bin", "hipcc"), body)

    def build(self):
        env = dict(os.environ, PATH=os.path.join(self.tmp, "bin") + os.pathsep + os.environ["PATH"])
        return subprocess.run([BASH, os.path.join(ROOT, "hipsp", "build.sh"), self.src, self.out, self.venv], env=env, capture_output=True, text=True, timeout=60)

    def test_a_failed_compile_fails_the_script_and_leaves_no_stale_library(self):
        open(self.out, "w").write("old library")
        self.hipcc('echo "error: kernel does not compile"; exit 1\n')
        r = self.build()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not compile", r.stdout)
        self.assertFalse(os.path.exists(self.out))

    def test_a_compiler_that_writes_nothing_fails_the_script(self):
        self.hipcc("exit 0\n")
        r = self.build()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("left no library", r.stderr)

    def test_a_good_build_prints_the_library(self):
        self.hipcc('while [ "$1" != "-o" ]; do shift; done; echo built > "$2"\n')
        r = self.build()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("libk.so", r.stdout)
        self.assertTrue(os.path.getsize(self.out) > 0)

    def test_a_missing_sdk_is_a_clear_error(self):
        r = self.build()                                       # no hipcc in the fake SDK
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no hipcc", r.stderr)


if __name__ == "__main__":
    unittest.main()
