"""CPU test of the decensor preset switch of the Hy4 server (serve_core.ablit_spec): off by default, "on" = the bundled preset, explicit path, off values.
Run: python -m unittest tests/test_hy4_ablit_default.py   (no GPU, no model)"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))
import serve_core as C  # noqa: E402

BUNDLED = os.path.join(TREE, "presets", "hy4-uncensor", "edit_spec.json")


class AblitSwitch(unittest.TestCase):
    def test_default_is_no_edit(self):
        self.assertIsNone(C.ablit_spec({}, TREE))

    def test_on_values_give_the_bundled_preset(self):
        for v in ("on", "ON", "1", "preset", "true", "yes", " on "):
            self.assertEqual(C.ablit_spec({"HY4_ABLIT_SPEC": v}, TREE), BUNDLED, v)
        self.assertTrue(os.path.isfile(BUNDLED))
        self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(BUNDLED), "edit_spec.safetensors")))

    def test_off_values_give_no_spec(self):
        for v in ("0", "", "  ", "off", "OFF", "false", "no", "none"):
            self.assertIsNone(C.ablit_spec({"HY4_ABLIT_SPEC": v}, TREE), v)

    def test_explicit_path_wins(self):
        self.assertEqual(C.ablit_spec({"HY4_ABLIT_SPEC": "/some/other/edit_spec.json"}, TREE), "/some/other/edit_spec.json")

    def test_missing_bundle_gives_no_spec(self):
        self.assertIsNone(C.ablit_spec({"HY4_ABLIT_SPEC": "on"}, os.path.join(TREE, "does_not_exist")))

    def test_cwd_does_not_matter(self):
        old = os.getcwd()
        try:
            os.chdir("/")
            self.assertEqual(C.ablit_spec({"HY4_ABLIT_SPEC": "on"}, TREE), BUNDLED)
        finally:
            os.chdir(old)

    def test_not_in_shipped_env(self):
        src = open(os.path.join(TREE, "tools", "hy4", "serve.py"), encoding="utf-8").read()
        a = src.index("SHIPPED_ENV = {")
        self.assertNotIn("HY4_ABLIT_SPEC", src[a:src.index("\n}\n", a)])


if __name__ == "__main__":
    unittest.main()
