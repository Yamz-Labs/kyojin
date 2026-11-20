"""CPU test of the decensor preset default of the Hy4 server (serve_core.ablit_spec): bundled preset when unset, off switches, explicit path wins.
Run: python -m unittest tests/test_hy4_ablit_default.py   (no GPU, no model)"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))
import serve_core as C  # noqa: E402


class AblitDefault(unittest.TestCase):
    def test_default_is_the_bundled_preset(self):
        p = C.ablit_spec({}, TREE)
        self.assertEqual(p, os.path.join(TREE, "presets", "hy4-uncensor", "edit_spec.json"))
        self.assertTrue(os.path.isfile(p))
        self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(p), "edit_spec.safetensors")))

    def test_off_values_give_no_spec(self):
        for v in ("0", "", "  ", "off", "OFF", "false", "no", "none"):
            self.assertIsNone(C.ablit_spec({"HY4_ABLIT_SPEC": v}, TREE), v)

    def test_explicit_path_wins(self):
        self.assertEqual(C.ablit_spec({"HY4_ABLIT_SPEC": "/some/other/edit_spec.json"}, TREE), "/some/other/edit_spec.json")

    def test_missing_bundle_gives_no_spec(self):
        self.assertIsNone(C.ablit_spec({}, os.path.join(TREE, "does_not_exist")))

    def test_cwd_does_not_matter(self):
        old = os.getcwd()
        try:
            os.chdir("/")
            self.assertTrue(os.path.isfile(C.ablit_spec({}, TREE)))
        finally:
            os.chdir(old)


if __name__ == "__main__":
    unittest.main()
