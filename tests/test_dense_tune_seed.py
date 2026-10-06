import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exllamav3.model.dense_warmup import seed_dense_tune


class SeedDenseTune(unittest.TestCase):
    def test_adds_missing_keys_once(self):
        with tempfile.TemporaryDirectory() as d:
            seed = os.path.join(d, "seed.txt")
            tune = os.path.join(d, "sub", "tune.txt")
            open(seed, "w").write("t|g|exact1 512 2560 4352 3 0 -5\nt|g|exact1 640 2560 4352 1 0 7\n")
            self.assertEqual(seed_dense_tune(seed, tune), 2)
            self.assertEqual(seed_dense_tune(seed, tune), 0)
            open(tune, "a").write("t|g|exact1 99 99 256 1 0 0\n")
            self.assertEqual(len(open(tune).read().splitlines()), 3)

    def test_existing_key_kept(self):
        with tempfile.TemporaryDirectory() as d:
            seed = os.path.join(d, "seed.txt")
            tune = os.path.join(d, "tune.txt")
            open(seed, "w").write("t|g|exact1 512 2560 4352 3 0 -5\n")
            open(tune, "w").write("t|g|exact1 512 2560 4352 3 0 11\n")
            self.assertEqual(seed_dense_tune(seed, tune), 0)
            self.assertEqual(open(tune).read().strip().split()[-1], "11")

    def test_shipped_seed_parses(self):
        for ln in open(os.path.join(os.path.dirname(__file__), "..", "exllamav3", "model", "dense_gemm_tune_seed.txt")):
            self.assertEqual(len(ln.split()), 7)


if __name__ == "__main__":
    unittest.main()
