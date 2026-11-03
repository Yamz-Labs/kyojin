# CPU tests of RowAlloc in spec_policy.py (no GPU, no model).   python tools/qwen/test_row_alloc.py
#  1. cost table: exact at the points, linear between, extrapolated beyond
#  2. width: fully accepted lookup drafts take the whole row budget, rejected drafts shrink it to a plain step
#  3. observe(): acceptance feedback moves the width; positions never tried inherit the rate before (a shallow width cannot freeze a prior)
#  4. a round of lookup drafts only is charged one drafter forward, a drafter round one per depth
import os, sys, unittest
os.environ.setdefault("HIP_VISIBLE_DEVICES", ""); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_policy as sp

PTS = sp.parse_row_cost(sp.ROW_COST_DENSE_CUDA)

class T(unittest.TestCase):
    def test_cost(self):
        self.assertAlmostEqual(sp.interp_cost(PTS, 16), 36.6)
        self.assertAlmostEqual(sp.interp_cost(PTS, 28), (48.3 + 55.4) / 2)
        self.assertGreater(sp.interp_cost(PTS, 40), 55.4)
        self.assertAlmostEqual(sp.interp_cost(PTS, 1), 21.5)

    def test_width(self):
        ra = sp.RowAlloc(PTS, 32); jobs = list("abcd")
        self.assertEqual(ra.best(jobs, ["ng"] * 4, [7] * 4), 7)             # priors are optimistic: full budget, 4 x 8 rows
        for _ in range(12):                                               # nothing is ever accepted
            for j in jobs:
                ra.cond(j, "mtp"); ra.pend[j] = (10, "mtp", 3); ra.observe(j, 11)
        self.assertEqual(ra.best(jobs, ["mtp"] * 4, [7] * 4), 0)           # a plain step beats paying rows for rejected drafts
        self.assertGreaterEqual(ra.best(jobs, ["mtp"] * 4, [7] * 4, wmin=1), 1)

    def test_observe(self):
        ra = sp.RowAlloc(PTS, 32)
        ra.pend["a"] = (10, "mtp", 4)
        self.assertEqual(ra.observe("a", 10 + 2 + 1), ("mtp", 4, 2))        # 2 accepted + the bonus token
        c = ra.c[("a", "mtp")]
        self.assertEqual([n for _, n in c[:5]], [1, 1, 1, 0, 0])             # tried: positions 1, 2 (accepted) and 3 (rejected)
        self.assertEqual(c[2][0], 0.0)
        eff = ra.cond("a", "mtp")
        self.assertEqual(eff[3], eff[2])                                  # an untried position inherits the rate before it
        self.assertIsNone(ra.observe("a", 99))                            # no pending round: nothing to score

    def test_shallow_width_does_not_freeze(self):
        ra = sp.RowAlloc(PTS, 32); jobs = list("abcd")
        for _ in range(10):
            W = ra.best(jobs, ["ng"] * 4, [7] * 4)
            for j in jobs: ra.pend[j] = (0, "ng", W); ra.observe(j, W + 1)  # everything drafted is accepted
        self.assertEqual(ra.best(jobs, ["ng"] * 4, [7] * 4), 7)

    def test_drafter_depth_charge(self):
        ra = sp.RowAlloc(PTS, 32)
        self.assertAlmostEqual(ra.ms(4, 7, ["ng"] * 4) - ra.ms(4, 7, ["mtp"] * 4), -6 * ra.dms)
        self.assertAlmostEqual(ra.ms(4, 0, ["mtp"] * 4), sp.interp_cost(PTS, 4) + ra.ovh)

if __name__ == "__main__":
    unittest.main()
