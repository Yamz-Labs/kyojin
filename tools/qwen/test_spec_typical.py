# CPU tests of the relaxed ("typical") acceptance rule in generator/spec_sample.py (no GPU, no model).   python tools/qwen/test_spec_typical.py
#  1. rule boundary: p >= min(eps, delta*exp(-H)) keeps the draft, just below does not
#  2. H = 0 (one-hot law): threshold is min(eps, delta) and the draft that carries all the mass is kept; any other draft has p = 0
#  3. tokens cut by top-k / top-p have p = 0 and are never kept, whatever eps and delta
#  4. flag off (default): EXL3_SPEC_TYPICAL unset -> typical_on() is False, the verify loop never calls typical_rows (code path guarded)
#  5. greedy jobs (SS_Argmax) are not eligible, so the rule never applies to them
import math, os, sys, unittest
os.environ.setdefault("HIP_VISIBLE_DEVICES", ""); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from exllamav3.generator import spec_sample as ss
from exllamav3.generator.sampler.custom import SS_Sort, SS_TopK, SS_TopP, SS_Temperature, SS_Normalize, SS_Sample, SS_Argmax, CustomSampler

V = 16
PRM_FULL = ("steps", [SS_Temperature(1.0), SS_Normalize()], V)


def row_from_probs(p):
    p = torch.tensor(p, dtype=torch.float)
    return torch.log(p).view(1, -1)   # log(0) = -inf: the token is outside the law


class TypicalRule(unittest.TestCase):
    def test_boundary(self):
        # law (0.5, 0.3, 0.2, 0 ...): H known; threshold for the draft with p = 0.2
        p = [0.5, 0.3, 0.2] + [0.0] * (V - 3)
        H = -sum(x * math.log(x) for x in p if x > 0)
        eps, delta = 0.09, 0.3
        thr = min(eps, delta * math.exp(-H))
        rows = row_from_probs(p)
        for d, want in ((2, True), (1, True), (0, True)):
            pp, hh, ok = ss.typical_rows(rows, [d], PRM_FULL, eps, delta)
            self.assertAlmostEqual(hh[0], H, places=4)
            self.assertEqual(ok[0], want)
        # push the threshold above the draft's probability: kept iff p >= thr
        pp, hh, ok = ss.typical_rows(rows, [2], PRM_FULL, 1.0, 1.0)   # thr = exp(-H) ~ 0.31 > 0.2
        self.assertFalse(ok[0])
        pp, hh, ok = ss.typical_rows(rows, [2], PRM_FULL, 0.2 - 1e-4, 10.0)   # thr = eps = 0.1999
        self.assertTrue(ok[0])
        pp, hh, ok = ss.typical_rows(rows, [2], PRM_FULL, 0.2 + 1e-3, 10.0)
        self.assertFalse(ok[0])

    def test_entropy_zero(self):
        p = [1.0] + [0.0] * (V - 1)
        rows = row_from_probs(p)
        pp, hh, ok = ss.typical_rows(rows.repeat(2, 1), [0, 5], PRM_FULL, 0.09, 0.3)
        self.assertAlmostEqual(hh[0], 0.0, places=5)
        self.assertEqual(ok, [True, False])
        self.assertEqual(pp[1], 0.0)

    def test_filtered_tokens_never_kept(self):
        p = [0.4, 0.3, 0.2, 0.05, 0.03, 0.02] + [0.0] * (V - 6)
        rows = row_from_probs(p)
        prm = ("steps", [SS_Temperature(1.0), SS_Sort(), SS_TopK(3), SS_Normalize()], V)
        pp, hh, ok = ss.typical_rows(rows.repeat(3, 1), [3, 4, 5], prm, 1e-9, 1e-9)   # tiny thresholds
        self.assertEqual(pp, [0.0, 0.0, 0.0])
        self.assertEqual(ok, [False, False, False])
        pp, hh, ok = ss.typical_rows(rows, [2], prm, 0.09, 0.3)
        self.assertTrue(ok[0] and pp[0] > 0.2)

    def test_topk_fast_path_equals_dense(self):
        g = torch.Generator().manual_seed(3)
        Vb = 200
        rows = torch.randn(6, Vb, generator=g) * 2.0
        prm = ("steps", [SS_Temperature(1.0), SS_Sort(), SS_TopK(20), SS_Normalize(), SS_Sort(), SS_TopP(0.95)], Vb)
        drafts = [int(rows[i].argmax()) if i % 2 == 0 else int(rows[i].argsort()[-5]) for i in range(6)]
        drafts[5] = int(rows[5].argsort()[0])   # far outside the top-K: p = 0
        pp, hh, ok = ss.typical_rows(rows, drafts, prm, 0.09, 0.3)
        for i, d in enumerate(drafts):
            pl = ss.law(rows[i], prm)
            H = float(-(pl * torch.log(pl.clamp_min(1e-30))).sum())
            self.assertAlmostEqual(pp[i], float(pl[d]), places=5)
            self.assertAlmostEqual(hh[i], H, places=4)
        self.assertEqual(pp[5], 0.0)
        self.assertFalse(ok[5])

    def test_default_off(self):
        self.assertFalse(ss.TYPICAL)
        self.assertFalse(ss.typical_on())

    def test_greedy_not_eligible(self):
        class J:
            filters = []; forced_ids = None; return_probs = False; return_top_tokens = False; device_logit_mask = None; sequences = [1]
            sampler = CustomSampler([SS_Argmax()])
            class generator:
                class tokenizer: actual_vocab_size = V
        self.assertIsNone(ss.eligible(J()))


if __name__ == "__main__":
    unittest.main()
