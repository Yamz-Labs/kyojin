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


class EntropyGate(unittest.TestCase):
    """EXL3_SPEC_TYPICAL_HMAX: relaxation only at rows with H <= HMAX."""
    P = [0.5, 0.3, 0.2] + [0.0] * (V - 3)
    H = -sum(x * math.log(x) for x in P if x > 0)

    def setUp(self): self.old = ss.TYP_HMAX
    def tearDown(self): ss.TYP_HMAX = self.old

    def run_rule(self):
        return ss.typical_rows(row_from_probs(self.P), [2], PRM_FULL, 0.09, 0.3)

    def test_default_off(self):
        self.assertIsNone(self.old)
        ss.TYP_HMAX = None
        pp, hh, ok = self.run_rule()
        self.assertTrue(ok[0])

    def test_above_hmax_never_relaxed(self):
        ss.TYP_HMAX = self.H - 0.01
        pp, hh, ok = self.run_rule()
        self.assertGreater(pp[0], 0.09)   # high p, still refused
        self.assertFalse(ok[0])

    def test_below_hmax_as_before(self):
        ss.TYP_HMAX = self.H + 0.01
        self.assertTrue(self.run_rule()[2][0])
        # a draft under the threshold stays refused
        pp, hh, ok = ss.typical_rows(row_from_probs(self.P), [2], PRM_FULL, 1.0, 1.0)
        self.assertFalse(ok[0])

    def test_greedy_not_eligible(self):
        class J:
            filters = []; forced_ids = None; return_probs = False; return_top_tokens = False; device_logit_mask = None; sequences = [1]
            sampler = CustomSampler([SS_Argmax()])
            class generator:
                class tokenizer: actual_vocab_size = V
        ss.TYP_HMAX = 0.1
        self.assertIsNone(ss.eligible(J()))


class CopiedDrafts(unittest.TestCase):
    """EXL3_SPEC_TYPICAL_LOOKUP: copied drafts keep the exact rule; model-head drafts stay relaxed."""
    class Gen:
        draft_ids_pinned = torch.zeros(2, 4, dtype=torch.long)

    def setUp(self):
        self.old = ss.TYPICAL_LOOKUP

    def tearDown(self):
        ss.TYPICAL_LOOKUP = self.old

    def test_default_is_exact_for_copies(self):
        self.assertEqual(ss.TYPICAL_LOOKUP, "rep")

    def test_model_head_vs_copy(self):
        g = self.Gen()
        mtp = g.draft_ids_pinned[:, :3]            # drafter round: slice of the pinned buffer
        copy = torch.tensor([[1, 2, 3]])           # lookup round: tensor of its own
        for flag, want_copy in (("0", False), ("1", True), ("rep", True)):
            ss.TYPICAL_LOOKUP = flag
            self.assertTrue(ss.relax_allowed(g, mtp, 0))          # MTP unchanged in both modes
            self.assertEqual(ss.relax_allowed(g, copy, 0), want_copy)

    def test_mixed_rows_marked(self):
        g = self.Gen()
        t = torch.tensor([[1, 2], [3, 4]])
        g.draft_copy_rows = (t, [True, False])
        ss.TYPICAL_LOOKUP = "0"
        self.assertFalse(ss.relax_allowed(g, t, 0))
        self.assertTrue(ss.relax_allowed(g, t, 1))
        g.draft_copy_rows = (torch.tensor([[9]]), [False, False])   # stale mark for another tensor: ignored
        self.assertFalse(ss.relax_allowed(g, t, 1))

    def test_copy_never_relaxed_end_to_end(self):
        # the generator keeps a relaxed draft only when relax_allowed; a draft that rule 1 would keep (ok) is dropped at 0
        p = [0.5, 0.3, 0.2] + [0.0] * (V - 3)
        _, _, ok = ss.typical_rows(row_from_probs(p), [2], PRM_FULL, 0.09, 0.3)
        self.assertTrue(ok[0])
        g = self.Gen(); copy = torch.tensor([[2]])
        ss.TYPICAL_LOOKUP = "0"
        self.assertFalse(ok[0] and ss.relax_allowed(g, copy, 0))
        ss.TYPICAL_LOOKUP = "1"
        self.assertTrue(ok[0] and ss.relax_allowed(g, copy, 0))

    def test_greedy_unchanged(self):
        # greedy jobs are never eligible, so neither rule is reached whatever the flag
        class J:
            filters = []; forced_ids = None; return_probs = False; return_top_tokens = False; device_logit_mask = None; sequences = [1]
            sampler = CustomSampler([SS_Argmax()])
            class generator:
                class tokenizer: actual_vocab_size = V
        for flag in ("0", "1", "rep"):
            ss.TYPICAL_LOOKUP = flag
            self.assertIsNone(ss.eligible(J()))


class FakeSeq:
    def __init__(self, ids): self.ids = list(ids)
    def __len__(self): return len(self.ids)
    def torch_slice(self, a, b): return torch.tensor(self.ids[a:b], dtype=torch.long)


class FakeJob:
    def __init__(self, prompt, gen=()):
        self.seq = FakeSeq(list(prompt) + list(gen)); self.new_tokens = len(gen)
        class S: pass
        s = S(); s.sequence_ids = self.seq; self.sequences = [s]
    def emit(self, t): self.seq.ids.append(t); self.new_tokens += 1


class RepetitionAware(unittest.TestCase):
    """Default "rep": the relaxed rule is refused for a draft that completes an n-gram the job already repeated >= 2 times."""
    def setUp(self):
        self.old = (ss.TYPICAL_LOOKUP, ss.REP_N, ss.REP_W); ss.TYPICAL_LOOKUP = "rep"; ss.REP_N = 8; ss.REP_W = 512
        self.n0 = ss.REP_REFUSED[0]

    def tearDown(self):
        ss.TYPICAL_LOOKUP, ss.REP_N, ss.REP_W = self.old

    def test_prompt_copy_stays_relaxed(self):
        prompt = list(range(100, 160))
        gen = [5, 6, 7, 8, 9, 10, 11, 12]       # output so far; the draft continues a span that exists only in the prompt
        j = FakeJob(prompt, gen)
        self.assertFalse(ss.rep_refused(j, 13))
        self.assertEqual(ss.REP_REFUSED[0], self.n0)

    def test_own_output_repeat_refused(self):
        loop = list(range(200, 220))             # a 20-token span emitted twice, third copy starts
        j = FakeJob([1, 2, 3], loop + loop + loop[:10])
        self.assertTrue(ss.rep_refused(j, loop[10]))      # would continue the third copy of a span seen twice
        self.assertFalse(ss.rep_refused(j, 999))          # a different token is not a repeat
        self.assertEqual(ss.REP_REFUSED[0], self.n0 + 1)
        self.assertEqual(j.rep_refused, 1)

    def test_seen_once_not_refused(self):
        loop = list(range(200, 220))
        j = FakeJob([1], loop + loop[:10])              # the span seen only once so far
        self.assertFalse(ss.rep_refused(j, loop[10]))

    def test_incremental_and_equality(self):
        loop = list(range(200, 220))
        j = FakeJob([1], loop + loop[:3])
        self.assertFalse(ss.rep_refused(j, loop[3]))
        for t in loop[3:] + loop + loop[:10]: j.emit(t)   # tracker catches up from the new tail
        self.assertTrue(ss.rep_refused(j, loop[10]))
        # equality still accepts: the generator only replaces the sampled token by a draft when not refused, so a refused draft
        # reaches acceptance only if the target sampled the same token (nothing in rep_refused touches that path)

    def test_window_evicts(self):
        ss.REP_W = 40
        loop = list(range(200, 220))
        j = FakeJob([1], loop + loop + list(range(300, 360)) + loop[:10])   # the repeats left the window
        self.assertFalse(ss.rep_refused(j, loop[10]))

    def test_other_arms_never_refuse(self):
        loop = list(range(200, 220))
        j = FakeJob([1], loop + loop + loop[:10])
        for arm in ("0", "1"):
            ss.TYPICAL_LOOKUP = arm
            self.assertFalse(ss.rep_refused(j, loop[10]))

    def test_mtp_source_checked_too(self):
        # the check is keyed on the output only: relax_allowed (source gate) is True for a model-head draft, rep_refused still applies
        class G: draft_ids_pinned = torch.zeros(1, 4, dtype=torch.long)
        g = G(); mtp = g.draft_ids_pinned[:, :2]
        loop = list(range(200, 220)); j = FakeJob([1], loop + loop + loop[:10])
        self.assertTrue(ss.relax_allowed(g, mtp, 0))
        self.assertTrue(ss.rep_refused(j, loop[10]))


if __name__ == "__main__":
    unittest.main()
