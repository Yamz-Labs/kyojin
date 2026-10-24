# CPU tests of the depth rules in spec_policy.py (no GPU, no model).   python tools/qwen/test_spec_rule.py
#  1. ProductRule == the pre-change inline rule (reference copy below) on random probability chains
#  2. apply_rule (truncation form) == early-stop form, for every rule
#  3. install() on a fake generator: early stop / fixed depth / log arm keep the same drafts; defaults are 0.6 / 0.3 / ndt
#  4. OnlineRule: scale rises when drafts are accepted more than predicted, falls in the opposite case, stays clipped
import os, sys, random, unittest
os.environ.setdefault("HIP_VISIBLE_DEVICES", ""); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import spec_policy as sp

def ref_keep(ps, thf=0.6, thv=0.3, maxd=3):
    """the rule exactly as it was inlined in install() before the rule objects: returns (keep, drafted forwards)"""
    reach = 1.0; i = 0
    for p in ps:
        reach *= p
        if i > 0 and reach < thv: return i, i + 1
        i += 1
        if reach < thf or i >= maxd: return i, i
    return i, i

def early(rule, ps):
    """early-stop form as used by sfs: returns (keep, forwards run)"""
    rule.begin(); keep = 0; fw = 0
    for i, p in enumerate(ps):
        fw += 1
        k, more = rule.step(i, p)
        if k: keep = i + 1
        if not more: break
    return keep, fw

class FakeJob:
    def __init__(self): self.sequences = [type("S", (), {"sequence_ids": []})()]
    def is_prefill_done(self): return True

class FakeGen:
    def __init__(self, ndt):
        self.num_draft_tokens = ndt; self.draft_ids_pinned = torch.zeros((1, 8), dtype=torch.long)
        self.job = FakeJob(); self.active_jobs = [self.job]; self.chain = []; self.forwards = 0
    def enqueue(self, job): return None
    def iterate_draftmodel_mtp_gen(self, results):
        for idx in range(self.num_draft_tokens):
            st = self.dm.forward(None, {}); self.lm.forward(None)
            new = self.dm.sample_from_state(st, {})
            self.draft_ids_pinned[:1, idx:idx + 1].copy_(new)
        return self.draft_ids_pinned[:, :self.num_draft_tokens]

class FakeDM:
    def __init__(self, gen, chain): self.gen, self.chain, self.k = gen, chain, 0
    def forward(self, ids, params): self.gen.forwards += 1; return torch.zeros(1)
    def sample_from_state(self, state, params):
        t = torch.tensor([[100 + self.k]]); self.k += 1; return t
class FakeLM:
    def __init__(self, dm): self.dm = dm
    def forward(self, *a, **k):
        p = self.dm.chain[self.dm.k]; v = 16
        lg = torch.full((1, 1, v), -30.0); lg[..., 0] = 0.0
        lg[..., 1:] = torch.log(torch.tensor((1 - p) / (v - 1)) + 1e-30) - torch.log(torch.tensor(max(p, 1e-9)))  # softmax max = p
        return lg
class FakeModel:
    class config: vocab_size = 16
    def __init__(self, lm): self.modules = [lm]; self.logit_layer_idx = 0

def run_install(ps, rounds=1, grow=None, **kw):
    g = FakeGen(kw.pop("ndt", 7)); dm = FakeDM(g, ps); lm = FakeLM(dm)
    g.dm, g.lm = dm, lm; dm.chain = list(ps) + [0.5] * 8
    model = FakeModel(lm)
    # wire the fake as the real thing: install wraps these attributes
    unin = sp.install(g, dm, model, **kw)
    for r in range(rounds):
        dm.k = 0
        out = g.iterate_draftmodel_mtp_gen([])
        if grow is not None: g.job.sequences[0].sequence_ids.extend([0] * (grow(out.shape[-1]) + 1))   # accepted + 1 new tokens
    return out, g

class T(unittest.TestCase):
    def rnd(self, n=3000):
        r = random.Random(1)
        for _ in range(n): yield [r.choice([r.random(), 0.99, 0.9, 0.5]) for _ in range(8)]

    def test_product_matches_reference(self):
        for ps in self.rnd():
            for (thf, thv, maxd) in [(0.6, 0.3, 3), (0.5, 0.2, 5), (0.8, 0.4, 4), (0.0, 0.0, 7)]:
                self.assertEqual(early(sp.ProductRule(thf, thv, maxd), ps), ref_keep(ps, thf, thv, maxd))

    def test_truncation_equals_early(self):
        rules = [lambda: sp.ProductRule(0.6, 0.3, 3), lambda: sp.CostRule(0.5, 0.3, 6, 1.2, 0.9), lambda: sp.OnlineRule(0.5, 0.3, 6)]
        for ps in self.rnd(1000):
            for mk in rules:
                self.assertEqual(sp.apply_rule(mk(), ps), early(mk(), ps)[0])

    def test_install_modes_agree(self):
        for ps in list(self.rnd(200))[:200]:
            ref, _ = ref_keep(ps)[0], None
            out, g = run_install(ps, thf=0.6, thv=0.3, maxd=3, ndt=3)
            self.assertEqual(out.shape[-1], ref)
            self.assertEqual(out[0].tolist(), [100 + i for i in range(ref)])
            self.assertEqual(g.forwards, ref_keep(ps)[1])
            out, g = run_install(ps, rule=sp.ProductRule(0.6, 0.3, 3), fixed=3, ndt=3)
            self.assertEqual(out.shape[-1], ref); self.assertEqual(g.forwards, 3)
            log = []
            out, g = run_install(ps, rule=sp.ProductRule(0.6, 0.3, 3), log=log, ndt=7)
            self.assertEqual(out.shape[-1], ref); self.assertEqual(g.forwards, 7)
            self.assertEqual(log[0]["keep"], ref); self.assertEqual(len(log[0]["ids"]), 7)
            self.assertEqual(len(log[0]["ps"]), 7)

    def test_tier_equals_product_up_to_four_rows(self):
        rng = random.Random(7)
        for _ in range(2000):
            ps = [rng.choice([rng.random(), 1.0 - 1e-3 * rng.random()]) for _ in range(7)]
            self.assertEqual(early(sp.TierRule(0.6, 0.3, 3, 0.9, 0.5), ps), early(sp.ProductRule(0.6, 0.3, 3), ps))
            k, fw = early(sp.TierRule(0.6, 0.3, 7, 0.9, 0.5), ps)
            kp, fwp = early(sp.ProductRule(0.6, 0.3, 3), ps)
            if k <= 3: self.assertEqual(k, kp)                  # same rows as the shipped rule when the deep tier adds none
            else: self.assertEqual(kp, 3)                       # a deep round is always the shipped rule's 4-row round plus extra rows
            self.assertEqual(early(sp.TierRule(0.6, 0.3, 7, 0.6, 0.3), ps), early(sp.ProductRule(0.6, 0.3, 7), ps))

    def test_tier_deep_gate(self):
        r = sp.TierRule(0.6, 0.3, 7, 0.9, 0.5)
        self.assertEqual(early(r, [0.99] * 7), (7, 7))            # confident chain goes to the cap
        self.assertEqual(early(r, [0.95, 0.95, 0.95, 0.9]), (3, 3)) # reach 0.857 < thfd: stops at depth 3, like the default rule
        self.assertEqual(early(r, [0.99, 0.99, 0.99, 0.5, 0.9]), (3, 4))  # depth 4 drafted, reach 0.48 < thvd: dropped
        self.assertEqual(early(r, [0.99, 0.99, 0.99, 0.6, 0.9]), (4, 4))  # kept (0.58 >= 0.5), reach < thfd: stop
        os.environ["QWSPEC_RULE"] = "tier"
        try: self.assertEqual((sp.rule_from_env(0.6, 0.3, 7).name, sp.rule_from_env(0.6, 0.3, 7).thfd), ("tier", 0.95))
        finally: del os.environ["QWSPEC_RULE"]

    def test_env_defaults(self):
        for k in ("QWSPEC_THF", "QWSPEC_THV", "QWSPEC_MAXD", "QWSPEC_RULE"): os.environ.pop(k, None)
        self.assertEqual(sp.env_defaults(7), (0.6, 0.3, 7))
        r = sp.rule_from_env(0.6, 0.3, 7); self.assertEqual((r.name, r.thf, r.thv, r.maxd, r.thfd, r.thvd), ("tier", 0.6, 0.3, 7, 0.95, 0.85))
        os.environ["QWSPEC_RULE"] = "product"
        try: r = sp.rule_from_env(0.6, 0.3, 3); self.assertEqual((r.name, r.thf, r.thv, r.maxd), ("product", 0.6, 0.3, 3))
        finally: del os.environ["QWSPEC_RULE"]
        os.environ["QWSPEC_THF"] = "0.5"; os.environ["QWSPEC_MAXD"] = "5"
        try: self.assertEqual(sp.env_defaults(3), (0.5, 0.3, 5))
        finally: del os.environ["QWSPEC_THF"]; del os.environ["QWSPEC_MAXD"]

    def test_ngram_from_env(self):
        os.environ.pop("QWSPEC_NGRAM", None)
        self.assertEqual(sp.ngram_from_env(), (2, 7, 16, 3))
        for v, want in (("2,3,5", (2, 3, 5)), ("off", None), ("0", None), (" 3, 7, 16 ", (3, 7, 16))):
            os.environ["QWSPEC_NGRAM"] = v
            try: self.assertEqual(sp.ngram_from_env(), want)
            finally: del os.environ["QWSPEC_NGRAM"]
        for v in ("2,7", "17,7,32", "2,0,5", "4,7,3", "2,7,16,3,1", "2,7,16,-1"):
            os.environ["QWSPEC_NGRAM"] = v
            try:
                with self.assertRaises(ValueError): sp.ngram_from_env()
            finally: del os.environ["QWSPEC_NGRAM"]
        for v, want in (("2,7,16,3", (2, 7, 16, 3)), ("2,7,16,0", (2, 7, 16, 0))):
            os.environ["QWSPEC_NGRAM"] = v
            try: self.assertEqual(sp.ngram_from_env(), want)
            finally: del os.environ["QWSPEC_NGRAM"]

    def test_install_lookup(self):
        # lookup-only installer (GLM): a lookup round runs ONE draft forward and returns the copied tokens; any other round
        # is the generator's own, at its own depth; the window is restored
        from exllamav3.util.tensor import SeqTensor
        row = lambda v: list(range(100, 120)) + [v] + [7, 8, 9]
        def gen(ctx):
            g = FakeGen(2); dm = FakeDM(g, [0.9] * 8); g.dm, g.lm = dm, FakeLM(dm); dm.chain = [0.9] * 16
            sid = SeqTensor((1, 1), torch.long, -1, init_cap=256); sid.append(torch.tensor([ctx], dtype=torch.long))
            g.job.sequences[0].sequence_ids = sid
            return g
        st = {}
        g = gen(row(1000) + list(range(100, 118))); un = sp.install_lookup(g, (2, 3, 16, 1), st)
        out = g.iterate_draftmodel_mtp_gen([])
        self.assertEqual((out.tolist(), g.forwards, g.num_draft_tokens, st), ([[118, 119, 1000]], 1, 2, {"ng": 1, "ng_tokens": 3}))
        g = gen(row(1000) + row(2000) + list(range(100, 118))); sp.install_lookup(g, (2, 3, 16, 1))
        self.assertEqual(g.iterate_draftmodel_mtp_gen([]).tolist(), [[118, 119]])            # cut where the two rows differ
        g = gen(row(1000) + row(2000) + list(range(100, 118))); sp.install_lookup(g, (2, 3, 16, 3))
        self.assertEqual((g.iterate_draftmodel_mtp_gen([]).shape[-1], g.forwards), (2, 2))   # under cmin: the drafter's own round
        g = gen(list(range(40))); un = sp.install_lookup(g, (2, 3, 16, 1))
        self.assertEqual((g.iterate_draftmodel_mtp_gen([]).shape[-1], g.forwards), (2, 2))   # no match
        g.active_jobs = [g.job, FakeJob()]
        self.assertEqual(g.iterate_draftmodel_mtp_gen([]).shape[-1], 2)                      # two requests: no lookup
        un(); self.assertNotIn("iterate_draftmodel_mtp_gen", g.__dict__)
        self.assertNotIn("enqueue", g.__dict__)
        # the index is primed at enqueue: the first decode round finds it built (no rebuild)
        g = gen(row(1000) + list(range(100, 118))); fols = []
        orig_sf = sp.SeqFollower
        class Spy(orig_sf):
            def __init__(s, n): orig_sf.__init__(s, n); fols.append(s)
        sp.SeqFollower = Spy
        try: sp.install_lookup(g, (2, 3, 16, 1))
        finally: sp.SeqFollower = orig_sf
        g.enqueue(g.job)
        self.assertEqual((fols[0].sid is g.job.sequences[0].sequence_ids, len(fols[0].idx)), (True, 42))
        self.assertEqual(g.iterate_draftmodel_mtp_gen([]).tolist(), [[118, 119, 1000]])

    def test_plain_lookup_from_env(self):
        # QWSPEC_PLAIN_LOOKUP = "lmin,K[,cmin[,rows]]"; unset / off / 0 / none -> None; rows defaults to the row_budget argument
        os.environ.pop("QWSPEC_PLAIN_LOOKUP", None)
        self.assertIsNone(sp.plain_lookup_from_env())
        for v in ("off", "0", "none", "OFF", " None "):
            os.environ["QWSPEC_PLAIN_LOOKUP"] = v
            try: self.assertIsNone(sp.plain_lookup_from_env())
            finally: del os.environ["QWSPEC_PLAIN_LOOKUP"]
        for v, want in (("2,15", (2, 15, 1, 0)), ("2,15,3", (2, 15, 3, 0)), (" 2 , 15 , 1 , 16 ", (2, 15, 1, 16))):
            os.environ["QWSPEC_PLAIN_LOOKUP"] = v
            try: self.assertEqual(sp.plain_lookup_from_env(), want)
            finally: del os.environ["QWSPEC_PLAIN_LOOKUP"]
        os.environ["QWSPEC_PLAIN_LOOKUP"] = "2,15"                 # 2 fields: rows = the default (row_budget)
        try: self.assertEqual(sp.plain_lookup_from_env(16), (2, 15, 1, 16))
        finally: del os.environ["QWSPEC_PLAIN_LOOKUP"]
        os.environ["QWSPEC_PLAIN_LOOKUP"] = "2,15,3"               # 3 fields: rows = the default
        try: self.assertEqual(sp.plain_lookup_from_env(16), (2, 15, 3, 16))
        finally: del os.environ["QWSPEC_PLAIN_LOOKUP"]
        os.environ["QWSPEC_PLAIN_LOOKUP"] = "2,15,1,4"             # explicit 4th field wins over the default
        try: self.assertEqual(sp.plain_lookup_from_env(16), (2, 15, 1, 4))
        finally: del os.environ["QWSPEC_PLAIN_LOOKUP"]
        for v in ("2", "2,15,1,4,9", "0,15", "2,0", "2,15,-1", "x,15"):
            os.environ["QWSPEC_PLAIN_LOOKUP"] = v
            try:
                with self.assertRaises(ValueError): sp.plain_lookup_from_env()
            finally: del os.environ["QWSPEC_PLAIN_LOOKUP"]

    def test_install_plain_lookup(self):
        # drafter-free lookup: answers iterate_ngram_gen. A match returns its continuation (K cap, cmin cut) as ONE draft per
        # round and needs NO drafter forward; no match = a plain round (no draft); uninstall restores the generator's slot.
        from exllamav3.util.tensor import SeqTensor
        row = lambda v: list(range(100, 120)) + [v] + [7, 8, 9]

        class PJob:
            def __init__(s, ids):
                sid = SeqTensor((1, 1), torch.long, -1, init_cap=256)
                sid.append(torch.tensor([ids], dtype=torch.long))
                s.sequences = [type("S", (), {"sequence_ids": sid})()]
            def is_prefill_done(s): return True

        class PGen:
            def __init__(s):
                s.active_jobs = []; s.forwards = 0
            def enqueue(s, j): s.active_jobs.append(j)
            def iterate_ngram_gen(s, results): s.forwards += 1; return None   # the generator's own slot

        def gen(ids):
            g = PGen(); st = {}; un = sp.install_plain_lookup(g, (2, 7, 1, 0), st); g.enqueue(PJob(ids)); return g, st, un
        # one earlier occurrence: the whole continuation (K = 7), one draft row, no drafter forward
        g1, st1, un = gen(row(1000) + list(range(100, 118)))
        out = g1.iterate_ngram_gen([])
        self.assertEqual((out.tolist(), out.shape[-1], g1.forwards, st1),
                         ([[118, 119, 1000, 7, 8, 9, 100]], 7, 0, {"ng": 1, "ng_tokens": 7}))
        # the two latest occurrences agree on only 2 tokens: cmin 1 cuts the draft to the common prefix
        g2, _, _ = gen(row(1000) + row(2000) + list(range(100, 118)))
        self.assertEqual(g2.iterate_ngram_gen([]).tolist(), [[118, 119]])
        # no match: no draft, and the generator's own slot was never used
        g3, _, _ = gen(list(range(40)))
        self.assertIsNone(g3.iterate_ngram_gen([])); self.assertEqual(g3.forwards, 0)
        self.assertIn("iterate_ngram_gen", g1.__dict__)                        # the lookup answers the slot
        un()
        self.assertNotIn("enqueue", g1.__dict__)
        self.assertNotIn("iterate_ngram_gen", g1.__dict__)                     # the generator's own slot is back
        self.assertEqual((g1.iterate_ngram_gen([]), g1.forwards), (None, 1))

    def test_witness_len(self):
        # a templated list: the constant part is copied, the copy stops at the value that changes from row to row
        row = lambda v: list(range(100, 120)) + [v, v + 1] + [7, 8, 9]
        idx = sp.NgramIndex(2)
        for t in row(1000) + row(2000) + list(range(100, 118)): idx.push(t)
        L, c = idx.match(7)
        self.assertEqual((L >= 16, c), (True, [118, 119, 2000, 2001, 7, 8, 9]))
        self.assertEqual(idx.witness_len(c), 2)
        # one earlier occurrence only: nothing to compare with, the whole continuation stands
        idx = sp.NgramIndex(2)
        for t in row(1000) + list(range(100, 118)): idx.push(t)
        L, c = idx.match(7)
        self.assertEqual(idx.witness_len(c), len(c))
        # against a direct scan on random streams with many repeats
        import random
        rng = random.Random(5)
        for _ in range(300):
            n = rng.randint(20, 400); per = [rng.randrange(4) for _ in range(rng.randint(3, 40))]
            ctx = [per[i % len(per)] if rng.random() > 0.03 else rng.randrange(4) for i in range(n)]
            idx = sp.NgramIndex(2); idx.extend(np.array(ctx))
            L, c = idx.match(7)
            if not L or idx._hit is None: continue
            j, m = idx._hit
            j2 = next((e for e in range(j - 1, m - 1, -1) if ctx[e - m:e] == ctx[j - m:j]), -1)
            want = len(c) if j2 < 0 else next((a for a in range(len(c)) if ctx[j2 + a] != c[a]), len(c))
            self.assertEqual(idx.witness_len(c), want)

    def test_online_scale(self):
        r = sp.OnlineRule(0.5, 0.3, 6)
        for _ in range(60):
            r.begin(); r.step(0, 0.7); r.step(1, 0.7); r.end_round(2, 2)    # predicted 0.7 + 0.49, got 2
        self.assertAlmostEqual(r.scale, 1.6)                          # clipped at hi
        r.new_request(); self.assertEqual(r.scale, 1.0)
        for _ in range(60):
            r.begin(); r.step(0, 0.9); r.step(1, 0.9); r.end_round(2, 0)
        self.assertAlmostEqual(r.scale, 0.5)                          # clipped at lo

class TT(unittest.TestCase):
    def test_table_depth_follows_acceptance(self):
        r = sp.TableRule(7)
        d0 = r.choose()                                     # optimistic prior
        for _ in range(40): r.end_round(d0 if False else 3, 0)   # nothing is ever accepted
        self.assertEqual(r.choose(), 1)                     # never below one draft
        r.new_request()
        for _ in range(40):
            D = r.choose(); r.end_round(D, D)               # everything accepted: drafts as deep as allowed
        self.assertEqual(r.choose(), 7)

    def test_table_probe_after_full_accept(self):
        r = sp.TableRule(7); r.a = [0.75] + [0.2] * 7       # low acceptance: best depth 1
        self.assertEqual(r.choose(), 1)
        r.end_round(1, 1); self.assertEqual(r.choose(), 2)  # censored full accept -> probe one deeper
        r.end_round(2, 0); self.assertEqual(r.choose(), 1)

    def test_table_install_draws_chosen_depth_without_reading_p(self):
        out, g = run_install([0.9] * 8, rule=sp.TableRule(5), ndt=5)
        self.assertEqual(g.forwards, out.shape[-1])
        self.assertEqual(out.shape[-1], sp.TableRule(5).choose())

class TE(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(early(sp.ProductRule(0.0, 0.5, 4), [1.0, 0.5, 1.0]), (3, 3))   # reach == thv is kept
        self.assertEqual(early(sp.ProductRule(0.0, 0.5, 4), [1.0, 0.49, 1.0]), (1, 2))  # below: dropped, its forward ran
        self.assertEqual(early(sp.ProductRule(0.5, 0.0, 4), [0.5, 1.0]), (2, 2))        # reach == thf keeps drafting

    def test_calibrated_reach(self):
        self.assertEqual(early(sp.CostRule(0.0, 0.3, 3, gamma=1.0, scale=2.0), [1.0, 0.2, 1.0]), (3, 3))   # 0.4 >= 0.3 with scale 2
        self.assertEqual(early(sp.CostRule(0.0, 0.3, 3, gamma=1.0, scale=1.0), [1.0, 0.2, 1.0]), (1, 2))   # 0.2 < 0.3
        self.assertEqual(early(sp.CostRule(0.0, 0.3, 3, gamma=2.0, scale=1.0), [1.0, 0.7, 1.0]), (3, 3))   # 0.49
        self.assertEqual(early(sp.CostRule(0.0, 0.3, 3, gamma=2.0, scale=1.0), [1.0, 0.5, 1.0]), (1, 2))   # 0.25

    def test_online_pred_counts_kept_depths_only(self):
        r = sp.OnlineRule(0.0, 0.5, 4); r.begin(); r.step(0, 0.9); r.step(1, 0.4)     # second depth dropped
        self.assertAlmostEqual(r.pred, 0.9)

class TI(unittest.TestCase):
    def test_online_adapts_through_install(self):
        rule = sp.OnlineRule(0.5, 0.3, 5)
        out, g = run_install([0.6] * 8, rounds=30, grow=lambda d: d, rule=rule, ndt=5)     # every kept draft is accepted
        self.assertGreater(rule.scale, 1.2)
        rule = sp.OnlineRule(0.5, 0.3, 5)
        out, g = run_install([0.9] * 8, rounds=30, grow=lambda d: 0, rule=rule, ndt=5)     # none accepted
        self.assertLess(rule.scale, 0.8)

    def test_table_adapts_through_install(self):
        rule = sp.TableRule(7)
        out, g = run_install([0.9] * 8, rounds=40, grow=lambda d: 0, rule=rule, ndt=7)
        self.assertEqual(out.shape[-1], 1)
        rule = sp.TableRule(7)
        out, g = run_install([0.9] * 8, rounds=40, grow=lambda d: d, rule=rule, ndt=7)
        self.assertEqual(out.shape[-1], 7)

    def test_num_draft_tokens_restored(self):
        rule = sp.TableRule(7)
        out, g = run_install([0.9] * 8, rounds=3, grow=lambda d: 0, rule=rule, ndt=7)
        self.assertEqual(g.num_draft_tokens, 7)
        out, g = run_install([0.9] * 8, rule=sp.ProductRule(0.6, 0.3, 3), ndt=3); self.assertEqual(g.num_draft_tokens, 3)

if __name__ == "__main__": unittest.main(verbosity=1)
