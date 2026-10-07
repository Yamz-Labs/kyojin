# CPU tests of the depth rules in spec_policy.py (no GPU, no model).   python tools/qwen/test_spec_rule.py
#  1. ProductRule == the pre-change inline rule (reference copy below) on random probability chains
#  2. apply_rule (truncation form) == early-stop form, for every rule
#  3. install() on a fake generator: early stop / fixed depth / log arm keep the same drafts; defaults are 0.6 / 0.3 / ndt
#  4. OnlineRule: scale rises when drafts are accepted more than predicted, falls in the opposite case, stays clipped
import os, sys, random, unittest
os.environ.setdefault("HIP_VISIBLE_DEVICES", ""); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
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
        try: self.assertEqual((sp.rule_from_env(0.6, 0.3, 7).name, sp.rule_from_env(0.6, 0.3, 7).thfd), ("tier", 0.9))
        finally: del os.environ["QWSPEC_RULE"]

    def test_env_defaults(self):
        for k in ("QWSPEC_THF", "QWSPEC_THV", "QWSPEC_MAXD", "QWSPEC_RULE"): os.environ.pop(k, None)
        self.assertEqual(sp.env_defaults(7), (0.6, 0.3, 7))
        r = sp.rule_from_env(0.6, 0.3, 7); self.assertEqual((r.name, r.thf, r.thv, r.maxd, r.thfd, r.thvd), ("tier", 0.6, 0.3, 7, 0.9, 0.5))
        os.environ["QWSPEC_RULE"] = "product"
        try: r = sp.rule_from_env(0.6, 0.3, 3); self.assertEqual((r.name, r.thf, r.thv, r.maxd), ("product", 0.6, 0.3, 3))
        finally: del os.environ["QWSPEC_RULE"]
        os.environ["QWSPEC_THF"] = "0.5"; os.environ["QWSPEC_MAXD"] = "5"
        try: self.assertEqual(sp.env_defaults(3), (0.5, 0.3, 5))
        finally: del os.environ["QWSPEC_THF"]; del os.environ["QWSPEC_MAXD"]

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
