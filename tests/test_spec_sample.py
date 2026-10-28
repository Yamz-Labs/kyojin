# CPU tests of exllamav3/generator/spec_sample.py (no GPU, no model).   python tests/test_spec_sample.py
#  1. law() of a step-by-step ComboSampler stack == an fp64 reference (softmax, sort, top-K, top-P cumsum <= p), padding gets 0
#  2. fused_law() == a slow reference of the fused kernel's truncation, with tie groups
#  3. draft d ~ q, verify(): emitted tokens follow p (chi-square), keep rate == sum min(p, q), never outside p's support
#  4. eligible(): sampled ComboSampler jobs qualify, greedy / masked / forced jobs do not
import os, sys, math, random, unittest
os.environ.setdefault("HIP_VISIBLE_DEVICES", ""); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hardware import require
require("ext")
from exllamav3.generator import spec_sample as ss
from exllamav3.generator.sampler import ComboSampler, ArgmaxSampler

def steps_prm(sampler, vs):
    return ("steps", sampler.steps[:-1], vs)

def ref_steps(l, T, k, tp):
    pr = torch.softmax(l.double() / T, 0); sp, si = torch.sort(pr, descending = True)
    if k: sp[k:] = 0; sp = sp / sp.sum()
    if tp < 1: m = torch.cumsum(sp, 0) <= tp; m[0] = True; sp = sp * m
    out = torch.zeros_like(pr); out[si] = sp
    return (out / out.sum()).float()

def ref_fused(l, prm):
    inv_t, inv_tf, filt, k, tp = prm
    v, order = torch.sort(l.double(), descending = True); v = v.tolist(); order = order.tolist()
    kept = [True] * len(v)
    if filt & 1 and 0 < k < len(v): kept = [x >= v[k - 1] for x in v]
    if filt & 2 and tp < 1:
        w = [math.exp((x - v[0]) * inv_tf) if kk else 0.0 for x, kk in zip(v, kept)]; z = sum(w); acc = 0.0; c = len(v) - 1
        for i, x in enumerate(w):
            acc += x
            if acc > tp * z: c = i; break
        kept = [kk and (x > v[c] or x == v[0]) for x, kk in zip(v, kept)]
    pw = [math.exp((x - v[0]) * inv_t) if kk else 0.0 for x, kk in zip(v, kept)]; z = sum(pw)
    out = [0.0] * len(v)
    for i, x in zip(order, pw): out[i] = x / z
    return torch.tensor(out)

class FakeJob:
    def __init__(self, sampler, **kw):
        self.sampler = sampler; self.filters = []; self.forced_ids = None; self.return_probs = False
        self.return_top_tokens = 0; self.device_logit_mask = None; self.sequences = [None]
        self.generator = type("G", (), {"tokenizer": type("T", (), {"actual_vocab_size": 100})()})()
        self.__dict__.update(kw)

class TestSpecSample(unittest.TestCase):

    def test_steps_law(self):
        g = torch.Generator().manual_seed(1)
        for T, k, tp in ((1.0, 0, 0.95), (1.0, 20, 0.95), (0.7, 0, 0.9)):
            prm = steps_prm(ComboSampler(temperature = T, top_k = k, top_p = tp), 300)
            for _ in range(100):
                l = torch.randn(310, generator = g) * (0.5 + 3 * torch.rand(1, generator = g))
                p = ss.law(l, prm)
                self.assertEqual(float(p[300:].abs().max()), 0.0)
                self.assertLess(float((p[:300] - ref_steps(l[:300], T, k, tp)).abs().max()), 1e-6)

    def test_fused_law_ties(self):
        g = torch.Generator().manual_seed(2)
        for prm in ((1.0, 1.0, 2, 0, 0.95), (1.0, 1.0, 3, 20, 0.95), (1 / 0.7, 1 / 0.7, 2, 0, 0.9)):
            for _ in range(100):
                l = (torch.randn(300, generator = g) * 2).half().float()
                l[torch.randint(0, 300, (40,), generator = g)] = l[torch.randint(0, 300, (1,), generator = g)]
                self.assertLess(float((ss.law(l, ("fused", prm, 300)) - ref_fused(l, prm)).abs().max()), 1e-6)

    def test_exact_in_distribution(self):
        g = torch.Generator().manual_seed(3); rng = random.Random(4)
        V, N = 48, 40000
        for prm in (steps_prm(ComboSampler(temperature = 1.0, top_p = 0.95), V), ("fused", (1.0, 1.0, 3, 20, 0.95), V)):
            lt = torch.randn(V, generator = g) * 2.0; ld = lt + torch.randn(V, generator = g) * 1.2
            p = ss.law(lt, prm); q = ss.law(ld, prm)
            cnt = torch.zeros(V); keep = 0
            for _ in range(N):
                d = int(ss.draw(q, rng.random())); x = int(ss.verify(lt, q, d, prm, rng)); cnt[x] += 1; keep += (x == d)
            m = p > 0
            self.assertEqual(int(cnt[~m].sum()), 0)
            e = p[m] * N; chi = float(((cnt[m] - e) ** 2 / e).sum()); dof = int(m.sum()) - 1
            self.assertLess((chi - dof) / math.sqrt(2 * dof), 4.0)
            self.assertLess(abs(keep / N - float(torch.minimum(p, q).sum())), 0.01)

    def test_eligible(self):
        self.assertEqual(ss.eligible(FakeJob(ComboSampler(temperature = 1.0, top_p = 0.95)))[0], "steps")
        self.assertEqual(ss.eligible(FakeJob(ComboSampler(temperature = 1.0, top_k = 20, top_p = 0.95)))[0], "steps")
        self.assertIsNone(ss.eligible(FakeJob(ArgmaxSampler())))
        self.assertIsNone(ss.eligible(FakeJob(ComboSampler(temperature = 0.0))))
        self.assertIsNone(ss.eligible(FakeJob(ComboSampler(temperature = 1.0, rep_p = 1.1))))
        self.assertIsNone(ss.eligible(FakeJob(ComboSampler(temperature = 1.0), device_logit_mask = torch.zeros(1))))
        self.assertIsNone(ss.eligible(FakeJob(ComboSampler(temperature = 1.0), forced_ids = [1])))

if __name__ == "__main__":
    unittest.main()
