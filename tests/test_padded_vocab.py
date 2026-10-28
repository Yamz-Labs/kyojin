"""
Padded vocabulary (GitHub issue 116): the model head is wider than the tokenizer's piece table (GLM-5.3: 154880 logit columns,
154856 pieces). No path may emit an id the tokenizer has no piece for, and an id that slips through must never kill the server.
CPU tests, no model; the sampler tests on the GPU run only where one is present.   pytest tests/test_padded_vocab.py
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hardware import require
require("ext")
from exllamav3.generator import spec_sample
from exllamav3.generator.generator import Generator
from exllamav3.generator.job import Job
from exllamav3.generator.sampler import custom, ArgmaxSampler, ComboSampler, CategoricalSampler
from exllamav3.util.tensor import SeqTensor

V, VS = 154880, 154856        # head columns, pieces


class StubTokenizer:
    actual_vocab_size = VS
    extended_id_to_piece = {}

    def get_id_to_piece_list(self, include_special_tokens = False):
        return [f"t{i}" for i in range(VS)]


class StubRng:
    def __init__(self, *u):
        self.u = list(u)

    def random(self):
        return self.u.pop(0)


def padded_law(seed = 0, spiky = False):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(V, generator = g) * (6.0 if spiky else 2.0)
    prm = ("fused", (1.0, 1.0, 0, 0, 1.0), VS)
    return logits, prm, spec_sample.law(logits, prm)


class TestSpecSamplePadding(unittest.TestCase):

    def test_law_is_zero_on_padding(self):
        _, _, p = padded_law()
        self.assertEqual(p.numel(), V)
        self.assertEqual(float(p[VS:].sum()), 0.0)
        self.assertAlmostEqual(float(p.sum()), 1.0, places = 5)

    def test_draw_never_returns_padding(self):
        _, _, p = padded_law()
        # u so close to 1 that total * u rounds up to total in fp32: the search used to run off the end of the row and the
        # clamp landed on the last column, a padded id
        for u in (0.0, 0.5, 1.0 - 2.0 ** -24, 1.0 - 1e-9):
            t = int(spec_sample.draw(p, u))
            self.assertLess(t, VS, f"u={u}")
            self.assertGreater(float(p[t]), 0.0, f"u={u}: drew a zero-mass index")

    def test_draw_without_mass_stays_in_vocabulary(self):
        # an all-zero law, and a NaN law (a broken row), used to end on the last column too
        for name, p in (("zero", torch.zeros(V)), ("nan", torch.full((V,), float("nan")))):
            for u in (0.0, 0.5, 0.999999):
                self.assertLess(int(spec_sample.draw(p, u)), VS, f"{name} u={u}")

    def test_draw_unchanged_on_normal_laws(self):
        _, _, p = padded_law(3, spiky = True)
        cs = torch.cumsum(p, 0)
        for u in (0.0, 0.1, 0.37, 0.5, 0.9, 0.99999):
            ref = int(torch.searchsorted(cs, cs[-1:] * u, right = True).clamp(max = V - 1))   # the draw as it was
            self.assertEqual(int(spec_sample.draw(p, u)), ref, f"u={u}")

    def test_verify_with_no_residual(self):
        # p == q leaves no residual mass; when the keep test fails by rounding (u1 ~ 1) the resample used to be the last column
        logits, prm, p = padded_law()
        d = int(torch.argmax(p))
        for u1 in (1.0 - 1e-12, 0.3):
            out = spec_sample.verify(logits, p.clone(), d, prm, StubRng(u1, 0.5))
            t = int(out)
            self.assertLess(t, VS)
            self.assertGreater(float(p[t]), 0.0)

    def test_verify_keeps_and_resamples_as_before(self):
        logits, prm, p = padded_law(1, spiky = True)
        q = padded_law(2, spiky = True)[2]
        d = int(torch.multinomial(q, 1))
        for u1, u2 in ((0.01, 0.2), (0.99, 0.2), (0.99, 0.8)):
            out = int(spec_sample.verify(logits, q, d, prm, StubRng(u1, u2)))
            keep = u1 * float(q[d]) < float(p[d])
            if keep:
                self.assertEqual(out, d)
            else:
                res = (p - q).clamp(min = 0.0)
                cs = torch.cumsum(res, 0)
                self.assertEqual(out, int(torch.searchsorted(cs, cs[-1:] * u2, right = True).clamp(max = V - 1)))


class TestSamplerMask(unittest.TestCase):

    def run_cpu_argmax(self, pos):
        saved = custom.fused_sampler_enable
        custom.fused_sampler_enable = False       # the fused kernel is CUDA only
        try:
            s = ArgmaxSampler()
        finally:
            custom.fused_sampler_enable = saved
        logits = torch.randn(1, 1, V)
        logits[..., pos] = 1e4
        return int(s.forward(logits, tokenizer = StubTokenizer()))

    def test_argmax_masks_padding_cpu(self):
        for pos in (VS, VS + 7, V - 1):
            self.assertLess(self.run_cpu_argmax(pos), VS)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_every_sampler_masks_padding_gpu(self):
        dev = "cuda:0"
        past = torch.randint(0, 1000, (1, 64), dtype = torch.long, device = dev)
        makers = [
            lambda: ArgmaxSampler(),
            lambda: ComboSampler(temperature = 0.0),
            lambda: ComboSampler(temperature = 0.0, rep_p = 1.1),
            lambda: ComboSampler(temperature = 0.7, top_p = 0.95),
            lambda: ComboSampler(temperature = 1.0, min_p = 0.05, top_k = 40),
            lambda: ComboSampler(temperature = 0.7, top_p = 0.9, rep_p = 1.1, pres_p = 0.1),
            lambda: ComboSampler(temperature = 0.7, top_p = 0.9, dry_multiplier = 0.8),
            lambda: ComboSampler(temperature = 0.7, adaptive_target = 0.5),
            lambda: CategoricalSampler(0.8),
        ]
        for mk in makers:
            for pos in (VS, VS + 13, V - 1):
                for dt in (torch.half, torch.float):
                    s = mk()
                    logits = torch.randn(1, 1, V, device = dev).to(dt)
                    logits[..., pos] = 60.0 if dt == torch.half else 1e4
                    try:
                        t = int(s.forward(logits, past if s.reqs_past_ids else None, 1234, StubTokenizer()))
                    except RuntimeError:       # a stack the kernels do not take for this dtype
                        continue
                    self.assertLess(t, VS, f"{type(s).__name__} {s.steps} pos={pos} {dt}")


class TestDraftBound(unittest.TestCase):

    def test_bound_draft_ids(self):
        g = type("G", (), {"vocab_limit": VS})()
        ids = torch.tensor([[5, VS - 1, VS, V - 1]])
        out = Generator._bound_draft_ids(g, ids)
        self.assertEqual(out.tolist(), [[5, VS - 1, 0, 0]])

    def test_no_limit_is_identity(self):
        g = type("G", (), {"vocab_limit": None})()
        ids = torch.tensor([[5, VS, V - 1]])
        self.assertIs(Generator._bound_draft_ids(g, ids), ids)


class FakeGenerator:
    num_draft_tokens = 0
    tokenizer = StubTokenizer()

    def __init__(self):
        self.unknown = []

    note_unknown_token = Generator.note_unknown_token

    _unknown_token_seen = False

    class model:
        class config:
            vocab_size = V


def bare_job(gen):
    job = Job(input_ids = torch.tensor([[1, 2, 3]]), max_new_tokens = 50, sampler = ArgmaxSampler(), stop_conditions = [])
    job.generator = gen
    job.sequences[0].sequence_ids = SeqTensor.from_tensor(torch.tensor([[1, 2, 3]]), 1)
    job.sequences[0].kv_position = 3
    job.new_tokens = 0
    job.max_rq_tokens = 10 ** 6
    job.held_tokens = SeqTensor((1, 0), dtype = torch.long, seq_dim = -1)
    return job


class TestReceiveSampleGuard(unittest.TestCase):

    def test_unknown_ids_decode_to_empty_text(self):
        gen = FakeGenerator()
        job = bare_job(gen)
        results = []
        for t in (7, VS, V - 1, 10 ** 9, -1):
            eos, tok, rq = job.receive_sample(None, torch.tensor([[t]]), None, None, None, results)
            self.assertFalse(eos)
        self.assertEqual("".join(r.get("text", "") for r in results), "t7")      # only the known id contributed text
        self.assertEqual(job.new_tokens, 5)

    def test_first_unknown_id_is_logged_once(self):
        gen = FakeGenerator()
        job = bare_job(gen)
        with self.assertLogs("exllamav3.generator.generator", level = "ERROR") as cm:
            for t in (VS, VS + 1, 10 ** 9):
                job.receive_sample(None, torch.tensor([[t]]), None, None, None, [])
        self.assertEqual(len(cm.output), 1)
        self.assertIn(str(VS), cm.output[0])


if __name__ == "__main__":
    unittest.main()
