"""
DSpark speculative decoding on the reduced REAL-dimension model (hidden 5120, moe 2304, 64 heads x 512, q_lora 1280,
o 8 x 1024, 4 experts top-2, 3 layers + 1 DSpark stage, vocab 32000, random bf16 weights; the checkpoint the GPU smoke of
the calibration used, written by ds41port/engine/scratch/gpu_smoke.py::real_stage). CPU fp32.

Opt-in: DSV41_SPEC_REAL=1 (loads ~6 GB). Directory: DSV41_REAL_DIMS (default: the ds41port scratch copy).
  greedy with drafting == greedy without, token for token, 3 prompts x 64 tokens, for the model's own drafter and for an
  oracle drafter that forces every acceptance length and rollback depth through the real-size caches. Strict on the graph
  without fake-quant rounding; on the full graph a divergence must sit at a near-tie within the measured numeric noise.
"""
import contextlib, os, sys, time
import numpy as np
import pytest
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "ds41"))
from exllamav3.dsv41.config import DSV41Config
from exllamav3.dsv41.safetensors_io import SafetensorsWeightSource
from exllamav3.dsv41.experts import SourceExpertProvider
from exllamav3.dsv41.model import DSV41Model
from exllamav3.dsv41.spec import PortSpec
import dspark as ctl
from dspark_refkit import OracleDrafter
from test_dsv41_spec import smooth_graph

torch.set_num_threads(int(os.environ.get("DSV41_THREADS", "8")))
DIR = os.environ.get("DSV41_REAL_DIMS", "./ds41_work/ds41port/scratch/real_dims")
pytestmark = pytest.mark.skipif(os.environ.get("DSV41_SPEC_REAL") != "1" or not os.path.isfile(os.path.join(DIR, "config.json")),
                                reason="set DSV41_SPEC_REAL=1 (needs the real-dims checkpoint)")
EOS = -1
N_NEW = 64


@pytest.fixture(scope="module")
def real():
    cfg = DSV41Config.from_file(os.path.join(DIR, "config.json"))
    src = SafetensorsWeightSource(DIR, cfg.num_hidden_layers)
    counts = {l: cfg.moe_counts(l)[0] for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers)}
    t0 = time.time()
    model = DSV41Model(cfg, src, SourceExpertProvider(src, counts, cache=True), dtype=torch.float32)
    print(f"\nloaded {time.time() - t0:.0f}s: hidden {cfg.hidden_size}, {cfg.num_hidden_layers} layers + {cfg.n_mtp_layers} stage(s), "
          f"block {cfg.dspark_block_size}, vocab {cfg.vocab_size}")
    return cfg, model


def plain(model, prompt, n, keep_logits=False):
    st = model.new_state(1, 256)
    lg, _ = model.forward(torch.tensor([prompt]), 0, st)
    out, rows = [int(lg[-1].argmax())], [lg[-1].clone()]
    while len(out) < n:
        lg, _ = model.forward(torch.tensor([[out[-1]]]), len(prompt) + len(out) - 1, st)
        out.append(int(lg[-1].argmax()))
        rows.append(lg[-1].clone())
    return (out, rows) if keep_logits else out


def prompts(cfg):
    g = torch.Generator().manual_seed(2026)
    return [torch.randint(0, cfg.vocab_size - 1, (n,), generator=g).tolist() for n in (18, 27, 33)]


def verify_noise(spec, prompt, want, rows):
    """max |logits of block verify - logits of decode steps| along the plain path (blocks of 6)."""
    spec.target.prefill(prompt)
    worst, i = 0.0, 0
    while i + 6 < len(want):
        o = spec.target.verify(want[i:i + 6], len(prompt) + i)
        worst = max(worst, max((o.logits[j] - rows[i + j + 1]).abs().max().item() for j in range(6)))
        i += 6
    return worst


@pytest.mark.parametrize("graph", ["smooth", "full"])
def test_real_dims_greedy_equality_real_drafter(real, graph):
    """smooth: the graph without its fp8 / fp4 fake-quant rounding steps (see test_dsv41_spec.smooth_graph): block verify and
    decode then differ by ulps only and the output is identical token for token. full: the real graph; block verify
    and decode differ by ~1e-3..1e-2 in the logits (one [6, 5120] GEMM vs six GEMVs, amplified by the fake-quant
    steps), so a divergence is legitimate only at a near-tie of the top two logits, which is asserted."""
    cfg, model = real
    spec = PortSpec(model, 256)
    ctx = smooth_graph() if graph == "smooth" else contextlib.nullcontext()
    tot = same = acc = prop = steps = 0
    t_plain = t_spec = 0.0
    with ctx:
        for p in prompts(cfg):
            t0 = time.time(); want, rows = plain(model, p, N_NEW, True); t_plain += time.time() - t0
            t0 = time.time(); got, dec = ctl.decode(spec.target, spec.drafter, p, N_NEW, EOS); t_spec += time.time() - t0
            assert len(got) == N_NEW
            k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
            tot += N_NEW; acc += dec.stats.accepted; prop += dec.stats.proposed; steps += dec.stats.steps
            assert spec.drafter.holes == 0
            if k is None:
                same += N_NEW
                continue
            assert graph == "full", f"smooth graph diverged at token {k}"
            top = rows[k].topk(2).values
            gap, noise = (top[0] - top[1]).item(), verify_noise(spec, p, want, rows)
            print(f"  prompt diverges at token {k}: top-2 gap {gap:.4f}, block-vs-decode logit noise {noise:.4f}")
            assert gap < 2 * noise
            same += k
    print(f"real-dims [{graph}], model's own drafter (random weights): {same}/{tot} tokens identical, accepted {acc}/{prop} drafts "
          f"in {steps} rounds ({acc / steps:.3f} per round); plain {t_plain:.0f}s, spec {t_spec:.0f}s (cpu fp32)")


def test_real_dims_greedy_equality_oracle_drafter(real):
    """Oracle drafts (30 % wrong) force every acceptance length and rollback depth through the real-size caches."""
    cfg, model = real
    spec = PortSpec(model, 256)
    hist = {}
    acc = prop = steps = 0
    with smooth_graph():
        for i, p in enumerate(prompts(cfg)):
            want = plain(model, p, N_NEW)
            drafter = OracleDrafter(want, len(p), cfg.vocab_size - 1, np.random.default_rng(i), 0.3, cfg.dspark_block_size)
            got, dec = ctl.decode(spec.target, drafter, p, N_NEW, EOS)
            assert got == want
            for k, v in dec.stats.accepted_hist.items():
                hist[k] = hist.get(k, 0) + v
            acc += dec.stats.accepted; prop += dec.stats.proposed; steps += dec.stats.steps
    assert len(hist) >= 3
    print(f"real-dims [smooth], oracle drafter p_err 0.3: identical, accepted {acc}/{prop} in {steps} rounds, hist {dict(sorted(hist.items()))}")
