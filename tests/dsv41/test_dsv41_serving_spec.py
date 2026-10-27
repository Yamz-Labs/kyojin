"""
Speculative decoding on the SERVING model (DSV41Serving), CPU, tiny random model with the real structure.

  * greedy with the DSpark drafter == plain greedy, token for token: 4 prompts x draft depth 1..5, plain / confidence
    stop / auto draft length (the real drafter, random heads: it rarely hits, so almost every round rolls back)
  * the same with an oracle drafter (true continuation with p_err random tokens) for every accept length
    1..5 and the confidence stop; accept rate per case is printed (pytest -s)
  * seeded temperature + top_p sampling (the engine's TopPSampler) == the plain loop under the same seed
  * engine level: PortEngine._run with num_draft / draft_auto / draft_conf_threshold == PortEngine._run without
  * state after a speculative run == state after a plain run for the committed positions (rings, compressor, indexer, hash)
"""
import os, sys
import numpy as np
import pytest
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "ds41"))
import test_dsv41_serving as S
import test_dsv41_spec as T
import dspark as ctl
from dspark_refkit import OracleDrafter
from ds41_model import PortEngine, TopPSampler
from exllamav3.dsv41.spec import PortSpec

EOS = T.EOS
N_NEW = 40
PROMPTS = [(11, 17), (12, 31), (13, 24), (14, 9)]


@pytest.fixture(scope="module")
def world():
    cfg, sd, tables, tmap = S.make_world(2, sliding_window=16)      # a verify block of 11 rows must fit the ring
    return cfg, S.build_serving(cfg, sd, tables, tmap)


def plain(mine, prompt, n_new, sampler=None):
    return T.plain_greedy(mine, prompt, n_new, sampler)


def make_tuner(depth):
    return ctl.DraftTuner(1, depth)


@pytest.mark.parametrize("mode", ["plain", "conf", "auto"])
def test_greedy_spec_equals_plain_with_the_real_drafter(world, mode):
    """Exact, token for token. Run on the smooth graph: a [k+1, d] block GEMM and k+1 GEMVs differ in the last bits, which
    sometimes flips an fp8/fp4 fake-quant grid cell (see test_dsv41_spec.smooth_graph); that is numerics, not cache logic."""
    with T.smooth_graph():
        _real_drafter_sweep(world, mode, exact=True)


def test_greedy_spec_on_the_real_graph_diverges_only_by_grid_flips(world):
    """Same sweep with the quantised graph: report the first divergence (if any); the common prefix must stay long."""
    _real_drafter_sweep(world, "plain", exact=False)


def _real_drafter_sweep(world, mode, exact):
    cfg, mine = world
    spec = PortSpec(mine, 256, max_depth=10)
    for depth in (1, 2, 3, 5, 7, 10):
        acc = prop = steps = 0
        for seed, n in PROMPTS:
            prompt = T.prompt_ids(cfg, seed, n)
            want = plain(mine, prompt, N_NEW)
            kw = {}
            if mode == "conf":
                kw["conf_threshold"] = 0.3
            if mode == "auto":
                kw["tuner"] = make_tuner(depth)
            got, dec = ctl.decode(spec.target, spec.drafter, prompt, N_NEW, EOS, num_draft=depth, **kw)
            if exact:
                assert got == want, (mode, depth, seed)
            else:
                same = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), len(want))
                if same < len(want):
                    print(f"  grid flip: depth {depth} seed {seed} first differs at token {same}")
                assert same >= 16
            assert max(dec.stats.draft_len_hist or {0: 1}) <= depth
            acc += dec.stats.accepted; prop += dec.stats.proposed; steps += dec.stats.steps
            assert spec.drafter.holes == 0
        print(f"real drafter {mode} depth {depth}: accepted {acc}/{prop} ({acc / max(prop, 1):.2f}) in {steps} rounds")


@pytest.mark.parametrize("p_err", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("mode", ["plain", "conf"])
def test_greedy_spec_equals_plain_with_an_oracle_drafter(world, p_err, mode):
    with T.smooth_graph():
        _oracle_sweep(world, p_err, mode)


def _oracle_sweep(world, p_err, mode):
    cfg, mine = world
    spec = PortSpec(mine, 256, max_depth=10)
    for depth in (1, 3, 5, 10):
        acc = prop = steps = tokens = 0
        hist = {}
        for seed, n in PROMPTS:
            prompt = T.prompt_ids(cfg, seed, n)
            want = plain(mine, prompt, N_NEW)
            drafter = OracleDrafter(want, len(prompt), cfg.vocab_size - 1, np.random.default_rng(seed), p_err, 10,
                                    conf_sigma=1.0 if mode == "conf" else None)
            got, dec = ctl.decode(spec.target, drafter, prompt, N_NEW, EOS, num_draft=depth,
                                  conf_threshold=0.3 if mode == "conf" else None)
            assert got == want, (p_err, mode, depth, seed)
            acc += dec.stats.accepted; prop += dec.stats.proposed; steps += dec.stats.steps; tokens += len(got)
            for k, v in dec.stats.accepted_hist.items():
                hist[k] = hist.get(k, 0) + v
        if p_err == 0.0 and mode == "plain":
            assert acc == prop and prop > 0
        print(f"oracle p_err={p_err} {mode} depth {depth}: accepted {acc}/{prop} ({acc / max(prop, 1):.2f}), "
              f"{tokens / max(steps, 1):.2f} tokens/round, hist {dict(sorted(hist.items()))}")


def test_chained_drafter_leaves_no_trace_in_the_rings(world):
    """A depth-10 draft (two passes) must leave the drafter window rings as a depth-5 draft does."""
    cfg, mine = world
    prompt = T.prompt_ids(cfg, 12, 31)
    rings = []
    for depth in (5, 10):
        spec = PortSpec(mine, 256, max_depth=depth)
        out = spec.target.prefill(prompt)
        tok = int(out.logits[-1].argmax())
        d = spec.drafter.draft(tok, out.hidden, len(prompt) - 1) if False else spec.drafter.draft(
            tok, out.hidden[-1:], len(prompt))
        assert len(d.tokens) == depth and len(d.conf) == depth
        n0 = cfg.num_hidden_layers
        rings.append([spec.st.win[n0 + j].clone() for j in range(mine.mtp.n_stages)])
    for a, b in zip(*rings):
        assert torch.equal(a, b)


@pytest.mark.parametrize("p_err", [0.0, 0.4])
def test_seeded_top_p_sampling_equals_the_plain_loop(world, p_err):
    cfg, mine = world
    spec = PortSpec(mine, 256)
    prompt = T.prompt_ids(cfg, 23, 20)

    def sampler():
        return TopPSampler(0.8, 0.9, torch.Generator().manual_seed(23))

    with T.smooth_graph():
        want = plain(mine, prompt, 50, sampler())
        dec = ctl.DSparkDecoder(spec.target, T._FollowDrafter(want, len(prompt), p_err), eos_id=EOS, sampler=sampler())
        got = [t for chunk in dec.stream(prompt, 50) for t in chunk]
        assert got == want
        assert dec.stats.accepted > 0
        print(f"top_p sampling p_err={p_err}: accepted {dec.stats.accepted}/{dec.stats.proposed}")


def _state(st):
    return T.state_tensors(st)


def test_committed_state_after_speculation_equals_plain(world):
    """After a speculative run (with real rollbacks) the committed positions hold what a plain run holds: ring windows,
    compressor pools, indexer keys and the Engram history of the trunk layers equal the plain run's."""
    with T.smooth_graph():                       # block GEMM vs GEMV last-bit flips of fake-quant grids are not cache logic
        _committed(world)


def _committed(world):
    cfg, mine = world
    spec = PortSpec(mine, 256)
    prompt = T.prompt_ids(cfg, 12, 31)
    want = plain(mine, prompt, 30)
    drafter = OracleDrafter(want, len(prompt), cfg.vocab_size - 1, np.random.default_rng(1), 0.4, 5)
    dec = ctl.DSparkDecoder(spec.target, drafter, eos_id=EOS)
    got = [t for chunk in dec.stream(prompt, 30) for t in chunk]
    assert got == want and dec.stats.rejected > 0
    # plain run to the same length: prompt + 29 tokens in the cache (the last emitted token is never fed)
    st = mine.new_state(1, 256)
    mine.forward(torch.tensor([prompt]), 0, st)
    for i, t in enumerate(want[:-1]):
        mine.forward(torch.tensor([[t]]), len(prompt) + i, st)
    n = len(prompt) + len(want) - 1
    assert spec.length == n
    a, b = _state(spec.st), _state(st)
    assert a.keys() == b.keys()
    bad = []
    for k in a:
        if k.startswith("win") and int(k[3:]) >= cfg.num_hidden_layers:
            continue                                  # drafter stage rings: a plain run never feeds them
        d = (a[k].float() - b[k].float()).abs().max().item()
        if d > 2e-3 * max(1.0, b[k].float().abs().max().item()):
            bad.append((k, d))
    assert not bad, bad[:6]


def test_engine_run_spec_equals_plain(world):
    cfg, mine = world
    ids = T.prompt_ids(cfg, 12, 31)

    def run(**kw):
        eng = PortEngine(mine, None, ctx=200, eos_id=EOS, **kw)
        out = []
        st = eng._run(ids, 36, TopPSampler(0.0), EOS, None, out.extend)
        return out, st

    want, _ = run()
    for kw in ({"num_draft": 3}, {"num_draft": 5}, {"num_draft": 10}, {"num_draft": 10, "draft_auto": True}, {"num_draft": 4, "draft_auto": True},
               {"num_draft": 5, "draft_conf_threshold": 0.4}, {"num_draft": 4, "draft_auto": True, "draft_conf_threshold": 0.5}):
        got, st = run(**kw)
        assert got == want, kw
        assert st["accepted_draft_tokens"] + st["rejected_draft_tokens"] >= 0 and st["new_tokens"] == 36
        print(f"engine {kw}: accepted {st['accepted_draft_tokens']} rejected {st['rejected_draft_tokens']} "
              f"steps {st['draft_steps']} tuner_k {st.get('tuner_k', 0)}")


@pytest.mark.parametrize("n_ids,fp16", [(128, False), (256, True), (0, True)])
def test_greedy_spec_with_a_cheaper_draft_head_equals_plain(world, n_ids, fp16):
    """The draft head sees only the first n_ids token ids; verify still uses the full head, so the text is unchanged
    (only the acceptance rate can move). The restricted rows are the exact rows of the full head."""
    cfg, mine = world
    spec = PortSpec(mine, 256, max_depth=10)
    heads = mine.mtp
    heads.set_draft_head(n_ids, markov_fp16=fp16)
    try:
        calls = []
        if n_ids:
            x = torch.randn(3, cfg.hidden_size, dtype=mine.dtype)
            full = mine.head_logits(x)
            assert torch.allclose(heads.rhead(x), full[:, heads.rhead.ids], atol=1e-5)
            assert heads.rhead.ids.numel() == n_ids
            orig = heads.rhead.__class__.__call__
            heads.rhead.__class__.__call__ = lambda self, x: (calls.append(1), orig(self, x))[1]
        else:
            calls.append(1)
        with T.smooth_graph():
            for depth in (1, 3, 5):
                for seed, n in PROMPTS:
                    prompt = T.prompt_ids(cfg, seed, n)
                    want = plain(mine, prompt, N_NEW)
                    got, dec = ctl.decode(spec.target, spec.drafter, prompt, N_NEW, EOS, num_draft=depth)
                    assert got == want, (n_ids, depth, seed)
        assert calls, 'the draft pass did not use the restricted head'
    finally:
        if n_ids:
            heads.rhead.__class__.__call__ = orig
        heads.set_draft_head(0)
