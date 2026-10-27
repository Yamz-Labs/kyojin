"""
DSpark speculative decoding on the port (CPU, fp32, tiny random model with the real structure: sliding-window ring
that wraps, compressors at ratio 2 and 1, indexers + two-level candidate selection, MoE, engram, 3-token... 5-token
draft block).

  * block verify == the same tokens decoded one at a time (logits, hidden, every cache buffer)
  * rollback restores the ring rows / compressor slots the rejected tail overwrote, without recomputing
  * feed() == the window seeding that discarded forward_spec calls leave behind
  * greedy with drafting == greedy without, token for token (real drafter, and an oracle drafter that makes every
    acceptance length and every rollback depth happen); seeded Gumbel sampling matches the plain loop too

16 index heads: with few heads many indexer scores are exactly 0 after the relu and top-k breaks those ties
differently for different tensor shapes (block vs one token), which is a property of the random toy, not of the port.
"""
import os, sys
import numpy as np
import pytest
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "ds41"))
import harness
from harness import FakeTok, make_state_dict, make_engram_tables, tiny_config
from test_dsv41_e2e import build_mine
from exllamav3.dsv41.engram import build_compressed_token_map, EngramLayout
from exllamav3.dsv41.spec import PortSpec
import dspark as ctl                               # tools/ds41/dspark.py, the controller
from dspark_refkit import OracleDrafter

import contextlib
from exllamav3.dsv41 import quant


@contextlib.contextmanager
def smooth_graph():
    """The graph without its fp8 / fp4 fake-quantisation steps (kv, attention output, indexer, compressed latents,
    fp8 linears' inputs). A [6, d] GEMM and six [1, d] GEMVs differ in the last bits; every fake-quant step is a
    rounding to a coarse grid, so a last-bit difference sometimes flips a grid cell and becomes a few-percent
    change that stays in the cache for good. That is a property of the numerics (block verify vs decode would show
    it on any engine), not of the cache logic; sampled-token parity under a seed needs it out of the way."""
    saved = (quant.act_quant_fp8_, quant.act_quant_fp8, quant.fp4_act_quant_)
    quant.act_quant_fp8_ = lambda x, block=32: x
    quant.act_quant_fp8 = lambda x, block=32: x
    quant.fp4_act_quant_ = lambda x, block=32, e4m3_scale=False: x
    try:
        yield
    finally:
        quant.act_quant_fp8_, quant.act_quant_fp8, quant.fp4_act_quant_ = saved


EOS = -1                                           # never sampled: every run produces its full budget


def build(seed=0, quantized=False, **over):
    cfg = tiny_config(index_n_heads=16, dspark_block_size=5, dspark_noise_token_id=511, **over)
    tok = FakeTok(cfg.vocab_size)
    tmap, nv = build_compressed_token_map(tok)
    cfg.engram_compressed_vocab_size = nv
    lay = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                             cfg.engram_n_heads, cfg.engram_head_dim)
    cfg.engram_num_embeddings = list(lay.num_embeddings)
    tables = make_engram_tables(cfg, lay.num_embeddings, seed + 1)
    sd = make_state_dict(cfg, seed, fp8=quantized, fp4=quantized)
    return cfg, build_mine(cfg, sd, tables, tmap)


@pytest.fixture(scope="module")
def tiny():
    return build(0)


def state_tensors(st):
    out = {}
    for l, t in st.win.items():
        out[f"win{l}"] = t
    for name in ("comp_kv", "comp_kv_state", "comp_score_state", "idx_k"):
        for l, t in getattr(st, name).items():
            out[f"{name}{l}"] = t
    if st.hasher is not None:
        out["hash"] = st.hasher.cache
    return {k: v.clone() for k, v in out.items()}


def prompt_ids(cfg, seed, n):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size - 1, (n,), generator=g).tolist()


# ------------------------------------------------------------------------------------------ block verify
@pytest.mark.parametrize("quantized,seed", [(False, 0), (True, 3)], ids=["plain", "fp8fp4"])
def test_block_verify_equals_token_by_token(quantized, seed):
    cfg, mine = build(seed, quantized)
    ids = torch.tensor([prompt_ids(cfg, seed, 43 + 40)])
    S = 43
    a, b = mine.new_state(1, 256), mine.new_state(1, 256)
    mine.forward(ids[:, :S], 0, a)
    mine.forward(ids[:, :S], 0, b)
    worst_l = worst_h = 0.0
    pos, n = S, 0
    while pos + 6 <= S + 40:
        L = 1 + pos % 6                                       # blocks of 1..6 tokens (a draft block is anchor + up to 5)
        lb, hb = mine.forward_block(ids[:, pos:pos + L], pos, b)
        assert lb.shape == (1, L, cfg.vocab_size) and hb.shape[:2] == (1, L)
        for i in range(L):
            l1, h1 = mine.forward(ids[:, pos + i:pos + i + 1], pos + i, a)
            worst_l = max(worst_l, (l1 - lb[:, i]).abs().max().item())
            worst_h = max(worst_h, (h1[:, 0] - hb[:, i]).abs().max().item())
            assert torch.equal(l1.argmax(-1), lb[:, i].argmax(-1))
        pos += L
        n += L
    assert n >= 30 and lb.abs().max() > 1.0
    assert worst_l < 1e-4 and worst_h < 1e-4, (worst_l, worst_h)
    sa, sb = state_tensors(a), state_tensors(b)
    for k in sa:
        assert (sa[k].float() - sb[k].float()).abs().max().item() < 1e-4, k
    print(f"block vs decode: logits {worst_l:.2e} hidden {worst_h:.2e}")


def test_block_rejects_prefill_and_oversize(tiny):
    cfg, mine = tiny
    st = mine.new_state(1, 256)
    ids = torch.tensor([prompt_ids(cfg, 1, 40)])
    mine.forward(ids[:, :20], 0, st)
    with pytest.raises(ValueError):
        mine.forward_block(ids[:, :3], 0, st)
    with pytest.raises(ValueError):
        st.begin_block(20, cfg.sliding_window + 1)
    with pytest.raises(ValueError):
        mine.forward(ids[:, 20:23], 20, st)                   # plain forward still refuses a multi-token decode


# ------------------------------------------------------------------------------------------ rollback
@pytest.mark.parametrize("start", [21, 22, 37])               # open / closed compressor group, ring wrapped or not
def test_rollback_restores_the_state(tiny, start):
    cfg, mine = tiny
    ids = torch.tensor([prompt_ids(cfg, 7, start + 8)])
    for keep in range(0, 6):
        st = mine.new_state(1, 256)
        mine.forward(ids[:, :start], 0, st)
        mine.forward_block(ids[:, start:start + 1], start, st)         # a verified block always follows decoded context
        s0 = state_tensors(st)
        # reference: the state after only the accepted tokens of the block
        ref = mine.new_state(1, 256)
        mine.forward(ids[:, :start], 0, ref)
        mine.forward_block(ids[:, start:start + 1], start, ref)
        if keep:
            mine.forward_block(ids[:, start + 1:start + 1 + keep], start + 1, ref)
        sref = state_tensors(ref)
        st.begin_block(start + 1, 5)
        mine.forward_block(ids[:, start + 1:start + 6], start + 1, st)
        changed = state_tensors(st)
        st.rollback(keep)
        got = state_tensors(st)
        if keep == 0:
            for k in s0:                                    # rollback to zero tokens: ring + compressor state exactly as before
                if k.startswith(("win", "comp_kv_state", "comp_score_state")):
                    assert torch.equal(s0[k], got[k]), (start, keep, k)
        W = cfg.sliding_window
        live = range(max(0, start + 1 + keep - W), start + 1 + keep)
        for k in sref:                                      # the live part equals the state of the accepted prefix
            if k.startswith("win"):
                slots = [p % W for p in live]
                a, b = got[k][:, slots], sref[k][:, slots]
            elif k.startswith(("comp_kv_state", "comp_score_state")):
                r = cfg.ratio(int(k[len("comp_kv_state"):])) if k.startswith("comp_kv_state") else cfg.ratio(int(k[len("comp_score_state"):]))
                m = (start + 1 + keep) % r
                a, b = got[k][:, :m], sref[k][:, :m]
            elif k.startswith(("comp_kv", "idx_k")):
                l = int(k[len("comp_kv"):]) if k.startswith("comp_kv") else int(k[len("idx_k"):])
                m = (start + 1 + keep) // cfg.ratio(l)
                a, b = got[k][:, :m], sref[k][:, :m]
            else:
                a, b = got[k][:, :start + 1 + keep], sref[k][:, :start + 1 + keep]
            assert a.shape == b.shape
            assert a.numel() == 0 or (a.float() - b.float()).abs().max().item() < 1e-4, (start, keep, k)
        # and the model goes on from there as if the rejected tail never happened
        nxt = ids[:, start + 1 + keep:start + 2 + keep]
        l1, _ = mine.forward(nxt, start + 1 + keep, st)
        l2, _ = mine.forward(nxt, start + 1 + keep, ref)
        assert (l1 - l2).abs().max().item() < 1e-4, (start, keep)


def test_rollback_without_block_or_out_of_range(tiny):
    cfg, mine = tiny
    st = mine.new_state(1, 256)
    with pytest.raises(RuntimeError):
        st.rollback(0)
    st.begin_block(5, 3)
    with pytest.raises(ValueError):
        st.rollback(4)


# ------------------------------------------------------------------------------------------ drafter feed
def test_feed_equals_discarded_forward_spec_calls(tiny):
    cfg, mine = tiny
    n, W = cfg.num_hidden_layers, cfg.sliding_window
    ids = torch.tensor([prompt_ids(cfg, 9, 60)])
    noise = torch.tensor([cfg.dspark_noise_token_id])
    S = 40                                                    # > W: the prefill seeding wraps the ring
    a, b, c = (mine.new_state(1, 256) for _ in range(3))
    _, hid = mine.forward(ids[:, :S], 0, a)
    rows = [hid[:, S - 1:S]]
    for i in range(S, S + 6):                                 # six more decoded positions
        _, h = mine.forward(ids[:, i:i + 1], i, a)
        rows.append(h)
    allh = torch.cat([hid] + rows[1:], dim=1)                 # rows of positions 0..S+5
    # the official way: start_pos 0 seeds rows 0..S-2, then one full call per later position (outputs discarded)
    ref = mine.new_state(1, 256)
    mine.mtp.forward_spec(ids[:, :1], allh[:, :S - 1], 0, ref)
    for i in range(S - 1, S + 6):
        mine.mtp.forward_spec(noise, allh[:, i:i + 1], i, ref)
    # feed(): the seed in one batch, the rest in one batch / row by row
    mine.mtp.feed(allh[:, :S - 1], 0, b)
    mine.mtp.feed(allh[:, S - 1:S + 6], S - 1, b)
    mine.mtp.feed(allh[:, :S - 1], 0, c)
    for i in range(S - 1, S + 6):
        mine.mtp.feed(allh[:, i:i + 1], i, c)
    for l in range(n, n + cfg.n_mtp_layers):
        assert (ref.win[l] - b.win[l]).abs().max().item() < 1e-5, l
        assert (ref.win[l] - c.win[l]).abs().max().item() < 1e-5, l
        assert ref.win[l].abs().max() > 0.1
    # and the drafts that come out are the same
    tok = ids[:, S + 6]
    r_ids, r_log, _ = mine.mtp.forward_spec(tok, allh[:, S + 5:S + 6], S + 5, ref)
    for st in (b, c):
        m_ids, m_log, _ = mine.mtp.forward_spec(tok, allh[:, S + 5:S + 6], S + 5, st)
        assert torch.equal(r_ids, m_ids) and (r_log - m_log).abs().max().item() < 1e-4


# ------------------------------------------------------------------------------------------ end to end
def plain_greedy(mine, prompt, n_new, sampler=None):
    sampler = sampler or ctl.GreedySampler()
    st = mine.new_state(1, 256)
    logits, _ = mine.forward(torch.tensor([prompt]), 0, st)
    out = [sampler.sample(logits[-1])]
    while len(out) < n_new:
        logits, _ = mine.forward(torch.tensor([[out[-1]]]), len(prompt) + len(out) - 1, st)
        out.append(sampler.sample(logits[-1]))
    return out


PROMPTS = [(11, 17), (12, 31), (13, 24)]


def test_greedy_with_drafting_equals_without(tiny):
    cfg, mine = tiny
    spec = PortSpec(mine, 256)
    total = acc = prop = 0
    for seed, n in PROMPTS:
        prompt = prompt_ids(cfg, seed, n)
        want = plain_greedy(mine, prompt, 64)
        got, dec = ctl.decode(spec.target, spec.drafter, prompt, 64, EOS)
        assert got == want, (seed, [i for i, (x, y) in enumerate(zip(got, want)) if x != y][:3])
        assert len(got) == 64 and dec.stats.steps > 0
        total += len(got); acc += dec.stats.accepted; prop += dec.stats.proposed
        assert spec.drafter.holes == 0
    print(f"real drafter (random heads): {total} tokens, accepted {acc}/{prop} drafts")


@pytest.mark.parametrize("p_err", [0.0, 0.25, 0.6, 1.0])
def test_greedy_with_oracle_drafts_equals_without(tiny, p_err):
    """The oracle proposes the true continuation with a fraction of random tokens: partial acceptance, full
    acceptance + bonus, immediate rejection; every rollback depth runs through the real cache."""
    cfg, mine = tiny
    spec = PortSpec(mine, 256)
    hist = {}
    acc = prop = steps = 0
    for seed, n in PROMPTS:
        prompt = prompt_ids(cfg, seed, n)
        want = plain_greedy(mine, prompt, 64)
        drafter = OracleDrafter(want, len(prompt), cfg.vocab_size - 1, np.random.default_rng(seed), p_err, 5)
        got, dec = ctl.decode(spec.target, drafter, prompt, 64, EOS)
        assert got == want, (p_err, seed)
        for k, v in dec.stats.accepted_hist.items():
            hist[k] = hist.get(k, 0) + v
        acc += dec.stats.accepted; prop += dec.stats.proposed; steps += dec.stats.steps
    if p_err == 0.0:
        assert acc == prop and set(hist) <= {5} | {0, 1, 2, 3, 4, 5}
        assert steps < 3 * 64 / 4                         # full acceptance: about 6 tokens per verify
    if p_err == 1.0:
        assert acc <= prop * 0.1
    if p_err in (0.25, 0.6):
        assert len([k for k, v in hist.items() if v]) >= 4, hist     # several accepted lengths, i.e. several rollback depths
    print(f"oracle p_err={p_err}: accepted {acc}/{prop} in {steps} rounds, hist {dict(sorted(hist.items()))}")


@pytest.mark.parametrize("p_err", [0.0, 0.4])
def test_seeded_sampling_matches_the_plain_loop(tiny, p_err):
    """Under sampling the controller draws the target token of each position in order with the official Gumbel
    rule and accepts a draft that equals the draw, so the tokens are the plain loop's tokens under the same seed
    whatever the drafter proposes. The drafter here follows the plain run's sampled path with errors."""
    cfg, mine = tiny
    spec = PortSpec(mine, 256)
    prompt = prompt_ids(cfg, 23, 20)
    with smooth_graph():
        _seeded(cfg, mine, spec, prompt, p_err)


def _seeded(cfg, mine, spec, prompt, p_err):
    want = plain_greedy(mine, prompt, 60, ctl.GumbelSampler(1.0, torch.Generator().manual_seed(23)))
    sampler = ctl.GumbelSampler(1.0, torch.Generator().manual_seed(23))
    dec = ctl.DSparkDecoder(spec.target, _FollowDrafter(want, len(prompt), p_err), eos_id=EOS, sampler=sampler)
    got = [t for chunk in dec.stream(prompt, 60) for t in chunk]
    assert got == want
    assert dec.stats.accepted > 0
    print(f"gumbel p_err={p_err}: accepted {dec.stats.accepted}/{dec.stats.proposed}")


class _FollowDrafter:
    """Drafts the sampled path `want` (known in advance from the plain run), each token replaced by noise with p_err."""

    block_size = 5

    def __init__(self, want, n_prompt, p_err):
        self.want, self.n_prompt, self.p_err = want, n_prompt, p_err
        self.rng = np.random.default_rng(3)

    def draft(self, token, hidden, start_pos):
        c = start_pos + 1 - self.n_prompt
        assert self.want[c] == token
        toks = []
        for i in range(5):
            t = self.want[c + 1 + i] if c + 1 + i < len(self.want) else 0
            if self.rng.random() < self.p_err:
                t = int(self.rng.integers(0, 500))
            toks.append(t)
        return ctl.DraftOut(toks)


def test_num_draft_caps_the_block(tiny):
    cfg, mine = tiny
    spec = PortSpec(mine, 256)
    prompt = prompt_ids(cfg, 11, 17)
    want = plain_greedy(mine, prompt, 40)
    for nd in (1, 2, 5):
        got, dec = ctl.decode(spec.target, spec.drafter, prompt, 40, EOS, num_draft=nd)
        assert got == want
        assert max(dec.stats.draft_len_hist) <= nd


def test_target_and_drafter_satisfy_the_protocols(tiny):
    cfg, mine = tiny
    spec = PortSpec(mine, 256)
    assert isinstance(spec.target, ctl.Target) and isinstance(spec.drafter, ctl.Drafter)
    assert spec.drafter.block_size == cfg.dspark_block_size == 5
