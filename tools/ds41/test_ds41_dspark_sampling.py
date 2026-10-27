"""DSpark: truncated sampling (top_p / top_k / min_p), confidence early stop and draft-length auto-tuning. CPU only.

All three must leave the output exactly as the plain sampler gives it: seeded equality with the plain loop over many seeds,
and chi-square tests of the end-to-end output law on a toy bigram model. The official-model checks are at the bottom.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

torch = pytest.importorskip("torch")
import dspark                                                   # noqa: E402
from test_ds41_dspark import EOS_TOY, V, ToyDrafter, ToyTarget, toy_table   # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


CONFIGS = [dict(top_p=0.9), dict(top_p=0.5), dict(top_k=3), dict(min_p=0.2), dict(top_p=0.8, top_k=5, min_p=0.05),
           dict(top_p=0.0), dict(top_k=1), dict()]


def make_sampler(temp, cfg, generator=None):
    return dspark.GumbelSampler(temp, generator, **cfg)


# ---- the truncation rule ------------------------------------------------------------------------------------------


def probs_of(*p):
    return torch.log(torch.tensor(p, dtype=torch.float32))


def test_truncated_probs_rules():
    row = probs_of(0.5, 0.25, 0.15, 0.07, 0.03)
    tp = lambda **kw: dspark.truncated_probs(row, 1.0, **kw)                     # noqa: E731
    assert torch.allclose(tp(), torch.tensor([0.5, 0.25, 0.15, 0.07, 0.03]), atol=1e-6)
    got = tp(top_p=0.8)                                  # 0.5 + 0.25 = 0.75 < 0.8: the third token is needed
    assert got[3:].sum() == 0 and torch.allclose(got[:3], torch.tensor([0.5, 0.25, 0.15]) / 0.9, atol=1e-6)
    got = tp(top_p=0.75)                                 # 0.5 + 0.25 reaches it exactly
    assert (got > 0).sum() == 2
    assert (tp(top_p=0.0) > 0).sum() == 1 and int(tp(top_p=0.0).argmax()) == 0       # the top token always stays
    assert (tp(top_k=2) > 0).sum() == 2 and (tp(top_k=99) > 0).sum() == 5
    assert (tp(min_p=0.35) > 0).sum() == 2               # >= 0.35 * 0.5 = 0.175
    assert (tp(top_k=4, top_p=0.85, min_p=0.1) > 0).sum() == 3
    for kw in (dict(top_p=0.6), dict(top_k=2), dict(min_p=0.5)):
        assert abs(float(tp(**kw).sum()) - 1) < 1e-6
    ties = dspark.truncated_probs(torch.zeros(6), 1.0, top_k=2)               # ties: lowest token ids stay
    assert [int(i) for i in (ties > 0).nonzero()] == [0, 1]
    assert torch.equal(dspark.truncated_probs(row, 0.7, top_p=0.9), dspark.truncated_probs(row.clone(), 0.7, top_p=0.9))


def test_sampler_arguments_are_checked():
    for bad in (dict(top_p=1.5), dict(top_k=-1), dict(min_p=2.0), dict(top_p=-0.1)):
        with pytest.raises(ValueError):
            dspark.GumbelSampler(1.0, **bad)


def test_default_sampler_is_bitwise_the_old_rule():
    g = torch.Generator().manual_seed(3)
    row = torch.randn(40, generator=g)
    for seed in range(20):
        torch.manual_seed(seed)
        a = dspark.GumbelSampler(0.9).sample(row)
        torch.manual_seed(seed)                          # the official `sample`
        probs = torch.softmax(row.reshape(1, -1) / 0.9, dim=-1, dtype=torch.float32)
        assert a == int(probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1))


# ---- seeded equality with the plain loop --------------------------------------------------------------------------


def plain_loop(table, prompt, max_tokens, eos, sampler):
    out, prev = [], prompt[-1]
    while True:
        tok = sampler.sample(table[prev])
        out.append(tok)
        if tok == eos or len(out) >= max_tokens:
            return out[:out.index(eos)] if eos in out else out
        prev = tok


class ConfToyDrafter(ToyDrafter):
    """ToyDrafter plus a synthetic confidence: +2 when the draft is the table's argmax after the previous draft (what a
    greedy target would pick), -2 when not, plus noise."""

    def __init__(self, table, rng, p_err=0.3, sigma=1.0, **kw):
        super().__init__(table, rng, p_err=p_err, **kw)
        self.sigma = sigma

    def draft(self, token, hidden, start_pos):
        out = super().draft(token, hidden, start_pos)
        prev, conf = token, []
        for t in out.tokens:
            conf.append((2.0 if t == int(self.table[prev].argmax()) else -2.0) + float(self.rng.normal()) * self.sigma)
            prev = t
        out.conf = conf
        return out


@pytest.mark.parametrize("seed", range(300))
def test_truncated_spec_equals_plain_under_seed(seed):
    rng = np.random.default_rng(seed)
    table = torch.tensor(toy_table(seed % 11) * 0.5)
    prompt = [int(x) for x in rng.integers(0, V - 1, size=int(rng.integers(2, 8)))]
    max_tokens, temp = int(rng.integers(1, 50)), [0.7, 1.0, 1.3][seed % 3]
    cfg = CONFIGS[seed % len(CONFIGS)]
    torch.manual_seed(seed)
    want = plain_loop(table, prompt, max_tokens, EOS_TOY, make_sampler(temp, cfg))
    kind = seed % 4                                       # fixed length, confidence stop, tuner, both
    kw = {}
    if kind in (1, 3):
        kw["conf_threshold"] = [0.3, 0.5, 0.8][seed % 3]
    if kind in (2, 3):
        kw["tuner"] = dspark.DraftTuner(cost_per_position=[0.05, 0.3][seed % 2])
    torch.manual_seed(seed)
    drafter = ConfToyDrafter(table.numpy(), rng, p_err=[0.0, 0.4, 0.8][seed % 3])
    got, dec = dspark.decode(ToyTarget(table), drafter, prompt, max_tokens, EOS_TOY, sampler=make_sampler(temp, cfg),
                             num_draft=[None, 2, 4][seed % 3], **kw)
    assert got == want
    if "tuner" in kw:
        assert all(1 <= k <= 4 for k in dec.stats.draft_len_hist)


def test_truncation_matters_in_the_seeded_tests():
    """Guard: the configs really change the output, so the equality above is not trivially true."""
    table = torch.tensor(toy_table(2) * 0.5)
    outs = set()
    for cfg in CONFIGS:
        torch.manual_seed(5)
        outs.add(tuple(plain_loop(table, [1, 2], 40, -1, make_sampler(1.0, cfg))))
    assert len(outs) >= 5


# ---- chi-square on the output law ---------------------------------------------------------------------------------


def chi2_check(counts: Counter, probs: dict, n: int, z: float = 3.72):
    """Pearson test, cells with expectation < 5 pooled; fails above the 1e-4 tail (Wilson-Hilferty)."""
    stat, cells, pooled_e, pooled_o = 0.0, 0, 0.0, 0
    assert set(counts) <= {k for k, p in probs.items() if p > 0}, "a token outside the truncated support was emitted"
    for key, p in probs.items():
        e = p * n
        if e < 5:
            pooled_e += e
            pooled_o += counts.get(key, 0)
            continue
        stat += (counts.get(key, 0) - e) ** 2 / e
        cells += 1
    if pooled_e >= 5:
        stat += (pooled_o - pooled_e) ** 2 / pooled_e
        cells += 1
    df = cells - 1
    assert df >= 3
    crit = df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3
    assert stat < crit, (stat, crit, df)
    return stat, crit


def exact_pairs(table, temp, cfg, prev):
    """Law of (t2, t3) for a completion that starts after `prev`: P(t1) P(t2|t1) P(t3|t2) summed over t1."""
    rows = torch.stack([dspark.truncated_probs(table[i], temp, **cfg) for i in range(V)]).double()
    p1 = rows[prev]
    joint = (p1[:, None, None] * rows[:, :, None] * rows[None, :, :]).sum(0)
    return {(a, b): float(joint[a, b]) for a in range(V) for b in range(V)}


class QToyDrafter:
    """Samples its drafts from its own law q_i = softmax(0.6 * table[prev]) (different from the target's); the
    confidence of slot i is the logit of max q_i, a function of the previous token only."""
    block_size = 5

    def __init__(self, table, gen):
        self.table, self.gen = table, gen

    def draft(self, token, hidden, start_pos):
        toks, qs, conf, prev = [], [], [], token
        for _ in range(self.block_size):
            q = torch.softmax(self.table[prev].float() * 0.6, dim=-1)
            t = int(torch.multinomial(q, 1, generator=self.gen))
            m = float(q.max())
            toks.append(t)
            qs.append(q)
            conf.append(math.log(m / (1 - m)))
            prev = t
        return dspark.DraftOut(toks, qs, conf)


N_LAW = 3000


@pytest.mark.parametrize("cfg", [dict(top_p=0.7), dict(top_k=4, min_p=0.1)])
def test_chi_square_plain_one_hot_and_q_drafts(cfg):
    table = torch.tensor(toy_table(4) * 0.8)
    temp, prompt = 1.0, [1, 2]
    exact = exact_pairs(table, temp, cfg, prompt[-1])
    runs = {"plain": Counter(), "onehot": Counter(), "q+conf+tuner": Counter()}
    for seed in range(N_LAW):
        torch.manual_seed(seed)
        p = plain_loop(table, prompt, 3, -1, make_sampler(temp, cfg))
        runs["plain"][tuple(p[1:])] += 1
        torch.manual_seed(seed + 10**6)
        rng = np.random.default_rng(seed)
        o, _ = dspark.decode(ToyTarget(table), ToyDrafter(table.numpy(), rng, p_err=0.5), prompt, 3, -1,
                             sampler=make_sampler(temp, cfg), tuner=dspark.DraftTuner())
        runs["onehot"][tuple(o[1:])] += 1
        torch.manual_seed(seed + 2 * 10**6)
        g = torch.Generator().manual_seed(seed)
        q, _ = dspark.decode(ToyTarget(table), QToyDrafter(table, g), prompt, 3, -1, sampler=make_sampler(temp, cfg),
                             conf_threshold=0.6, tuner=dspark.DraftTuner(cost_per_position=0.2))
        runs["q+conf+tuner"][tuple(q[1:])] += 1
    for name, counts in runs.items():
        chi2_check(counts, exact, N_LAW)


def test_chi_square_catches_a_wrong_law():
    """The test has power: sampling with top_p 0.9 against the law of top_p 0.5 fails."""
    table = torch.tensor(toy_table(4) * 0.8)
    wrong = Counter()
    for seed in range(4000):
        torch.manual_seed(seed)
        wrong[tuple(plain_loop(table, [1, 2], 3, -1, make_sampler(1.0, dict(top_p=0.9)))[1:])] += 1
    with pytest.raises(AssertionError):
        chi2_check(wrong, exact_pairs(table, 1.0, dict(top_p=0.5), 2), 4000)


# ---- early stop and the cost it saves -----------------------------------------------------------------------------


def test_conf_early_stop_is_exact_and_saves_verify_positions(capsys):
    fixed = dict(pos=0, emitted=0, wasted=0, calls=0)
    early = dict(pos=0, emitted=0, wasted=0, calls=0)
    for seed in range(120):
        rng = np.random.default_rng(seed)
        table = toy_table(seed % 9)
        prompt = [int(x) for x in rng.integers(0, V - 1, size=4)]
        want = None
        for acc, thr in ((fixed, None), (early, 0.5)):
            drafter = ConfToyDrafter(table, np.random.default_rng(seed), p_err=0.35, sigma=1.0)
            got, dec = dspark.decode(ToyTarget(table), drafter, prompt, 60, -1, num_draft=4, conf_threshold=thr)
            want = got if want is None else want
            assert got == want
            st = dec.stats
            acc["pos"] += st.verify_positions
            acc["emitted"] += st.emitted
            acc["wasted"] += st.rejected
            acc["calls"] += st.verify_calls
    # Decode is memory bound: a verify of k+1 positions costs about one step plus c per extra position. Cost model of
    # DraftTuner: calls + c * (positions - calls). The confidence here is synthetic (+-2 plus N(0, 1) noise, close to an
    # oracle), so the saving is an upper bound that assumes a calibrated head; the tiny official model has random weights.
    def cost(acc, c):
        return acc["calls"] + c * (acc["pos"] - acc["calls"])

    saving = {c: 1 - cost(early, c) / cost(fixed, c) for c in (0.05, 0.1, 0.3)}
    with capsys.disabled():
        print(f"\n[toy] fixed k=4 vs confidence stop (threshold 0.5), same {fixed['emitted']} tokens: verify calls "
              f"{fixed['calls']} -> {early['calls']}, positions {fixed['pos']} -> {early['pos']}, rejected drafts verified "
              f"{fixed['wasted']} -> {early['wasted']}; verify cost saving " +
              ", ".join(f"{v:.1%} at c={c}" for c, v in saving.items()))
    assert saving[0.1] > 0.05 and early["wasted"] < fixed["wasted"] and early["pos"] < fixed["pos"]
    assert early["calls"] < 1.1 * fixed["calls"]                  # the stop does not cost many extra rounds


def test_conf_stop_edge_cases():
    table = toy_table(1)
    rng = np.random.default_rng(0)
    # a threshold nobody reaches: no draft is ever verified, the loop is the plain loop
    class Low(ConfToyDrafter):
        def draft(self, *a):
            out = super().draft(*a)
            out.conf = [-50.0] * len(out.tokens)
            return out
    t = ToyTarget(table)
    got, dec = dspark.decode(t, Low(table, rng), [1, 2, 3], 20, -1, conf_threshold=0.5)
    assert got == dspark.decode(ToyTarget(table), ConfToyDrafter(table, np.random.default_rng(1)), [1, 2, 3], 20, -1)[0]
    assert dec.stats.steps == 0 and dec.stats.plain_steps == 19 and dec.stats.conf_dropped > 0
    assert dec.stats.verify_positions == 19
    # drafts without a confidence are never cut
    _, dec = dspark.decode(ToyTarget(table), ToyDrafter(table, np.random.default_rng(1), p_err=0.1), [1, 2, 3], 20, -1,
                           conf_threshold=0.99)
    assert dec.stats.conf_dropped == 0
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            dspark.DSparkDecoder(t, ToyDrafter(table, rng), eos_id=-1, conf_threshold=bad)


# ---- draft length auto-tuning -------------------------------------------------------------------------------------


def test_tuner_unit():
    t = dspark.DraftTuner()
    assert 1 <= t.choose() <= 4
    for _ in range(40):
        t.observe(4, 4, 4)                               # every draft accepted
    assert t.choose() == 4 and t.rate > 0.95
    for _ in range(60):
        t.observe(4, 1, 0)                               # first draft always rejected
    assert t.choose() == 1 and t.rate < 0.1
    assert t.choose(cap=1) == 1
    t2 = dspark.DraftTuner(cost_per_position=0.1)
    for _ in range(80):
        t2.observe(4, 4, 2)                              # mixed: accepts 2 of 4
    assert t2.choose() in (1, 2, 3, 4) and t2.choose(cap=2) <= 2
    before = (t2.succ, t2.trials)
    t2.observe(0, 0, 0)
    assert (t2.succ, t2.trials) == before
    for bad in ((0, 4), (3, 2)):
        with pytest.raises(ValueError):
            dspark.DraftTuner(*bad)


def test_tuner_follows_acceptance_and_stays_exact():
    table = toy_table(6)
    prompt = [3, 4, 5]
    ref = dspark.decode(ToyTarget(table), ToyDrafter(table, np.random.default_rng(0), p_err=0.0), prompt, 80, -1)[0]
    cost = {}
    for name, p_err in (("good", 0.0), ("bad", 1.0)):
        for mode in ("fixed", "auto"):
            kw = dict(num_draft=4) if mode == "fixed" else dict(tuner=dspark.DraftTuner())
            got, dec = dspark.decode(ToyTarget(table), ToyDrafter(table, np.random.default_rng(0), p_err=p_err), prompt,
                                     80, -1, **kw)
            assert got == ref
            cost[name, mode] = dec.stats
    assert cost["good", "auto"].tuner_k == 4 and cost["bad", "auto"].tuner_k == 1
    assert cost["bad", "auto"].verify_positions < 0.6 * cost["bad", "fixed"].verify_positions
    assert cost["good", "auto"].verify_positions <= cost["good", "fixed"].verify_positions * 1.2
    assert set(cost["bad", "auto"].draft_len_hist) <= {1, 2, 3, 4}


def test_engine_stats_carry_the_new_fields():
    table = toy_table(2)
    _, dec = dspark.decode(ToyTarget(table), ConfToyDrafter(table, np.random.default_rng(0)), [1, 2, 3], 40, -1,
                           conf_threshold=0.6, tuner=dspark.DraftTuner())
    es = dec.stats.as_engine_stats()
    assert {"draft_len_mean", "verify_calls", "verify_positions", "conf_cut_rounds", "conf_dropped_tokens",
            "tuner_k"} <= set(es)
    assert es["verify_positions"] == sum(k * c for k, c in dec.stats.draft_len_hist.items()) + dec.stats.verify_calls


# ---- the official model code --------------------------------------------------------------------------------------


def _official():
    for mod in ("sympy", "PIL", "tokenizers"):
        pytest.importorskip(mod)
    import dspark_refkit as kit
    return kit


def official_plain(kit, model, prompt, max_tokens, sampler):
    """The plain loop on the official target, one position per verify call."""
    target = kit.OfficialTarget(model)
    out = target.prefill(prompt)
    tok = sampler.sample(out.logits[-1])
    toks, pos = [tok], len(prompt)
    while len(toks) < max_tokens:
        out = target.verify([tok], pos)
        tok = sampler.sample(out.logits[0])
        toks.append(tok)
        pos += 1
    return toks


@pytest.mark.parametrize("model_seed", range(2))
@pytest.mark.parametrize("cfg", [dict(top_p=0.9), dict(top_k=4), dict(min_p=0.1, top_p=0.95)])
def test_official_truncated_sampling_with_conf_and_tuner_equals_plain(model_seed, cfg):
    kit = _official()
    model = kit.build_tiny(model_seed)
    rng = np.random.default_rng(model_seed + 7)
    for i, max_tokens in enumerate((5, 14, 22)):
        seed = 100 * model_seed + i
        prompt = [int(x) for x in rng.integers(0, 31, size=int(rng.integers(3, 12)))]
        torch.manual_seed(seed)
        want = official_plain(kit, model, prompt, max_tokens, make_sampler(0.9, cfg))
        for kw in (dict(), dict(conf_threshold=0.5), dict(tuner=dspark.DraftTuner()),
                   dict(conf_threshold=0.3, tuner=dspark.DraftTuner(), num_draft=3)):
            torch.manual_seed(seed)
            drafter = kit.OfficialDrafter(model)
            got, _ = dspark.decode(kit.OfficialTarget(model), drafter, prompt, max_tokens, -1,
                                   sampler=make_sampler(0.9, cfg), **kw)
            assert got == want, (cfg, kw, seed)


def test_official_drafter_returns_the_confidence_scores():
    kit = _official()
    model = kit.build_tiny(0)
    target = kit.OfficialTarget(model)
    prompt = [3, 9, 4, 5, 6, 7]
    out = target.prefill(prompt)
    d = kit.OfficialDrafter(model).draft(int(out.logits[-1].argmax()), out.hidden, len(prompt) - 1)
    assert len(d.tokens) == 5 and len(d.conf) == 5 and all(math.isfinite(c) for c in d.conf)
    # a threshold above every score drops the whole block; below every score keeps it
    lo, hi = min(d.conf), max(d.conf)
    p_hi = 1 / (1 + math.exp(-(hi + 1.0)))
    p_lo = 1 / (1 + math.exp(-(lo - 1.0)))
    for thr, want in ((p_hi, 0), (p_lo, 4)):
        dec = dspark.DSparkDecoder(None, kit.OfficialDrafter(model), eos_id=-1, conf_threshold=thr, num_draft=4)
        assert dec._conf_len(d, 4) == want
