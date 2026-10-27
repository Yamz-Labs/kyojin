"""DSpark controller tests on a bigram table (CPU, no weights): greedy equals plain decoding, seeded sampling equals
the plain loop, the p/q rule has the target law, edge cases, statistics. The tests on the official model code are in
test_ds41_dspark_official.py."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

torch = pytest.importorskip("torch")
import dspark                      # noqa: E402

V = 12
EOS_TOY = 11


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """Tiny tensors: torch's thread pool only adds contention."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ---- toy world ----------------------------------------------------------------------------------------------------


class ToyTarget:
    """Next-token logits depend on the previous token only: a table. Checks the controller's calls."""

    def __init__(self, table):
        self.table, self.cache = table, []

    def prefill(self, tokens):
        self.cache = list(tokens)
        rows = self.table[list(tokens)]
        return dspark.TargetOut(rows, np.arange(len(tokens), dtype=float)[:, None])

    def verify(self, tokens, start_pos):
        assert start_pos == len(self.cache)
        self.cache += list(tokens)
        return dspark.TargetOut(self.table[list(tokens)], np.arange(start_pos, start_pos + len(tokens), dtype=float)[:, None])

    def truncate(self, length):
        assert length <= len(self.cache)
        del self.cache[length:]


class ToyDrafter:
    """Follows the table (so it is often right) with probability 1 - p_err per token; checks the feed contract:
    every call gets the hidden rows of exactly the positions after the last call, ending at start_pos."""

    def __init__(self, table, rng, p_err=0.3, block_size=5, allow_hole=False):
        self.table, self.rng, self.p_err, self.block_size = table, rng, p_err, block_size
        self.fed, self.allow_hole = 0, allow_hole

    def draft(self, token, hidden, start_pos):
        n = len(hidden)
        assert start_pos > 0
        if self.allow_hole:                        # one-token prompt: position 0 is never sent
            assert start_pos - n + 1 >= self.fed
        else:
            assert start_pos - n + 1 == self.fed, (start_pos, n, self.fed)
            assert [int(h[0]) for h in hidden] == list(range(self.fed, start_pos + 1))
        self.fed = start_pos + 1
        toks, prev = [], token
        for _ in range(self.block_size):
            t = int(self.table[prev].argmax())
            if self.rng.random() < self.p_err:
                t = int(self.rng.integers(V))
            toks.append(t)
            prev = t
        return dspark.DraftOut(toks)


def toy_table(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(V, V)) * 3.0


def plain_greedy(table, prompt, max_tokens, eos):
    out, tok = [], int(table[prompt[-1]].argmax())
    while True:
        out.append(tok)
        if tok == eos or len(out) >= max_tokens:
            break
        tok = int(table[tok].argmax())
    return out[:out.index(eos)] if eos in out else out


@pytest.mark.parametrize("seed", range(200))
def test_toy_greedy_equals_plain(seed):
    rng = np.random.default_rng(1000 + seed)
    table = toy_table(seed % 25)
    prompt = [int(x) for x in rng.integers(0, V - 1, size=int(rng.integers(2, 12)))]
    max_tokens = int(rng.integers(1, 40))
    eos = EOS_TOY if seed % 3 else -1
    num_draft = [None, 1, 2, 3, 5, 9][seed % 6]
    target = ToyTarget(table)
    drafter = ToyDrafter(table, rng, p_err=[0.0, 0.2, 0.6, 1.0][seed % 4])
    got, dec = dspark.decode(target, drafter, prompt, max_tokens, eos, num_draft=num_draft)
    want = plain_greedy(table, prompt, max_tokens, eos)
    assert got == want
    emitted = len(want) + (1 if eos in plain_greedy_raw(table, prompt, max_tokens, eos) else 0)
    assert dec.stats.emitted == emitted
    assert len(target.cache) == len(prompt) + emitted - 1            # the last token is never fed
    assert target.cache == (prompt + plain_greedy_raw(table, prompt, max_tokens, eos))[:len(target.cache)]
    cap = min(num_draft or 5, 5)
    assert all(1 <= k <= cap for k in dec.stats.draft_len_hist)
    assert sum(dec.stats.draft_len_hist.values()) == dec.stats.steps
    assert sum(dec.stats.accepted_hist.values()) == dec.stats.steps
    assert dec.stats.accepted <= dec.stats.proposed == sum(k * c for k, c in dec.stats.draft_len_hist.items())


def plain_greedy_raw(table, prompt, max_tokens, eos):
    out, tok = [], int(table[prompt[-1]].argmax())
    while True:
        out.append(tok)
        if tok == eos or len(out) >= max_tokens:
            return out
        tok = int(table[tok].argmax())


def plain_sampled(table, prompt, max_tokens, eos, temperature, seed):
    torch.manual_seed(seed)
    s = dspark.GumbelSampler(temperature)
    out, prev = [], prompt[-1]
    while True:
        tok = s.sample(torch.tensor(table[prev]))
        out.append(tok)
        if tok == eos or len(out) >= max_tokens:
            return out[:out.index(eos)] if eos in out else out
        prev = tok


@pytest.mark.parametrize("seed", range(60))
def test_toy_sampling_matches_plain_loop_under_seed(seed):
    rng = np.random.default_rng(seed)
    table = toy_table(seed % 7) * 0.5
    prompt = [int(x) for x in rng.integers(0, V - 1, size=int(rng.integers(2, 8)))]
    max_tokens, temp = int(rng.integers(1, 50)), [0.7, 1.0, 1.3][seed % 3]
    want = plain_sampled(table, prompt, max_tokens, EOS_TOY, temp, seed)
    torch.manual_seed(seed)
    got, dec = dspark.decode(ToyTarget(torch.tensor(table)), ToyDrafter(table, rng, p_err=[0.0, 0.4, 0.8][seed % 3]),
                             prompt, max_tokens, EOS_TOY, sampler=dspark.GumbelSampler(temp))
    # drafts are argmax of the table, so they are right sometimes; the output must not care
    assert got == want


def test_toy_perfect_drafter_accepts_whole_blocks():
    table = np.zeros((V, V))
    for i in range(V):
        table[i, (i * 5 + 3) % (V - 1)] = 9.0                  # a deterministic cycle, never EOS
    target = ToyTarget(table)
    got, dec = dspark.decode(target, ToyDrafter(table, np.random.default_rng(0), p_err=0.0), [1, 2], 31, EOS_TOY)
    assert got == plain_greedy(table, [1, 2], 31, EOS_TOY)
    st = dec.stats
    assert st.accepted_hist[5] >= 4 and st.draft_len_hist[5] >= 4          # whole blocks, plus a bonus token each
    assert st.accepted_per_step > 4 and st.tokens_per_step > 5
    assert st.rejected == st.proposed - st.accepted


def test_toy_stats_histogram_shape():
    table = toy_table(3)
    got, dec = dspark.decode(ToyTarget(table), ToyDrafter(table, np.random.default_rng(1), p_err=0.5), [1, 2, 3], 60, -1)
    st = dec.stats
    assert st.emitted == len(got) == 60
    assert isinstance(st.draft_len_hist, Counter) and set(st.draft_len_hist) <= {1, 2, 3, 4, 5}
    es = st.as_engine_stats()
    assert es["accepted_draft_tokens"] == st.accepted and es["rejected_draft_tokens"] == st.rejected
    assert es["draft_steps"] == st.steps and es["accepted_per_step"] == round(st.accepted / st.steps, 4)


def test_toy_edge_eos_inside_block_and_budget():
    # a chain 1 -> 2 -> 3 -> 4 -> EOS: EOS lands inside the first block
    table = np.zeros((V, V))
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, EOS_TOY)]:
        table[a, b] = 9.0
    for p_err in (0.0, 1.0):
        t = ToyTarget(table)
        got, dec = dspark.decode(t, ToyDrafter(table, np.random.default_rng(0), p_err=p_err), [5, 0], 50, EOS_TOY)
        assert got == [1, 2, 3, 4]
        assert t.cache == [5, 0, 1, 2, 3, 4]                       # EOS sampled, never fed
    t = ToyTarget(table)                                         # budget ends mid-block
    got, dec = dspark.decode(t, ToyDrafter(table, np.random.default_rng(0), p_err=0.0), [5, 0], 3, EOS_TOY)
    assert got == [1, 2, 3] and t.cache == [5, 0, 1, 2]
    assert max(dec.stats.draft_len_hist) <= 2
    got, _ = dspark.decode(ToyTarget(table), ToyDrafter(table, np.random.default_rng(0)), [5, 0], 1, EOS_TOY)
    assert got == [1]
    assert dspark.decode(ToyTarget(table), ToyDrafter(table, np.random.default_rng(0)), [5, 0], 0, EOS_TOY)[0] == []


def test_toy_first_token_is_eos_and_bad_num_draft():
    table = np.zeros((V, V))
    table[:, EOS_TOY] = 5.0
    t = ToyTarget(table)
    got, dec = dspark.decode(t, ToyDrafter(table, np.random.default_rng(0)), [1, 2], 10, EOS_TOY)
    assert got == [] and t.cache == [1, 2] and dec.stats.steps == 0
    with pytest.raises(ValueError):
        dspark.DSparkDecoder(t, ToyDrafter(table, np.random.default_rng(0)), eos_id=0, num_draft=0)


def test_toy_one_token_prompt_never_drafts_from_position_zero():
    for seed in range(20):
        table = toy_table(seed)
        t = ToyTarget(table)
        got, dec = dspark.decode(t, ToyDrafter(table, np.random.default_rng(seed), p_err=0.2, allow_hole=True),
                                 [3], 25, -1)
        assert got == plain_greedy(table, [3], 25, -1)
        assert dec.stats.plain_steps >= 1 and dec.stats.steps >= 1


def test_toy_stream_yields_rounds():
    table = toy_table(5)
    dec = dspark.DSparkDecoder(ToyTarget(table), ToyDrafter(table, np.random.default_rng(2), p_err=0.1), eos_id=-1)
    chunks = list(dec.stream([1, 2, 3], 25))
    assert sum(map(len, chunks)) == 25 and len(chunks[0]) == 1 and len(chunks) < 25


# ---- speculative sampling with draft laws: exact output law ---------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_pq_rule_has_the_target_law(seed):
    g = torch.Generator().manual_seed(seed)
    n = 4
    rows = torch.randn(3, n, generator=g) * 1.2                                  # target logits for 2 drafts + bonus
    qs = [torch.softmax(torch.randn(n, generator=g) * 1.5, -1) for _ in range(2)]  # arbitrary draft laws
    p = [torch.softmax(r, -1) for r in rows]
    dec = dspark.DSparkDecoder(None, type("D", (), {"block_size": 2})(), eos_id=-1, sampler=dspark.GumbelSampler(1.0, g))
    first, second = Counter(), Counter()
    trials = 30000
    for _ in range(trials):
        d = [int(torch.multinomial(q, 1, generator=g)) for q in qs]
        toks, _ = dec._walk(dspark.TargetOut(rows, None), dspark.DraftOut(d, qs), 2, 3)
        first[toks[0]] += 1
        if len(toks) > 1:
            second[toks[1]] += 1
    for i in range(n):
        assert abs(first[i] / trials - float(p[0][i])) < 0.012
    m = sum(second.values())
    for i in range(n):
        assert abs(second[i] / m - float(p[1][i])) < 0.015


def test_one_hot_draft_acceptance_is_p_of_draft():
    g = torch.Generator().manual_seed(0)
    row = torch.randn(1, 6, generator=g)
    dec = dspark.DSparkDecoder(None, type("D", (), {"block_size": 1})(), eos_id=-1, sampler=dspark.GumbelSampler(1.0, g))
    hits = sum(dec._walk(dspark.TargetOut(torch.cat([row, row]), None), dspark.DraftOut([2]), 1, 2)[1] for _ in range(20000))
    assert abs(hits / 20000 - float(torch.softmax(row[0], -1)[2])) < 0.012
