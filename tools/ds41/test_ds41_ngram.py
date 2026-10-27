"""Prompt lookup (NgramLookup) and its merge with the drafter: greedy output equals plain decoding in every mode (CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
pytest.importorskip("torch")
import dspark                                                                                   # noqa: E402
from test_ds41_dspark import EOS_TOY, ToyDrafter, ToyTarget, plain_greedy, toy_table, V       # noqa: E402,F401


def test_lookup_finds_latest_earlier_occurrence_and_match_length():
    lk = dspark.NgramLookup(3)
    lk.extend([9, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3])
    assert lk.find(4) == ([4, 5, 6, 7], 3)
    lk.extend([4, 5])                                         # tail 3,4,5 now matches too; match length grows to 5 back
    cont, m = lk.find(3)
    assert cont == [6, 7, 1] and m == 5
    lk2 = dspark.NgramLookup(3)
    lk2.extend([1, 2, 3, 4])
    assert lk2.find(2) == ([], 0)                             # the tail itself is not an earlier occurrence
    lk3 = dspark.NgramLookup(3)
    lk3.extend([1, 2, 3, 1, 2, 3])
    assert lk3.find(5) == ([1, 2, 3], 3)                      # continuation cut at the context end


def test_lookup_incremental_equals_fresh():
    rng = np.random.default_rng(3)
    seq = [int(x) for x in rng.integers(0, 4, size=300)]
    inc = dspark.NgramLookup(3)
    for i in range(0, 300, 7):
        inc.extend(seq[i:i + 7])
        fresh = dspark.NgramLookup(3)
        fresh.extend(seq[:min(i + 7, 300)])
        assert inc.find(6) == fresh.find(6)


@pytest.mark.parametrize("seed", range(120))
@pytest.mark.parametrize("mode", ["strong", "extend"])
def test_ngram_modes_equal_plain_greedy(seed, mode):
    rng = np.random.default_rng(7000 + seed)
    table = toy_table(seed % 25)
    prompt = [int(x) for x in rng.integers(0, V - 1, size=int(rng.integers(2, 30)))]
    max_tokens = int(rng.integers(1, 60))
    eos = EOS_TOY if seed % 3 else -1
    target = ToyTarget(table)
    drafter = ToyDrafter(table, rng, p_err=[0.0, 0.3, 0.7, 1.0][seed % 4], block_size=[5, 10][seed % 2])
    dec = dspark.DSparkDecoder(target, drafter, eos_id=eos, num_draft=[2, 4, 8, 10][seed % 4],
                               tuner=dspark.DraftTuner(1, 8) if seed % 2 else None,
                               ngram=mode, ngram_min_match=[3, 4, 6][seed % 3])
    out = []
    for chunk in dec.stream(prompt, max_tokens):
        out.extend(chunk)
    if eos in out:
        out = out[:out.index(eos)]
    assert out == plain_greedy(table, prompt, max_tokens, eos)


def test_ngram_actually_drafts_on_repeating_text():
    # a table that cycles 0 -> 1 -> 2 -> 3 -> 0 ...: the lookup predicts the whole cycle, a useless drafter does not
    table = np.full((V, V), -5.0)
    for i in range(4):
        table[i, (i + 1) % 4] = 5.0
    prompt = [0, 1, 2, 3] * 6
    target, drafter = ToyTarget(table), ToyDrafter(table, np.random.default_rng(0), p_err=1.0)
    dec = dspark.DSparkDecoder(target, drafter, eos_id=-1, num_draft=8, ngram="strong", ngram_min_match=4)
    out = [t for ch in dec.stream(prompt, 40) for t in ch]
    assert out == plain_greedy(table, prompt, 40, -1)
    assert dec.stats.ngram_rounds > 0 and dec.stats.accepted > dec.stats.steps * 3


class WantDrafter(ToyDrafter):
    supports_want = True

    def draft(self, token, hidden, start_pos, want=None):
        out = super().draft(token, hidden, start_pos)
        return dspark.DraftOut(out.tokens[:want] if want else out.tokens)


@pytest.mark.parametrize("seed", range(120))
@pytest.mark.parametrize("ng", ["", "strong", "extend"])
def test_timed_tuner_and_want_equal_plain_greedy(seed, ng):
    rng = np.random.default_rng(9000 + seed)
    table = toy_table(seed % 25)
    prompt = [int(x) for x in rng.integers(0, V - 1, size=int(rng.integers(2, 30)))]
    max_tokens = int(rng.integers(1, 60))
    eos = EOS_TOY if seed % 3 else -1
    target = ToyTarget(table)
    drafter = WantDrafter(table, rng, p_err=[0.0, 0.3, 0.7, 1.0][seed % 4], block_size=5)
    dec = dspark.DSparkDecoder(target, drafter, eos_id=eos, num_draft=[2, 4, 5][seed % 3],
                               tuner=dspark.TimedTuner(0, [2, 4, 5][seed % 3]), ngram=ng, ngram_min_match=3, want_cap=True)
    out = [t for ch in dec.stream(prompt, max_tokens) for t in ch]
    if eos in out:
        out = out[:out.index(eos)]
    assert out == plain_greedy(table, prompt, max_tokens, eos)


def test_timed_tuner_backs_off_when_drafts_fail_and_rounds_are_costly():
    t = dspark.TimedTuner(0, 6)
    for _ in range(30):
        t.observe(4, 1, 0)
        t.observe_time(4, 0.5)
        t.observe_time(0, 0.1)
    assert t.choose() == 0
    u = dspark.TimedTuner(0, 6)
    for _ in range(30):
        u.observe(6, 6, 6)
        u.observe_time(6, 0.12)
        u.observe_time(0, 0.1)
    assert u.choose() == 6
