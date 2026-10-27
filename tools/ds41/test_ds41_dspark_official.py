"""DSpark controller against the official model code (CPU, no weights).

A tiny random-weight Transformer is built from the unmodified official model.py (vendored in testdata/reference/inference;
CUDA kernels replaced by torch equivalents, see dspark_refkit.py), with the real structure: sliding-window ring, compressors
at ratios 2 and 1, indexers, engram, MoE, hyper-connections, three DSpark stages with the Markov head. The official
generate.py is the reference loop. The official code has no speculative loop, so a drafted run must equal the plain loop.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

torch = pytest.importorskip("torch")
for _mod in ("sympy", "PIL", "tokenizers"):                      # imported by the official engram / image code
    pytest.importorskip(_mod)
import dspark                      # noqa: E402
import dspark_refkit as kit        # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """Tiny tensors: torch's thread pool only adds contention."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@functools.lru_cache(maxsize=None)
def tiny(seed):
    return kit.build_tiny(seed)


def rand_prompt(rng, vocab=32, lo=2, hi=24):
    return [int(x) for x in rng.integers(0, vocab - 1, size=int(rng.integers(lo, hi)))]


def spec_run(model, prompt, max_tokens, eos, drafter, sampler=None, num_draft=None, target=None):
    target = target or kit.OfficialTarget(model)
    return dspark.decode(target, drafter, prompt, max_tokens, eos, sampler=sampler, num_draft=num_draft) + (target,)


def test_official_vendored_files_are_the_unmodified_model():
    model, _ = kit.load_official()
    cfg = kit.official_config()
    assert cfg["n_mtp_layers"] == 3 and cfg["dspark_block_size"] == 5 and cfg["dspark_noise_token_id"] == 128799
    assert cfg["dspark_target_layer_ids"] == [37, 38, 39] and cfg["dspark_markov_rank"] == 256
    assert cfg["dspark_n_routed_experts"] == 128 and cfg["dspark_n_activated_experts"] == 3
    m = tiny(0)
    assert len(m.mtp) == 3 and m.mtp[0].block_size == 5 and m.target_layer_ids == (3, 4, 5)
    assert model.ModelArgs(**kit.TINY).dspark_block_size == 5


def test_official_engram_state_is_part_of_the_snapshot_and_matters():
    model = tiny(0)
    assert [i for i, layer in enumerate(model.layers) if layer.engram is not None] == [1, 3]
    assert "engram_hash.cache" in kit.capture(model)
    a, b = [3, 9, 4, 5, 6, 7], [3, 9, 4, 5, 6, 7]
    b[3] = 20
    ids = lambda seq: model.engram_hash(torch.tensor([seq]), 0)[0, -1]       # noqa: E731
    assert not torch.equal(ids(a), ids(b))                                   # a 3-gram reaches back two tokens
    ha = kit.OfficialTarget(model).prefill(a).hidden[-1]
    hb = kit.OfficialTarget(model).prefill(b).hidden[-1]
    assert not torch.allclose(ha, hb)


@pytest.mark.parametrize("model_seed", range(4))
@pytest.mark.parametrize("kind", ["oracle0", "oracle2", "oracle6", "stage"])
def test_official_greedy_identical_to_generate(model_seed, kind):
    model = tiny(model_seed)
    rng = np.random.default_rng(model_seed * 7 + 1)
    for max_tokens in (1, 2, 5, 6, 7, 17, 31):
        prompt = rand_prompt(rng)
        ref = kit.reference_generate(model, prompt, max_tokens, -1)
        if kind == "stage":
            drafter = kit.OfficialDrafter(model)
        else:
            drafter = kit.OracleDrafter(ref, len(prompt), 32, rng, p_err=int(kind[-1]) / 10)
        got, dec, target = spec_run(model, prompt, max_tokens, -1, drafter)
        assert got == ref, (kind, prompt, max_tokens)
        assert target.length == len(prompt) + max_tokens - 1
        if kind == "oracle0" and max_tokens > 6:
            assert dec.stats.accepted_per_step > 3


@pytest.mark.parametrize("model_seed", range(3))
@pytest.mark.parametrize("temperature", [0.8, 1.0])
@pytest.mark.parametrize("kind", ["oracle", "stage"])
def test_official_sampling_matches_generate_under_seed(model_seed, temperature, kind):
    model = tiny(model_seed)
    rng = np.random.default_rng(model_seed + 100)
    for i, max_tokens in enumerate((1, 4, 9, 25)):
        seed, prompt = 1000 * model_seed + i, rand_prompt(rng)
        ref = kit.reference_generate(model, prompt, max_tokens, -1, temperature=temperature, seed=seed)
        drafter = (kit.OfficialDrafter(model) if kind == "stage"
                   else kit.OracleDrafter(ref, len(prompt), 32, rng, p_err=0.3))
        torch.manual_seed(seed)
        got, dec, _ = spec_run(model, prompt, max_tokens, -1, drafter, sampler=dspark.GumbelSampler(temperature))
        assert got == ref, (kind, temperature, seed)
        assert model.temperature == 0
        if kind == "oracle" and max_tokens == 25:
            assert dec.stats.accepted > 0


def test_official_sampled_drafts_with_laws_run_and_are_valid():
    model = tiny(1)
    rng = np.random.default_rng(5)
    for i in range(6):
        prompt = rand_prompt(rng)
        torch.manual_seed(i)
        drafter = kit.OfficialDrafter(model, temperature=1.0)
        got, dec, target = spec_run(model, prompt, 14, -1, drafter, sampler=dspark.GumbelSampler(1.0))
        assert len(got) == 14 and all(0 <= t < 32 for t in got)
        assert target.length == len(prompt) + 13
        assert dec.stats.proposed > 0
    model.mtp[0].temperature = 0
    for stage in model.mtp:
        stage.temperature = 0


class CheckingTarget(kit.OfficialTarget):
    """After every truncate, the live cache entries must equal those of a fresh prefill of the kept tokens."""

    def __init__(self, model):
        super().__init__(model)
        self.checks = 0
        self.seq: list[int] = []

    def prefill(self, tokens):
        self.seq = list(tokens)
        return super().prefill(tokens)

    def verify(self, tokens, start_pos):
        self.seq = self.seq[:start_pos] + list(tokens)
        return super().verify(tokens, start_pos)

    def truncate(self, length):
        super().truncate(length)
        mine = kit.live_view(self.model, length)
        saved, saved_len, saved_ver = kit.capture(self.model), self.length, self._verified
        super().prefill(self.seq[:length])
        fresh = kit.live_view(self.model, length)
        kit.restore(self.model, saved)
        self.length, self._verified = saved_len, saved_ver
        assert mine.keys() == fresh.keys()
        for k in mine:
            assert torch.allclose(mine[k], fresh[k], atol=1e-4, rtol=1e-4), k
        self.checks += 1


@pytest.mark.parametrize("model_seed", range(3))
def test_official_rollback_leaves_cache_equal_to_fresh_prefill(model_seed):
    model = tiny(model_seed)
    rng = np.random.default_rng(model_seed + 50)
    total = 0
    for p_err in (0.15, 0.5, 1.0):
        prompt = rand_prompt(rng, lo=6, hi=22)             # longer than the 8-slot ring after a few steps: wrap-around
        ref = kit.reference_generate(model, prompt, 30, -1)
        target = CheckingTarget(model)
        got, dec, _ = spec_run(model, prompt, 30, -1, kit.OracleDrafter(ref, len(prompt), 32, rng, p_err=p_err), target=target)
        assert got == ref
        total += target.checks
        # the final cache too
        n = target.length
        mine = kit.live_view(model, n)
        full = prompt + got
        fresh_t = kit.OfficialTarget(model)
        fresh_t.prefill(full[:n])
        fresh = kit.live_view(model, n)
        for k in mine:
            assert torch.allclose(mine[k], fresh[k], atol=1e-4, rtol=1e-4), k
    assert total >= 10


def test_official_draft_stage_windows_hold_every_accepted_position():
    model = tiny(2)
    rng = np.random.default_rng(9)
    prompt = rand_prompt(rng, lo=10, hi=14)
    ref = kit.reference_generate(model, prompt, 24, -1)
    # a draft stage that is mostly right: use the stage for the cache, but verify with the oracle's tokens by
    # decoding with the real stage and the real target (acceptance is whatever it is)
    drafter = kit.OfficialDrafter(model)
    got, dec, target = spec_run(model, prompt, 24, -1, drafter)
    assert got == ref
    fed = drafter.fed
    assert fed >= len(prompt) + 10
    mine = {k: v for k, v in kit.live_view(model, target.length, draft_n=fed).items() if k.startswith("mtp.")}
    seq = prompt + got
    fresh_t = kit.OfficialTarget(model)
    out = fresh_t.prefill(seq[:fed])
    kit.OfficialDrafter(model).draft(seq[fed - 1], out.hidden, fed - 1)
    fresh = {k: v for k, v in kit.live_view(model, target.length, draft_n=fed).items() if k.startswith("mtp.")}
    assert mine.keys() == fresh.keys() and len(mine) == 3
    for k in mine:
        assert torch.allclose(mine[k], fresh[k], atol=1e-4, rtol=1e-4), k


def first_new_token(ref, start=2):
    """A token whose first occurrence in `ref` is at index >= start (EOS candidate)."""
    for i, t in enumerate(ref):
        if i >= start and t not in ref[:i]:
            return i, t
    raise AssertionError("no candidate")


@pytest.mark.parametrize("model_seed", range(3))
def test_official_eos_inside_draft_block(model_seed):
    model = tiny(model_seed)
    rng = np.random.default_rng(model_seed + 300)
    found = 0
    for _ in range(4):
        prompt = rand_prompt(rng)
        ref = kit.reference_generate(model, prompt, 20, -1)
        idx, eos = first_new_token(ref, start=2)
        want = kit.reference_generate(model, prompt, 20, eos)
        assert want == ref[:idx]
        for kind in ("oracle0", "oracle3", "stage"):
            truth = ref[:idx + 1]
            drafter = (kit.OfficialDrafter(model) if kind == "stage"
                       else kit.OracleDrafter(truth, len(prompt), 32, rng, p_err=int(kind[-1]) / 10))
            got, dec, target = spec_run(model, prompt, 20, eos, drafter)
            assert got == want
            assert target.length == len(prompt) + idx               # EOS sampled, never fed
            assert dec.stats.emitted == idx + 1
            if kind == "oracle0" and idx <= 5:
                # EOS is draft number idx of the first block: every draft up to it is accepted, EOS included
                assert dec.stats.accepted == idx and dec.stats.steps == 1
                found += 1
    assert found >= 1


@pytest.mark.parametrize("max_tokens", [2, 3, 4, 5, 6, 7, 8, 9, 12])
def test_official_max_tokens_hit_mid_block(max_tokens):
    model = tiny(0)
    rng = np.random.default_rng(max_tokens)
    prompt = rand_prompt(rng)
    ref = kit.reference_generate(model, prompt, 40, -1)
    drafter = kit.OracleDrafter(ref, len(prompt), 32, rng, p_err=0.0)
    got, dec, target = spec_run(model, prompt, max_tokens, -1, drafter)
    assert got == ref[:max_tokens]
    assert target.length == len(prompt) + max_tokens - 1
    assert dec.stats.emitted == max_tokens
    start, fed = target._verified[0], target._verified[1]
    assert start + len(fed) <= len(prompt) + max_tokens
    # never drafts past the budget: the longest verify fits the tokens still owed
    assert all(k <= max_tokens - 1 for k in dec.stats.draft_len_hist)


def test_official_one_token_prompt():
    model = tiny(1)
    rng = np.random.default_rng(4)
    ref = kit.reference_generate(model, [7], 20, -1)
    got, dec, target = spec_run(model, [7], 20, -1, kit.OracleDrafter(ref, 1, 32, rng, p_err=0.2))
    assert got == ref and target.length == 20
    got, dec, _ = spec_run(model, [7], 20, -1, kit.OfficialDrafter(model))
    assert got == ref


def test_official_num_draft_caps_at_block_size():
    model = tiny(0)
    rng = np.random.default_rng(1)
    prompt = rand_prompt(rng)
    ref = kit.reference_generate(model, prompt, 30, -1)
    for nd in (1, 2, 4, 9):
        drafter = kit.OracleDrafter(ref, len(prompt), 32, rng, p_err=0.0)
        got, dec, _ = spec_run(model, prompt, 30, -1, drafter, num_draft=nd)
        assert got == ref
        assert max(dec.stats.draft_len_hist) == min(nd, 5)
