"""Prompt-state checkpoints (exllamav3/dsv41/prefix_cache.py) behind PortEngine, plain and drafted decode. CPU, tiny random
models with the real structure (the reference model and the serving model, ring of 16 so that chunk ends and ring wrap are hit).

  * store: longest strict prefix, LRU under a byte budget and an entry count, an older reply of the same conversation is superseded
  * snapshot / restore: the restored state equals the state that was copied (rings, pools, open compressor groups, Engram history)
  * resume == full prefill: next-token logits within tolerance, greedy tokens identical over 64 tokens
  * every kind of snapshot is hit: end of prompt, the generation-prompt mark (a render that flips the token after <Assistant>),
    the system-block mark, long-prompt chunk ends, the reply (verbatim agent loop); a diverging prompt falls back to the LCP snapshot
  * drafted decode takes the same snapshots, rebuilds the drafter rings from the saved target rows, and equals plain decode
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(REPO, "tests", "dsv41"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_dsv41_serving as S  # noqa: E402
import test_dsv41_spec as T  # noqa: E402
from ds41_model import PortEngine, TopPSampler  # noqa: E402
from exllamav3.dsv41 import prefix_cache as pc  # noqa: E402

EOS = -1
USER_ID, ASSISTANT_ID = 5, 6                         # stand-ins for <User> / <Assistant> in a tiny vocabulary


def store(gib=0.5, n=48):
    return pc.PrefixStore(int(gib * (1 << 30)), n, min_free_gib=0.0)


@pytest.fixture(scope="module")
def serving():
    cfg, sd, tables, tmap = S.make_world(2, sliding_window=16)
    return cfg, S.build_serving(cfg, sd, tables, tmap, prefill_chunk=24)


@pytest.fixture(scope="module")
def reference():
    return T.build(1, sliding_window=16)


def ids_of(cfg, seed, n):
    ids = T.prompt_ids(cfg, seed, n)
    return [i if i not in (USER_ID, ASSISTANT_ID, 0) else 7 for i in ids]


def greedy(eng, ids, n, **kw):
    out = []
    st = eng._run(list(ids), n, TopPSampler(0.0), EOS, None, out.extend, **kw)
    return out, st


def fresh(model, **kw):
    return PortEngine(model, None, ctx=400, eos_id=EOS, **kw)


# ---------------------------------------------------------------------------------------------------- store


def snap(keys, kind="end", nbytes=100):
    return pc.Snapshot(list(keys), len(keys), kind, {}, None, nbytes)


def test_lcp():
    assert pc.lcp([1, 2, 3, 4], [1, 2, 9]) == 2 and pc.lcp([1, 2], [1, 2, 3]) == 2 and pc.lcp([], [1]) == 0 and pc.lcp([5], [6]) == 0


def test_lookup_takes_the_longest_strict_prefix():
    s = store()
    s.put(snap(range(10)))
    s.put(snap(range(20)))
    s.put(snap([0, 1, 2, 99]))
    assert s.lookup(list(range(30))).n == 20
    assert s.lookup(list(range(20))).n == 10                 # a snapshot of the whole prompt leaves nothing to prefill: not usable
    assert s.lookup(list(range(5))) is None
    assert s.lookup([0, 1, 2, 99, 4]).n == 4
    assert s.best_lcp([0, 1, 2, 3, 77]) [0] in (3, 4)


def test_budget_count_and_supersede():
    s = pc.PrefixStore(250, 3, min_free_gib=0.0)
    for i in range(4):
        s.put(snap([i] * (i + 1), nbytes=100))
    assert len(s) == 2 and s.nbytes() <= 250                  # bytes: oldest dropped first
    s = store(n=3)
    for i in range(5):
        s.put(snap([i] * 3))
    assert len(s) == 3
    s = store()
    s.put(snap([1, 2, 3], "reply"))
    s.put(snap([9, 9, 9], "reply"))
    s.put(snap([1, 2, 3, 4, 5], "reply"))                      # the same conversation, longer: the older reply goes
    assert sorted(e.n for e in s.entries) == [3, 5] and [e.keys for e in s.entries if e.n == 3] == [[9, 9, 9]]
    s.put(snap([9, 9, 9], "reply"))
    assert len(s) == 2                                          # same position again: replaced, not duplicated
    assert pc.PrefixStore(10, 3, min_free_gib=0.0).put(snap([1], nbytes=100)) is None


def test_marks():
    ids = list(range(100, 140))
    ids[3] = 7
    assert pc.marks_for(ids, 0, user_id=None, assistant_id=None) == []
    ids[30] = ASSISTANT_ID
    assert pc.marks_for(ids, 0, user_id=None, assistant_id=ASSISTANT_ID) == [31]
    assert pc.marks_for(ids, 31, user_id=None, assistant_id=ASSISTANT_ID) == []
    ids[20] = USER_ID
    assert pc.marks_for(ids, 0, user_id=USER_ID, assistant_id=ASSISTANT_ID, min_sys=16) == [20, 31]
    assert pc.marks_for(ids, 0, user_id=USER_ID, assistant_id=ASSISTANT_ID, min_sys=512) == [31]


# ---------------------------------------------------------------------------------------------------- snapshot / restore


def state_equal(cfg, a, b, n):
    for l in a.win:
        assert torch.equal(a.win[l], b.win[l]), f"win {l}"
    for name in pc.POOLS:
        for l, t in (getattr(a, name, None) or {}).items():
            rows = -(-n // max(int(cfg.ratio(l)), 1))
            assert torch.equal(t[:, :rows], getattr(b, name)[l][:, :rows]), f"{name} {l}"
    for l in a.comp_kv_state:
        assert torch.equal(a.comp_kv_state[l], b.comp_kv_state[l]) and torch.equal(a.comp_score_state[l], b.comp_score_state[l])
    if a.hasher is not None:
        assert torch.equal(a.hasher.cache[:, :n], b.hasher.cache[:, :n])


@pytest.mark.parametrize("which", ["serving", "reference"])
def test_snapshot_restore_roundtrip(which, serving, reference):
    cfg, model = serving if which == "serving" else reference
    ids = ids_of(cfg, 3, 53)                                   # not a multiple of any compression ratio: open groups are live
    st = model.new_state(1, 256)
    model.forward(torch.tensor([ids]), 0, st)
    sn = pc.snapshot(cfg, st, len(ids), ids, None, "end")
    back = pc.restore(model, sn, 256)
    state_equal(cfg, st, back, len(ids))
    assert sn.nbytes > 0 and sn.keys == ids


# ---------------------------------------------------------------------------------------------------- resume == full prefill


def full_logits(model, ids):
    st = model.new_state(1, 400)
    lg, _ = (model.forward(torch.tensor([ids]), 0, st))
    return lg[-1]


@pytest.mark.parametrize("which", ["serving", "reference"])
def test_resume_logits_and_greedy_tokens_equal_a_full_prefill(which, serving, reference):
    cfg, model = serving if which == "serving" else reference
    base = ids_of(cfg, 4, 70)
    longer = base + ids_of(cfg, 5, 37)
    with T.smooth_graph():
        s = store()
        a = pc.prefill(model, s, base, max_seq=400)
        assert a.cached == 0 and torch.allclose(a.logits[-1], full_logits(model, base), atol=1e-4)
        b = pc.prefill(model, s, longer, max_seq=400)
        assert b.cached == len(base)
        assert torch.allclose(b.logits[-1], full_logits(model, longer), atol=1e-4, rtol=1e-4)
        eng_c, eng_w = fresh(model, prefix_cache=False), fresh(model)
        want, sw = greedy(eng_c, longer, 64)
        greedy(eng_w, base, 4)
        got, sg = greedy(eng_w, longer, 64)
        assert sw["cached_tokens"] == 0 and sg["cached_tokens"] == len(base)
        assert got == want


def test_every_snapshot_kind_is_hit(serving):
    cfg, model = serving
    ids = ids_of(cfg, 6, 130)
    ids[11] = USER_ID                                           # system block = ids[:11] is too short: raise min via tail below
    sys_len = 60
    ids[sys_len] = USER_ID
    ids[100] = ASSISTANT_ID                                     # generation prompt: mark at 101
    with T.smooth_graph():
        eng = fresh(model)
        eng._user_id, eng._assistant_id = USER_ID, ASSISTANT_ID
        s = eng.pfx = store()
        # min_sys is 512 in production; here the system block is below it, so only the generation-prompt mark exists
        out, st0 = greedy(eng, ids, 8)
        kinds = {(e.kind, e.n) for e in s.entries}
        assert ("end", len(ids)) in kinds and ("mark", 101) in kinds
        assert any(k == "grid" for k, _ in kinds), kinds         # prefill_chunk 24 < 130: chunk ends are snapshotted
        assert ("reply", len(ids) + 7) in kinds                  # 8 tokens produced, the last one never fed
        # 1. the client drops the reasoning: the render flips the token after <Assistant>; the mark at 101 is the LCP
        flipped = ids[:101] + [9, 9] + ids_of(cfg, 7, 30)
        _, st = greedy(eng, flipped, 4)
        assert st["cached_tokens"] == 101
        # 2. the client sends the reply back verbatim and a follow-up
        follow = ids + out[:-1] + ids_of(cfg, 8, 12)
        _, st = greedy(eng, follow, 4)
        assert st["cached_tokens"] == len(ids) + 7
        # 3. a prompt that diverges in the middle falls back to a chunk end below the divergence
        mid = ids[:70] + [8] * 20
        _, st = greedy(eng, mid, 2)
        assert 0 < st["cached_tokens"] <= 70 and st["cached_tokens"] % 24 == 0
        # 4. a different conversation: nothing to reuse
        _, st = greedy(eng, ids_of(cfg, 9, 50), 2)
        assert st["cached_tokens"] == 0


def test_cache_prompt_false_neither_reads_nor_writes(serving):
    cfg, model = serving
    eng = fresh(model)
    eng.pfx = store()
    ids = ids_of(cfg, 10, 40)
    greedy(eng, ids, 3, cache_prompt=False)
    assert len(eng.pfx) == 0
    greedy(eng, ids, 3)
    n = len(eng.pfx)
    _, st = greedy(eng, ids + ids_of(cfg, 11, 5), 3, cache_prompt=False)
    assert st["cached_tokens"] == 0 and len(eng.pfx) == n


def test_reset_prefix_cache_and_warmup_leave_nothing(serving):
    import asyncio
    cfg, model = serving
    eng = fresh(model)
    eng.pfx = store()
    greedy(eng, ids_of(cfg, 12, 40), 3)
    assert len(eng.pfx) > 0
    asyncio.run(eng.reset_prefix_cache())
    assert len(eng.pfx) == 0


def test_persisted_snapshots_resume_a_new_store_with_the_same_tokens(serving, tmp_path, monkeypatch):
    """Mark / end / reply snapshots go to the disk (background writer, token-hash key); a new store with the same fingerprint (a
    restarted lane) indexes them, reads the deepest one lazily and resumes with the same greedy tokens as the in-RAM resume;
    a different fingerprint or a torn payload loads nothing; clear() forgets the disk entries too."""
    cfg, model = serving
    monkeypatch.setattr(pc, "PERSIST_MIN", 20)
    real = pc.marks_for
    monkeypatch.setattr(pc, "marks_for", lambda *a, **k: real(*a, **{**k, "min_sys": 20}))
    monkeypatch.setenv("EXL3_PREFIX_DISK_FLOOR_GB", "0")
    ids = ids_of(cfg, 21, 90)
    ids[40] = USER_ID
    mk = lambda fp="a": pc.PrefixStore(1 << 29, 48, min_free_gib=0.0, persist_dir=str(tmp_path), fingerprint=fp)  # noqa: E731
    with T.smooth_graph():
        ref = fresh(model)                                           # pure RAM: the reference of the disk resume
        ref._user_id, ref._assistant_id = USER_ID, ASSISTANT_ID
        ref.pfx = store()
        cold_ref, _ = greedy(ref, ids, 6)
        follow = ids + cold_ref[:-1] + ids_of(cfg, 8, 12)
        ram, st_ram = greedy(ref, follow, 4)
        assert st_ram["cached_tokens"] == len(ids) + 5
        eng = fresh(model)
        eng._user_id, eng._assistant_id = USER_ID, ASSISTANT_ID
        eng.pfx = mk()
        cold, _ = greedy(eng, ids, 6)
        assert cold == cold_ref
        eng.pfx.disk.flush()
        kinds = sorted(n.split("_")[0] for n in eng.pfx.disk.entries())
        assert kinds == ["end", "mark", "reply"], kinds              # no grid kind on disk
        assert mk("other").load_dir() == 0
        eng2 = fresh(model)                                          # a restarted lane: empty RAM, same disk
        eng2._user_id, eng2._assistant_id = USER_ID, ASSISTANT_ID
        eng2.pfx = mk()
        n_files = eng2.pfx.load_dir()
        assert n_files == 3 and len(eng2.pfx) == 0                   # lazy: nothing in RAM yet
        disk, st = greedy(eng2, follow, 4)
        assert st["cached_tokens"] == len(ids) + 5 and disk == ram
        # a torn payload (index present, payload junk) is dropped and the request still runs
        eng3 = fresh(model)
        eng3._user_id, eng3._assistant_id = USER_ID, ASSISTANT_ID
        eng3.pfx = mk()
        eng3.pfx.load_dir()
        eng3.pfx.disk.flush()
        for n in eng3.pfx.disk.entries():
            if n.startswith("reply"):
                (tmp_path / eng3.pfx.fingerprint / (n + ".pt")).write_bytes(b"junk")
        out3, st3 = greedy(eng3, follow, 4)
        assert out3 == ram and 0 < st3["cached_tokens"] < len(ids) + 5          # falls back to the prompt end / the mark
        eng3.pfx.clear()
        assert eng3.pfx.disk.entries() == [] and eng3.pfx.disk_index == {}


# ---------------------------------------------------------------------------------------------------- drafted decode


@pytest.mark.parametrize("which", ["serving", "reference"])
def test_drafted_decode_reuses_the_cache_and_equals_plain(which, serving, reference):
    cfg, model = serving if which == "serving" else reference
    ids = ids_of(cfg, 13, 90)
    ids[50] = ASSISTANT_ID
    with T.smooth_graph():
        plain = fresh(model, prefix_cache=False)
        spec = fresh(model, num_draft=4)
        spec._user_id, spec._assistant_id = USER_ID, ASSISTANT_ID
        spec.spec.attach_prefix(spec.pfx, USER_ID, ASSISTANT_ID)
        out0, st0 = greedy(spec, ids, 20)
        assert st0["cached_tokens"] == 0 and any(e.kind == "mark" for e in spec.pfx.entries)
        assert out0 == greedy(plain, ids, 20)[0]
        # the reply goes back verbatim (agent loop) with a short follow-up
        follow = ids + out0[:-1] + ids_of(cfg, 14, 9)
        want, _ = greedy(plain, follow, 64)
        got, st1 = greedy(spec, follow, 64)
        assert st1["cached_tokens"] == len(ids) + len(out0) - 1
        assert got == want
        # client dropped the reasoning: the render flips right after the mark at 51
        flipped = ids[:51] + [9] + ids_of(cfg, 15, 25)
        want, _ = greedy(plain, flipped, 64)
        got, st2 = greedy(spec, flipped, 64)
        assert st2["cached_tokens"] == 51 and got == want
        # the drafter rings were rebuilt: it still proposes (drafts accepted in the resumed run)
        assert st1["draft_steps"] > 0 and spec.spec.drafter.holes == 0


def test_drafted_resume_logits_equal_a_full_prefill(serving):
    cfg, model = serving
    from exllamav3.dsv41.spec import PortSpec
    base = ids_of(cfg, 16, 80)
    longer = base + ids_of(cfg, 17, 21)
    with T.smooth_graph():
        spec = PortSpec(model, 400, max_depth=4)
        s = store()
        spec.begin_request(None, s)
        spec.target.prefill(base)
        spec.begin_request(None, s)
        out = spec.target.prefill(longer)
        assert spec.cached == len(base)
        assert torch.allclose(out.logits[-1], full_logits(model, longer), atol=1e-4, rtol=1e-4)
        assert out.hidden.size(0) <= cfg.sliding_window and spec.drafter._next == len(longer) - out.hidden.size(0)
