"""CPU tests of the multi-session slot ops (loader/mu_slots.py): batched rows of several sessions == each session stepped alone (solo ops of step_graph /
ref_fused), slab isolation (neighbour slots untouched, scrambled neighbours change nothing), bounds (slots outside the active ones keep a sentinel).
Tiny config: kpool 4, indexer head dim 16, top_k 16 (n_top 3: scoring starts past 16 positions), MLA R 8 + rope 4."""
import math
import os
import sys
import types
import importlib.util

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.environ.get("MU_DEV", "cpu")      # MU_DEV=cuda: the same tests on the GPU (compiled Triton kernels)
torch.set_default_device(DEV)


def _load():
    for n, s in (("exllamav3", ""), ("exllamav3.loader", "loader")):
        m = sys.modules.get(n) or types.ModuleType(n)
        m.__path__ = [os.path.join(ROOT, "exllamav3", s)]
        sys.modules[n] = m
    from exllamav3.loader import mu_slots, step_graph, indexer_fused, ref_fused
    return mu_slots, step_graph, indexer_fused, ref_fused


MS, SG, IXF, RF = _load()
K, D, H, E, RQ = 4, 16, 2, 24, 12
CAP, S = 64, 4


def make_ix(seed=0, cap=CAP):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.5
    w = {"attn_k": r(D, E), "q_b": r(H * D, RQ), "value": r(D, D), "gate": r(D, D), "k_norm": 1 + r(D) * 0.1, "proj": r(H, E), "ape": r(K, D) * 0.1,
         "pool_norm": 1 + r(D) * 0.1}
    return IXF.FusedKPoolIndexer(w, DEV, n_head=H, head_dim=D, top_k=16, kpool=K, select_tail=True, rope_dim=8, rope_base=1e4, eps=1e-5, cap=cap)


def solo_ix(w_from, cap=CAP):
    ix = make_ix(0, cap)
    ix.w = w_from.w
    SG.indexer_static_init(ix, cap)
    return ix


def causal(pos, Nb):
    return torch.arange(Nb)[None] <= pos[:, None]


def run_case(plan, scramble=False, poolk=False, seed=1):
    """plan: list of steps, each a list of (slot, T) rows; returns (ok flags) comparing the batched slab ops with solo ops per session"""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ixb = make_ix(0)
    SG.indexer_static_init(ixb, CAP)
    sl = MS.IxSlabs(ixb, S, CAP)
    if scramble:
        sl.vgb.normal_(generator=g)
        sl.pool.normal_(generator=g)
    solo = {}
    ar = torch.arange(CAP)
    nxt = {}
    for step in plan:
        rows = []
        for slot, T in step:
            n0 = nxt.get(slot, 0)
            nxt[slot] = n0 + T
            rows.append((slot, n0, T))
        xs = {r: torch.randn(r[2], E, generator=g) for r in rows}
        qs = {r: torch.randn(r[2], RQ, generator=g) for r in rows}
        pos = torch.cat([torch.arange(n0, n0 + T) for _, n0, T in rows])
        base = torch.cat([torch.full((T,), slot * CAP) for slot, _, T in rows])
        x = torch.cat([xs[r] for r in rows])
        qr = torch.cat([qs[r] for r in rows])
        Nb = min(CAP, ((int((pos).max()) + 1 - 1) // 16 + 1) * 16)
        al_b = MS.indexer_forward_slots(ixb, sl, x, qr, pos, base, Nb, ar, poolk=poolk)
        if al_b is None:
            al_b = causal(pos, Nb)
        o = 0
        for slot, n0, T in rows:
            ixs = solo.get(slot)
            if ixs is None:
                ixs = solo[slot] = solo_ix(ixb)
            p = torch.arange(n0, n0 + T)
            Nbs = min(CAP, ((n0 + T - 1) // 16 + 1) * 16)
            al_s = SG.indexer_forward_static(ixs, xs[(slot, n0, T)], qs[(slot, n0, T)], p, Nbs, ar)
            if al_s is None:
                al_s = causal(p, Nbs)
            got = al_b[o:o + T]
            assert torch.equal(got[:, :Nbs], al_s), f"allow differs slot {slot} n0 {n0}"
            assert not got[:, Nbs:].any() or bool((got[:, Nbs:] <= causal(p, Nb)[:, Nbs:]).all())
            o += T
    for slot, ixs in solo.items():
        n = nxt[slot]
        pv, vv = sl.view(slot)
        assert torch.equal(vv[:n], ixs.vgb[:n]), f"vgb rows differ slot {slot}"
        ng = n // K                                    # completed groups: pooled keys
        tol = dict(rtol=0, atol=0) if not poolk else dict(rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pv[:ng], ixs.pool[:ng], **tol)
    return sl, nxt


PLAN = [[(1, 3), (2, 3)], [(1, 1), (2, 1)], [(1, 4), (2, 2)]] + [[(1, 1), (2, 1)]] * 25 + [[(2, 2)], [(1, 3)]] + [[(1, 1), (2, 4)]] * 6


def test_indexer_batched_equals_solo_past_scoring_bound():
    run_case(PLAN)


def test_indexer_staggered_sessions_unequal_positions():
    plan = [[(0, 5)], [(0, 5), (3, 2)], [(3, 3)]] + [[(0, 1), (3, 1)]] * 30 + [[(3, 1)]] * 7 + [[(0, 1), (3, 1)]] * 4
    run_case(plan)


def test_indexer_neighbour_garbage_changes_nothing_and_idle_slots_untouched():
    sl, nxt = run_case(PLAN, scramble=False)
    k0 = None
    # scrambled slab content (finite garbage in every slot incl. the active ones beyond their position): same allow / same rows below the position
    sl2, _ = run_case(PLAN, scramble=True)
    for slot in (0, 3):                                  # idle slots: never written
        assert not sl.vgb[slot * CAP:(slot + 1) * CAP].any() and not sl.pool[slot * CAP // K:(slot + 1) * CAP // K].any()


def test_indexer_sentinel_slots_outside_the_batch_keep_their_values():
    ixb = make_ix(0)
    SG.indexer_static_init(ixb, CAP)
    sl = MS.IxSlabs(ixb, S, CAP)
    sl.vgb.fill_(7.0)
    sl.pool.fill_(7.0)
    ar = torch.arange(CAP)
    g = torch.Generator(device=DEV).manual_seed(5)
    T = 8
    for n0 in range(0, CAP, T):                           # slot S - 1 (the last slab) and slot 1 run to their last position
        pos = torch.cat([torch.arange(n0, n0 + T)] * 2)
        base = torch.cat([torch.full((T,), (S - 1) * CAP), torch.full((T,), CAP)])
        MS.indexer_forward_slots(ixb, sl, torch.randn(2 * T, E, generator=g), torch.randn(2 * T, RQ, generator=g), pos, base, min(CAP, n0 + T + 8), ar)
    for slot in (0, 2):
        assert bool((sl.vgb[slot * CAP:(slot + 1) * CAP] == 7.0).all()) and bool((sl.pool[slot * CAP // K:(slot + 1) * CAP // K] == 7.0).all())
    assert not bool((sl.vgb[S * CAP - 1] == 7.0).all()) and not bool((sl.pool[-1] == 7.0).all())


def test_indexer_poolk_triton_interpreter_matches_torch_path():
    if (os.environ.get("TRITON_INTERPRET") != "1" and DEV == "cpu"):
        pytest.skip("needs TRITON_INTERPRET=1")
    run_case(PLAN[:12], poolk=True)


# ------------------------------------------------------------------------------------------------ attention core
def attn_inputs(seed=3):
    g = torch.Generator(device=DEV).manual_seed(seed)
    R, P = 8, 4
    kflat = torch.randn(S * CAP, R + P, generator=g)
    return g, R, P, kflat


@pytest.mark.parametrize("window", [0, 7])
def test_mla_slots_ref_equals_per_row_solo_reference(window):
    g, R, P, kflat = attn_inputs()
    rows = [(1, 40), (2, 3), (1, 41), (3, 63), (0, 0)]
    T = len(rows)
    ql, qp = torch.randn(T, 2, R, generator=g), torch.randn(T, 2, P, generator=g)
    sinks = torch.randn(2, generator=g)
    qpos = torch.tensor([p for _, p in rows])
    base = torch.tensor([s * CAP for s, _ in rows])
    N = 64
    allow = torch.rand(T, N, generator=g) > 0.3
    for al in (None, allow):
        got = MS.mla_slots_ref(ql, qp, kflat, qpos, base, N, 0.3, sinks, window, al)
        for t, (s, p) in enumerate(rows):
            kk = kflat[s * CAP:s * CAP + N]
            want = RF.mla_sink_attend_ref(ql[t:t + 1], qp[t:t + 1], kk[:, :R], kk[:, R:], qpos[t:t + 1], torch.arange(N), 0.3, sinks, window, None if al is None else al[t:t + 1])
            torch.testing.assert_close(got[t], want[0], rtol=1e-5, atol=1e-6)


def test_mla_slots_other_slabs_and_future_rows_are_not_read():
    g, R, P, kflat = attn_inputs()
    T = 3
    ql, qp = torch.randn(T, 2, R, generator=g), torch.randn(T, 2, P, generator=g)
    sinks = torch.randn(2, generator=g)
    qpos = torch.tensor([30, 10, 63])
    base = torch.tensor([CAP, 2 * CAP, 3 * CAP])
    a = MS.mla_slots_ref(ql, qp, kflat, qpos, base, 64, 0.3, sinks, 0, None)
    k3 = kflat.clone()
    k3[:CAP] = float("nan")
    for s_, p_ in ((1, 30), (2, 10)):
        k3[s_ * CAP + p_ + 1:(s_ + 1) * CAP] = float("nan")
    b = MS.mla_slots_ref(ql, qp, k3, qpos, base, 64, 0.3, sinks, 0, None)
    assert torch.isfinite(b).all() and torch.equal(a, b)


@pytest.mark.skipif((os.environ.get("TRITON_INTERPRET") != "1" and DEV == "cpu"), reason="needs TRITON_INTERPRET=1")
@pytest.mark.parametrize("window", [0, 9])
def test_mla_slots_kernel_interpreter_equals_reference(window):
    g, R, P, kflat = attn_inputs()
    rows = [(1, 40), (2, 3), (3, 63), (0, 17)]
    T = len(rows)
    ql, qp = torch.randn(T, 2, R, generator=g), torch.randn(T, 2, P, generator=g)
    sinks = torch.randn(2, generator=g)
    qpos = torch.tensor([p for _, p in rows])
    base = torch.tensor([s * CAP for s, _ in rows])
    allow = torch.rand(T, 64, generator=g) > 0.3
    for al in (None, allow):
        want = MS.mla_slots_ref(ql, qp, kflat, qpos, base, 64, 0.3, sinks, window, al)
        got = MS.mla_attend_slots(ql, qp, kflat, qpos, base, 64, 0.3, sinks, window, al)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
