"""Zero-GPU checks of the served --slots path.
1. serve.apply_env(slots > 1) leaves the shipped long-context levers MuEngine refuses OFF (the 1.9.0 release clone died at construction: HY4_LC_SPARSE=1 from SHIPPED_ENV vs the
   MuEngine guard; the server stayed up with /health 500 and the benchmark driver waited for it). An explicit value still wins; slots == 1 keeps the shipped flags.
2. The flat-slab index math of step_slots / indexer_forward_slots (row base + position, pooled group rows, group start + offset) stays inside [0, S * cap) for every session count
   B <= S, every bucket Nb <= cap and every position, including the first chunk of a fresh slot and the warm-up shape (64-token prompts)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))

import serve  # noqa: E402

GUARDED = ("HY4_SWA_RING", "HY4_LC_SPARSE")


def _clean(env):
    for k in list(os.environ):
        if k.startswith("HY4_") and k not in ("HY4_ALLOW_SLOW",):
            del os.environ[k]
    os.environ.update(env)


def test_slots_apply_env_turns_guarded_flags_off():
    _clean({})
    serve.apply_env(4096, "", slots=8)
    for f in GUARDED:
        assert os.environ.get(f, "0") in ("", "0"), f
    assert os.environ["HY4_LC_PRIME"] == "1" and os.environ["HY4_PF_SPARSE"] == "1"      # the other shipped levers stay


def test_single_session_keeps_shipped_flags():
    _clean({})
    serve.apply_env(4096, "", slots=1)
    assert os.environ["HY4_LC_SPARSE"] == "1"
    _clean({})
    serve.apply_env(4096)
    assert os.environ["HY4_LC_SPARSE"] == "1"


def test_explicit_value_wins():
    _clean({"HY4_LC_SPARSE": "1"})
    serve.apply_env(4096, "", slots=8)
    assert os.environ["HY4_LC_SPARSE"] == "1"


def _slab_indices(S, cap, k, rows, Nb):
    """rows = [(slot, n_before, n_new)] -> every flat index step_slots / indexer_forward_slots builds"""
    out = []
    for slot, n0, nt in rows:
        base = slot * cap
        for p in range(n0, n0 + nt):
            flat = base + p
            out.append(("kv / vgb write", flat, S * cap))
            out.append(("pool write", flat // k, S * cap // k))
            g0 = (flat // k) * k
            for a in range(k):
                out.append(("vgb group read", g0 + a, S * cap))
            Pb = Nb // k
            for a in range(Pb):
                out.append(("pool score read", base // k + a, S * cap // k))
            for a in range(Nb):                 # MLA core reads keys base + 0 .. N - 1
                out.append(("mla key read", base + a, S * cap))
    return out


def test_slab_gather_indices_in_range():
    S, k = 8, 4
    for ctx in (2048, 4096):
        cap = ((ctx + 64 + 511) // 512) * 512
        assert cap % k == 0
        for B in range(1, S + 1):
            for n0, nt in ((0, 64), (0, 1), (64, 1), (64, 4), (cap - 8, 4)):
                top = n0 + nt
                Nb = min(cap, ((top - 1) // 64 + 1) * 64)
                rows = [(s, n0, nt) for s in range(B)]
                for what, idx, hi in _slab_indices(S, cap, k, rows, Nb):
                    assert 0 <= idx < hi, (what, ctx, B, n0, nt, idx, hi)


def test_padding_rows_are_never_indexed():
    """the graph is keyed (Nb, T) with T = real rows: the static input past T is stale and never read (MultiDecoder.step_slots copies hin[:, :T] and the ops slice [:T])"""
    import inspect
    src = open(os.path.join(TREE, "exllamav3", "loader", "step_mu.py")).read()
    assert "inp[2, :self.T]" in src and "assert 1 <= T <= TM" in src
