"""CPU tests of the Hy4 prefill head-state window (tools/hy4/engine.py).

The MTP draft head reads the trunk head states of prompt rows [lo, n-1] only. prefill() used to
build the state of every prompt row and torch.cat them, which costs 2 x n x 3072 x 4 B of peak
memory - 24.6 GiB at a 1M prompt - and OOMs long-context prefill. It now keeps only the rows the
draft head reads.

These tests pin the invariant that makes that safe: the rows the collector keeps are the rows
_prime_draft() reads, with the same values and the same order.

No GPU, no model. Run: PYTHONPATH=. pytest -q tests/test_hy4_prefill_head_states_cpu.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TREE, "tools", "hy4"))

os.environ.setdefault("HY4_PRIME_ROWS", "7")       # small, so the draft fill takes several slices
from engine import PRIME_ROWS, copy_head_state_rows, head_state_window  # noqa: E402

EMB = 8          # fake n_embd


def chunk_plan(total, chunk):
    """the chunk boundaries prefill walks (full chunks of `chunk` rows)"""
    out, s = [], 0
    while s < total:
        e = min(total, s + chunk)
        out.append((s, e))
        s = e
    return out


def collect(plan, n, lo):
    """-> the buffer prefill builds: one row per prompt row in [lo, n-1]"""
    out = torch.full((n - lo, EMB), -1.0)
    for s, e in plan:
        st = torch.arange(s, e, dtype=torch.float32).unsqueeze(1).repeat(1, EMB)
        copy_head_state_rows(out, st, s, e, lo)
    return out


def row(j):
    return torch.full((EMB,), float(j))


def test_window_covers_the_whole_prompt_when_prime_rows_is_zero():
    # prime_rows = 0 means "fill the draft cache for the whole prompt": nothing may be dropped
    for n in (1, 100, 999983):
        assert head_state_window(n, 0) == (0, n)


def test_window_is_the_draft_tail_when_prime_rows_is_set():
    n, prime_rows = 999983, 2048
    lo, count = head_state_window(n, prime_rows)
    assert lo == n - 1 - prime_rows
    assert count == prime_rows + 1                       # rows [lo, n-1] inclusive
    # a prompt shorter than the window keeps everything
    assert head_state_window(100, 2048) == (0, 100)
    assert head_state_window(2049, 2048) == (0, 2049)


def test_collected_rows_are_exactly_the_rows_the_draft_head_reads():
    n, chunk, prime_rows = 1000, 256, 64
    lo, count = head_state_window(n, prime_rows)
    H = collect(chunk_plan(n, chunk), n, lo)
    assert H.shape == (count, EMB)

    # every collected row carries the value of its prompt row
    for j in range(lo, n):
        assert torch.equal(H[j - lo], row(j))

    # the rows _prime_draft() reads are present, in order, with the right values
    read, s = [], lo
    while s < n - 1:
        e = min(n - 1, s + PRIME_ROWS)
        read.extend(range(s, e))
        s = e
    assert read == list(range(lo, n - 1))
    for r in read:
        assert torch.equal(H[r - lo], row(r))


def test_a_chunk_that_starts_before_lo_is_truncated_not_dropped():
    # the chunk containing lo starts before lo: its head is kept from lo on, its prefix is dropped
    n, chunk, prime_rows = 1000, 256, 64
    lo, _ = head_state_window(n, prime_rows)
    plan = chunk_plan(n, chunk)
    s, e = next((s, e) for s, e in plan if e > lo)
    assert s < lo < e
    H = collect(plan, n, lo)
    for j in range(lo, n):
        assert torch.equal(H[j - lo], row(j))


def test_rows_before_lo_are_never_copied():
    n, chunk, prime_rows = 1000, 256, 64
    lo, _ = head_state_window(n, prime_rows)
    out = torch.full((n - lo, EMB), -1.0)
    st = torch.arange(0, 100, dtype=torch.float32).unsqueeze(1).repeat(1, EMB)
    copy_head_state_rows(out, st, 0, 100, lo)
    assert torch.equal(out, torch.full((n - lo, EMB), -1.0))
