"""CPU checks for the 9..32 row verify caps on ROCm: how the union MoE splits a step into exact <= 8 row calls."""
import pytest
import torch

from exllamav3.modules import block_sparse_mlp as B
from exllamav3.util.row_inv import ROW_INV_MAX_ROWS


def _x(jobs, w):
    return torch.empty((jobs, w, 4), dtype=torch.half, device="meta")


def test_caps_are_arch_keyed():
    assert ROW_INV_MAX_ROWS == (32 if torch.version.hip else 8)
    assert B.UNION_ROWS == (32 if torch.version.hip else B.MAX_BSZN)


@pytest.mark.parametrize("jobs", range(1, 33))
@pytest.mark.parametrize("w", range(1, 9))
def test_slices_cover_rows_exactly(jobs, w):
    rows = jobs * w
    if rows > 32:
        pytest.skip("above the union cap")
    sl = B._union_slices(_x(jobs, w), rows)
    assert sl[0][0] == 0 and sl[-1][1] == rows
    assert all(a[1] == b[0] for a, b in zip(sl, sl[1:]))                    # contiguous, in order, no overlap
    for lo, hi in sl:
        assert hi - lo <= B.MAX_BSZN                                        # each call is the exact <= 8 row kernel
        if rows > 1:
            assert hi - lo >= 2 or rows <= B.MAX_BSZN                       # a lone row would take the batch-1 kernel
        if rows > B.MAX_BSZN and w > 1:
            assert lo % w == 0 and hi % w == 0                              # whole jobs stay in one slice


def test_slice_plans_of_the_target_steps():
    assert B._union_slices(_x(4, 3), 12) == [(0, 6), (6, 12)]                # N=4 depth 2
    assert B._union_slices(_x(4, 4), 16) == [(0, 8), (8, 16)]                # N=4 depth 3
    assert B._union_slices(_x(4, 2), 8) == [(0, 8)]                          # N=4 depth 1: one call, unchanged
    assert B._union_slices(_x(3, 3), 9) == [(0, 6), (6, 9)]
    assert B._union_slices(_x(12, 1), 12) == [(0, 6), (6, 12)]               # plain multi-session step, balanced
    assert B._union_slices(torch.empty((1, 11, 4), device="meta"), 11) == [(0, 6), (6, 11)]   # one long job
    assert B._union_slices(torch.empty((20, 4), device="meta"), 20) == [(0, 7), (7, 14), (14, 20)]
