"""CPU checks for the row-invariance plumbing of multi-row verify forwards (issue 28).

The arithmetic itself is proven on the GPU (tools/glm/pair_inv.py); this file pins the pure logic:
router row chunks, the row caps, the per-job block table slice and the generator's table widths.
"""
import os
import torch
import pytest

from exllamav3.util.row_inv import ROW_INV, ROW_INV_MAX_ROWS
from exllamav3.modules import block_sparse_mlp_routing as R
from exllamav3.modules.mla_attn import MLAttention, _rowloop_max


@pytest.fixture
def rowinv():
    old = ROW_INV["on"]
    yield ROW_INV
    ROW_INV["on"] = old


def test_router_chunks_are_2_to_4_and_sum_to_rows():
    for bsz in range(2, 9):
        ch = R._router_row_chunks(bsz)
        assert sum(ch) == bsz
        assert all(2 <= k <= R.ROUTER_ROWS_KERNEL_MAX for k in ch), (bsz, ch)
    assert R._router_row_chunks(6) == [3, 3]
    assert R._router_row_chunks(8) == [4, 4]
    assert R._router_row_chunks(3) == [3]


def test_row_caps_follow_the_flag(rowinv):
    if "EXL3_MLA_DEC_ROWLOOP_MAX" in os.environ:
        pytest.skip("explicit rowloop cap in the environment")
    rowinv["on"] = True
    assert R._router_rows_max() == ROW_INV_MAX_ROWS == 8
    assert _rowloop_max() == 8
    rowinv["on"] = False
    assert R._router_rows_max() == R.DEC_ROUTER_ROWS_MAX == 4
    assert _rowloop_max() == 4


def test_row_table_is_the_solo_width_slice():
    bt = torch.arange(2 * 48, dtype=torch.int32).view(2, 48)
    params = {"dflash_table_pages": [16, 32]}
    host = [300, 5000]
    t0 = MLAttention._row_table(params, bt, 0, host, 3)
    t1 = MLAttention._row_table(params, bt, 1, host, 3)
    assert tuple(t0.shape) == (1, 16) and tuple(t1.shape) == (1, 32)
    assert t0.is_contiguous() and t1.is_contiguous()
    assert torch.equal(t0, bt[0:1, :16]) and torch.equal(t1, bt[1:2, :32])
    assert MLAttention._row_table(params, bt, 1, host, 3) is t1      # cached for the other layers
    # no widths from the generator: the full table width
    assert tuple(MLAttention._row_table({}, bt, 0, host, 3).shape) == (1, 48)
    # a width wider than the staged table is clamped
    assert tuple(MLAttention._row_table({"dflash_table_pages": [64, 64]}, bt, 1, host, 3).shape) == (1, 48)


def test_row_table_refuses_a_table_shorter_than_the_context():
    from exllamav3.constants import PAGE_SIZE
    bt = torch.zeros((2, 48), dtype=torch.int32)
    # job 1 holds 16 * PAGE_SIZE - 2 tokens: with 3 new rows it needs 17 pages, the solo width says 16
    host = [10, 16 * PAGE_SIZE - 2]
    with pytest.raises(AssertionError):
        MLAttention._row_table({"dflash_table_pages": [16, 16]}, bt, 1, host, 3)
    MLAttention._row_table({"dflash_table_pages": [16, 32]}, bt, 1, host, 3)


def test_generator_table_width_covers_every_verify_row():
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.generator.generator import block_table_pages
    nd, dw = 2, 2
    assert block_table_pages(1) == 16 and block_table_pages(16 * PAGE_SIZE) == 16 and block_table_pages(16 * PAGE_SIZE + 1) == 32
    for max_len in range(1, 6 * PAGE_SIZE, 37):
        w = block_table_pages(max_len + nd + dw)
        # the longest position a verify forward of this job can touch: max_len - 1 cached + 1 + dw new rows
        assert w * PAGE_SIZE >= max_len + dw + 1 and w % 16 == 0


def test_nogroup_torch_router_breaks_score_ties_by_expert_index():
    from types import SimpleNamespace
    E, K = 16, 4
    cfg = SimpleNamespace(num_experts=E, num_experts_per_tok=K, e_score_correction_bias=None, routed_scaling_factor=1.0)
    y = torch.zeros((3, 8))
    # every row has a tie across the k-th place: experts 3, 5, 9, 12 share the second best score
    row = torch.full((E,), 0.1)
    row[7] = 0.9
    row[[3, 5, 9, 12]] = 0.5
    scores = row.repeat(3, 1)
    sel, w = R._routing_nogroup_torch(cfg, y, {}, scores)
    # best first, then the tied experts by ascending index; the 4th pick is the lowest-index tie
    assert sel.tolist() == [[7, 3, 5, 9]] * 3
    assert torch.equal(w[0], w[1]) and torch.equal(w[0], w[2])
