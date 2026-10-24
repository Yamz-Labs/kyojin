import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.generator import block_table_pages
from exllamav3.generator.pagetable import Sequence


def make_seq(n_tokens, n_pages):
    seq = Sequence.__new__(Sequence)
    seq.sequence_ids = list(range(n_tokens))
    seq.block_index_tensor = torch.arange(n_pages, dtype = torch.int32).unsqueeze(0)
    return seq


def test_prefill_table_is_sliced_to_used_pages():
    # a request without max_tokens reserves the whole cache: 2048 pages at -c 524288
    seq = make_seq(8192, 2048)
    bt = seq.prefill_block_table()
    assert bt.shape == (1, block_table_pages(8192 + 1))
    assert bt.is_contiguous()
    assert torch.equal(bt, seq.block_index_tensor[:, :bt.shape[1]])
    assert bt.shape[1] * PAGE_SIZE > 8192          # covers every page the chunks write


def test_prefill_table_never_wider_than_allocation():
    seq = make_seq(8192, 40)                       # prompt + 256 reply tokens: 33 pages, rounded up to 48 by the helper
    assert seq.prefill_block_table() is seq.block_index_tensor
    seq = make_seq(8192, 33)
    assert seq.prefill_block_table() is seq.block_index_tensor


def test_prefill_table_width_is_independent_of_cache_size():
    widths = {make_seq(8192, p).prefill_block_table().shape[1] for p in (1064, 2048, 4096)}
    assert len(widths) == 1


def test_prefill_table_covers_page_boundary_token():
    # len + 1 so the page after an exact multiple of PAGE_SIZE is inside the table
    for n in (PAGE_SIZE * 16 - 1, PAGE_SIZE * 16, PAGE_SIZE * 16 + 1):
        assert make_seq(n, 4096).prefill_block_table().shape[1] * PAGE_SIZE >= n + 1
