"""
The row-sliced prefill grid (EXL3_PF_ROW_SLICE: programs launched slice by slice over the query rows) computes every program
exactly as the single wide q-block axis does: the output must be BIT-identical for any slice size, with causal masks, windows,
GQA, a ragged last q block and a past that is not tile-aligned.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU (Triton)")

from hardware import require
require("ext")
from exllamav3.modules.attention_fn import triton_paged as tp

PAGE = 256
device = "cuda:0"


def _run(q, k, v, kc, vc, bt, past, slice_rows, **kw):
    old = tp._pf_row_slice
    tp._pf_row_slice = slice_rows
    try:
        sl = torch.tensor([past], dtype = torch.int32, device = device)
        return tp.paged_attn_triton_prefill(q, k, v, kc, vc, bt, sl, causal = True, num_splits = 1, **kw)
    finally:
        tp._pf_row_slice = old


@pytest.mark.parametrize("hd, n_heads, n_kv, q_len, past, window", [
    (256, 24, 4, 4096, 3000, None),   # Qwen3.8-27B full-attention geometry
    (256, 24, 4, 1000, 0, None),      # ragged last q block, no past
    (256, 24, 4, 2500, 777, None),    # past not tile-aligned
    (128, 32, 8, 1536, 5000, None),
    (256, 16, 2, 2048, 4096, 1024),   # sliding window
])
@pytest.mark.parametrize("slice_rows", [128, 256, 1024])
def test_row_slice_bit_identical(hd, n_heads, n_kv, q_len, past, window, slice_rows):
    torch.manual_seed(1)
    total = past + q_len
    pages = (total + PAGE - 1) // PAGE
    kc = (torch.randn((pages, PAGE, n_kv, hd), device = device) * 0.5).half()
    vc = torch.randn((pages, PAGE, n_kv, hd), device = device).half()
    bt = torch.arange(pages, dtype = torch.int32, device = device)[None]
    q = torch.randn((1, q_len, n_heads, hd), device = device).half()
    k = torch.randn((1, q_len, n_kv, hd), device = device).half()
    v = torch.randn((1, q_len, n_kv, hd), device = device).half()
    kw = dict(softmax_scale = hd ** -0.5, window_size = window, max_kv_len = past)
    kc2, vc2 = kc.clone(), vc.clone()
    ref = _run(q, k, v, kc, vc, bt, past, 0, **kw)
    out = _run(q, k, v, kc2, vc2, bt, past, slice_rows, **kw)
    assert torch.equal(ref, out)
