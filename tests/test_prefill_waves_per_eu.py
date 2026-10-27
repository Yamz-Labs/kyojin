"""
The waves_per_eu launch hint of the ROCm prefill kernel (EXL3_PF_WAVES_PER_EU, default 2) only changes the register allocation:
the output must be bit-identical to the compiler default (0) for causal / windowed / ragged / GQA geometries.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason = "needs a ROCm GPU (Triton)")

from exllamav3.modules.attention_fn import triton_paged as tp

PAGE = 256
device = "cuda:0"


@pytest.mark.parametrize("hd, n_heads, n_kv, q_len, past, window", [
    (256, 24, 4, 4096, 3000, None),
    (256, 24, 4, 1000, 0, None),
    (256, 24, 4, 2500, 777, None),
    (128, 32, 4, 2048, 5000, None),
    (128, 32, 8, 1536, 5000, None),
    (256, 16, 2, 2048, 4096, 1024),
])
def test_waves_per_eu_bit_identical(hd, n_heads, n_kv, q_len, past, window):
    torch.manual_seed(1)
    pages = (past + q_len + PAGE - 1) // PAGE
    kc = (torch.randn((pages, PAGE, n_kv, hd), device = device) * 0.5).half()
    vc = torch.randn((pages, PAGE, n_kv, hd), device = device).half()
    bt = torch.arange(pages, dtype = torch.int32, device = device)[None]
    q = torch.randn((1, q_len, n_heads, hd), device = device).half()
    k = torch.randn((1, q_len, n_kv, hd), device = device).half()
    v = torch.randn((1, q_len, n_kv, hd), device = device).half()
    outs = []
    for w in (0, 2):
        old = tp._pf_waves_per_eu
        tp._pf_waves_per_eu = w
        try:
            sl = torch.tensor([past], dtype = torch.int32, device = device)
            outs.append(tp.paged_attn_triton_prefill(
                q, k, v, kc.clone(), vc.clone(), bt, sl, causal = True, softmax_scale = hd ** -0.5,
                window_size = window, max_kv_len = past))
        finally:
            tp._pf_waves_per_eu = old
    assert torch.equal(outs[0], outs[1])
