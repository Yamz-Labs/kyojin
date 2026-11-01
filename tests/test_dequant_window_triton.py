import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason = "ROCm only")

from exllamav3 import ext_fallbacks as fb
from exllamav3.modules.attention_fn.qdeq_window_triton import dequant_window, dequant_window_supported


def _case(kb, vb, past, bonus, bsz, n_kvh = 4, hd = 256, pages = 48, seed = 1):
    d = torch.device("cuda:0")
    g = torch.Generator(device = d).manual_seed(seed)
    G = n_kvh * hd // 32
    pps = -(-(past + bonus) // 256)
    kin = torch.randint(-2**31, 2**31 - 1, (pages, 256, G * kb), dtype = torch.int32, device = d, generator = g)
    vin = torch.randint(-2**31, 2**31 - 1, (pages, 256, G * vb), dtype = torch.int32, device = d, generator = g)
    ksc = (torch.rand(pages, 256, G, device = d, generator = g) * 3 + 0.01).half()
    vsc = (torch.rand(pages, 256, G, device = d, generator = g) * 3 + 0.01).half()
    bt = torch.stack([torch.randperm(pages, device = d, generator = g)[:pps] for _ in range(bsz)]).int().contiguous()
    cs = torch.full((bsz,), past, dtype = torch.int32, device = d)
    alloc = 1 << (bsz * pps - 1).bit_length()
    fill = lambda: torch.full((alloc, 256, n_kvh, hd), 7.0, dtype = torch.half, device = d)
    return kin, ksc, vin, vsc, bt, cs, fill


@pytest.mark.parametrize("kb, vb", [(4, 4), (6, 6), (4, 6), (3, 5), (8, 8), (2, 7)])
@pytest.mark.parametrize("past, bonus, bsz", [(5000, 4096, 1), (3000, 17, 1), (4097, 1, 1), (255, 1, 1), (2600, 700, 3)])
def test_bit_identical_to_torch_fallback(kb, vb, past, bonus, bsz):
    kin, ksc, vin, vsc, bt, cs, fill = _case(kb, vb, past, bonus, bsz)
    k1, v1, k2, v2 = fill(), fill(), fill(), fill()
    assert dequant_window_supported(kin, ksc, k2, vin, vsc, v2, cs, bt, 256, 0.0)
    fb._dequant_cache_paged(kin, ksc, k1, vin, vsc, v1, cs, bt, 256, 0.0, -1, bonus, True)
    dequant_window(kin, ksc, k2, vin, vsc, v2, cs, bt, bonus)
    # rows past cache_seqlens + bonus_len stay untouched (the sentinel 7.0), written rows match bit for bit
    assert torch.equal(k1, k2) and torch.equal(v1, v2)


def test_unsupported_cases_decline():
    kin, ksc, vin, vsc, bt, cs, fill = _case(4, 4, 1000, 8, 1)
    k = fill()
    assert not dequant_window_supported(kin, ksc, k, vin, vsc, k, cs, bt, 256, 0.25)     # compand
    assert not dequant_window_supported(kin, ksc, k, vin, vsc, k, cs, bt, 128, 0.0)      # page size
    assert not dequant_window_supported(kin, ksc, k.float(), vin, vsc, k, cs, bt, 256, 0.0)


def test_public_entry_matches_fallback(monkeypatch):
    from exllamav3.ext import exllamav3_ext as ext
    kin, ksc, vin, vsc, bt, cs, fill = _case(4, 4, 2000, 512, 1)
    k1, v1, k2, v2 = fill(), fill(), fill(), fill()
    monkeypatch.setenv("EXL3_DEQ_WINDOW_TRITON", "0")
    ext.dequant_cache_paged_window(kin, ksc, k1, vin, vsc, v1, cs, bt, 256, 512, 0.0)
    monkeypatch.setenv("EXL3_DEQ_WINDOW_TRITON", "1")
    ext.dequant_cache_paged_window(kin, ksc, k2, vin, vsc, v2, cs, bt, 256, 512, 0.0)
    assert torch.equal(k1, k2) and torch.equal(v1, v2)
