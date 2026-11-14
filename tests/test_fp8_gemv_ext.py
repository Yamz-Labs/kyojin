"""C++ FP8 decode GEMV (fq_gemv_fp8.cu) vs the fp16 dequant reference and the Triton kernel it replaces: 1, 3 and 16 activation rows,
per-row and 128x128 block-grid scales, ragged N. Needs a GPU and an extension built with fq_gemv_fp8."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _ext():
    from exllamav3.ext import exllamav3_ext as ext
    if not hasattr(ext, "fq_gemv_fp8"):
        pytest.skip("extension built without fq_gemv_fp8")
    return ext


@pytest.mark.parametrize("rows", [1, 3, 16])
@pytest.mark.parametrize("grid", [False, True])
def test_fq_gemv_fp8_matches_dequant_and_triton(rows, grid, monkeypatch):
    ext = _ext()
    from exllamav3.modules.quant import fp8 as F
    torch.manual_seed(0)
    N, K = 1000, 4096                                        # ragged N (not a multiple of the 4 columns per block)
    q = torch.randint(0, 0x7E, (N, K), dtype=torch.uint8, device="cuda")
    q[:, ::7] |= 0x80                                        # negative codes too
    block = (128, 128) if grid else (1, 1)
    s = (torch.rand((-(-N // 128), K // 128) if grid else (N,), device="cuda") * 0.01 + 1e-3).float()
    x = torch.randn((rows, K), dtype=torch.half, device="cuda")
    ref = x.float() @ F.dequant(q, s, grid, block).float().t()
    y = torch.empty((rows, N), dtype=torch.float32, device="cuda")
    ext.fq_gemv_fp8(x, q, s, y, s.shape[1] if grid else 1, block[0], block[1], grid)
    rel = float((y - ref).abs().max() / ref.abs().max())
    assert rel < 2e-3, rel
    monkeypatch.setenv("BOLT_FP8_GEMV_EXT", "0")
    monkeypatch.setattr(F, "_EXT", [])
    tri = F.gemv(x, q, s, grid, block)
    assert float((y - tri).abs().max() / tri.abs().max()) < 2e-3
    monkeypatch.setenv("BOLT_FP8_GEMV_EXT", "1")
    monkeypatch.setattr(F, "_EXT", [])
    assert torch.equal(F.gemv(x, q, s, grid, block), y)     # the module entry takes the C++ kernel
