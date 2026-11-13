"""LinearFP8 (FP8 served as stored on HIP / CUDA): Triton decode GEMV and the wide path equal dequant + matmul; per-row and block-grid scales."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")


def _case(N, K, grid, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 256, (N, K), dtype=torch.uint8, generator=g)
    q[(q & 127) == 127] = 0x38                                              # no NaN codes
    if grid:
        s = torch.rand((-(-N // 128), -(-K // 128)), generator=g) * 2e-3 + 1e-3
    else:
        s = torch.rand(N, generator=g) * 2e-3 + 1e-3
    return q.cuda(), s.cuda()


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("N,K", [(256, 512), (300, 768), (5120, 17408), (1000, 1000)])
@pytest.mark.parametrize("R", [1, 3, 16, 40])
def test_linear_fp8_matches_dequant_matmul(grid, N, K, R):
    from exllamav3.modules.quant.fp8 import LinearFP8, dequant, ok
    q, s = _case(N, K, grid, N + K)
    if not ok(N, K, grid, (128, 128)):
        pytest.skip("tiling")
    lin = LinearFP8(K, N, q, s, grid, (128, 128), None, torch.half)
    x = torch.randn((R, K), device="cuda", dtype=torch.half)
    ref = x.float() @ dequant(q, s, grid, (128, 128)).float().T
    y = lin.forward(x, {}, torch.float).float()
    assert torch.isfinite(y).all()
    err = (y - ref).abs().max().item()
    assert err <= 2e-3 * ref.abs().max().item() + 1e-3, err


def test_e4m3_decode_is_exact_on_every_code():
    from exllamav3.modules.quant.fp8 import gemv
    codes = torch.arange(256, dtype=torch.uint8)
    codes = codes[(codes & 127) != 127]
    n = codes.numel()
    q = torch.zeros((n, 128), dtype=torch.uint8); q[:, 0] = codes                # row i holds code i at column 0
    x = torch.zeros((1, 128), dtype=torch.half); x[0, 0] = 1.0
    y = gemv(x.cuda(), q.cuda(), torch.ones(n, device="cuda"), False, (128, 128))[0].cpu()
    assert torch.equal(y, codes.view(torch.float8_e4m3fn).float())
