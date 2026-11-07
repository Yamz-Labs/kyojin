"""CUDA dense GEMM tuner (EXL3_CUDA_GEMM_TUNE, hgemm.cu ltune): tuned results are bit-exact with cublasGemmEx."""
import os
import pytest
import torch

from hardware import require
require("ext")
from exllamav3.ext import exllamav3_ext as ext


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires a CUDA device (the ROCm build has its own tuner, EXL3_DENSE_GEMM_TUNE)",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("M,K,N,pad_a,pad_c", [
    (512, 1024, 2048, 0, 0), (1000, 2048, 1024, 0, 0), (777, 1024, 1536, 64, 0), (1024, 1536, 1024, 0, 32),
])
@torch.inference_mode()
def test_tuned_hgemm_bit_exact(tmp_path, monkeypatch, dtype, M, K, N, pad_a, pad_c):
    monkeypatch.setenv("EXL3_CUDA_GEMM_TUNE_FILE", str(tmp_path / "tune.txt"))
    torch.manual_seed(7)
    a_full = torch.empty((M, K + pad_a), device="cuda", dtype=torch.half).normal_(std=0.5)
    a = a_full[:, :K]
    b = torch.empty((K, N), device="cuda", dtype=torch.half).normal_(std=K ** -0.5)
    c_full = torch.full((M, N + pad_c), 77, device="cuda", dtype=dtype)
    c = c_full[:, :N]
    ref = torch.empty((M, N), device="cuda", dtype=dtype)
    monkeypatch.setenv("EXL3_CUDA_GEMM_TUNE", "0")
    ext.hgemm(a, b, ref)
    monkeypatch.setenv("EXL3_CUDA_GEMM_TUNE", "1")
    for _ in range(2):           # first call tunes (or reads the cache), second call uses the cached winner
        c.fill_(0)
        ext.hgemm(a, b, c)
        torch.cuda.synchronize()
        assert torch.equal(c, ref)
    if pad_c:
        assert bool((c_full[:, N:] == 77).all())
    assert (tmp_path / "tune.txt").exists() or M < 256
