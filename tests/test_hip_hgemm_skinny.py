"""ROCm skinny hgemm regressions for padded row counts.

For m=9..32 the skinny path uses padded row instantiations and a one-column
per-lane kernel. A grid sized for two columns per lane left n=64/128 upper
columns unwritten. These shapes exercise that path directly; a zero/sentinel
output makes an unwritten column fail instead of looking like a valid result.
"""
import os

import pytest
import torch

from hardware import require
require("ext")
from exllamav3.ext import exllamav3_ext as ext


pytestmark = pytest.mark.skipif(
    not (torch.version.hip and torch.cuda.is_available()),
    reason="requires a ROCm GPU",
)


@pytest.mark.parametrize("rows", range(9, 33))
@pytest.mark.parametrize("n", [64, 128])
@torch.inference_mode()
def test_skinny_hgemm_padded_rows_write_all_columns(rows, n):
    if os.environ.get("EXL3_HIP_SKINNY_GEMM") == "0":
        pytest.skip("skinny hgemm path disabled by EXL3_HIP_SKINNY_GEMM=0")

    k = 128
    torch.manual_seed(0)
    a = torch.randn((rows, k), dtype=torch.float16, device="cuda") * 0.05
    b = torch.randn((k, n), dtype=torch.float16, device="cuda") * 0.05
    # The kernel must overwrite every output element. A sentinel makes an
    # unwritten column (the old grid bug) fail the comparison even when the
    # expected value happens to be near zero.
    c = torch.full((rows, n), 123.0, dtype=torch.float16, device="cuda")

    ext.hgemm(a, b, c)

    assert torch.isfinite(c).all()
    ref = a.float() @ b.float()
    torch.testing.assert_close(c.float(), ref, rtol=0.01, atol=0.02)
