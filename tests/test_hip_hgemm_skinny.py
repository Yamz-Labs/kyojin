"""ROCm skinny hgemm regressions for padded row counts.

For m=9..32 the skinny path uses padded row instantiations and a one-column
per-lane kernel. A grid sized for two columns per lane left n=64/128 upper
columns unwritten. These shapes exercise that path directly; a zero/sentinel
output makes an unwritten column fail instead of looking like a valid result.

k=128 is exactly one K_CHUNK tile, so it never enters the kernel's partial-tile
branch. k=130 adds a two-row last tile in a second k-split (only warps 0 and 1
contribute there), so an off-by-one or zero-fill bug in that branch fails here
too.
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


# Mirrors skinny::max_rows() in hgemm.cu: clamp(atoi(EXL3_HIP_SKINNY_MAX_M), 1, 32),
# default 32. Below 32 the padded rows fall back to hipBLASLt, which is correct, so
# the comparisons would pass without ever running the kernel under test.
def _effective_max_rows():
    raw = os.environ.get("EXL3_HIP_SKINNY_MAX_M")
    if not raw:
        return 32
    try:
        v = int(raw)
    except ValueError:
        v = 0
    return min(max(v, 1), 32)


if os.environ.get("EXL3_HIP_SKINNY_GEMM") == "0":
    pytest.skip("skinny hgemm path disabled by EXL3_HIP_SKINNY_GEMM=0", allow_module_level = True)
if _effective_max_rows() < 32:
    pytest.skip("EXL3_HIP_SKINNY_MAX_M routes padded rows to the fallback", allow_module_level = True)


@pytest.mark.parametrize("rows", range(9, 33))
@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("k", [128, 130])
@torch.inference_mode()
def test_skinny_hgemm_padded_rows_write_all_columns(rows, n, k):
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
