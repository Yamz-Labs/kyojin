"""hgemm must launch on the output tensor's device, not the process's ambient device.

A single process can run kernels on two GPUs (layer split). hgemm_gr used the ambient
device for its stream and for skinny's per-device scratch, so a call with tensors on
cuda:1 while the current device was 0 read foreign pointers. These cases run the public
hgemm path both ways and compare with torch.matmul; a shape <= 32 rows exercises the
skinny path (the one with per-device scratch), 33 rows stays on the GEMM fallback.
"""
import pytest
import torch

from hardware import require
require("ext")
from exllamav3.ext import exllamav3_ext as ext


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires at least two GPUs",
)


@pytest.mark.parametrize("rows,k,n", [(16, 128, 128), (9, 128, 64), (33, 130, 128)])
@pytest.mark.parametrize("src,cur", [(1, 0), (0, 1)])
@torch.inference_mode()
def test_hgemm_uses_the_output_device(rows, k, n, src, cur):
    if src >= torch.cuda.device_count() or cur >= torch.cuda.device_count():
        pytest.skip("two-GPU case requires both devices")
    prev = torch.cuda.current_device()
    try:
        torch.cuda.set_device(cur)
        dev = torch.device("cuda", src)
        torch.manual_seed(0)
        a = torch.randn((rows, k), dtype=torch.float16, device=dev) * 0.05
        b = torch.randn((k, n), dtype=torch.float16, device=dev) * 0.05
        # Sentinel: an unwritten output element must fail, not look like a valid product
        c = torch.full((rows, n), 123.0, dtype=torch.float16, device=dev)

        ext.hgemm(a, b, c)

        assert c.device == dev
        assert torch.cuda.current_device() == cur, "hgemm changed the ambient device"
        ref = a.float() @ b.float()
        torch.testing.assert_close(c.float(), ref, rtol=0.01, atol=0.02)
    finally:
        torch.cuda.set_device(prev)
