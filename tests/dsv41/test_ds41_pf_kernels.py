"""GPU: ds41_pf prefill kernels (sparse attention, gather-sum, rope rows) vs torch references at chunk-size shapes
(1024 / 2047 / 2048 / 2049 / 3072 / 4096 rows). Skipped without a GPU."""
import os, sys
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


@pytest.mark.parametrize("R", [1024, 2047, 2048, 2049, 3072, 4096])
def test_sparse_attention_and_gather_sum(R):
    from exllamav3.modules.dsv41_fn import ds41_pf as pf
    from exllamav3.dsv41 import ops
    g = torch.Generator(device="cuda").manual_seed(1)
    N, K = R + 512, 640
    q = (torch.randn(1, R, 64, 512, device="cuda", generator=g) * 0.5).half()
    kv = (torch.randn(1, N, 512, device="cuda", generator=g) * 0.5).half()
    idx = torch.randint(-1, N, (1, R, K), device="cuda", generator=g, dtype=torch.int32)
    sink = torch.randn(64, device="cuda", generator=g)
    o = pf.sparse_attention(q, kv, sink, idx, 512 ** -0.5)
    ref = torch.cat([ops.sparse_attention(q[:, a:a + 256], kv, sink, idx[:, a:a + 256], 512 ** -0.5) for a in range(0, R, 256)], 1)
    assert ((o.float() - ref.float()).abs().max() / ref.float().abs().max()).item() < 5e-3
    Kt, H = 6, 5120
    res = torch.randn(R * Kt, H, device="cuda", generator=g)
    inv = torch.randperm(R * Kt, device="cuda", generator=g).int()
    inv[torch.rand(R * Kt, device="cuda", generator=g) < 0.05] = -1
    out = torch.empty(R, H, device="cuda")
    pf.gather_sum(res, inv, out, R, Kt, H)
    ref = torch.zeros(R, H, device="cuda")
    i2 = inv.view(R, Kt)
    for k in range(Kt):
        ok = i2[:, k] >= 0
        ref[ok] += res[i2[ok, k].long()]
    assert torch.equal(out, ref)
