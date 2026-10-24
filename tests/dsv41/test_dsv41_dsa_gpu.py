"""Existing Kyojin sparse attention (Triton dsa_attn) at the DS4.1 layout against ops.sparse_attention: 64 heads, head_dim 512
(448 + 64 rope), sliding window 128 (ring + the current row) ++ gathered pool entries (-1 padded), per-head attn_sink in the
softmax denominator, fused inverse rope + group-major output (8 o-groups). KERNELS.md section 2.4 marks this op "OK" from code
reading only; this is the device check. Opt-in like test_dsv41_hipk.py (DSV41_DEVICE=cuda, under the GPU lock)."""
import os, sys
import pytest
import torch

DEV = os.environ.get("DSV41_DEVICE", "")
if DEV != "cuda" or not torch.cuda.is_available():
    pytest.skip("device test: DSV41_DEVICE=cuda under the GPU lock", allow_module_level=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exllamav3.dsv41 import ops
from exllamav3.modules.attention_fn.dsa_triton import dsa_attn

H, D, DR, WIN = 64, 512, 64, 128


@pytest.mark.parametrize("p,n_pool,topk", [(300, 150, 100), (1000, 500, 200), (200, 100, 0)])
@pytest.mark.parametrize("derot", [False, True])
def test_dsa_attn_ds41_decode(p, n_pool, topk, derot):
    g = torch.Generator().manual_seed(p)
    rows = 512
    q = (torch.randn(1, H, D, generator=g) * 0.5).half()
    pool = (torch.randn(rows, D, generator=g) * 0.5).half()
    ring_pos = torch.arange(p - WIN + 1, p)                          # window positions before the current token
    ring = (torch.randn(WIN - 1, D, generator=g) * 0.5).half()
    cur = (torch.randn(1, D, generator=g) * 0.5).half()
    sink = torch.randn(H, generator=g)
    KL = 256
    idx = torch.full((1, KL), -1, dtype=torch.int32)
    if topk:
        idx[0, :topk] = torch.randperm(n_pool, generator=g)[:topk].sort().values.int()
    # reference: window rows (oldest first) + current, then the gathered pool rows
    kv_all = torch.cat([ring, cur, pool], 0).unsqueeze(0)
    ridx = torch.cat([torch.arange(WIN), torch.where(idx[0] >= 0, idx[0] + WIN, torch.full_like(idx[0], -1))]).int().view(1, 1, -1)
    ref = ops.sparse_attention(q.unsqueeze(0), kv_all, sink, ridx, D ** -0.5)[0]          # [1, H, D]
    inv = 1.0 / (10000.0 ** (torch.arange(0, DR, 2, dtype=torch.float32) / DR))
    kw = {}
    if derot:
        fc = ops.rope_tables(DR, p + 1, 0, 10000.0, 16.0, 32, 1)
        ops.rope_apply_(ref.unsqueeze(0)[..., -DR:], fc[p:p + 1], True)      # [1, 1, H, DR] view, in place
        kw = dict(derot_inv_freq=(-inv).cuda(), groups=8)
    cu = lambda t: t.cuda().contiguous()
    out = dsa_attn(cu(q), cu(pool[:, :D - DR]), cu(pool[:, D - DR:]), torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
                   sinks=cu(sink), ring=cu(ring), kv_chunk=cu(cur), win_len=WIN, win_floor=p - WIN + 1, ring_beg=p - WIN + 1,
                   indices=cu(idx), k_len=KL, pool_len=n_pool, q_pos0=p, compress_rate=2, page_size=512, **kw)
    torch.cuda.synchronize()
    if derot:                                                        # group-major [8, R, 8 * 512]
        ref = ref.view(1, 8, -1).permute(1, 0, 2)
    torch.testing.assert_close(out.cpu().float().view(ref.shape), ref.float(), rtol=2e-2, atol=2e-2)
