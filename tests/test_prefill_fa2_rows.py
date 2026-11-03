import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "CUDA only")

from hardware import require
require("ext")
from exllamav3.modules.attention_fn import triton_paged as tp


def _reference(q, k, v, past, scale):
    # fp32 attention, queries aligned to the end of the keys, GQA by repeat
    q_len, n_heads, hd = q.shape
    n_kv = k.shape[1]
    kf = k[:past + q_len].float().repeat_interleave(n_heads // n_kv, dim = 1)
    vf = v[:past + q_len].float().repeat_interleave(n_heads // n_kv, dim = 1)
    s = torch.einsum("qhd,khd->hqk", q.float(), kf) * scale
    mask = torch.arange(past + q_len, device = q.device)[None, :] > (past + torch.arange(q_len, device = q.device))[:, None]
    s = s.masked_fill(mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), vf)


@pytest.mark.parametrize("q_len, past", [(300, 0), (1024, 0), (2500, 3000), (4096, 5000), (777, 2000)])
def test_split_rows_equal_whole_and_close_to_reference(monkeypatch, q_len, past):
    torch.manual_seed(0)
    hd, n_heads, n_kv = 256, 24, 4
    q = torch.randn(1, q_len, n_heads, hd, device = "cuda", dtype = torch.half)
    kd = torch.randn(1, past + q_len + 100, n_kv, hd, device = "cuda", dtype = torch.half)
    vd = torch.randn(1, past + q_len + 100, n_kv, hd, device = "cuda", dtype = torch.half)
    scale = hd ** -0.5

    outs = {}
    for rows in (0, 1024, 512):
        monkeypatch.setattr(tp, "_pf_fa2_rows", rows)
        outs[rows] = tp._prefill_fa2_contig(q, kd, vd, past, n_kv, hd, scale, torch.empty_like(q))

    # every row sees the same keys in the same order whatever the split. Whole 1024-row sub-calls are bit-identical; a short
    # remainder sub-call may take FA2's split-kv schedule (different summation order), so there the check is rounding-level
    if q_len % 1024 == 0:
        assert torch.equal(outs[1024], outs[0])
    assert (outs[1024].float() - outs[0].float()).abs().max().item() < 2e-3
    assert (outs[512].float() - outs[0].float()).abs().max().item() < 2e-3
    ref = _reference(q[0], kd[0], vd[0], past, scale)
    assert (outs[1024][0].float() - ref).abs().max().item() < 5e-3
