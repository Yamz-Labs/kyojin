"""GPU check of the indexer scoring kernels against the torch reference (dense scan and candidate slots)."""
import pytest, torch
from hardware import require
require("ext")
from exllamav3.dsv41 import quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _ref(q, w, k, start, ratio):
    R, nt = q.size(0), k.size(0)
    s = torch.einsum("shd,td->sht", q.float(), k.float()).relu_()
    sc = (s * w.unsqueeze(-1)).sum(1)
    clen = ((start + torch.arange(1, R + 1, device=q.device)) // ratio).unsqueeze(-1)
    return sc.masked_fill(torch.arange(nt, device=q.device) >= clen, -torch.inf)


def _data(R, nt, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(R, 32, 128, generator=g, device="cuda", dtype=torch.half)
    k = torch.randn(nt + 5, 128, generator=g, device="cuda", dtype=torch.half)
    quant.fp4_act_quant_(q, 32); quant.fp4_act_quant_(k, 32)
    w = torch.rand(R, 32, generator=g, device="cuda") * 0.05
    return q, w, k


@pytest.mark.parametrize("R,nt,start,ratio", [(1, 1000, 1999, 2), (1, 70001, 70000, 1), (7, 4099, 4090, 1), (300, 9000, 8800, 1), (64, 2000, 3900, 2)])
def test_dense(R, nt, start, ratio):
    from exllamav3.modules.dsv41_fn import dsv41_idx as I
    q, w, k = _data(R, nt)
    got = I.score_dense(q, w, k, nt, start, ratio)[:, :nt]
    ref = _ref(q, w, k[:nt], start, ratio)
    fin = torch.isfinite(ref)
    assert torch.equal(fin, torch.isfinite(got))
    assert (got[fin] - ref[fin]).abs().max() <= 2e-5 * ref[fin].abs().max()


def test_cand():
    from exllamav3.modules.dsv41_fn import dsv41_idx as I
    R, nt, start, ratio, nsel = 40, 6000, 5900, 1, 50
    q, w, k = _data(R, nt)
    g = torch.Generator(device="cuda").manual_seed(3)
    blk = torch.stack([torch.randperm(nt // 8, generator=g, device="cuda")[:nsel] for _ in range(R)]).int()
    blk[:, -3:] = -1
    got = I.score_cand(q, w, k, blk, start, ratio)
    full = _ref(q, w, k[:nt], start, ratio)
    pos = (blk.long().clamp_min(0) * 8).unsqueeze(-1) + torch.arange(8, device="cuda")
    ref = torch.gather(full, 1, pos.flatten(1)).masked_fill((blk < 0).repeat_interleave(8, 1), -torch.inf)
    fin = torch.isfinite(ref)
    assert torch.equal(fin, torch.isfinite(got))
    assert (got[fin] - ref[fin]).abs().max() <= 2e-5 * ref[fin].abs().max()


def test_pack_roundtrip_and_packed_scores():
    from exllamav3.modules.dsv41_fn import dsv41_idx as I
    q, w, k = _data(5, 3000)
    kq, ks = I.pack_keys(k)
    nt = 3000
    a = I.score_dense(q, w, k, nt, 2990, 1)
    b = I.score_dense(q, w, k, nt, 2990, 1, packed=(kq, ks))
    fin = torch.isfinite(a)
    assert torch.equal(fin, torch.isfinite(b)) and torch.allclose(a[fin], b[fin], rtol=2e-5, atol=1e-5)   # same inputs; only fp32 summation order may differ
    blk = torch.arange(40, device="cuda", dtype=torch.int32).repeat(5, 1).contiguous()
    c, d = I.score_cand(q, w, k, blk, 2990, 1), I.score_cand(q, w, k, blk, 2990, 1, packed=(kq, ks))
    fin = torch.isfinite(c)
    assert torch.equal(fin, torch.isfinite(d)) and torch.allclose(c[fin], d[fin], rtol=2e-5, atol=1e-5)
