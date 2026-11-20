"""GPU check for EXL3_FQ_MMVQ_MXFP4: random packed MXFP4 experts at small shapes, flag-on output of
BlockSparseMLP._forward_mxfp4_grouped (R == 1) and of the raw kernels vs an fp32 dequant reference, rel L2 < 2e-2.
Skipped without CUDA, the built extension or the fq_mxfp4_moe_* kernels."""
import os
import types

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.ext,
              pytest.mark.skipif(not torch.cuda.is_available() or bool(torch.version.hip), reason="needs CUDA")]

E, S, H, K, I, LIM = 6, 4, 192, 256, 128, 7.0       # hidden 192 padded to K = 256 (as gpt-oss 2880 -> 2944); down: N = 256 rows, K = I = 128


def _packed(rows, k, gen):
    nb = k // 32
    t = torch.randint(0, 256, (E, rows, nb, 17), dtype=torch.uint8, generator=gen)
    t[..., 0] = torch.randint(118, 127, (E, rows, nb), dtype=torch.uint8, generator=gen)      # scales around 2^-9 .. 2^-1
    return t.reshape(E, rows * nb * 17).cuda()


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_mmvq_mxfp4_moe_vs_dequant_reference(monkeypatch):
    from exllamav3.ext import exllamav3_ext as ext
    if not (hasattr(ext, "fq_mxfp4_moe_gu") and hasattr(ext, "fq_mxfp4_moe_down")):
        pytest.skip("extension built without fq_mxfp4_moe_*")
    from exllamav3.foreign_quant import gap_formats as GF
    from exllamav3.foreign_quant import triton_moe_mxfp4 as MX
    from exllamav3.modules import block_sparse_mlp as B

    gen = torch.Generator().manual_seed(0)
    gp, up, dp = _packed(I, K, gen), _packed(I, K, gen), _packed(K, I, gen)
    gb, ub, db = [(torch.randn(E, n, generator=gen) * 0.1).half().cuda() for n in (I, I, K)]
    x = (torch.randn(1, H, generator=gen) * 0.5).half().cuda()
    sel = torch.tensor([[3, 0, 5, 1]], dtype=torch.int64).cuda()
    rw = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).cuda()

    def tbl(t):
        return torch.tensor([t[e].data_ptr() for e in range(E)], dtype=torch.int64).cuda()

    def mod(t, b, n, k):
        return types.SimpleNamespace(inner=types.SimpleNamespace(
            fl=types.SimpleNamespace(packed=t, in_features=k, out_features=n, format="mxfp4"), bias=b))

    gates = [mod(gp[e], gb[e], I, K) for e in range(E)]
    ups = [mod(up[e], ub[e], I, K) for e in range(E)]
    downs = [mod(dp[e], db[e], K, I) for e in range(E)]
    me = types.SimpleNamespace(num_experts_per_tok=S, activation_fn="swiglu_oai", act_limit=LIM, shared_experts=None,
                               gates=gates, ups=ups, downs=downs)
    me._mxfp4_grouped_ok = lambda y: True
    me._mxfp4_mmvq_forward = types.MethodType(B.BlockSparseMLP._mxfp4_mmvq_forward, me)

    # fp32 reference from the engine's own dequant of the same bytes
    def deq(t, n, k):
        return torch.stack([GF.dequant("mxfp4", t[e].reshape(n, -1)).float() for e in range(E)])
    Wg, Wu, Wd = deq(gp, I, K), deq(up, I, K), deq(dp, K, I)
    xp = torch.nn.functional.pad(x, (0, K - H)).float()[0]
    ref_d = torch.empty(S, K, device="cuda")
    ref_a = torch.empty(S, I, device="cuda")
    for s in range(S):
        e = sel[0, s].item()
        g = (Wg[e] @ xp + gb[e].float()).half().float().clamp(max=LIM)
        u = (Wu[e] @ xp + ub[e].float()).half().float().clamp(-LIM, LIM)
        a = ((u + 1.0) * g * torch.sigmoid(1.702 * g)).half()
        ref_a[s] = a.float()
        ref_d[s] = Wd[e] @ a.float() + db[e].float()
    ref_out = (ref_d[:, :H] * rw.reshape(S, 1)).sum(0, keepdim=True)

    # raw kernels
    ssel = sel.reshape(-1).to(torch.int32).contiguous()
    qx = ext.fq_q8_scratch(0, 1, K)
    ext.fq_quant_q8_1_rows(torch.nn.functional.pad(x, (0, K - H)).contiguous(), qx)
    a = torch.empty(S, I, dtype=torch.float16, device="cuda")
    ext.fq_mxfp4_moe_gu(qx, ssel, tbl(gp), tbl(gb), tbl(up), tbl(ub), a, I, K, LIM)
    assert _rel(a, ref_a) < 2e-2, _rel(a, ref_a)
    qa = ext.fq_q8_scratch(0, S, I)
    ext.fq_quant_q8_1_rows(a, qa)
    d = torch.empty(S, K, dtype=torch.float32, device="cuda")
    ext.fq_mxfp4_moe_down(qa, ssel, tbl(dp), tbl(db), d, K, I)
    assert _rel(d, ref_d) < 2e-2, _rel(d, ref_d)

    # flag on: whole decode branch of _forward_mxfp4_grouped
    monkeypatch.setenv("EXL3_FQ_MMVQ_MXFP4", "1")
    fires = MX.MMVQ_FIRES
    out = B.BlockSparseMLP._forward_mxfp4_grouped(me, x, x, sel, rw, {}, None, None)
    assert MX.MMVQ_FIRES == fires + 1, "mmvq branch did not fire"
    assert out.shape == (1, H)
    assert _rel(out, ref_out) < 2e-2, _rel(out, ref_out)
