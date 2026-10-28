import os
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason = "ROCm only")

from hardware import require
require("ext")
from exllamav3.modules.attention_fn import triton_paged as tp

_SO = os.environ.get("EXL3_PF_CK_SO") or os.path.expanduser("~/.cache/kyojin/ck-fmha/libckpf.so")


def _reference(q, k, v, past, scale, rows):
    # fp32 attention for the selected query rows, queries aligned to the end of the keys, GQA by repeat
    q_len, n_heads, hd = q.shape
    n_kv = k.shape[1]
    out = []
    for r in rows:
        n = past + r + 1
        kf = k[:n].float().repeat_interleave(n_heads // n_kv, dim = 1)
        vf = v[:n].float().repeat_interleave(n_heads // n_kv, dim = 1)
        s = torch.einsum("hd,khd->hk", q[r].float(), kf) * scale
        out.append(torch.einsum("hk,khd->hd", s.softmax(-1), vf))
    return torch.stack(out)


@pytest.mark.skipif(not os.path.isfile(_SO), reason = "libckpf.so not built (tools/strix_halo/ck_fmha/build.sh)")
@pytest.mark.parametrize("hd, n_heads", [(256, 24), (128, 32)])
@pytest.mark.parametrize("q_len, past", [(300, 0), (1024, 0), (2500, 3000), (4096, 5000), (777, 2001), (4096, 0)])
def test_ck_matches_fp32_reference_and_triton(monkeypatch, hd, n_heads, q_len, past):
    monkeypatch.setenv("EXL3_PF_CK_SO", _SO)
    torch.manual_seed(0)
    n_kv = 4
    q = torch.randn(1, q_len, n_heads, hd, device = "cuda", dtype = torch.half)
    kd = (torch.randn(1, past + q_len + 100, n_kv, hd, device = "cuda") * .5).half()
    vd = torch.randn(1, past + q_len + 100, n_kv, hd, device = "cuda", dtype = torch.half)
    scale = hd ** -0.5
    out = torch.zeros_like(q)
    assert tp._prefill_ck_contig(q, kd, vd, past, n_kv, hd, scale, out) is out
    rows = sorted({0, 1, q_len // 2, q_len - 1})
    ref = _reference(q[0], kd[0], vd[0], past, scale, rows)
    assert (out[0, rows].float() - ref).abs().max().item() / ref.abs().max().item() < 2e-3
    # same K/V through the Triton kernel (paged, identity table): both are fp16-noise away from the reference
    page = 256
    pages = (past + q_len + page - 1) // page
    kc = torch.zeros(pages, page, n_kv, hd, device = "cuda", dtype = torch.half)
    vc = torch.zeros_like(kc)
    kc.view(-1, n_kv, hd)[:past + q_len] = kd[0, :past + q_len]
    vc.view(-1, n_kv, hd)[:past + q_len] = vd[0, :past + q_len]
    bt = torch.arange(pages, dtype = torch.int32, device = "cuda")[None]
    sl = torch.tensor([past], dtype = torch.int32, device = "cuda")
    tri = tp.paged_attn_triton_prefill(q, None, None, kc, vc, bt, sl, causal = True, softmax_scale = scale,
                                       k_new = kd[:, past:past + q_len].contiguous(), v_new = vd[:, past:past + q_len].contiguous(),
                                       max_kv_len = past)
    assert (out.float() - tri.float()).abs().max().item() < 5e-3
