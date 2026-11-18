"""CPU checks for EXL3_MOE_FOLD_FUSED (fq_moe_fold): reduce-order emulation + dispatch with stubbed kernels.
No GPU, no kernel launch: the extension is replaced by stubs. Needs the repo importable (exllamav3 package, built extension present so the
import does not start a build); skipped otherwise. Run: CUDA_VISIBLE_DEVICES= python -m pytest tests/test_moe_fold_fused_cpu.py -q"""
import glob
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not glob.glob(os.path.join(ROOT, "exllamav3_ext*.so")):
    pytest.skip("extension not built in " + ROOT + " (importing exllamav3 would start the build)", allow_module_level=True)
sys.path.insert(0, ROOT)
import torch



def fold_seq(d, rw, H):             # kernel / torch reduce order for a size-S non-fastest dim: ((0+v0)+v1)+v2)+v3, products rounded to fp32 first
    acc = np.zeros(H, dtype=np.float32)
    for s in range(d.shape[0]):
        acc = acc + (d[s, :H] * np.float32(rw[s])).astype(np.float32)
    return acc


def test_fold_order():
    import torch
    rng = np.random.default_rng(3)
    bad = 0
    for S, H, N in ((4, 2880, 2880), (4, 2880, 2944), (4, 4096, 4096)):
        d = (rng.standard_normal((S, N)) * 5).astype(np.float32)
        rw = rng.random(S).astype(np.float16)
        ref = (torch.from_numpy(d).view(1, S, -1)[..., :H] * torch.from_numpy(rw).reshape(1, S, 1).float()).sum(1).numpy().reshape(-1)
        mine = fold_seq(d, rw.astype(np.float32), H)
        bad += int((ref != mine).sum())
        # the other plausible order, pairwise (v0+v2)+(v1+v3), for contrast
        p = (d[:, :H] * rw.astype(np.float32)[:, None]).astype(np.float32)
        pw = (p[0] + p[2]) + (p[1] + p[3])
        print(f"fold S={S} H={H} N={N}: sequential vs torch-CPU {int((ref != mine).sum())} diffs; pairwise vs torch-CPU {int((ref != pw).sum())} diffs")
    # not asserted: torch CPU is not torch CUDA. The CUDA order is sequential (Reduce.cuh thread_reduce_impl, block_height does not split a 4-long dim: values_per_thread 4 < 16 * block_height).
    print("fold order emulation done (CPU torch diffs total", bad, ")")


class _FakeCudaBias:
    """Stand-in for a CUDA half bias on a CPU box."""
    is_cuda = True
    dtype = torch.half
    def __init__(self, n): self.n = n
    def is_contiguous(self): return True
    def numel(self): return self.n


@pytest.fixture(params=["0", "1"])
def flag(request, monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return request.param == "1"

def test_dispatch(flag, monkeypatch):
    from exllamav3.modules import block_sparse_mlp as bs
    monkeypatch.setenv("EXL3_MOE_FOLD_FUSED", "1" if flag else "0")
    S, H, N = 4, 16, 16
    rng = np.random.default_rng(5)
    dd = torch.from_numpy((rng.standard_normal((S, N)) * 4).astype(np.float32))
    fold_calls = []
    def moe_down(qa, sel, dw, db, d, N_, I_): d.copy_(dd)
    def fold(d, rw, out):
        fold_calls.append(1)
        out.copy_(torch.from_numpy(fold_seq(d.numpy(), rw.float().numpy(), out.numel())).reshape(out.shape).to(out.dtype))
    bext = types.SimpleNamespace(fq_q8_scratch=lambda dev, R, K: torch.zeros(R, K // 32 * 36, dtype=torch.uint8),
                                 fq_quant_q8_1_rows=lambda x, qb: None,
                                 fq_mxfp4_moe_gu=lambda *a: None, fq_mxfp4_moe_down=moe_down, fq_moe_fold=fold)
    monkeypatch.setattr(bs, "ext", bext)
    mod = object.__new__(bs.BlockSparseMLP)
    mod.num_experts_per_tok = S; mod.act_limit = 7.0; mod.shared_experts = None
    mod.downs = [types.SimpleNamespace(inner=types.SimpleNamespace(fl=types.SimpleNamespace(out_features=N)))]
    xin = torch.zeros(1, 1, H, dtype=torch.half); h = torch.zeros(1, H, dtype=torch.half)
    rw = torch.from_numpy(rng.random(S).astype(np.float16))
    tbl = (None, None, None, None, torch.zeros(1), None)
    outs = {}
    for od in (None, torch.half, torch.float):
        fold_calls.clear()
        outs[od] = mod._mxfp4_mmvq_forward(xin, h, torch.zeros(S, dtype=torch.long), tbl, 8, H, H, rw.reshape(1, S), {}, od)
        assert len(fold_calls) == (1 if flag else 0), (od, fold_calls)
    ref = (dd.view(1, S, -1)[..., :H] * rw.reshape(1, S, 1).float()).sum(1).view(xin.shape)
    assert torch.equal(outs[None].float(), ref) and torch.equal(outs[torch.half], ref.half()) and outs[torch.half].dtype == torch.half
    assert outs[torch.float].dtype == torch.float32
