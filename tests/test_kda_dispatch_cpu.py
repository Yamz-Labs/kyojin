"""CPU test of the chunk_kda default dispatch (EXL3_KDA_FUSE unset = 3) on an arch whose fused Triton output is not
known-stable: {HIP, non-HIP} x {single device, layer split}. Kernels are stubbed; the test records which path runs.

Non-HIP keeps the previous rule (deterministic fla chain only under KYOJIN_SPLIT_ACTIVE, the Triton fused path otherwise).
HIP with no runnable HIP kernel takes the deterministic chain in both modes. A known-stable arch keeps the fused path."""
import sys, types
import pytest
import torch

import exllamav3.vendor.fla as F


@pytest.fixture
def stubs(monkeypatch):
    calls = []
    monkeypatch.delenv("EXL3_KDA_FUSE", raising = False)
    monkeypatch.delenv("EXL3_KDA_INTRA", raising = False)
    hf = types.ModuleType("exllamav3.vendor.fla.hip.kda_hf_hip")
    hf.kda_hf_ok = lambda *a, **k: False
    hf.chunk_kda_hf = None
    monkeypatch.setitem(sys.modules, "exllamav3.vendor.fla.hip.kda_hf_hip", hf)
    monkeypatch.setattr(F, "_hip_ok", lambda *a, **k: False)

    def intra_pinned(q, k, gk, beta, scale, chunk_size):
        B, T, H, K = q.shape
        return torch.zeros(B, T, H, 64), torch.zeros(B, T, H, 64)

    def fused(q, k, v, gk, beta, Akk, Aqk, scale, initial_state, output_final_state, chunk_size, fuse_o):
        calls.append("triton_fused")
        return None, None, torch.zeros_like(v), None

    def chain_intra(q, k, v, gk, beta, scale, chunk_size, **kw):
        calls.append("chain")
        z = torch.zeros(1)
        return z, z, z, z, z, z

    monkeypatch.setattr(F, "chunk_kda_fwd_intra_pinned", intra_pinned)
    monkeypatch.setattr(F, "chunk_kda_fwd_h_fused", fused)
    monkeypatch.setattr(F, "chunk_kda_fwd_intra", chain_intra)
    monkeypatch.setattr(F, "chunk_gated_delta_rule_fwd_h", lambda **kw: (None, None, None))
    monkeypatch.setattr(F, "chunk_gla_fwd_o_gk", lambda **kw: torch.zeros(1))
    return calls


def _run():
    B, T, H, K = 1, 512, 2, 128
    q = torch.zeros(B, T, H, K, dtype = torch.bfloat16)
    v = torch.zeros(B, T, H, K, dtype = torch.bfloat16)
    g = torch.zeros(B, T, H, K)
    beta = torch.zeros(B, T, H, dtype = torch.bfloat16)
    F.chunk_kda(q, q.clone(), v, g, beta, g_cumsum = True)


@pytest.mark.parametrize("hip, split, stable, expected", [
    (False, False, False, "triton_fused"),   # CUDA / unknown arch, one device: as before this change
    (False, True, False, "chain"),           # CUDA / unknown arch under a split: as before
    (True, False, False, "chain"),           # gfx12, no runnable HIP kernel: deterministic chain
    (True, True, False, "chain"),
    (False, False, True, "triton_fused"),    # known-stable arch keeps the fused path
    (True, True, True, "triton_fused"),
])
def test_default_dispatch(stubs, monkeypatch, hip, split, stable, expected):
    monkeypatch.setattr(torch.version, "hip", "7.0" if hip else None)
    if split:
        monkeypatch.setenv("KYOJIN_SPLIT_ACTIVE", "1")
    else:
        monkeypatch.delenv("KYOJIN_SPLIT_ACTIVE", raising = False)
    monkeypatch.setattr(F, "_fused_output_stable", lambda device = None: stable)
    _run()
    assert stubs == [expected]
