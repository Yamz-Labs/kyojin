"""Fused HIP decode router (routing_std_gfx12_bsz1) vs the torch routing path on the Qwen3.6-35B-A3B shape (2048 x 256, top-8):
same experts in the same order, same fp16 weights, rows 1..4, random inputs plus tied logits. Needs a gfx11.5 / gfx12 wave32 GPU.
Run: hiprun.sh python tests/test_router_shapes_hip.py (or pytest)."""
import types

import torch

from exllamav3.modules import block_sparse_mlp_routing as R


def _cfg(gate):
    E = gate.shape[1]
    return types.SimpleNamespace(gate_tensor=gate, gate_tensor_t=None, num_experts=E, num_experts_per_tok=8, per_expert_scale=None,
                                 router_bias=None, router_logits_bsz1=torch.empty((1, E), dtype=torch.half, device="cuda"),
                                 selected_experts_bsz1=torch.empty((1, 8), dtype=torch.long, device="cuda"),
                                 routing_weights_bsz1=torch.empty((1, 8), dtype=torch.half, device="cuda"))


def test_router_2048_256_top8_matches_torch():
    g = torch.Generator(device="cpu").manual_seed(0)
    bad = 0
    for trial in range(200):
        gate = (torch.randn(2048, 256, generator=g) * 0.02).half().cuda().contiguous()
        if trial % 4 == 0:
            gate[:, 1::2] = gate[:, 0::2]                  # tied logits: experts 2i and 2i+1 equal -> lowest index first
        for rows in (1, 2, 4):
            y = (torch.randn(rows, 2048, generator=g)).half().cuda().contiguous()
            cfg = _cfg(gate)
            assert R._hip_router_call_supported(rows, cfg, y, {}), "fused router not eligible"
            s1, w1 = R.routing_std(rows, cfg, y, {})
            s0, w0 = R._routing_std_torch(cfg, y, {})
            if not (torch.equal(s1.cpu(), s0.cpu()) and torch.equal(w1.cpu(), w0.cpu())):
                bad += 1
    assert bad == 0, f"{bad} of 600 calls differ from the torch routing"


if __name__ == "__main__":
    test_router_2048_256_top8_matches_torch()
    print("PASS")
