"""Grouped IQ dequant of the prefill MoE (fq_moe_prefill_iq._dq_iq) vs the per-expert references (gap_formats.dequant for IQ2_S /
IQ3_XXS, ext.fq_dequant_iq4_xs for IQ4_XS) on real experts of an unsloth UD-IQ3_XXS MoE GGUF (default: Qwen3.6-35B-A3B). Same fp16
values expected (max abs diff ~0, both decode in fp32 then round). Usage: python tests/test_moe_iq_prefill.py [GGUF]"""
import sys

import torch
import triton

sys.path.insert(0, "tests")
from test_moe_iq_q8 import index, GGUF as G0  # noqa: E402

import exllamav3_ext as ext  # noqa: E402
from exllamav3.foreign_quant import gap_formats  # noqa: E402
from exllamav3.foreign_quant.triton_iq import _tabs  # noqa: E402
from exllamav3.modules.fq_moe_prefill_iq import _dq_iq, _kv, FMT_ID  # noqa: E402

import numpy as np  # noqa: E402

GGUF = sys.argv[1] if len(sys.argv) > 1 else G0
TYPES = {18: ("iq3_xxs", 98), 22: ("iq2_s", 82), 23: ("iq4_xs", 136)}


def check(f, infos, base, name, ne=4):
    dims, t, off = infos[name]
    fmt, bb = TYPES[t]
    Kin, N = dims[0], dims[1]
    per = N * (Kin // 256) * bb
    f.seek(base + off)
    raw = torch.from_numpy(np.frombuffer(f.read(per * ne), dtype=np.uint8).reshape(ne, N, -1).copy()).cuda()
    ptrs = torch.tensor([raw[e].data_ptr() for e in range(ne)], dtype=torch.int64, device="cuda")
    out = torch.empty(ne, N, Kin, dtype=torch.half, device="cuda")
    t3x, _, tks, _, t2s, _ = _tabs(raw.device)
    _dq_iq[(triton.cdiv(N, 32), Kin // 256, ne)](ptrs, out, 0, N, Kin, t3x, tks, t2s, _kv(raw.device), FMT=FMT_ID[fmt], BN=32, num_warps=4)
    worst = 0.0
    for e in range(ne):
        if fmt == "iq4_xs":
            ref = torch.empty(N, Kin, dtype=torch.half, device="cuda")
            ext.fq_dequant_iq4_xs(raw[e].contiguous(), ref)
        else:
            ref = gap_formats.dequant(fmt, raw[e]).reshape(N, Kin)
        worst = max(worst, float((out[e].float() - ref.float()).abs().max() / ref.float().abs().max()))
    print(f"{name} {fmt} [{ne}, {N}, {Kin}] max rel diff {worst:.2e}")
    return worst


def test_moe_iq_prefill_dequant():
    f, infos, base = index(GGUF)
    names = []
    for want in (22, 18, 23):
        for kind in ("ffn_gate_exps", "ffn_down_exps"):
            n = next((f"blk.{i}.{kind}.weight" for i in range(200) if infos.get(f"blk.{i}.{kind}.weight", (0, -1))[1] == want), None)
            if n and n not in names:
                names.append(n)
                break
    worst = max(check(f, infos, base, n) for n in names)
    assert worst < 1e-3, f"dequant differs: {worst:.2e}"


if __name__ == "__main__":
    test_moe_iq_prefill_dequant()
    print("PASS")
