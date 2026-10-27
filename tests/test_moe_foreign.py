"""MoE-path test: IQ2_XXS routed experts through BlockSparseMLP dispatch.

Builds a tiny 2-expert gateless-relu2 BlockSparseMLP whose ups/downs are
ForeignLinear (raw IQ2_XXS payloads), drives forward() with pre-routed
picks (params["dec_routed"], no router needed), and checks against a manual
numpy-dequant + torch reference. Runs fused (fq_gemm_*) and unfused
(dequant + matmul) and requires both to match.

Standalone script (CUDA venv has no pytest):
    python tests/test_moe_foreign.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGINE)


def _load_pkg(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


iq2_xxs = _load_pkg("fq_iq2_ref", os.path.join(ENGINE, "exllamav3", "foreign_quant", "iq2_xxs.py"))

from exllamav3.modules.foreign_linear import ForeignLinear  # noqa: E402
import exllamav3.modules.foreign_linear as fl_mod  # noqa: E402
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    (PASS if cond else FAIL).append(name)


def make_expert_pack(rng, out_f, in_f):
    assert in_f % 256 == 0
    p = rng.integers(0, 256, size=(out_f, in_f // 256, 66)).astype(np.uint8)
    # Small fixed scale (real Bolt expert scales are ~1e-2; random fp16
    # scales up to 6e4 overflow fp16 matmuls to inf -> inf-inf NaN diffs;
    # and 0.005-scale weights amplify 1-ulp GEMM diffs through relu2 to 4%
    # at the composed output, while 0.002 keeps the composed check tight).
    p[:, :, 0] = np.float16(0.002).tobytes()[0]
    p[:, :, 1] = np.float16(0.002).tobytes()[1]
    return p


def main():
    # Dispatch-equivalence contract is the fp dots': pin all q8 paths off
    # (their numeric contract lives in test_foreign_quant_cuda's q8 gates).
    os.environ["BOLT_FQ_Q8ACT"] = "0"
    os.environ["BOLT_FQ_Q8MOE"] = "0"
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")
    rng = np.random.default_rng(21)
    H, I, E, TOP = 256, 256, 2, 2

    packs_up = [make_expert_pack(rng, I, H) for _ in range(E)]
    packs_dn = [make_expert_pack(rng, H, I) for _ in range(E)]
    # Reference fp16 weights.
    ref_up = [torch.from_numpy(iq2_xxs.dequant_iq2_xxs(p).reshape(I, H)).half().cuda()
              for p in packs_up]
    ref_dn = [torch.from_numpy(iq2_xxs.dequant_iq2_xxs(p).reshape(H, I)).half().cuda()
              for p in packs_dn]

    ups, downs = [], []
    for e in range(E):
        u = ForeignLinear(None, f"moe.exp{e}.up", H, I, "iq2_xxs")
        u.packed = torch.from_numpy(packs_up[e]).cuda()
        u.device = dev
        d = ForeignLinear(None, f"moe.exp{e}.down", I, H, "iq2_xxs",
                          out_dtype=torch.float)
        d.packed = torch.from_numpy(packs_dn[e]).cuda()
        d.device = dev
        ups.append(u)
        downs.append(d)

    moe = BlockSparseMLP(
        None, "moe", hidden_size=H, intermediate_size=I,
        num_experts=E, num_experts_per_tok=TOP,
        activation_fn="relu2", gates=[], ups=ups, downs=downs,
    )
    for m in ups + downs:
        moe.register_submodule(m)
    moe.device = dev

    xr = rng.standard_normal((3, H)).astype(np.float32) * 0.5
    sel = torch.tensor([[0, 1], [1, 0], [1, 1]], dtype=torch.long, device=dev)
    wts = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]], dtype=torch.float16, device=dev)

    def reference(x):
        outs = []
        for r in range(x.shape[0]):
            acc = torch.zeros(H, dtype=torch.float32, device=dev)
            for k in range(TOP):
                e = int(sel[r, k])
                u = torch.matmul(x[r:r + 1].half(), ref_up[e].t())
                a = torch.square(torch.relu(u))
                o = torch.matmul(a.half(), ref_dn[e].t())
                acc += float(wts[r, k]) * o.float().ravel()
            outs.append(acc)
        return torch.stack(outs)

    def tol(name, got, ref):
        d = (got.float() - ref.float()).abs()
        worst = d.max().item()
        wrel = (d / ref.float().abs().clamp_min(1.0)).max().item()
        check(name, worst <= 1e-2 or wrel <= 5e-3, f"max_abs={worst:.2e} wrel={wrel:.2e}")

    x = torch.from_numpy(xr).half().cuda()
    fl_mod._FUSED = False
    y_unfused = moe.forward(x, {"dec_routed": (sel, wts)})
    tol("moe dispatch unfused vs manual", y_unfused, reference(x))
    assert y_unfused.shape == (3, H), y_unfused.shape
    fl_mod._FUSED = True
    y_fused = moe.forward(x, {"dec_routed": (sel, wts)})
    tol("moe dispatch fused vs unfused", y_fused, y_unfused)

    print(f"done: {len(PASS)} pass, {len(FAIL)} fail")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
