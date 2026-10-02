#!/usr/bin/env python
"""Diagnose the 10 test_glm_dec2 failures seen on the test box. Measures, does not assert.

    PYTHONPATH=<repo> python tests/diag_glm_dec2_fail.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("EXL3_REPO", os.path.dirname(_HERE)))
import torch  # noqa: E402
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402
sys.path.append(os.path.expanduser("~/.local/lib/python3.12/site-packages"))
sys.path.insert(0, _HERE)
import test_glm_dec2 as T  # noqa: E402

DEV = T.DEV


def stages(x, proj, K, sel, act_limit, mcg, alias=False):
    """Per-stage maxima of the reference: g, u, a (fp16), d fp16 vs d computed at 2^-8 scale."""
    mg = mu = ma = md16 = md32 = 0.0
    nonfin = 0
    interm = proj["down"][0][1].numel()
    a = torch.empty((1, interm), dtype=torch.half, device=DEV)
    for r in range(x.shape[0]):
        xr = x[r:r + 1].half()
        for k in range(sel.shape[1]):
            e = 0 if alias else int(sel[r, k])
            g = T._ref_linear(xr, *proj["gate"][e], K, mcg)
            u = T._ref_linear(xr, *proj["up"][e], K, mcg)
            ext.silu_mul(g, u, a, act_limit)
            d16 = T._ref_linear(a, *proj["down"][e], K, mcg)
            d32 = T._ref_linear(a * (2 ** -8), *proj["down"][e], K, mcg).float() * 2 ** 8
            nonfin += int((~torch.isfinite(d16)).sum())
            mg = max(mg, float(g.float().abs().max())); mu = max(mu, float(u.float().abs().max()))
            ma = max(ma, float(a.float().abs().max()))
            md16 = max(md16, float(d16.float().abs().nan_to_num(posinf=7e4).max()))
            md32 = max(md32, float(d32.abs().max()))
    return dict(g=mg, u=mu, a=ma, d16=md16, d32=md32, d16_nonfinite=nonfin)


def mpw_case(shape, K, mcg, act_limit):
    if shape == "glm":
        hidden, interm, experts, topk = T.GLM_H, T.GLM_I, T.EXPERTS, T.GLM_TOPK
    else:
        hidden, interm, experts, topk = T.QWEN_H, T.QWEN_I, T.EXPERTS, T.QWEN_TOPK
    seed = 1000 + K * 10 + (3 if mcg else 0)
    proj = {
        "gate": T._mpw_projection(hidden, interm, K, seed + 1, mcg, experts),
        "up":   T._mpw_projection(hidden, interm, K, seed + 2, mcg, experts),
        "down": T._mpw_projection(interm, hidden, K, seed + 3, mcg, experts),
    }
    g = torch.Generator(device=DEV).manual_seed(seed)
    scale = 1.0 if act_limit else 0.35
    x = (torch.randn(T.ROWS, hidden, generator=g, device=DEV) * scale).half()
    sel = torch.stack([torch.randperm(experts, generator=g, device=DEV)[:topk]
                       for _ in range(T.ROWS)]).to(torch.long)
    w = torch.rand(T.ROWS, topk, generator=g, device=DEV)
    w = w / w.sum(dim=-1, keepdim=True)
    st = stages(x, proj, K, sel, act_limit, mcg)
    got = T._mpw_run(x, proj, K, sel, w, act_limit, mcg, hidden, interm, experts)
    print(f"mpw {shape} K={K} mcg={mcg} L={act_limit}: {st} got finite={bool(torch.isfinite(got).all())}"
          f" max|got|={float(got.abs().max()):.1f}")


def discriminating(K, mcg):
    hidden, interm, experts, topk = T.GLM_H, T.GLM_I, T.EXPERTS, T.GLM_TOPK
    proj = {
        "gate": T._mpw_projection(hidden, interm, K, 77, mcg, experts),
        "up":   T._mpw_projection(hidden, interm, K, 78, mcg, experts),
        "down": T._mpw_projection(interm, hidden, K, 79, mcg, experts),
    }
    g = torch.Generator(device=DEV).manual_seed(555)
    x0 = torch.randn(16, hidden, generator=g, device=DEV)
    sel = torch.stack([torch.randperm(experts, generator=g, device=DEV)[:topk]
                       for _ in range(16)]).to(torch.long)
    w = torch.rand(16, topk, generator=g, device=DEV)
    w = w / w.sum(dim=-1, keepdim=True)
    for s in (1.0, 0.5, 0.35, 0.25):
        x = (x0 * s).half()
        closed = T._mpw_run(x, proj, K, sel, w, T.ACT_LIMIT, mcg, hidden, interm, experts)
        open_ = T._mpw_run(x, proj, K, sel, w, 0.0, mcg, hidden, interm, experts)
        st = stages(x[:4], proj, K, sel[:4], 0.0, mcg)
        eff = float((closed - open_).abs().max() / open_.abs().max().clamp_min(1e-6))
        print(f"disc K={K} mcg={mcg} scale={s}: closed finite={bool(torch.isfinite(closed).all())}"
              f" open finite={bool(torch.isfinite(open_).all())} (nan {int(open_.isnan().sum())})"
              f" clamp_effect={eff:.3f} ref-stages(4 rows, L=0)={st}")


def pair(mcg):
    from exllamav3.modules.mla_attn import dec_proj_single
    attn, _, _, _ = T._mla()
    q_a = T._Lin(T.HIDDEN, T.Q_LORA, 3, 41, mcg)
    kv_a = T._Lin(T.HIDDEN, T.KV_A, 3, 42, mcg)
    attn.q_a_proj, attn.kv_a_proj_with_mqa = q_a, kv_a
    g = torch.Generator(device=DEV).manual_seed(43)
    x = (torch.randn(1, T.HIDDEN, generator=g, device=DEV) * 0.5).half()
    for forced in (None, "4", "8"):
        if forced:
            os.environ["EXL3_DEC_KTW"] = forced
        else:
            os.environ.pop("EXL3_DEC_KTW", None)
        fused = attn._dec_proj_pair(x, {})
        for name, lin in (("q_a", q_a), ("kv_a", kv_a)):
            s = dec_proj_single(lin, x)
            p = fused[id(lin)]
            d = (p.float() - s.float()).abs()
            print(f"pair mcg={mcg} ktw={forced or 'auto'} {name}: equal={torch.equal(p, s)}"
                  f" ndiff={int((d > 0).sum())}/{d.numel()} maxdiff={float(d.max()):.3e}"
                  f" max|y|={float(s.float().abs().max()):.2f}")
    os.environ.pop("EXL3_DEC_KTW", None)


if __name__ == "__main__":
    with torch.inference_mode():
        for args in [("glm", 2, False, 0.0), ("glm", 3, False, 0.0), ("glm", 2, True, 0.0),
                     ("qwen", 2, True, 0.0), ("glm", 3, True, 0.0), ("qwen", 3, True, 0.0),
                     ("glm", 3, True, 10.0)]:
            mpw_case(*args)
        discriminating(3, True)
        discriminating(3, False)
        pair(False)
        pair(True)
