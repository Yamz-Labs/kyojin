"""exl3_dec_router with E % 64 != 0 (GLM-5.3: E = 288): same selection and weights as the fp32
torch reference of routing_dots (fp16 logits, sigmoid, + fp32 bias, top-k, renormalize, scale)
and as the generic torch path (_routing_nogroup_torch) it replaces. Run: python tests/test_dec_router_ragged_e.py"""
import sys, torch
from exllamav3.ext import exllamav3_ext as ext  # noqa
from exllamav3.modules.quant.exl3 import dec_workspace
from exllamav3.modules.block_sparse_mlp_routing import _routing_nogroup_torch

def ref(y, G, bias, k, scale):
    logits = (y.float() @ G.float()).half().float()
    s = torch.sigmoid(logits)
    ch = s + bias
    top = torch.topk(ch, k, dim=-1)
    w = s.gather(-1, top.indices)
    return top.indices, w / (w.sum(-1, keepdim=True) + 1e-20) * scale, ch

def run(E, H=4096, k=8, scale=2.5, trials=200, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed + E)
    dev = "cuda"
    G = (torch.randn(H, E, device=dev, generator=g) * 0.02).half().contiguous()
    bias = (torch.randn(E, device=dev, generator=g) * 0.05 + 34.0).float()   # GLM-like large offset
    scratch, counters = dec_workspace(torch.device(dev))
    sel = torch.empty(k, dtype=torch.long, device=dev); wts = torch.empty(k, dtype=torch.half, device=dev)
    # generic path buffers (what E = 288 used before)
    scores = torch.empty(1, E, dtype=torch.half, device=dev)
    sel_g = torch.empty(1, k, dtype=torch.long, device=dev); wts_g = torch.empty(1, k, dtype=torch.half, device=dev)
    bias_h = (bias - bias.mean()).half()
    bad = ties = 0; werr = werr_g = 0.0; agree_g = 0
    for t in range(trials):
        y = (torch.randn(1, H, device=dev, generator=g)).half().contiguous()
        ext.exl3_dec_router(y, G, bias, sel, wts, scratch, counters, scale)
        ri, rw, ch = ref(y, G, bias, k, scale)
        # generic path E = 288 took before: routing_dots -> _routing_nogroup_torch (torch ops)
        from types import SimpleNamespace
        cfg = SimpleNamespace(e_score_correction_bias=bias, num_experts_per_tok=k, routed_scaling_factor=scale, num_experts=E)
        sg, wg = _routing_nogroup_torch(cfg, y, {}, torch.sigmoid(torch.matmul(y, G).float()))
        sel_g[:], wts_g[:] = sg, wg
        torch.cuda.synchronize()
        srt = ch[0].sort(descending=True).values
        tie = (srt[k - 1] - srt[k]).item() < 1e-5
        same = set(sel.tolist()) == set(ri[0].tolist())
        if not same:
            if tie: ties += 1
            else: bad += 1
            continue
        o = torch.argsort(sel); ro = torch.argsort(ri[0])
        werr = max(werr, (wts[o].float() - rw[0][ro]).abs().max().item())
        if set(sel_g[0].tolist()) == set(sel.tolist()):
            agree_g += 1
            og = torch.argsort(sel_g[0])
            werr_g = max(werr_g, (wts_g[0][og].float() - wts[o].float()).abs().max().item())
    print(f"E={E:4d}: set mismatches {bad} (+{ties} ties) / {trials}, max |w - ref| {werr:.2e}, "
          f"generic agrees {agree_g}/{trials}, max |w - generic| {werr_g:.2e}")
    return bad == 0 and werr < 2e-3

def test_dec_router_ragged_e():
    assert all([run(E) for E in (64, 256, 288, 320, 1000, 1016)])


if __name__ == "__main__":
    ok = all([run(E) for E in (64, 256, 288, 320, 1000, 1016)])
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
