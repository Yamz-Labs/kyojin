"""Engram fused vs Hy4Engram.apply reference (random shapes like the model: hc 4, D 4096? small here). GPU if present else TRITON_INTERPRET=1"""
import os, sys, importlib.util, types
import torch
dev = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1" else "cpu"
L = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "exllamav3", "loader")
sp = importlib.util.spec_from_file_location("engf", os.path.join(L, "engram_fused.py")); EF = importlib.util.module_from_spec(sp); sys.modules["engf"] = EF; sp.loader.exec_module(EF)
def rms(x, eps): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
class E: pass
def ref(e, x, rows, valid):                                  # copy of Hy4Engram.apply (q8_act off)
    T, eps = x.shape[0], e.eps
    r = rms(rows, eps) * e.lookup_norm * valid[:, :, None]
    r = r.reshape(T, -1)
    kv = (r @ e.kv.T).reshape(T, e.hc + 1, e.n_embd) + e.kv_b
    key, value = kv[:, :e.hc], kv[:, e.hc]
    dot = (rms(x, eps) * rms(key, eps) * e.gate).sum(-1) / e.n_embd ** 0.5
    mag = dot.abs().clamp_min(1e-6).sqrt()
    sign = 1.0 - 2.0 * (-dot > 0).float()
    gate = torch.sigmoid(mag * sign)
    signal = rms(value, eps) * e.pre_gamma
    delta = signal[:, None, :] * (gate[:, :, None] + e.hc_scale)
    anyv = (valid.sum(-1) > 0).float()
    return x + delta * anyv[:, None, None]
torch.manual_seed(1)
hc, D, nc, hd = 4, 384, 6, 48
e = E(); e.eps = 1.1920929e-07; e.hc, e.n_embd = hc, D
e.lookup_norm = torch.randn(nc, hd, device=dev); e.kv = torch.randn((hc + 1) * D, nc * hd, device=dev) * 0.05
e.kv_b = torch.randn(hc + 1, D, device=dev); e.gate = torch.randn(hc, D, device=dev); e.hc_scale = torch.randn(hc, D, device=dev); e.pre_gamma = torch.randn(D, device=dev)
bad = 0
for T in (1, 2, 4, 6):
    x = torch.randn(T, hc, D, device=dev); rows = torch.randn(T, nc, hd, device=dev)
    valid = (torch.rand(T, nc, device=dev) > 0.3).float(); valid[0] = 1.0
    if T > 1: valid[1] = 0.0
    a, b = ref(e, x, rows, valid), EF.apply(e, x, rows, valid)
    err = ((a - b).abs().max() / a.abs().max()).item(); print("T", T, "rel err %.2e" % err); bad += err > 1e-4
print("ENGRAM_FUSED_OK" if not bad else "ENGRAM_FUSED_FAIL", dev)
