"""rows test of the fused K-row / q-rope kernels (T = 1..8): bit-compare with the torch reference chains. GPU if available, else TRITON_INTERPRET=1 on CPU."""
import os, sys, importlib.util
import torch
dev = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1" else "cpu"
L = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "exllamav3", "loader")
def load(n):
    sp = importlib.util.spec_from_file_location("kvt_" + n, os.path.join(L, n + ".py")); m = importlib.util.module_from_spec(sp); sys.modules["kvt_" + n] = m; sp.loader.exec_module(m); return m
AE = load("attn_eager") if False else None
import triton
KV = load("kv_fused")
torch.manual_seed(0)
R, P, H, NOPE, base, eps, CAP = 512, 64, 8, 128, 10000.0, 1e-6, 64
h = P // 2
def build(n):
    inv = 1.0 / base ** (torch.arange(0, P, 2, dtype=torch.float64) / P)
    th = torch.arange(n, dtype=torch.float64)[:, None] * inv[None]
    c, s = th.cos().float(), th.sin().float()
    t = torch.empty(n, 2, h, 2); t[:, 0, :, 0], t[:, 0, :, 1], t[:, 1, :, 0], t[:, 1, :, 1] = c, s, -s, c
    return t
tab = build(CAP).to(dev)
def rope_ref(x, pos):
    g = tab.index_select(0, pos)
    a, b = g[:, 0].reshape(x.shape[0], 1, h, 2), g[:, 1].reshape(x.shape[0], 1, h, 2)
    xv = x.unflatten(-1, (h, 2))
    return (xv[..., 0:1] * a + xv[..., 1:2] * b).flatten(-2)
bad = 0
for T in (1, 2, 4, 6, 8):
    pos = torch.randint(0, CAP, (T,), device=dev)
    qq = torch.randn(T, H, NOPE + P, device=dev)
    got = KV.rope_q(qq, tab, pos, NOPE, P)
    ref = rope_ref(qq[..., NOPE:], pos)
    e1 = (got - ref).abs().max().item()
    c = torch.randn(T, R + P, device=dev); w = torch.randn(R, device=dev)
    for ring in (0, 16):
        kb = torch.zeros(CAP, R + P, device=dev)
        pos2 = torch.arange(40, 40 + T, device=dev)
        KV.write_row(c, w, tab, pos2, kb, R, P, eps, ring=ring)
        x = c[:, :R]; kref = torch.cat([x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w, rope_ref(c[:, R:].reshape(T, 1, P), pos2).reshape(T, P)], -1)
        slot = pos2 % ring if ring else pos2
        e2 = (kb[slot] - kref).abs().max().item()
        print("T", T, "ring", ring, "rope_q err %.2e" % e1, "kv row err %.2e" % e2)
        bad += (e1 > 1e-5) + (e2 > 1e-4)
print("KVROWS_OK" if not bad else "KVROWS_FAIL", dev)
