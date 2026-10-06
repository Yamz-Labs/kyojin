"""CPU reference of the 8-bit cache format (writer semantics of q_cache_kernels.cuh quant_block_x4<8>, reader of _qc_load_kt/_qc_load_v)."""
import torch
def h32():
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < 32: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 32 ** 0.5
H = h32()
def pack8(x):
    """x (N, token_dim) fp16 -> words (N, token_dim//32*8) int32, scales (N, token_dim//32) fp16. Rotated-domain midpoint grid."""
    N, D = x.shape; G = D // 32
    v = (x.double().view(N, G, 32) @ H)                       # rotate (H symmetric, 1/sqrt32 folded in)
    s = v.abs().amax(-1) + 1e-10
    inv = 1.0 / s
    q = torch.floor(v * inv[..., None] * 128 + 128).clamp(0, 255).to(torch.int64)   # (N,G,32)
    qb = q.view(N, G, 8, 4)
    w = qb[..., 0] | (qb[..., 1] << 8) | (qb[..., 2] << 16) | (qb[..., 3] << 24)    # (N,G,8) as uint32 in int64
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)
    return w.reshape(N, G * 8).contiguous(), s.to(torch.float16).float().half()
def dequant8(words, scales, rotated=False):
    N = words.shape[0]; G = scales.shape[1]
    w = words.view(N, G, 8).to(torch.int64) & 0xFFFFFFFF
    q = torch.stack([(w >> (8 * j)) & 255 for j in range(4)], -1).view(N, G, 32).double()
    v = (q - 127.5) * (scales.double()[..., None] / 128.0)
    if not rotated: v = v @ H
    return v.view(N, G * 32).float()
