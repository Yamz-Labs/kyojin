"""int8 (per output row scale) routed-expert stacks for the MTP draft head + two Triton kernels (decode, one row).

The draft layer's experts are stored bf16 in the GGUF (3 x 2.25 GiB). A draft only changes the acceptance rate, never the output (the trunk
verifies every token), so the stacks are quantised on load: w ~ q * s, q int8, s fp32 per output row (max |w| / 127). 1.1 GiB per stack.

  gate_up(z [K], sel [S]) -> h [S, N]   h[s] = silu(min(Wg[e] z, lim)) * clamp(Wu[e] z, -lim, lim)          e = sel[s]
  down(h [S, N], sel, w [S]) -> lat [N]  lat = sum_s w[s] * (Wd[e_s] h[s])
Weights are [E, N, K] row-major (ggml ne0 = K fastest), K = N = 1536 here; generic in (E, N, K)."""
import numpy as np
import torch
import triton
import triton.language as tl


class Int8Stack:
    def __init__(self, g, name, dev, chunk_experts=16):
        ty, dims, off = g.infos[name]
        assert ty == 30 and len(dims) == 3, (name, ty, dims)
        K, N, E = dims
        self.E, self.N, self.K = E, N, K
        self.q = torch.empty(E, N, K, dtype=torch.int8, device=dev)
        self.s = torch.empty(E, N, dtype=torch.float32, device=dev)
        per = N * K * 2
        for e0 in range(0, E, chunk_experts):
            n = min(chunk_experts, E - e0)
            raw = torch.from_numpy(np.array(g._pread(off + e0 * per, n * per))).to(dev)
            w = raw.view(torch.bfloat16).float().reshape(n, N, K)
            s = w.abs().amax(-1).clamp_min(1e-12) / 127.0
            self.q[e0:e0 + n] = torch.round(w / s[..., None]).clamp(-127, 127).to(torch.int8)
            self.s[e0:e0 + n] = s


@triton.jit
def _k_gu(wg_ptr, wu_ptr, sg_ptr, su_ptr, sel_ptr, z_ptr, h_ptr, N, K, lim, BN: tl.constexpr, BK: tl.constexpr):
    s = tl.program_id(0)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    rm = rows < N
    e = tl.load(sel_ptr + s).to(tl.int64)
    accg = tl.zeros([BN], dtype=tl.float32)
    accu = tl.zeros([BN], dtype=tl.float32)
    base = (e * N + rows)[:, None] * K
    for k0 in range(0, K, BK):
        cols = k0 + tl.arange(0, BK)
        cm = cols < K
        m2 = rm[:, None] & cm[None, :]
        zv = tl.load(z_ptr + cols, mask=cm, other=0.0)
        wg = tl.load(wg_ptr + base + cols[None, :], mask=m2, other=0).to(tl.float32)
        wu = tl.load(wu_ptr + base + cols[None, :], mask=m2, other=0).to(tl.float32)
        accg += tl.sum(wg * zv[None, :], axis=1)
        accu += tl.sum(wu * zv[None, :], axis=1)
    sg = tl.load(sg_ptr + e * N + rows, mask=rm, other=0.0)
    su = tl.load(su_ptr + e * N + rows, mask=rm, other=0.0)
    g = tl.minimum(accg * sg, lim)
    u = tl.minimum(tl.maximum(accu * su, -lim), lim)
    tl.store(h_ptr + s * N + rows, g * (1.0 / (1.0 + tl.exp(-g))) * u, mask=rm)


@triton.jit
def _k_down(wd_ptr, sd_ptr, sel_ptr, h_ptr, w_ptr, o_ptr, N, K, S: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    rm = rows < N
    out = tl.zeros([BN], dtype=tl.float32)
    for s in tl.static_range(S):
        e = tl.load(sel_ptr + s).to(tl.int64)
        ws = tl.load(w_ptr + s)
        acc = tl.zeros([BN], dtype=tl.float32)
        base = (e * N + rows)[:, None] * K
        for k0 in range(0, K, BK):
            cols = k0 + tl.arange(0, BK)
            cm = cols < K
            m2 = rm[:, None] & cm[None, :]
            hv = tl.load(h_ptr + s * K + cols, mask=cm, other=0.0)
            wd = tl.load(wd_ptr + base + cols[None, :], mask=m2, other=0).to(tl.float32)
            acc += tl.sum(wd * hv[None, :], axis=1)
        out += ws * acc * tl.load(sd_ptr + e * N + rows, mask=rm, other=0.0)
    tl.store(o_ptr + rows, out, mask=rm)


def experts_forward(gate, up, down, z, sel, w, lim):
    """z [K] fp32, sel [S] int64 device, w [S] fp32 -> lat [N] fp32 (K = gate/up K, N = gate/up N = down K, down N = latent)"""
    S = sel.numel()
    h = torch.empty(S, gate.N, dtype=torch.float32, device=z.device)
    _k_gu[(S, triton.cdiv(gate.N, 16))](gate.q, up.q, gate.s, up.s, sel, z, h, gate.N, gate.K, lim, BN=16, BK=128, num_warps=4)
    out = torch.empty(down.N, dtype=torch.float32, device=z.device)
    _k_down[(triton.cdiv(down.N, 8),)](down.q, down.s, sel, h, w, out, down.N, down.K, S=S, BN=8, BK=128, num_warps=4)
    return out
