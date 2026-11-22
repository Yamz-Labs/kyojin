"""Inkling 2 Mini fused block-depth-attention mix (Triton, gfx1151; same file runs on CPU with TRITON_INTERPRET=1).

mix(blocks, cur, query) = softmax_s(query . rms(state_s) / sqrt(D)) applied to rms(state_s), states = blocks + [cur]   (rms eps 1e-6, query [Q, D], out [Q, T, D])
torch chain = 10 aten launches per call (stack, rms 5, bmm, div, softmax, bmm), 145 calls per token (48 attn + 48 ffn + 48 message mixes of the loop passes + the output mix).
Fused = ONE launch: grid (T, Q), one program per (token, query) streams the states once with an online softmax (the rows of D = 2304 stay in registers, no [S, T, D] copy), the
optional epilogue is the input norm that follows the mix in the step: out = rms(mixed) * nw[q] (6 more aten launches saved; the program owns the whole row, no cross-program reduction).
`bank [NBMAX, 8, D]` fp32 = the block states stored in place by the step (row stride per block = bank.stride(0)); `nb` = number of valid blocks (python int, constant per application);
the current state is a separate [T, D] tensor. T <= 8 per call. No host sync, no allocation inside a call: graph safe.
"""
import math
import torch
import triton
import triton.language as tl

BLK = 4096


@triton.jit
def _step(x, qv, m, l, acc, eps, inv_sqrt_d, D: tl.constexpr):
    r = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
    xn = x * r
    sc = tl.sum(qv * xn, 0) * inv_sqrt_d
    m_new = tl.maximum(m, sc)
    a = tl.exp(m - m_new)
    p = tl.exp(sc - m_new)
    return m_new, l * a + p, acc * a + p * xn


@triton.jit
def _mix_kernel(BANK, CUR, QRY, NW, OUT, NB, s_bank, s_bank_t, s_cur, s_out_q, s_out_t, eps, eps_n, inv_sqrt_d,
                D: tl.constexpr, BLOCK: tl.constexpr, HAS_NW: tl.constexpr, NW_PER_Q: tl.constexpr):
    t = tl.program_id(0)
    q = tl.program_id(1)
    k = tl.arange(0, BLOCK)
    km = k < D
    qv = tl.load(QRY + q * D + k, mask=km, other=0.0)
    m = -1e30
    l = 0.0
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in range(NB):
        x = tl.load(BANK + s * s_bank + t * s_bank_t + k, mask=km, other=0.0)
        m, l, acc = _step(x, qv, m, l, acc, eps, inv_sqrt_d, D)
    x = tl.load(CUR + t * s_cur + k, mask=km, other=0.0)
    m, l, acc = _step(x, qv, m, l, acc, eps, inv_sqrt_d, D)
    o = acc / l
    if HAS_NW:
        w = tl.load(NW + (q * D if NW_PER_Q else 0) + k, mask=km, other=0.0)
        o = o * tl.rsqrt(tl.sum(o * o, 0) / D + eps_n) * w
    tl.store(OUT + q * s_out_q + t * s_out_t + k, o, mask=km)


def mix_fused(bank, nb, cur, query, out, nw=None, eps=1e-6, eps_n=1e-6):
    """bank [NBMAX, 8, D] fp32 (blocks 0..nb-1 hold [T, D] rows), cur [T, D], query [Q, D], out [Q, T, D] fp32 (contiguous per row);
    nw None | [D] (one norm weight for all queries) | [Q, D] (per query): epilogue out = rms(mixed, eps_n) * nw."""
    T, D = cur.shape
    Q = query.shape[0]
    assert T <= 8 and query.shape[1] == D and bank.shape[2] == D and nb <= bank.shape[0] and cur.stride(1) == 1 and out.stride(2) == 1 and query.stride(1) == 1
    per_q = nw is not None and nw.dim() == 2 and nw.shape[0] > 1
    _mix_kernel[(T, Q)](bank, cur, query, nw if nw is not None else cur, out, nb, bank.stride(0), bank.stride(1), cur.stride(0), out.stride(0), out.stride(1),
                        eps, eps_n, 1.0 / math.sqrt(D), D=D, BLOCK=BLK, HAS_NW=nw is not None, NW_PER_Q=per_q, num_warps=4)
    return out


def mix_ref(blocks, cur, query, eps=1e-6):
    rms = lambda x: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    states = rms(torch.stack(list(blocks) + [cur], 0))
    scores = torch.einsum("qd,std->qts", query, states) / math.sqrt(cur.shape[1])
    return torch.einsum("qts,std->qtd", torch.softmax(scores, -1), states)
