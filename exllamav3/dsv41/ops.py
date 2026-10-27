"""
Stateless math of DeepSeek-V4.1 (rope/yarn, RMSNorm, hyper-connection mixing, sparse attention,
candidate-block selection, router). Plain torch, CPU or HIP. Each function documents the reference
line it implements (DeepSeek-V4.1-Flash inference/model.py unless noted).

Numerics contract for the fp32 gate: where a downstream fp8/fp4 fake-quant step can flip on a
last-bit difference, the reduction formulation (einsum signature, sum dims, op order) is kept
identical to the reference so results are bit-identical on the same CPU.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------- rope
def rope_tables(rd: int, seqlen: int, orig_len: int, base: float, factor: float, beta_fast: float,
                beta_slow: float, device=None):
    """Complex rotation table fc [seqlen, rd/2] (complex64): exp(i * pos * inv_freq). orig_len > 0 enables
    YaRN (model.py:368 precompute_freqs_cis). Built with torch.polar so each entry is bit-identical to the
    reference's (cos/sin evaluated separately differ by an ulp, which a downstream fp8/fp4 fake-quant
    can amplify)."""
    inv = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    if orig_len > 0:
        def corrected(rot):
            return rd * math.log(orig_len / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected(beta_fast)), 0)
        high = min(math.ceil(corrected(beta_slow)), rd - 1)
        ramp = ((torch.arange(rd // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        inv = inv / factor * (1 - smooth) + inv * smooth
    ang = torch.outer(torch.arange(seqlen), inv)
    return torch.polar(torch.ones_like(ang), ang).to(device)


def rope_apply_(x: torch.Tensor, fc: torch.Tensor, inverse: bool = False):
    """Rotate adjacent element pairs of x's LAST dim in place (pair = complex number). x: [b, s, d] or
    [b, s, h, d]; fc: [s, d/2] complex rows for the s positions. inverse=True conjugates (model.py:392)."""
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        fc = fc.conj()
    fc = fc.view(1, z.size(1), z.size(-1)) if x.ndim == 3 else fc.view(1, z.size(1), 1, z.size(-1))
    x.copy_(torch.view_as_real(z * fc).flatten(-2))
    return x


# --------------------------------------------------------------------------------------- norm
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (w * xf).to(dt)


# --------------------------------------------------------------------------------------- hyper-connections
def hc_split_sinkhorn(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int,
                      eps: float):
    """kernel.py hc_split_sinkhorn_kernel: mixes [..., (2+hc)*hc] -> pre[...,hc] post[...,hc] comb[...,hc,hc]."""
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int,
             iters: int, hc_eps: float, norm_eps: float):
    """model.py:950. x [b,s,hc,d] -> pre, post, comb. Statistic is over the whole flattened hc*d stream."""
    xf = x.flatten(2).float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, fn) * r
    return hc_split_sinkhorn(mixes, scale, base, hc, iters, hc_eps)


def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """[b,s,hc,d] x [b,s,hc] -> [b,s,d] (model.py:959)."""
    return torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """y[c] = post[c] * x + sum_j comb[j,c] * residual[j] (model.py:964)."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)


# --------------------------------------------------------------------------------------- attention
_STEP_MEMO = {}


def step_memo(key, make):
    """Per-step host-built device tensors (window indices, ring slots) are identical for every layer of a step:
    build once, keep only the latest key per kind (a rebuild is a host->device copy = a sync)."""
    k = key[0]
    hit = _STEP_MEMO.get(k)
    if hit is not None and hit[0] == key:
        return hit[1]
    v = make()
    _STEP_MEMO[k] = (key, v)
    return v


_TABLE_MEMO = {}


def table_memo(key, make):
    """Same, for tensors that repeat with a small period (decode: ring phase): all kept, built once each."""
    v = _TABLE_MEMO.get(key)
    if v is None:
        v = _TABLE_MEMO[key] = make()
    return v


def window_idxs(win: int, bsz: int, seqlen: int, start_pos: int, device=None) -> torch.Tensor:
    """Slots of the sliding-window ring each query attends to, -1 = empty (model.py:410)."""
    if start_pos > 0:     # decode: depends on the ring phase only once the window is full
        sp = start_pos if start_pos < win else win + start_pos % win
        return table_memo(("widx", win, bsz, seqlen, sp, str(device)),
                          lambda: _window_idxs(win, bsz, seqlen, sp, device))
    return step_memo(("widx", win, bsz, seqlen, start_pos, str(device)),
                     lambda: _window_idxs(win, bsz, seqlen, start_pos, device))


def _window_idxs(win: int, bsz: int, seqlen: int, start_pos: int, device=None) -> torch.Tensor:
    if start_pos == 0:
        dev = device if seqlen >= 64 else None            # prompt chunks: build on the GPU (the CPU tensors take 60+ ms at 2048 rows)
        end = torch.arange(seqlen, device=dev).unsqueeze(1)
        idxs = (end - win + 1).clamp(0) + torch.arange(min(seqlen, win), device=dev)
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % win + 1
        idxs = torch.cat([torch.arange(oldest, win), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous().to(device)


def window_idxs_block(win: int, bsz: int, seqlen: int, start_pos: int, device=None) -> torch.Tensor:
    return step_memo(("widxb", win, bsz, seqlen, start_pos, str(device)),
                     lambda: _window_idxs_block(win, bsz, seqlen, start_pos, device))


def _window_idxs_block(win: int, bsz: int, seqlen: int, start_pos: int, device=None) -> torch.Tensor:
    """Verify block of `seqlen` tokens at start_pos > 0. Indices into cat([ring before the block (win rows),
    the block's keys (seqlen rows)]): query i reads the `win` positions ending at start_pos + i, oldest first,
    left-padded with -1 while fewer than `win` positions exist (same order and width as `window_idxs` decode)."""
    dev = device if seqlen >= 64 else None
    p = start_pos + torch.arange(seqlen, device=dev)
    pos = (p - win + 1).unsqueeze(1) + torch.arange(win, device=dev)
    idxs = torch.where(pos < start_pos, pos % win, win + (pos - start_pos))
    idxs = torch.where(pos >= 0, idxs, -1)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous().to(device)


def sparse_attention(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, idx: torch.Tensor,
                     scale: float, kv2: torch.Tensor = None) -> torch.Tensor:
    """kernel.py sparse_attn_kernel. q [b,s,h,d], kv [b,n,d] (one shared latent: MQA), idx [b,s,k] int32,
    -1 = masked. Softmax over the gathered keys plus a per-head sink logit that only enters the
    denominator. A row with no valid key returns 0. With `kv2` the keys are the rows of cat([kv, kv2]) but the two
    tensors are gathered separately (the concatenation would copy the whole pool every step)."""
    b, s, h, d = q.shape
    valid = idx >= 0
    if kv2 is not None and kv2.size(1) == 0:      # empty pool (fewer tokens than one compression block): nothing to gather
        kv2 = None

    def take(t, i):
        return torch.gather(t.unsqueeze(1).expand(b, s, -1, d), 2, i.clamp_min(0).long().unsqueeze(-1).expand(-1, -1, -1, d))
    if kv2 is not None and kv2.size(1) == 0:
        kv2 = None
    if kv2 is None:
        g = take(kv, idx)
    else:
        n = kv.size(1)
        inw = idx < n
        g = torch.where(inw.unsqueeze(-1), take(kv, torch.where(inw, idx, 0)), take(kv2, torch.where(inw, 0, idx - n)))
    sc = torch.einsum("bshd,bskd->bshk", q.float(), g.float()) * scale
    sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = sc.amax(-1, keepdim=True).clamp_min(-1e30)
    p = torch.exp(sc - mx)
    den = p.sum(-1, keepdim=True) + torch.exp(sink.float().view(1, 1, h, 1) - mx)
    o = torch.einsum("bshk,bskd->bshd", p, g.float()) / den
    return o.to(q.dtype)


def shared_key_attention(q: torch.Tensor, keys: torch.Tensor, sink: torch.Tensor, scale: float) -> torch.Tensor:
    """sparse_attention for the case where every query row sees the same keys (the dspark draft block: no causal mask, one
    key set for all rows): no gather, two batched matmuls. q [b,s,h,d], keys [b,n,d], same softmax with the sink in the
    denominator as `sparse_attention`."""
    b, s, h, d = q.shape
    kf = keys.float()
    sc = torch.matmul(q.float().reshape(b, s * h, d), kf.transpose(1, 2)).view(b, s, h, -1) * scale
    mx = sc.amax(-1, keepdim=True)
    p = torch.exp(sc - mx)
    den = p.sum(-1, keepdim=True) + torch.exp(sink.float().view(1, 1, h, 1) - mx)
    o = torch.matmul(p.view(b, s * h, -1), kf).view(b, s, h, d) / den
    return o.to(q.dtype)


def select_candidate_blocks(logits: torch.Tensor, compress_lens, topk_blocks: int, block: int) -> torch.Tensor:
    """Level one of the two-level top-k (model.py:584): bool mask shaped like logits that keeps the
    `topk_blocks` blocks with the best (max) score; the block holding the newest reachable position is
    pinned. logits has unreachable positions at -inf."""
    width = logits.size(-1)
    sc = F.pad(logits, (0, -width % block), value=-torch.inf).unflatten(-1, (-1, block)).amax(-1)
    nb = sc.size(-1)
    last = (compress_lens - 1) // block
    sc = sc.masked_fill(torch.arange(nb, device=logits.device) == last, torch.inf)
    top = sc.topk(min(topk_blocks, nb), dim=-1)
    keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block, dim=-1)[..., :width]


# --------------------------------------------------------------------------------------- MoE routing
def route(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, topk: int, func: str, norm: bool,
          scale: float, temp: float = 1.0):
    """model.py:811 Gate.forward. x [n,dim] -> (weights [n,k] fp32, indices [n,k]). The bias only
    steers the selection; weights come from the unbiased score."""
    scores = F.linear(x.float(), w.float()) / temp
    if func == "softmax":
        scores = scores.softmax(-1)
    elif func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    idx = (scores + bias).topk(topk, dim=-1)[1]
    wts = scores.gather(1, idx)
    if norm and topk > 1:
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
    return wts * scale, idx


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """model.py:843: up clamped both ways, gate only from above, fp32."""
    gate, up = gate.float(), up.float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up
