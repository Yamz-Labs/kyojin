"""KVWIN kernels (see cache/kvwin.py): ring writer and the fp16 ring partial of the flash-decoding split."""
import torch
import triton
import triton.language as tl
from ...cache.kvwin import KVWinCtx


@triton.jit
def _ov_write_kernel(
    k, v, ov_k, ov_v, slots, seqlens,
    q_len, S, Rc, R,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    HD_PAD: tl.constexpr,
):
    """One program per (batch, row j, kv head). j < S handles the sink row of position j (if it lies in this call),
    j >= S the (j - S)-th of the last min(q_len, Rc) rows. A row that is both a sink and in the tail is written
    twice with the same bytes."""
    b = tl.program_id(0)
    j = tl.program_id(1)
    h = tl.program_id(2)
    L = tl.load(seqlens + b)
    slot = tl.load(slots + b)
    t = tl.where(j < S, j, tl.maximum(q_len - Rc, 0) + (j - S))
    p = L + t
    ok = (t < q_len) & (slot >= 0) & ((j >= S) | (p < S))
    row = tl.where(p < S, p, S + (p - S) % Rc)
    offs = tl.arange(0, HD_PAD)
    m = (offs < head_dim) & ok
    src = ((b * q_len + t) * n_kv_heads + h).to(tl.int64) * head_dim
    dst = ((slot.to(tl.int64) * R + row) * n_kv_heads + h) * head_dim
    tl.store(ov_k + dst + offs, tl.load(k + src + offs, mask = m, other = 0.0), mask = m)
    tl.store(ov_v + dst + offs, tl.load(v + src + offs, mask = m, other = 0.0), mask = m)


def ov_write(ctx: KVWinCtx, k: torch.Tensor, v: torch.Tensor, cache_seqlens: torch.Tensor, q_len: int):
    """Write the K/V rows just appended to the pool (k, v: [bsz, q_len, kvh, hd] fp16, positions cache_seqlens[b]..+q_len)
    into the ring: the sinks that lie in this call and its last Rc rows (earlier rows of a long prefill chunk would be
    overwritten inside the same call)."""
    st, slots = ctx.state, ctx.slots
    cfg = st.cfg
    bsz, _, n_kv_heads, head_dim = k.shape
    k = k.contiguous()
    v = v.contiguous()
    n_rows = cfg.S + min(q_len, cfg.Rc)
    with torch.cuda.device(k.device):
        _ov_write_kernel[(bsz, n_rows, n_kv_heads)](
            k, v, st.ov_k, st.ov_v, slots, cache_seqlens,
            q_len, cfg.S, cfg.Rc, cfg.R,
            n_kv_heads, head_dim, triton.next_power_of_2(head_dim),
            num_warps = 1,
        )


@triton.jit
def _ring_split_kernel(
    q, ov_k, ov_v, slots, cache_seqlens, h32,
    partial_o, partial_ml,
    ns_total,             # runtime: partial stride (quant splits + 1); the ring partial is the last one
    sinks_n, win, Rc, R,  # runtime: S, W, Rc, R
    q_len: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    HD_PAD: tl.constexpr,
    scale: tl.constexpr,
    ROT_V: tl.constexpr,         # quantized V: the partial must live in the H32-rotated domain like the pool partials
    LAYOUT: tl.constexpr,        # partial layout of the pool kernel: 0 = dense (query, head) rows (1.6 Triton), 1 = padded rows
                                 # (legacy Triton), 2 = qdec_hip (rows q-major, log2-domain max, unpadded columns)
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RPP: tl.constexpr = 0,       # LAYOUT 2: rows per program in the partial buffers (RP of qdec_hip)
):
    """fp16 pass over the ring: sinks (positions < S) and the window [q-W+1, q] of every query row, stored as one more
    flash-decoding partial (m, l, unnormalised acc) for the existing combine."""
    pid = tl.program_id(0)
    group_size = n_q_heads // n_kv_heads
    rows = tl.arange(0, BLOCK_ROWS)
    if LAYOUT == 2:
        bh = pid
        row_q = rows // group_size
        row_h_local = rows % group_size
        valid_row = rows < q_len * group_size
    elif LAYOUT == 0:
        h_blocks = tl.cdiv(q_len * group_size, BLOCK_ROWS)
        h_block = pid % h_blocks
        bh = pid // h_blocks
        row_idx = h_block * BLOCK_ROWS + rows
        row_q = row_idx % q_len
        row_h_local = row_idx // q_len
        valid_row = row_idx < q_len * group_size
    else:
        h_blocks = tl.cdiv(group_size, BLOCK_H)
        h_block = pid % h_blocks
        bh = pid // h_blocks
        row_q = rows % BLOCK_M
        row_h_local = h_block * BLOCK_H + (rows // BLOCK_M)
        valid_row = (row_q < q_len) & (row_h_local < group_size)
    batch = bh // n_kv_heads
    kv_head = bh - batch * n_kv_heads
    q_head = kv_head * group_size + row_h_local

    offs_d = tl.arange(0, HD_PAD)
    d_mask = offs_d < head_dim
    q_base = ((batch * q_len + row_q) * n_q_heads + q_head) * head_dim
    q_tile = tl.load(q + q_base[:, None] + offs_d[None, :], mask = valid_row[:, None] & d_mask[None, :], other = 0.0)

    L = tl.load(cache_seqlens + batch)
    slot = tl.load(slots + batch).to(tl.int64)
    q_abs = L + row_q
    last = L + q_len - 1          # newest position in the ring

    m = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_ROWS,), tl.float32)
    acc = tl.zeros((BLOCK_ROWS, HD_PAD), tl.float32)
    for r0 in range(0, R, BLOCK_N):
        offs = r0 + tl.arange(0, BLOCK_N)
        in_ring = offs < R
        # position stored in ring row `offs` (rows >= S are cyclic: the one of the last Rc positions with that residue)
        back = (last - sinks_n - (offs - sinks_n)) % Rc
        back = tl.where(back < 0, back + Rc, back)
        pos = tl.where(offs < sinks_n, offs, last - back)
        base = ((slot * R + offs) * n_kv_heads + kv_head) * head_dim
        k_tile = tl.load(ov_k + base[None, :] + offs_d[:, None], mask = in_ring[None, :] & d_mask[:, None], other = 0.0)
        scores = tl.dot(q_tile, k_tile) * scale
        is_sink = (offs < sinks_n)[None, :]
        pos2 = pos[None, :]
        ok = tl.where(
            is_sink,
            pos2 <= q_abs[:, None],
            (pos2 >= sinks_n) & (pos2 <= q_abs[:, None]) & (pos2 > q_abs[:, None] - win),
        )
        valid = valid_row[:, None] & in_ring[None, :] & ok
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis = 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
        l = l * alpha + tl.sum(p, axis = 1)
        v_tile = tl.load(ov_v + base[:, None] + offs_d[None, :], mask = in_ring[:, None] & d_mask[None, :], other = 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new

    if ROT_V:
        hm = tl.load(h32 + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]).to(tl.float32)
        acc = tl.reshape(tl.dot(tl.reshape(acc, (BLOCK_ROWS * HD_PAD // 32, 32)), hm, input_precision = "ieee"),
                         (BLOCK_ROWS, HD_PAD))
    s_idx = ns_total - 1
    if LAYOUT == 2:
        # qdec_hip partials: RPP rows of head_dim columns, running max in the log2 domain (the sum l is domain-free)
        pb = (pid * ns_total + s_idx) * RPP
        st_rows = rows < RPP
        tl.store(partial_o + (pb + rows[:, None]) * head_dim + offs_d[None, :], acc, mask = st_rows[:, None] & d_mask[None, :])
        tl.store(partial_ml + (pb + rows) * 2, m * 1.4426950408889634, mask = st_rows)
        tl.store(partial_ml + (pb + rows) * 2 + 1, l, mask = st_rows)
    else:
        po_base = (pid * ns_total + s_idx) * BLOCK_ROWS * HD_PAD
        tl.store(partial_o + po_base + rows[:, None] * HD_PAD + offs_d[None, :], acc)
        ml_base = (pid * ns_total + s_idx) * BLOCK_ROWS * 2
        tl.store(partial_ml + ml_base + rows * 2, m)
        tl.store(partial_ml + ml_base + rows * 2 + 1, l)


def ring_split(ctx: KVWinCtx, q, cache_seqlens, h32, partial_o, partial_ml, ns_total, programs,
               q_len, n_q_heads, n_kv_heads, head_dim, hd_pad, scale, rot_v, layout, block_m, block_h, block_rows, rpp = 0):
    cfg = ctx.cfg
    st = ctx.state
    with torch.cuda.device(q.device):
        _ring_split_kernel[(programs,)](
            q, st.ov_k, st.ov_v, ctx.slots, cache_seqlens, h32,
            partial_o, partial_ml,
            ns_total, cfg.S, cfg.W, cfg.Rc, cfg.R,
            q_len, n_q_heads, n_kv_heads, head_dim, hd_pad, float(scale), bool(rot_v), int(layout),
            block_m, block_h, block_rows, 64, int(rpp),
            num_warps = 4, num_stages = 1,
        )


@triton.jit
def _stage_patch_kernel(
    k_new, v_new, kd, vd, ov_k, ov_v, slots, seqlens,
    stride, q_len, S, W, Rc, R,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    HD_PAD: tl.constexpr,
    HAS_CHUNK: tl.constexpr,
):
    """Overwrite rows of the dequantized prefill scratch (kd / vd, batch b's tokens at b * stride + position) with exact
    values: the sinks and the last W positions of the past from the ring (as it was before this chunk) and, with
    HAS_CHUNK, every row of the chunk from the fp16 K/V that was just quantized."""
    b = tl.program_id(0)
    j = tl.program_id(1)
    h = tl.program_id(2)
    L = tl.load(seqlens + b)
    slot = tl.load(slots + b).to(tl.int64)
    offs = tl.arange(0, HD_PAD)
    dmask = offs < head_dim
    is_sink = j < S
    is_win = (j >= S) & (j < S + W)
    t = j - S - W
    pos = tl.where(is_sink, j, tl.where(is_win, L - W + (j - S), L + t))
    from_ring = is_sink | is_win
    ok_ring = from_ring & (pos >= 0) & (pos < L) & (is_sink | (pos >= S)) & (slot >= 0)
    dst = ((b * stride + pos).to(tl.int64) * n_kv_heads + h) * head_dim
    row = tl.where(pos < S, pos, S + (pos - S) % Rc)
    rsrc = ((slot * R + row) * n_kv_heads + h) * head_dim
    m_ring = dmask & ok_ring
    tl.store(kd + dst + offs, tl.load(ov_k + rsrc + offs, mask = m_ring, other = 0.0), mask = m_ring)
    tl.store(vd + dst + offs, tl.load(ov_v + rsrc + offs, mask = m_ring, other = 0.0), mask = m_ring)
    if HAS_CHUNK:
        ok_new = (j >= S + W) & (t < q_len)
        csrc = ((b * q_len + t).to(tl.int64) * n_kv_heads + h) * head_dim
        m_new = dmask & ok_new
        tl.store(kd + dst + offs, tl.load(k_new + csrc + offs, mask = m_new, other = 0.0), mask = m_new)
        tl.store(vd + dst + offs, tl.load(v_new + csrc + offs, mask = m_new, other = 0.0), mask = m_new)


def stage_patch(ctx: KVWinCtx, kd, vd, cache_seqlens, stride: int, q_len: int):
    """kd / vd: [pages, page_size, kvh, hd] dequantized scratch in block-table order, stride = tokens per batch row."""
    cfg, st = ctx.cfg, ctx.state
    bsz = ctx.slots.shape[0]
    n_kv_heads, head_dim = kd.shape[2], kd.shape[3]
    has_chunk = ctx.chunk_k is not None
    ck = ctx.chunk_k.contiguous() if has_chunk else kd
    cv = ctx.chunk_v.contiguous() if has_chunk else vd
    n_rows = cfg.S + cfg.W + (q_len if has_chunk else 0)
    with torch.cuda.device(kd.device):
        _stage_patch_kernel[(bsz, n_rows, n_kv_heads)](
            ck, cv, kd, vd, st.ov_k, st.ov_v, ctx.slots, cache_seqlens,
            stride, q_len, cfg.S, cfg.W, cfg.Rc, cfg.R,
            n_kv_heads, head_dim, triton.next_power_of_2(head_dim), has_chunk,
            num_warps = 1,
        )
