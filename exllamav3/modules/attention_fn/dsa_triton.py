"""
Production DSA (DeepSeek Sparse Attention) Triton kernels.

Constexpr toggles on the core sparse attention kernel:

  HAS_WINDOW   include a sliding-window phase over raw K=V rows before the gathered phase
               (V4: window 128 from the ring; V3.2-on-MLA: off)
  HAS_SINKS    per-head learnable sink logit folded into the softmax state init (gpt-oss
               style; V4: on)
  DENSE_POOL   attend to ALL pool entries with a per-query causal entry bound instead of a
               gathered index list (V4 HCA; also the short-context fast path for CSA when
               the pool is smaller than index_topk)
  DEROTATE     eq. 26 output de-rotation fused into the epilogue: the output's trailing
               D_r slice is GPT-J-rotated at the query's absolute position using a caller
               frequency table (pass the NEGATED table for de-rotation). One program = one
               query row, so this is a single theta vector broadcast over heads
  HPG          heads-per-group > 0 stores the output group-major, (H / HPG, R, HPG * D),
               so each group's wo_a GEMM reads a contiguous slice with no host permute
               (V4 grouped output projection); 0 = token-major (R, H, D)

The score is always two-dot: D_c-wide nope part + D_r-wide rope part, V IS K. For V4 the
cache rows are (448 nope | 64 rotated rope) split across two tensors exactly like the MLA
ckv/kpe layout; for V3.2-on-MLA they are (512 latent | 64 rope) and the same kernel serves
with DEROTATE=0, HAS_WINDOW=0.

The lightning-indexer scoring kernel is shared as-is between raw-token (V3.2) and pooled
(V4 CSA) keys -- only the key tensor differs.
"""

import os
import torch
from ...util.tensor import g_tensor_cache

# EXL3_DSA_DEBUG_BOUNDS=1: compile the JIT DSA kernels with device-side bounds asserts on
# every block-table page read and gathered pool index (names the kernel and traps at the
# bad index instead of faulting downstream). Triton only emits device_assert when kernels
# compile in debug mode, so force TRITON_DEBUG before any compilation
dsa_debug_bounds = os.environ.get("EXL3_DSA_DEBUG_BOUNDS", "0") != "0"

# gates only. dsa_glue_expand additionally runs the eager chain and raises on any
# bitwise mismatch. Read once (it sits on the per-layer decode path).
dsa_glue_check = os.environ.get("EXL3_DSA_GLUE_CHECK", "0") == "1"

import triton
import triton.language as tl

from .triton_paged import _rot_h32, _qc_load_v, _get_h32

@triton.jit(do_not_specialize = [
    "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0", "R",
    "win_floor", "ring_beg",
], debug = dsa_debug_bounds)
def _dsa_attn_kernel(
    q,                   # (R, H, D_c + D_r) fp16 token-major, rope slice pre-rotated
    ring,                # (ring_rows, D_c + D_r) fp16: rows at abs < q_pos0, linearly
                         # addressed as abs - ring_beg (HAS_WINDOW)
    kv_chunk,            # (R, D_c + D_r) fp16: this chunk's K = V rows, abs >= q_pos0
    pool_c,              # paged pool, nope part: (pages * page_size, D_c) fp16
    pool_r,              # paged pool, rope part: (pages * page_size, D_r) fp16
    block_table,         # (R, num_pages_per_row) int32
    indices,             # (R, K_pad) int32 pool-entry indices, -1 padded (not DENSE_POOL)
    sinks,               # (H,) fp32 (HAS_SINKS)
    derot_inv_freq,      # (D_r // 2,) fp32 epilogue frequency table (DEROTATE)
    out,                 # (R, H, D_c + D_r) fp16, or (H / HPG, R, HPG * D) when HPG > 0
    k_len,               # runtime: valid gathered entries per row
    win_len,             # runtime: sliding window width
    pool_len,            # runtime: pool entry count (DENSE_POOL bound base)
    num_pages_per_row,   # runtime
    q_pos0,              # runtime: absolute position of query row 0 (window/DENSE_POOL
                         # bounds and DEROTATE query position base)
    R,                   # runtime: query row count (HPG group stride)
    win_floor,           # runtime: lowest absolute position visible to the window
    ring_beg,            # runtime: absolute position of ring row 0
    pool_s,              # (pages * page_size, D_c // 32) fp16 group scales (QC > 0)
    h32,                 # (32, 32) fp16 H32 / sqrt(32) (QC > 0)
    H: tl.constexpr,
    page_size: tl.constexpr,
    D_c: tl.constexpr,
    D_c_pad: tl.constexpr,
    D_r: tl.constexpr,
    K_pad: tl.constexpr,
    compress_rate: tl.constexpr,   # DENSE_POOL causal bound: entry w < (pos + 1) // m
    scale: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    DENSE_POOL: tl.constexpr,
    DEROTATE: tl.constexpr,
    HPG: tl.constexpr,             # heads per output group; 0 = token-major store
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,         # window tile; smaller than BLOCK_N (two-source smem)
    DEBUG_BOUNDS: tl.constexpr = 0,
    DEBUG_PAGES: tl.constexpr = 0,
    NC_BLOCK: tl.constexpr = 0,    # DSpark draft mode: every row sees the SAME range
                                   # [win_floor, q_pos0 + R) (window history ++ whole
                                   # chunk, non-causal); history rows are PAGED, read
                                   # via block_table at absolute-position pages
    NC_CHUNK: tl.constexpr = 0,    # non-causal image chunk (DeepSeek-V4 vision): every
                                   # row sees the WHOLE chunk plus its own causal window
                                   # of NC_HIST history rows from the ring; win_len is
                                   # the scan length (NC_HIST - 1 + R), top = chunk end
    NC_HIST: tl.constexpr = 0,
    Q_SPLIT: tl.constexpr = 0,     # GLM-5.2 absorbed queries: q is the HEAD-MAJOR latent
                                   # (H, R, D_c) and the (unused, windowless) ring slot
                                   # carries the token-major rope queries (R, H, D_r)
    OUT_LATENT: tl.constexpr = 0,  # store only the latent half, head-major (H, R, D_c),
                                   # the exact mla_unfold input; the rope half of the
                                   # weighted sum is never accumulated
    QC: tl.constexpr = 0,          # pool_c is the packed quantized pool (QC bits per
                                   # value, 32-value groups, H32-rotated domain)
):
    """One program per (query row, head block); heads are the MMA M dim. Consecutive
    programs cover one query's head blocks so gathers stay L2-resident. Two KV phases:
    the sliding ring (dense, short) and the pool (gathered via per-query index list, or
    dense with a causal entry bound when DENSE_POOL)."""
    pid = tl.program_id(0)
    h_blocks = tl.cdiv(H, BLOCK_H)
    row = pid // h_blocks
    h_block = pid % h_blocks

    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H
    offs_c = tl.arange(0, D_c_pad)
    valid_c = offs_c < D_c
    offs_r = tl.arange(0, D_r if D_r > 0 else 1)
    D = D_c + D_r

    if Q_SPLIT:
        qc = tl.load(q + (offs_h * R + row)[:, None] * D_c + offs_c[None, :],
                     mask = valid_h[:, None] & valid_c[None, :], other = 0.0)
        if D_r > 0:
            qr = tl.load(ring + (row * H + offs_h)[:, None] * D_r + offs_r[None, :],
                         mask = valid_h[:, None], other = 0.0)
    else:
        q_base = q + (row * H + offs_h)[:, None] * D
        qc = tl.load(q_base + offs_c[None, :], mask = valid_h[:, None] & valid_c[None, :], other = 0.0)
        if D_r > 0:
            qr = tl.load(q_base + D_c + offs_r[None, :], mask = valid_h[:, None], other = 0.0)

    if QC > 0:
        # Packed pool values live in the H32-rotated domain (orthonormal, block-diagonal
        # per 32 values): rotate q once so every score dot is exact, rotate the fp16
        # window tiles into the same domain, and rotate the accumulated output back in
        # the epilogue (one-shot kernel) or the combine (split kernels)
        qc = _rot_h32(qc, h32, BLOCK_H, D_c_pad)

    if HAS_SINKS:
        sink = tl.load(sinks + offs_h, mask = valid_h, other = -float("inf"))
        m_state = sink
        l = tl.full((BLOCK_H,), 1.0, tl.float32)
    else:
        m_state = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l = tl.zeros((BLOCK_H,), tl.float32)
    acc_c = tl.zeros((BLOCK_H, D_c_pad), tl.float32)
    acc_r = tl.zeros((BLOCK_H, D_r if D_r > 0 else 1), tl.float32)

    # Phase 1: sliding-window rows, addressed by absolute position: query row sees
    # positions [q_abs - win_len + 1, q_abs] clipped to win_floor; rows at abs >= q_pos0
    # come from this chunk's kv rows, older rows from the ring at abs - ring_beg
    if HAS_WINDOW:
        q_abs = q_pos0 + row
        if NC_BLOCK or NC_CHUNK:
            top = q_pos0 + R - 1
        else:
            top = q_abs
        for n0 in tl.range(0, win_len, BLOCK_W, num_stages = 1):
            offs_j = n0 + tl.arange(0, BLOCK_W)
            abs_pos = top - offs_j
            in_range = (offs_j < win_len) & (abs_pos >= win_floor)
            if NC_CHUNK:
                # chunk rows: all visible; history rows: this row's own window
                in_range = in_range & ((abs_pos >= q_pos0) | (abs_pos > q_abs - NC_HIST))
            mc = in_range & (abs_pos >= q_pos0)
            mr = in_range & (abs_pos < q_pos0)
            idx_c = tl.where(mc, abs_pos - q_pos0, 0)
            if NC_BLOCK:
                ap = tl.where(mr, abs_pos, 0)
                w_phys = tl.load(block_table + row * num_pages_per_row + ap // page_size,
                                 mask = mr, other = 0)
                idx_r = w_phys * page_size + ap % page_size
            else:
                idx_r = tl.where(mr, abs_pos - ring_beg, 0)
            vc = tl.load(kv_chunk + idx_c[:, None] * D + offs_c[None, :],
                         mask = mc[:, None] & valid_c[None, :], other = 0.0) \
               + tl.load(ring + idx_r[:, None] * D + offs_c[None, :],
                         mask = mr[:, None] & valid_c[None, :], other = 0.0)
            if QC > 0:
                vc = _rot_h32(vc, h32, BLOCK_W, D_c_pad)
            scores = tl.dot(qc, tl.trans(vc))
            if D_r > 0:
                vr = tl.load(kv_chunk + idx_c[:, None] * D + D_c + offs_r[None, :],
                             mask = mc[:, None], other = 0.0) \
                   + tl.load(ring + idx_r[:, None] * D + D_c + offs_r[None, :],
                             mask = mr[:, None], other = 0.0)
                scores = tl.dot(qr, tl.trans(vr), acc = scores)
            scores = scores * scale
            scores = tl.where(in_range[None, :], scores, -float("inf"))
            m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(scores - m_exp[:, None])
            p = tl.where(in_range[None, :], p, 0.0)
            alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
            l = l * alpha + tl.sum(p, axis = 1)
            pv = p.to(vc.dtype)
            acc_c = acc_c * alpha[:, None] + tl.dot(pv, vc)
            if (not OUT_LATENT) and D_r > 0:
                acc_r = acc_r * alpha[:, None] + tl.dot(pv, vr)
            m_state = m_new

    # Phase 2: pool entries -- gathered by index list, or dense with causal bound
    if DENSE_POOL:
        bound = (q_pos0 + row + 1) // compress_rate
        n_end = tl.minimum(bound, pool_len)
    else:
        n_end = k_len
    for n0 in range(0, n_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        if DENSE_POOL:
            idx = tl.where(offs_n < n_end, offs_n, -1)
        else:
            idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < n_end, other = -1)
        in_range = idx >= 0
        idx_s = tl.where(in_range, idx, 0)
        page = idx_s // page_size
        phys = tl.load(block_table + row * num_pages_per_row + page, mask = in_range, other = 0)
        if DEBUG_BOUNDS:
            tl.device_assert(tl.where(in_range, idx_s < pool_len, True), "dsa_attn: entry idx >= pool_len")
            tl.device_assert(tl.where(in_range, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_attn: pool page OOB")
        tok = phys * page_size + idx_s % page_size
        if QC > 0:
            vc = _qc_load_v(pool_c, pool_s, tok, 0, offs_c, in_range, QC, 1, D_c, D_c_pad)
        else:
            vc = tl.load(pool_c + tok[:, None] * D_c + offs_c[None, :],
                         mask = in_range[:, None] & valid_c[None, :], other = 0.0)
        scores = tl.dot(qc, tl.trans(vc))
        if D_r > 0:
            vr = tl.load(pool_r + tok[:, None] * D_r + offs_r[None, :],
                         mask = in_range[:, None], other = 0.0)
            scores = tl.dot(qr, tl.trans(vr), acc = scores)
        scores = scores * scale
        scores = tl.where(in_range[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(in_range[None, :], p, 0.0)
        alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
        l = l * alpha + tl.sum(p, axis = 1)
        pv = p.to(vc.dtype)
        acc_c = acc_c * alpha[:, None] + tl.dot(pv, vc)
        if (not OUT_LATENT) and D_r > 0:
            acc_r = acc_r * alpha[:, None] + tl.dot(pv, vr)
        m_state = m_new

    denom = tl.where(l == 0.0, 1.0, l)
    oc = acc_c / denom[:, None]
    if QC > 0:
        # Accumulated in the rotated domain: one inverse rotation (H32 is involutory)
        oc = _rot_h32(oc, h32, BLOCK_H, D_c_pad).to(tl.float32)

    if OUT_LATENT:
        ob = out + (offs_h * R + row)[:, None] * D_c
        tl.store(ob + offs_c[None, :], oc.to(tl.float16),
                 mask = valid_h[:, None] & valid_c[None, :])
        return

    o_r = acc_r / denom[:, None]

    if DEROTATE:
        # eq. 26: rotate the output's rope slice at the query's absolute position (the
        # caller passes a negated table for de-rotation). Rotation is linear, so applying
        # it after the softmax-weighted sum equals summing rotated rows. Standard GPT-J
        # pairs (2i, 2i+1) via split/rotate/interleave, all in registers, fp32
        theta = tl.load(derot_inv_freq + tl.arange(0, D_r // 2)) * (q_pos0 + row)
        cos = tl.cos(theta)[None, :]
        sin = tl.sin(theta)[None, :]
        o_e, o_o = tl.split(tl.reshape(o_r, (BLOCK_H, D_r // 2, 2)))
        o_r = tl.interleave(o_e * cos - o_o * sin, o_o * cos + o_e * sin)

    D_out = D_c + D_r
    if HPG > 0:
        # Group-major: out[h // HPG, row, (h % HPG) * D + d]
        base_h = (offs_h // HPG) * (R * HPG * D_out) + row * (HPG * D_out) + (offs_h % HPG) * D_out
    else:
        base_h = (row * H + offs_h) * D_out
    out_base = out + base_h[:, None]
    tl.store(out_base + offs_c[None, :], oc.to(tl.float16), mask = valid_h[:, None] & valid_c[None, :])
    tl.store(out_base + D_c + offs_r[None, :], o_r.to(tl.float16), mask = valid_h[:, None])


@triton.jit(do_not_specialize = ["k_len", "num_pages_per_row", "R", "zero"], debug = dsa_debug_bounds)
def _dsa_attn_latent_kernel(
    q,                   # (H, R, D_c) fp16 head-major absorbed latent queries
    pool_c,              # (pages * page_size, D_c) fp16 paged latent pool (K = V)
    block_table,         # (R, num_pages_per_row) int32 (stride 0 when shared)
    indices,             # (R, K_pad) int32 pool-entry indices, -1 padded
    out,                 # (H, R, D_c) fp16 head-major latent output
    k_len,
    num_pages_per_row,
    R,
    zero,                # runtime 0: keeps the per-chunk q loads inside the key loop
    H: tl.constexpr,
    page_size: tl.constexpr,
    D_c: tl.constexpr,
    K_pad: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DC: tl.constexpr,              # QK reduction chunk
    TRIM: tl.constexpr = 1,        # stop the key loop at the row's last valid index entry
    KV1: tl.constexpr = 0,         # 1 = load each key tile once (full D_c) and reuse it for QK and PV
    PF: tl.constexpr = 0,          # 1 = software-prefetch the next tile's indices + page lookup
    NS: tl.constexpr = 1,          # key-loop num_stages
    OSPLIT: tl.constexpr = 1,      # split the D_c output (PV accumulator) across OSPLIT programs
):
    """GLM sparse prefill (Q_SPLIT, OUT_LATENT, D_r = 0, gathered, no window/sinks/QC).
    Same math as _dsa_attn_kernel, but the QK dot runs over DC-wide chunks of the latent
    with q re-read from L1/L2 per key tile: the one-shot kernel holds the whole (BLOCK_H,
    512) q as a hoisted WMMA A operand (~256 VGPRs) and spills hundreds of registers."""
    pid = tl.program_id(0)
    h_blocks = tl.cdiv(H, BLOCK_H)
    row = pid // (h_blocks * OSPLIT)
    h_block = (pid // OSPLIT) % h_blocks
    osp = pid % OSPLIT
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H
    offs_c = tl.arange(0, D_c)
    offs_o = osp * (D_c // OSPLIT) + tl.arange(0, D_c // OSPLIT)
    offs_d = tl.arange(0, DC)
    q_row = q + (offs_h * R + row)[:, None] * D_c

    m_state = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, D_c // OSPLIT), tl.float32)
    n_end = k_len
    if TRIM:
        # Rows whose context is shorter than the top-k keep a -1 padded tail (up to the
        # whole list for early queries of a >2048-row chunk): scan the int32 list once
        # (8 KB) and walk key tiles only up to the last valid entry. Interior -1 entries,
        # if any, stay masked below, so this does not assume a compacted list.
        last = -1
        for s0 in tl.range(0, k_len, 256, num_stages = 1):
            offs_s = s0 + tl.arange(0, 256)
            ids = tl.load(indices + row * K_pad + offs_s, mask = offs_s < k_len, other = -1)
            last = tl.maximum(last, tl.max(tl.where(ids >= 0, offs_s, -1), axis = 0))
        n_end = last + 1
    if PF:
        offs_n = tl.arange(0, BLOCK_N)
        idx_nx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < n_end, other = -1)
        phys_nx = tl.load(block_table + row * num_pages_per_row + tl.where(idx_nx >= 0, idx_nx, 0) // page_size,
                          mask = idx_nx >= 0, other = 0)
    for n0 in tl.range(0, n_end, BLOCK_N, num_stages = NS):
        if PF:
            idx = idx_nx
            phys = phys_nx
            in_range = idx >= 0
            idx_s = tl.where(in_range, idx, 0)
            offs_nx = n0 + BLOCK_N + tl.arange(0, BLOCK_N)
            idx_nx = tl.load(indices + row * K_pad + offs_nx, mask = offs_nx < n_end, other = -1)
            phys_nx = tl.load(block_table + row * num_pages_per_row + tl.where(idx_nx >= 0, idx_nx, 0) // page_size,
                              mask = idx_nx >= 0, other = 0)
        else:
            offs_n = n0 + tl.arange(0, BLOCK_N)
            idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < n_end, other = -1)
            in_range = idx >= 0
            idx_s = tl.where(in_range, idx, 0)
            phys = tl.load(block_table + row * num_pages_per_row + idx_s // page_size,
                           mask = in_range, other = 0)
        tok = phys * page_size + idx_s % page_size
        k_row = pool_c + tok[:, None] * D_c
        if KV1:
            vc = tl.load(k_row + offs_c[None, :], mask = in_range[:, None], other = 0.0)
            qf = tl.load(q_row + n0 * zero + offs_c[None, :], mask = valid_h[:, None], other = 0.0)
            scores = tl.dot(qf, tl.trans(vc))
        else:
            scores = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
            for d0 in tl.static_range(0, D_c, DC):
                qd = tl.load(q_row + (d0 + n0 * zero) + offs_d[None, :], mask = valid_h[:, None], other = 0.0)
                kd = tl.load(k_row + d0 + offs_d[None, :], mask = in_range[:, None], other = 0.0)
                scores = tl.dot(qd, tl.trans(kd), acc = scores)
        scores = scores * scale
        scores = tl.where(in_range[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(in_range[None, :], p, 0.0)
        alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
        l = l * alpha + tl.sum(p, axis = 1)
        if not KV1:
            vc = tl.load(k_row + offs_o[None, :], mask = in_range[:, None], other = 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vc)
        m_state = m_new

    denom = tl.where(l == 0.0, 1.0, l)
    oc = acc / denom[:, None]
    tl.store(out + (offs_h * R + row)[:, None] * D_c + offs_o[None, :], oc.to(tl.float16),
             mask = valid_h[:, None])


@triton.jit
def _dsa_lat_trim(indices, row, k_len, K_pad: tl.constexpr, TRIM: tl.constexpr):
    n_end = k_len
    if TRIM:
        last = -1
        for s0 in tl.range(0, k_len, 256, num_stages = 1):
            offs_s = s0 + tl.arange(0, 256)
            ids = tl.load(indices + row * K_pad + offs_s, mask = offs_s < k_len, other = -1)
            last = tl.maximum(last, tl.max(tl.where(ids >= 0, offs_s, -1), axis = 0))
        n_end = last + 1
    return n_end


@triton.jit(do_not_specialize = ["k_len", "num_pages_per_row", "R", "zero"], debug = dsa_debug_bounds)
def _dsa_lat2p_stats_kernel(
    q, pool_c, block_table, indices,
    ml,                  # (2, H, R) fp32 out: row max (scaled scores), row sum of exp
    sbuf,                # (R, H, K_pad) fp32 out: scaled scores (STORE = 1 only)
    k_len, num_pages_per_row, R, zero, row0,
    H: tl.constexpr, page_size: tl.constexpr, D_c: tl.constexpr, K_pad: tl.constexpr,
    scale: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, DC: tl.constexpr,
    TRIM: tl.constexpr = 1, STORE: tl.constexpr = 0,
):
    """Two-pass DSA latent, pass 1: QK only over the top-k -> exact row max and sum."""
    pid = tl.program_id(0)
    h_blocks = tl.cdiv(H, BLOCK_H)
    row = row0 + pid // h_blocks
    h_block = pid % h_blocks
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H
    offs_d = tl.arange(0, DC)
    q_row = q + (offs_h * R + row)[:, None] * D_c
    m_state = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    n_end = _dsa_lat_trim(indices, row, k_len, K_pad, TRIM)
    for n0 in tl.range(0, n_end, BLOCK_N, num_stages = 1):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < n_end, other = -1)
        in_range = idx >= 0
        idx_s = tl.where(in_range, idx, 0)
        phys = tl.load(block_table + row * num_pages_per_row + idx_s // page_size, mask = in_range, other = 0)
        k_row = pool_c + (phys * page_size + idx_s % page_size)[:, None] * D_c
        scores = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
        for d0 in tl.static_range(0, D_c, DC):
            qd = tl.load(q_row + (d0 + n0 * zero) + offs_d[None, :], mask = valid_h[:, None], other = 0.0)
            kd = tl.load(k_row + d0 + offs_d[None, :], mask = in_range[:, None], other = 0.0)
            scores = tl.dot(qd, tl.trans(kd), acc = scores)
        scores = scores * scale
        scores = tl.where(in_range[None, :], scores, -float("inf"))
        if STORE:
            tl.store(sbuf + ((row - row0) * H + offs_h)[:, None] * K_pad + offs_n[None, :], scores,
                     mask = valid_h[:, None] & (offs_n < n_end)[None, :])
        m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
        p = tl.where(in_range[None, :], tl.exp(scores - m_exp[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis = 1)
        m_state = m_new
    tl.store(ml + offs_h * R + row, m_state, mask = valid_h)
    tl.store(ml + (H + offs_h) * R + row, l, mask = valid_h)


@triton.jit(do_not_specialize = ["k_len", "num_pages_per_row", "R", "zero"], debug = dsa_debug_bounds)
def _dsa_lat2p_pv_kernel(
    q, pool_c, block_table, indices, ml, sbuf, out,
    k_len, num_pages_per_row, R, zero, row0,
    H: tl.constexpr, page_size: tl.constexpr, D_c: tl.constexpr, K_pad: tl.constexpr,
    scale: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, DC: tl.constexpr,
    TRIM: tl.constexpr = 1, STORE: tl.constexpr = 0, OSPLIT: tl.constexpr = 1,
):
    """Two-pass DSA latent, pass 2: PV with the final row max, no rescale (WMMA accumulate).
    STORE = 1 reads pass-1 scores, 0 recomputes QK."""
    pid = tl.program_id(0)
    h_blocks = tl.cdiv(H, BLOCK_H)
    row = row0 + pid // (h_blocks * OSPLIT)
    h_block = (pid // OSPLIT) % h_blocks
    osp = pid % OSPLIT
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H
    offs_o = osp * (D_c // OSPLIT) + tl.arange(0, D_c // OSPLIT)
    offs_d = tl.arange(0, DC)
    q_row = q + (offs_h * R + row)[:, None] * D_c
    m = tl.load(ml + offs_h * R + row, mask = valid_h, other = 0.0)
    l = tl.load(ml + (H + offs_h) * R + row, mask = valid_h, other = 0.0)
    m_exp = tl.where(m == -float("inf"), 0.0, m)
    acc = tl.zeros((BLOCK_H, D_c // OSPLIT), tl.float32)
    n_end = _dsa_lat_trim(indices, row, k_len, K_pad, TRIM)
    for n0 in tl.range(0, n_end, BLOCK_N, num_stages = 1):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < n_end, other = -1)
        in_range = idx >= 0
        idx_s = tl.where(in_range, idx, 0)
        phys = tl.load(block_table + row * num_pages_per_row + idx_s // page_size, mask = in_range, other = 0)
        k_row = pool_c + (phys * page_size + idx_s % page_size)[:, None] * D_c
        if STORE:
            scores = tl.load(sbuf + ((row - row0) * H + offs_h)[:, None] * K_pad + offs_n[None, :],
                             mask = valid_h[:, None] & in_range[None, :], other = -float("inf"))
        else:
            scores = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
            for d0 in tl.static_range(0, D_c, DC):
                qd = tl.load(q_row + (d0 + n0 * zero) + offs_d[None, :], mask = valid_h[:, None], other = 0.0)
                kd = tl.load(k_row + d0 + offs_d[None, :], mask = in_range[:, None], other = 0.0)
                scores = tl.dot(qd, tl.trans(kd), acc = scores)
            scores = scores * scale
        p = tl.where(in_range[None, :], tl.exp(scores - m_exp[:, None]), 0.0)
        vc = tl.load(k_row + offs_o[None, :], mask = in_range[:, None], other = 0.0)
        acc = tl.dot(p.to(tl.float16), vc, acc = acc)
    denom = tl.where(l == 0.0, 1.0, l)
    oc = acc / denom[:, None]
    tl.store(out + (offs_h * R + row)[:, None] * D_c + offs_o[None, :], oc.to(tl.float16),
             mask = valid_h[:, None])


@triton.jit(do_not_specialize = [
    "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0",
    "win_floor", "ring_beg", "ring_stride",
], debug = dsa_debug_bounds)
def _dsa_attn_split_kernel(
    q,                   # (R, H, D_c + D_r) fp16
    ring,                # (ring_rows, D) fp16, rows at abs - ring_beg
    kv_chunk,            # (R, D) fp16, rows at abs - q_pos0
    pool_c,              # (pages * page_size, D_c) fp16
    pool_r,              # (pages * page_size, D_r) fp16
    block_table,         # (R, num_pages_per_row) int32
    indices,             # (R, K_pad) int32, -1 padded (not DENSE_POOL)
    ws_ml,               # (R * HB * S * BLOCK_H * 2) fp32 partial m / l
    ws_acc,              # (R * HB * S * BLOCK_H * D) fp32 partial numerators
    k_len,
    win_len,
    pool_len,
    num_pages_per_row,
    q_pos0,
    win_floor,
    ring_beg,
    slot_ids,            # MULTIROW: (B,) i32 ring slot per job (else ignored, pass 0)
    ring_stride,         # MULTIROW: ring slot stride in elements
    pool_s,              # (pages * page_size, D_c // 32) fp16 group scales (QC > 0)
    h32,                 # (32, 32) fp16 H32 / sqrt(32) (QC > 0)
    H: tl.constexpr,
    page_size: tl.constexpr,
    D_c: tl.constexpr,
    D_c_pad: tl.constexpr,
    D_r: tl.constexpr,
    K_pad: tl.constexpr,
    compress_rate: tl.constexpr,
    scale: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    DENSE_POOL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
    SEQ: tl.constexpr = 1,
    MULTIROW: tl.constexpr = 0,
    DEBUG_BOUNDS: tl.constexpr = 0,
    DEBUG_PAGES: tl.constexpr = 0,
    Q_SPLIT: tl.constexpr = 0,     # GLM-5.2 absorbed queries: q is the HEAD-MAJOR latent
                                   # (H, R, D_c), ring carries the token-major rope
                                   # queries (R, H, D_r) and ring_stride carries R (both
                                   # slots are free: Q_SPLIT excludes windows, and under
                                   # MULTIROW keeps the per-job state args scalar)
    OUT_LATENT: tl.constexpr = 0,  # accumulate/store only the latent half; ws_acc rows
                                   # are D_c wide and the combine emits (H, R, D_c)
    QC: tl.constexpr = 0,          # packed quantized pool (see _dsa_attn_kernel); the
                                   # partials stay in the rotated domain
):
    """Flash-decoding split phase: each program covers one (query row, head block) and a
    contiguous slice of that row's VIRTUAL key sequence [window keys ++ pool entries],
    writing softmax partials (m, l, acc) to the workspace. Sinks, normalization, the
    eq. 26 de-rotation and the output store all live in the combine kernel."""
    pid = tl.program_id(0)
    split = tl.program_id(1)
    n_splits = tl.num_programs(1)
    h_blocks: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    row = pid // h_blocks

    # MULTIROW: rows are B jobs x SEQ; the position/window/ring state args are per-job
    # i32 arrays, the ring is the stacked (slots, rows, D) tensor addressed by slot, and
    # the block table holds one row per JOB (paged pools). Under Q_SPLIT (gathered mode,
    # no window) the per-job state is dead -- causality lives in the selection and k_len
    # is the uniform K_pad -- so only the job's block-table row matters and the state
    # args stay scalars
    if MULTIROW:
        job = row // SEQ
        loc = row % SEQ
        if not Q_SPLIT:
            q_pos0 = tl.load(q_pos0 + job)
            win_floor = tl.load(win_floor + job)
            ring_beg = tl.load(ring_beg + job)
            pool_len = tl.load(pool_len + job)
            k_len = tl.load(k_len + job)
            slot = tl.load(slot_ids + job)
            ring = ring + slot.to(tl.int64) * ring_stride
        bt_row = job
        cbase = job * SEQ
    else:
        loc = row
        cbase = 0
        bt_row = row

    offs_h = (pid % h_blocks) * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H
    offs_c = tl.arange(0, D_c_pad)
    valid_c = offs_c < D_c
    offs_r = tl.arange(0, D_r if D_r > 0 else 1)
    D = D_c + D_r

    if Q_SPLIT:
        qc = tl.load(q + (offs_h * ring_stride + row)[:, None] * D_c + offs_c[None, :],
                     mask = valid_h[:, None] & valid_c[None, :], other = 0.0)
        if D_r > 0:
            qr = tl.load(ring + (row * H + offs_h)[:, None] * D_r + offs_r[None, :],
                         mask = valid_h[:, None], other = 0.0)
    else:
        q_base = q + (row * H + offs_h)[:, None] * D
        qc = tl.load(q_base + offs_c[None, :], mask = valid_h[:, None] & valid_c[None, :], other = 0.0)
        if D_r > 0:
            qr = tl.load(q_base + D_c + offs_r[None, :], mask = valid_h[:, None], other = 0.0)

    if QC > 0:
        # Packed pool values live in the H32-rotated domain (orthonormal, block-diagonal
        # per 32 values): rotate q once so every score dot is exact, rotate the fp16
        # window tiles into the same domain, and rotate the accumulated output back in
        # the epilogue (one-shot kernel) or the combine (split kernels)
        qc = _rot_h32(qc, h32, BLOCK_H, D_c_pad)

    # This row's virtual key range for this split
    if DENSE_POOL:
        n_pool = tl.minimum((q_pos0 + loc + 1) // compress_rate, pool_len)
    else:
        n_pool = k_len
    n_win = win_len if HAS_WINDOW else 0
    n_tot = n_win + n_pool
    chunk = (n_tot + n_splits - 1) // n_splits
    j0 = split * chunk
    j1 = tl.minimum(j0 + chunk, n_tot)

    m_state = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    acc_c = tl.zeros((BLOCK_H, D_c_pad), tl.float32)
    acc_r = tl.zeros((BLOCK_H, D_r if D_r > 0 else 1), tl.float32)

    if HAS_WINDOW:
        q_abs = q_pos0 + loc
        w1 = tl.minimum(j1, n_win)
        for n0 in tl.range(j0, w1, BLOCK_W, num_stages = 1):
            offs_j = n0 + tl.arange(0, BLOCK_W)
            abs_pos = q_abs - offs_j
            in_range = (offs_j < w1) & (abs_pos >= win_floor)
            mc = in_range & (abs_pos >= q_pos0)
            mr = in_range & (abs_pos < q_pos0)
            idx_c = tl.where(mc, cbase + abs_pos - q_pos0, 0)
            idx_r = tl.where(mr, abs_pos - ring_beg, 0)
            vc = tl.load(kv_chunk + idx_c[:, None] * D + offs_c[None, :],
                         mask = mc[:, None] & valid_c[None, :], other = 0.0) \
               + tl.load(ring + idx_r[:, None] * D + offs_c[None, :],
                         mask = mr[:, None] & valid_c[None, :], other = 0.0)
            if QC > 0:
                vc = _rot_h32(vc, h32, BLOCK_W, D_c_pad)
            scores = tl.dot(qc, tl.trans(vc))
            if D_r > 0:
                vr = tl.load(kv_chunk + idx_c[:, None] * D + D_c + offs_r[None, :],
                             mask = mc[:, None], other = 0.0) \
                   + tl.load(ring + idx_r[:, None] * D + D_c + offs_r[None, :],
                             mask = mr[:, None], other = 0.0)
                scores = tl.dot(qr, tl.trans(vr), acc = scores)
            scores = scores * scale
            scores = tl.where(in_range[None, :], scores, -float("inf"))
            m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(scores - m_exp[:, None])
            p = tl.where(in_range[None, :], p, 0.0)
            alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
            l = l * alpha + tl.sum(p, axis = 1)
            pv = p.to(vc.dtype)
            acc_c = acc_c * alpha[:, None] + tl.dot(pv, vc)
            if (not OUT_LATENT) and D_r > 0:
                acc_r = acc_r * alpha[:, None] + tl.dot(pv, vr)
            m_state = m_new

    p0 = tl.maximum(j0 - n_win, 0)
    p1 = j1 - n_win
    for n0 in range(p0, p1, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        if DENSE_POOL:
            idx = tl.where(offs_n < p1, offs_n, -1)
        else:
            idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < p1, other = -1)
        in_range = idx >= 0
        idx_s = tl.where(in_range, idx, 0)
        page = idx_s // page_size
        phys = tl.load(block_table + bt_row * num_pages_per_row + page, mask = in_range, other = 0)
        if DEBUG_BOUNDS:
            tl.device_assert(tl.where(in_range, idx_s < pool_len, True), "dsa_split: entry idx >= pool_len")
            tl.device_assert(tl.where(in_range, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_split: pool page OOB")
        tok = phys * page_size + idx_s % page_size
        if QC > 0:
            vc = _qc_load_v(pool_c, pool_s, tok, 0, offs_c, in_range, QC, 1, D_c, D_c_pad)
        else:
            vc = tl.load(pool_c + tok[:, None] * D_c + offs_c[None, :],
                         mask = in_range[:, None] & valid_c[None, :], other = 0.0)
        scores = tl.dot(qc, tl.trans(vc))
        if D_r > 0:
            vr = tl.load(pool_r + tok[:, None] * D_r + offs_r[None, :],
                         mask = in_range[:, None], other = 0.0)
            scores = tl.dot(qr, tl.trans(vr), acc = scores)
        scores = scores * scale
        scores = tl.where(in_range[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(in_range[None, :], p, 0.0)
        alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
        l = l * alpha + tl.sum(p, axis = 1)
        pv = p.to(vc.dtype)
        acc_c = acc_c * alpha[:, None] + tl.dot(pv, vc)
        if (not OUT_LATENT) and D_r > 0:
            acc_r = acc_r * alpha[:, None] + tl.dot(pv, vr)
        m_state = m_new

    # Partials out (fp32)
    hloc = tl.arange(0, BLOCK_H)
    base = ((pid * n_splits + split) * BLOCK_H + hloc) * 2
    tl.store(ws_ml + base, m_state)
    tl.store(ws_ml + base + 1, l)
    if OUT_LATENT:
        abase = ((pid * n_splits + split) * BLOCK_H + hloc)[:, None] * D_c
        tl.store(ws_acc + abase + offs_c[None, :], acc_c, mask = valid_c[None, :])
    else:
        abase = ((pid * n_splits + split) * BLOCK_H + hloc)[:, None] * D
        tl.store(ws_acc + abase + offs_c[None, :], acc_c, mask = valid_c[None, :])
        tl.store(ws_acc + abase + D_c + offs_r[None, :], acc_r)

@triton.jit(do_not_specialize = [
    "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0",
    "win_floor", "ring_beg",
])
def _dsa_attn_split_dt_kernel(
    q, ring, kv_chunk, pool_c, pool_r, block_table, indices, ws_ml, ws_acc,
    k_len, win_len, pool_len, num_pages_per_row, q_pos0, win_floor, ring_beg,
    slot_ids, ring_stride, pool_s, h32,
    H: tl.constexpr,
    page_size: tl.constexpr,
    D_c: tl.constexpr,
    K_pad: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_H: tl.constexpr,       # heads per program (16 / 32 / 64)
    BLOCK_N: tl.constexpr,
    BLOCK_DT: tl.constexpr,      # output latent columns per program
    BLOCK_DK: tl.constexpr,      # QK reduction chunk (keeps q and the key tile out of VGPRs)
    WS_BH: tl.constexpr,         # head block of the workspace layout (the combine's BLOCK_H)
    Q_VOLATILE: tl.constexpr = True,
    QK_DYN: tl.constexpr = 0,
):
    """r38 decode split for the GLM absorbed-MLA DSA form (Q_SPLIT, OUT_LATENT, D_r = 0,
    fp16 pages, gathered, single row per job). Same runtime args and workspace layout as
    _dsa_attn_split_kernel (so the unchanged combine and the bc launcher serve it), but the
    program x index covers (head block, latent column tile) instead of head blocks only and
    the QK dot is accumulated in BLOCK_DK chunks: the monolithic form holds a 16 x 512 q, a
    32 x 512 key tile and a 16 x 512 fp32 acc at once and spills (256 VGPRs, ~290 spills).
    Programs per row = (H / BLOCK_H) * (D_c / BLOCK_DT) must equal H / WS_BH (the launcher's
    head-block count); column tiles other than 0 recompute the scores and skip the m / l store."""
    pid = tl.program_id(0)
    split = tl.program_id(1)
    n_splits = tl.num_programs(1)
    n_dt: tl.constexpr = D_c // BLOCK_DT
    per_row: tl.constexpr = (H // BLOCK_H) * n_dt
    row = pid // per_row
    g = pid % per_row
    hblk = g // n_dt
    dt = g % n_dt

    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = dt * BLOCK_DT + tl.arange(0, BLOCK_DT)
    offs_k = tl.arange(0, BLOCK_DK)
    q_rows = (offs_h * ring_stride + row) * D_c        # head-major latent q (H, R, D_c)

    n_tot = k_len
    chunk = (n_tot + n_splits - 1) // n_splits
    j0 = split * chunk
    j1 = tl.minimum(j0 + chunk, n_tot)

    m_state = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DT), tl.float32)

    for n0 in range(j0, j1, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < j1, other = -1)
        in_range = idx >= 0
        idx_s = tl.where(in_range, idx, 0)
        phys = tl.load(block_table + row * num_pages_per_row + idx_s // page_size,
                       mask = in_range, other = 0)
        tok = (phys * page_size + idx_s % page_size).to(tl.int64) * D_c
        scores = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
        if QK_DYN == 0:
            for k0 in tl.static_range(0, D_c, BLOCK_DK):
                qk = tl.load(q + q_rows[:, None] + (k0 + offs_k)[None, :], volatile = Q_VOLATILE)
                vk = tl.load(pool_c + tok[:, None] + (k0 + offs_k)[None, :],
                             mask = in_range[:, None], other = 0.0)
                scores = tl.dot(qk, tl.trans(vk), acc = scores)
        else:
            # r38: a dynamic (not unrolled) reduction loop keeps one q/key chunk live at a
            # time; the unrolled form keeps all D_c/BLOCK_DK q chunks in WMMA A layout and spills
            for k0 in range(0, D_c, BLOCK_DK):
                qk = tl.load(q + q_rows[:, None] + (k0 + offs_k)[None, :], volatile = Q_VOLATILE)
                vk = tl.load(pool_c + tok[:, None] + (k0 + offs_k)[None, :],
                             mask = in_range[:, None], other = 0.0)
                if QK_DYN == 2:
                    scores += tl.trans(tl.dot(vk, tl.trans(qk)))
                else:
                    scores = tl.dot(qk, tl.trans(vk), acc = scores)
        scores = scores * scale
        scores = tl.where(in_range[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(in_range[None, :], p, 0.0)
        alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
        l = l * alpha + tl.sum(p, axis = 1)
        v = tl.load(pool_c + tok[:, None] + offs_d[None, :], mask = in_range[:, None], other = 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_state = m_new

    # Workspace in the combine's (row, head block WS_BH, split, head, col) layout
    hb_ws: tl.constexpr = H // WS_BH
    wrow = ((row * hb_ws + offs_h // WS_BH) * n_splits + split) * WS_BH + offs_h % WS_BH
    if dt == 0:
        tl.store(ws_ml + wrow * 2, m_state)
        tl.store(ws_ml + wrow * 2 + 1, l)
    tl.store(ws_acc + wrow[:, None] * D_c + offs_d[None, :], acc)


@triton.jit(do_not_specialize = [
    "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0",
    "win_floor", "ring_beg",
])
def _dsa_attn_split_fast_kernel(
    q, ring, kv_chunk, pool_c, pool_r, block_table, indices, ws_ml, ws_acc,
    k_len, win_len, pool_len, num_pages_per_row, q_pos0, win_floor, ring_beg,
    slot_ids, ring_stride, pool_s, h32,
    H: tl.constexpr,
    page_size: tl.constexpr,
    D_c: tl.constexpr,
    K_pad: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DT: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    WS_BH: tl.constexpr,
    K_P2: tl.constexpr,
    REBAL: tl.constexpr = 1,     # split the valid prefix, not the -1 padded k_len
    PIPE: tl.constexpr = 0,      # issue the next tile's index -> page -> row chain early
    QK_DYN: tl.constexpr = 1,
):
    """EXL3_DEC_DSA_FAST: _dsa_attn_split_dt_kernel with a shorter dependent-load
    chain. REBAL: every program first reads the row's whole index vector (one K_pad-wide load)
    and splits [0, last valid + 1) instead of k_len: at 4K the selection holds 1024 valid
    entries of 2048, so each program walks half the tiles. Exact for any -1 pattern (a
    trailing run of -1 is the only thing skipped). PIPE: the next tile's indices and page
    lookup are loaded before the current tile's dots, so the idx -> block_table -> row chain
    overlaps compute instead of heading every iteration. Same args / workspace as dt."""
    pid = tl.program_id(0)
    split = tl.program_id(1)
    n_splits = tl.num_programs(1)
    n_dt: tl.constexpr = D_c // BLOCK_DT
    per_row: tl.constexpr = (H // BLOCK_H) * n_dt
    row = pid // per_row
    g = pid % per_row
    hblk = g // n_dt
    dt = g % n_dt

    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = dt * BLOCK_DT + tl.arange(0, BLOCK_DT)
    offs_k = tl.arange(0, BLOCK_DK)
    q_rows = (offs_h * ring_stride + row) * D_c
    ind_row = indices + row * K_pad

    if REBAL:
        offs_all = tl.arange(0, K_P2)
        ids_all = tl.load(ind_row + offs_all, mask = offs_all < k_len, other = -1)
        n_tot = tl.max(tl.where(ids_all >= 0, offs_all + 1, 0), axis = 0)
    else:
        n_tot = k_len
    chunk = (n_tot + n_splits - 1) // n_splits
    chunk = ((chunk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    j0 = split * chunk
    j1 = tl.minimum(j0 + chunk, n_tot)

    m_state = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DT), tl.float32)

    offs_n = j0 + tl.arange(0, BLOCK_N)
    idx = tl.load(ind_row + offs_n, mask = offs_n < j1, other = -1)
    in_range = idx >= 0
    idx_s = tl.where(in_range, idx, 0)
    phys = tl.load(block_table + row * num_pages_per_row + idx_s // page_size, mask = in_range, other = 0)
    for n0 in range(j0, j1, BLOCK_N):
        tok = (phys * page_size + idx_s % page_size).to(tl.int64) * D_c
        cur_in = in_range
        if PIPE:
            offs_nn = n0 + BLOCK_N + tl.arange(0, BLOCK_N)
            idx_n = tl.load(ind_row + offs_nn, mask = offs_nn < j1, other = -1)
        scores = tl.zeros((BLOCK_H, BLOCK_N), tl.float32)
        if QK_DYN == 0:
            for k0 in tl.static_range(0, D_c, BLOCK_DK):
                qk = tl.load(q + q_rows[:, None] + (k0 + offs_k)[None, :], volatile = True)
                vk = tl.load(pool_c + tok[:, None] + (k0 + offs_k)[None, :],
                             mask = cur_in[:, None], other = 0.0)
                scores = tl.dot(qk, tl.trans(vk), acc = scores)
        else:
            for k0 in range(0, D_c, BLOCK_DK):
                qk = tl.load(q + q_rows[:, None] + (k0 + offs_k)[None, :], volatile = True)
                vk = tl.load(pool_c + tok[:, None] + (k0 + offs_k)[None, :],
                             mask = cur_in[:, None], other = 0.0)
                scores = tl.dot(qk, tl.trans(vk), acc = scores)
        if PIPE:
            in_n = idx_n >= 0
            idx_sn = tl.where(in_n, idx_n, 0)
            phys = tl.load(block_table + row * num_pages_per_row + idx_sn // page_size, mask = in_n, other = 0)
        v = tl.load(pool_c + tok[:, None] + offs_d[None, :], mask = cur_in[:, None], other = 0.0)
        scores = scores * scale
        scores = tl.where(cur_in[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(cur_in[None, :], p, 0.0)
        alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
        l = l * alpha + tl.sum(p, axis = 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_state = m_new
        if PIPE:
            in_range = in_n
            idx_s = idx_sn
        else:
            offs_nn = n0 + BLOCK_N + tl.arange(0, BLOCK_N)
            idx_n = tl.load(ind_row + offs_nn, mask = offs_nn < j1, other = -1)
            in_range = idx_n >= 0
            idx_s = tl.where(in_range, idx_n, 0)
            phys = tl.load(block_table + row * num_pages_per_row + idx_s // page_size,
                           mask = in_range, other = 0)

    hb_ws: tl.constexpr = H // WS_BH
    wrow = ((row * hb_ws + offs_h // WS_BH) * n_splits + split) * WS_BH + offs_h % WS_BH
    if dt == 0:
        tl.store(ws_ml + wrow * 2, m_state)
        tl.store(ws_ml + wrow * 2 + 1, l)
    tl.store(ws_acc + wrow[:, None] * D_c + offs_d[None, :], acc)


@triton.jit(do_not_specialize = ["q_pos0", "R", "n_splits"])
def _dsa_attn_combine_kernel(
    ws_ml,               # (R * HB * S * BLOCK_H * 2) fp32
    ws_acc,              # (R * HB * S * BLOCK_H * D) fp32
    sinks,               # (H,) fp32 (HAS_SINKS)
    derot_inv_freq,      # (D_r // 2,) fp32 (DEROTATE)
    out,                 # (R, H, D) fp16 or (H / HPG, R, HPG * D) when HPG > 0
    q_pos0,
    R,
    n_splits,
    h32,                 # (32, 32) fp16 H32 / sqrt(32) (QC > 0)
    H: tl.constexpr,
    D_c: tl.constexpr,
    D_r: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    DEROTATE: tl.constexpr,
    HPG: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,       # even; D_c is even so rope pairs never straddle tiles
    SEQ: tl.constexpr = 1,
    MULTIROW: tl.constexpr = 0,  # q_pos0 is a per-job i32 array when set
    OUT_LATENT: tl.constexpr = 0,  # ws rows are D_c wide; emit head-major (H, R, D_c)
    QC: tl.constexpr = 0,          # split partials came from a packed pool (rotated)
):
    """Combine phase: merge the split partials (sink folded in as one more partial),
    normalize, de-rotate (identity rotation, theta = 0, below the rope columns) and
    store. One program per (row, head block, D tile); the m/l merge is recomputed per D
    tile, which is trivial next to the acc traffic."""
    pid = tl.program_id(0)
    dtile = tl.program_id(1)
    h_blocks: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    row = pid // h_blocks
    offs_h = (pid % h_blocks) * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H
    hloc = tl.arange(0, BLOCK_H)
    D: tl.constexpr = D_c if OUT_LATENT else D_c + D_r
    offs_d = dtile * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_d = offs_d < D

    if HAS_SINKS:
        m_run = tl.load(sinks + offs_h, mask = valid_h, other = -float("inf"))
        l_run = tl.full((BLOCK_H,), 1.0, tl.float32)
    else:
        m_run = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l_run = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    for s in range(n_splits):
        base = ((pid * n_splits + s) * BLOCK_H + hloc)
        m_s = tl.load(ws_ml + base * 2)
        l_s = tl.load(ws_ml + base * 2 + 1)
        a_s = tl.load(ws_acc + base[:, None] * D + offs_d[None, :], mask = valid_d[None, :], other = 0.0)
        m_new = tl.maximum(m_run, m_s)
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.where(m_run == -float("inf"), 0.0, tl.exp(m_run - m_exp))
        beta = tl.where(m_s == -float("inf"), 0.0, tl.exp(m_s - m_exp))
        l_run = l_run * alpha + l_s * beta
        acc = acc * alpha[:, None] + a_s * beta[:, None]
        m_run = m_new

    denom = tl.where(l_run == 0.0, 1.0, l_run)
    o = acc / denom[:, None]

    if QC > 0:
        # Partials from a packed pool are in the rotated domain: one inverse rotation per
        # 32-group of the latent columns (the rope columns are fp16 pages, never rotated;
        # D_c is a multiple of 32, so no group straddles the boundary)
        o_rot = _rot_h32(o, h32, BLOCK_H, BLOCK_D).to(tl.float32)
        o = tl.where(offs_d[None, :] < D_c, o_rot, o)

    if DEROTATE:
        # Uniform pair rotation: theta = 0 (identity) below the rope columns
        offs_p = (dtile * BLOCK_D) // 2 + tl.arange(0, BLOCK_D // 2)
        col_e = offs_p * 2
        in_rope = col_e >= D_c
        fr = tl.load(derot_inv_freq + tl.where(in_rope, (col_e - D_c) // 2, 0),
                     mask = in_rope, other = 0.0)
        if MULTIROW:
            qp = tl.load(q_pos0 + row // SEQ) + row % SEQ
        else:
            qp = q_pos0 + row
        theta = fr * qp
        cos = tl.cos(theta)[None, :]
        sin = tl.sin(theta)[None, :]
        o_e, o_o = tl.split(tl.reshape(o, (BLOCK_H, BLOCK_D // 2, 2)))
        o = tl.interleave(o_e * cos - o_o * sin, o_o * cos + o_e * sin)

    if OUT_LATENT:
        base_h = (offs_h * R + row) * D_c
    elif HPG > 0:
        base_h = (offs_h // HPG) * (R * HPG * D) + row * (HPG * D) + (offs_h % HPG) * D
    else:
        base_h = (row * H + offs_h) * D
    tl.store(out + base_h[:, None] + offs_d[None, :], o.to(tl.float16),
             mask = valid_h[:, None] & valid_d[None, :])

@triton.jit(do_not_specialize = ["T", "R", "q_pos0", "bound_max"],
            debug = dsa_debug_bounds)
def _dsa_indexer_kernel(
    q_idx,               # (R, H_i, D_i) fp16, rope applied
    w,                   # (R, H_i) fp16 raw head weights (scales folded into `scale`)
    k_idx,               # (T, D_i) fp16 indexer keys, rope applied; paged when EPP > 0
    scores,              # (R, S_stride) fp16 out
    T,                   # runtime: valid keys
    R,                   # runtime: valid query rows
    q_pos0,              # runtime: absolute position of query row 0
    bound_max,           # runtime: entry count clamp; bound[r] =
                         #   min((q_pos0 + r + 1) // compress_rate, bound_max)
    block_table,         # EPP > 0: (npr,) i32 page table of the (single) job
    H_i: tl.constexpr,
    D_i: tl.constexpr,
    S_stride: tl.constexpr,
    compress_rate: tl.constexpr,
    scale: tl.constexpr,             # D_i ** -0.5 * H_i ** -0.5
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPP: tl.constexpr = 0,           # pool entries per page; 0 = contiguous k_idx
    DEBUG_BOUNDS: tl.constexpr = 0,
    DEBUG_PAGES: tl.constexpr = 0,
):
    """Lightning-indexer scoring: scores[r, s] = sum_h w[r, h] * relu(q[r, h] . k[s]) * scale.
    GEMM-shaped with a per-head ReLU epilogue; the head loop runs H_i full MMA dots against
    a resident key tile. Shared between raw-token keys (V3.2-on-MLA) and pooled keys (V4
    CSA), only the key tensor differs. Causal entry bound applied in the epilogue."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_i)
    valid_m = offs_m < R
    valid_n = offs_n < T

    if EPP > 0:
        phys = tl.load(block_table + offs_n // EPP, mask = valid_n, other = 0)
        if DEBUG_BOUNDS:
            tl.device_assert(tl.where(valid_n, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_indexer: pool page OOB")
        k_rows = phys * EPP + offs_n % EPP
    else:
        k_rows = offs_n
    kt = tl.load(k_idx + k_rows[None, :] * D_i + offs_d[:, None],
                 mask = valid_n[None, :], other = 0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for h in range(H_i):
        qh = tl.load(q_idx + (offs_m * H_i + h)[:, None] * D_i + offs_d[None, :],
                     mask = valid_m[:, None], other = 0.0)
        logits = tl.dot(qh, kt)
        wh = tl.load(w + offs_m * H_i + h, mask = valid_m, other = 0.0)
        acc += tl.maximum(logits, 0.0) * wh[:, None].to(tl.float32)

    acc = acc * scale
    bound = tl.minimum((q_pos0 + offs_m + 1) // compress_rate, bound_max)
    acc = tl.where(offs_n[None, :] < bound[:, None], acc, -float("inf"))
    tl.store(scores + offs_m[:, None] * S_stride + offs_n[None, :], acc.to(tl.float16),
             mask = valid_m[:, None] & valid_n[None, :])


@triton.jit(do_not_specialize = ["T", "R", "q_pos0", "bound_max", "num_pages_per_row"],
            debug = dsa_debug_bounds)
def _dsa_indexer_fewq_kernel(
    q_idx,               # (R, H_i, D_i) fp16, rope applied
    w,                   # (R, H_i) fp16 raw head weights
    k_idx,               # (T, D_i) fp16 indexer keys, rope applied; paged when EPP > 0
    scores,              # (R, S_stride) fp16 out
    T,
    R,
    q_pos0,
    bound_max,
    block_table,         # EPP > 0: i32 page table, one row per job (row 0 if not MULTIROW)
    num_pages_per_row,   # EPP > 0, MULTIROW: block table row stride
    H_i: tl.constexpr,
    H_pad: tl.constexpr,
    D_i: tl.constexpr,
    S_stride: tl.constexpr,
    compress_rate: tl.constexpr,
    scale: tl.constexpr,             # D_i ** -0.5 * H_i ** -0.5
    BLOCK_N: tl.constexpr,
    SEQ: tl.constexpr = 1,
    MULTIROW: tl.constexpr = 0,      # T/q_pos0/bound_max are per-job i32 arrays
    EPP: tl.constexpr = 0,           # pool entries per page; 0 = contiguous k_idx
    DEBUG_BOUNDS: tl.constexpr = 0,
    DEBUG_PAGES: tl.constexpr = 0,
):
    """Few-query variant (decode): one program per (query row, key tile) with HEADS as
    the MMA M dim. A single dot replaces the head loop, which is a serial latency chain
    when the row tile is nearly empty."""
    r = tl.program_id(0)
    pid_n = tl.program_id(1)
    if MULTIROW:
        job = r // SEQ
        loc = r % SEQ
        T = tl.load(T + job)
        q_pos0 = tl.load(q_pos0 + job)
        bound_max = tl.load(bound_max + job)
        block_table = block_table + job * num_pages_per_row
    else:
        loc = r
    # In a captured graph the grid is sized for the FULL score buffer (pool capacity)
    # with T patched per replay; tiles past T retire immediately (their region of the
    # scores buffer holds the required -inf from the one-time fill)
    if pid_n * BLOCK_N >= T:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_i)
    offs_h = tl.arange(0, H_pad)
    valid_n = offs_n < T
    valid_h = offs_h < H_i

    qh = tl.load(q_idx + (r * H_i + offs_h)[:, None] * D_i + offs_d[None, :],
                 mask = valid_h[:, None], other = 0.0)                     # (H_pad, D)
    if EPP > 0:
        phys = tl.load(block_table + offs_n // EPP, mask = valid_n, other = 0)
        if DEBUG_BOUNDS:
            tl.device_assert(tl.where(valid_n, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_fewq: pool page OOB")
        k_rows = phys * EPP + offs_n % EPP
    else:
        k_rows = offs_n
    kt = tl.load(k_idx + k_rows[None, :] * D_i + offs_d[:, None],
                 mask = valid_n[None, :], other = 0.0)                     # (D, N)
    logits = tl.dot(qh, kt)                                                # (H_pad, N)
    wh = tl.load(w + r * H_i + offs_h, mask = valid_h, other = 0.0)
    acc = tl.sum(tl.maximum(logits, 0.0) * wh[:, None].to(tl.float32), axis = 0) * scale

    bound = tl.minimum((q_pos0 + loc + 1) // compress_rate, bound_max)
    acc = tl.where(offs_n < bound, acc, -float("inf"))
    tl.store(scores + r * S_stride + offs_n, acc.to(tl.float16), mask = valid_n)


@triton.jit(do_not_specialize = ["num_pages_per_row", "append_len"])
def _dsa_pool_update_kernel(
    plane,               # flat (pages * page_size, 2 * D) packed [k || gate] rows
    pool_plane,          # flat (pages * EPP_POOL, D) pooled keys
    ape,                 # (P, D) fp16 learned in-pool position embedding
    block_table,         # (bsz, num_pages_per_row) i32
    cache_seqlens,       # (bsz,) i32, pre-append counts
    num_pages_per_row,
    append_len,
    page_size: tl.constexpr,
    P: tl.constexpr,     # tokens per pool
    D: tl.constexpr,
    MAXPOOLS: tl.constexpr,   # grid height: append_len // P + 1
):
    """(Re)build the pooled keys touched by this append: softmax(gate + ape) over the
    pool's present members, weighted mean of their keys. Branch-free w.r.t. pool
    completion: partially filled pools are written too but never selected (the causal
    bound admits only complete pools), and the write that completes a pool sees all P
    members. Pools never straddle pages (page_size % P == 0)."""
    b = tl.program_id(0)
    pi = tl.program_id(1)
    pos0 = tl.load(cache_seqlens + b)
    t_end = pos0 + append_len
    pool = pos0 // P + pi
    if pool * P >= t_end:
        return

    offs_d = tl.arange(0, D)
    bt = block_table + b * num_pages_per_row

    # Per-dim softmax over the present members (j: token pool*P + j < t_end)
    m = tl.full((D,), -float("inf"), tl.float32)
    for j in range(P):
        tok = pool * P + j
        if tok < t_end:
            phys = tl.load(bt + tok // page_size)
            row = phys * page_size + tok % page_size
            gv = tl.load(plane + row * (2 * D) + D + offs_d).to(tl.float32) \
               + tl.load(ape + j * D + offs_d).to(tl.float32)
            m = tl.maximum(m, gv)
    den = tl.zeros((D,), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    for j in range(P):
        tok = pool * P + j
        if tok < t_end:
            phys = tl.load(bt + tok // page_size)
            row = phys * page_size + tok % page_size
            gv = tl.load(plane + row * (2 * D) + D + offs_d).to(tl.float32) \
               + tl.load(ape + j * D + offs_d).to(tl.float32)
            e = tl.exp(gv - m)
            den += e
            acc += e * tl.load(plane + row * (2 * D) + offs_d).to(tl.float32)
    pk = acc / den

    phys0 = tl.load(bt + (pool * P) // page_size)
    prow = phys0 * (page_size // P) + pool % (page_size // P)
    tl.store(pool_plane + prow * D + offs_d, pk.to(tl.float16))


@triton.jit(do_not_specialize = ["q_pos0"])
def _dsa_pool_expand_kernel(
    pool_idx,            # (R, KP_pool) i32 selected pool ids, -1 padded
    out,                 # (R, K_pad) i32 raw token indices
    q_pos0,              # scalar (patched) or per-job i32 array (MULTIROW)
    P: tl.constexpr,
    SEL: tl.constexpr,   # pools selected per row (topk // P)
    K_pad: tl.constexpr,
    KP_pool: tl.constexpr,
    TAIL: tl.constexpr,
    SEQ: tl.constexpr = 1,
    MULTIROW: tl.constexpr = 0,
    BLOCK: tl.constexpr = 256,
):
    """Expand selected pools to raw token indices (pool * P + j) and append the query's
    incomplete tail pool as raw tokens. -1 entries pass through; unwritten columns pad
    with -1."""
    r = tl.program_id(0)
    c0 = tl.program_id(1) * BLOCK
    offs = c0 + tl.arange(0, BLOCK)
    if MULTIROW:
        job = r // SEQ
        loc = r % SEQ
        q_pos = tl.load(q_pos0 + job) + loc
    else:
        q_pos = q_pos0 + r % SEQ
    vis = q_pos + 1

    # Expanded region [0, SEL * P)
    pool = tl.load(pool_idx + r * KP_pool + offs // P, mask = offs < SEL * P, other = -1)
    v = tl.where(pool >= 0, pool * P + offs % P, -1)

    # Tail region [SEL * P, SEL * P + P - 1)
    if TAIL:
        tcount = vis % P
        tstart = vis - tcount
        ti = offs - SEL * P
        tv = tl.where(ti < tcount, tstart + ti, -1)
        v = tl.where((ti >= 0) & (ti < P - 1), tv, v)

    v = tl.where(offs < SEL * P + (P - 1 if TAIL else 0), v, -1)
    tl.store(out + r * K_pad + offs, v, mask = offs < K_pad)

# Packed-pool prefill staging: the gathered kernel dequantizes every selected entry per query
# row, so a prefill chunk of R rows re-expands R * k entries per layer (R = 4096, k = 2048:
# 8.4M expansions), several times the whole context. From EXL3_DSA_QC_STAGE_MIN_R rows on, the
# referenced pool window (pool_len entries, i.e. the actual context, never the cache pool) is
# gathered and dequantized once into a per-call fp16 transient and the fp16 kernel runs over
# it: measured 3.9x faster attention at 16k context (GLM-5.3, q6). EXL3_DSA_QC_STAGE=0 keeps
# the online path; _MAX_ENTRIES bounds the transient (1M entries * 576 dims ~ 1.2 GB)
_dsa_qc_stage = os.environ.get("EXL3_DSA_QC_STAGE", "1") != "0"
_dsa_qc_stage_min_r = int(os.environ.get("EXL3_DSA_QC_STAGE_MIN_R", "64"))
_dsa_qc_stage_max_entries = int(os.environ.get("EXL3_DSA_QC_STAGE_MAX_ENTRIES", str(1 << 20)))


def _stage_packed_pool(pool_c, qc, pool_r, block_table, page_size, pool_len, D_c, D_r):
    """Gather the pages holding entries [0, pool_len) of a packed pool and dequantize them into
    a contiguous fp16 pool with an identity block table. Returns (pool_c, pool_r, block_table)
    for the fp16 kernel path."""
    from ...ext import exllamav3_ext as ext
    pool_s, bits = qc
    G = D_c // 32
    npw = -(-pool_len // page_size)
    pages = block_table[0, :npw].long()
    pc = pool_c.reshape(-1, page_size, G * bits)[pages]
    ps = pool_s.reshape(-1, page_size, G)[pages]
    n_rows = npw * page_size
    rows_alloc = max(page_size, 1 << (n_rows - 1).bit_length())      # pow2: few distinct sizes
    out = torch.empty((rows_alloc, D_c), dtype = torch.half, device = pool_c.device)
    ext.dequant_cache_cont(pc.view(-1, G * bits), ps.view(-1, G), out[:n_rows], 0.0)
    pr = pool_r.reshape(-1, page_size, D_r)[pages].contiguous() if D_r > 0 else pool_r
    bt = torch.arange(npw, dtype = torch.int32, device = pool_c.device).unsqueeze(0)
    return out[:n_rows].view(npw, page_size, D_c), pr, bt


def dsa_attn(
    q,                       # (R, H, D_c + D_r) fp16, rope slice pre-rotated
    pool_c,                  # (pages, page_size, D_c) or (rows, D_c) fp16
    pool_r,                  # matching rope-part tensor, D_r wide
    block_table,             # (R, num_pages_per_row) int32
    sinks = None,            # (H,) fp32 or None
    ring = None,             # (ring_rows, D_c + D_r) fp16: rows at abs - ring_beg (window)
    kv_chunk = None,         # (R, D_c + D_r) fp16: this chunk's rows at abs - q_pos0
    win_len = 0,
    win_floor = 0,           # lowest absolute position visible to the window
    ring_beg = 0,            # absolute position of ring row 0
    indices = None,          # (R, K_pad) int32, -1 padded (gathered mode)
    k_len = 0,
    pool_len = 0,            # dense mode: pool entry count
    q_pos0 = 0,              # dense mode: absolute position of query row 0
    compress_rate = 1,       # dense mode causal bound divisor
    scale = None,
    derot_inv_freq = None,   # (D_r // 2,) fp32: fuse eq. 26 output rotation at q_pos0 + row
                             # (pass the NEGATED frequency table for de-rotation)
    groups = 1,              # > 1: store output group-major (groups, R, (H // groups) * D)
    group_major = None,      # force group-major layout even at groups == 1 (TP shard
                             # holding a single o_group); default = (groups > 1)
    out = None,
    n_splits = 0,            # flash-decoding splits; 0 = auto (few queries -> split path)
    block_h = 32,
    block_n = 32,
    num_warps = 4,
    num_stages = 3,
    page_size = 256,         # pool entries per block-table page (PAGE_SIZE // m for the
                             # paged pools; 256 with an identity table for contiguous pools)
    nc_block = False,        # DSpark draft mode: non-causal chunk + paged window history
                             # (single job per call; forces the one-shot kernel)
    nc_chunk = False,        # non-causal image chunk (DeepSeek-V4 vision): every query sees
                             # the whole chunk (bidirectional) plus its own win_len-wide
                             # causal window into the ring history; pool phases unchanged
                             # (single job, one-shot kernel)
    multirow = None,         # batched jobs: dict(q_pos, win_floor, ring_beg, pool_len,
                             # k_len (B,) i32; slot_ids (B,) i32; ring_stride int; seq int)
                             # -- scalar args of the same names are ignored, ring is the
                             # stacked (slots, rows, D) tensor, block_table holds one row
                             # per JOB, R = B * seq
    q_pe = None,             # GLM-5.2 absorbed form: q is the HEAD-MAJOR latent (H, R, D_c)
                             # and q_pe the token-major rope queries (R, H, D_r); the kernel
                             # reads both directly, no packed copy exists anywhere
    out_latent = False,      # emit only the latent half, head-major (H, R, D_c): the exact
                             # mla_unfold input, with the rope half never accumulated
    qc = None,               # (scales, bits): pool_c is the packed quantized pool (pages,
                             # page_size, D_c // 32 * bits) int32 with fp16 group scales
                             # (pages, page_size, D_c // 32); read online in the kernels
):
    """Core DSA attention over [sliding ring ++ pool entries], V IS K. Gathered mode when
    `indices` is given, dense-pool mode otherwise (per-query causal entry bound). With
    derot_inv_freq the eq. 26 output de-rotation is fused into the epilogue (otherwise the
    output's rope slice is still rotated and de-rotation is the caller's epilogue); with
    groups > 1 the output is written group-major for the grouped o_proj."""
    q_split = q_pe is not None
    if q_split:
        # Split absorbed queries exclude every feature that would touch the packed layout
        assert not (win_len and kv_chunk is not None) and sinks is None and \
            derot_inv_freq is None and groups == 1 and not group_major and \
            multirow is None and not nc_block and out_latent
        H, R, D_c = q.shape
        D_r = q_pe.shape[-1]
        D = D_c + D_r
        assert q_pe.shape == (R, H, D_r) and q_pe.is_contiguous() and q.is_contiguous()
    else:
        R, H, D = q.shape
        D_r = pool_r.shape[-1]
        D_c = D - D_r
    if scale is None:
        scale = D ** -0.5
    dense_pool = indices is None
    assert H % groups == 0
    if group_major is None:
        group_major = groups > 1
    hpg = H // groups if group_major else 0
    assert out_latent or q_pe is None or q_pe.shape[-1] > 0, \
        "dsa_attn: D_r == 0 requires out_latent"
    if out_latent:
        out_shape = (H, R, D_c)
        if out is None:
            # Owned by the caller (results may be held across calls); not cached
            out = torch.empty(out_shape, dtype = torch.half, device = q.device)
        else:
            assert out.shape == out_shape
    else:
        out_shape = (groups, R, hpg * D) if group_major else (R, H, D)
        if out is None:
            out = g_tensor_cache.get(q.device, out_shape, torch.half, "dsa_out")
        else:
            assert out.shape == out_shape

    dummy_i = block_table  # any valid int32 pointer for unused int args
    if (qc is not None and _dsa_qc_stage and multirow is None and not nc_block
            and R >= _dsa_qc_stage_min_r and block_table.shape[0] == 1
            and 0 < pool_len <= _dsa_qc_stage_max_entries):
        pool_c, pool_r, block_table = _stage_packed_pool(
            pool_c, qc, pool_r, block_table, page_size, pool_len, D_c, D_r)
        qc = None
        dummy_i = block_table
    if qc is not None:
        pool_s, qc_bits = qc
        assert D_c % 32 == 0 and pool_c.dtype == torch.int32 and pool_s.dtype == torch.half
        h32_t = _get_h32(q.device)
        pc_arg = pool_c
    else:
        pool_s, qc_bits, h32_t = q, 0, q   # any valid fp16 pointers for the unused args
        pc_arg = pool_c.reshape(-1, D_c)
    has_window = win_len > 0 and kv_chunk is not None
    if q_split:
        # The windowless ring slot carries q_pe; kv_chunk stays a dummy pointer
        ring, kv_chunk = q_pe, q
    elif not has_window:
        ring, kv_chunk = q, q
    elif ring is None:
        ring = kv_chunk    # window fully inside the chunk (win_floor >= q_pos0)
    if indices is None:
        indices, K_pad = dummy_i, 32
    else:
        K_pad = indices.shape[1]
        k_len = k_len or K_pad
    if sinks is None:
        sinks_t, has_sinks = q, False
    else:
        sinks_t, has_sinks = sinks, True
    if derot_inv_freq is None:
        derot_t, derotate = q, False   # any valid fp pointer for the unused arg
    else:
        derot_t, derotate = derot_inv_freq, True

    # A single-row block table is shared by every query row (contiguous per-slot pools):
    # stride 0 makes the kernel's bt[row * npr + page] hit row 0 for all rows
    npr = block_table.shape[1] if block_table.shape[0] > 1 else 0
    dbg = 1 if dsa_debug_bounds else 0
    if dsa_debug_bounds:
        # Debug codegen inflates the one-shot kernel's smem footprint past 99 KB devices;
        # halve the head tile (perf is irrelevant with asserts on)
        num_stages = min(num_stages, 2)
        block_h = min(block_h, 16)
    if dsa_debug_bounds:
        pool_rows = pool_s.numel() // (D_c // 32) if qc is not None else pool_c.numel() // max(D_c, 1)
        dbg_pages = -(-pool_rows // max(page_size, 1))
    else:
        dbg_pages = 0
    fast_cfg = _dsa_dec_fast_cfg(R) if (q_split and out_latent and D_r == 0 and qc is None and multirow is None) else None
    if nc_block:
        n_splits = 1
    nc_hist = 0
    if nc_chunk:
        assert has_window and not nc_block and multirow is None, "nc_chunk: single windowed job only"
        n_splits = 1
        # Scan the whole chunk plus (win_len - 1) history rows below it; the kernel keeps
        # only each row's own history window
        nc_hist = win_len
        win_len = win_len + R - 1
    if multirow is not None:
        n_splits = n_splits or 8
    if n_splits == 0:
        if R <= 8:
            est = (win_len if has_window else 0) + \
                  (k_len if indices is not dummy_i else min(pool_len, (q_pos0 + R) // max(compress_rate, 1)))
            n_splits = 16 if est > 256 else 8
            # EXL3_DEC_DSA_FAST split count (gathered latent decode only)
            if fast_cfg is not None and fast_cfg[2] > 0 and indices is not dummy_i and out_latent:
                n_splits = fast_cfg[2]
        else:
            n_splits = 1

    if n_splits > 1:
        block_h = min(block_h, 16)   # more head-blocks: the split path wants parallelism
        # Flash-decoding split: the monolithic kernel at R = 1 is 2 blocks walking the key
        # tiles serially (pure latency); the split phase spreads the keys over
        # R * h_blocks * n_splits programs and the combine folds sinks/derot/store
        hb = triton.cdiv(H, block_h)
        D_out = D_c if out_latent else D
        ws_ml = g_tensor_cache.get(q.device, (R * hb * n_splits * block_h * 2,),
                                   torch.float, "dsa_ws_ml")
        ws_acc = g_tensor_cache.get(q.device, (R * hb * n_splits * block_h * D_out,),
                                    torch.float, "dsa_ws_acc")
        if multirow is not None:
            mr = multirow
            a_klen, a_pool = mr["k_len"], mr["pool_len"]
            a_qpos, a_floor, a_beg = mr["q_pos"], mr["win_floor"], mr["ring_beg"]
            a_slots, a_rstride, a_seq = mr["slot_ids"], mr["ring_stride"], mr["seq"]
            npr = block_table.shape[1]     # one block-table row per job
        else:
            a_klen, a_pool, a_qpos, a_floor, a_beg = k_len, pool_len, q_pos0, win_floor, ring_beg
            # ring_stride carries R for the head-major q_lat load in split-q mode
            a_slots, a_rstride, a_seq = 0, (R if q_split else 0), 1
        # glm-mtp step 9: R <= EXL3_DSA_DEC_DT_R (default 4) so MTP verify rows take the same
        # kernel as the R = 1 plain decode (per-row programs, n_splits fixed for R <= 8: row r
        # reduces bit-equal to a batch-1 call). EXL3_DSA_DEC_DT_R=1 = r39 behaviour (R = 1 only).
        dt_cfg = _dsa_dec_dt_cfg() if (
            R <= int(os.environ.get("EXL3_DSA_DEC_DT_R", "4")) and multirow is None and q_split and out_latent and D_r == 0 and qc is None
            and not has_window and not dense_pool and indices is not dummy_i and not dbg
            and not dbg_pages and block_h == 16) else None
        if dt_cfg is not None and fast_cfg is not None and len(fast_cfg) == 10:
            dt_cfg = fast_cfg[3:]   # the fast preset carries its own BH,DT,BN,DK,NW,NS,QK
        if dt_cfg is not None and (H % dt_cfg[0] or D_c % dt_cfg[1]
                                   or (H // dt_cfg[0]) * (D_c // dt_cfg[1]) != hb):
            dt_cfg = None
        with torch.cuda.device(q.device):
            if dt_cfg is not None and fast_cfg is not None:
                # dt kernel with the valid-prefix split / pipelined index chain
                bh, bdt, bn, bdk, nw, ns, qd = dt_cfg
                _dsa_attn_split_fast_kernel[(R * hb, n_splits)](
                    q, ring, kv_chunk, pc_arg, pool_r.reshape(-1),
                    block_table, indices, ws_ml, ws_acc,
                    a_klen, win_len, a_pool, npr, a_qpos, a_floor, a_beg,
                    a_slots, a_rstride, pool_s, h32_t,
                    H = H, page_size = page_size, D_c = D_c, K_pad = K_pad, scale = scale,
                    BLOCK_H = bh, BLOCK_N = bn, BLOCK_DT = bdt, BLOCK_DK = bdk, WS_BH = block_h,
                    K_P2 = triton.next_power_of_2(K_pad),
                    REBAL = fast_cfg[0], PIPE = fast_cfg[1], QK_DYN = 0 if qd == 0 else 1,
                    num_warps = nw, num_stages = ns,
                )
            elif dt_cfg is not None:
                if os.environ.get("EXL3_DSA_DEC_DEBUG"):
                    print(f"[dsa_dec] dt {dt_cfg} splits {n_splits} k_len {a_klen}", flush = True)
                # r38: column-tiled split without VGPR spills (same args, grid, ws layout)
                bh, bdt, bn, bdk, nw, ns, qd = dt_cfg
                _dsa_attn_split_dt_kernel[(R * hb, n_splits)](
                    q, ring, kv_chunk, pc_arg, pool_r.reshape(-1),
                    block_table, indices, ws_ml, ws_acc,
                    a_klen, win_len, a_pool, npr, a_qpos, a_floor, a_beg,
                    a_slots, a_rstride, pool_s, h32_t,
                    H = H, page_size = page_size, D_c = D_c, K_pad = K_pad, scale = scale,
                    BLOCK_H = bh, BLOCK_N = bn, BLOCK_DT = bdt, BLOCK_DK = bdk, WS_BH = block_h,
                    Q_VOLATILE = True, QK_DYN = qd,
                    num_warps = nw, num_stages = ns,
                )
            else:
                _dsa_attn_split_kernel[(R * hb, n_splits)](
                    q, ring, kv_chunk, pc_arg, (pool_r.reshape(-1, D_r) if D_r > 0 else pool_r.reshape(-1)),
                    block_table, indices, ws_ml, ws_acc,
                    a_klen, win_len, a_pool, npr, a_qpos, a_floor, a_beg,
                    a_slots, a_rstride, pool_s, h32_t,
                    H = H, page_size = page_size, D_c = D_c, D_c_pad = triton.next_power_of_2(D_c),
                    D_r = D_r, K_pad = K_pad, compress_rate = compress_rate, scale = scale,
                    HAS_WINDOW = has_window, DENSE_POOL = dense_pool,
                    BLOCK_H = block_h, BLOCK_N = block_n, BLOCK_W = 16,
                    SEQ = a_seq, MULTIROW = multirow is not None,
                    DEBUG_BOUNDS = dbg, DEBUG_PAGES = dbg_pages,
                    Q_SPLIT = 1 if q_split else 0, OUT_LATENT = 1 if out_latent else 0,
                    QC = qc_bits,
                    num_warps = num_warps, num_stages = 2,
                )
            _dsa_attn_combine_kernel[(R * hb, triton.cdiv(D_out, 128))](
                ws_ml, ws_acc, sinks_t, derot_t, out,
                a_qpos, R, n_splits, h32_t,
                H = H, D_c = D_c, D_r = D_r,
                HAS_SINKS = has_sinks, DEROTATE = derotate, HPG = hpg,
                BLOCK_H = block_h, BLOCK_D = 128,
                SEQ = a_seq, MULTIROW = multirow is not None,
                OUT_LATENT = 1 if out_latent else 0,
                QC = qc_bits,
                num_warps = 4, num_stages = 2,
            )
        return out

    # EXL3_DSA_LATENT (read per call): 1 = chunked-QK latent kernel for the GLM sparse
    # prefill shape (default), 0 = one-shot kernel, "BH:BN:NW:DC[:KV1[:PF[:NS]]]" = sweep a tile config
    _lat = os.environ.get("EXL3_DSA_LATENT", "1")
    # EXL3_DSA_TRIM (read per call): 1 = latent kernel stops each row at its last valid
    # index entry (default), 0 = walk all k_len tiles
    _trim = 0 if os.environ.get("EXL3_DSA_TRIM", "1") == "0" else 1
    if (_lat != "0" and q_split and out_latent and D_r == 0 and not has_window and not has_sinks
            and not derotate and qc_bits == 0 and not dense_pool and not nc_block and not nc_chunk
            and D_c == triton.next_power_of_2(D_c) and not dsa_debug_bounds):
        bh, bn, nw, dc, *ext = (int(x) for x in _lat.split(":")) if ":" in _lat else (32, 16, 4, 64)
        kv1, pf, ns = (list(ext) + [0, 0, 1][len(ext):])[:3]
        # EXL3_DSA_LAT_OSPLIT (read per call, opt-in): split the D_c output accumulator across n programs
        osplit = int(os.environ.get("EXL3_DSA_LAT_OSPLIT", "1"))
        assert osplit == 1 or not kv1, "EXL3_DSA_LAT_OSPLIT > 1 needs KV1 = 0"
        # EXL3_DSA_LAT_2P (read per call): "BN1:NW1:BN2:NW2[:STORE]" = two-pass latent kernel (pass 1 QK ->
        # exact row max/sum, pass 2 PV without rescale; STORE=1 keeps pass-1 scores in fp32, row-sliced
        # <= 512 MB, and pass 2 skips QK). Default 64:4:16:4:1 since mla-prefill step 4 (-36 % kernel,
        # PPL 10x4096 +0.054 %, KL 3.5e-3); "0" = the one-pass kernel above.
        _2p = os.environ.get("EXL3_DSA_LAT_2P", "64:4:16:4:1")
        with torch.cuda.device(q.device):
            if _2p != "0":
                bn1, nw1, bn2, nw2, *st = (int(x) for x in _2p.split(":"))
                store = st[0] if st else 0
                ml = torch.empty((2, H, R), dtype = torch.float32, device = q.device)
                # STORE: fp32 scores per row slice, bounded to ~512 MB (2048 rows at H 32, top-k 2048)
                rs = max(1, min(R, (512 << 20) // (H * K_pad * 4))) if store else R
                sbuf = torch.empty((rs, H, K_pad) if store else (1,), dtype = torch.float32, device = q.device)
                hb = triton.cdiv(H, bh)
                for r0 in range(0, R, rs):
                    nr = min(rs, R - r0)
                    _dsa_lat2p_stats_kernel[(nr * hb,)](
                        q, pc_arg, block_table, indices, ml, sbuf, k_len, npr, R, 0, r0,
                        H = H, page_size = page_size, D_c = D_c, K_pad = K_pad, scale = scale,
                        BLOCK_H = bh, BLOCK_N = bn1, DC = dc, TRIM = _trim, STORE = store,
                        num_warps = nw1, num_stages = 1,
                    )
                    _dsa_lat2p_pv_kernel[(nr * hb * osplit,)](
                        q, pc_arg, block_table, indices, ml, sbuf, out, k_len, npr, R, 0, r0,
                        H = H, page_size = page_size, D_c = D_c, K_pad = K_pad, scale = scale,
                        BLOCK_H = bh, BLOCK_N = bn2, DC = dc, TRIM = _trim, STORE = store, OSPLIT = osplit,
                        num_warps = nw2, num_stages = 1,
                    )
                return out
            _dsa_attn_latent_kernel[(R * triton.cdiv(H, bh) * osplit,)](
                q, pc_arg, block_table, indices, out, k_len, npr, R, 0,
                H = H, page_size = page_size, D_c = D_c, K_pad = K_pad, scale = scale,
                BLOCK_H = bh, BLOCK_N = bn, DC = dc, TRIM = _trim, KV1 = kv1, PF = pf, NS = ns,
                OSPLIT = osplit, num_warps = nw, num_stages = 1,
            )
        return out
    grid = (R * triton.cdiv(H, block_h),)
    with torch.cuda.device(q.device):   # layer split: launch on the tensor's device
        _dsa_attn_kernel[grid](
            q, ring, kv_chunk, pc_arg, (pool_r.reshape(-1, D_r) if D_r > 0 else pool_r.reshape(-1)),
            block_table, indices, sinks_t, derot_t, out,
            k_len, win_len, pool_len, npr, q_pos0, R, win_floor, ring_beg, pool_s, h32_t,
            H = H, page_size = page_size, D_c = D_c, D_c_pad = triton.next_power_of_2(D_c),
            D_r = D_r, K_pad = K_pad, compress_rate = compress_rate,
            scale = scale,
            HAS_WINDOW = has_window,
            HAS_SINKS = has_sinks,
            DENSE_POOL = dense_pool,
            DEROTATE = derotate,
            HPG = hpg,
            BLOCK_H = block_h, BLOCK_N = block_n, BLOCK_W = 16,
            DEBUG_BOUNDS = dbg, DEBUG_PAGES = dbg_pages,
            NC_BLOCK = 1 if nc_block else 0,
            NC_CHUNK = 1 if nc_chunk else 0, NC_HIST = nc_hist,
            Q_SPLIT = 1 if q_split else 0, OUT_LATENT = 1 if out_latent else 0,
            QC = qc_bits,
            num_warps = num_warps, num_stages = num_stages,
        )
    return out


def dsa_indexer_scores(
    q_idx,                   # (R, H_i, D_i) fp16, rope applied
    weights,                 # (R, H_i) fp16 raw head weights (H_i ** -0.5 folded in-kernel)
    k_idx,                   # (T, D_i) fp16 key pool, rope applied; contiguous rows, or the
                             # flat paged pool when block_table is given
    q_pos0,                  # absolute position of query row 0
    compress_rate,
    bound_max,               # entry count clamp (valid pool entries)
    scores = None,
    block_m = 64,
    block_n = 128,
    num_warps = 8,
    num_stages = 2,
    block_table = None,      # (npr,) or (1, npr) i32 page table of the (single) job
    epp = 0,                 # pool entries per page (paged mode)
    scale = None,            # None: D_i ** -0.5 * H_i ** -0.5 (DSA); QSA passes dk ** -0.5
):
    """Indexer scores (R, T) fp16 with -inf past each query's causal entry bound
    min((q_pos0 + r + 1) // compress_rate, bound_max); feed to topk."""
    R, H_i, D_i = q_idx.shape
    if scale is None:
        scale = D_i ** -0.5 * H_i ** -0.5
    T = bound_max
    if block_table is None:
        T = k_idx.shape[0]
        bt, epp = 0, 0
    else:
        bt = block_table.reshape(-1)
    dbg = 1 if (dsa_debug_bounds and epp) else 0
    dbg_pages = -(-k_idx.shape[0] // epp) if dbg else 0
    # S_stride is a kernel constexpr (row stride of the score matrix), so every distinct value
    # is a separate Triton compile. It must therefore NOT track the visible context: a caller
    # that hands in a backing sets it (a fixed tile width, one compile per model); without a
    # backing the per-call allocation is rounded to a power of two, bounding the compile count
    # by log2 of the context. The kernels only write columns < T either way
    T_pad = triton.cdiv(max(T, 1), block_n) * block_n
    if scores is None:
        # Deliberately a per-call allocation: the shape grows with the visible context, so it
        # is not tensor-cache material.
        # TODO: The buffer should disappear entirely once scoring and top-k are fused into a streaming kernel
        S_stride = max(block_n, triton.next_power_of_2(T_pad))
        scores = torch.empty((R, S_stride), dtype = torch.half, device = q_idx.device)
    else:
        S_stride = scores.shape[1]
        assert scores.shape[0] >= R and S_stride >= T_pad and S_stride % 8 == 0 \
            and scores.stride(1) == 1 and scores.stride(0) == S_stride, \
            "dsa_indexer_scores: score backing must be a row-contiguous (>= R, >= T rounded to the tile) view"
    with torch.cuda.device(q_idx.device):
        if R <= 4:
            # Few-query (decode) shape: heads as the MMA M dim, one dot per key tile --
            # the query-tiled kernel degenerates to a serial head loop over padding here
            grid = (R, triton.cdiv(max(T, 1), block_n))
            _dsa_indexer_fewq_kernel[grid](
                q_idx, weights, k_idx, scores, T, R, q_pos0, bound_max, bt, 0,
                H_i = H_i, H_pad = max(triton.next_power_of_2(H_i), 16), D_i = D_i,
                S_stride = S_stride, compress_rate = compress_rate,
                scale = scale,
                BLOCK_N = block_n, EPP = epp,
                DEBUG_BOUNDS = dbg, DEBUG_PAGES = dbg_pages,
                num_warps = num_warps, num_stages = num_stages,
            )
        else:
            grid = (triton.cdiv(R, block_m), triton.cdiv(max(T, 1), block_n))
            _dsa_indexer_kernel[grid](
                q_idx, weights, k_idx, scores, T, R, q_pos0, bound_max, bt,
                H_i = H_i, D_i = D_i, S_stride = S_stride, compress_rate = compress_rate,
                scale = scale,
                BLOCK_M = block_m, BLOCK_N = block_n, EPP = epp,
                DEBUG_BOUNDS = dbg, DEBUG_PAGES = dbg_pages,
                num_warps = num_warps, num_stages = num_stages,
            )
    return scores[:, :T]


def _dsa_dec_fast_cfg(R = 1):
    """EXL3_DEC_DSA_FAST (read per call, default 0 = off) = "REBAL,PIPE,SPLITS" for
    _dsa_attn_split_fast_kernel on the dt decode path (SPLITS 0 = launcher default);
    "REBAL,PIPE,SPLITS,BH,DT,BN,DK,NW,NS,QK" also overrides the dt tile config (EXL3_DSA_DEC_CFG);
    "1" = the preset below. Needs the dt path (EXL3_DSA_DEC_DT=1, the default)."""
    v = os.environ.get("EXL3_DEC_DSA_FAST", "0")
    if v == "0":
        return None
    if v == "1":
        # test box microbench : at R >= 2 (MTP verify, ~94 % of the
        # served decode calls) BH 32 x DT 256 halves the redundant QK of the 64 x 128 dt tile,
        # 12 splits + pipelined index chain + num_stages 2: 0.091 -> 0.057 ms. R = 1: no
        # variant beats dt (all within +-5 %), so R = 1 keeps it
        if R == 1:
            return None
        v = "0,1,12,32,256,32,64,4,2,1"
    return tuple(int(x) for x in v.split(","))


def _dsa_dec_dt_cfg():
    """r38: EXL3_DSA_DEC_DT=1 (default since r39; 0 = original split) selects _dsa_attn_split_dt_kernel for the batch-1 gathered
    latent decode split; EXL3_DSA_DEC_CFG = "BH,DT,BN,DK,NW,NS,QK_DYN". None = original."""
    if os.environ.get("EXL3_DSA_DEC_DT", "1") != "1":
        return None
    return tuple(int(x) for x in os.environ.get("EXL3_DSA_DEC_CFG", "64,128,32,64,4,1,1").split(","))


@triton.jit(do_not_specialize = [
    "k_sel", "out_w", "tail_off", "q_pos0", "s_pool", "s_out",
])
def _dsa_glue_expand_kernel(
    pool_ptr, out_ptr,
    k_sel, out_w, tail_off, q_pos0,
    s_pool, s_out,
    P: tl.constexpr, DO_TAIL: tl.constexpr, BLOCK: tl.constexpr = 128,
):
    """one program per (row, 128-col block) of the DSA indexer pool expand.

    Fuses the decode/verify torch glue after dsa_topk (mla_attn._indexer_topk_kpool):
    pool_idx * P + arange(P) expand, the >= 0 where-mask, the out copy, and the
    incomplete-tail-pool append. Pure int32 arithmetic, bitwise identical to the
    eager chain. out_ptr rows must be pre-filled with -1 beyond out_w.
    q_pos0 (int32, absolute position of row 0) keeps every value under 2**31,
    so int32 arithmetic matches the eager int64-then-.int() path exactly."""
    r = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    ok = cols < out_w
    exp_w = k_sel * P
    is_exp = cols < exp_w
    j = cols // P
    p = cols % P
    pid = tl.load(pool_ptr + r * s_pool + j, mask = ok & is_exp, other = -1)
    ev = tl.where(pid >= 0, pid * P + p, -1)
    if DO_TAIL:
        t = cols - tail_off
        q_pos = q_pos0 + r
        vis = q_pos + 1
        tc = vis % P
        ts = vis - tc
        tv = tl.where((t >= 0) & (t < tc), ts + t, -1)
        ev = tl.where((~is_exp) & ok, tv, ev)
    tl.store(out_ptr + r * s_out + cols, ev, mask = ok)


def dsa_glue_expand(pool_idx, out_slab, P, q_pos0, do_tail):
    """fused pool-expand + tail-append for one _indexer_topk_kpool slab.

    pool_idx (rows, k_sel) int32 top-k pool ids; out_slab (rows, k_pad) int32 slice
    of the -1-filled selection; q_pos0 = host_seqlens[b] + r0. Returns the expand
    width (k_sel * P, eager base_cols semantics). EXL3_DSA_GLUE_CHECK=1 also runs
    the eager chain and raises on any bitwise mismatch (gates only, default off)."""
    rows, k_sel = pool_idx.shape
    assert pool_idx.dtype == torch.int32 and out_slab.dtype == torch.int32
    assert pool_idx.stride(1) == 1 and out_slab.stride(1) == 1
    assert 0 < q_pos0 + rows < 1 << 30
    exp_w = k_sel * P
    tail_cols = (P - 1) if do_tail else 0
    out_w = exp_w + tail_cols
    if rows == 0 or out_w <= 0:
        return exp_w
    dev = pool_idx.device
    if dsa_glue_check:
        pre = out_slab.clone()
    with torch.cuda.device(dev):
        _dsa_glue_expand_kernel[(rows, triton.cdiv(out_w, 128))](
            pool_idx, out_slab, k_sel, out_w, exp_w, int(q_pos0),
            pool_idx.stride(0), out_slab.stride(0),
            P = P, DO_TAIL = bool(do_tail),
        )
    if dsa_glue_check:
        ref_e = (pool_idx.unsqueeze(-1) * P +
                 torch.arange(P, dtype = torch.int32, device = dev)).flatten(1)
        ref_e = torch.where(
            pool_idx.unsqueeze(-1).expand(-1, -1, P).flatten(1) >= 0,
            ref_e, ref_e.new_full((), -1))
        ref = pre.clone()
        ref[:, :ref_e.shape[1]] = ref_e
        if do_tail:
            q_pos = int(q_pos0) + torch.arange(rows, device = dev)
            vis = q_pos + 1
            tc = vis.remainder(P)
            ts = vis - tc
            toff = torch.arange(P - 1, device = dev)
            tail = ts.unsqueeze(1) + toff.unsqueeze(0)
            tail = torch.where(toff.unsqueeze(0) < tc.unsqueeze(1),
                               tail, tail.new_full((), -1))
            ref[:, exp_w:exp_w + P - 1] = tail.int()
        bad = (ref[:, :out_w] != out_slab[:, :out_w])
        if bool(bad.any().item()):
            r, c = bad.nonzero()[0].tolist()
            raise AssertionError(
                f"dsa_glue_expand mismatch at ({r},{c}): fused {out_slab[r, c].item()} "
                f"vs eager {ref[r, c].item()} (rows={rows} k_sel={k_sel} P={P} q_pos0={q_pos0})")
    return exp_w
