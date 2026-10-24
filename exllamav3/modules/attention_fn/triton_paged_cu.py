"""
CUDA-only paged decode attention (flash-decoding split + combine kernels).

Ported from upstream exllamav3 1.6.0 (MIT licence, Turboderp), modules/attention_fn/triton_paged.py
(_paged_attn_decode_split_kernel, _paged_attn_decode_combine_kernel, decode_row_layout, combine_subtiles,
paged_attn_triton_decode, _paged_kv_update_rows_kernel) and modules/attention_fn/smem.py (the shared-memory
ladder, trimmed to what the decode launcher uses). Changes from 1.6.0:
  - head_dim == v_head_dim (Kyojin has no asymmetric V head dim): the combine kernel's V_DIM is passed as head_dim
  - split_batch (DFlash verify rows, triton_paged.paged_attn_triton_decode_rows) is kept
  - ATTN_SPLIT_WARPS_STAGES / attn_decode_config (util/backend.py upstream) are inlined for CUDA: (4, 2)

What the 1.6 kernels change against the Kyojin ones in triton_paged.py:
  - sliding-window layers split only the keys >= (first query position - window) across programs
  - a kv head's (query position, sibling q head) rows are packed densely (decode_row_layout)
  - the combine kernel walks the splits S_BLK at a time and is sub-tiled over grid axis 1
  - the eager KV append is row-tiled

triton_paged.py dispatches here only when torch.version.hip is falsy and EXL3_ATTN16 != "0"; ROCm keeps calling
the original kernels there. bc_attn.py (CUDA only) compiles its decode kernels from this module under the same switch.
The QSA sparse kernels keep reducing through triton_paged._paged_attn_decode_combine_kernel (their partial layout).
"""

import math
import os

import torch
import triton
import triton.language as tl

from .triton_paged import (
    _get_h32, _rot_h32, _qc_load_kt, _qc_load_v,
    _normalize_window, _check_tensor, _prep_sinks, _same_device,
)

# The EXL3_ATTN16 A/B switch lives in triton_paged.ATTN16 (read once at import)


# ---- shared-memory ladder (upstream smem.py) ------------------------------------------------------------------

_limit = {}
_env_limit = int(os.environ.get("EXL3_TRITON_SMEM_LIMIT", "0") or "0")
_debug = bool(os.environ.get("EXL3_TRITON_SMEM_DEBUG"))
_picks = {}


class NoFittingConfig(RuntimeError):
    """No candidate in a ladder fits the device's shared memory."""


def _dev_index(device) -> int:
    idx = device.index if hasattr(device, "index") else device
    return torch.cuda.current_device() if idx is None else idx


def smem_limit(device) -> int:
    """Dynamic shared memory per block the device grants (the opt-in maximum), in bytes."""
    idx = _dev_index(device)
    lim = _limit.get(idx)
    if lim is None:
        props = torch.cuda.get_device_properties(idx)
        lim = getattr(props, "shared_memory_per_block_optin", 0)
        if not lim:
            lim = 64 * 1024 if props.major < 8 else 96 * 1024
        if _env_limit:
            lim = min(lim, _env_limit)
        _limit[idx] = lim
    return lim


def pick_config(device, name: str, key, candidates, probe):
    """First config in `candidates` whose compiled kernel fits the device's shared memory (cached per key)."""
    idx = _dev_index(device)
    limit = smem_limit(idx)
    full_key = (torch.cuda.get_device_capability(idx), limit, name, key)
    cfg = _picks.get(full_key, ...)
    if cfg is None:
        raise NoFittingConfig(f"{name}: no tile configuration fits the device's {limit} B of dynamic shared memory")
    if cfg is not ...:
        return cfg
    need = None
    for cand in candidates:
        need = probe(cand)
        if _debug:
            print(f" -- smem: {name} {key} config {cand}: {need} B ({'fits' if need <= limit else 'over'} {limit} B)", flush = True)
        if need <= limit:
            _picks[full_key] = cand
            return cand
    _picks[full_key] = None
    raise NoFittingConfig(
        f"{name}: no tile configuration fits the device's {limit} B of dynamic shared memory "
        f"(the smallest candidate needs {need} B)"
    )


def shared_bytes(kernel, args, **kwargs) -> int:
    """Compile-only warmup of a JIT kernel with the launch's real arguments."""
    return kernel.run(*args, grid = (1,), warmup = True, **kwargs).metadata.shared


def halving_ladder(value: int, minimum: int = 16):
    """value, value / 2, ... down to `minimum`."""
    cands = [value]
    while value // 2 >= minimum:
        value //= 2
        cands.append(value)
    return cands


def attn_decode_config(device, hd_pad: int, qc_block_rows: int | None = None) -> tuple[int, int, int]:
    """(block_n, num_warps, num_stages) for the decode split kernel (upstream util/backend.py, CUDA branch).
    hd_pad 512 (Gemma-4 global layers): 32 keys x 1 stage instead of 16 x 2, with the 16-row programs of decode_row_layout
    (RTX 3090, 6-bit cache, 8K-32K keys: x1.35-1.5 at 1 row, x2.8 at 9 verify rows together with the row cap; 2 or 3 stages
    give back most of the 1-row gain, 64 keys lose).
    Quantized caches with HD_PAD <= 256 (qc_block_rows = the program's BLOCK_ROWS): the same budget, a kv tile of
    2^18 / (BLOCK_ROWS * HD_PAD) keys with one stage. The quantized split loop is bound by per-tile work (barriers, row
    reductions, layout conversions), not by residency or DRAM bandwidth, so the widest tile that stays in registers wins
    (RTX 3090: Qwen3.8 KV6, hd 256, x0.86 at 64K context and x0.93 at 8 verify rows; Gemma-3 KV8, hd 128, x0.82)"""
    if hd_pad >= 512:
        return 32, 4, 1
    if qc_block_rows is not None:
        return max(16, 262144 // (qc_block_rows * hd_pad)), 4, 1
    return max(16, 8192 // hd_pad), 4, 2


# ---- row-tiled KV append (upstream _paged_kv_update_rows_kernel) ----------------------------------------------

# Row-tiled variant of the append above for the eager paths. For one token, all kv heads form one contiguous
# n_kv_heads * head_dim row in both the source [bsz, len, n_kv, hd] and the cache [pages, page_size, n_kv, hd],
# so a program copies BLOCK_T tokens x the whole row with one page lookup per token, instead of one small program
# per (token, kv head). A pure copy: the cache contents are identical. _paged_kv_update_kernel stays for bc_attn,
# which AOT-compiles it for the graph path
@triton.jit
def _paged_kv_update_rows_kernel(
    k,
    v,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    num_pages_per_seq,
    kv_append_len,
    t_blocks,
    ROW: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // t_blocks
    t = (pid - batch * t_blocks) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < kv_append_len
    logical_t = tl.load(cache_seqlens + batch) + t
    page = logical_t // page_size
    phys = tl.load(block_table + batch * num_pages_per_seq + page, mask=tmask, other=0)
    dst_row = (phys.to(tl.int64) * page_size + (logical_t - page * page_size)) * ROW
    src_row = (batch * kv_append_len + t).to(tl.int64) * ROW
    for w0 in tl.static_range(0, ROW, BLOCK_W):
        w = w0 + tl.arange(0, BLOCK_W)
        m = tmask[:, None] & (w < ROW)[None, :]
        src = src_row[:, None] + w[None, :]
        dst = dst_row[:, None] + w[None, :]
        tl.store(k_cache + dst, tl.load(k + src, mask=m), mask=m)
        tl.store(v_cache + dst, tl.load(v + src, mask=m), mask=m)



_KV_UPDATE_BLOCK_T = 8
_KV_UPDATE_WARPS = 4


def _paged_kv_update_rows(k, v, k_cache, v_cache, block_table, cache_seqlens, num_pages_per_seq, bsz, kv_append_len,
                          n_kv_heads, page_size, head_dim):
    """Append kv_append_len new tokens per sequence to the paged cache (eager paths); contiguous tensors only,
    the caller falls back to the per-head kernel otherwise"""
    row = n_kv_heads * head_dim
    t_blocks = triton.cdiv(kv_append_len, _KV_UPDATE_BLOCK_T)
    _paged_kv_update_rows_kernel[(bsz * t_blocks,)](
        k, v, k_cache, v_cache, block_table, cache_seqlens, num_pages_per_seq, kv_append_len, t_blocks,
        row, page_size, _KV_UPDATE_BLOCK_T, min(1024, triton.next_power_of_2(row)),
        num_warps = _KV_UPDATE_WARPS,
    )



# ---- software-pipelined quantized tile loads ------------------------------------------------------------------
# The quantized split loop feeds loaded words to ALU unpacking, not to a dot, so Triton's loop pipeliner never overlaps
# the loads with compute (num_stages is moot). The kernel therefore issues the NEXT tile's raw words (and scales) before
# it expands and consumes the current one: _qp_* load, _qe_* expand. Same values as _qc_load_kt / _qc_load_v.

@triton.jit
def _qp_ld_k(qh, row_words, mask_n, pbase, W: tl.constexpr, BITS: tl.constexpr, head_dim: tl.constexpr, HD_PAD: tl.constexpr):
    WPH: tl.constexpr = head_dim * W // 32
    WPH_PAD: tl.constexpr = HD_PAD * W // 32
    garr = tl.arange(0, WPH_PAD)
    cols = (garr // W) * BITS + pbase + (garr % W)
    if WPH_PAD > WPH:
        return tl.load(qh + row_words[None, :] + cols[:, None], mask = mask_n[None, :] & (garr < WPH)[:, None], other = 0)
    return tl.load(qh + row_words[None, :] + cols[:, None], mask = mask_n[None, :], other = 0)


@triton.jit
def _qp_ld_v(qh, row_words, mask_n, pbase, W: tl.constexpr, BITS: tl.constexpr, head_dim: tl.constexpr, HD_PAD: tl.constexpr):
    WPH: tl.constexpr = head_dim * W // 32
    WPH_PAD: tl.constexpr = HD_PAD * W // 32
    garr = tl.arange(0, WPH_PAD)
    cols = (garr // W) * BITS + pbase + (garr % W)
    if WPH_PAD > WPH:
        return tl.load(qh + row_words[:, None] + cols[None, :], mask = mask_n[:, None] & (garr < WPH)[None, :], other = 0)
    return tl.load(qh + row_words[:, None] + cols[None, :], mask = mask_n[:, None], other = 0)


@triton.jit
def _qp_planes_k(qwords, scales, tok_rows, kv_head, mask_n,
                 BITS: tl.constexpr, n_kv_heads: tl.constexpr, head_dim: tl.constexpr, HD_PAD: tl.constexpr):
    """Raw (HD_PAD-orientation) plane word tiles of one K tile + its scales; absent planes are (1, 1) zeros."""
    GPT: tl.constexpr = n_kv_heads * head_dim // 32
    G: tl.constexpr = head_dim // 32
    G_PAD: tl.constexpr = HD_PAD // 32
    qh = qwords + kv_head * (G * BITS)
    row_words = tok_rows * (GPT * BITS)
    z = tl.zeros((1, 1), tl.int32)
    w8 = z
    w4 = z
    w2 = z
    w1 = z
    pbase = 0
    if BITS & 8:
        w8 = _qp_ld_k(qh, row_words, mask_n, pbase, 8, BITS, head_dim, HD_PAD)
        pbase += 8
    if BITS & 4:
        w4 = _qp_ld_k(qh, row_words, mask_n, pbase, 4, BITS, head_dim, HD_PAD)
        pbase += 4
    if BITS & 2:
        w2 = _qp_ld_k(qh, row_words, mask_n, pbase, 2, BITS, head_dim, HD_PAD)
        pbase += 2
    if BITS & 1:
        w1 = _qp_ld_k(qh, row_words, mask_n, pbase, 1, BITS, head_dim, HD_PAD)
    garr = tl.arange(0, G_PAD)
    sc = tl.load(scales + tok_rows[None, :] * GPT + (kv_head * G + garr)[:, None],
                 mask = mask_n[None, :] & (garr < G)[:, None], other = 0.0)
    return w8, w4, w2, w1, sc


@triton.jit
def _qp_planes_v(qwords, scales, tok_rows, kv_head, mask_n,
                 BITS: tl.constexpr, n_kv_heads: tl.constexpr, head_dim: tl.constexpr, HD_PAD: tl.constexpr):
    GPT: tl.constexpr = n_kv_heads * head_dim // 32
    G: tl.constexpr = head_dim // 32
    G_PAD: tl.constexpr = HD_PAD // 32
    qh = qwords + kv_head * (G * BITS)
    row_words = tok_rows * (GPT * BITS)
    z = tl.zeros((1, 1), tl.int32)
    w8 = z
    w4 = z
    w2 = z
    w1 = z
    pbase = 0
    if BITS & 8:
        w8 = _qp_ld_v(qh, row_words, mask_n, pbase, 8, BITS, head_dim, HD_PAD)
        pbase += 8
    if BITS & 4:
        w4 = _qp_ld_v(qh, row_words, mask_n, pbase, 4, BITS, head_dim, HD_PAD)
        pbase += 4
    if BITS & 2:
        w2 = _qp_ld_v(qh, row_words, mask_n, pbase, 2, BITS, head_dim, HD_PAD)
        pbase += 2
    if BITS & 1:
        w1 = _qp_ld_v(qh, row_words, mask_n, pbase, 1, BITS, head_dim, HD_PAD)
    garr = tl.arange(0, G_PAD)
    sc = tl.load(scales + tok_rows[:, None] * GPT + (kv_head * G + garr)[None, :],
                 mask = mask_n[:, None] & (garr < G)[None, :], other = 0.0)
    return w8, w4, w2, w1, sc


@triton.jit
def _qe_k(w8, w4, w2, w1, sc, BITS: tl.constexpr, head_dim: tl.constexpr, HD_PAD: tl.constexpr):
    """(HD_PAD, BLOCK_N) fp16 tile from _qp_planes_k output; bit-identical to _qc_load_kt."""
    G_PAD: tl.constexpr = HD_PAD // 32
    raw = tl.zeros((1, 1), tl.int32)
    first = True
    if BITS & 8:
        raw = _qe_plane_k(w8, 8, HD_PAD)
        first = False
    if BITS & 4:
        p = _qe_plane_k(w4, 4, HD_PAD)
        raw = p if first else (raw << 4) | p
        first = False
    if BITS & 2:
        p = _qe_plane_k(w2, 2, HD_PAD)
        raw = p if first else (raw << 2) | p
        first = False
    if BITS & 1:
        p = _qe_plane_k(w1, 1, HD_PAD)
        raw = p if first else (raw << 1) | p
    scx = tl.reshape(tl.broadcast_to(sc[:, None, :], (G_PAD, 32, sc.shape[1])), (HD_PAD, sc.shape[1]))
    mh = (1 << (BITS - 1)) - 0.5
    inv_m = 1.0 / (1 << (BITS - 1))
    return ((raw.to(tl.float32) - mh) * (scx.to(tl.float32) * inv_m)).to(tl.float16)


@triton.jit
def _qe_v(w8, w4, w2, w1, sc, BITS: tl.constexpr, head_dim: tl.constexpr, HD_PAD: tl.constexpr):
    """(BLOCK_N, HD_PAD) fp16 tile from _qp_planes_v output; bit-identical to _qc_load_v."""
    G_PAD: tl.constexpr = HD_PAD // 32
    raw = tl.zeros((1, 1), tl.int32)
    first = True
    if BITS & 8:
        raw = _qe_plane_v(w8, 8, HD_PAD)
        first = False
    if BITS & 4:
        p = _qe_plane_v(w4, 4, HD_PAD)
        raw = p if first else (raw << 4) | p
        first = False
    if BITS & 2:
        p = _qe_plane_v(w2, 2, HD_PAD)
        raw = p if first else (raw << 2) | p
        first = False
    if BITS & 1:
        p = _qe_plane_v(w1, 1, HD_PAD)
        raw = p if first else (raw << 1) | p
    scx = tl.reshape(tl.broadcast_to(sc[:, :, None], (sc.shape[0], G_PAD, 32)), (sc.shape[0], HD_PAD))
    mh = (1 << (BITS - 1)) - 0.5
    inv_m = 1.0 / (1 << (BITS - 1))
    return ((raw.to(tl.float32) - mh) * (scx.to(tl.float32) * inv_m)).to(tl.float16)


@triton.jit
def _qe_plane_k(w, W: tl.constexpr, HD_PAD: tl.constexpr):
    VPW: tl.constexpr = 32 // W
    nib = (w[:, None, :] >> (tl.arange(0, VPW) * W)[None, :, None]) & ((1 << W) - 1)
    return tl.reshape(nib, (HD_PAD, w.shape[1]))


@triton.jit
def _qe_plane_v(w, W: tl.constexpr, HD_PAD: tl.constexpr):
    VPW: tl.constexpr = 32 // W
    nib = (w[:, :, None] >> (tl.arange(0, VPW) * W)[None, None, :]) & ((1 << W) - 1)
    return tl.reshape(nib, (w.shape[0], HD_PAD))


# ---- decode kernels (upstream 1.6.0 triton_paged.py) ----------------------------------------------------------

@triton.jit
def _paged_attn_decode_split_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    out,
    partial_o,
    partial_ml,
    k_scales,
    v_scales,
    h32,
    split_len,           # runtime: derived from the block-table bound, changes as it grows
    num_pages_per_seq,   # runtime: block-table width can grow without recompiling
    num_splits,          # runtime: the grid may be launched wider (graph path); extra splits idle
    sinks,               # last runtime arg: the BC launch appends it after the patched ints
    QCK: tl.constexpr,
    QCV: tl.constexpr,
    q_len: tl.constexpr,
    kv_append_len: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    HD_PAD: tl.constexpr,      # power of two >= head_dim: tile width, zero-padded columns
    scale: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    SOFTCAP: tl.constexpr,
    FINAL: tl.constexpr,       # num_splits == 1: skip the combine pass, store directly to out
    HAS_SINKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash-decoding phase 1: one program per (batch, kv_head, row block, kv split). The rows of a
    kv head are its (query position, sibling q head) pairs packed densely (decode_row_layout), so a
    program reads and dequantizes each K/V tile once for up to BLOCK_ROWS of them."""
    pid = tl.program_id(0)
    split = tl.program_id(1)

    group_size = n_q_heads // n_kv_heads
    h_blocks = tl.cdiv(q_len * group_size, BLOCK_ROWS)
    h_block = pid % h_blocks
    bh = pid // h_blocks
    batch = bh // n_kv_heads
    kv_head = bh - batch * n_kv_heads

    rows = tl.arange(0, BLOCK_ROWS)
    row_idx = h_block * BLOCK_ROWS + rows
    row_q = row_idx % q_len
    row_h_local = row_idx // q_len
    q_head = kv_head * group_size + row_h_local
    valid_row = row_idx < q_len * group_size

    offs_d = tl.arange(0, HD_PAD)
    d_mask = offs_d < head_dim
    q_base = ((batch * q_len + row_q) * n_q_heads + q_head) * head_dim
    q_tile = tl.load(q + q_base[:, None] + offs_d[None, :], mask=valid_row[:, None] & d_mask[None, :], other=0.0)
    if QCK > 0:
        q_tile = _rot_h32(q_tile, h32, BLOCK_ROWS, HD_PAD)

    total_k_len = tl.load(cache_seqlens + batch) + kv_append_len
    q_abs = total_k_len - q_len + row_q

    if WINDOW_LEFT >= 0:
        # Sliding-window layers only need keys >= (first query position - window): split that range, not the whole
        # sequence, across the programs. Splitting the sequence and skipping the tiles below the window leaves the
        # window to the one or two splits that cover it (10x slower at 64K than at 16K for a 2048 window). The
        # skipped keys carry exactly zero weight, so the result is unchanged
        w_lo = tl.maximum(total_k_len - q_len - WINDOW_LEFT, 0)
        w_lo = (w_lo // BLOCK_N) * BLOCK_N
        w_span = tl.cdiv(tl.cdiv(total_k_len - w_lo, num_splits), BLOCK_N) * BLOCK_N
        n_start = w_lo + split * w_span
        n_end = tl.minimum(n_start + w_span, total_k_len)
    else:
        n_start = split * split_len
        n_end = tl.minimum(n_start + split_len, total_k_len)

    m = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_ROWS,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_ROWS, HD_PAD), tl.float32)

    # quantized K and V: software-pipelined loads (see _qp_planes_k); the dummies keep the loop-carried names defined
    z0 = tl.zeros((1, 1), tl.int32)
    zs = tl.zeros((1, 1), tl.float16)
    kw8 = z0
    kw4 = z0
    kw2 = z0
    kw1 = z0
    vw8 = z0
    vw4 = z0
    vw2 = z0
    vw1 = z0
    ksc = zs
    vsc = zs
    if QCK > 0 and QCV > 0:
        offs_p = n_start + tl.arange(0, BLOCK_N)
        page_p = offs_p // page_size
        phys_p = tl.load(block_table + batch * num_pages_per_seq + page_p, mask=offs_p < n_end, other=0)
        rows_p = phys_p * page_size + (offs_p - page_p * page_size)
        kw8, kw4, kw2, kw1, ksc = _qp_planes_k(k_cache, k_scales, rows_p, kv_head, offs_p < n_end, QCK, n_kv_heads, head_dim, HD_PAD)
        vw8, vw4, vw2, vw1, vsc = _qp_planes_v(v_cache, v_scales, rows_p, kv_head, offs_p < n_end, QCV, n_kv_heads, head_dim, HD_PAD)

    for n0 in range(n_start, n_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        page = offs_n // page_size
        page_off = offs_n - page * page_size
        phys = tl.load(
            block_table + batch * num_pages_per_seq + page,
            mask=offs_n < n_end,
            other=0,
        )

        if QCK > 0 and QCV > 0:
            offs_x = offs_n + BLOCK_N
            page_x = offs_x // page_size
            phys_x = tl.load(block_table + batch * num_pages_per_seq + page_x, mask=offs_x < n_end, other=0)
            rows_x = phys_x * page_size + (offs_x - page_x * page_size)
            nk8, nk4, nk2, nk1, nksc = _qp_planes_k(k_cache, k_scales, rows_x, kv_head, offs_x < n_end, QCK, n_kv_heads, head_dim, HD_PAD)
            nv8, nv4, nv2, nv1, nvsc = _qp_planes_v(v_cache, v_scales, rows_x, kv_head, offs_x < n_end, QCV, n_kv_heads, head_dim, HD_PAD)

        if QCK > 0 and QCV > 0:
            k_tile = _qe_k(kw8, kw4, kw2, kw1, ksc, QCK, head_dim, HD_PAD)
        elif QCK > 0:
            tok_rows = phys * page_size + page_off
            k_tile = _qc_load_kt(k_cache, k_scales, tok_rows, kv_head, offs_d, offs_n < n_end, QCK, n_kv_heads, head_dim, HD_PAD)
        else:
            k_ptrs = k_cache + (((phys[None, :] * page_size + page_off[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
            if HD_PAD > head_dim:
                k_tile = tl.load(k_ptrs, mask=(offs_n[None, :] < n_end) & d_mask[:, None], other=0.0)
            else:
                k_tile = tl.load(k_ptrs, mask=offs_n[None, :] < n_end, other=0.0)
        scores = tl.dot(q_tile, k_tile) * scale
        if SOFTCAP > 0.0:
            scores_scaled = scores / SOFTCAP
            scores = (2.0 / (1.0 + tl.exp(-2.0 * scores_scaled)) - 1.0) * SOFTCAP

        valid = valid_row[:, None] & (offs_n[None, :] < n_end)
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= q_abs[:, None])
        if WINDOW_LEFT >= 0:
            valid = valid & (offs_n[None, :] >= q_abs[:, None] - WINDOW_LEFT)
        if WINDOW_RIGHT >= 0:
            valid = valid & (offs_n[None, :] <= q_abs[:, None] + WINDOW_RIGHT)
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
        l_new = l * alpha + tl.sum(p, axis=1)

        if QCK > 0 and QCV > 0:
            v_tile = _qe_v(vw8, vw4, vw2, vw1, vsc, QCV, head_dim, HD_PAD)
        elif QCV > 0:
            tok_rows_v = phys * page_size + page_off
            v_tile = _qc_load_v(v_cache, v_scales, tok_rows_v, kv_head, offs_d, offs_n < n_end, QCV, n_kv_heads, head_dim, HD_PAD)
        else:
            v_ptrs = v_cache + (((phys[:, None] * page_size + page_off[:, None]) * n_kv_heads + kv_head) * head_dim + offs_d[None, :])
            if HD_PAD > head_dim:
                v_tile = tl.load(v_ptrs, mask=(offs_n[:, None] < n_end) & d_mask[None, :], other=0.0)
            else:
                v_tile = tl.load(v_ptrs, mask=offs_n[:, None] < n_end, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new
        l = l_new
        if QCK > 0 and QCV > 0:
            kw8 = nk8
            kw4 = nk4
            kw2 = nk2
            kw1 = nk1
            ksc = nksc
            vw8 = nv8
            vw4 = nv4
            vw2 = nv2
            vw1 = nv1
            vsc = nvsc

    if FINAL:
        if HAS_SINKS:
            sink = tl.load(sinks + q_head, mask=valid_row, other=0.0).to(tl.float32)
            m_top = tl.maximum(m, sink)
            alpha_s = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_top))
            acc = acc * alpha_s[:, None]
            l = l * alpha_s + tl.exp(sink - m_top)
        out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
        if QCV > 0:
            out_tile = _rot_h32(out_tile, h32, BLOCK_ROWS, HD_PAD)
        out_base = ((batch * q_len + row_q) * n_q_heads + q_head) * head_dim
        tl.store(out + out_base[:, None] + offs_d[None, :], out_tile, mask=valid_row[:, None] & d_mask[None, :])
    else:
        if split < num_splits:
            po_base = (pid * num_splits + split) * BLOCK_ROWS * HD_PAD
            tl.store(partial_o + po_base + rows[:, None] * HD_PAD + offs_d[None, :], acc)
            ml_base = (pid * num_splits + split) * BLOCK_ROWS * 2
            tl.store(partial_ml + ml_base + rows * 2, m)
            tl.store(partial_ml + ml_base + rows * 2 + 1, l)


@triton.jit
def _paged_attn_decode_combine_kernel(
    partial_o,
    partial_ml,
    out,
    h32,
    num_splits,          # runtime
    sinks,               # last runtime arg: the BC launch appends it after the patched int
    QCV: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    q_len: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    HD_PAD: tl.constexpr,
    V_DIM: tl.constexpr,       # output lanes per head (< head_dim when V rides zero-padded in the cache)
    BLOCK_ROWS: tl.constexpr,
    ROWS_SUB: tl.constexpr,
    D_SUB: tl.constexpr,
):
    """Flash-decoding phase 2: reduce the per-split partial accumulators. The output row is
    written V_DIM wide per head: with an asymmetric V head dim the padded lanes are dropped
    here, so the result feeds o_proj directly. Grid axis 1 splits
    the program's (BLOCK_ROWS, HD_PAD) tile into (ROWS_SUB, D_SUB) sub-tiles so the serial
    walk over the splits runs on many CTAs (decode launches only a few programs)."""
    pid = tl.program_id(0)
    sub = tl.program_id(1)
    D_CHUNKS: tl.constexpr = HD_PAD // D_SUB
    r_c = sub // D_CHUNKS
    d_c = sub - r_c * D_CHUNKS

    group_size = n_q_heads // n_kv_heads
    h_blocks = tl.cdiv(q_len * group_size, BLOCK_ROWS)
    h_block = pid % h_blocks
    bh = pid // h_blocks
    batch = bh // n_kv_heads
    kv_head = bh - batch * n_kv_heads

    # Same dense (query, head) rows as the split kernel
    rows = r_c * ROWS_SUB + tl.arange(0, ROWS_SUB)
    row_idx = h_block * BLOCK_ROWS + rows
    row_q = row_idx % q_len
    row_h_local = row_idx // q_len
    q_head = kv_head * group_size + row_h_local
    valid_row = row_idx < q_len * group_size

    offs_d = d_c * D_SUB + tl.arange(0, D_SUB)
    d_mask = offs_d < head_dim

    # Splits are walked S_BLK at a time as one masked tile load per pass: a scalar loop with a
    # runtime trip count and no pipelining serializes a full load latency per split (65 splits
    # at a 2K sparse context cost 20 us), a tile load amortizes it
    S_BLK: tl.constexpr = 16
    offs_s = tl.arange(0, S_BLK)
    m_max = tl.full((ROWS_SUB,), -float("inf"), tl.float32)
    for s0 in range(0, num_splits, S_BLK):
        sidx = s0 + offs_s
        s_mask = sidx < num_splits
        ml_idx = ((pid * num_splits + sidx)[:, None] * BLOCK_ROWS + rows[None, :]) * 2
        m_s = tl.load(partial_ml + ml_idx, mask=s_mask[:, None], other=-float("inf"))
        m_max = tl.maximum(m_max, tl.max(m_s, axis=0))

    if HAS_SINKS:
        # Learned per-head sink joins the softmax denominator at the final reduction
        sink = tl.load(sinks + q_head, mask=valid_row, other=0.0).to(tl.float32)
        m_max = tl.maximum(m_max, sink)

    l_sum = tl.zeros((ROWS_SUB,), tl.float32)
    acc = tl.zeros((ROWS_SUB, D_SUB), tl.float32)
    m_safe = tl.where(m_max == -float("inf"), 0.0, m_max)
    for s0 in range(0, num_splits, S_BLK):
        sidx = s0 + offs_s
        s_mask = sidx < num_splits
        ml_idx = ((pid * num_splits + sidx)[:, None] * BLOCK_ROWS + rows[None, :]) * 2
        m_s = tl.load(partial_ml + ml_idx, mask=s_mask[:, None], other=-float("inf"))
        l_s = tl.load(partial_ml + ml_idx + 1, mask=s_mask[:, None], other=0.0)
        w = tl.where(m_s == -float("inf"), 0.0, tl.exp(m_s - m_safe[None, :]))
        po_idx = (pid * num_splits + sidx)[:, None, None] * (BLOCK_ROWS * HD_PAD) \
            + rows[None, :, None] * HD_PAD + offs_d[None, None, :]
        o_s = tl.load(partial_o + po_idx, mask=s_mask[:, None, None], other=0.0)
        acc += tl.sum(o_s * w[:, :, None], axis=0)
        l_sum += tl.sum(l_s * w, axis=0)

    if HAS_SINKS:
        l_sum += tl.exp(sink - m_safe)
    out_tile = acc / tl.where(l_sum[:, None] == 0.0, 1.0, l_sum[:, None])
    if QCV > 0:
        out_tile = _rot_h32(out_tile, h32, ROWS_SUB, D_SUB)   # 32-wide groups: D_SUB % 32 == 0
    out_base = ((batch * q_len + row_q) * n_q_heads + q_head) * V_DIM
    tl.store(out + out_base[:, None] + offs_d[None, :], out_tile, mask=valid_row[:, None] & (offs_d < V_DIM)[None, :])


def decode_row_layout(q_len: int, group_size: int, hd_pad: int) -> tuple[int, int]:
    """(BLOCK_ROWS, row blocks per kv head) for the decode split / combine kernels.

    A kv head's rows are its q_len * group_size (query position, sibling q head) pairs, packed densely
    and cut into programs of up to 32 rows, each of which reads and dequantizes the head's K/V tiles
    once. (Padding query positions up to a power of two and giving a program 16 / that many heads used
    to split a 6-head group across three programs for a 5-token verify step, tripling the tile work.)
    32 rows pays off at every head_dim up to 256 on Ampere, Blackwell and RDNA3 alike, the fp32
    accumulator notwithstanding, since the tile loads dominate; at least 16 rows, the dots' minimum.
    At head_dim 512 the 32 x 512 fp32 accumulator of a 4-warp program spills: 16-row programs are x1.8-1.95 faster
    on a 9-row verify (72 rows per kv head) although each one re-reads the head's K/V (RTX 3090, 6-bit cache)."""
    rows = q_len * group_size
    block_rows = max(16, min(16 if hd_pad >= 512 else 32, triton.next_power_of_2(rows)))
    return block_rows, triton.cdiv(rows, block_rows)


def combine_subtiles(block_rows: int, hd_pad: int) -> tuple[int, int]:
    """(ROWS_SUB, D_SUB) for the combine kernel: the smallest sub-tile whose h32 rotation still
    forms a >= 16-row tl.dot (ROWS_SUB * D_SUB >= 512), D_SUB a multiple of 32."""
    d_sub = min(128, hd_pad)
    rows_sub = min(block_rows, max(1, 512 // d_sub))
    while rows_sub * d_sub < 512 and d_sub < hd_pad:
        d_sub *= 2
    if rows_sub * d_sub < 512:
        rows_sub, d_sub = block_rows, hd_pad
    return rows_sub, d_sub




# ---- launcher (upstream paged_attn_triton_decode; Kyojin's split_batch kept) -----------------------------------

_decode_sm_count = {}


_QDEC_CHECK = os.environ.get("EXL3_QDEC_CHECK") == "1"
_QDEC_STATS = []


def paged_attn_triton_decode_cu(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: int | tuple[int, int] | None = None,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_n: int | None = None,
    num_splits: int | None = None,
    max_kv_len: int | None = None,
    qc: tuple | None = None,            # (k_scales, v_scales, k_bits, v_bits): caches are packed int32
    pre_appended_len: int = 0,          # new tokens already written to the cache; count but don't append
    n_kv_heads_override: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_batch: int | None = None,     # batch size the split count is derived from (default: bsz)
) -> torch.Tensor:
    """Flash-decoding paged attention for short queries: the kv sequence is split across
    programs (sized from the block table, so no host sync on cache_seqlens) and reduced in a
    second pass. GQA sibling q heads share K/V tiles within a program."""
    _check_tensor("q", q)
    _check_tensor("block_table", block_table, None)
    _check_tensor("cache_seqlens", cache_seqlens, None)

    bsz, q_len, n_q_heads, head_dim = q.shape
    if qc is None:
        _check_tensor("k_cache", k_cache)
        _check_tensor("v_cache", v_cache)
        if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError("q, k_cache and v_cache must be rank-4 tensors")
        _, page_size, n_kv_heads, cache_dim = k_cache.shape
        if v_cache.shape != k_cache.shape:
            raise ValueError("v_cache must have the same shape as k_cache")
        if cache_dim != head_dim:
            raise ValueError("q and cache head dimensions must match")
    else:
        page_size = k_cache.shape[1]
        n_kv_heads = n_kv_heads_override
    if not _same_device(q, k_cache, v_cache, block_table, cache_seqlens):
        raise ValueError("q, caches, block_table and cache_seqlens must be on the same CUDA device")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")
    if head_dim > 512:
        raise ValueError("paged_attn_triton_decode supports head_dim <= 512")
    if qc is not None and head_dim % 32 != 0:
        raise ValueError("quantized caches need head_dim to be a multiple of 32")
    hd_pad = triton.next_power_of_2(head_dim)   # tile width; non-power-of-two dims are zero-padded
    if q_len > 16:
        raise ValueError("paged_attn_triton_decode supports q_len <= 16")

    kv_append_len = pre_appended_len
    if k is not None or v is not None:
        if k is None or v is None:
            raise ValueError("k and v must be provided together")
        _check_tensor("k", k)
        _check_tensor("v", v)
        if k.shape[:1] != (bsz,) or k.shape[2:] != (n_kv_heads, head_dim):
            raise ValueError("k/v shape must be [batch, seqlen_new, kv_heads, head_dim]")
        kv_append_len = k.shape[1]

    if out is None:
        out = torch.empty_like(q)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    window_left, window_right = _normalize_window(window_size)
    sinks, has_sinks = _prep_sinks(sinks, n_q_heads, q)

    if qc is not None:
        k_scales, v_scales, qck, qcv = qc
        h32 = _get_h32(q.device)
    else:
        k_scales, v_scales, qck, qcv = q, q, 0, 0
        h32 = q

    # K + V tiles in smem across num_stages; on small-smem devices the ladder halves the kv tile
    qc_rows = decode_row_layout(q_len, n_q_heads // n_kv_heads, hd_pad)[0] if qc is not None else None
    cfg_block_n, cfg_warps, cfg_stages = attn_decode_config(q.device, hd_pad, qc_rows)
    if num_warps is None:
        num_warps = cfg_warps
    if num_stages is None:
        num_stages = cfg_stages
    candidates = halving_ladder(cfg_block_n) if block_n is None else [block_n]

    group_size = n_q_heads // n_kv_heads
    block_rows, h_blocks = decode_row_layout(q_len, group_size, hd_pad)
    num_pages_per_seq = block_table.shape[1]

    # Upper bound on kv length: caller-provided hint, else from the block table shape
    # (cache_seqlens stays on the device; no sync). A loose bound wastes split parallelism when
    # the table is allocated much larger than the live sequence
    max_k_len = num_pages_per_seq * page_size + kv_append_len
    if max_kv_len is not None:
        max_k_len = min(max_k_len, max_kv_len + kv_append_len)

    programs = bsz * n_kv_heads * h_blocks
    dev = q.device.index
    if dev not in _decode_sm_count:
        _decode_sm_count[dev] = torch.cuda.get_device_properties(q.device).multi_processor_count
    target = 2 * _decode_sm_count[dev]
    programs_split = programs if split_batch is None else programs // bsz * split_batch
    req_splits = num_splits

    if qc is not None and qck > 0 and qcv > 0:
        from . import qdec_cuda
        qcfg = qdec_cuda.make_cfg(q_len, n_q_heads, n_kv_heads, head_dim, qck, qcv, page_size, causal, softmax_scale,
                                  block_rows, window_left, window_right, softcap, kv_append_len)
        if qcfg is not None and (h_blocks == 1 or qcfg.get("rows_kernel")):
            if qcfg.get("rows_kernel"):
                # many-row kernel (cu/qdec_rows.cu): all of a kv head's (query, head) rows in one program
                block_rows, h_blocks = qcfg["block_rows"], 1
                programs = bsz * n_kv_heads
                programs_split = programs if split_batch is None else programs // bsz * split_batch
            # tensor-core CUDA split kernel on the packed words (cu/qdec.cu): one CTA per SM-share, 16-token slabs
            qsplits = req_splits if req_splits is not None else \
                max(1, min(qdec_cuda.split_cap(qcfg, q.device, programs_split), triton.cdiv(max_k_len, 16), 128))
            qsplit_len = triton.cdiv(triton.cdiv(max_k_len, qsplits), 16) * 16
            partial_o = torch.empty(programs * qsplits * block_rows * hd_pad, dtype = torch.float32, device = q.device)
            partial_ml = torch.empty(programs * qsplits * block_rows * 2, dtype = torch.float32, device = q.device)
            with torch.cuda.device(q.device):
                qdec_cuda.get_kernel(qcfg).launch(
                    (programs, qsplits),
                    [q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), block_table.data_ptr(), cache_seqlens.data_ptr(),
                     out.data_ptr(), partial_o.data_ptr(), partial_ml.data_ptr(), k_scales.data_ptr(), v_scales.data_ptr(),
                     h32.data_ptr()],
                    [qsplit_len, num_pages_per_seq, qsplits])
                rows_sub, d_sub = combine_subtiles(block_rows, hd_pad)
                _paged_attn_decode_combine_kernel[(programs, (block_rows // rows_sub) * (hd_pad // d_sub))](
                    partial_o, partial_ml, out, h32,
                    qsplits, sinks, qcv, has_sinks, q_len, n_q_heads, n_kv_heads, head_dim, hd_pad, head_dim,
                    block_rows, rows_sub, d_sub,
                    num_warps=4, num_stages=1,
                )
            if _QDEC_CHECK:
                ref = paged_attn_triton_decode_cu.__wrapped_ref__(q, k, v, k_cache, v_cache, block_table, cache_seqlens, causal,
                    softmax_scale, window_size, softcap, sinks, None, block_n, num_splits, max_kv_len, qc, pre_appended_len,
                    n_kv_heads_override, num_warps, num_stages, split_batch)
                d = (ref.float() - out.float()).abs().max().item(); mx = ref.float().abs().max().item()
                _QDEC_STATS.append((d, mx, d / max(mx, 1e-9), q_len, head_dim, qck, qcv))
            return out

    def prepare(block_n):
        """Split count, partial buffers and the split kernel's argument list for one kv tile."""
        splits = req_splits
        if splits is None:
            splits = max(1, min(target // programs_split, triton.cdiv(max_k_len, 1 * block_n), 128))
        split_len = triton.cdiv(triton.cdiv(max_k_len, splits), block_n) * block_n
        if splits > 1:
            partial_o = torch.empty(programs * splits * block_rows * hd_pad, dtype = torch.float32, device = q.device)
            partial_ml = torch.empty(programs * splits * block_rows * 2, dtype = torch.float32, device = q.device)
        else:
            partial_o = q   # unused
            partial_ml = q  # unused
        args = (
            q, k_cache, v_cache, block_table, cache_seqlens, out, partial_o, partial_ml,
            k_scales, v_scales, h32,
            split_len, num_pages_per_seq, splits, sinks,
            qck, qcv, q_len, kv_append_len, n_q_heads, n_kv_heads,
            page_size, head_dim, hd_pad, float(softmax_scale),
            bool(causal), int(window_left), int(window_right), float(softcap or 0.0),
            splits == 1, has_sinks, block_rows, block_n,
        )
        return args, splits, partial_o, partial_ml

    with torch.cuda.device(q.device):
        pick_key = (head_dim, hd_pad, qck, qcv, q_len, block_rows, h_blocks, bool(causal), window_left >= 0,
                    window_right >= 0, has_sinks, num_warps, num_stages, candidates[0])
        block_n = pick_config(
            q.device, "paged_attn_decode_cu", pick_key, candidates,
            lambda bn: shared_bytes(_paged_attn_decode_split_kernel, prepare(bn)[0],
                                    num_warps = num_warps, num_stages = num_stages))
        args, num_splits, partial_o, partial_ml = prepare(block_n)

        if k is not None and kv_append_len:
            if all(t.is_contiguous() for t in (k, v, k_cache, v_cache, block_table)):
                _paged_kv_update_rows(k, v, k_cache, v_cache, block_table, cache_seqlens, num_pages_per_seq, bsz,
                                      kv_append_len, n_kv_heads, page_size, head_dim)
            else:
                from .triton_paged import _paged_kv_update_kernel
                update_block_d = triton.next_power_of_2(head_dim)
                _paged_kv_update_kernel[(bsz * kv_append_len, n_kv_heads, triton.cdiv(head_dim, update_block_d))](
                    k, v, k_cache, v_cache, block_table, cache_seqlens,
                    num_pages_per_seq, kv_append_len, n_kv_heads, page_size, head_dim, update_block_d,
                    num_warps=2, num_stages=3,
                )

        _paged_attn_decode_split_kernel[(programs, num_splits)](
            *args, num_warps = num_warps, num_stages = num_stages,
        )

        if num_splits > 1:
            rows_sub, d_sub = combine_subtiles(block_rows, hd_pad)
            _paged_attn_decode_combine_kernel[(programs, (block_rows // rows_sub) * (hd_pad // d_sub))](
                partial_o, partial_ml, out, h32,
                num_splits, sinks, qcv, has_sinks, q_len, n_q_heads, n_kv_heads, head_dim, hd_pad, head_dim,
                block_rows, rows_sub, d_sub,
                num_warps=4, num_stages=1,
            )
    return out


if _QDEC_CHECK:
    import atexit
    _orig = paged_attn_triton_decode_cu

    def _ref_call(*a):
        from . import qdec_cuda
        qdec_cuda.ENABLED = False
        try:
            return _orig(*a)
        finally:
            qdec_cuda.ENABLED = True
    paged_attn_triton_decode_cu.__wrapped_ref__ = _ref_call

    @atexit.register
    def _report():
        if _QDEC_STATS:
            w = max(_QDEC_STATS, key = lambda r: r[2])
            print(f"QDEC_CHECK {len(_QDEC_STATS)} calls; worst rel diff {w[2]:.3e} (abs {w[0]:.3e}, ref max {w[1]:.3e}, q_len {w[3]}, hd {w[4]}, bits {w[5]}/{w[6]}); "
                  f"mean rel {sum(r[2] for r in _QDEC_STATS) / len(_QDEC_STATS):.3e}", flush = True)
