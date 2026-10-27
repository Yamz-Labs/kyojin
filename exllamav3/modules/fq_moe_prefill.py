"""Grouped MoE prefill for MiniCPM IQ4_XS experts.

One Triton launch dequantizes every expert matrix of a layer to fp16 scratch
(replacing 256 x 3 per-expert dequant + cuBLAS calls), then torch._grouped_mm
runs gate / up / down over the expert-sorted rows with device-side offsets
(no host sync). Same math as the per-expert path: fp16 weights, fp32 accum.
"""
import torch
import triton
import triton.language as tl

_KV_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
_kv_tables = {}
_scratch = {}


def _kv(dev):
    t = _kv_tables.get(dev)
    if t is None:
        t = _kv_tables[dev] = torch.tensor(_KV_IQ4NL, dtype=torch.float32, device=dev)
    return t


@triton.jit
def _dq_iq4xs_grouped(ptrs, kv, out, n_blocks, BLK: tl.constexpr):
    e = tl.program_id(1)
    sb = tl.program_id(0) * BLK + tl.arange(0, BLK)            # [BLK] super-block ids
    m = sb < n_blocks
    base = tl.load(ptrs + e).to(tl.pointer_type(tl.uint8))
    j = tl.arange(0, 256)                                       # element inside super-block
    ib = j >> 5
    jj = j & 31
    bp = base + sb.to(tl.int64) * 136                           # [BLK] block byte pointers
    b0 = tl.load(bp, mask=m, other=0).to(tl.uint16)
    b1 = tl.load(bp + 1, mask=m, other=0).to(tl.uint16)
    d = (b0 | (b1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)      # [BLK]
    h0 = tl.load(bp + 2, mask=m, other=0).to(tl.int32)
    h1 = tl.load(bp + 3, mask=m, other=0).to(tl.int32)
    sh = h0 | (h1 << 8)                                         # [BLK]
    sl = tl.load(bp[:, None] + 4 + (ib >> 1)[None, :], mask=m[:, None], other=0).to(tl.int32)
    ls = ((sl >> (4 * (ib & 1))[None, :]) & 0xF) | (((sh[:, None] >> (2 * ib)[None, :]) & 3) << 4)
    q = tl.load(bp[:, None] + 8 + (16 * ib + (jj & 15))[None, :], mask=m[:, None], other=0).to(tl.int32)
    nib = tl.where((jj & 16)[None, :] != 0, q >> 4, q & 0xF)
    dl = d[:, None] * (ls - 32).to(tl.float32)
    v = dl * tl.load(kv + nib)
    optr = out + (e * n_blocks + sb[:, None]).to(tl.int64) * 256 + j[None, :]
    tl.store(optr, v.to(tl.float16), mask=m[:, None])


def dequant_iq4xs_grouped(ptrs: torch.Tensor, out: torch.Tensor, n_blocks: int):
    """ptrs: int64 [E] device pointers to [n_blocks*136] byte payloads;
    out: fp16 contiguous [E * n_blocks * 256]."""
    E = ptrs.shape[0]
    BLK = 4
    grid = (triton.cdiv(n_blocks, BLK), E)
    _dq_iq4xs_grouped[grid](ptrs, _kv(ptrs.device), out, n_blocks, BLK=BLK, num_warps=4)


def get_scratch(dev, name, numel):
    t = _scratch.get((dev, name))
    if t is None or t.numel() < numel:
        _scratch.pop((dev, name), None)
        t = _scratch[(dev, name)] = torch.empty(numel, dtype=torch.half, device=dev)
    return t[:numel]


# ---------------------------------------------------------------------------
# Fused grouped GEMM: IQ4_XS weights decoded inside the K loop (no fp16 scratch)
# ---------------------------------------------------------------------------

@triton.jit
def _iq4_sb(blk, kv, nmask, BN: tl.constexpr):
    """Decode one IQ4_XS super-block per row of the tile: blk [BN] byte
    pointers. Returns (wl, wh), each [128, BN] fp16 k-major: wl[c] is the low
    nibble element of qs byte c (K index 32*(c//16)+c%16 inside the
    super-block), wh[c] the high-nibble element (same index + 16). Per-element
    op order matches fq_dq_iq4_xs_kernel."""
    c = tl.arange(0, 128)
    ib = tl.arange(0, 8)
    b0 = tl.load(blk, mask=nmask, other=0).to(tl.uint16)
    b1 = tl.load(blk + 1, mask=nmask, other=0).to(tl.uint16)
    d = (b0 | (b1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)   # [BN]
    h0 = tl.load(blk + 2, mask=nmask, other=0).to(tl.int32)
    h1 = tl.load(blk + 3, mask=nmask, other=0).to(tl.int32)
    sh = h0 | (h1 << 8)                                                 # [BN]
    sl = tl.load(blk[:, None] + 4 + (ib >> 1)[None, :], mask=nmask[:, None], other=0).to(tl.int32)  # [BN, 8]
    ls = ((sl >> (4 * (ib & 1))[None, :]) & 0xF) | (((sh[:, None] >> (2 * ib)[None, :]) & 3) << 4)
    dl = d[:, None] * (ls - 32).to(tl.float32)                          # [BN, 8]
    dl = tl.reshape(tl.broadcast_to(dl[:, :, None], (BN, 8, 16)), (BN, 128))
    q = tl.load(blk[:, None] + 8 + c[None, :], mask=nmask[:, None], other=0).to(tl.int32)   # [BN, 128]
    wl = (dl * tl.load(kv + (q & 0xF))).to(tl.float16)
    wh = (dl * tl.load(kv + (q >> 4))).to(tl.float16)
    return tl.trans(wl), tl.trans(wh)


@triton.jit
def _gmm_iq4(a_ptr, wptrs, kv, out_ptr, tile_e, tile_r0, offs_ptr, wrow_ptr,
             N, K, stride_am, stride_om, E,
             SWIGLU: tl.constexpr, SCALE_ROWS: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    """out[r, n] = sum_k A[r, k] * W_e[n, k] for the rows of tile t's expert.
    SWIGLU: wptrs holds gate ptrs [0:E] and up ptrs [E:2E]; out = silu(g)*u
    (g, u rounded to fp16 first, like the unfused path). SCALE_ROWS: multiply
    the fp16 result by wrow[r] (routing weight, fp16 mul). K is walked one
    256-wide super-block at a time in a permuted (nibble-split) order that is
    applied identically to A and W (sum order only): A arrives
    pre-permuted (see perm_cols; the SWIGLU epilogue writes its output permuted)."""
    pid_n = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.load(tile_e + t)
    if e >= E:
        return
    r0 = tl.load(tile_r0 + t)
    r_end = tl.load(offs_ptr + e)
    offs_m = r0 + tl.arange(0, BM)
    mmask = offs_m < r_end
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < N
    c = tl.arange(0, 128)
    kb = K // 256
    wb = tl.load(wptrs + e).to(tl.pointer_type(tl.uint8))
    bp_n = wb + offs_n.to(tl.int64) * (kb * 136)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if SWIGLU:
        wb2 = tl.load(wptrs + E + e).to(tl.pointer_type(tl.uint8))
        bp_n2 = wb2 + offs_n.to(tl.int64) * (kb * 136)
        acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    a_row = a_ptr + offs_m.to(tl.int64)[:, None] * stride_am
    for sb in range(0, kb):
        al = tl.load(a_row + (sb * 256 + c)[None, :], mask=mmask[:, None], other=0.0)
        ah = tl.load(a_row + (sb * 256 + 128 + c)[None, :], mask=mmask[:, None], other=0.0)
        wl, wh = _iq4_sb(bp_n + sb * 136, kv, nmask, BN)
        acc = tl.dot(al, wl, acc)
        acc = tl.dot(ah, wh, acc)
        if SWIGLU:
            wl2, wh2 = _iq4_sb(bp_n2 + sb * 136, kv, nmask, BN)
            acc2 = tl.dot(al, wl2, acc2)
            acc2 = tl.dot(ah, wh2, acc2)
    if SWIGLU:
        g = acc.to(tl.float16).to(tl.float32)
        u = acc2.to(tl.float16).to(tl.float32)
        res = (g / (1.0 + tl.exp(-g)) * u).to(tl.float16)
    else:
        res = acc.to(tl.float16)
        if SCALE_ROWS:
            wr = tl.load(wrow_ptr + offs_m, mask=mmask, other=0.0).to(tl.float32)
            res = (res.to(tl.float32) * wr[:, None]).to(tl.float16)
    if SWIGLU:
        # store the activation K-permuted (nibble-split) for the down GEMM's A loads
        s_ = offs_n & 255
        pos = tl.where((s_ & 16) != 0, 128 + (s_ >> 5) * 16 + (s_ & 15), (s_ >> 5) * 16 + (s_ & 15))
        ocol = (offs_n - s_) + pos
    else:
        ocol = offs_n
    o = out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + ocol[None, :]
    tl.store(o, res, mask=mmask[:, None] & nmask[None, :])


def build_tiles(counts: torch.Tensor, M: int, BM: int):
    """Sync-free tile map. counts: int64 [E] rows per expert (sorted layout).
    Returns (tile_e int32 [T], tile_r0 int32 [T], offs int32 [E]) with
    T = M // BM + E (static bound); unused tiles have tile_e == E."""
    E = counts.shape[0]
    dev = counts.device
    T = M // BM + E
    offs = counts.cumsum(0)
    starts = offs - counts
    tiles = (counts + BM - 1) // BM
    cum = tiles.cumsum(0)
    t = torch.arange(T, device=dev)
    te = torch.searchsorted(cum, t, right=True)                # first expert with cum > t; E if none
    tec = te.clamp(max=E - 1)
    local = t - (cum[tec] - tiles[tec])
    r0 = starts[tec] + local * BM
    return te.to(torch.int32), r0.to(torch.int32), offs.to(torch.int32)


def gmm_iq4(a, wptrs, out, tile_e, tile_r0, offs, N, K, E, swiglu=False, wrow=None, BM=64, BN=64,
            num_warps=4, num_stages=1):
    T = tile_e.shape[0]
    grid = (triton.cdiv(N, BN), T)
    _gmm_iq4[grid](a, wptrs, _kv(a.device), out, tile_e, tile_r0, offs,
                   wrow if wrow is not None else offs, N, K, a.stride(0), out.stride(0), E,
                   SWIGLU=swiglu, SCALE_ROWS=wrow is not None, BM=BM, BN=BN,
                   num_warps=num_warps, num_stages=num_stages)


def perm_cols(x: torch.Tensor) -> torch.Tensor:
    """K-permute [M, K] (K % 256 == 0) into the nibble-split order _gmm_iq4 expects."""
    M, K = x.shape
    return x.view(M, K // 256, 8, 2, 16).permute(0, 1, 3, 2, 4).reshape(M, K)
