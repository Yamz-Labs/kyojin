"""
TurboQuant KV cache layer ("tq"): per KV head vector, rotation + Lloyd-Max codebook + one fp16 norm.

Write: x -> y = R x (R = Hadamard x random signs, orthonormal), n = ||y||, codes = nearest Lloyd-Max centroid of y / n
(the codebook of one coordinate of a random unit vector in R^head_dim), stored scale s = n / ||c|| (norm correction: the
centroid vector c is re-normalised before scaling, so the reconstruction has exactly the norm of the input). Read:
x^ = R^T (s * c). The scale is ONE fp16 per head vector, so b-bit codes cost b + 16 / head_dim bits per value.

The cache lives in the ROTATED domain: R is orthonormal, so q . k = (R q) . (R k) and o = R^T (sum p (R v)). The attention
dispatcher rotates q, k and v on the way in and the output on the way out, so the paged pool, the KVWIN ring and every
kernel see rotated vectors; the Triton loaders expand a code with a codebook gather and a single per-head scale instead
of the per-32 midpoint grid. The packed code layout (32-value groups, power-of-two bit planes) is CacheLayer_quant's, so
copy_page, save / restore and the prefix disk cache move the same four tensors (codes + scales) as before.
"""
from __future__ import annotations
import math
import os
from functools import lru_cache
import torch
from typing_extensions import override
from ..constants import PAGE_SIZE
from .quant import CacheLayer_quant

TQ_SEED = 20240611
_REF_ROT = os.environ.get("EXL3_TQ_REF_ROT") == "1"           # torch GEMM rotation (A/B and debugging)
_REF_WRITE = os.environ.get("EXL3_TQ_REF_WRITE") == "1"     # torch reference writer (A/B and debugging)


@lru_cache(maxsize = None)
def codebook(dim: int, bits: int) -> torch.Tensor:
    """Lloyd-Max centroids (fp32 CPU, ascending) for a coordinate of a uniform random unit vector in R^dim."""
    from .tq_codebooks import CODEBOOKS
    assert (dim, bits) in CODEBOOKS, f"no TurboQuant codebook for head_dim {dim}, {bits} bits (have {sorted(CODEBOOKS)})"
    return torch.tensor(CODEBOOKS[dim, bits], dtype = torch.float32)


def supported(dim: int, bits: int) -> bool:
    from .tq_codebooks import CODEBOOKS
    return (dim, bits) in CODEBOOKS


@lru_cache(maxsize = None)
def rotation(dim: int) -> torch.Tensor:
    """M with y = x @ M.T = H diag(s) x / sqrt(dim): fp32 CPU, orthonormal."""
    h = torch.ones(1, 1, dtype = torch.float64)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    g = torch.Generator().manual_seed(TQ_SEED)
    s = torch.where(torch.rand(dim, generator = g) < 0.5, -1.0, 1.0).double()
    return ((h / math.sqrt(dim)) * s).float()


class TQState:
    """Per (head_dim, k_bits, v_bits, device) constants shared by all layers: rotation, codebooks, midpoints."""
    _cache = {}

    @classmethod
    def get(cls, dim: int, k_bits: int, v_bits: int, device) -> "TQState":
        key = (dim, k_bits, v_bits, str(device))
        if key not in cls._cache:
            cls._cache[key] = cls(dim, k_bits, v_bits, device)
        return cls._cache[key]

    def __init__(self, dim, k_bits, v_bits, device):
        self.dim = dim
        self.device = device
        m = rotation(dim)
        self.rot_t = m.t().contiguous().to(device, torch.float16)       # x @ rot_t = R x      (rotate)
        self.rot = m.contiguous().to(device, torch.float16)             # y @ rot   = R^T y    (un-rotate)
        self.rot_t32 = m.t().contiguous().to(device)
        self.rot32 = m.contiguous().to(device)
        self.cbk = codebook(dim, k_bits).to(device)
        self.cbv = codebook(dim, v_bits).to(device)
        self.cbk16 = self.cbk.half().contiguous()
        self.cbv16 = self.cbv.half().contiguous()
        self.midk = ((self.cbk[1:] + self.cbk[:-1]) / 2).contiguous()
        self.midv = ((self.cbv[1:] + self.cbv[:-1]) / 2).contiguous()
        self.eye32 = torch.eye(32, dtype = torch.float16, device = device)   # the kernels' H32 slot: rotation is already outside

    def rotate(self, *xs: torch.Tensor):
        """fp16 tensors [..., dim] -> R x, same shapes (one Triton launch for all of them on a GPU)."""
        if xs[0].is_cuda and not _REF_ROT:
            from ..modules.attention_fn.tq_triton import tq_rotate
            return tq_rotate(self.rot_t, *xs) if len(xs) > 1 else tq_rotate(self.rot_t, *xs)[0]
        out = [(x.reshape(-1, self.dim) @ self.rot_t).view(x.shape) for x in xs]
        return out if len(xs) > 1 else out[0]

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        if y.is_cuda and not _REF_ROT:
            from ..modules.attention_fn.tq_triton import tq_rotate
            return tq_rotate(self.rot, y)[0]
        return (y.reshape(-1, self.dim) @ self.rot).view(y.shape)


def encode_ref(y: torch.Tensor, cb: torch.Tensor, mid: torch.Tensor, bits: int):
    """Reference writer. y [N, dim] (rotated, fp16 or fp32) -> (words int32 [N, dim/32*bits], scale fp16 [N])."""
    from ..ext_fallbacks import _pack_cache_codes
    n, dim = y.shape
    yf = y.float()
    nrm = yf.norm(dim = -1, keepdim = True).clamp_min(1e-12)
    codes = torch.bucketize(yf / nrm, mid)
    cn = cb[codes].norm(dim = -1, keepdim = True).clamp_min(1e-12)
    words = _pack_cache_codes(codes.view(n, dim // 32, 32), bits).reshape(n, dim // 32 * bits)
    return words, (nrm / cn).squeeze(-1).to(torch.float16)


def decode_ref(words: torch.Tensor, scale: torch.Tensor, cb: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Inverse of encode_ref in the rotated domain: -> fp16 [N, dim]."""
    from ..ext_fallbacks import _unpack_cache_codes
    n = words.shape[0]
    codes = _unpack_cache_codes(words.view(n, dim // 32, bits).to(torch.int64) & 0xffffffff, bits)
    return (cb[codes.reshape(n, dim)] * scale.float().unsqueeze(-1)).to(torch.float16)


class CacheLayer_tq(CacheLayer_quant):
    """TurboQuant KV pool. Tensors: qk / qv packed codes (CacheLayer_quant layout), sk / sv one fp16 scale per head vector."""

    tq = True

    def __init__(self, config, attention, cache_id: int, max_num_tokens: int, k_bits: int, v_bits: int, compand_a: float = 0.0):
        assert compand_a == 0.0
        super().__init__(config, attention, cache_id, max_num_tokens, k_bits, v_bits, 0.0)
        assert attention is not None and supported(attention.head_dim, k_bits) and supported(attention.head_dim, v_bits), \
            f"tq cache: head_dim {attention.head_dim} with {k_bits}/{v_bits} bits is not supported"
        self.head_dim = attention.head_dim
        self.num_kv_heads = attention.num_kv_heads
        self.qshape_s = (max_num_tokens // PAGE_SIZE, PAGE_SIZE, self.num_kv_heads)
        self.state: TQState | None = None

    @override
    def alloc(self, device: torch.device):
        super().alloc(device)
        self.state = TQState.get(self.head_dim, self.k_bits, self.v_bits, device)

    @override
    def free(self):
        super().free()
        self.state = None

    def get_tq(self) -> TQState:
        return self.state

    # -- writers / readers (rotated domain) -------------------------------------------------------------------------
    def _rows(self, cache_seqlens, block_table, bsz: int, length: int):
        """Physical pool row (page * PAGE_SIZE + offset) of each of the bsz x length new positions."""
        tok = cache_seqlens.long().unsqueeze(1) + torch.arange(length, device = cache_seqlens.device).unsqueeze(0)
        page = block_table.long().gather(1, tok // PAGE_SIZE)
        return (page * PAGE_SIZE + tok % PAGE_SIZE).reshape(-1)

    def write_rot(self, cache_seqlens, block_table, k, v, length: int):
        """k, v: rotated fp16 [bsz, length, kvh, hd] (contiguous). Reference writer."""
        st = self.state
        bsz = block_table.shape[0]
        rows = self._rows(cache_seqlens, block_table, bsz, length)
        for x, q, s, bits, cb, mid in ((k, self.qk, self.sk, self.k_bits, st.cbk, st.midk),
                                       (v, self.qv, self.sv, self.v_bits, st.cbv, st.midv)):
            words, sc = encode_ref(x.reshape(-1, self.head_dim), cb, mid, bits)
            q.view(-1, q.shape[-1]).index_copy_(0, rows, words.view(rows.numel(), -1))
            s.view(-1, self.num_kv_heads).index_copy_(0, rows, sc.view(rows.numel(), -1))

    def read_rot(self, rows: torch.Tensor):
        """Dequantized rotated K, V fp16 [len(rows), kvh, hd] for physical pool rows."""
        st = self.state
        out = []
        for q, s, bits, cb in ((self.qk, self.sk, self.k_bits, st.cbk), (self.qv, self.sv, self.v_bits, st.cbv)):
            w = q.view(-1, q.shape[-1]).index_select(0, rows).reshape(-1, self.head_dim // 32 * bits)
            sc = s.view(-1, self.num_kv_heads).index_select(0, rows).reshape(-1)
            out.append(decode_ref(w, sc, cb, bits, self.head_dim).view(rows.numel(), self.num_kv_heads, self.head_dim))
        return out

    @override
    def update_kv_direct(self, cache_seqlens, block_table, k, v, length: int):
        # k, v are already rotated by the dispatcher
        k, v = k.contiguous(), v.contiguous()
        if k.is_cuda and not _REF_WRITE:
            from ..modules.attention_fn.tq_triton import tq_write
            st = self.state
            tq_write(k, self.qk, self.sk, st.cbk, st.midk, cache_seqlens, block_table, self.k_bits, self.num_kv_heads, self.head_dim, PAGE_SIZE)
            tq_write(v, self.qv, self.sv, st.cbv, st.midv, cache_seqlens, block_table, self.v_bits, self.num_kv_heads, self.head_dim, PAGE_SIZE)
        else:
            self.write_rot(cache_seqlens, block_table, k, v, length)

    @override
    def get_kv(self, cache_seqlens, block_table, sliding_window: int = -1):
        """Natural-domain fp16 K / V of the whole pool (callers outside the quant-direct path expect plain vectors)."""
        st = self.state
        k = torch.empty(self.shape, dtype = torch.half, device = self.device)
        v = torch.empty(self.shape, dtype = torch.half, device = self.device)
        n = self.shape[0] * PAGE_SIZE
        for lo in range(0, n, 1 << 16):
            rows = torch.arange(lo, min(n, lo + (1 << 16)), device = self.device)
            kr, vr = self.read_rot(rows)
            k.view(n, *self.shape[2:])[lo:lo + rows.numel()] = st.unrotate(kr)
            v.view(n, *self.shape[2:])[lo:lo + rows.numel()] = st.unrotate(vr)
        return k, v

    @override
    def update_kv(self, cache_seqlens, block_table, k, v, length: int):
        # k, v: natural-domain fp16 pool-shaped scratch (get_kv layout): re-encode the new rows
        st = self.state
        bsz = block_table.shape[0]
        rows = self._rows(cache_seqlens, block_table, bsz, length)
        kr = st.rotate(k.view(-1, self.num_kv_heads, self.head_dim).index_select(0, rows))
        vr = st.rotate(v.view(-1, self.num_kv_heads, self.head_dim).index_select(0, rows))
        # scatter back through write_rot's row logic
        for x, q, s, bits, cb, mid in ((kr, self.qk, self.sk, self.k_bits, st.cbk, st.midk),
                                       (vr, self.qv, self.sv, self.v_bits, st.cbv, st.midv)):
            words, sc = encode_ref(x.reshape(-1, self.head_dim), cb, mid, bits)
            q.view(-1, q.shape[-1]).index_copy_(0, rows, words.view(rows.numel(), -1))
            s.view(-1, self.num_kv_heads).index_copy_(0, rows, sc.view(rows.numel(), -1))

    @override
    def tp_export(self, plan):
        raise NotImplementedError("the tq KV cache does not support tensor parallel loading")
