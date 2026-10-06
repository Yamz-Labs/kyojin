"""8-bit paged cache -> rotated-domain fp16 staging for the HIP sparse prefill kernel.
Packed 8-bit cache: per 32-value group 32 offset-binary codes (four per int32 word, low byte first) + one fp16 scale; the stored
values are in the H32-rotated domain. Staged values d' = (code - 127.5) * (scale / 128), so the true K/V of a group is H_n d' with
H_n = H32 / sqrt(32) (symmetric, orthonormal): q . k = (H_n q) . d' and sum p v = H_n (sum p d'). The caller rotates q before and
un-rotates the output after the unchanged fp16 HIP kernel. Pages are staged into a compact buffer in block-table order."""
import os
import torch
import triton
import triton.language as tl
from .qwrite_triton import _h32_stage

TB = 8   # tokens per program (256 % TB == 0)


@triton.jit
def _deq8_kernel(qk, sk, qv, sv, ok, ov, btable, P, TBc: tl.constexpr):
    j = tl.program_id(0)
    tb = tl.program_id(1)
    page = tl.minimum(tl.maximum(tl.load(btable + j), 0), P - 1).to(tl.int64)
    tok = (tb * TBc + tl.arange(0, TBc))[:, None].to(tl.int64)
    offs = tl.arange(0, 512)[None, :]
    src = (page * 256 + tok) * 512 + offs
    sidx = (page * 256 + tok) * 16 + offs // 32
    dst = (j.to(tl.int64) * 256 + tok) * 512 + offs
    ck = tl.load(qk + src).to(tl.float32)
    ak = (ck - 127.5) * (tl.load(sk + sidx).to(tl.float32) / 128.0)
    tl.store(ok + dst, ak.to(tl.float16))
    cv = tl.load(qv + src).to(tl.float32)
    av = (cv - 127.5) * (tl.load(sv + sidx).to(tl.float32) / 128.0)
    tl.store(ov + dst, av.to(tl.float16))


def dequant8_pages(qk, sk, qv, sv, pages, out_k, out_v):
    """qk/qv: (P, 256, 128) int32, sk/sv: (P, 256, 16) fp16 (2 kv heads x 256 dims), pages: (n,) int32 block-table slice,
    out_k/out_v: (>= n*256, 2, 256) fp16 contiguous."""
    P = qk.shape[0]
    n = pages.shape[0]
    assert qk.shape[1:] == (256, 128) and sk.shape[1:] == (256, 16) and qv.shape == qk.shape and sv.shape == sk.shape
    assert out_k.shape[0] >= n * 256 and out_k.shape[1:] == (2, 256) and out_v.shape == out_k.shape
    assert pages.dtype == torch.int32 and pages.is_contiguous() and out_k.is_contiguous() and out_v.is_contiguous()
    _deq8_kernel[(n, 256 // TB)](qk.view(torch.uint8), sk, qv.view(torch.uint8), sv, out_k, out_v, pages, P, TBc=TB, num_warps=4)


@triton.jit
def _deqfull8_kernel(qk, sk, qv, sv, ok, ov, btable, P, TBc: tl.constexpr, G: tl.constexpr):
    """True fp16 dequantisation (H32 un-rotation included), same float32 operation order as ext_fallbacks.dequant_cache_cont:
    (code - 127.5) * ((scale * r32) / 128), unnormalised butterfly, one rounding to fp16."""
    j = tl.program_id(0)
    tb = tl.program_id(1)
    page = tl.minimum(tl.maximum(tl.load(btable + j), 0), P - 1).to(tl.int64)
    tok = (tb * TBc + tl.arange(0, TBc))[:, None, None].to(tl.int64)
    grp = tl.arange(0, G)[None, :, None]
    lane = tl.arange(0, 32)[None, None, :]
    src = ((page * 256 + tok) * G + grp) * 32 + lane
    sidx = (page * 256 + tok) * G + grp
    dst = ((j.to(tl.int64) * 256 + tok) * G + grp) * 32 + lane
    for which in tl.static_range(2):
        if which == 0:
            c = tl.load(qk + src).to(tl.float32)
            sc = tl.load(sk + sidx).to(tl.float32)
        else:
            c = tl.load(qv + src).to(tl.float32)
            sc = tl.load(sv + sidx).to(tl.float32)
        x = (c - 127.5) * ((sc * 0.17677669529663688) / 128.0)
        x = tl.reshape(x, (TBc * G, 32))
        x = _h32_stage(x, TBc * G, 1)
        x = _h32_stage(x, TBc * G, 2)
        x = _h32_stage(x, TBc * G, 4)
        x = _h32_stage(x, TBc * G, 8)
        x = _h32_stage(x, TBc * G, 16)
        x = tl.reshape(x, (TBc, G, 32)).to(tl.float16)
        if which == 0:
            tl.store(ok + dst, x)
        else:
            tl.store(ov + dst, x)


def dequant8_full(qk, sk, qv, sv, pages, out_k, out_v):
    """qk/qv (P, 256, G*8) int32, sk/sv (P, 256, G) fp16, pages (n,) int32 -> out_k/out_v (>= n*256 tokens, G*32) fp16 holding true K/V
    (any trailing shape with G*32 values per token), compact in `pages` order."""
    P, G = qk.shape[0], sk.shape[-1]
    n = pages.shape[0]
    assert qk.shape[1:] == (256, G * 8) and sk.shape[1:] == (256, G) and qv.shape == qk.shape and sv.shape == sk.shape and G & (G - 1) == 0
    assert pages.dtype == torch.int32 and pages.is_contiguous() and out_k.is_contiguous() and out_v.is_contiguous()
    assert out_k.numel() >= n * 256 * G * 32 and out_v.numel() >= n * 256 * G * 32 and out_k.dtype == out_v.dtype == torch.half
    _deqfull8_kernel[(n, 256 // 4)](qk.view(torch.uint8), sk, qv.view(torch.uint8), sv, out_k, out_v, pages, P, TBc=4, G=G, num_warps=4)


@triton.jit
def _rot32_kernel(x, y, n, BG: tl.constexpr):
    row = (tl.program_id(0).to(tl.int64) * BG + tl.arange(0, BG))[:, None]
    offs = row * 32 + tl.arange(0, 32)[None, :]
    m = row < n
    v = tl.load(x + offs, mask=m, other=0.0).to(tl.float32)
    v = _h32_stage(v, BG, 1)
    v = _h32_stage(v, BG, 2)
    v = _h32_stage(v, BG, 4)
    v = _h32_stage(v, BG, 8)
    v = _h32_stage(v, BG, 16)
    tl.store(y + offs, (v * 0.17677669529663688).to(tl.float16), mask=m)


def rot32(x, out = None):
    """x (..., 32k) fp16 contiguous -> H32/sqrt(32) applied to every group of 32 (fp32 butterfly, one rounding to fp16). Self-inverse."""
    assert x.dtype == torch.half and x.is_contiguous() and x.numel() % 32 == 0
    out = torch.empty_like(x) if out is None else out
    n = x.numel() // 32
    BG = 64
    _rot32_kernel[((n + BG - 1) // BG,)](x, out, n, BG=BG, num_warps=4)
    return out


_STAGE = {}


def _h32_f32(dev):
    h = torch.ones(1, 1, dtype = torch.float32)
    while h.shape[0] < 32:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / 32 ** 0.5).to(dev).contiguous()


def qsa_prefill_q8(q_rows, qk, sk, qv, sv, indices, sm_scale, block_table, n_used, page_size = 256):
    """Sparse prefill attention over an 8-bit paged cache through the fp16 HIP kernel (same call shape as qsa_prefill_hip).
    q_rows (R, 24, 256) fp16, block_table (>= n_used,) int32 logical page -> cache page. Returns (R, 24, 256) fp16."""
    from .qsa_prefill_hip import qsa_prefill_hip
    dev = q_rows.device
    st = _STAGE.get(dev)
    if st is None or st["k"].shape[0] < n_used * page_size:
        cap = max(n_used * page_size, 2 * (st["k"].shape[0] if st else 0))
        st = _STAGE[dev] = dict(
            k = torch.empty((cap, 2, 256), dtype = torch.half, device = dev),
            v = torch.empty((cap, 2, 256), dtype = torch.half, device = dev),
            ids = torch.arange(cap // page_size, dtype = torch.int32, device = dev),
            h = _h32_f32(dev))
    if os.environ.get("EXL3_QSA_Q8_STAGE", "full") == "full":
        # true fp16 K/V (un-rotation inside the staging kernel), unrotated q, no output rotation
        dequant8_full(qk, sk, qv, sv, block_table[:n_used].int().contiguous(), st["k"], st["v"])
        return qsa_prefill_hip(q_rows, st["k"][: n_used * page_size], st["v"][: n_used * page_size], indices, sm_scale, st["ids"][:n_used], page_size)
    dequant8_pages(qk, sk, qv, sv, block_table[:n_used].int().contiguous(), st["k"], st["v"])
    tri = os.environ.get("EXL3_QSA_Q8_ROT", "triton") == "triton"
    qr = rot32(q_rows.contiguous()) if tri else torch.matmul(q_rows.float().view(-1, 32), st["h"]).half().view(q_rows.shape)
    o = qsa_prefill_hip(qr, st["k"][: n_used * page_size], st["v"][: n_used * page_size], indices, sm_scale, st["ids"][:n_used], page_size)
    return rot32(o) if tri else torch.matmul(o.float().view(-1, 32), st["h"]).half().view(o.shape)
