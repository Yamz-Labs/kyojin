"""Optional sparse CK prefill over an 8-bit paged KV cache.

Quantized direct-cache calls use a separate dispatcher list, so the fp16 sparse
handler cannot see them. This adapter dequantizes the used past pages into a
persistent fp16 staging buffer in logical order, then uses the same selector and
CK compaction path as fp16 sparse prefill. It intentionally stages both K and V
for all used past pages in this first implementation: K is required to score every
page and selected-V-only staging is a future optimization, not a property claimed
by this module.

MiMo has 24 32-value quantization groups per token. The existing generic kernel
assumes a power-of-two group count, so this dequantizer pads the Triton program to
32 groups while masking the eight unused groups.
"""
import torch
import triton
import triton.language as tl

from ...util.attention_mode import CK_SPARSE_RETAIN, CK_SPARSE_MIN_SK
from .ck_pagedkv import _kernel, _supported_device
from .qwrite_triton import _h32_stage

_page = 256
_R32 = 0.17677669529663688   # 1/sqrt(32), applied after the unnormalised butterfly

_stats = {"qc_calls": 0, "qc_staged_pages": 0, "qc_kept_v_pages": 0}


@triton.jit
def _deqfull8_pad_kernel(qk, sk, qv, sv, ok, ov, btable, P, G: tl.constexpr,
                         G_PAD: tl.constexpr, TBc: tl.constexpr):
    """True fp16 dequantisation (H32 un-rotation included) of the pages listed in btable,
    G non-pow2 groups per token covered by G_PAD >= 2^ceil(log2(G)) with masked lanes.
    Same float32 op order as qdeq_triton._deqfull8_kernel / ext_fallbacks.dequant_cache_cont."""
    j = tl.program_id(0)
    tb = tl.program_id(1)
    page = tl.minimum(tl.maximum(tl.load(btable + j), 0), P - 1).to(tl.int64)
    tok = (tb * TBc + tl.arange(0, TBc))[:, None, None].to(tl.int64)
    grp = tl.arange(0, G_PAD)[None, :, None]
    gm = grp < G
    lane = tl.arange(0, 32)[None, None, :]
    src = ((page * 256 + tok) * G + grp) * 32 + lane
    sidx = (page * 256 + tok) * G + grp
    dst = ((j.to(tl.int64) * 256 + tok) * G + grp) * 32 + lane
    for which in tl.static_range(2):
        if which == 0:
            c = tl.load(qk + src, mask=gm, other=0.0).to(tl.float32)
            sc = tl.load(sk + sidx, mask=gm, other=0.0).to(tl.float32)
        else:
            c = tl.load(qv + src, mask=gm, other=0.0).to(tl.float32)
            sc = tl.load(sv + sidx, mask=gm, other=0.0).to(tl.float32)
        x = (c - 127.5) * ((sc * 0.17677669529663688) / 128.0)
        x = tl.reshape(x, (TBc * G_PAD, 32))
        x = _h32_stage(x, TBc * G_PAD, 1)
        x = _h32_stage(x, TBc * G_PAD, 2)
        x = _h32_stage(x, TBc * G_PAD, 4)
        x = _h32_stage(x, TBc * G_PAD, 8)
        x = _h32_stage(x, TBc * G_PAD, 16)
        x = tl.reshape(x, (TBc, G_PAD, 32)).to(tl.float16)
        if which == 0:
            tl.store(ok + dst, x, mask=gm)
        else:
            tl.store(ov + dst, x, mask=gm)


def _dequant8_pages_pad(qk, sk, qv, sv, pages, out_k, out_v, G, G_PAD):
    """qk/qv (P, 256, G*8) int32 (four 8-bit codes per word, low byte first); sk/sv (P, 256, G) fp16;
    pages (n,) int32 logical->physical; out_k/out_v hold >= n*256 tokens x G*32 fp16 values,
    compact in `pages` order. G may be any multiple of 1; G_PAD >= G power of two."""
    P = qk.shape[0]
    n = pages.shape[0]
    assert qk.shape[1:] == (256, G * 8) and sk.shape[1:] == (256, G), (qk.shape, sk.shape)
    assert qv.shape == qk.shape and sv.shape == sk.shape
    assert pages.dtype == torch.int32 and pages.is_contiguous()
    assert out_k.numel() >= n * 256 * G * 32 and out_v.numel() >= n * 256 * G * 32
    assert out_k.dtype == out_v.dtype == torch.half and out_k.is_contiguous() and out_v.is_contiguous()
    _deqfull8_pad_kernel[(n, 256 // 4)](
        qk.view(torch.uint8), sk, qv.view(torch.uint8), sv, out_k, out_v, pages, P,
        G=G, G_PAD=G_PAD, TBc=4, num_warps=4)


# Persistent scratch, one pair per device under the engine's single-stream contract.
# Concurrent streams/external views across growth are unsupported. Capacity grows in
# 32-page steps, bounded by the active packed cache; a layout/bound change retires it.
_STAGE = {}


def _stage_buffers(dev, cap_pages, vals_per_token, max_pages=None):
    if max_pages is not None and cap_pages > max_pages:
        raise ValueError(f"staging needs {cap_pages} pages but packed cache holds {max_pages}")
    st = _STAGE.get(dev)
    if (st is None or st["cap_pages"] < cap_pages
            or st["vals_per_token"] != vals_per_token
            or (max_pages is not None and st["cap_pages"] > max_pages)):
        new_cap = ((cap_pages + 31) // 32) * 32
        if max_pages is not None:
            new_cap = min(new_cap, max_pages)
        # Scratch is fully overwritten on every accepted call. Retire the old pair
        # before allocating replacements; retaining it inflates the growth peak.
        _STAGE.pop(dev, None)
        st = None
        st = _STAGE[dev] = dict(
            k=torch.empty((new_cap * 256, vals_per_token), dtype=torch.half, device=dev),
            v=torch.empty((new_cap * 256, vals_per_token), dtype=torch.half, device=dev),
            ident=None,   # identity table, rebuilt when cap grows
            cap_pages=new_cap,
            vals_per_token=vals_per_token,
        )
    st["used_pages"] = cap_pages
    return st


def _owned_qc(args):
    """Mirror of ck_pagedkv_sparse._owned plus the quantized-cache specifics this port supports."""
    return (
        args.q_cache is not None
        and _supported_device(args.q.device)
        and not args.is_varlen()
        and not args.is_swa()
        and args.cu_seqlens is None
        and args.softcap == 0.0
        and args.sinks is None
        and getattr(args, "ov", None) is None
        and getattr(args, "tq", None) is None
        and args.causal
        and not getattr(args, "non_causal_spans", None)
        and args.dim == 192
        and args.num_q_heads == 64
        and args.num_kv_heads == 4
        and args.bsz == 1
        and args.v_dim == 128
        and args.kv_len == args.q_len
        and args.q_len >= 16          # prefill chunks only: decode (q_len==1) and tiny tails
                                      # go to the Triton qc kernels; the selector cannot pay
                                      # for itself below a chunk
        and args.q.dtype == torch.float16
    )


def fn_ck_paged_attn_sparse_qc(args):
    """Sparse CK prefill over the 8-bit paged cache: stage, select, run the unchanged fp16 path.

    Returns the output tensor, or None when the shape is not owned (the dispatcher then falls
    through to the Triton online-dequant kernels, exactly as before this module existed).
    """
    from .ck_pagedkv_sparse import _stats as sparse_stats, _page, select_pages, fn_ck_paged_attn_sparse

    if not _owned_qc(args):
        return None
    if _kernel() is None:
        return None
    qk, sk, qv, sv, k_bits, v_bits = args.q_cache
    if k_bits != 8 or v_bits != 8:
        return None   # the staging kernel is 8-bit specific

    q = args.q
    bt = args.block_table
    cs = args.cache_seqlens
    past = int(cs[0].item())
    q_len = q.shape[1]
    if past + q_len < CK_SPARSE_MIN_SK:
        return None

    sparse_stats["owned_calls"] = (sparse_stats.get("owned_calls") or 0) + 1
    _r = past % _page
    _pm = sparse_stats.get("past_mods") or {}
    _pm[str(_r)] = _pm.get(str(_r), 0) + 1
    sparse_stats["past_mods"] = _pm
    if _r != 0:
        # ragged trailing chunk: decline, the Triton qc prefill handles it (same policy as the
        # fp16 sparse path's ragged guard)
        sparse_stats["declined_unaligned"] = (sparse_stats.get("declined_unaligned") or 0) + 1
        if sparse_stats["declined_unaligned"] <= 3:
            print(f"[ck_pagedkv_sparse_qc] DECLINED unaligned past={past} (page={_page})",
                  flush=True)
        return None

    n_past_pages = past // _page
    n_cur_pages = (q_len + _page - 1) // _page
    n_pages_needed = n_past_pages + n_cur_pages
    P = qk.shape[0]
    G = sk.shape[-1]                    # groups per token (24 for MiMo qk192 x 4 kv heads)
    G_PAD = max(16, 1 << (G - 1).bit_length())
    vals = G * 32                       # fp16 values per staged token (768)
    dev = q.device

    st = _stage_buffers(dev, n_pages_needed, vals, max_pages=P)
    cap_pages = st["cap_pages"]
    if st["ident"] is None or st["ident"].shape[0] < cap_pages:
        st["ident"] = torch.arange(cap_pages, dtype=torch.int32, device=dev)

    # 1) stage ALL past K pages (the selector needs every page's mean) and, for now, all past V.
    #    Staged in logical order (identity table slice), so logical page i lives at row i.
    pages_k = bt[0, :n_past_pages].to(torch.int32).contiguous()
    _dequant8_pages_pad(qk, sk, qv, sv, pages_k, st["k"], st["v"], G, G_PAD)
    _stats["qc_calls"] += 1
    _stats["qc_staged_pages"] += n_past_pages

    # 2) build the full fp16-shape args over the staged buffers: past pages staged, current
    #    pages left as-is (zeros); fn_ck_paged_attn_sparse writes the new rows itself via
    #    ext.paged_kv_cache_update at the true positions (identity table, cs = past).
    k_full = st["k"][: n_pages_needed * 256].view(n_pages_needed, 256, args.num_kv_heads, args.dim)
    v_full = st["v"][: n_pages_needed * 256].view(n_pages_needed, 256, args.num_kv_heads, args.dim)
    if n_cur_pages:
        k_full[n_past_pages:].zero_()
        v_full[n_past_pages:].zero_()
    bt2 = st["ident"][:n_pages_needed][None]
    args2 = args._replace(k_cache=k_full, v_cache=v_full, block_table=bt2, q_cache=None)

    # 3) unchanged fp16 sparse machinery: select over staged K, compact, kernel
    keep, info = select_pages(
        k_full, bt2, past, q, q_len,
        retain=CK_SPARSE_RETAIN,
        n_sink=2,
        n_local=1,
        probe=512,
        n_heads_k=k_full.shape[2],
        mode="global",
    )
    o = fn_ck_paged_attn_sparse(args2, keep)
    if o is None:
        sparse_stats["declined"] = (sparse_stats.get("declined") or 0) + 1
        return None

    _stats["qc_kept_v_pages"] += int(info.get("kept", 0)) if info else 0
    return o


def install_sparse_qc(_fns_qc):
    """Prepend the qc sparse wrapper to the quantized-cache dispatch chain (mirrors the
    EXL3_ATTN_BACKEND=ck install pattern for attn_fns). No-op replaces: the Triton qc kernels
    stay in the chain as the decline path."""
    if any(getattr(f, "__name__", "") == "fn_ck_paged_attn_sparse_qc" for f in _fns_qc):
        return _fns_qc
    _fns_qc.insert(0, fn_ck_paged_attn_sparse_qc)
    return _fns_qc
