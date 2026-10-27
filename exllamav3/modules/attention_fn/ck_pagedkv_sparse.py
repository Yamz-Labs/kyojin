"""Page selection and compaction for optional CK sparse prefill.

Selected past pages are gathered in chronological order before current pages. The
CK kernel uses bottom-right causal alignment, so every selected past token remains
visible to every current query while the current chunk retains its ordinary causal
prefix. Page-aligned past context is required; a partial past page declines to the
normal attention backend.
"""

import ctypes
import math
import os

import torch

from ...ext import exllamav3_ext as ext
from ...util.attention_mode import CK_SPARSE_RETAIN, CK_SPARSE_MIN_SK
from .ck_pagedkv import _kernel, _PAGE as _page, _supported_device

_stats = {"dense": 0, "sparse": 0, "declined": 0}


def compact_table(bt, past, q_len, keep_past_pages):
    """Build the compacted physical page list and its compacted key length.

    bt              : [1, n_pages] int32 logical->physical
    past            : cached token length BEFORE this chunk
    q_len           : new tokens in this chunk
    keep_past_pages : bool tensor over the past's logical pages (chronological order)

    Returns (phys_pages int64 [n_compact], s_k_compact int).
    """
    page = _page
    if past % page != 0:
        raise ValueError(f"compaction requires page-aligned past, got past={past} page={page}")
    n_past_pages = past // page
    if keep_past_pages.numel() != n_past_pages:
        raise ValueError(f"keep mask has {keep_past_pages.numel()} entries, expected {n_past_pages}")
    n_cur_pages = math.ceil(q_len / page)
    total_pages = bt.shape[1]
    if n_past_pages + n_cur_pages > total_pages:
        raise ValueError(f"block table too short: {n_past_pages}+{n_cur_pages} > {total_pages}")

    past_phys = bt[0, :n_past_pages].long()
    cur_phys = bt[0, n_past_pages:n_past_pages + n_cur_pages].long()

    sel = past_phys[keep_past_pages]          # chronological order preserved
    pages = torch.cat([sel, cur_phys])         # current chunk LAST
    s_k_compact = int(sel.numel()) * page + q_len
    return pages, s_k_compact


def fn_ck_paged_attn_sparse(args, keep_past_pages):
    """Sparse variant of fn_ck_paged_attn: same kernel, compacted block table.

    Returns the output tensor, or None if the arguments are not owned (the caller then falls
    back to the dense path).
    """
    owned = (
        _supported_device(args.q.device)
        and not args.is_varlen()
        and not args.is_swa()
        and args.has_kv_cache()
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
        and args.q_len >= 16
    )
    if not owned:
        _stats["declined"] += 1
        return None

    kernel = _kernel()
    if kernel is None:
        _stats["declined"] += 1
        return None

    q = args.q
    k = args.k
    v = args.v
    k_cache = args.k_cache
    v_cache = args.v_cache
    bt = args.block_table
    cs = args.cache_seqlens

    q_len = q.shape[1]
    past = int(cs[0].item())
    s_k = past + q_len

    # 1) write the new K/V rows at their TRUE logical positions (original table)
    ext.paged_kv_cache_update(k, v, k_cache, v_cache, bt, cs)

    # 2) compact the table
    pages, s_k_compact = compact_table(bt, past, q_len, keep_past_pages)

    # 3) gather ONLY the compacted pages and repack to canonical (page, head, token, dim)
    kcan = k_cache[pages].permute(0, 2, 1, 3).contiguous()
    vcan = v_cache[pages].permute(0, 2, 1, 3).contiguous()
    n_pages = int(pages.numel())
    bt_id = torch.arange(n_pages, dtype=torch.int32, device=q.device)

    # 4) run the kernel with the compacted length
    head_pad = int(k_cache.shape[3])
    o = torch.zeros(1, q_len, 64, head_pad, device=q.device, dtype=torch.float16)
    skl = torch.tensor([s_k_compact], dtype=torch.int32, device=q.device)
    t = kernel(
        q.data_ptr(), kcan.data_ptr(), vcan.data_ptr(), o.data_ptr(),
        bt_id.data_ptr(), skl.data_ptr(),
        q_len, s_k_compact, n_pages, float(args.sm_scale), 0, head_pad,
        torch.cuda.current_stream().cuda_stream, 0,
    )
    if t == -1.0:
        _stats["declined"] += 1
        return None
    if not math.isfinite(t) or t <= 0:
        raise RuntimeError(f"CK paged-prefill returned an invalid result (rc={t})")
    _stats["sparse"] += 1
    return o


# --------------------------------------------------------------------------------------
# numerics: dense CK vs compacted CK vs a hand-masked fp32 reference over the SAME set
# --------------------------------------------------------------------------------------

def _ref_over_selected(kfull, vfull, q, dv, s_k, s, keep_token_mask):
    """fp32 reference: dense attention over all logical tokens, masked to the selected set.

    The causal mask is bottom-right (nm > s_k - s + qi), matching the shipped selftest.
    """
    nkv = kfull.shape[1]
    # q is [1, s, nq, dq] -- so nq is shape[2] and dq is shape[3]. Taking them from
    # shape[1]/shape[2] silently built a [s, nkv, s//nkv, nq] view and threw.
    nq = q.shape[2]
    dq = q.shape[3]
    qf = q[0].float().view(s, nkv, nq // nkv, dq)
    sc = torch.einsum("shgd,nhd->shgn", qf, kfull) * (dq ** -0.5)
    qi = torch.arange(s, device=q.device).view(s, 1, 1, 1)
    nm = torch.arange(s_k, device=q.device).view(1, 1, 1, -1)
    causal = nm <= (s_k - s + qi)
    sel = keep_token_mask.view(1, 1, 1, -1)
    sc = sc.masked_fill(~(causal & sel), float("-inf"))
    p = sc.softmax(dim=-1)
    ref = torch.einsum("shgn,nhd->shgd", p, vfull).reshape(s, nq, dv)
    return ref


def numerics_test(s=512, s_k=1536, keep_pages=(0, 3), seed=0, bt_list=None, verbose=True):
    """Compare dense CK, compacted CK, and the masked reference over the selected set."""
    from .common import AttnArgs

    torch.manual_seed(seed)
    nq, nkv, dq, dv, page = 64, 4, 192, 128, _page
    if s_k % page or s % page:
        raise ValueError("test geometry must be page-aligned")
    pages_total = s_k // page
    if bt_list is None:
        bt_list = list(range(pages_total))
    dev = "cuda"
    old = s_k - s

    q = ((torch.rand(1, s, nq, dq, device=dev, dtype=torch.float16) * 2) - 1) * 0.5
    knew = ((torch.rand(1, s, nkv, dq, device=dev, dtype=torch.float16) * 2) - 1) * 0.5
    vnew = torch.zeros(1, s, nkv, dq, device=dev, dtype=torch.float16)
    vnew[..., :dv].uniform_(-0.5, 0.5)
    k_cache = torch.zeros(pages_total, page, nkv, dq, device=dev, dtype=torch.float16)
    v_cache = torch.zeros(pages_total, page, nkv, dq, device=dev, dtype=torch.float16)
    # fill every logical page (past pages and the current chunk's page) with random data
    for lp, pp in enumerate(bt_list):
        lo, hi = lp * page, min(s_k, (lp + 1) * page)
        if hi <= lo:
            break
        k_cache[pp, :hi - lo] = knew.new_empty(hi - lo, nkv, dq).uniform_(-0.5, 0.5)
        v_cache[pp, :hi - lo, :, :dv] = vnew.new_empty(hi - lo, nkv, dv).uniform_(-0.5, 0.5)
    bt = torch.tensor(bt_list, dtype=torch.int32, device=dev).unsqueeze(0)
    cs = torch.tensor([old], dtype=torch.int32, device=dev)

    os.environ["EXL3_ATTN_BACKEND"] = "ck"
    args = AttnArgs(
        1, s, nq, dq,
        s, nkv,
        q, knew, vnew,
        k_cache, v_cache,
        True,
        float(dq) ** -0.5,
        None, None,
        None, 0.0,
        bt, cs,
        None, None, None,
        max_kv_len=None, v_dim=dv,
    )

    n_past_pages = old // page
    keep_mask = torch.zeros(n_past_pages, dtype=torch.bool, device=dev)
    for p in keep_pages:
        keep_mask[p] = True

    # --- compacted CK ---
    o_sparse = fn_ck_paged_attn_sparse(args, keep_mask)
    torch.cuda.synchronize()
    assert o_sparse is not None, "sparse path declined valid args"

    # --- dense CK on the same data (full table) ---
    from .ck_pagedkv import fn_ck_paged_attn
    os.environ["EXL3_ATTN_BACKEND"] = "ck"
    o_dense = fn_ck_paged_attn(args)
    torch.cuda.synchronize()
    assert o_dense is not None, "dense path declined valid args"

    # --- reference: dense masked to the SAME selected set ---
    n_idx = torch.arange(s_k, device=dev)
    phys = torch.tensor(bt_list, device=dev)[n_idx // page] * page + (n_idx % page)
    kfull = k_cache.view(-1, nkv, dq)[phys].float()
    vfull = v_cache.view(-1, nkv, dq)[phys][:, :, :dv].float()
    # the current chunk's rows are written through by the kernel call; recover them
    kfull[old:] = knew[0].float()
    vfull[old:] = vnew[0][:, :, :dv].float()

    keep_token_mask = torch.zeros(s_k, dtype=torch.bool, device=dev)
    for p in range(n_past_pages):
        if keep_mask[p]:
            keep_token_mask[p * page:(p + 1) * page] = True
    keep_token_mask[old:] = True           # the current chunk is always kept

    ref = _ref_over_selected(kfull, vfull, q, dv, s_k, s, keep_token_mask)

    err_sparse = (ref - o_sparse[0][..., :dv].float()).abs().max().item()
    err_dense = (ref - o_dense[0][..., :dv].float()).abs().max().item()
    # dense CK sees MORE keys than the reference's selected set, so it must DIFFER
    diff_dense_vs_sparse = (o_dense[0][..., :dv].float()
                            - o_sparse[0][..., :dv].float()).abs().max().item()

    ok = err_sparse < 2e-2
    if verbose:
        print(f"[sparse.selftest] s={s} s_k={s_k} keep_past_pages={list(keep_pages)} "
              f"of {n_past_pages}")
        print(f"   compacted CK vs masked reference : max_abs_err={err_sparse:.5f} "
              f"{'PASS' if ok else 'FAIL'}")
        print(f"   dense CK    vs masked reference : max_abs_err={err_dense:.5f} "
              f"(expected >0: dense sees the dropped pages)")
        print(f"   dense CK    vs compacted CK      : max_abs_diff={diff_dense_vs_sparse:.5f} "
              f"(expected >0: the dropped pages must change the result)")
    return ok


if __name__ == "__main__":
    import sys
    ok1 = numerics_test(s=256, s_k=1024, keep_pages=(0, 2))
    ok2 = numerics_test(s=512, s_k=1536, keep_pages=(0, 3), seed=7,
                        bt_list=[2, 0, 3, 1, 5, 4])
    # keep-all control: 3 past pages exist (s_k=1024, s=256, page=256) -> keep 0,1,2.
    # With every past page kept the compacted table equals the dense table, so the
    # compacted result must reproduce dense CK.
    ok3 = numerics_test(s=256, s_k=1024, keep_pages=(0, 1, 2), seed=3)
    print(f"[sparse.selftest] dropped={ok1 and ok2} keep_all={ok3}")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


# --------------------------------------------------------------------------------------
# Page-level SHARED keep-set selector (the deployable shape)
# --------------------------------------------------------------------------------------
# A paged cache page holds ALL FOUR KV HEADS, so a page is kept or dropped for all heads
# jointly: a per-KV-head keep set is NOT expressible in this cache format (see ledger
# row 178). This selector therefore produces ONE shared keep set over pages:
#   score each page per KV head (page-mean K dotted with a strided probe of the current
#   chunk's queries, maxed over probe rows and over the query heads of the group), then
#   UNION over the 4 KV heads ("keep the page if any head wants it"), then take the top-k
#   of the UNFORCED pages. Only the attention SINKS (first past pages) and the current
#   chunk's own pages are forced; current pages carry the diagonal and are appended
#   unconditionally by compaction. Most-recent past pages are ranked, not forced.
#
# Scoring is the bsref pool proxy at PAGE granularity: the page's MEAN K dotted with probe
# queries. Cost is O(n_past_pages) per call, and the mean is taken in FP32 (k_cache is
# fp16) so a 256-row page sum neither loses precision nor overflows.

def select_pages(k_cache, bt, past, q, q_len, retain=0.5, n_sink=2, n_local=1,
                 probe=512, n_heads_k=4, mode='union'):
    """Shared keep set over PAST cache pages -> bool mask [n_past_pages].

    k_cache : [n_pages, page, H_kv, D] fp16 (the engine's paged layout)
    bt      : [1, n_pages] int32 logical->physical
    past    : cached token length BEFORE this chunk (must be page-aligned)
    q       : [1, q_len, n_q_heads, D] fp16 current-chunk queries
    mode    : 'union'  -- per-KV-head top-k, then union  (mirrors bsref's shared=1 arm)
              'global' -- one global top-k over the per-head-max score

    THE SCORING MIRRORS bsref's VALIDATED pool selector exactly: for each KV head h, take
    the query heads of that head's GQA group, score each page by
    max over (probe rows, group heads) of q . mean_k[page, h], and keep that head's top-k.
    The two modes differ only in how the four per-head sets are COMBINED, and both are
    offered because they have different speed/quality trade-offs that must be measured:
      union  -> up to 4x the per-head budget of pages (the inflated keep of bsref shared=1)
      global -> exactly one budget of pages, ranked by the peak per-head demand
    NOTE: the first version of this function scored pages with
    einsum('pgd,nhd->pgn'), which SUMS the four KV heads instead of scoring per head. The
    keep COUNT is budget-determined so the speed numbers were unaffected, but the RANKING was
    not the proxy whose quality was validated. Fixed here.

    FORCED SET: the current chunk's own pages are always kept (they carry the diagonal) and
    are appended unconditionally by the compaction, so they are NOT part of this mask. The
    past-side forced set is therefore just the attention SINKS; bsref's 'local' window is
    the last n_local blocks of the whole sequence, which lies INSIDE the diagonal band and
    is therefore already kept -- it must not be counted a second time (the first version
    double-counted it in n_forced, which shrank the top-k budget).
    """
    import math as _math
    page = _page
    n_past_pages = past // page
    if past % page != 0:
        raise ValueError(f"page-aligned past required, got {past}")
    phys = bt[0, :n_past_pages].long()
    kp = k_cache[phys]                                  # [n_past_pages, page, H, D] fp16
    # past is page-aligned, so every past page is FULL: the mean over all `page` tokens is
    # the valid-token mean. (An unaligned past would need a per-page valid count.)
    mean_k = kp.mean(dim=1, dtype=torch.float32)        # [n_past_pages, H, D] fp32
    del kp

    n_cur_pages = (q_len + page - 1) // page
    nB = n_past_pages + n_cur_pages
    n_keep_total = max(n_local, _math.ceil(retain * nB))
    # n_forced counts what is kept WITHOUT being ranked: the current chunk's pages (always
    # kept) plus the sinks. 'local' is inside the current chunk's band, so it is already
    # included in n_cur_pages and is NOT counted again.
    n_forced = n_cur_pages + min(n_sink, n_past_pages)
    n_top = max(0, n_keep_total - n_forced)

    # per-KV-head importance, mirroring bsref's validated pool selector
    qs = q[0, ::max(1, probe)]                          # [P, n_q_heads, D]
    n_q_heads = qs.shape[1]
    G = n_q_heads // n_heads_k
    imp_per_head = []
    for h in range(n_heads_k):
        qp = qs[:, h * G:(h + 1) * G, :].float()        # [P, G, D]
        sc = torch.einsum("pgd,nd->pgn", qp, mean_k[:, h])
        imp_per_head.append(sc.amax(dim=(0, 1)))         # [n_past_pages] per head

    # the past-side forced set: the attention sinks only
    mask = torch.zeros(n_past_pages, dtype=torch.bool, device=q.device)
    if n_sink > 0:
        mask[:min(n_sink, n_past_pages)] = True
    n_forced_past = int(mask.sum().item())
    keep = mask.clone()
    avail = n_past_pages - n_forced_past

    if mode == 'union':
        # per-head top-k, then union: mirrors bsref's shared=1 arm
        for s in imp_per_head:
            s_r = s.masked_fill(mask, float("-inf"))
            k_h = min(n_top, avail)
            if k_h > 0:
                keep[torch.topk(s_r, k=k_h, largest=True).indices] = True
    elif mode == 'global':
        imp = torch.stack(imp_per_head).amax(dim=0)
        imp_r = imp.masked_fill(mask, float("-inf"))
        k_g = min(n_top, avail)
        if k_g > 0:
            keep[torch.topk(imp_r, k=k_g, largest=True).indices] = True
    else:
        raise ValueError(f"unknown mode {mode}")

    return keep, dict(n_past_pages=n_past_pages, n_cur_pages=n_cur_pages,
                      n_keep_total=n_keep_total, n_forced=n_forced,
                      n_forced_past=n_forced_past, n_top=n_top, mode=mode,
                      kept=int(keep.sum().item()))


def _owned(args):
    """Mirror the dense CK ownership test before spending selector work.

    Unsupported devices or shapes must fall through to the normal handler without
    selector, cache, or native work. Keep this predicate aligned with the direct
    sparse entry's duplicated guard.
    """
    return (
        _supported_device(args.q.device)
        and not args.is_varlen()
        and not args.is_swa()
        and args.has_kv_cache()
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
        and args.q_len >= 16
    )


def fn_ck_paged_attn_sparse_auto(args, retain=None, n_sink=None, n_local=None, probe=None,
                                 mode=None):
    """ENGINE-FACING sparse path: compute the shared keep set, then run the sparse kernel.

    Dispatch registers the engine wrapper at import time when EXL3_CK_PAGEDKV=sparse;
    the wrapper calls this selector entry and declines unsupported calls to the normal
    backend. Registration does not establish quality or end-to-end speedup.

    EXL3_CK_SPARSE_RETAIN (default 0.5) is the nominal keep fraction of total pages,
    including current pages. Current and sink pages are forced; most-recent past pages
    are not forced. The remaining options are function arguments, not environment
    switches: n_sink=2, n_local=1 (minimum nominal budget), probe=512, mode="global".
    """
    # Ownership FIRST: a shape the sparse path does not own must fall through to the dense
    # handler without the selector having done any work.
    if not _owned(args):
        return None, None
    if _kernel() is None:
        _stats["declined"] = (_stats.get("declined") or 0) + 1
        return None, None

    retain = CK_SPARSE_RETAIN if retain is None else retain
    n_sink = 2 if n_sink is None else n_sink
    n_local = 1 if n_local is None else n_local
    probe = 512 if probe is None else probe
    mode = "global" if mode is None else mode

    q = args.q
    k_cache = args.k_cache
    bt = args.block_table
    cs = args.cache_seqlens
    past = int(cs[0].item())
    q_len = q.shape[1]
    # Like retain, this threshold is frozen with the persistent cache identity.
    if past + q_len < CK_SPARSE_MIN_SK:
        _stats["declined"] += 1
        return None, None
    # RAGGED FINAL-CHUNK GUARD. The selector needs page-aligned `past`: every page it scores
    # must be FULL, because mean_k averages all `page` rows of the page and a partial page would
    # average in never-written lanes. So it refuses a misaligned past -- but it must REFUSE BY
    # DECLINING, not by raising, because the dispatcher contract is that a backend returns None
    # for a shape it does not own and the next function in the chain handles it. Raising here
    # took the whole request down: the engine's last chunk of a 32775-token prefill has
    # past=32769 (not a multiple of 256), and the first end-to-end candidate request died with
    #   RuntimeError: page-aligned past required, got 32769
    # Dense CK tolerates exactly this shape (ck_pagedkv.py:97 ceils: pages_needed = (s_k +
    # _page - 1) // _page), so declining hands this chunk to dense CK -- correct, and it costs
    # only the trailing chunk.
    #
    # Counted separately from the generic decline so the telemetry says WHICH refusal happened:
    # if this counter comes back as 1-2 per request the misalignment is the final chunk only and
    # declining is all that is needed; if it tracks the chunk count, past is misaligned on every
    # chunk and the selector needs to score floor-aligned pages and force-keep the trailing
    # partial page (verifier ready for that: verify_select_pages.py 7/7).
    # How many OWNED calls reach here, and what their past%page residues are. Without this
    # the misalignment question is unanswerable: the smoke either raises on the first ragged
    # past (loud) or declines everything (silent). The histogram separates "only the trailing
    # chunk is ragged" (declining is correct and cheap) from "the engine's cache length carries a
    # BOS offset so EVERY past is off by one" (declining would disable the sparse path entirely,
    # and the selector must instead score floor-aligned pages and force-keep the partial tail).
    _stats["owned_calls"] = (_stats.get("owned_calls") or 0) + 1
    _pm = _stats.get("past_mods") or {}
    _r = past % _page
    _pm[str(_r)] = _pm.get(str(_r), 0) + 1
    _stats["past_mods"] = _pm
    if past % _page != 0:
        # `_stats` values are reset to None elsewhere in this module, so `or 0` is load-
        # bearing: `.get(k, 0)` returns None for a key that EXISTS with a None value, and
        # None + 1 raises TypeError -- a crash on the exact path the counter exists to
        # measure. The offline test missed it because both cases declined earlier at
        # _owned(), so the guard never actually executed.
        _stats["declined_unaligned"] = (_stats.get("declined_unaligned") or 0) + 1
        if _stats["declined_unaligned"] <= 3:
            print(f"[ck_pagedkv_sparse] DECLINED unaligned past={past} (page={_page}); "
                  f"handing this chunk to the next backend", flush=True)
        return None, None
    n_heads_k = k_cache.shape[2]
    keep, info = select_pages(k_cache, bt, past, q, q_len, retain=retain,
                              n_sink=n_sink, n_local=n_local, probe=probe,
                              n_heads_k=n_heads_k, mode=mode)
    o = fn_ck_paged_attn_sparse(args, keep)
    return o, info
