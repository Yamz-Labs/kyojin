"""Install the optional CK sparse prefill handler into the normal dispatcher.

Installation happens once during ``dispatch`` import when
``EXL3_CK_PAGEDKV=sparse``. Replacing the dense CK entry in place means the
normal dispatch hint caches this wrapper, rather than bypassing sparse prefill
on later chunks. A declined shape returns ``None`` and therefore uses the
existing Triton fallback.
"""

from .ck_pagedkv import fn_ck_paged_attn
from .ck_pagedkv_sparse import fn_ck_paged_attn_sparse_auto

_stats = {"accepted": 0, "declined": 0, "pages_kept": 0, "pages_total": 0}


def fn_ck_paged_attn_sparse_engine(args):
    output, info = fn_ck_paged_attn_sparse_auto(args)
    if output is None:
        _stats["declined"] += 1
        return None
    _stats["accepted"] += 1
    if info:
        _stats["pages_kept"] += int(info["kept"])
        _stats["pages_total"] += int(info["n_past_pages"])
    return output


def install_sparse_engine(attn_fns):
    """Install once and preserve the ordinary backend as a decline fallback."""
    if fn_ck_paged_attn_sparse_engine in attn_fns:
        return attn_fns
    try:
        index = attn_fns.index(fn_ck_paged_attn)
    except ValueError:
        attn_fns.insert(0, fn_ck_paged_attn_sparse_engine)
    else:
        attn_fns[index] = fn_ck_paged_attn_sparse_engine
    return attn_fns
