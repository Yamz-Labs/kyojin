"""Optional Composable Kernel paged-prefill backend for MiMo qk192.

This backend is deliberately narrow: ROCm gfx1151, batch one, causal MiMo global
attention (64 query heads, 4 KV heads, Q/K=192, V=128 padded to 192), and
page-aligned KV cache pages. It returns ``None`` for every other shape so the
normal Triton dispatcher remains the fallback.

The shared object is built from the pinned AMD Composable Kernel source by
``tools/strix_halo/build_ck_pagedkv.sh``. No binary is checked into this
repository. Set ``EXL3_CK_PAGEDKV_SO`` to test a different locally-built copy.
"""

import ctypes
import os
from pathlib import Path

import torch

from .common import AttnArgs

_PAGE = 256
_fn: ctypes._CFuncPtr | None = None
_stats = {"accepted": 0, "rejected": 0}
_unavailable: str | None = None   # why the native library cannot be used; set once, then every call declines


def ck_library_path() -> Path:
    """Resolve an explicit test override or the reproducible builder output."""
    configured = os.environ.get("EXL3_CK_PAGEDKV_SO")
    if configured:
        return Path(configured).expanduser()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "kyojin" / "ck-pagedkv" / "libck_pagedkv.so"


def _supported_device(device: torch.device) -> bool:
    if not torch.version.hip or not torch.cuda.is_available() or device.type != "cuda":
        return False
    try:
        arch = str(torch.cuda.get_device_properties(device).gcnArchName)
    except Exception:
        return False
    return arch.split(":", 1)[0] == "gfx1151"


def _kernel() -> ctypes._CFuncPtr:
    global _fn
    if _fn is None:
        path = ck_library_path()
        if not path.is_file():
            raise RuntimeError(
                f"CK paged-prefill library not found at {path}. "
                "Build it with tools/strix_halo/build_ck_pagedkv.sh or set EXL3_CK_PAGEDKV_SO."
            )
        library = ctypes.CDLL(str(path))
        fn = library.ck_pagedkv_forward
        fn.restype = ctypes.c_float
        fn.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_void_p] * 2
            + [ctypes.c_int] * 3
            + [ctypes.c_float]
            + [ctypes.c_int] * 2
            + [ctypes.c_void_p]
            + [ctypes.c_int]
        )
        _fn = fn
    return _fn


def kernel_or_none() -> ctypes._CFuncPtr | None:
    """The native entry point, or None (with one warning) if the library is missing or does not load.

    The caller declines to the normal attention backend instead of raising, so a missing build never takes a
    request down.
    """
    global _unavailable
    if _unavailable is not None:
        return None
    try:
        return _kernel()
    except (RuntimeError, OSError, AttributeError) as exc:
        _unavailable = str(exc)
        print(f"[ck_pagedkv] disabled, using the normal attention backend: {exc}", flush=True)
        return None


def owns(args: AttnArgs) -> bool:
    return (
        _supported_device(args.q.device)
        and not args.is_varlen()
        and not args.is_swa()
        and args.has_kv_cache()
        and args.cu_seqlens is None
        and args.softcap == 0.0
        and args.sinks is None
        and args.causal
        and args.dim == 192
        and args.num_q_heads == 64
        and args.num_kv_heads == 4
        and args.bsz == 1
        and args.v_dim == 128
        and args.kv_len == args.q_len
        and args.q_len >= 16
    )


def fn_ck_paged_attn(args: AttnArgs) -> torch.Tensor | None:
    """Run the loaded CK backend, or decline to the next dispatcher handler."""
    if not owns(args):
        _stats["rejected"] += 1
        return None

    kernel = kernel_or_none()
    if kernel is None:
        return None
    from ...ext import exllamav3_ext as ext

    q, k, v = args.q, args.k, args.v
    k_cache, v_cache = args.k_cache, args.v_cache
    block_table, cache_seqlens = args.block_table, args.cache_seqlens
    q_len = q.shape[1]
    total_kv = int(cache_seqlens[0].item()) + q_len
    pages_needed = (total_kv + _PAGE - 1) // _PAGE

    # Preserve the engine's logical page order; physical cache pages need not be contiguous.
    ext.paged_kv_cache_update(k, v, k_cache, v_cache, block_table, cache_seqlens)
    page_indices = block_table[0, :pages_needed].long()
    k_canonical = k_cache[page_indices].permute(0, 2, 1, 3).contiguous()
    v_canonical = v_cache[page_indices].permute(0, 2, 1, 3).contiguous()
    identity_table = torch.arange(pages_needed, dtype=torch.int32, device=q.device)

    value_stride = int(k_cache.shape[3])
    out = torch.zeros(1, q_len, 64, value_stride, device=q.device, dtype=torch.float16)
    lengths = torch.tensor([total_kv], dtype=torch.int32, device=q.device)
    result = kernel(
        q.data_ptr(), k_canonical.data_ptr(), v_canonical.data_ptr(), out.data_ptr(),
        identity_table.data_ptr(), lengths.data_ptr(), q_len, total_kv, pages_needed,
        float(args.sm_scale), 0, value_stride, torch.cuda.current_stream().cuda_stream, 0,
    )
    if result < 0:
        _stats["rejected"] += 1
        return None   # the cache rows are already written (same data again on the fallback path)
    _stats["accepted"] += 1
    return out
