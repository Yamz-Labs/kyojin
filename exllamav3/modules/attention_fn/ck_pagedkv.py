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
import math
import os
from pathlib import Path
import warnings

import torch

from ...ext import exllamav3_ext as ext
from .common import AttnArgs

_PAGE = 256
_fn: ctypes._CFuncPtr | None = None
_load_failed = False
_stats = {"accepted": 0, "rejected": 0}


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


def _kernel() -> ctypes._CFuncPtr | None:
    """Resolve once per process; an unavailable optional library uses the fallback."""
    global _fn, _load_failed
    if _load_failed:
        return None
    if _fn is None:
        path = ck_library_path()
        reason = None
        fn = None
        if not path.is_file():
            reason = "file not found"
        else:
            try:
                library = ctypes.CDLL(str(path))
            except OSError as exc:
                reason = str(exc)
            else:
                try:
                    fn = library.ck_pagedkv_forward
                except AttributeError as exc:
                    reason = str(exc)
        if fn is None:
            _load_failed = True
            warnings.warn(
                f"CK paged-prefill library unavailable at {path}: {reason}; "
                "using the ordinary fallback. Build it with tools/strix_halo/build_ck_pagedkv.sh "
                "or set EXL3_CK_PAGEDKV_SO before starting the process.",
                RuntimeWarning, stacklevel=2,
            )
            return None
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


def owns(args: AttnArgs) -> bool:
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


def fn_ck_paged_attn(args: AttnArgs) -> torch.Tensor | None:
    """Run the loaded CK backend, or decline to the next dispatcher handler."""
    if not owns(args):
        _stats["rejected"] += 1
        return None

    kernel = _kernel()
    if kernel is None:
        _stats["rejected"] += 1
        return None

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
    if result == -1.0:
        # Explicit CK decline only. Invocation/device exceptions are not recoverable
        # here. The preceding cache update does not advance cache_seqlens, so the
        # fallback writes the same K/V to the same positions, not an extra append.
        _stats["rejected"] += 1
        return None
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError(f"CK paged-prefill returned an invalid result (rc={result})")
    _stats["accepted"] += 1
    return out
