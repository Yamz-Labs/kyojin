"""Frozen attention policy shared by import-time dispatch and persistent caches.

EXL3_CK_PAGEDKV=sparse enables approximate prefill. EXL3_CK_SPARSE_RETAIN
(default 0.5) and EXL3_CK_MIN_SK (default 0) affect its state and cache identity.
Changing these environment variables requires a new process, not a new session.
"""

import os


CK_SPARSE_ENABLED = os.environ.get("EXL3_CK_PAGEDKV", "").lower() == "sparse"
# Ignore sparse-only parameters in normal mode, including malformed values.
CK_SPARSE_RETAIN = float(os.environ.get("EXL3_CK_SPARSE_RETAIN", "0.5")) if CK_SPARSE_ENABLED else 0.5
CK_SPARSE_MIN_SK = int(os.environ.get("EXL3_CK_MIN_SK", "0")) if CK_SPARSE_ENABLED else 0

# None is the legacy normal mode. Version and all state-affecting sparse knobs
# travel together so a cache cannot silently cross attention policies.
ATTENTION_CACHE_MODE = (
    f"ck-sparse-v1:retain={CK_SPARSE_RETAIN.hex()}:min_sk={CK_SPARSE_MIN_SK}"
    if CK_SPARSE_ENABLED else None
)
