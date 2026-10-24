"""Optional log of Triton JIT compiles (EXL3_LOG_JIT=1): one line per kernel compiled in this process, with its
compile-time constants and seconds. Kernels served from the on-disk cache are not logged. Use it to see what a start
(or the first request after READY) still compiles."""
from __future__ import annotations
import os, sys, time

_t0: dict = {}


def install(log=None) -> bool:
    if os.environ.get("EXL3_LOG_JIT", "0") != "1":
        return False
    try:
        from triton import knobs
    except Exception:   # noqa: BLE001  older triton: no hooks
        return False
    out = log or (lambda s: print(s, file=sys.stderr, flush=True))
    started = time.perf_counter()

    def pre(*, key, repr, fn, compile, is_manual_warmup, already_compiled):
        if not already_compiled:
            _t0[key] = time.perf_counter()
        return False

    def post(*, key, repr, fn, compile, is_manual_warmup, already_compiled):
        t = _t0.pop(key, None)
        if t is None or already_compiled:
            return
        now = time.perf_counter()
        name = str(repr).split("[", 1)[0]
        consts = ""
        try:
            consts = " ".join(f"{k[-1] if isinstance(k, tuple) else k}={v}" for k, v in (compile.get("constants") or {}).items())
        except Exception:   # noqa: BLE001  diagnostics never break a compile
            pass
        out(f"[jit] t={now - started:7.1f}s took {now - t:5.2f}s {name} {str(repr)[len(name):].split(']', 1)[0]}] {consts[:300]}")

    knobs.runtime.jit_cache_hook = pre
    knobs.runtime.jit_post_compile_hook = post
    return True
