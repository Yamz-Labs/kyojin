"""HY4_MU_SYNC_DEBUG=1: locate a device fault (HSA 0x1016 / hipErrorLaunchFailure) of the multi-session path.

A GPU fault is asynchronous: it surfaces at the next host sync (a .tolist(), a .to(), a torch.tensor(..., device=...)), far from the kernel that caused it. With the env set,
every stage of the slots path (prefill chunk, draft chain, each graph segment replay, the eager probe of a new graph, the draft catch-up steps) logs `mu-sync begin <stage>` and
waits for the device when it ends. The LAST `begin` line before the crash is the stage that faulted (a Python-level fault is also logged as `mu-sync FAULT in <stage>`).
A new (bucket, rows) trunk graph is first run once eagerly (`probe`) with a sync + log around every layer, indexer, K-row write, Engram stage, MLA core and FFN, so one rerun names the
layer and the op. Off by default: `stage` is a null context and `probe_wrap` does nothing."""
from __future__ import annotations

import contextlib
import os

import torch


def enabled() -> bool:
    return os.environ.get("HY4_MU_SYNC_DEBUG", "0") not in ("", "0")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def _stage(name):
    print(f"mu-sync begin {name}", flush=True)
    try:
        yield
        sync()
    except BaseException as ex:       # noqa: BLE001
        print(f"mu-sync FAULT in {name}: {ex!r}", flush=True)
        raise


def stage(name):
    return _stage(name) if enabled() else contextlib.nullcontext()


_MISSING = object()


@contextlib.contextmanager
def probe_wrap(targets):
    """targets = [(obj, attr, label)]: while active, obj.attr(...) is a synced, logged stage `label` (the first int argument, the layer index, is appended). Restores on exit."""
    saved = []
    for obj, attr, label in targets:
        fn = getattr(obj, attr)

        def wrapped(*a, _fn=fn, _label=label, **k):
            idx = next((x for x in a[:2] if isinstance(x, int)), None)
            with _stage(_label if idx is None else f"{_label} L{idx}"):
                return _fn(*a, **k)
        saved.append((obj, attr, obj.__dict__.get(attr, _MISSING)))
        setattr(obj, attr, wrapped)
    try:
        yield
    finally:
        for obj, attr, old in reversed(saved):
            if old is _MISSING:
                delattr(obj, attr)
            else:
                setattr(obj, attr, old)
