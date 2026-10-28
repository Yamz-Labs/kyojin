"""Run-time switchable A/B knobs.

knob(name, default) reads os.environ when it is called (never at import), so one server can flip a switch between requests.
set_knobs({name: value}) is the only writer: it sets or unsets the environment, and when a value really changed it drops everything
that may have baked the old one in (captured CUDA graphs, block graphs, per-shape kernel caches registered with on_change).
Only names in KNOBS are accepted; the table holds the default each call site passes, so the effective value of an unset knob is known.
"""
import os
import weakref
from typing import Callable, Optional

# name -> default (str). The call sites pass the same default to knob(); tests/test_knobs.py checks the two agree.
KNOBS = {
    "EXL3_DEQ_WINDOW_TRITON": "1",
    "EXL3_GEMV_TRI": "1",
    "EXL3_GEMV_R_ARR1": "1",
    "EXL3_GEMV_R_WIDE8": "1",
    "EXL3_GEMV_R_RPB78": "4",
    "EXL3_BC_KVWIN": "1",
    "EXL3_CUDA_GRAPH_RING": "1",
    "EXL3_DEC_MOE_KBS_A": "0",
    "EXL3_DEC_MOE_KBS_B": "0",
    "EXL3_PF_CK": "auto",
    "EXL3_PF_WAVES_PER_EU": "2",
    "EXL3_VERIFY_ATTN_LOOP": "0",
    "EXL3_QDEC": "1",
    "EXL3_QDEC_HIP": "1",
    "EXL3_QDEC_GRAPH": "1",
    "EXL3_QDEC_ROWS": "1",
    "EXL3_QDEC_L2HINT": "1",
    "EXL3_QDEC_FASTDQ": "1",
    "EXL3_QDEC_RNW": "0",
    "EXL3_QDEC_RST": "",
    "EXL3_QDEC_SPL": "",
    "EXL3_QDEC_ABL": "",
    "EXL3_QDEC_HIP_NW": "0",
    "EXL3_QDEC_HIP_DW": "",
    "EXL3_QDEC_HIP_MAXO": "96",
    "EXL3_QDEC_HIP_FLAGS": "",
}

_hooks: list = []
_tracked: list = []      # (weakref to an object, attribute name of a dict it caches knob-dependent state in)


def knob(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


def knob_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None or v == "" else int(v)


def knob_on(name: str, default: str = "1") -> bool:
    """True unless the value is '0' (the 'X != "0"' idiom of the switches)"""
    return os.environ.get(name, default) != "0"


def on_change(fn: Callable[[], None]) -> Callable[[], None]:
    """Register a cache clearer that set_knobs runs after a value changed. Decorator-friendly."""
    _hooks.append(fn)
    return fn


def clear_on_change(obj, attr: str) -> None:
    """obj.<attr> (a dict) holds objects built from knob values (for example the per-layer BC attention graphs): set_knobs empties it."""
    _tracked[:] = [t for t in _tracked if t[0]() is not None]
    if not any(t[0]() is obj and t[1] == attr for t in _tracked):
        _tracked.append((weakref.ref(obj), attr))


def effective(names=None) -> dict:
    return {n: os.environ.get(n, KNOBS[n]) for n in (names or KNOBS)}


def invalidate() -> None:
    """Drop every captured graph and registered per-shape cache. Graphs are imported lazily: a CPU-only process has none."""
    import sys
    for modname, fn in (("exllamav3.model.cuda_graph", "purge"), ("exllamav3.modules.block_graph", "purge")):
        m = sys.modules.get(modname)
        if m is not None:
            getattr(m, fn)()
    for ref, attr in _tracked:
        obj = ref()
        if obj is not None:
            getattr(obj, attr).clear()
    for h in list(_hooks):
        h()


def set_knobs(values: dict) -> dict:
    """Apply {name: value} (None unsets). Returns the effective values of ALL knobs. Raises ValueError, nothing applied, on an
    unknown name. Graphs and caches are dropped only when an effective value changed."""
    bad = [n for n in values if n not in KNOBS]
    if bad:
        raise ValueError("unknown knob(s): " + ", ".join(sorted(bad)) + "; known: " + ", ".join(sorted(KNOBS)))
    before = effective()
    for n, v in values.items():
        if v is None:
            os.environ.pop(n, None)
        else:
            os.environ[n] = str(v)
    after = effective()
    if after != before:
        invalidate()
    return after
