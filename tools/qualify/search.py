"""Max-context search (prior work: rel174c3 srch/srch2.sh, nanb by-hand trials). Pure: the probe is injected.

A probe(ctx) -> (ok, note) starts the server at -c ctx and returns ok only when the startup guard passes, two distinct
near-full prompts are served back to back and the server log has no out-of-memory line.
"""
from __future__ import annotations

from typing import Callable

STEP = 2048          # grid of -c values (nanb: 73,728 = 36 x 2048; 77,824 = 38 x 2048)
FILL = 0.97          # share of -c the probe prompts use


def grid_floor(x: int, step: int = STEP) -> int:
    return max(0, x // step * step)


def upper_bound(free_bytes: float, bytes_per_token: float, headroom_bytes: float, cap: int | None, step: int = STEP) -> int:
    """First guess of the largest -c: free VRAM after weights minus headroom, divided by KV bytes per token, capped by max_pos."""
    ub = grid_floor(int(max(0.0, free_bytes - headroom_bytes) / bytes_per_token), step)
    return min(ub, grid_floor(cap, step)) if cap else ub


def bisect_max(lo: int, hi: int, probe: Callable[[int], tuple[bool, str]], step: int = STEP, max_probes: int = 8) -> dict:
    """Largest ctx on the grid in [lo, hi] for which probe is ok. hi is tried first (it is often the answer when capped by the model),
    then bisection. lo is assumed good only if probed ok: returns best=None when even lo fails (no OOM-free config in the range)."""
    tried: list[dict] = []
    if hi < step:
        return dict(best=None, tried=tried)

    def run(c: int) -> bool:
        ok, note = probe(c)
        tried.append(dict(ctx=c, ok=ok, note=note))
        return ok

    if run(hi):
        return dict(best=hi, tried=tried)
    good, bad = None, hi
    # find a passing floor first: lo, then keep bisecting between good and bad
    if lo < hi and run(lo):
        good = lo
    else:
        return dict(best=None, tried=tried)
    while bad - good > step and len(tried) < max_probes:
        mid = grid_floor((good + bad) // 2, step)
        if mid <= good or mid >= bad:
            break
        if run(mid):
            good = mid
        else:
            bad = mid
    return dict(best=good, tried=tried)
