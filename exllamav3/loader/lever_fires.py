"""fires counters of the Hy4 decode levers (HY4_DENSE_V2, HY4_MOE_V2, HY4_DRAFT_MOE_TILES, HY4_DRAFT_MLA_BG, HY4_DRAFT_IX_GV).
A lever that silently falls back (shape, alignment, missing code object) must be visible: an A/B arm whose flagged lever shows 0 hits, or any fallback reason, fails the cell.
Counters are python-side: a HIP graph replay does not re-run python, so they count the capture and eager call sites, not the replays."""
import threading

REG = {}
_LOCK = threading.Lock()


class Fires:
    def __init__(self, name, hit="hit", miss="fallback"):
        self.name, self.hit_key, self.miss_key = name, hit, miss
        self.hits, self.fallback = 0, {}
        REG[name] = self

    def hit(self):
        self.hits += 1

    def miss(self, reason):
        with _LOCK:
            self.fallback[reason] = self.fallback.get(reason, 0) + 1

    def reset(self):
        self.hits, self.fallback = 0, {}

    def snapshot(self):
        return {self.hit_key: self.hits, self.miss_key: dict(self.fallback)}


def line():
    """one printable line with every registered counter that was touched: FIRES name {hit: n, fallback: {...}} ..."""
    parts = [f"FIRES {n} {f.snapshot()}" for n, f in sorted(REG.items()) if f.hits or f.fallback]
    return " | ".join(parts) if parts else "FIRES none (no lever flag on, or no call yet)"
