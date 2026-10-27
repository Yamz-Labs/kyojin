"""Acceptance-driven draft depth for block drafters (DFlash2): how many drafted tokens a round verifies, 0 = plain decode.

One DraftDepth per request. It keeps an EMA of the drafted tokens a speculative round accepted and picks full depth, a short
block, or off. While off it probes one full round after `probe` plain steps; a failed probe doubles the interval (up to
probe_max), a good one switches speculation back on. Pure Python: no torch, unit-testable without a model.
"""
from __future__ import annotations

import os


class DraftDepth:

    def __init__(self, full: int, low: int = None, alpha: float = None, off_below: float = None, low_below: float = None,
                 full_above: float = None, probe_min: int = None, probe_max: int = None, warm: int = None):
        env = os.environ.get
        self.full = full
        self.low = min(full, low or int(env("EXL3_DFLASH_ADAPT_LOW", "3")))
        self.alpha = alpha or float(env("EXL3_DFLASH_ADAPT_ALPHA", "0.3"))
        self.off_below = off_below if off_below is not None else float(env("EXL3_DFLASH_ADAPT_OFF", "0.45"))
        self.low_below = low_below if low_below is not None else float(env("EXL3_DFLASH_ADAPT_LOW_BELOW", "1.0"))
        self.full_above = full_above if full_above is not None else float(env("EXL3_DFLASH_ADAPT_FULL_ABOVE", "2.2"))
        self.probe_min = probe_min or int(env("EXL3_DFLASH_ADAPT_PROBE", "4"))
        self.probe_max = probe_max or int(env("EXL3_DFLASH_ADAPT_PROBE_MAX", "32"))
        self.warm = warm if warm is not None else int(env("EXL3_DFLASH_ADAPT_WARM", "3"))   # rounds at full depth before any cut
        self.ema = None             # accepted drafted tokens per speculative round
        self.depth = full
        self.probe = self.probe_min
        self.since_probe = 0
        self.rounds = 0

    def want(self) -> int:
        """Drafted tokens to verify in the next round (0 = plain step)."""
        if self.depth == 0 and self.since_probe >= self.probe:
            return self.full
        return self.depth

    def record(self, drafted: int, accepted: int):
        """Result of one step: `drafted` tokens verified (0 = plain step), `accepted` of them kept."""
        if drafted == 0:
            self.since_probe += 1
            return
        probing = self.depth == 0
        self.rounds += 1
        a = float(accepted)
        if self.ema is None:
            self.ema = a
        else:
            w = 0.5 if probing else self.alpha
            self.ema += w * (a - self.ema)
        if probing:
            self.since_probe = 0
            if self.ema >= self.off_below:
                self.depth = self.full if self.ema >= self.low_below else self.low
                self.probe = self.probe_min
            else:
                self.probe = min(self.probe * 2, self.probe_max)
            return
        if self.rounds < self.warm:
            return
        # a short block caps the accepted count at `low`, so promote on the share of that cap rather than the full-depth figure
        if self.ema < self.off_below:
            self.depth = 0
            self.since_probe = 0
        elif self.depth == self.full:
            if self.ema < self.low_below:
                self.depth = self.low
        elif self.ema >= min(self.full_above, 0.75 * self.low):
            self.depth = self.full
