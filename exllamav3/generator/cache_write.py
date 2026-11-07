"""Logical input positions admitted to the reusable prompt cache by a job.

GPU forwards, copies and checkpoint admission are separate events. A forward
alone is not a cache write: only computed input positions covered by a retained
reusable checkpoint/page are credited. This module contains no model/GPU code.
"""
from threading import RLock


def merge_ranges(ranges, limit = None):
    """Union half-open token ranges, optionally restricted to an input bound."""
    out = []
    for start, end in sorted(ranges):
        start, end = max(0, int(start)), int(end)
        if limit is not None:
            end = min(end, limit)
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out


def subtract_ranges(ranges, excluded):
    """The positions in ranges which are absent from excluded."""
    excluded = merge_ranges(excluded)
    out = []
    for start, end in merge_ranges(ranges):
        for left, right in excluded:
            if right <= start:
                continue
            if left >= end:
                break
            if left > start:
                out.append((start, left))
            start = max(start, right)
            if start >= end:
                break
        if start < end:
            out.append((start, end))
    return out


def intersect_ranges(ranges, start, end):
    return [(max(left, start), min(right, end)) for left, right in ranges
            if max(left, start) < min(right, end)]


class CacheWriteTracker:
    """Request-owned ranges; the same object survives physical job requeues.

    Limits are captured before decode/requeue adds output to sequence_ids. Token
    healing supplies its shortened input length, so a sampled replacement does
    not become an input write. Per-sequence counts use the generator's existing
    integer-mean convention for cache reads.
    """

    def __init__(self, limits):
        self._lock = RLock()
        self.limits = tuple(int(n) for n in limits)
        self.computed = [[] for _ in self.limits]
        self.reads = [[] for _ in self.limits]
        self.covered = [[] for _ in self.limits]
        self.writes = [[] for _ in self.limits]

    def compute(self, seq, start, end):
        with self._lock:
            self.computed[seq] = merge_ranges(self.computed[seq] + [(start, end)], self.limits[seq])

    def read(self, seq, start, end):
        # Restoring this logical request's own work after a requeue is not a
        # second external cache hit, nor a new write of the copied rows.
        with self._lock:
            incoming = subtract_ranges([(start, min(end, self.limits[seq]))], self.computed[seq])
            self.reads[seq] = merge_ranges(self.reads[seq] + incoming, self.limits[seq])

    def commit(self, seq, start, end, *, new = True):
        with self._lock:
            end = min(end, self.limits[seq])
            if end <= start:
                return
            if new:
                fresh = subtract_ranges(intersect_ranges(self.computed[seq], start, end),
                                        self.reads[seq] + self.covered[seq])
                self.writes[seq] = merge_ranges(self.writes[seq] + fresh, self.limits[seq])
            self.covered[seq] = merge_ranges(self.covered[seq] + [(start, end)], self.limits[seq])

    def write_ranges(self, seq = 0):
        with self._lock:
            return list(self.writes[seq])

    def pending(self, seq):
        with self._lock:
            return bool(subtract_ranges(self.computed[seq], self.reads[seq] + self.covered[seq]))

    def read_ranges(self, seq = 0):
        with self._lock:
            return list(self.reads[seq])

    def count(self):
        with self._lock:
            return sum(end - start for ranges in self.writes for start, end in ranges) // max(len(self.limits), 1)
