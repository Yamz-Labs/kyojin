"""EXL3_SCHED_CHUNK_IL=1 (default off): chunked-prefill interleave planner. Pure python (no torch), importable without a GPU.

The engine already prefills in chunks (EXL3_PREFILL_CHUNK, 4096 rows; Job.prefill runs ONE chunk per Generator.iterate()
step and keeps kv_position / pages / recurrent slot between steps). This planner only decides, per step, WHICH jobs may run
their next chunk, so a big prefill (the "leader": first-arrived unfinished prefill with more than EXL3_CIL_SHORT_TOK rows left)
alternates with the queued short work of other jobs:

  shorts slice : leader parked, the other jobs run (short prefills, ONE batch-1 decode per step) for up to EXL3_CIL_SLICE_S s
  leader chunk : exactly one leader chunk; every other job is parked / held, so the chunk does not share its step with them
  ... repeat until the leader's deferral budget is used, then leader chunks only (= FIFO order) until its prefill is done.

Decode priority (EXL3_CIL_DECODE_PRIO=1, for GPUs that decode several users in one batch): while another job is decode-ready the
deferral budget does not apply and a leader chunk is not exclusive. The leader takes EXL3_CIL_DPRIO_CHUNK rows (default 768, about
0.75 s on a 3090) per step and shares the step with the decode round of every ready job; between two leader chunks the ready jobs
decode, all of them in one batch, for up to EXL3_CIL_SLICE_S (a short prefill beside a decoding job takes the same short chunks). A running stream waits one chunk, not the whole big prefill; the
leader keeps about two thirds of the GPU time. Without a decode-ready job the leader runs at full chunk size as before.

Hard bound: the leader's deferral (wall time of steps in which it did not run) is capped per leader by
min(EXL3_CIL_MAX_DEFER_S, EXL3_CIL_DEFER_FRAC x its estimated prefill time), so a smaller big prefill pays a proportionally
smaller price (default 8 s and 18 %: a 45K prefill of 36 s defers at most 6.5 s, an 8K one of 6.5 s at most 1.2 s). A slice step
runs the oldest short prefills whose rows still fit:  used + estimated_cost(step) <= budget  (cost = rows to prefill x measured
s/row of the leader's own chunks + EXL3_CIL_DECODE_S per decode step), so the overshoot is the estimate error of one step, not one
whole short request. A second big prefill is parked while the leader prefills (FIFO order between bigs, no round-robin between
equals). With fewer than two active jobs nothing is parked or held (single-user path untouched).

Only WHEN a chunk runs changes: chunk boundaries come from the job's own kv_position, never from other jobs, so the sequence of
chunk calls of every job is the one of the non-interleaved path (tests: test_chunk_il.py).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, NamedTuple


def chunk_il_enabled() -> bool:
    return os.environ.get("EXL3_SCHED_CHUNK_IL", "0") == "1"


class View(NamedTuple):
    key: Any            # stable job identity (arrival order = sort order of `order`)
    order: int          # arrival rank (Job.serial_number)
    remaining: int      # prompt rows still to prefill (0 once prefill is done)
    done: bool          # prefill done = decode-ready
    next_rows: int      # rows its next prefill chunk will take (0 when done)
    kv: int             # kv_position now (progress measure for the s/row estimate)


class Plan(NamedTuple):
    parked: set         # keys that skip prefill this step
    mode: str | None    # None (no interleave), 'shorts', 'chunk'
    leader: Any
    chunk_rows: int = 0  # decode priority: prefill rows per job this step (0 = the generator's own chunk size)


class _Leader:
    def __init__(self, key: Any, kv: int, now: float):
        self.key = key
        self.mode = None
        self.used = 0.0           # deferral so far (s)
        self.slice_used = 0.0
        self.t_last = now
        self.kv_last = kv
        self.slices = 0
        self.chunks = 0
        self.rows_chunked = 0
        self.t_start = now
        self.rows = 0             # prompt rows left when it took the lead (budget base)
        self.dprio_on = False     # the last plan ran under decode priority (its slice time is not charged to the deferral budget)


class ChunkPlanner:
    def __init__(self, short_tok: int | None = None, max_defer_s: float | None = None, slice_s: float | None = None,
                 row_s: float | None = None, decode_s: float | None = None, single_decode: bool | None = None,
                 defer_frac: float | None = None, clock=time.monotonic) -> None:
        e = os.environ.get
        self.short_tok = int(e("EXL3_CIL_SHORT_TOK", "4096")) if short_tok is None else short_tok
        self.max_defer_s = float(e("EXL3_CIL_MAX_DEFER_S", "8")) if max_defer_s is None else max_defer_s
        self.defer_frac = float(e("EXL3_CIL_DEFER_FRAC", "0.18")) if defer_frac is None else defer_frac
        self.slice_s = float(e("EXL3_CIL_SLICE_S", "0.6")) if slice_s is None else slice_s
        self.row_s = float(e("EXL3_CIL_ROW_S", "0.0007")) if row_s is None else row_s     # 2.9 s per 4096-row chunk until measured
        self.decode_s = float(e("EXL3_CIL_DECODE_S", "0.05")) if decode_s is None else decode_s
        self.dprio = e("EXL3_CIL_DECODE_PRIO", "0") == "1"
        self.dprio_chunk = max(256, int(e("EXL3_CIL_DPRIO_CHUNK", "768")) // 256 * 256)
        self.single_decode = (e("EXL3_CIL_SINGLE_DECODE", "1") != "0" and not self.dprio) if single_decode is None else single_decode
        self._clock = clock
        self._lead: _Leader | None = None
        self._resync = False
        self._tick = 0
        self._last_decoded: dict[Any, int] = {}
        self.finished: list[dict] = []      # stats of the last leaders that completed (bounded)

    # ---- helpers
    def _stats(self, ld: _Leader, now: float) -> dict:
        return {"deferred_s": round(ld.used, 4), "budget_s": round(self._budget(ld), 3), "slices": ld.slices, "chunks": ld.chunks,
                "span_s": round(now - ld.t_start, 3), "row_s": round(self.row_s, 6)}

    def _budget(self, ld: _Leader) -> float:
        """Deferral allowed for this leader: a fraction of its estimated prefill time (rows left when it took the lead x measured
        s/row), never more than the absolute cap. Fraction 0 = the cap alone."""
        if self.defer_frac <= 0:
            return self.max_defer_s
        return min(self.max_defer_s, self.defer_frac * ld.rows * self.row_s)

    def _pick(self, shorts: list[View], room: float) -> set:
        """Keys of the shorts a slice step runs: the oldest prefills whose rows still fit in `room` (s of deferral left), plus the
        decode of the ready ones. Empty when not even one of them fits."""
        ready = {v.key for v in shorts if v.done}
        cost = self.decode_s if ready else 0.0
        run = set()
        for v in sorted((v for v in shorts if not v.done), key=lambda v: v.order):
            c = v.next_rows * self.row_s
            if cost + c <= room:
                run.add(v.key)
                cost += c
        return (run | ready) if run or cost <= room else set()

    def solo(self) -> None:
        """The generator ran a step with fewer than two active jobs and did not call the planner. The time and progress of those steps
        are not interleave time: the next plan re-bases the leader's clock instead of charging them to its deferral or to the
        s/row estimate."""
        self._resync = self._lead is not None

    # ---- prefill round
    def plan_prefill(self, views: list[View]) -> Plan:
        now = self._clock()
        self._tick += 1
        live = [v for v in views if not v.done]
        ld = self._lead
        if ld is not None and not any(v.key == ld.key and not v.done for v in views):
            self.finished = self.finished[-15:] + [self._stats(ld, now)]    # leader finished its prefill (or left): the interleave is over
            logging.info("chunk_il leader done %s", self.finished[-1])
            self._lead = ld = None
        if ld is None:
            bigs = sorted((v for v in live if v.remaining > self.short_tok), key=lambda v: v.order)
            if not bigs:
                # decode priority also for the short prefills (a 2-3K-row tool result): they take the short chunk while someone decodes
                return Plan(set(), None, None, self.dprio_chunk if self.dprio and live and any(v.done for v in views) else 0)
            lv = bigs[0]
            ld = self._lead = _Leader(lv.key, lv.kv, now)
            ld.rows = lv.remaining
        lv = next(v for v in views if v.key == ld.key)
        others = [v for v in views if v.key != ld.key]
        if not others:
            ld.t_last, ld.kv_last = now, lv.kv
            return Plan(set(), None, ld.key)
        if self._resync:
            self._resync = False
            ld.t_last, ld.kv_last, ld.mode = now, lv.kv, None
        dt = max(now - ld.t_last, 0.0)
        if ld.mode == "shorts":
            if not ld.dprio_on:
                ld.used += dt
            ld.slice_used += dt
        elif ld.mode == "chunk" and lv.kv > ld.kv_last and dt > 0:
            rows = lv.kv - ld.kv_last                          # the leader's own chunk(s) since the last step: measured s/row
            ld.chunks += 1
            ld.rows_chunked += rows
            self.row_s = 0.5 * self.row_s + 0.5 * (dt / rows)
        ld.t_last, ld.kv_last = now, lv.kv
        bigs_other = [v for v in others if not v.done and v.remaining > self.short_tok]
        shorts = [v for v in others if v not in bigs_other]
        budget = self._budget(ld)
        # decode priority: with a decode-ready job beside the leader the shorts slice is not bounded by the deferral budget
        dprio = self.dprio and any(v.done for v in others)
        ld.dprio_on = dprio
        # next mode
        run = self._pick(shorts, float("inf") if dprio else budget - ld.used)
        fits = bool(run)
        if ld.mode in (None, "chunk"):
            if fits:
                ld.mode, ld.slice_used = "shorts", 0.0
                ld.slices += 1
            else:
                ld.mode = "chunk"
        elif not fits or ld.slice_used >= self.slice_s:         # 'shorts' continues until its slice / the budget is spent
            ld.mode = "chunk"
        if ld.mode == "shorts":
            return Plan({ld.key} | {v.key for v in others if not v.done and v.key not in run}, "shorts", ld.key, self.dprio_chunk if dprio else 0)
        return Plan({v.key for v in others if not v.done}, ld.mode, ld.key, self.dprio_chunk if dprio else 0)

    # ---- generation round (after the prefill round: jobs that just finished their prefill are decode-ready)
    def plan_decode(self, views: list[View], plan: Plan) -> set:
        """Keys of decode-ready jobs held this step."""
        if plan.mode is None:
            return set()
        ready = [v for v in views if v.done and v.key != plan.leader]     # the leader (just finished) decodes at once
        if plan.mode == "chunk":
            return set() if plan.chunk_rows else {v.key for v in ready}   # decode priority: the decode round shares the leader's chunk step
            # (otherwise the leader chunk does not share its step with a decode)
        if not self.single_decode or len(ready) <= 1:
            return set()
        run = min(ready, key=lambda v: (self._last_decoded.get(v.key, -1), v.order))   # one batch-1 decode, least recently served
        self._last_decoded[run.key] = self._tick
        if len(self._last_decoded) > 64:
            alive = {v.key for v in views}
            self._last_decoded = {k: t for k, t in self._last_decoded.items() if k in alive}
        return {v.key for v in ready if v.key != run.key}
