"""Loop guard shared by the Qwen and GLM servers: an n-gram period detector and a text rule for loops that drift.

Thinking uses the release-gate rule of tools/evalgate/loops.py; an answer needs proof of replay on top, so that valid
repetitive code or data is not cut (see answer_text_loop). CPU only, no model needed: tools/qwen/loopguard_gate.py replays a
labelled corpus through it with the real tokenizers.
"""
from __future__ import annotations

import os
import re
from typing import Any


# An n-gram period detector, plus a text rule for loops that drift (no exact period): the release-gate rule of tools/evalgate/loops.py (zlib ratio of the last 1000 chars
# < 30 % AND a 50-char span seen >= 3 times in the last 3000 chars) that must hold for a sustained stretch. Greedy
# thinking on a near-tie (the Qwen3.8 trait, ifeval-0015/0114/0195) is the target: the guard closes the think block.
GUARD_MAX_N = 64
GUARD_THINK = {"k": 4, "min_span": 48, "sustain": 320}    # tokens; GLM thresholds for the period rule
GUARD_ANSWER = {"k": 6, "min_span": 96, "sustain": 640}   # stricter: an answer may repeat items legitimately
GUARD_CHECK_EVERY = 16                                    # tokens between two evaluations of the text rule
TEXT_WINDOW, TEXT_RATIO, TEXT_MIN_CHARS = 1000, 0.30, 200
SPAN_TAIL, SPAN_LEN, SPAN_MIN = 3000, 50, 3


def loop_guard_default() -> bool:
    return os.environ.get("EXL3_LOOP_GUARD", "1") != "0"


def text_loop(text: str) -> bool:
    """tools/evalgate/loops.is_loop (kept identical; test_serve checks both agree)."""
    import zlib
    from collections import Counter
    w = text[-TEXT_WINDOW:]
    if len(w) < TEXT_MIN_CHARS or len(zlib.compress(w.encode())) / max(len(w.encode()), 1) >= TEXT_RATIO:
        return False
    t = text[-SPAN_TAIL:]
    return max(Counter(t[i:i + SPAN_LEN] for i in range(len(t) - SPAN_LEN + 1)).values(), default=0) >= SPAN_MIN


ANSWER_REPEAT_MIN, ANSWER_SHINGLE = 0.5, 60   # answer rule: share of 60-character shingles of the tail already seen in it
ANSWER_SATURATED = 0.07                       # answer rule: zlib ratio of the last 3000 characters
ANSWER_UNIT_MIN, ANSWER_UNITS_MIN, ANSWER_UNITS_ALONE = 25, 0.3, 0.6   # answer rule: share of repeated lines / sentences
DATA_LETTERS_MAX, DATA_MIN_CHARS, DATA_SUSTAIN = 0.10, 40, 16384   # content check: letters share of a data tail; tokens a data repetition may run before it is cut anyway
COUNTER_SUSTAIN, COUNTER_RATIO = 2000, 0.09   # counter arm: tokens it must hold, zlib ratio of the last 3000 characters (valid ramps: 0.10 and up)


def counter_drift(text: str) -> bool:
    """The last lines (25 characters or more) are one template whose numbers only move by 0 or 1 from line to line:
    `fillBox(cx-1,cx+1,254,254,...)`, `fillBox(...,255,255,...)`, ... A loop that counts has no repeated stretch and a
    tail that compresses to about 0.07; a table or a list of values has a field that jumps, and a valid ramp (a list of
    300 numbered links) still compresses less (0.10 and up). Needs 8 lines."""
    import zlib
    from collections import Counter
    b = text[-SPAN_TAIL:].encode()
    if len(zlib.compress(b)) / max(len(b), 1) >= COUNTER_RATIO:
        return False
    lines = [ln for ln in text[-SPAN_TAIL:].split("\n") if len(ln.strip()) >= ANSWER_UNIT_MIN]
    if len(lines) < 8:
        return False
    tmpl = [re.sub(r"\d+", "#", ln) for ln in lines]
    top, n = Counter(tmpl).most_common(1)[0]
    if n < 0.8 * len(lines):
        return False
    rows = [[int(x) for x in re.findall(r"\d+", ln)] for ln, t in zip(lines, tmpl) if t == top]
    steps = [all(abs(a - b) <= 1 for a, b in zip(r1, r2)) for r1, r2 in zip(rows, rows[1:])]
    return sum(steps) >= 0.9 * len(steps)


def _json_doc(line: str) -> bool:
    import json
    try:
        return isinstance(json.loads(line), (dict, list))
    except ValueError:
        return False


def data_like(text: str) -> bool:
    """The tail of an answer is data, not prose or code: almost no letters (a grid of zeros, a zero-padded hex dump, a table of
    numbers), or JSON lines (every complete line of the tail is a JSON object or array). Such output repeats because the data
    does (80 rows of 40 zeros, 60 identical records), not because the model is stuck, and a stop is final. The answer rules
    still cut it, but only after DATA_SUSTAIN tokens, so a loop on data does not run to the token limit."""
    t = text[-SPAN_TAIL:]
    body = [c for c in t if not c.isspace()]
    if len(body) < DATA_MIN_CHARS:
        return False
    if sum(c.isalpha() for c in body) <= DATA_LETTERS_MAX * len(body):
        return True
    lines = [ln.strip() for ln in t.split("\n")[1:-1] if ln.strip()]   # the first and last lines may be cut by the window
    return len(lines) >= 4 and all(_json_doc(ln) for ln in lines)


def shingle_repeat(text: str, k: int = ANSWER_SHINGLE) -> float:
    """Share of the k-character shingles of the last 3000 characters that already occurred earlier in that tail. A loop
    brings back whole stretches of text, so nearly every shingle is a repeat (0.55 and more); valid repetitive output
    (dense code, tables, JSON records) shares short idioms but writes new text around them (0.43 at most in our corpus, real code 0.16)."""
    t = text[-SPAN_TAIL:]
    seen: set[str] = set()
    hits = 0
    for i in range(len(t) - k + 1):
        s = t[i:i + k]
        if s in seen:
            hits += 1
        else:
            seen.add(s)
    return hits / max(len(t) - k + 1, 1)


def repeated_units(text: str) -> float:
    """Share of the lines and sentences (25 characters or more) of the last 3000 characters that repeat an earlier one
    word for word; 0.0 below 6 units. A loop that drifts re-emits the same few units in changing order, so no long stretch
    repeats but the units do. Valid code and data write a new line each time (0.2 at most in our corpus)."""
    units = [u.strip() for u in re.split(r"\n+|(?<=[.!?])\s+", text[-SPAN_TAIL:]) if len(u.strip()) >= ANSWER_UNIT_MIN]
    return 1 - len(set(units)) / len(units) if len(units) >= 6 else 0.0


def answer_text_loop(text: str) -> bool:
    """The release-gate rule (text_loop) is the right trigger for a thinking block, where a restart costs a few tokens.
    In an answer a stop is final, and valid code or data is often repetitive enough to trip it (a one-shot HTML page with
    a dozen dense helper functions did: zlib ratio 0.296, one 50-character idiom four times). So the answer needs proof
    that nothing new is written, on top of a repeated 50-character span: most of the tail replays itself
    (shingle_repeat >= 0.5), the same lines come back (repeated_units >= 0.3), or the tail compresses to almost nothing
    although no stretch is replayed (a template that only changes a counter). Lines that come back in more than half of
    the units are a loop without any span. The 0.30 zlib ratio of the gate rule is not asked here: it misses loops in
    scripts whose UTF-8 bytes compress poorly (Hindi, Thai); the replay shares do not depend on the script."""
    import zlib
    t = text[-SPAN_TAIL:]
    if len(t) < TEXT_MIN_CHARS:
        return False
    units = repeated_units(t)
    if units >= ANSWER_UNITS_ALONE:
        return True
    if max_span_repeat(t) < SPAN_MIN:
        return False
    if units >= ANSWER_UNITS_MIN or shingle_repeat(t) >= ANSWER_REPEAT_MIN:
        return True
    b = t.encode()
    return len(zlib.compress(b)) / max(len(b), 1) < ANSWER_SATURATED


class LoopMonitor:
    """Watches the tokens of one generation. feed() returns None, or a dict
    {phase, kind ("period"|"text"|"budget"), cut (tokens of the current phase to keep), at, period}.
    Phases: "think" (until the </think> token) then "answer"; a request without a think block starts in "answer".
    With detect=False only the thinking budget is enforced."""

    def __init__(self, phase: str, close_id: int | None, budget: int | None = None, detect: bool = True):
        self.close_id, self.budget, self.detect = close_id, budget, detect
        self.start_phase(phase)

    def start_phase(self, phase: str) -> None:
        self.phase = phase
        self.cfg = GUARD_THINK if phase == "think" else GUARD_ANSWER
        self.ids: list[int] = []
        self.run = [0] * (GUARD_MAX_N + 1)
        self.text = ""
        self.last_check = 0
        self.flag_from: int | None = None
        self.counter_from: int | None = None
        self._data_at, self._data = -GUARD_CHECK_EVERY, False   # data_like() of the tail, refreshed every GUARD_CHECK_EVERY tokens

    def _data_tail(self, at: int) -> bool:
        if at - self._data_at >= GUARD_CHECK_EVERY:
            self._data_at, self._data = at, data_like(self.text)
        return self._data

    def feed(self, tokens, text: str = "") -> dict[str, Any] | None:
        tokens = [int(t) for t in tokens]
        if self.phase == "think" and self.close_id is not None and self.close_id in tokens:
            k = tokens.index(self.close_id)
            hit = self._feed(tokens[:k], text)
            if hit:
                return hit
            self.start_phase("answer")
            return self._feed(tokens[k + 1:], "")
        return self._feed(tokens, text)

    def _feed(self, tokens, text: str) -> dict[str, Any] | None:
        ids, run, cfg = self.ids, self.run, self.cfg
        self.text += text
        for t in tokens:
            i = len(ids)
            ids.append(t)
            if self.phase == "think" and self.budget is not None and i + 1 >= self.budget:
                return {"phase": "think", "kind": "budget", "cut": self.budget, "at": i + 1, "period": 0}
            if not self.detect:
                continue
            for n in range(1, GUARD_MAX_N + 1):
                r = run[n] + 1 if i >= n and t == ids[i - n] else 0
                run[n] = r
                if r >= max(n * (cfg["k"] - 1), cfg["min_span"] - n):
                    if r < DATA_SUSTAIN and self.phase == "answer" and self._data_tail(i):
                        continue   # legit repetitive data: the period must hold for DATA_SUSTAIN tokens before it is a loop
                    return {"phase": self.phase, "kind": "period", "cut": max(i - r + 1, 1), "at": i + 1, "period": n}
        if self.detect and len(ids) - self.last_check >= GUARD_CHECK_EVERY:
            prev, self.last_check = self.last_check, len(ids)
            if (answer_text_loop if self.phase == "answer" else text_loop)(self.text[-(SPAN_TAIL + 200):]):
                if self.flag_from is None:
                    self.flag_from = prev
                if len(ids) - self.flag_from >= (DATA_SUSTAIN if self.phase == "answer" and self._data_tail(len(ids)) else cfg["sustain"]):
                    return {"phase": self.phase, "kind": "text", "cut": max(self.flag_from, 1), "at": len(ids), "period": 0}
            else:
                self.flag_from = None
            if self.phase == "answer" and counter_drift(self.text[-(SPAN_TAIL + 200):]):
                if self.counter_from is None:
                    self.counter_from = prev
                if len(ids) - self.counter_from >= (DATA_SUSTAIN if self._data_tail(len(ids)) else COUNTER_SUSTAIN):
                    return {"phase": "answer", "kind": "counter", "cut": max(self.counter_from, 1), "at": len(ids), "period": 0}
            else:
                self.counter_from = None
        if len(self.text) > 4 * SPAN_TAIL:
            self.text = self.text[-SPAN_TAIL * 2:]
        return None


def max_span_repeat(text: str) -> int:
    t = text[-SPAN_TAIL:]
    from collections import Counter
    return max(Counter(t[i:i + SPAN_LEN] for i in range(len(t) - SPAN_LEN + 1)).values(), default=0)


class AnswerTap:
    """Answer-phase guard for a server that streams events of text and token ids and cannot retract (GLM).

    A reply that opens inside a think block (`thinking=True`) is not watched until the closing tag shows up in the
    text; after that, or from the start without a think block, every event feeds an answer-phase LoopMonitor.
    feed() returns None, or the monitor's hit dict (kind, at, cut, period) once the answer loops."""

    def __init__(self, thinking: bool, close_tag: str = "</think>"):
        self.close_tag, self.waiting, self.tail = close_tag, thinking, ""
        self.mon = None if thinking else LoopMonitor("answer", None)

    def feed(self, tokens, text: str) -> dict[str, Any] | None:
        if self.waiting:
            joined = self.tail + text
            k = joined.find(self.close_tag)
            if k < 0:
                self.tail = joined[-len(self.close_tag):]
                return None
            self.waiting, text = False, joined[k + len(self.close_tag):]
            tokens = []                    # the tokens of this event are not split at the tag: only the text after it is fed
            self.mon = LoopMonitor("answer", None)
        return self.mon.feed(tokens, text)
