"""Greedy-loop detector (hard release gate: 0/100). CPU only, model-agnostic.

A generation is a loop when its last 1000 characters compress below 30 % with zlib (same rule as loop1/loop2) AND the
tail holds an explicit repeated span: some 50-character span occurs at least 3 times in the last 3000 characters.
The span check is script-agnostic. The zlib ratio alone flags short valid answers in non-Latin scripts (Thai, 3 bytes per
character in UTF-8 compress differently: ifeval-0183, ratio 0.291), because zlib runs on UTF-8 bytes.
"""
import zlib
from collections import Counter

WINDOW, RATIO, MIN_CHARS = 1000, 0.30, 200
SPAN_TAIL, SPAN_LEN, SPAN_MIN = 3000, 50, 3
# the gate arm: 100 pinned IFEval prompts, cap 1536 (the old cap 640 hid loops: loop1), greedy. sampler_kw None = engine default
# (serve.DEFAULT_SAMPLER_KW / SERVE_SAMPLER_KW); a profile sets the shipped penalties, e.g. {"rep_p": 1.10} for GLM.
DEFAULT_ARM = {"task": "ifeval", "n": 100, "cap": 1536, "temp": 0.0, "top_p": 1.0, "sampler_kw": None}


def zlib_low(text):
    t = text[-WINDOW:]
    return len(t) >= MIN_CHARS and len(zlib.compress(t.encode())) / max(len(t.encode()), 1) < RATIO


def max_span_repeat(text):
    """Highest occurrence count of any SPAN_LEN-character span in the last SPAN_TAIL characters."""
    t = text[-SPAN_TAIL:]
    return max(Counter(t[i:i + SPAN_LEN] for i in range(len(t) - SPAN_LEN + 1)).values(), default=0)


def is_loop(text):
    return zlib_low(text) and max_span_repeat(text) >= SPAN_MIN


def summarize(recs):
    """recs: run_suite jsonl records (_id, text, hit_cap) -> {n, loops, ids, capped}."""
    ids = sorted(r["_id"] for r in recs if is_loop(r["text"]))
    return {"n": len(recs), "loops": len(ids), "ids": ids, "capped": sum(bool(r.get("hit_cap")) for r in recs)}
