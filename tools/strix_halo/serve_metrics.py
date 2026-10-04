#!/usr/bin/env python3
"""Shared Prometheus /metrics collector for the kyojin servers (llama.cpp metric names).

Counters accumulate when a request completes (the HTTP layer observes the engine's eos stats);
the gauges requests_processing, requests_deferred and kv_cache_usage_ratio are read live at scrape.
All mutation happens on the server's event loop, so no locking is needed.
"""
from __future__ import annotations

# (name, type, help) in the order the /metrics body renders them.
METRICS = [
    ("llamacpp:prompt_tokens_total", "counter", "Total prompt tokens"),
    ("llamacpp:prompt_tokens_cached_total", "counter", "Total prompt tokens reused from cache"),
    ("llamacpp:prompt_seconds_total", "counter", "Prompt process time"),
    ("llamacpp:tokens_predicted_total", "counter", "Total tokens generated"),
    ("llamacpp:tokens_predicted_seconds_total", "counter", "Predict process time"),
    ("llamacpp:n_tokens_max", "counter", "Largest observed n_tokens."),
    ("llamacpp:spec_decode_num_drafts_total", "counter", "Total speculative verification rounds"),
    ("llamacpp:spec_decode_num_draft_tokens_total", "counter", "Total draft tokens proposed"),
    ("llamacpp:spec_decode_num_accepted_tokens_total", "counter", "Total draft tokens accepted"),
    ("llamacpp:prompt_tokens_seconds", "gauge", "Prompt processing speed in tokens per second"),
    ("llamacpp:predicted_tokens_seconds", "gauge", "Generation speed in tokens per second"),
    ("llamacpp:requests_processing", "gauge", "Number of admitted requests, including cache preparation"),
    ("llamacpp:requests_deferred", "gauge", "Number of requests waiting for a session"),
    ("llamacpp:kv_cache_usage_ratio", "gauge",
     "In-flight prompt and generated tokens over sessions times context; excludes retained cache"),
]


def fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.6f}"


class Metrics:
    """Counters plus the last request's speeds, observed from the engine's eos event."""

    def __init__(self) -> None:
        self.prompt_tokens_total = 0.0
        self.prompt_cached_total = 0.0
        self.prompt_seconds_total = 0.0
        self.tokens_predicted_total = 0.0
        self.predicted_seconds_total = 0.0
        self.n_tokens_max = 0.0
        self.drafts_total = 0.0
        self.draft_tokens_total = 0.0
        self.accepted_tokens_total = 0.0
        self.prompt_tokens_seconds = 0.0
        self.predicted_tokens_seconds = 0.0

    def observe(self, *, prompt_tokens: int, cached: int, predicted: int,
                prefill_s: float, generate_s: float,
                drafts: int = 0, draft_tokens: int = 0, accepted: int = 0) -> None:
        self.prompt_tokens_total += prompt_tokens
        self.prompt_cached_total += cached
        self.prompt_seconds_total += prefill_s
        self.tokens_predicted_total += predicted
        self.predicted_seconds_total += generate_s
        self.n_tokens_max = max(self.n_tokens_max, prompt_tokens + predicted)
        self.drafts_total += drafts
        self.draft_tokens_total += draft_tokens
        self.accepted_tokens_total += accepted
        if prefill_s:
            self.prompt_tokens_seconds = (prompt_tokens - cached) / prefill_s
        if generate_s:
            self.predicted_tokens_seconds = predicted / generate_s

    def render(self, *, processing: int, deferred: int, kv_ratio: float) -> bytes:
        values = {
            "llamacpp:prompt_tokens_total": self.prompt_tokens_total,
            "llamacpp:prompt_tokens_cached_total": self.prompt_cached_total,
            "llamacpp:prompt_seconds_total": self.prompt_seconds_total,
            "llamacpp:tokens_predicted_total": self.tokens_predicted_total,
            "llamacpp:tokens_predicted_seconds_total": self.predicted_seconds_total,
            "llamacpp:n_tokens_max": self.n_tokens_max,
            "llamacpp:spec_decode_num_drafts_total": self.drafts_total,
            "llamacpp:spec_decode_num_draft_tokens_total": self.draft_tokens_total,
            "llamacpp:spec_decode_num_accepted_tokens_total": self.accepted_tokens_total,
            "llamacpp:prompt_tokens_seconds": self.prompt_tokens_seconds,
            "llamacpp:predicted_tokens_seconds": self.predicted_tokens_seconds,
            "llamacpp:requests_processing": processing,
            "llamacpp:requests_deferred": deferred,
            "llamacpp:kv_cache_usage_ratio": kv_ratio,
        }
        out = []
        for name, mtype, help_text in METRICS:
            out.append(f"# HELP {name} {help_text}\n# TYPE {name} {mtype}\n{name} {fmt(values[name])}\n")
        return "".join(out).encode()


def assert_reported(samples: dict, test, *, prompt_tokens: int, cached: int, predicted: int,
                    prefill_s: float, generate_s: float, drafts: int, draft_tokens: int,
                    accepted: int) -> None:
    """unittest helper: rendered /metrics samples must match one observed request."""
    expected = {
        "llamacpp:prompt_tokens_total": prompt_tokens,
        "llamacpp:prompt_tokens_cached_total": cached,
        "llamacpp:prompt_seconds_total": prefill_s,
        "llamacpp:tokens_predicted_total": predicted,
        "llamacpp:tokens_predicted_seconds_total": generate_s,
        "llamacpp:n_tokens_max": prompt_tokens + predicted,
        "llamacpp:spec_decode_num_drafts_total": drafts,
        "llamacpp:spec_decode_num_draft_tokens_total": draft_tokens,
        "llamacpp:spec_decode_num_accepted_tokens_total": accepted,
        "llamacpp:prompt_tokens_seconds": (prompt_tokens - cached) / prefill_s if prefill_s else 0,
        "llamacpp:predicted_tokens_seconds": predicted / generate_s if generate_s else 0,
    }
    for name, value in expected.items():
        test.assertAlmostEqual(samples[name], value, places=6, msg=name)
