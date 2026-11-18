#!/usr/bin/env python3
"""Shared Prometheus /metrics collector for the kyojin servers (llama.cpp metric names).

Counters accumulate when a request completes (the HTTP layer observes the engine's eos stats);
the gauges requests_processing, requests_deferred and kv_cache_usage_ratio are read live at scrape.
All mutation happens on the server's event loop, so no locking is needed.

The kyojin:* metrics have no llama.cpp counterpart: the request count, failures, cancellations and finish
reasons, the queue time and the time to the first token, what the server is (version, server, model, sessions,
context) and, read live at scrape, the age of the oldest running request, the cache's token counts and
evictions and the device memory. A value a server does not pass is left out of the body.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

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
    ("kyojin:requests_total", "counter", "Finished requests"),
    ("kyojin:requests_failed_total", "counter", "Requests that ended in an engine error"),
    ("kyojin:requests_cancelled_total", "counter", "Requests the client cancelled, while queued or while running"),
    ("kyojin:queue_seconds_total", "counter", "Time the finished requests waited for a session"),
    ("kyojin:first_token_seconds_total", "counter", "Time the finished requests took from their start to the first token"),
    ("kyojin:oldest_request_age_seconds", "gauge", "Run time of the oldest running request, 0 when none runs"),
    ("kyojin:start_time_seconds", "gauge", "Unix time the server began to serve, after the model loaded"),
]

# Rendered only when the server passes the value: (name, type, help).
OPTIONAL = [
    ("kyojin:sessions", "gauge", "Requests the server runs at once"),
    ("kyojin:context_tokens", "gauge", "Context length of one session in tokens"),
    ("kyojin:kv_cache_max_tokens", "gauge", "Capacity of the paged KV cache in tokens"),
    ("kyojin:kv_cache_used_tokens", "gauge", "Tokens in pages referenced by running requests"),
    ("kyojin:kv_cache_free_tokens", "gauge", "Tokens in unreferenced pages that hold no reusable content"),
    ("kyojin:prompt_cache_tokens", "gauge", "Tokens in pages a later request can reuse (the prompt cache)"),
    ("kyojin:prompt_cache_evictions_total", "counter",
     "Reusable prompt-cache pages given to new content; starts again at 0 when the cache is dropped"),
    ("kyojin:gpu_memory_allocated_bytes", "gauge", "Device memory held by tensors of this process"),
    ("kyojin:gpu_memory_reserved_bytes", "gauge", "Device memory reserved by the allocator of this process"),
    ("kyojin:gpu_memory_free_bytes", "gauge", "Free device memory, all processes"),
    ("kyojin:gpu_memory_total_bytes", "gauge", "Total device memory"),
]
INFO = ("kyojin:info", "gauge", "Constant 1; the labels name the Kyojin version, the server and the model")
# One sample per stop reason seen so far (label reason); rendered once a request has finished.
FINISH = ("kyojin:finish_reasons_total", "counter", "Finished requests by the stop reason the engine reports")
# get_cache_stats() key and gpu_memory() key of each live OPTIONAL gauge
CACHE_KEYS = {"kyojin:kv_cache_max_tokens": "max_tokens", "kyojin:kv_cache_used_tokens": "used_tokens",
              "kyojin:kv_cache_free_tokens": "free_tokens", "kyojin:prompt_cache_tokens": "cached_tokens"}
GPU_KEYS = {"kyojin:gpu_memory_allocated_bytes": "allocated", "kyojin:gpu_memory_reserved_bytes": "reserved",
            "kyojin:gpu_memory_free_bytes": "free", "kyojin:gpu_memory_total_bytes": "total"}


def fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.6f}"


def kyojin_version() -> str:
    """The version in exllamav3/version.py, read as text: importing the package would load the extension."""
    try:
        text = (Path(__file__).resolve().parents[2] / "exllamav3" / "version.py").read_text()
        return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)
    except Exception:                                            # noqa: BLE001
        return "unknown"


def gpu_memory(torch) -> dict | None:
    """Device memory for render(gpu=...): the allocator counters of this process and the device's free and total
    bytes. None without torch or without a device (fake engines in tests)."""
    try:
        free, total = torch.cuda.mem_get_info()
        return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                "free": free, "total": total}
    except Exception:                                            # noqa: BLE001
        return None


def cache_evictions(generator) -> int | None:
    """Evicted prompt-cache pages for render(evictions=...): the complete pages the page table gave to new
    content. None without a paged cache (fake engines in tests)."""
    try:
        return generator.pagetable.metrics["evictions_live"]
    except Exception:                                            # noqa: BLE001
        return None


def label(value) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class Metrics:
    """Counters plus the last request's speeds, observed from the engine's eos event."""

    def __init__(self, *, server: str | None = None, model: str | None = None,
                 sessions: int | None = None, context: int | None = None) -> None:
        self.info = {"version": kyojin_version(), **({"server": server} if server else {}),
                     **({"model": model} if model else {})}
        self.sessions, self.context = sessions, context
        self.start_time = time.time()
        self.requests_total = 0.0
        self.requests_failed_total = 0.0
        self.requests_cancelled_total = 0.0
        self.queue_seconds_total = 0.0
        self.first_token_seconds_total = 0.0
        self.finish_reasons: dict[str, int] = {}
        self.running: dict[int, float] = {}     # begin() token -> start time of a request that runs now
        self.serial = 0
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
                drafts: int = 0, draft_tokens: int = 0, accepted: int = 0, queue_s: float = 0.0,
                first_token_s: float = 0.0, reason: str | None = None) -> None:
        self.requests_total += 1
        self.queue_seconds_total += queue_s
        self.first_token_seconds_total += first_token_s
        if reason:
            self.finish_reasons[str(reason)] = self.finish_reasons.get(str(reason), 0) + 1
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

    def failed(self) -> None:
        """A request that ended in an engine error (it is not observed, so it is not in requests_total)."""
        self.requests_failed_total += 1

    def cancelled(self) -> None:
        """A request the client cancelled, queued or running (it is not observed, so it is not in requests_total)."""
        self.requests_cancelled_total += 1

    def begin(self) -> int:
        """A request starts to run: the token for end(). The oldest one that still runs sets oldest_request_age_seconds."""
        self.serial += 1
        self.running[self.serial] = time.perf_counter()
        return self.serial

    def end(self, token: int | None) -> None:
        self.running.pop(token, None)

    def render(self, *, processing: int, deferred: int, kv_ratio: float,
               cache: dict | None = None, gpu: dict | None = None, evictions: int | None = None) -> bytes:
        """`cache` is the generator's get_cache_stats(), `gpu` is gpu_memory(), `evictions` is cache_evictions():
        all read by the server at scrape."""
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
            "kyojin:requests_total": self.requests_total,
            "kyojin:requests_failed_total": self.requests_failed_total,
            "kyojin:requests_cancelled_total": self.requests_cancelled_total,
            "kyojin:queue_seconds_total": self.queue_seconds_total,
            "kyojin:first_token_seconds_total": self.first_token_seconds_total,
            "kyojin:oldest_request_age_seconds":
                time.perf_counter() - min(self.running.values()) if self.running else 0.0,
            "kyojin:start_time_seconds": self.start_time,
            "kyojin:sessions": self.sessions,
            "kyojin:context_tokens": self.context,
            **{name: (cache or {}).get(key) for name, key in CACHE_KEYS.items()},
            **{name: (gpu or {}).get(key) for name, key in GPU_KEYS.items()},
            "kyojin:prompt_cache_evictions_total": evictions,
        }
        out = []
        for name, mtype, help_text in METRICS + OPTIONAL:
            if values[name] is not None:
                out.append(f"# HELP {name} {help_text}\n# TYPE {name} {mtype}\n{name} {fmt(values[name])}\n")
        if self.finish_reasons:
            name, mtype, help_text = FINISH
            out.append(f"# HELP {name} {help_text}\n# TYPE {name} {mtype}\n")
            out.extend(f'{name}{{reason="{label(r)}"}} {n}\n' for r, n in sorted(self.finish_reasons.items()))
        name, mtype, help_text = INFO
        labels = ",".join(f'{k}="{label(v)}"' for k, v in self.info.items())
        out.append(f"# HELP {name} {help_text}\n# TYPE {name} {mtype}\n{name}{{{labels}}} 1\n")
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
