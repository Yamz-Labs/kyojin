"""vLLM-style request helpers shared by tools/qwen/serve.py and tools/glm/serve.py (pure Python, no engine import).

- token-id prompts for POST /v1/completions (`prompt` = [int, ...] or [[int, ...]])
- the `cache_salt` request field
- the POST /tokenize reply
- the `usage` object with optional prompt cache read/write counts
- the llama.cpp `timings` object (one shape on every server), final and progressive

See doc/prompt_cache_usage.md for the model-independent cache accounting contract.
"""
from __future__ import annotations

import time
from typing import Any, Callable


class CompatError(ValueError):
    """A request field is malformed (the servers answer 400)."""


def cache_salt(body: dict[str, Any]) -> str | None:
    """The request's `cache_salt`: None when absent. Requests with different salts never share prompt cache."""
    salt = body.get("cache_salt")
    if salt is None:
        return None
    if not isinstance(salt, str) or not salt:
        raise CompatError("cache_salt must be a non-empty string")
    return salt


def token_prompt(prompt: Any, vocab_size: int | None) -> list[int] | None:
    """The token ids of a prompt given as ids, None when it is text. `[int, ...]` and a single `[[int, ...]]` are
    accepted. An id outside 0 <= id < vocab_size is refused: the embedding lookup of the model would read out of bounds."""
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], list):
        prompt = prompt[0]
    if not isinstance(prompt, list) or not prompt or all(isinstance(x, str) for x in prompt):
        return None                                  # text, or a batch of texts (the servers take one prompt)
    if any(isinstance(x, list) for x in prompt):
        raise CompatError("one prompt per request: a list of token id lists must hold exactly one list")
    for x in prompt:
        if isinstance(x, bool) or not isinstance(x, int):
            raise CompatError("prompt token ids must be integers")
        if x < 0 or (vocab_size is not None and x >= vocab_size):
            raise CompatError(f"prompt token id {x} is outside the vocabulary (0..{(vocab_size or 0) - 1})")
    return list(prompt)


def tokenize_reply(ids: list[int], max_model_len: int | None) -> dict[str, Any]:
    """vLLM POST /tokenize reply."""
    return {"count": len(ids), "max_model_len": max_model_len, "tokens": ids, "token_strs": None}


def usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int | None = None, *,
          cache_write_tokens: int | None = None) -> dict[str, Any]:
    """OpenAI usage object with disjoint prompt cache read/write counts.

    Writes count input positions successfully retained during this request,
    not checkpoint copies, output tokens or cache occupancy. None means the
    engine does not report writes; zero means it measured none. Prompt/total
    counts remain inclusive of both cache buckets.
    """
    out: dict[str, Any] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                           "total_tokens": prompt_tokens + completion_tokens}
    details = {}
    if cached_tokens is not None:
        details["cached_tokens"] = max(0, min(int(cached_tokens), prompt_tokens))
    if cache_write_tokens is not None:
        room = max(0, prompt_tokens - details.get("cached_tokens", 0))
        details["cache_write_tokens"] = max(0, min(int(cache_write_tokens), room))
    if details:
        out["prompt_tokens_details"] = details
    return out


def timings(prompt_tokens: int, cache_n: int, prefill_s: float, decode_s: float, predicted_n: int, **extra: Any) -> dict[str, Any]:
    """The llama.cpp `timings` object: prompt_n (tokens prefilled, cache hits left out), prompt_ms, prompt_per_second, predicted_n,
    predicted_ms, predicted_per_second, cache_n. The swap orchestrator reads cache_n as the verdict of a KV restore."""
    prompt_n = max(prompt_tokens - cache_n, 0)
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": prefill_s * 1000,
            "prompt_per_second": prompt_n / prefill_s if prefill_s else 0.0,
            "predicted_n": predicted_n, "predicted_ms": decode_s * 1000,
            "predicted_per_second": predicted_n / decode_s if decode_s else 0.0, **extra}


def stats_timings(prompt_tokens: int, st: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """`timings` from the engine's per-request stats (cached_tokens, time_prefill, time_generate, new_tokens)."""
    return timings(prompt_tokens, int(st.get("cached_tokens", 0)), st.get("time_prefill") or 0.0,
                   st.get("time_generate") or 0.0, int(st.get("new_tokens", 0)), **extra)


class LiveTimings:
    """Progressive `timings` for one streamed reply: `take()` returns the object at most once per `period` seconds, None otherwise.

    Measured numbers only. The worker stamps the job dict: `_job_at` (the job left the queue), `_first_at` (its first token) and
    `_live["n"]` (the engine's own count of generated tokens, updated at each decode event). Prefill time is the span between the first two,
    decode time runs from `_first_at`. While the engine has not reported a count, `take()` returns None; a server whose engine cannot
    report one never builds a LiveTimings and sends `timings` on the final chunk only."""

    PERIOD_S = 1.0

    def __init__(self, job: dict[str, Any], prompt_tokens: int, stats: Callable[[], dict[str, Any]] = lambda: {},
                 period: float | None = None, clock: Callable[[], float] = time.perf_counter) -> None:
        self.job, self.prompt_tokens, self.stats = job, prompt_tokens, stats
        self.period, self.clock = self.PERIOD_S if period is None else period, clock
        self.last = None

    def take(self) -> dict[str, Any] | None:
        first, n = self.job.get("_first_at"), (self.job.get("_live") or {}).get("n")
        if first is None or n is None:
            return None
        now = self.clock()
        if self.last is None:
            self.last = first
        if now - self.last < self.period:
            return None
        self.last = now
        return timings(self.prompt_tokens, int(self.stats().get("cached_tokens") or 0),
                       max(first - self.job.get("_job_at", first), 0.0), now - first, n)


RQ_MIN_TOKENS = 2048   # smallest reply span a job holds pages for before it yields them (each yield costs a short re-prefill)


def rq_budget(ctx: int | None, sessions: int, prompt_tokens: int, max_new: int, draft: int = 0, page: int = 256) -> int | None:
    """Job `max_rq_tokens` for a request, or None to leave the generator's default.

    The generator reserves the cache pages for prompt + max_new_tokens + 1 + draft when a job starts, so a request that
    allows a reply as long as the context (no max_tokens) takes every page and the other sessions wait for it to end.
    With several sessions each job holds pages for its fair share of the context (ctx // sessions rounded down to pages, at
    least RQ_MIN_TOKENS of reply) and, if the reply goes on, the generator requeues it: the pages are released, the job re-enters the queue
    at the front and takes the pages again when they are free (the full pages are found by hash, only the tail is prefilled).
    A lone request therefore still reaches the end of the context; a reply that fits the share is untouched.
    None for one session (nothing competes) and when the whole reply fits the share."""
    if sessions <= 1 or not ctx:
        return None
    room = ctx // page * page - prompt_tokens          # slots the page table can give this prompt's reply
    share = ctx // sessions // page * page - prompt_tokens   # the same for a whole number of pages of the fair share
    budget = min(max(share, RQ_MIN_TOKENS), room)
    if budget < 1 or max_new + 1 + draft <= budget:
        return None
    return budget
