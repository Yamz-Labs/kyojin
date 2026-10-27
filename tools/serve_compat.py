"""vLLM-style request helpers shared by tools/qwen/serve.py and tools/glm/serve.py (pure Python, no engine import).

- token-id prompts for POST /v1/completions (`prompt` = [int, ...] or [[int, ...]])
- the `cache_salt` request field
- the POST /tokenize reply
- the `usage` object with `prompt_tokens_details.cached_tokens`
"""
from __future__ import annotations

from typing import Any


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


def usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int | None = None) -> dict[str, Any]:
    """OpenAI usage object. cached_tokens is clamped to 0..prompt_tokens; None leaves `prompt_tokens_details` out."""
    out: dict[str, Any] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                           "total_tokens": prompt_tokens + completion_tokens}
    if cached_tokens is not None:
        out["prompt_tokens_details"] = {"cached_tokens": max(0, min(int(cached_tokens), prompt_tokens))}
    return out


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
