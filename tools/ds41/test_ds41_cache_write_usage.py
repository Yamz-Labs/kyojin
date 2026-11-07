"""HTTP cache-write usage over the existing no-weight DS4.1 stub (CPU only)."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys

import pytest
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ds41_serve import MODEL, assemble, chat_body, serve, sse_events
from ds41_engine import StubEngine


UNSUPPORTED = object()


class ReportedCacheStatsEngine(StubEngine):
    """Keep the real stub generation/tokenization; override engine cache metrics."""

    def __init__(self, write_tokens, *, sessions):
        super().__init__("OK", sessions=sessions, chunk=1)
        self.write_tokens = write_tokens

    async def generate(self, prompt, *, stats=None, **kwargs):
        async for delta in super().generate(prompt, stats=stats, **kwargs):
            yield delta
        # Exercise both HTTP worker contracts: last_stats for one session and
        # the request-owned stats dict for multiple sessions.
        reported = dict(self.last_stats, cached_tokens=3)
        if self.write_tokens is not UNSUPPORTED:
            reported["cache_write_tokens"] = self.write_tokens
        else:
            reported.pop("cache_write_tokens", None)
        self.last_stats = reported
        if stats is not None:
            stats.clear()
            stats.update(reported)


@asynccontextmanager
async def client_for_stats(write_tokens, sessions):
    engine = ReportedCacheStatsEngine(write_tokens, sessions=sessions)
    client = TestClient(TestServer(serve.create_app(engine, MODEL)))
    await client.start_server()
    try:
        yield client, engine
    finally:
        await client.close()


async def request_usage(client, endpoint, stream):
    body = {"model": MODEL, "max_tokens": 16, "stream": stream}
    if endpoint == "/v1/chat/completions":
        body = chat_body(thinking=False, **body)
    else:
        # Known input-token count independent of chat rendering and tokenizer
        # changes. The output still uses the real stub's byte tokenizer.
        body["prompt"] = list(range(12))
    if stream:
        body["stream_options"] = {"include_usage": True}
    response = await client.post(endpoint, json=body)
    assert response.status == 200, await response.text()
    if not stream:
        result = await response.json()
        choice = result["choices"][0]
        assert (choice["message"]["content"] if endpoint == "/v1/chat/completions" else choice["text"]) == "OK"
        return result["usage"]
    events = sse_events(await response.text())
    assert events[-1] == "[DONE]"
    chunks = [event for event in events if isinstance(event, dict)]
    assert chunks and not any("error" in event for event in chunks)
    usage_chunks = [event for event in chunks if "usage" in event]
    assert len(usage_chunks) == 1
    assert usage_chunks[0] is chunks[-1]
    assert chunks[-1]["choices"][0]["finish_reason"] is not None
    if endpoint == "/v1/chat/completions":
        assert assemble(events)["message"]["content"] == "OK"
    else:
        assert "".join(event["choices"][0]["text"] for event in chunks) == "OK"
    return chunks[-1]["usage"]


def assert_usage(usage, engine, expected_write):
    prompt_tokens = len(engine.calls[-1]["ids"])
    assert usage["prompt_tokens"] == prompt_tokens
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == prompt_tokens + 2
    details = usage["prompt_tokens_details"]
    assert details["cached_tokens"] == 3
    assert "cache_write_tokens" not in usage  # Metric belongs only in input details.
    if expected_write is UNSUPPORTED:
        assert "cache_write_tokens" not in details
    else:
        assert type(details["cache_write_tokens"]) is int
        assert details["cache_write_tokens"] == expected_write
        assert details["cached_tokens"] + details["cache_write_tokens"] <= prompt_tokens


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions"], ids=["chat", "completions"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("sessions", [1, 2], ids=["last-stats", "request-stats"])
@pytest.mark.parametrize("write_tokens", [UNSUPPORTED, None, 0, 4, -5, 10**6],
                         ids=["unsupported", "none", "no-save", "saved", "negative", "over-fresh-prompt"])
def test_cache_write_usage_propagates_and_keeps_totals(endpoint, stream, sessions, write_tokens):
    async def go():
        async with client_for_stats(write_tokens, sessions) as (client, engine):
            usage = await request_usage(client, endpoint, stream)
            expected = UNSUPPORTED if write_tokens is UNSUPPORTED or write_tokens is None else max(0, min(write_tokens, len(engine.calls[-1]["ids"]) - 3))
            assert_usage(usage, engine, expected)
    asyncio.run(go())


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions"], ids=["chat", "completions"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("sessions", [1, 2], ids=["last-stats", "request-stats"])
def test_a_previous_save_does_not_leak_into_later_request_usage(endpoint, stream, sessions):
    async def go():
        async with client_for_stats(4, sessions) as (client, engine):
            assert_usage(await request_usage(client, endpoint, stream), engine, 4)
            engine.write_tokens = 0
            assert_usage(await request_usage(client, endpoint, stream), engine, 0)
            engine.write_tokens = UNSUPPORTED
            assert_usage(await request_usage(client, endpoint, stream), engine, UNSUPPORTED)
    asyncio.run(go())
