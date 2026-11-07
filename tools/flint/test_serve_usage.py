"""Flint's native JSON/SSE usage path with no model or GPU imports."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
from aiohttp.test_utils import TestClient, TestServer

_spec = importlib.util.spec_from_file_location('flint_serve_usage', Path(__file__).with_name('serve.py'))
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)
UNMEASURED = object()


class Tokens:
    def numel(self):
        return 2


class Tokenizer:
    def encode(self, text, **kwargs):
        return [list(range(10))]


class Engine:
    ctx = 100
    tok = Tokenizer()

    def __init__(self, writes):
        self.writes = writes
        self.outq = {}
        self.cancelled = []

    async def submit(self, prompt, max_new, sampler, stop):
        q = asyncio.Queue()
        event = dict(text='OK', token_ids=Tokens(), eos=True,
                     eos_reason='stop_token', cached_tokens=3)
        if self.writes is not UNMEASURED:
            event['cache_write_tokens'] = self.writes
        q.put_nowait(event)
        self.outq[1] = q
        return 1, 10, q

    def cancel(self, ident):
        self.cancelled.append(ident)


@pytest.fixture
def sampler_stub(monkeypatch):
    # make_app needs only the sampler constructor. Preserve real imports in
    # other test files by letting monkeypatch restore every touched module.
    for name in ('exllamav3', 'exllamav3.generator', 'exllamav3.generator.sampler'):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['exllamav3.generator.sampler'].ComboSampler = lambda **kwargs: kwargs


async def response_usage(client, stream):
    response = await client.post('/v1/chat/completions', json={
        'messages': [{'role': 'user', 'content': 'hello'}], 'stream': stream,
        'stream_options': {'include_usage': True}, 'max_tokens': 8})
    assert response.status == 200, await response.text()
    if not stream:
        result = await response.json()
        assert result['choices'][0]['message']['content'] == 'OK'
        return result['usage']
    text = await response.text()
    assert text.endswith('data: [DONE]\n\n')
    events = [json.loads(line[6:]) for line in text.splitlines()
              if line.startswith('data: ') and line != 'data: [DONE]']
    assert ''.join(e['choices'][0]['delta'].get('content', '') for e in events if e['choices']) == 'OK'
    usage = [e['usage'] for e in events if 'usage' in e]
    assert len(usage) == 1 and events[-1]['choices'] == []
    return usage[0]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('writes,expected', [(UNMEASURED, None), (None, None), (0, 0), (5, 5), (-1, 0), (100, 7)])
def test_measured_writes_reach_json_and_sse(sampler_stub, stream, writes, expected):
    async def run():
        engine = Engine(writes)
        defaults = dict(temperature=1, top_p=.95, top_k=20, max_new=8)
        client = TestClient(TestServer(serve.make_app(engine, 'flint-test', '{{ messages[0].content }}', defaults)))
        await client.start_server()
        try:
            usage = await response_usage(client, stream)
            assert (usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens']) == (10, 2, 12)
            details = usage['prompt_tokens_details']
            assert details['cached_tokens'] == 3
            if expected is None:
                assert 'cache_write_tokens' not in details
            else:
                assert details['cache_write_tokens'] == expected
            assert not engine.outq and not engine.cancelled
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('stream', [False, True])
def test_previous_request_writes_do_not_leak(sampler_stub, stream):
    async def run():
        engine = Engine(5)
        defaults = dict(temperature=1, top_p=.95, top_k=20, max_new=8)
        client = TestClient(TestServer(serve.make_app(engine, 'flint-test', '{{ messages[0].content }}', defaults)))
        await client.start_server()
        try:
            assert (await response_usage(client, stream))['prompt_tokens_details']['cache_write_tokens'] == 5
            engine.writes = 0
            assert (await response_usage(client, stream))['prompt_tokens_details']['cache_write_tokens'] == 0
            engine.writes = UNMEASURED
            assert 'cache_write_tokens' not in (await response_usage(client, stream))['prompt_tokens_details']
        finally:
            await client.close()
    asyncio.run(run())
