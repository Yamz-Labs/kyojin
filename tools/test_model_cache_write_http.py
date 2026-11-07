"""Cache-write usage on existing Qwen/GLM/MiMo OpenAI surfaces, without weights."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
UNSUPPORTED = object()
MODEL = "cache-write-test"
TEMPLATE = ("baseline one two three four five six seven "
            "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}assistant")
SERVERS = {}


def server(family):
    if family not in SERVERS:
        # Resolve the shared symlink targets even on source-copy test stages.
        for name in ("image_cache", "serve_metrics", "serve_reqlog", "serve_ping"):
            sys.modules.setdefault(name, importlib.import_module("tools.strix_halo." + name))
        name = "cache_write_http_" + family
        spec = importlib.util.spec_from_file_location(name, HERE / family / "serve.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        SERVERS[family] = module
    return SERVERS[family]


class TokenIds(list):
    def numel(self):
        return len(self)

    def flatten(self):
        return self

    def tolist(self):
        return list(self)


class FakeEngine:
    ctx = 4096
    supports_vision = False
    spec_on = False
    vocab_size = 4096

    def __init__(self, write_tokens, sessions):
        self.write_tokens, self.sessions = write_tokens, sessions
        self.calls, self.last_stats = [], {}
        self.report_stats = True

    def count_tokens(self, text):
        return len(text.split())

    def prompt_ids(self, prompt):
        return TokenIds(range(self.count_tokens(prompt)))

    def encode_ids(self, text, add_special_tokens=False):
        return self.prompt_ids(text)

    async def reset_prefix_cache(self):
        pass

    def spec_stats(self):
        return {"speculative": False}

    async def generate(self, prompt, *, stats=None, prompt_ids=None, **kwargs):
        prompt_tokens = len(prompt_ids) if prompt_ids is not None else self.count_tokens(prompt)
        self.calls.append({"prompt_tokens": prompt_tokens, "kwargs": kwargs})
        yield "O"
        await asyncio.sleep(0)
        yield "K"
        if self.report_stats:
            measured = {"prompt_tokens": prompt_tokens, "new_tokens": 1, "cached_tokens": 3,
                        "time_prefill": 0.1, "time_generate": 0.1, "eos_reason": "stop_token"}
            if self.write_tokens is not UNSUPPORTED:
                measured["cache_write_tokens"] = self.write_tokens
            self.last_stats = measured
            if stats is not None:
                stats.update(measured)


@asynccontextmanager
async def client_for(family, sessions, write_tokens):
    module = server(family)
    engine = FakeEngine(write_tokens, sessions)
    args = (engine, MODEL, TEMPLATE)
    app = module.create_app(*args, defaults=dict(module.load_defaults("/nonexistent"), reasoning_effort=None)) if family == "qwen" else module.create_app(*args)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield client, engine
    finally:
        await client.close()


async def request_usage(client, endpoint, stream, **extra):
    body = {"model": MODEL, "max_tokens": 16, "stream": stream}
    if endpoint == "/v1/chat/completions":
        body["messages"] = [{"role": "user", "content": "hello there"}]
    else:
        body["prompt"] = "one two three four five six seven eight nine ten"
    if stream:
        body["stream_options"] = {"include_usage": True}
    body.update(extra)
    response = await client.post(endpoint, json=body)
    assert response.status == 200, await response.text()
    if not stream:
        result = await response.json()
        choice = result["choices"][0]
        assert (choice["message"]["content"] if endpoint == "/v1/chat/completions" else choice["text"]) == "OK"
        return result["usage"]
    raw = await response.text()
    assert "data: [DONE]" in raw
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    assert events and not any("error" in event for event in events)
    usage_events = [event for event in events if "usage" in event]
    assert len(usage_events) == 1 and usage_events[0] is events[-1]
    assert events[-1]["choices"][0]["finish_reason"] is not None
    if endpoint == "/v1/chat/completions":
        assert "".join(event["choices"][0]["delta"].get("content", "") for event in events) == "OK"
    else:
        assert "".join(event["choices"][0]["text"] for event in events) == "OK"
    return events[-1]["usage"]


def assert_usage(usage, engine, expected, cached=3):
    prompt = engine.calls[-1]["prompt_tokens"]
    assert usage["prompt_tokens"] == prompt
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == prompt + 1
    details = usage["prompt_tokens_details"]
    assert details["cached_tokens"] == cached
    assert "cache_write_tokens" not in usage
    if expected is UNSUPPORTED:
        assert "cache_write_tokens" not in details
    else:
        assert type(details["cache_write_tokens"]) is int
        assert details["cache_write_tokens"] == expected
        assert cached + expected <= prompt


SURFACES = [(family, sessions, endpoint)
            for family, sessions in [("qwen", 2), ("glm", 1), ("glm", 2), ("mimo", 1)]
            for endpoint in (["/v1/chat/completions"] if family == "mimo" else ["/v1/chat/completions", "/v1/completions"])]


@pytest.mark.parametrize("family,sessions,endpoint", SURFACES)
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("write_tokens", [UNSUPPORTED, None, 0, 4], ids=["unsupported", "none", "zero", "saved"])
def test_openai_cache_write_usage(family, sessions, endpoint, stream, write_tokens):
    async def go():
        async with client_for(family, sessions, write_tokens) as (client, engine):
            expected = UNSUPPORTED if write_tokens is UNSUPPORTED or write_tokens is None else write_tokens
            assert_usage(await request_usage(client, endpoint, stream), engine, expected)
    asyncio.run(go())


@pytest.mark.parametrize("family,sessions,endpoint", SURFACES)
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_request_metrics_never_leak_a_previous_save(family, sessions, endpoint, stream):
    async def go():
        async with client_for(family, sessions, 4) as (client, engine):
            assert_usage(await request_usage(client, endpoint, stream), engine, 4)
            engine.write_tokens = 0
            assert_usage(await request_usage(client, endpoint, stream), engine, 0)
            engine.write_tokens = UNSUPPORTED
            assert_usage(await request_usage(client, endpoint, stream), engine, UNSUPPORTED)
            # An unsupported custom hook might not produce a terminal report at
            # all. Its old last_stats must not become this request's cache write.
            engine.report_stats = False
            assert_usage(await request_usage(client, endpoint, stream), engine, UNSUPPORTED, cached=0)
    asyncio.run(go())


@pytest.mark.parametrize("family,sessions,endpoint", [
    (family, 1 if family == "qwen" else sessions, endpoint)
    for family, sessions, endpoint in SURFACES
])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_http_cache_flag_does_not_override_measured_backend_writes(family, sessions, endpoint, stream):
    async def go():
        async with client_for(family, sessions, 4) as (client, engine):
            assert_usage(await request_usage(client, endpoint, stream, cache_prompt=False), engine, 4)
            if family == "qwen":
                assert engine.calls[-1]["kwargs"]["cache_prompt"] is False
    asyncio.run(go())


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_qwen_multisession_cache_disable_is_rejected_before_generation(endpoint, stream):
    async def go():
        async with client_for("qwen", 2, 4) as (client, engine):
            body = {"model": MODEL, "max_tokens": 16, "stream": stream, "cache_prompt": False}
            if endpoint == "/v1/chat/completions":
                body["messages"] = [{"role": "user", "content": "hello there"}]
            else:
                body["prompt"] = "one two three four five six seven eight nine ten"
            response = await client.post(endpoint, json=body)
            assert response.status == 400
            assert "needs a server started with --sessions 1" in await response.text()
            assert engine.calls == []
    asyncio.run(go())


def job(read_ranges, writes, read_tokens=None):
    return SimpleNamespace(cache_read_ranges=lambda: list(read_ranges),
                           cache_read_tokens=lambda: sum(b - a for a, b in read_ranges) if read_tokens is None else read_tokens,
                           cache_write_ranges=lambda: list(writes))


def test_qwen_restarted_jobs_keep_original_input_scope_and_first_read_baseline():
    tracker = server("qwen")._RequestCacheUsage(10)
    tracker.observe(job([(0, 3)], [(3, 7)]))
    # The replacement prompt includes generated reasoning. It reads writes from
    # this same logical request; those are not a new request's cacheRead.
    tracker.observe(job([(0, 7)], [(5, 18)]))
    stats = {"cached_tokens": 7, "cache_write_tokens": 13}
    tracker.apply(stats)
    assert stats == {"cached_tokens": 3, "cache_write_tokens": 7}


def test_qwen_read_holes_are_positions_not_a_contiguous_read_count():
    tracker = server("qwen")._RequestCacheUsage(12)
    tracker.observe(job([(4, 8)], [(0, 4), (8, 10)]))
    tracker.observe(job([(0, 10)], [(0, 14)]))
    stats = {}
    tracker.apply(stats)
    assert stats == {"cached_tokens": 4, "cache_write_tokens": 8}


def test_qwen_duplicate_and_generated_only_writes_do_not_inflate_usage():
    tracker = server("qwen")._RequestCacheUsage(10)
    tracker.observe(job([(0, 3)], [(3, 7), (3, 7)]))
    tracker.observe(job([(0, 7)], [(12, 20)]))
    stats = {}
    tracker.apply(stats)
    assert stats == {"cached_tokens": 3, "cache_write_tokens": 4}


def test_qwen_supported_no_commit_is_zero_but_custom_job_is_omitted():
    tracker = server("qwen")._RequestCacheUsage(10)
    tracker.observe(job([], []))
    stats = {}
    tracker.apply(stats)
    assert stats == {"cached_tokens": 0, "cache_write_tokens": 0}
    tracker.observe(SimpleNamespace())
    tracker.apply(stats)
    assert stats == {"cached_tokens": 0}


def test_qwen_guard_final_usage_resamples_earlier_completed_checkpoint():
    module = server("qwen")

    class Ids(TokenIds):
        dtype, device = "long", "cpu"

        def to(self, device):
            return self

    class Monitor:
        def __init__(self, *args, **kwargs):
            pass

        def feed(self, tokens, text):
            return dict(phase="think", kind="budget", at=1, period=0, cut=1)

        def start_phase(self, phase):
            pass

    async def go():
        engine = object.__new__(module.QwenEngine)
        engine.draft_model, engine.spec_on = None, False
        engine.ctx, engine.sessions = 4096, 2
        engine.slot_store, engine.prefix_disk = None, None
        engine.generator = SimpleNamespace(num_draft_tokens=0)
        engine.torch = SimpleNamespace(long="long", tensor=lambda rows, **kw: Ids(rows[0]),
                                       cat=lambda rows, **kw: Ids(token for row in rows for token in row))
        engine.eos, engine._routes = [], {}
        engine._sampler = lambda sampling: None
        engine._close_ids = lambda: (99, [99])
        jobs, first_writes = [], [(3, 6)]

        def make_job(**kwargs):
            number = len(jobs)
            current = job([(0, 3)] if not number else [(0, 9)], first_writes if not number else [(12, 16)])
            current.serial_number = number
            jobs.append(current)
            return current

        async def post(fn, *args):
            return fn(*args)

        async def drop(current):
            pass

        async def submit(current):
            queue = asyncio.Queue()
            if current.serial_number == 0:
                queue.put_nowait([dict(token_ids=Ids([10]), text="x")])
            else:
                # The first Job's outstanding checkpoint becomes confirmed
                # while the replacement runs. Nothing waits on its Future.
                first_writes.append((6, 9))
                queue.put_nowait([dict(token_ids=Ids([11]), text="OK", eos=True,
                                      prompt_tokens=12, cached_tokens=9,
                                      new_tokens=1, time_generate=0.1)])
            return queue

        engine.Job, engine._post, engine._drop, engine._submit = make_job, post, drop, submit
        stats, mine = {}, []
        with patch.object(module, "LoopMonitor", Monitor):
            pieces = [piece async for piece in engine._generate(
                "prompt", max_tokens=8, sampling={}, stop=[], speculative=False,
                prompt_ids=list(range(10)), reasoning_first=True, thinking_budget=1,
                loop_guard=True, loop_final=False, stats=stats, _mine=mine)]
        assert len(mine) == 2 and pieces[-1] == "OK"
        assert stats["prompt_tokens"] == 10 and stats["cached_tokens"] == 3
        assert stats["cache_write_tokens"] == 6

    asyncio.run(go())


@pytest.mark.parametrize("shared", [False, True], ids=["single", "shared-driver"])
@pytest.mark.parametrize("measured", [False, True], ids=["unsupported-job", "measured-job"])
def test_glm_guard_stats_sample_only_after_existing_cancel(shared, measured):
    module = server("glm")

    async def go():
        state = {"writes": 0, "cancelled": False}
        current = SimpleNamespace(serial_number=1)
        if measured:
            current.cache_read_tokens = lambda: 3
            current.cache_write_tokens = lambda: state["writes"]
        event = {"text": "OK", "token_ids": TokenIds([1])}

        class Generator:
            pending_jobs = []
            deferred_jobs = []

            def __init__(self):
                self.active_jobs = []

            def enqueue(self, item):
                self.active_jobs.append(item)
                return item.serial_number

            def iterate(self):
                return [event]

            def cancel(self, item):
                self.active_jobs.remove(item)
                # A cancellation may finish admitting a retained checkpoint.
                # Sampling before the existing cancel would incorrectly get0.
                state.update(writes=4, cancelled=True)

            def num_remaining_jobs(self):
                return len(self.active_jobs)

            def clear_queue(self):
                for item in list(self.active_jobs):
                    self.cancel(item)

        engine = object.__new__(module.ResidentEngine)
        engine.sessions = 2 if shared else 1
        engine.greedy_generator = Generator()
        hit = {"phase": "answer", "kind": "repeat", "at": 1, "cut": 0, "period": 1}
        with patch.object(module, "guard_hit", return_value=hit):
            if shared:
                engine._routes = {}
                engine._driver = SimpleNamespace(done=lambda: False)

                async def post(fn, *args):
                    result = fn(*args)
                    if fn.__name__ == "enqueue":
                        engine._routes[current.serial_number].put_nowait([event])
                    return result

                engine._post = post
                stats = {}
                output = [piece async for piece in engine._run_shared(current, TokenIds(range(12)),
                          asyncio.Event(), stats, 0.0, None, loop_guard=True)]
            else:
                # Only the already-constructed engine's control flow runs. No
                # real tensor/model/sampler or GPU initialization is involved.
                torch = ModuleType("torch")
                torch.long = object()
                torch.tensor = lambda rows, dtype=None: TokenIds(rows[0])
                sampler = ModuleType("exllamav3.generator.sampler")
                sampler.ComboSampler = lambda **kwargs: object()
                sampler.GreedySampler = lambda: object()
                engine.config = SimpleNamespace(eos_token_id_list=[])
                engine.Job = lambda **kwargs: current
                engine.settle_reply = lambda: asyncio.sleep(0)
                with patch.dict(sys.modules, {"torch": torch, "exllamav3.generator.sampler": sampler}):
                    output = [piece async for piece in engine.generate("", max_tokens=8,
                              temperature=0, top_p=1, stop=[], prompt_ids=list(range(12)), loop_guard=True)]
                stats = engine.last_stats
        assert output == ["OK"] and state["cancelled"]
        assert stats["eos_reason"] == "loop_guard" and stats["new_tokens"] == 1
        if measured:
            assert stats["cached_tokens"] == 3 and stats["cache_write_tokens"] == 4
        else:
            assert "cache_write_tokens" not in stats

    asyncio.run(go())
