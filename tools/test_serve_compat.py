#!/usr/bin/env python3
"""Shared vLLM-style helpers, and RigMark's own request builders and parsers driven against both servers.

The RigMark part runs bench.py (read-only, outside this repo) over real HTTP against qwen/serve.py and glm/serve.py with a
fake engine that keeps a whole-prompt cache and drops it in reset_prefix_cache(). Model-free, CPU only. It is skipped when
the RigMark source is not at $RIGMARK_SRC.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import threading
import unittest
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_compat

RIGMARK = Path(os.environ.get("RIGMARK_SRC", "/nonexistent"))


class RqBudgetTests(unittest.TestCase):
    def test_one_session_or_no_ctx_keeps_the_default(self):
        self.assertIsNone(serve_compat.rq_budget(65536, 1, 1000, 60000))
        self.assertIsNone(serve_compat.rq_budget(None, 4, 1000, 60000))

    def test_reply_that_fits_the_share_is_untouched(self):
        self.assertIsNone(serve_compat.rq_budget(65536, 4, 1000, 8000))   # share 16384 - 1000 = 15384

    def test_open_ended_reply_holds_its_share_only(self):
        self.assertEqual(serve_compat.rq_budget(65536, 4, 1000, 64000), 65536 // 4 - 1000)   # 16384 is whole pages

    def test_long_prompt_gets_the_minimum_span_within_the_page_table(self):
        self.assertEqual(serve_compat.rq_budget(65536, 4, 30000, 35000), serve_compat.RQ_MIN_TOKENS)
        self.assertEqual(serve_compat.rq_budget(16384, 4, 40, 9000), 4096 - 40)   # four jobs fit the page table
        self.assertEqual(serve_compat.rq_budget(65536, 4, 64000, 5000), 65536 - 64000)   # never beyond the cache


class HelperTests(unittest.TestCase):
    def test_cache_salt(self):
        self.assertIsNone(serve_compat.cache_salt({}))
        self.assertEqual(serve_compat.cache_salt({"cache_salt": "s"}), "s")
        for bad in ("", 1, [], True):
            with self.assertRaises(serve_compat.CompatError):
                serve_compat.cache_salt({"cache_salt": bad})

    def test_token_prompt(self):
        self.assertEqual(serve_compat.token_prompt([1, 2], 10), [1, 2])
        self.assertEqual(serve_compat.token_prompt([[1, 2]], 10), [1, 2])
        self.assertIsNone(serve_compat.token_prompt("text", 10))
        self.assertIsNone(serve_compat.token_prompt(["a"], 10))
        self.assertIsNone(serve_compat.token_prompt([], 10))
        self.assertEqual(serve_compat.token_prompt([9], None), [9])          # no vocabulary known: only the sign is checked
        for bad in ([-1], [10], [True], [1.0], [[1], [2]], [1, [2]]):
            with self.assertRaises(serve_compat.CompatError, msg=bad):
                serve_compat.token_prompt(bad, 10)

    def test_usage_and_tokenize_reply(self):
        self.assertEqual(serve_compat.usage(4, 2), {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
        self.assertEqual(serve_compat.usage(4, 2, 9)["prompt_tokens_details"], {"cached_tokens": 4})
        self.assertEqual(serve_compat.usage(4, 2, -1)["prompt_tokens_details"], {"cached_tokens": 0})
        self.assertEqual(serve_compat.tokenize_reply([1], 8), {"count": 1, "max_model_len": 8, "tokens": [1], "token_strs": None})


def load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class SimEngine:
    """Fake engine with a whole-prompt cache: a prompt seen since the last reset reports cached_tokens == its length."""
    ctx = 20000
    vocab_size = 1000
    supports_vision = False
    spec_on = True
    sessions = 1

    def __init__(self):
        self.cache, self.resets, self.salted_seen = set(), 0, []
        self.last_stats = {}

    def encode_ids(self, text, add_special_tokens=False):
        return [1 + zlib.crc32(w.encode()) % 900 for w in text.split()]

    def count_tokens(self, text):
        return len(self.encode_ids(text))

    def spec_stats(self):
        return {}

    async def reset_prefix_cache(self):
        self.cache.clear()
        self.resets += 1

    async def generate(self, prompt, stats=None, prompt_ids=None, ignore_eos=False, **kw):
        ids = tuple(prompt_ids) if prompt_ids is not None else tuple(self.encode_ids(prompt))
        cached = len(ids) if ids in self.cache else 0
        self.cache.add(ids)
        st = {"cached_tokens": cached, "prompt_tokens": len(ids), "new_tokens": 6, "eos_reason": "stop_token" if not ignore_eos else "max_new_tokens",
              "time_prefill": 0.01, "time_generate": 0.05}
        self.last_stats = st
        if stats is not None:
            stats.update(st)
        for piece in ("one ", "two ", "three ", "four ", "five ", "six"):
            await asyncio.sleep(0.01)
            yield piece


class Live:
    """An aiohttp app on 127.0.0.1 in its own thread, for the urllib client of RigMark."""

    def __init__(self, app):
        from aiohttp import web
        self.loop = asyncio.new_event_loop()
        self.runner = web.AppRunner(app)
        self.loop.run_until_complete(self.runner.setup())
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        self.loop.run_until_complete(site.start())
        self.port = site._server.sockets[0].getsockname()[1]
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.run_until_complete(self.runner.cleanup())
        self.loop.close()


@unittest.skipUnless((RIGMARK / "bench.py").exists(), "RigMark source not found (set RIGMARK_SRC)")
class RigmarkDriveTests(unittest.TestCase):
    """RigMark's own prefill, decode and concurrency code against each server."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(RIGMARK))
        cls.bench = load("rigmark_bench", RIGMARK / "bench.py")
        cls.prompts, _ = cls.bench.load_prompts(RIGMARK / "prompts.json")
        cls.template = "{% for m in messages %}{{ m.content }} {% endfor %}answer:"

    def servers(self):
        qwen = load("qwen_serve", HERE / "qwen" / "serve.py")
        glm = load("glm_serve", HERE / "glm" / "serve.py")
        return {"qwen": qwen.create_app, "glm": glm.create_app}

    def run_server(self, name, fn):
        engine = SimEngine()
        live = Live(self.servers()[name](engine, "m", self.template))
        try:
            client = self.bench.Client(f"http://127.0.0.1:{live.port}", "", 60)
            return fn(client, engine)
        finally:
            live.close()

    def test_prefill_phase_cold_then_replay_per_salt(self):
        for name in ("qwen", "glm"):
            def go(client, engine):
                return self.bench.run_prefill(client, "m", self.prompts, [512, 1024], 2, "cmp-test"), engine
            result, engine = self.run_server(name, go)
            for depth in ("512", "1024"):
                for cold, warm in zip(result[depth]["cold"]["runs"], result[depth]["warm_replay"]["runs"]):
                    self.assertEqual(cold["prompt_tokens"], int(depth), name)
                    self.assertEqual(cold["requested_prompt_tokens"], int(depth), name)
                    self.assertEqual(cold["cached_prompt_tokens"], 0, name)
                    self.assertEqual(warm["cached_prompt_tokens"], int(depth), name)
                    self.assertEqual(len(cold["cache_salt_sha256"]), 64)
            self.assertEqual(engine.resets, 4, name)         # one reset per cold request (a new salt), none for a replay

    def test_prefill_without_a_salt_would_hit_the_cache(self):
        """Control: the same ids twice under one salt do hit; RigMark's cold check is what the salt reset satisfies."""
        for name in ("qwen", "glm"):
            def go(client, engine):
                ids = self.bench.exact_token_ids(client, "m", 600, self.prompts["prefill_unit"], "n1")
                a = self.bench.prefill_once(client, "m", ids, "same")
                b = self.bench.prefill_once(client, "m", ids, "same")
                c = self.bench.prefill_once(client, "m", ids, "other")
                return a, b, c
            a, b, c = self.run_server(name, go)
            self.assertEqual((a["cached_prompt_tokens"], b["cached_prompt_tokens"], c["cached_prompt_tokens"]), (0, 600, 0), name)

    def test_decode_and_concurrency_fields_are_accepted(self):
        extra = {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}
        for name in ("qwen", "glm"):
            def go(client, engine):
                dec = self.bench.run_decode(client, "m", self.prompts, 1, 64, 20260905, extra, "cmp-test")
                con = self.bench.run_concurrency(client, "m", self.prompts, [1, 2], 1, 32, 20260905, extra, "code", "cmp-test")
                return dec, con
            dec, con = self.run_server(name, go)
            for workload in dec.values():
                row = workload["runs"][0]
                self.assertEqual(row["finish_reason"], "stop", name)
                self.assertTrue(row["stream_done_marker"], name)
                self.assertGreater(row["prompt_tokens"], 0)
            self.assertEqual(sorted(con), ["1", "2"])

    def test_rigmark_payload_fields_pass_both_chat_endpoints(self):
        payload = self.bench.build_chat_payload("m", "sys", "hi", 16, 1, {"chat_template_kwargs": {"enable_thinking": False}})
        for name in ("qwen", "glm"):
            def go(client, engine):
                return client.stream("/v1/chat/completions", payload)
            row = self.run_server(name, go)
            self.assertEqual(row["finish_reason"], "stop", name)


if __name__ == "__main__":
    unittest.main()
