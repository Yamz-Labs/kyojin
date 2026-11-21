"""HTTP-level CPU test of tools/hy4/serve.py with a fake engine (no GPU, no model): routes, streaming, error bodies, queueing.
Run: python -m unittest tests/test_hy4_serve_http.py   (needs aiohttp + jinja2 + tokenizers)"""
import argparse
import asyncio
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "hy4"))
sys.path.insert(0, HERE)
import serve as S  # noqa: E402
import serve_core as C  # noqa: E402
from test_hy4_serve_core import tiny_meta  # noqa: E402

try:
    from aiohttp.test_utils import TestClient, TestServer
except ImportError:      # pragma: no cover
    TestClient = None


class FakeEngine:
    """replies with a fixed text, 3 tokens per round, like the speculative loop"""

    def __init__(self, vocab, reply):
        self.ids = vocab.encode(reply)
        self.calls = []

    def generate(self, ids, max_new, on_tokens, speculative=True, cancel=lambda: False):
        self.calls.append((list(ids), max_new, speculative))
        out, n = self.ids, 0
        stop = on_tokens(out[:1])
        n = 1
        rounds = 0
        while n < max_new and not stop and n < len(out):
            new = out[n:n + 3][:max_new - n]
            n += len(new)
            rounds += 1
            stop = on_tokens(new)
        return {"prompt_tokens": len(ids), "prefill_s": 0.5, "prefill_trunk_s": 0.4, "prime_s": 0.1, "prime_mode": "fake", "decode_s": 0.25,
                "decode_tokens": n - 1, "ttft_s": 0.5, "rounds": rounds, "accept_hist": [0, 0, 0, rounds], "tau": (n - 1) / max(rounds, 1), "speculative": speculative}


def make_state(reply, ctx=4096):
    args = argparse.Namespace(model_id="Hy4-test", max_ctx=ctx, num_draft=3, default_max_tokens=64)
    st = S.State(args)
    st.vocab = C.Vocab(tiny_meta())
    st.engine = FakeEngine(st.vocab, reply)
    st.ready = True
    return st


@unittest.skipIf(TestClient is None, "aiohttp not installed")
class Http(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.st = make_state("Bonjour le monde")
        self.st.lock = asyncio.Lock()
        self.client = TestClient(TestServer(S.build_app(self.st)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_health_models(self):
        r = await self.client.get("/health")
        self.assertEqual((r.status, (await r.json())["status"]), (200, "ok"))
        self.st.ready = False
        r = await self.client.get("/health")
        self.assertEqual(r.status, 503)
        self.st.ready = True
        m = await (await self.client.get("/v1/models")).json()
        self.assertEqual(m["data"][0]["id"], "Hy4-test")

    async def test_chat_non_stream(self):
        r = await self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "salut"}], "max_tokens": 50, "reasoning_effort": "no_think"})
        j = await r.json()
        self.assertEqual(r.status, 200, j)
        self.assertEqual(j["choices"][0]["message"]["content"], "Bonjour le monde")
        self.assertEqual(j["choices"][0]["finish_reason"], "length")
        self.assertEqual(j["usage"]["completion_tokens"], len(self.st.engine.ids))
        self.assertAlmostEqual(j["timings"]["prompt_per_second"], j["usage"]["prompt_tokens"] / 0.5)
        ids = self.st.engine.calls[0][0]
        self.assertEqual(ids[0], 256)       # <｜hy_start...｜> is one id

    async def test_chat_default_effort_splits_reasoning(self):
        self.st.engine = FakeEngine(self.st.vocab, "pense</think:6124c78e>Reponse")
        r = await self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}], "max_tokens": 50})
        j = await r.json()
        self.assertEqual((j["choices"][0]["message"]["reasoning_content"], j["choices"][0]["message"]["content"]), ("pense", "Reponse"))

    async def test_chat_stream(self):
        r = await self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}], "max_tokens": 50, "stream": True,
                                                                 "reasoning_effort": "no_think", "stream_options": {"include_usage": True}})
        self.assertEqual(r.status, 200)
        raw = (await r.read()).decode()
        events = [e[6:] for e in raw.split("\n\n") if e.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        self.assertEqual(text, "Bonjour le monde")
        self.assertEqual([c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1], "length")
        self.assertIn("usage", chunks[-1])

    async def test_completion_text_and_ids(self):
        r = await self.client.post("/v1/completions", json={"prompt": "Il etait", "max_tokens": 4, "speculative": False})
        j = await r.json()
        self.assertEqual((r.status, j["choices"][0]["text"], j["choices"][0]["finish_reason"]), (200, "Bonj", "length"))
        self.assertFalse(self.st.engine.calls[-1][2])
        r = await self.client.post("/v1/completions", json={"prompt": [65, 66], "max_tokens": 2})
        self.assertEqual(r.status, 200)
        self.assertEqual(self.st.engine.calls[-1][0], [65, 66])

    async def test_stop_string(self):
        r = await self.client.post("/v1/completions", json={"prompt": "x", "max_tokens": 50, "stop": ["le"]})
        j = await r.json()
        self.assertEqual((j["choices"][0]["text"], j["choices"][0]["finish_reason"]), ("Bonjour ", "stop"))

    async def test_errors(self):
        for path, body, status in (("/v1/chat/completions", {"messages": []}, 400), ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "tools": [1]}, 400),
                                   ("/v1/completions", {"prompt": ""}, 400)):
            r = await self.client.post(path, json=body)
            j = await r.json()
            self.assertEqual(r.status, status, j)
            self.assertIn("error", j)
        r = await self.client.post("/v1/completions", data="not json")
        self.assertEqual(r.status, 400)
        self.st.ready = False
        r = await self.client.post("/v1/completions", json={"prompt": "x"})
        self.assertEqual(r.status, 503)

    async def test_tokenize_and_template(self):
        j = await (await self.client.post("/tokenize", json={"content": "ab"})).json()
        self.assertEqual(j["tokens"], [97, 98])
        j = await (await self.client.post("/apply-template", json={"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "no_think"})).json()
        self.assertTrue(j["prompt"].endswith("</think:6124c78e>"))


if __name__ == "__main__":
    unittest.main()
