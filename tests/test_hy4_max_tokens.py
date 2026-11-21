"""Reply budget of the Hy4 server (issue #222: an agent client's context compaction ended with stop reason `length`).
A request without max_tokens gets all the room left in the context window (--default-max-tokens 0, the default); a positive default is a cap;
a max_tokens larger than the room is clipped to it (no 400); only a prompt with no room for one token is a 400. Chat and completions, streamed or not.
CPU test of tools/hy4/serve.py with a fake engine (no GPU, no model).
Run: python -m unittest tests/test_hy4_max_tokens.py   (needs aiohttp + jinja2 + tokenizers)"""
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

CTX = 16384
DRAFT = 3
FILL = 97                       # byte id of "a"
TMO = 20
CHAT = "/v1/chat/completions"
COMP = "/v1/completions"


class FillEngine:
    """generates `fill` tokens of "a" (None = never stops) or max_new of them, whichever is smaller; ends with the EOS token when it stops by itself.
    Records the max_new it was given and refuses a call that would not fit the context window, as the real engine does."""

    def __init__(self, ctx, k, fill=None, eos=258):
        self.ctx, self.k, self.fill, self.eos, self.calls = ctx, k, fill, eos, []

    def generate(self, ids, max_new, on_tokens, speculative=True, cancel=lambda: False):
        assert len(ids) + max_new + self.k + 2 <= self.ctx, (len(ids), max_new, self.ctx)
        self.calls.append((len(ids), max_new))
        total = max_new if self.fill is None else min(self.fill, max_new)
        n, stop = 0, False
        while n < total and not stop:
            step = min(64, total - n)
            n += step
            stop = on_tokens([FILL] * step)
        if self.fill is not None and self.fill < max_new and not stop:
            on_tokens([self.eos])
        return {"prompt_tokens": len(ids), "prefill_s": 0.5, "prefill_trunk_s": 0.4, "prime_s": 0.1, "prime_mode": "fake", "decode_s": 0.25,
                "decode_tokens": n, "ttft_s": 0.5, "rounds": 1, "accept_hist": [0, 0, 0, 1], "tau": 1.0, "speculative": speculative}


def make_state(default_max=0, ctx=CTX, fill=None):
    args = argparse.Namespace(model_id="Hy4-test", max_ctx=ctx, num_draft=DRAFT, default_max_tokens=default_max)
    st = S.State(args)
    st.vocab = C.Vocab(tiny_meta())
    st.engine = FillEngine(ctx, DRAFT, fill=fill)
    st.ready = True
    st.lock = asyncio.Lock()
    return st


def long_text(n_tokens):
    """a text of exactly n_tokens tokens in the byte-level test vocabulary"""
    return "b" * n_tokens


def chat_body(n_prompt, **kw):
    return dict(messages=[{"role": "user", "content": long_text(n_prompt)}], reasoning_effort="no_think", **kw)


@unittest.skipIf(TestClient is None, "aiohttp not installed")
class MaxTokens(unittest.IsolatedAsyncioTestCase):
    async def start(self, **kw):
        self.st = make_state(**kw)
        self.client = TestClient(TestServer(S.build_app(self.st)))
        await self.client.start_server()
        return self.st

    async def asyncTearDown(self):
        await self.client.close()

    async def call(self, path, body):
        """-> (status, finish_reason, completion_tokens or None, error message or None); a streamed reply is read to [DONE]"""
        async def go():
            r = await self.client.post(path, json=body)
            raw = await r.read()
            return r.status, raw
        status, raw = await asyncio.wait_for(go(), TMO)
        if status != 200:
            return status, None, None, json.loads(raw)["error"]["message"]
        if not body.get("stream"):
            j = json.loads(raw)
            return status, j["choices"][0]["finish_reason"], j["usage"]["completion_tokens"], None
        chunks = [json.loads(l[6:]) for l in raw.decode().split("\n\n") if l.startswith("data: {")]
        finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
        usage = [c["usage"]["completion_tokens"] for c in chunks if c.get("usage")]
        self.assertTrue(raw.rstrip().endswith(b"[DONE]"))
        return status, finish[-1], (usage[-1] if usage else None), None

    def prompt_len(self, body, path):
        """number of prompt tokens the server counts for this body"""
        if path == CHAT:
            r = C.parse_chat(body, 0)
            return len(self.st.vocab.encode(self.st.vocab.chat_prompt(r)[0]))
        return len(self.st.vocab.encode(body["prompt"]))

    def room(self, n):
        return CTX - n - DRAFT - 2

    def cases(self, n, **kw):
        """the same request on both endpoints, streamed and not"""
        for stream in (False, True):
            extra = dict(kw, stream=stream, stream_options={"include_usage": True}) if stream else dict(kw)
            yield CHAT, chat_body(n, **extra)
            yield COMP, dict(prompt=long_text(n), **extra)

    # -- no max_tokens: the whole room left
    async def test_no_max_tokens_long_prompt_gets_the_room(self):
        await self.start(default_max=0)
        for path, body in self.cases(6000):
            with self.subTest(path=path, stream=body.get("stream", False)):
                self.st.engine.calls.clear()
                status, finish, n_gen, err = await self.call(path, body)
                self.assertEqual(status, 200, err)
                n = self.prompt_len(body, path)
                self.assertGreater(n, 6000 - 1)
                self.assertEqual(self.st.engine.calls, [(n, self.room(n))])
                self.assertEqual(finish, "length")            # the filler never stops: it used all the budget
                self.assertGreater(self.room(n), 1024)
                if n_gen is not None:
                    self.assertEqual(n_gen, self.room(n))

    async def test_no_max_tokens_reply_shorter_than_room_stops(self):
        await self.start(default_max=0, fill=3000)
        for path, body in self.cases(6000):
            with self.subTest(path=path, stream=body.get("stream", False)):
                status, finish, n_gen, err = await self.call(path, body)
                self.assertEqual(status, 200, err)
                self.assertEqual(finish, "stop")
                if n_gen is not None:
                    self.assertGreater(n_gen, 1024)          # not cut at the old default

    async def test_no_max_tokens_short_prompt(self):
        await self.start(default_max=0)
        for path, body in self.cases(10):
            n = self.prompt_len(body, path)
            self.st.engine.calls.clear()
            status, finish, _, err = await self.call(path, body)
            self.assertEqual((status, finish), (200, "length"), err)
            self.assertEqual(self.st.engine.calls, [(n, self.room(n))])

    # -- explicit max_tokens larger than the room: clipped
    async def test_explicit_too_large_is_clipped_not_400(self):
        await self.start(default_max=0)
        for big in (100000, CTX, CTX - 100):
            for path, body in self.cases(6000, max_tokens=big):
                with self.subTest(big=big, path=path, stream=body.get("stream", False)):
                    self.st.engine.calls.clear()
                    status, finish, _, err = await self.call(path, body)
                    self.assertEqual(status, 200, err)
                    n = self.prompt_len(body, path)
                    self.assertEqual(self.st.engine.calls, [(n, min(big, self.room(n)))])
                    self.assertEqual(finish, "length")

    async def test_max_completion_tokens_alias_is_clipped_too(self):
        await self.start(default_max=0)
        status, finish, _, err = await self.call(CHAT, chat_body(6000, max_completion_tokens=100000))
        self.assertEqual((status, finish), (200, "length"), err)
        n = self.prompt_len(chat_body(6000), CHAT)
        self.assertEqual(self.st.engine.calls, [(n, self.room(n))])
        self.st.engine.calls.clear()
        status, _, _, err = await self.call(CHAT, chat_body(6000, max_completion_tokens=40, max_tokens=100000))      # the newer name wins
        self.assertEqual(status, 200, err)
        self.assertEqual(self.st.engine.calls, [(n, 40)])

    # -- explicit small: unchanged
    async def test_explicit_small_is_unchanged(self):
        await self.start(default_max=0)
        for path, body in self.cases(6000, max_tokens=50):
            with self.subTest(path=path, stream=body.get("stream", False)):
                self.st.engine.calls.clear()
                status, finish, n_gen, err = await self.call(path, body)
                self.assertEqual((status, finish), (200, "length"), err)
                self.assertEqual(self.st.engine.calls[0][1], 50)
                if n_gen is not None:
                    self.assertEqual(n_gen, 50)

    # -- --default-max-tokens N > 0 keeps the cap
    async def test_positive_default_is_a_cap(self):
        await self.start(default_max=1024)
        for path, body in self.cases(6000):
            with self.subTest(path=path, stream=body.get("stream", False)):
                self.st.engine.calls.clear()
                status, finish, n_gen, err = await self.call(path, body)
                self.assertEqual((status, finish), (200, "length"), err)
                self.assertEqual(self.st.engine.calls[0][1], 1024)
                if n_gen is not None:
                    self.assertEqual(n_gen, 1024)
        # an explicit value is still honoured above the cap (up to the room), and a cap larger than the room is clipped
        self.st.engine.calls.clear()
        status, _, _, err = await self.call(CHAT, chat_body(6000, max_tokens=3000))
        self.assertEqual(status, 200, err)
        self.assertEqual(self.st.engine.calls[0][1], 3000)
        self.st.engine.calls.clear()
        self.st.args.default_max_tokens = 50000
        status, _, _, err = await self.call(CHAT, chat_body(6000))
        n = self.prompt_len(chat_body(6000), CHAT)
        self.assertEqual((status, self.st.engine.calls), (200, [(n, self.room(n))]), err)

    # -- a prompt that does not fit is a 400, with or without max_tokens
    async def test_prompt_that_does_not_fit_is_400(self):
        await self.start(default_max=0)
        for n_prompt in (CTX + 500, CTX, CTX - 3, CTX - DRAFT - 2):
            for extra in ({}, {"max_tokens": 10}):
                for path, body in self.cases(n_prompt, **extra):
                    with self.subTest(n=n_prompt, extra=extra, path=path, stream=body.get("stream", False)):
                        status, _, _, err = await self.call(path, body)
                        self.assertEqual(status, 400, err)
                        self.assertIn("does not fit the context window", err)
                        self.assertIn(str(CTX), err)
        self.assertEqual(self.st.engine.calls, [])                  # a refused request never reached the engine
        status, finish, _, err = await self.call(CHAT, chat_body(10, max_tokens=5))      # and the server answers the next one
        self.assertEqual((status, finish), (200, "length"), err)

    async def test_one_token_of_room_is_served(self):
        await self.start(default_max=0)
        body = dict(prompt=long_text(CTX - DRAFT - 3))
        status, finish, _, err = await self.call(COMP, body)
        self.assertEqual((status, finish), (200, "length"), err)
        self.assertEqual(self.st.engine.calls, [(CTX - DRAFT - 3, 1)])


@unittest.skipIf(TestClient is None, "aiohttp not installed")
class Flag(unittest.TestCase):
    def test_flag_default_is_zero_and_negative_is_refused(self):
        import io
        from contextlib import redirect_stderr
        with unittest.mock.patch.object(S, "apply_env", side_effect=SystemExit("stop here")), unittest.mock.patch.object(S, "check_kernels"):
            with self.assertRaises(SystemExit) as cm:
                S.main(["--model", "x.gguf"])
            self.assertEqual(str(cm.exception), "stop here")
            err = io.StringIO()
            with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                S.main(["--model", "x.gguf", "--default-max-tokens", "-5"])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--default-max-tokens", err.getvalue())

    def test_parse_default_zero_means_no_budget_yet(self):
        self.assertIsNone(C.parse_chat({"messages": [{"role": "user", "content": "x"}]}, 0)["max_tokens"])
        self.assertIsNone(C.parse_completion({"prompt": "x"}, 300, 0)["max_tokens"])
        self.assertEqual(C.parse_chat({"messages": [{"role": "user", "content": "x"}], "max_tokens": 7}, 0)["max_tokens"], 7)
        self.assertEqual(C.parse_chat({"messages": [{"role": "user", "content": "x"}]}, 512)["max_tokens"], 512)
        with self.assertRaises(C.RequestError):
            C.parse_chat({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, 0)


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
