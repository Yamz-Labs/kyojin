#!/usr/bin/env python3
"""POST /admin/knobs on the Qwen and GLM servers: off unless --admin-knobs, loopback only, drains the requests, switches the knob.

    python -m unittest tools.test_admin_knobs        (model-free: a fake engine, like tools/test_serve_logging.py)
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_serve_logging import FakeEngine, load_server  # noqa: E402


class Engine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def run_exclusive(self, fn, *args):
        self.calls.append("run")
        return fn(*args)

    async def reset_prefix_cache(self) -> None:
        self.calls.append("reset")


class AdminKnobs(unittest.TestCase):
    def setUp(self) -> None:
        for n in ("EXL3_GEMV_TRI",):
            os.environ.pop(n, None)

    def tearDown(self) -> None:
        os.environ.pop("EXL3_GEMV_TRI", None)

    def test_both_servers(self) -> None:
        for name in ("qwen", "glm"):
            with self.subTest(server=name):
                os.environ.pop("EXL3_GEMV_TRI", None)
                serve = load_server(name)
                self.assertIn("/admin/knobs", serve.ROUTES)

                async def go() -> None:
                    from aiohttp.test_utils import TestClient, TestServer
                    eng = Engine()
                    off = TestClient(TestServer(serve.create_app(eng, "m", "{{ messages[0].content }}")))
                    await off.start_server()
                    try:
                        self.assertEqual((await off.get("/admin/knobs")).status, 404)       # opt-in
                    finally:
                        await off.close()
                    on = TestClient(TestServer(serve.create_app(eng, "m", "{{ messages[0].content }}", admin_knobs=True)))
                    await on.start_server()
                    try:
                        r = await on.post("/admin/knobs", json={"EXL3_GEMV_TRI": 0})
                        body = await r.json()
                        self.assertEqual(r.status, 200)
                        self.assertEqual(body["effective"]["EXL3_GEMV_TRI"], "0")
                        self.assertEqual(os.environ["EXL3_GEMV_TRI"], "0")
                        self.assertEqual(eng.calls, ["run", "reset"])
                        r = await on.post("/admin/knobs", json={"EXL3_GEMV_TRI": 0}, headers={"X-Forwarded-For": "1.2.3.4"})
                        self.assertEqual(r.status, 403)
                        r = await on.post("/admin/knobs", json={"NOPE": 1})
                        self.assertEqual(r.status, 400)
                        r = await on.get("/admin/knobs")
                        self.assertEqual((await r.json())["effective"]["EXL3_GEMV_TRI"], "0")
                    finally:
                        await on.close()

                asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
