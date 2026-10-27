"""End-to-end tests of the GPU-stage code against a fake server (no GPU, no torch): max-ctx search with start-up OOM, served matrix, sha, conformance."""
from __future__ import annotations

import socket
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import backends  # noqa: E402
import load  # noqa: E402
import stages  # noqa: E402
from test_plan import fake_pack, args, packmod  # noqa: E402


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class FakeEnv:
    def __enter__(self):
        self._free = backends.gpu_free_mib
        backends.gpu_free_mib = lambda backend: 24000.0     # the real card is busy: fixed figure
        self.t = tempfile.TemporaryDirectory()
        t = Path(self.t.name)
        (t / "pack").mkdir()
        self.meta = packmod.read_pack(fake_pack(t / "pack"))
        self.meta["serve_entry"] = str(HERE / "fake_serve.py")
        self.meta["weights_bytes"] = 1
        self.a = args(port=free_port(), sessions=[1, 2], kv_kl_max=0.02, disk_floor_gb=1, out=str(t / "out"), repo=str(t), kv_bits=[8, 6], hf=None)
        (t / "out").mkdir()
        prof = dict(python=sys.executable, env={}, locks=[str(t / "gpu.lock")])
        logs = []
        self.c = stages.Ctx(self.meta, self.a, t / "out", t, prof, logs.append)
        self.logs = logs
        return self

    def __exit__(self, *e):
        backends.gpu_free_mib = self._free
        self.t.cleanup()


class GpuPathTests(unittest.TestCase):
    def test_client_calibrates_and_builds_near_full_prompts(self):
        with FakeEnv() as e:
            from server import running_server
            argv = [e.meta["serve_entry"], "--port", str(e.a["port"]), "-c", "30000"]
            with running_server(e.c.prof, e.c.repo, argv, e.a["port"], e.c.out / "s.log") as (p, ok):
                self.assertTrue(ok)
                cl = load.Client(e.a["port"], "qual")
                cpt = cl.calibrate()
                self.assertAlmostEqual(cpt, 4.0, delta=0.4)
                r = cl.near_full(30000, 0)
                self.assertFalse(r["err"]); self.assertGreater(r["ptok"], 0.9 * 30000 * 0.95)
                self.assertNotEqual(cl.prompt(1000, 0), cl.prompt(1000, 0))      # distinct every call
                self.assertTrue(cl.conformance()["ok"])
                agg = cl.concurrent(3, 2000, 16)
                self.assertEqual(len(agg["ttft"]), 3); self.assertEqual(agg["errors"], [])

    def test_maxctx_search_finds_the_startup_limit(self):
        import os
        os.environ["LIMIT"] = "45000"
        try:
            with FakeEnv() as e:
                e.a["sessions"] = [1]; e.a["ctx_cap"] = 65536
                e.meta["max_pos"] = 65536
                r = stages.job_maxctx(e.c)
                self.assertEqual(r["1"], 43008)                     # largest 2048-step value <= 45000
                self.assertFalse(any(not t["ok"] and "near-full" in t["note"] for t in r["tried_s1"]))
        finally:
            os.environ.pop("LIMIT")

    def test_serve_job_produces_matrix_and_sha(self):
        with FakeEnv() as e:
            e.c.data["maxctx"] = {"2": 24576}
            e.a["sessions"] = [1, 2]
            r = stages.job_serve(e.c, True, False)
            s = r["serve"]
            self.assertTrue(s["conformance"]["ok"]); self.assertTrue(s["sha_self_equal"]); self.assertFalse(s["oom"])
            self.assertEqual([a["n"] for a in s["agg"]], [1, 2])
            self.assertGreaterEqual(len(s["curve"]), 2)
            self.assertIn("--sessions", r["cmd"])


if __name__ == "__main__":
    unittest.main()
