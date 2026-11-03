"""CPU tests of the day-0 autotune: shape enumeration from safetensors headers, row classes, tuning-cache file format/merge, planner."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

import autotune  # noqa: E402
import plan as planmod  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("tune_cache", HERE.parents[2] / "exllamav3/util/tune_cache.py")
tc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tc)    # no torch needed for the file format


def write_st(path: Path, tensors: dict):
    hdr = {k: {"dtype": "I16", "shape": list(v), "data_offsets": [0, 0]} for k, v in tensors.items()}
    b = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(b)) + b)


def rec(h, tag=1, bd=128, sms=82, conc=1):
    return tc.REC.pack(h, tag, bd, sms, conc, 0, 0)


class T(unittest.TestCase):
    def test_row_class_mirrors_hash(self):
        self.assertEqual([autotune.row_class(m) for m in (1, 2, 3, 5, 9, 16, 17, 24, 25, 33, 64, 65, 192)], [2, 2, 4, 8, 16, 16, 24, 24, 32, 48, 64, 64, 64])
        self.assertEqual(sorted({autotune.row_class(m) for m in autotune.ROW_CLASSES}), [2, 4, 8, 16, 24, 32, 48, 64])

    def test_shapes_from_headers(self):
        with tempfile.TemporaryDirectory() as d:
            t = {"m.layers.0.q.trellis": (448, 448, 64), "m.layers.0.q.mul1": (), "m.layers.1.q.trellis": (448, 448, 64), "m.layers.1.q.mul1": (),
                 "m.layers.0.g.trellis": (448, 1376, 96), "m.layers.0.g.mcg": (), "m.layers.0.e.experts.3.trellis": (64, 64, 48), "m.embed.weight": (10, 10)}
            write_st(Path(d) / "a.safetensors", t)
            s = autotune.pack_shapes(d)
            self.assertEqual(s["dense"], {(7168, 7168, 4.0, False, True): 2, (7168, 22016, 6.0, True, False): 1})
            self.assertEqual(len(s["experts"]), 1)
            p = autotune.tune_plan(s)
            self.assertEqual(len(p), 2 * len(autotune.ROW_CLASSES) * 2)
            self.assertEqual(p[0][:2], (7168, 22016))       # biggest first

    def test_half_integer_k_skipped(self):
        p = autotune.tune_plan({"dense": {(128, 128, 2.5, False, True): 1}, "experts": {}})
        self.assertEqual(p, [])

    def test_cache_roundtrip_and_merge(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            tc.write_records(d / "pack.bin", {1: rec(1), 2: rec(2, tag=5)})
            self.assertEqual(sorted(tc.read_records(d / "pack.bin")), [1, 2])
            self.assertEqual(tc.merge_into(d / "live.bin", [d / "pack.bin"]), 2)          # empty live: file created with header
            self.assertEqual(tc.merge_into(d / "live.bin", [d / "pack.bin"]), 0)          # idempotent
            tc.write_records(d / "p2.bin", {2: rec(2, tag=9), 3: rec(3)})
            self.assertEqual(tc.merge_into(d / "live.bin", [d / "p2.bin"]), 1)            # live wins, only key 3 is new
            r = tc.read_records(d / "live.bin")
            self.assertEqual(sorted(r), [1, 2, 3])
            self.assertEqual(tc.decode(r[2])["tag"], 5)

    def test_reader_skips_embedded_header_and_torn_tail(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "c.bin"
            hdr = tc.HDR.pack(tc.MAGIC, tc.FORMAT, tc.REC.size)
            f.write_bytes(hdr + rec(1) + hdr + rec(2) + b"\x01\x02\x03")
            self.assertEqual(sorted(tc.read_records(f)), [1, 2])
            f.write_bytes(b"garbage" * 10)
            self.assertEqual(tc.read_records(f), {})

    def test_install_off_switch_and_missing_dir(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            os.environ["EXLLAMAV3_TUNE_CACHE"] = str(Path(d) / "live.bin")
            try:
                self.assertEqual(tc.install_pack(d), 0)                                   # no tuning dir: no-op
                os.environ["EXL3_PACK_TUNE"] = "0"
                self.assertEqual(tc.install_pack(d), 0)
            finally:
                os.environ.pop("EXL3_PACK_TUNE", None); os.environ.pop("EXLLAMAV3_TUNE_CACHE", None)

    def test_stage_in_plan(self):
        self.assertIn("autotune", planmod.ORDER)
        self.assertLess(planmod.ORDER.index("meta"), planmod.ORDER.index("autotune"))
        self.assertLess(planmod.ORDER.index("autotune"), planmod.ORDER.index("serve"))


if __name__ == "__main__":
    unittest.main()


class TestReplay(unittest.TestCase):
    def test_sweep_covers_every_row_class(self):
        import replay
        rq = replay.sweep_requests(sessions=4, ctx=16384)
        rows = {autotune.row_class(r["tokens"]) for r in rq if r["kind"] == "prompt"} | {autotune.row_class(r["tokens"] - replay.CHUNK) for r in rq if r["kind"] == "tail"}
        rows |= {autotune.row_class(r["conc"]) for r in rq if r["kind"] == "decode"}
        self.assertEqual(rows, {2, 4, 8, 16, 24, 32, 48, 64})
        self.assertEqual([r["conc"] for r in rq if r["kind"] == "decode"], [2, 3, 4])
        self.assertTrue(all(r["tokens"] <= 16384 - 64 for r in rq))

    def test_sweep_respects_small_ctx(self):
        import replay
        rq = replay.sweep_requests(sessions=1, ctx=512)
        self.assertTrue(all(r["tokens"] <= 448 for r in rq))
        self.assertFalse(any(r["kind"] == "decode" for r in rq))

    def test_token_prompts_distinct_and_ordinary(self):
        import replay
        a, b = replay.token_prompt(64, 1), replay.token_prompt(64, 2)
        self.assertNotEqual(a[:8], b[:8])
        self.assertEqual(a, replay.token_prompt(64, 1))
        self.assertTrue(all(1000 <= t < 30000 for t in a))

    def test_run_sweep_against_fake_server(self):
        import replay
        import subprocess, time, urllib.request
        port = 8597
        p = subprocess.Popen([sys.executable, str(HERE / "fake_serve.py"), "--port", str(port), "-c", "4096", "--sessions", "3"])
        try:
            for _ in range(50):
                try:
                    if urllib.request.urlopen(f"http://127.0.0.1:{port}/health").status == 200:
                        break
                except Exception:
                    time.sleep(0.2)
            res = replay.run_sweep(port, 3, 4096, log=lambda *_: None)
            self.assertTrue(all(r["ok"] for r in res), [r for r in res if not r["ok"]][:2])
        finally:
            p.terminate(); p.wait(10)


class TestDense(unittest.TestCase):
    def test_dense_text_merge_keeps_live_lines(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "live.txt").write_text("1 2 3 4 5 6 7\n")
            (d / "pack.txt").write_text("1 2 3 4 5 6 9\n7 7 7 7 7 7 7\nbad line\n")
            self.assertEqual(tc.merge_dense_text(d / "live.txt", d / "pack.txt"), 1)
            self.assertEqual((d / "live.txt").read_text().splitlines(), ["1 2 3 4 5 6 7", "7 7 7 7 7 7 7"])
            self.assertEqual(tc.merge_dense_text(d / "new.txt", d / "pack.txt"), 2)


class TestPlan(unittest.TestCase):
    def test_autotune_skipped_without_tuner_backend(self):
        meta = dict(index_sha="x", problems=[])
        for be, want in (("cuda", "todo"), ("rocm", "todo"), ("mlx", "skip")):
            st = {s.name: s for s in planmod.build_plan(meta, dict(backend=be, sessions=[1]), {})}
            self.assertEqual(st["autotune"].status, want, be)


class TestDenseNew(unittest.TestCase):
    def test_only_lines_beyond_the_seed(self):
        import stages
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "seed.txt").write_text("a 1 2 3 4 5 9\n")
            (d / "t.txt").write_text("a 1 2 3 4 5 9\nb 1 2 3 4 5 8\n")
            self.assertEqual(stages._new_dense_lines(d / "t.txt", d / "seed.txt"), ["b 1 2 3 4 5 8"])
            self.assertEqual(stages._new_dense_lines(d / "none.txt", d / "seed.txt"), [])
