"""CPU tests of the conversion step: template carry-over, argv, memory guard pause/resume on a real child process."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import backends  # noqa: E402
import convert_job as cj  # noqa: E402


class ConvertTests(unittest.TestCase):
    def test_argv_resume_and_extra(self):
        a = cj.convert_argv("/s", "/w", "/p", 4.0, 6, True, ["-hq"])
        self.assertEqual(a, ["convert.py", "-i", "/s", "-w", "/w", "-o", "/p", "-b", "4.0", "-hb", "6", "-r", "-hq"])

    def test_template_carried_from_file_then_tokenizer_config(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); src, pack = t / "src", t / "pack"; src.mkdir(); pack.mkdir()
            (src / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ x }}"}))
            self.assertEqual(cj.carry_template(src, pack), "tokenizer_config")
            self.assertEqual((pack / "chat_template.jinja").read_text(), "{{ x }}")
            self.assertEqual(cj.carry_template(src, pack), "pack")                 # never overwrites the pack's own
            (pack / "chat_template.jinja").unlink(); (src / "chat_template.jinja").write_text("FILE")
            self.assertEqual(cj.carry_template(src, pack), "source file")
            (pack / "chat_template.jinja").unlink(); (src / "chat_template.jinja").unlink(); (src / "tokenizer_config.json").write_text("{}")
            self.assertIsNone(cj.carry_template(src, pack))

    def test_pack_complete_needs_config_quant_index_and_shards(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            self.assertFalse(cj.pack_complete(p))
            (p / "config.json").write_text("{}"); (p / "quantization_config.json").write_text("{}")
            (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s.safetensors"}}))
            self.assertFalse(cj.pack_complete(p))                                  # shard missing
            (p / "s.safetensors").write_bytes(b"x")
            self.assertTrue(cj.pack_complete(p))

    def test_single_shard_pack_without_index_is_complete(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            (p / "config.json").write_text("{}"); (p / "quantization_config.json").write_text("{}")
            self.assertFalse(cj.pack_complete(p))
            (p / "model.safetensors").write_bytes(b"x")
            self.assertTrue(cj.pack_complete(p))

    def test_memguard_stops_and_continues_only_its_group(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        real = backends.mem_available_gib
        try:
            logs = []
            backends.mem_available_gib = lambda: 1.0                               # below the pause line
            g = cj.MemGuard(proc.pid, 6, 10, logs.append, poll_s=0.05); g.start()
            time.sleep(0.4)
            self.assertTrue(g.stopped)
            self.assertIn("T", Path(f"/proc/{proc.pid}/stat").read_text().split(") ")[1][:1])   # state T = stopped
            backends.mem_available_gib = lambda: 50.0
            time.sleep(0.4)
            self.assertFalse(g.stopped)
            self.assertNotIn("T", Path(f"/proc/{proc.pid}/stat").read_text().split(") ")[1][:1])
            self.assertEqual(g.pauses, 1)
            g.finish(); g.join(1)
        finally:
            backends.mem_available_gib = real
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()


if __name__ == "__main__":
    unittest.main()
