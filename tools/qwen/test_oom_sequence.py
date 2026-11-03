#!/usr/bin/env python3
"""GPU regression: the driver sequence that ended in CUDA out of memory on the second distinct 32K prompt (3090, Qwen3.8-27B,
k6v6, --sessions 3, -c 183040: 1551 MiB left after load, the 150 MiB draft-head subset appears during the first 32K request).
A start that the VRAM guard accepts must survive it; a start the guard refuses must name a -c that survives it.
Opt-in: EXL3_GPU_TEST_MODEL=<pack dir> [EXL3_GPU_TEST_CTX=183040] [EXL3_GPU_TEST_SESSIONS=3] [EXL3_GPU_TEST_KV=6] [EXL3_GPU_TEST_FULL=1] [EXL3_GPU_TEST_PYTHON=...] python3 tools/qwen/test_oom_sequence.py"""
import json, os, re, socket, subprocess, sys, tempfile, time, unittest, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("EXL3_GPU_TEST_MODEL", "")
CTX = int(os.environ.get("EXL3_GPU_TEST_CTX", "183040"))
SESSIONS = os.environ.get("EXL3_GPU_TEST_SESSIONS", "3")
KV = os.environ.get("EXL3_GPU_TEST_KV", "6")       # K and V bits


def _text() -> str:
    return "\n".join(p.read_text(errors="replace") for p in sorted((ROOT / "exllamav3").rglob("*.py")))


def _ask(port: int, text: str, i: int, chars: int, max_new: int) -> dict:
    off = 7000 * (i + 1) * 3
    body = {"model": "t", "temperature": 0, "top_k": 1, "max_tokens": max_new, "stream": True, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"[SEQ{i}] Explain this extract:\n\n{text[off:off + chars]}"}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer x"})
    err, got = None, 0
    for line in urllib.request.urlopen(req, timeout=900):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        ev = json.loads(line[5:])
        err = ev.get("error") or err
        got += sum(len((c.get("delta") or {}).get("content") or "") for c in ev.get("choices", []))
    return {"error": err, "chars": got}


@unittest.skipUnless(MODEL, "set EXL3_GPU_TEST_MODEL to a Qwen3.8-27B EXL3 pack to run the GPU sequence")
class OomSequence(unittest.TestCase):
    def start(self, ctx: int):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        log = tempfile.NamedTemporaryFile("w+", suffix=".log")
        cmd = [os.environ.get("EXL3_GPU_TEST_PYTHON", sys.executable), "-u", str(ROOT / "tools/qwen/serve.py"), "--model", MODEL, "--port", str(port), "-c", str(ctx),
               "--k-bits", KV, "--v-bits", KV, "--sessions", SESSIONS, "--model-id", "t", "--default-max-tokens", "512"]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, EXL3_CUDA_GRAPH="1", PYTHONPATH=str(ROOT)), cwd=ROOT)
        self.addCleanup(lambda: (proc.terminate(), proc.wait(timeout=60)))
        for _ in range(400):
            if proc.poll() is not None:
                log.seek(0); return None, log.read()
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200:
                    return port, log
            except OSError:
                pass
            time.sleep(3)
        self.fail("server did not become healthy")

    def test_distinct_32k_prompts_after_the_short_and_4k_warm_up(self):
        port, log = self.start(CTX)
        ctx = CTX
        for _ in range(4):                  # the guard refused: it must name a context that works (each refusal lowers it, the figure is measured after load)
            if port is not None:
                break
            m = re.findall(r"Start with -c (\d+)", log)
            self.assertTrue(m, log[-800:])
            self.assertLess(int(m[-1]), ctx)
            ctx = int(m[-1])
            port, log = self.start(ctx)
        self.assertIsNotNone(port, log)
        print(f"FINAL_CTX={ctx} sessions={SESSIONS} kv={KV}", flush=True)
        text = _text()
        for i in range(3):
            self.assertIsNone(_ask(port, text, i, 700, 256)["error"])
        for i in range(3):
            self.assertIsNone(_ask(port, text, 10 + i, 10000, 256)["error"])
        for i in range(3):                  # 80K chars is about 31K tokens; the third one is the one the lane sequence never reached
            r = _ask(port, text, 20 + i, 80000, 256)
            self.assertIsNone(r["error"], f"32K prompt {i}")
        if os.environ.get("EXL3_GPU_TEST_FULL"):      # two distinct prompts that fill the context (1 session k6v6 -c 262144 died on the second one)
            for i in range(2):
                r = _ask(port, text, 40 + i, int((ctx - 1500) * 2.2), 64)
                self.assertIsNone(r["error"], f"full-context prompt {i}")
        log.seek(0); self.assertNotIn("out of memory", log.read())


if __name__ == "__main__":
    unittest.main()
