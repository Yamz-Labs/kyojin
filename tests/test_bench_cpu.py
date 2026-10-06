"""CPU test for tools/bench.py against a small mock OpenAI server: the prefill and decode arithmetic and
the Markdown block. bench.py reads time.perf_counter() once per measurement point, so a fake clock that
moves STEP seconds per read makes every rate exact: prefill = prompt tokens / STEP, decode = 1 / STEP."""
import contextlib
import importlib.util
import io
import itertools
import json
import re
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("bench", Path(__file__).parent.parent / "tools" / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

STEP = 0.05
MODEL = "mock-model"


class MockOpenAI(BaseHTTPRequestHandler):
    """/v1/models, and /v1/chat/completions: prompt_tokens = words in the prompt; a stream sends one
    content chunk per completion token, then a usage chunk. "mock_completion_tokens" in the request
    overrides the completion_tokens it reports."""
    protocol_version = "HTTP/1.1"
    requests = []                                   # every request body, in order (reset per test by the server fixture)

    def log_message(self, *args):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockOpenAI.requests.append(req)
        prompt_tokens = len(req["messages"][-1]["content"].split())
        n = req.get("max_tokens", 16)
        if not req.get("stream"):
            return self._json({"choices": [{"index": 0, "message": {"role": "assistant", "content": "x"},
                                            "finish_reason": "length"}],
                               "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        events = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}]
        events += [{"choices": [{"index": 0, "delta": {"content": f"t{i} "}}]} for i in range(n)]
        events += [{"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": req.get("mock_completion_tokens", n),
                              "total_tokens": prompt_tokens + n}}]
        for e in events:
            self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True


@pytest.fixture
def server(monkeypatch):
    # bench.py calls urllib.request.urlopen(), which reuses a global opener built from the proxy settings
    # of its first call. Replace it with one that ignores proxies, so a proxy in the environment cannot
    # intercept requests to the mock server, whatever ran earlier in the same session.
    monkeypatch.setattr(bench.urllib.request, "_opener",
                        bench.urllib.request.build_opener(bench.urllib.request.ProxyHandler({})))
    monkeypatch.setattr(MockOpenAI, "requests", [])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAI)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def fake_clock(monkeypatch):
    ticks = itertools.count(1)                     # never 0.0: bench.py keeps the first chunk time with `first or now`
    clock = types.SimpleNamespace(perf_counter=lambda: next(ticks) * STEP, time=time.time, strftime=time.strftime)
    monkeypatch.setattr(bench, "time", clock)


def test_prefill_is_prompt_tokens_over_wall_time(server, fake_clock):
    n, rate = bench.prefill(server, MODEL, "one two three four five six seven eight")
    assert n == 8
    assert rate == pytest.approx(8 / STEP)


def test_decode_skips_the_first_token(server, fake_clock):
    # 128 content chunks read the clock 128 times: 127 steps between the first and the last.
    assert bench.decode(server, MODEL, "hello", 128) == pytest.approx(127 / (127 * STEP))


def test_decode_needs_two_tokens(server, fake_clock):
    assert bench.decode(server, MODEL, "hello", 1) is None


def test_decode_trusts_usage_over_chunk_count(server, fake_clock, monkeypatch):
    # Two content chunks but usage reports one token: one token gives no rate, whatever the chunks say.
    post = bench.post
    monkeypatch.setattr(bench, "post", lambda base, body, timeout=3600: post(base, body | {"mock_completion_tokens": 1}, timeout))
    assert bench.decode(server, MODEL, "hello", 2) is None


def test_markdown_block(server, fake_clock, monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(" ".join(f"w{i}" for i in range(9000)))
    monkeypatch.setattr(bench, "gpu_name", lambda: "gfx-test")
    monkeypatch.setattr(bench, "rocm_version", lambda: "rocm-test")
    monkeypatch.setattr(sys, "argv", ["bench.py", "--base", server, "--corpus", str(corpus), "--max-tokens", "16"])
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        bench.main()
    md = out.getvalue()

    assert md.startswith("### Kyojin benchmark\n")
    assert f"- Model: `{MODEL}`" in md
    assert "- GPU target: gfx-test" in md and "- ROCm: rocm-test" in md
    assert f"- Server: {server}" in md

    # Every prompt has the same length, so the median is that length over one clock step.
    m = re.search(r"\| Prefill, ~(\d+) tokens \(median of 3\) \| ([\d.]+) tok/s \|", md)
    assert m, md
    assert float(m.group(2)) == pytest.approx(int(m.group(1)) / STEP, abs=0.05)
    # The word count comes from the warm-up's tokens-per-word ratio, so the prompt lands near the target, not on it.
    assert int(m.group(1)) == pytest.approx(bench.PREFILL_TARGET_TOKENS, rel=0.02)

    for cat in bench.DECODE_PROMPTS:
        assert f"| Decode, {cat}, 16 tokens, temperature 0 (median of 3) | {1 / STEP:.1f} tok/s |" in md


def test_requests_are_greedy(server, fake_clock):
    bench.prefill(server, MODEL, "one two three")
    bench.decode(server, MODEL, "hello", 4)
    pre, dec = MockOpenAI.requests
    assert pre["temperature"] == 0 and pre["max_tokens"] == 1
    assert dec["temperature"] == 0 and dec["stream"] is True


def test_table_reports_medians_not_means(server, monkeypatch, tmp_path):
    # Three unequal runs per row, so the median (200, 20) and the mean (300, 30) differ.
    def prefill(base, model, text):
        salt = text[1:text.index("]")]
        return (300, 1.0) if salt.endswith("warm") else (1000, [100.0, 200.0, 600.0][int(salt.rsplit("-", 1)[1])])

    def decode(base, model, prompt, max_tokens):
        return next([10.0, 20.0, 60.0][p.index(prompt)] for p in bench.DECODE_PROMPTS.values() if prompt in p)

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(" ".join(f"w{i}" for i in range(9000)))
    monkeypatch.setattr(bench, "prefill", prefill)
    monkeypatch.setattr(bench, "decode", decode)
    monkeypatch.setattr(bench, "gpu_name", lambda: "gfx-test")
    monkeypatch.setattr(bench, "rocm_version", lambda: "rocm-test")
    monkeypatch.setattr(sys, "argv", ["bench.py", "--base", server, "--model", MODEL, "--corpus", str(corpus)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        bench.main()
    md = out.getvalue()
    assert "| Prefill, ~1000 tokens (median of 3) | 200.0 tok/s |" in md, md
    for cat in bench.DECODE_PROMPTS:
        assert f"(median of 3) | 20.0 tok/s |" in [line[line.index("(median"):] for line in md.splitlines() if line.startswith(f"| Decode, {cat},")][0]


def test_json_block_matches_the_table(server, monkeypatch, tmp_path):
    # Unequal runs, so the JSON must carry the median (not the mean) and every run; "code" never gives a rate.
    def prefill(base, model, text):
        salt = text[1:text.index("]")]
        return (300, 1.0) if salt.endswith("warm") else (1000, [100.0, 200.0, 600.0][int(salt.rsplit("-", 1)[1])])

    def decode(base, model, prompt, max_tokens):
        if prompt in bench.DECODE_PROMPTS["code"]:
            return None
        return next([10.0, 20.0, 60.0][p.index(prompt)] for p in bench.DECODE_PROMPTS.values() if prompt in p)

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(" ".join(f"w{i}" for i in range(9000)))
    monkeypatch.setattr(bench, "prefill", prefill)
    monkeypatch.setattr(bench, "decode", decode)
    monkeypatch.setattr(bench, "gpu_name", lambda: "gfx-test")
    monkeypatch.setattr(bench, "rocm_version", lambda: "rocm-test")
    argv = ["bench.py", "--base", server, "--model", MODEL, "--corpus", str(corpus)]
    outs = []
    for extra in ([], ["--json"]):
        monkeypatch.setattr(sys, "argv", argv + extra)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            bench.main()
        outs.append(out.getvalue())
    plain, with_json = outs
    assert "```json" not in plain
    assert with_json.startswith(plain.rstrip("\n"))          # the Markdown block itself is unchanged
    d = json.loads(re.search(r"```json\n(.*)\n```", with_json, re.S).group(1))
    assert (d["model"], d["gpu_target"], d["rocm"], d["server"]) == (MODEL, "gfx-test", "rocm-test", server)
    assert d["prefill"] == {"prompt_tokens": 1000, "max_tokens": 1, "temperature": 0, "tok_s": 200.0, "runs": [100.0, 200.0, 600.0]}
    assert d["decode"]["prose"] == d["decode"]["chat"] == {"max_tokens": 128, "temperature": 0, "tok_s": 20.0, "runs": [10.0, 20.0, 60.0]}
    assert d["decode"]["code"] == {"max_tokens": 128, "temperature": 0, "tok_s": None, "runs": []}
    # the table shows the same medians
    assert "| Prefill, ~1000 tokens (median of 3) | 200.0 tok/s |" in plain
    for cat in ("prose", "chat"):
        assert f"| Decode, {cat}, 128 tokens, temperature 0 (median of 3) | 20.0 tok/s |" in plain
    assert "| Decode, code, 128 tokens, temperature 0 (median of 0) | n/a tok/s |" in plain
