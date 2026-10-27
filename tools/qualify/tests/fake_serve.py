"""Stand-in for tools/qwen/serve.py in the qualify tests: same CLI subset, /health 503 then 200, SSE chat completions with usage,
/apply-template. -c above LIMIT exits like a CUDA OOM at start-up; FAIL_ROUND2=1 makes the 2nd concurrent round fail."""
import argparse, json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int); ap.add_argument("-c", type=int); ap.add_argument("--sessions", type=int, default=1)
ap.add_argument("--model"); ap.add_argument("--model-id"); ap.add_argument("--k-bits"); ap.add_argument("--v-bits"); ap.add_argument("--draft-policy")
ap.add_argument("--no-vision", action="store_true"); ap.add_argument("--chat-template")
a = ap.parse_args()
LIMIT = int(os.environ.get("LIMIT", "50000"))
if a.c > LIMIT:
    print("torch.OutOfMemoryError: CUDA out of memory", flush=True); sys.exit(1)
ready = time.time() + 0.5


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def do_GET(self):
        ok = time.time() >= ready
        self.send_response(200 if ok else 503); self.end_headers(); self.wfile.write(b"{}")

    def do_POST(self):
        n = int(self.headers["Content-Length"]); b = json.loads(self.rfile.read(n))
        if self.path == "/apply-template":
            m = b["messages"]; p = "TOOLMARK " + m[0]["content"] + " " + m[1]["content"]
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"prompt": p}).encode()); return
        if self.path == "/v1/completions":
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({"usage": {"prompt_tokens": len(b["prompt"]), "completion_tokens": b["max_tokens"]}}).encode()); return
        text = b["messages"][-1]["content"]
        pt = max(1, len(text) // 4); mt = b["max_tokens"]
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        if pt + mt > a.c:
            self.wfile.write(b'data: {"error": "context overflow"}\n\n'); return
        time.sleep(pt / 2e5)
        for i in range(min(mt, 20)):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": f"w{(hash(text[:50]) + i) % 7 if False else i} "}}]}).encode() + b"\n\n"); self.wfile.flush(); time.sleep(0.002)
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": min(mt, 20)}}).encode() + b"\n\ndata: [DONE]\n\n")


ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
