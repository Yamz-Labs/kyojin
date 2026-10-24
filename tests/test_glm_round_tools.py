"""CPU check of round_probe + round_replay against a fake generator and a fake SSE server."""
import json, os, pathlib, sys, threading, time, types, http.server, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools" / "glm"))
import round_probe, round_replay

round_probe.torch.cuda.synchronize = lambda: None
class Seq: sequence_ids = list(range(100)); 
class Job:
    sequences = [Seq()]; new_tokens = 0; draft_stats = []
    def is_prefill_done(self): return True
job = Job()
class G:
    active_jobs = [job]
    def iterate(self):
        time.sleep(0.002); job.new_tokens += 3; job.draft_stats.append((job.new_tokens, 2, 2)); return []
g = G(); ls = {}
import tempfile; log = tempfile.mktemp(suffix=".jsonl"); open(log, "w").close()
round_probe.install(g, log, ls)
for _ in range(5): g.iterate()
recs, off = round_replay.read_rounds(log, 0)
assert "req" in recs[0] and len(recs) == 6 and recs[1]["produced"] == 3 and recs[1]["acc"] == 2 and recs[1]["window"] == 2, recs
print("probe ok", recs[1])

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers["content-length"]); self.rfile.read(n)
        self.send_response(200); self.send_header("content-type", "text/event-stream"); self.end_headers()
        for i in range(8):
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": f"w{i} "}}]}) + "\n\n").encode()); self.wfile.flush(); time.sleep(0.01)
        self.wfile.write(("data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 8, "prompt_tokens": 10}, "timings": {"predicted_per_second": 33.0, "predicted_ms": 240, "prompt_per_second": 500, "prompt_ms": 20}}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
a = argparse.Namespace(base=f"http://127.0.0.1:{srv.server_port}", model="m", roundlog=log, max_tokens=8, think="{}", reqlog_dir="", salt="cold")
for _ in range(4): g.iterate()
r, off = round_replay.run_point(a, "x" * 1000, 500, "t", off)
print({k: v for k, v in r.items() if k not in ("rounds_detail", "slowest5", "text")})
assert r["rounds"] == 4 and r["decode_srv"] == 33.0 and r["acc_per_round"] == 2
