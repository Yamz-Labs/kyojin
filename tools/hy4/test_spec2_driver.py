"""Offline exec-check of step/spec2_arms.py main() (no GPU, no server): python3 tools/hy4/test_spec2_driver.py <head_tau.passes>
Stubs the server (Popen, health, chat) with a fake engine that writes the real server-log line format using tools/hy4/depth_gate.DepthGate (modes static / learn / carry),
so the whole job path runs: boot loop, per-request log parsing, FIRES gate_det (per arm + N vs N2), tables, verdict, exit code. Expected: arms N, N2, O gate_det 0 and rc 0;
arm C (carry) gate_det > 0 (negative control) -> run with SP2_ORDER=ACN to see the C failure path."""
import hashlib, io, json, os, random, runpy, sys, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from depth_gate import DepthGate
passes = sys.argv[1]; order = os.environ.get("SP2_ORDER", "AONN")
tmp = tempfile.mkdtemp(prefix="sp2chk_")
os.environ.update(OD_NAME="chk", OD_PORT="8999", N1119=tmp, B2_H=tmp, B2_DIR=tmp, B2_PASSES=passes, SP2_LEVERS="0", SP2_ORDER=order, OD_TMAX="3300", GGUF="x.gguf")
import subprocess, urllib.request, time
state = {}
class FakeProc:
    pid = 0
    def __init__(self, cmd, env=None, stdout=None, **kw):
        self.env, self.log = env, stdout
        mode = env.get("HY4_SPEC2_MODE") if env.get("HY4_SPEC2_GATE") == "1" else None
        self.g = DepthGate({1: 61.8, 2: 76.7, 3: 90.9, 4: 104.0, 5: 113.3, 6: 128.5}, 3.0, 0.3, 2, 1, mode=mode, theta=float(env.get("HY4_SPEC2_THETA", "0.6"))) if mode else None
        self.n, self.fires = 0, {"rounds": 0, "depth2": 0, "gate_skip": 0, "gate_run": 0}
        state["srv"] = self
    def poll(self): return None
    def request(self, body):
        self.n += 1
        label = json.dumps(body["messages"])[:40]
        seed = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
        rng = random.Random(seed)
        rounds, tau, h, ext = 110, 2.3, [30, 30, 30, 20, 0, 0, 0, 0, 0], 0
        if self.g:
            self.g.reset()
            for _ in range(rounds):
                c = [rng.random() ** 0.5, rng.random() ** 0.5]; go, b = self.g.decide(c); self.g.update(b, go, min(rng.randrange(4), 3 if go else 2))
                self.fires["rounds"] += 1; self.fires["depth2"] += 1; self.fires["gate_run" if go else "gate_skip"] += 1
            st = self.g.stats(); gate = f" | gate {st} | fires {self.fires}"
            md5 = hashlib.md5((label + st["digest"]).encode()).hexdigest()      # fake text depends on the gate decisions (as the real text does through the row count T)
        else:
            gate = ""; md5 = hashlib.md5(label.encode()).hexdigest()
        self.log.write(f"req {self.n}: prompt 3310 tok prefill 4.1 s (800 tok/s) | decode 255 tok 11.0 s = 23.0 tok/s spec=True rounds {rounds} tau {tau} hist {h} | length{gate}\n"); self.log.flush()
        return md5
subprocess.Popen = FakeProc
class R:
    def __init__(s, b): s.b = b
    def read(s): return s.b
urllib.request.urlopen = lambda *a, **k: R(b"{}")
time.sleep = lambda *_: None
sys.argv = ["spec2_arms.py"]
src = open(os.path.join(HERE, "step", "spec2_arms.py")).read().replace('if __name__ == "__main__":\n    main()', "")
g = {"__name__": "spec2_arms"}; exec(compile(src, "spec2_arms.py", "exec"), g)
def chat(body):
    md5 = state["srv"].request(body)
    return {"tps": 23.0, "tau": 2.3, "n": 255, "prompt_n": 3310, "md5": md5[:12]}
g["chat"] = chat; g["stop"] = lambda s: None
try:
    g["main"]()
    print("RC 0")
except SystemExit as e:
    print("RC", e.code)
