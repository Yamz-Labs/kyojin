"""Gate plumbing on the stub engine: score route -> gate.py -> verdict. Run: python3 -m pytest tools/ds41qual/test_gate.py -q"""
import asyncio, json, os, subprocess, sys, threading
import numpy as np
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ds41"))
sys.path.insert(0, HERE)
import serve                                   # noqa: E402
from ds41_engine import StubEngine             # noqa: E402
import gate                                    # noqa: E402


def test_verdict_bars():
    assert gate.verdict(94.0, 0.02) == "STRONG"
    assert gate.verdict(90.0, 0.10) == "PASS"
    assert gate.verdict(89.0, 0.10) == "FAIL"
    assert gate.verdict(94.0, 0.20) == "FAIL"


def test_row_metrics_identical_distribution_is_zero():
    lp = np.log(np.array([[0.5, 0.3, 0.1]]))
    ref = {"top_lp": lp, "argmax": np.array([7])}
    pos = [{"top1": 7, "cand_logps": lp[0].tolist()}]
    k, t = gate.row_metrics(ref, pos, 0)
    assert abs(k[0]) < 1e-9 and t[0] == 1
    pos = [{"top1": 8, "cand_logps": np.log([0.3, 0.3, 0.3]).tolist()}]
    k, t = gate.row_metrics(ref, pos, 0)
    assert k[0] > 0 and t[0] == 0


def test_end_to_end_with_stub_server(tmp_path):
    rng = np.random.default_rng(0)
    rows = [rng.integers(1, 1000, 300).astype(np.int64) for _ in range(2)]
    d = tmp_path / "data"; d.mkdir()
    np.savez(d / "gate_ids.npz", **{f"row{i:03d}": r for i, r in enumerate(rows)})
    import hashlib
    sha = hashlib.sha256(b"".join(r.astype(np.int32).tobytes() for r in rows)).hexdigest()
    json.dump({"names": ["a", "b"], "sha256": sha}, open(d / "gate_meta.json", "w"))
    ref = tmp_path / "ref"; ref.mkdir()
    for i, r in enumerate(rows):
        P = len(r) - 1
        idx = np.stack([r[1:], r[1:] + 1], -1).astype(np.int32)          # reference: argmax = next token
        lp = np.log(np.tile([0.9, 0.1], (P, 1))).astype(np.float32)
        np.savez(ref / f"ref_row{i:03d}.npz", top_idx=idx, top_lp=lp, argmax=r[1:].astype(np.int32),
                 tgt_lp=lp[:, 0], lse=np.zeros(P, np.float32))
    json.dump({"sha256": sha, "start": 64, "topk": 2}, open(ref / "manifest.json", "w"))

    loop = asyncio.new_event_loop(); ready = threading.Event(); box = {}
    def run():
        asyncio.set_event_loop(loop)
        app = serve.create_app(StubEngine("x"), "m")
        runner = web.AppRunner(app); loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0); loop.run_until_complete(site.start())
        box["port"] = site._server.sockets[0].getsockname()[1]; ready.set(); loop.run_forever()
    threading.Thread(target=run, daemon=True).start(); ready.wait(10)
    out = tmp_path / "res.json"
    p = subprocess.run([sys.executable, os.path.join(HERE, "gate.py"), "--base-url", f"http://127.0.0.1:{box['port']}",
                        "--ref", str(ref), "--ids", str(d / "gate_ids.npz"), "--out", str(out)], capture_output=True, text=True)
    loop.call_soon_threadsafe(loop.stop)
    res = json.load(open(out))
    assert res["positions"] == 2 * (299 - 64) and 85 < res["top1_pct"] < 95, p.stdout[-400:]
    assert res["verdict"] == "FAIL" or res["verdict"] in ("PASS", "STRONG")      # stub KLD is arbitrary; plumbing only
