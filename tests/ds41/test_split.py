"""Offline CPU tests of the coordinator/worker split (tools/ds41/split.py) with the stub quantizer.

No network and no GPU: a tiny DS4.1-shaped checkpoint is served by a local HTTP server that implements
Range requests (DS41_BASE_URL), and every process runs with the GPUs hidden.
"""

import http.server
import json
import os
import signal
import struct
import subprocess
import sys
import threading
import time

import pytest
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "tools", "ds41")
sys.path.insert(0, TOOLS)

import common as C  # noqa: E402
import plan as P  # noqa: E402
import split as SP  # noqa: E402
import stream_convert as S  # noqa: E402
import tensor_map as TM  # noqa: E402

SC = os.path.join(TOOLS, "stream_convert.py")
REPO, REV = "ds/mini", "r1"
HID, INTER, N_EXP = 64, 32, 12
PY = sys.executable


# ---------------------------------------------------------------- tiny checkpoint over HTTP

def _fp4(g, out_f, in_f):
    return torch.randint(0, 256, (out_f, in_f // 2), generator=g, dtype=torch.uint8).view(torch.int8)


def _e8m0(g, *shape):
    return torch.randint(120, 130, shape, generator=g, dtype=torch.uint8)


def _fp8(g, *shape):
    return (torch.randn(*shape, generator=g) * 2).to(torch.float8_e4m3fn).view(torch.uint8)


def build_checkpoint():
    g = torch.Generator().manual_seed(7)
    T = []   # (name, dtype, shape, bytes)

    def add(name, dtype, t):
        T.append((name, dtype, list(t.shape), t.contiguous().view(torch.uint8).numpy().tobytes()))

    L = "layers.1"
    add(f"{L}.attn_norm.weight", "BF16", torch.randn(HID, generator=g).to(torch.bfloat16))
    add(f"{L}.ffn.gate.weight", "BF16", torch.randn(N_EXP, HID, generator=g).to(torch.bfloat16))
    for nm, o, i in (("attn.wq_a", 32, HID), ("ffn.shared_experts.w1", INTER, HID),
                     ("ffn.shared_experts.w2", HID, INTER)):
        add(f"{L}.{nm}.weight", "F8_E4M3", _fp8(g, o, i))
        add(f"{L}.{nm}.scale", "F8_E8M0", _e8m0(g, o // 32 if o >= 32 else 1, i // 32 if i >= 32 else 1))
    for e in range(N_EXP):
        for w, o, i in (("w1", INTER, HID), ("w2", HID, INTER), ("w3", INTER, HID)):
            add(f"{L}.ffn.experts.{e}.{w}.weight", "I8", _fp4(g, o, i))
            add(f"{L}.ffn.experts.{e}.{w}.scale", "F8_E8M0", _e8m0(g, o, i // 32))
    items, pos = [], 0
    for name, dt, shape, raw in T:
        items.append((name, dt, shape, len(raw), iter([raw])))
    import tempfile
    d = tempfile.mkdtemp()
    path = os.path.join(d, "model-00001-of-00001.safetensors")
    S.write_streamed(path, items)
    shard = open(path, "rb").read()
    idx = {"metadata": {"total_size": sum(len(r) for *_, r in T)}, "weight_map": {n: "model-00001-of-00001.safetensors" for n, *_ in T}}
    return {"model-00001-of-00001.safetensors": shard,
            "config.json": json.dumps({"model_type": "mini", "quantization_config": {"quant_method": "fp8"}}).encode(),
            "model.safetensors.index.json": json.dumps(idx).encode()}


class _Handler(http.server.BaseHTTPRequestHandler):
    files = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        name = self.path.rsplit("/", 1)[-1]
        data = self.files.get(name)
        if data is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    _Handler.files = build_checkpoint()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["DS41_BASE_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"
    yield
    srv.shutdown()
    del os.environ["DS41_BASE_URL"]


@pytest.fixture(scope="module")
def mapfile(server, tmp_path_factory):
    tm = TM.build(str(tmp_path_factory.mktemp("m") / "map.json"), REPO, REV, workers=1, log=lambda *_: None)
    p = str(tmp_path_factory.mktemp("m2") / "map.json")
    json.dump(tm, open(p, "w"))
    return p


def common_args(mapfile, out):
    return ["--map", mapfile, "--out", out, "--layers", "1", "--target", "2.3", "--quantizer", "stub"]


def worker_cmd(name, delay=0.0):
    return f"{name}=env DS41_WORKER_NAME={name} {PY} {SC} --role worker"


def run_coord(mapfile, out, workers=("A", "B"), extra=(), delay=0.0, wait=True):
    cmd = [PY, SC, "--role", "coordinator", *common_args(mapfile, out), "--chunk", "2", "--stub-delay", str(delay),
           "--retry-s", "0.2", *extra]
    for w in workers:
        cmd += ["--worker", worker_cmd(w)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait:
        return p
    out_txt = p.communicate(timeout=300)[0]
    assert p.returncode == 0, out_txt
    return out_txt


@pytest.fixture(scope="module")
def baseline(mapfile, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("single"))
    S.main(common_args(mapfile, out))
    return out


@pytest.fixture(scope="module")
def baseline_hess(mapfile, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("hess1"))
    run_coord(mapfile, out, workers=("A",), extra=["--hessians", "synthetic"])
    return out


def final_files(out):
    names = ["layers.1.safetensors", "layers.1.done.json", "model.safetensors.index.json",
             "quantization_config.json", "config.json"]
    return {n: open(os.path.join(out, n), "rb").read() for n in names}


def expert_workers(out):
    d = os.path.join(out, "experts", "layers.1")
    if not os.path.isdir(d):
        return {}
    r = {}
    for fn in os.listdir(d):
        if fn.endswith(".json") and not fn.endswith(".tmp"):
            try:
                r[fn] = json.load(open(os.path.join(d, fn)))["worker"]
            except (ValueError, KeyError):
                pass
    return r


# ---------------------------------------------------------------- unit tests

def test_hessian_blob_roundtrip():
    H, n = SP.synthetic_hessian(48, 3)
    blob = SP.pack_hessian(H, n, 123, (1, 2))
    assert len(blob) < 48 * 48 * 4          # triangle only
    u = SP.unpack_hessian(blob)
    assert torch.equal(u["H"], H.triu() + H.triu(1).T) and u["count"] == n and u["inf_nan"] == [1, 2]
    bad = bytearray(blob)
    bad[40] ^= 1
    with pytest.raises(ValueError):
        SP.unpack_hessian(bytes(bad))


def test_frame_roundtrip():
    import io
    b = io.BytesIO()
    SP.send_frame(b, {"op": "x", "n": 1}, [b"abc", b"def"])
    SP.send_frame(b, {"op": "y"})
    b.seek(0)
    assert SP.recv_frame(b) == ({"op": "x", "n": 1}, b"abcdef")
    assert SP.recv_frame(b) == ({"op": "y"}, b"")
    assert SP.recv_frame(b) is None


def test_queue_requeue_and_reissue():
    q = SP.LayerQueue("u", [1, 2, 3, 4], 1, 4)
    q.release([1, 2, 3, 4])
    j1 = q.take("A", timeout=0.01)
    j2 = q.take("B", timeout=0.01)
    j2b = q.take("B", timeout=0.01)
    assert [j1.experts, j2.experts, j2b.experts] == [[1, 2], [3], [4]]    # guided sizing shrinks at the tail
    assert q.take("A", timeout=0.01) is None           # nothing pending, nothing old enough
    q.requeue(j1, "A")                                  # A died
    j3 = q.take("B", timeout=0.01)
    j3b = q.take("B", timeout=0.01)
    assert j3.experts + j3b.experts == j1.experts      # back at the front, re-chunked
    j2.t0 -= 1000                                       # B's first job straggles
    j4 = q.take("C", reissue_after=10, timeout=0.01)
    assert j4 is j2 and j4.holders == {"B", "C"}
    q.finished(j4, "C")
    q.finished(j3, "B")
    q.finished(j3b, "B")
    q.finished(j2, "B")
    q.finished(j2b, "B")
    q.wait_all(lambda: True)


def test_expert_marker_validation(tmp_path):
    out = str(tmp_path)
    assert SP.store_expert(out, "u", 3, b"abcdef", {"e": 3})
    assert SP.expert_done(out, "u", 3)
    assert not SP.store_expert(out, "u", 3, b"zzz", {"e": 3})    # duplicate loses
    f, _ = SP.expert_paths(out, "u", 3)
    open(f, "wb").write(b"abc")                                    # truncated file
    assert not SP.expert_done(out, "u", 3)


# ---------------------------------------------------------------- end to end

def test_split_equals_single(mapfile, baseline, tmp_path):
    out = str(tmp_path)
    run_coord(mapfile, out, delay=0.01, extra=["--keep-experts"])
    a, b = final_files(baseline), final_files(out)
    for n in a:
        assert a[n] == b[n], n
    who = set(expert_workers(out).values())
    assert who == {"A", "B"}, who     # both workers took part


def test_calibrated_two_workers_equal_one(mapfile, baseline, baseline_hess, tmp_path):
    out = str(tmp_path)
    run_coord(mapfile, out, extra=["--hessians", "synthetic", "--synthetic-slices", "3", "--synthetic-gap", "0.2"],
              delay=0.02)
    a, b = final_files(baseline_hess), final_files(out)
    for n in a:
        assert a[n] == b[n], n
    assert a["layers.1.safetensors"] != final_files(baseline)["layers.1.safetensors"]   # Hessians reached the quantizer
    assert not os.path.exists(os.path.join(out, "spool", "layers.1"))                     # cleaned up


def _worker_pid(coord_pid, name):
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
            if ppid == coord_pid and f"DS41_WORKER_NAME={name}".encode() in open(f"/proc/{pid}/environ", "rb").read():
                return int(pid)
        except (OSError, IndexError, ValueError):
            continue
    return None


def _wait(cond, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_kill_worker_mid_layer(mapfile, baseline, tmp_path):
    out = str(tmp_path)
    p = run_coord(mapfile, out, delay=0.15, extra=["--worker-restarts", "0", "--keep-experts"], wait=False)
    try:
        assert _wait(lambda: len(expert_workers(out)) >= 3)
        pid = _worker_pid(p.pid, "A")
        assert pid, "worker A not found"
        os.kill(pid, signal.SIGKILL)
        txt = p.communicate(timeout=300)[0]
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, txt
    assert "gave up" in txt
    a, b = final_files(baseline), final_files(out)
    for n in a:
        assert a[n] == b[n], n
    who = expert_workers(out)
    assert len(who) == N_EXP and "B" in who.values()


def test_kill_coordinator_resume(mapfile, baseline_hess, tmp_path):
    out = str(tmp_path)
    p = run_coord(mapfile, out, delay=0.15, extra=["--hessians", "synthetic", "--keep-experts"], wait=False)
    try:
        assert _wait(lambda: len(expert_workers(out)) >= 4)
        p.kill()
        p.wait()
    finally:
        if p.poll() is None:
            p.kill()
    first = {}
    d = os.path.join(out, "experts", "layers.1")
    for fn in os.listdir(d):
        if fn.endswith(".json"):
            first[fn] = open(os.path.join(d, fn), "rb").read()
    assert 4 <= len(first) < N_EXP
    assert not os.path.exists(os.path.join(out, "layers.1.done.json"))
    time.sleep(0.5)   # workers see EOF and exit
    # resume: same command, one worker only this time
    txt = run_coord(mapfile, out, workers=("C",), extra=["--hessians", "synthetic", "--keep-experts"], delay=0.0)
    assert f"{len(first)} already done" in txt or "already done" in txt
    a, b = final_files(baseline_hess), final_files(out)
    for n in a:
        assert a[n] == b[n], n
    for fn, raw in first.items():      # finished experts were not recomputed
        assert open(os.path.join(d, fn), "rb").read() == raw
    assert "C" in expert_workers(out).values()
