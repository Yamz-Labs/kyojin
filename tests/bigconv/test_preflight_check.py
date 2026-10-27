"""Pre-flight checks, release-day arch check and the HTTP range retry."""
import gzip, http.server, json, os, threading
import pytest

from bigconv import adapters  # noqa
from bigconv import preflight as P
from bigconv import archcheck as A
from bigconv.defaults import DEFAULTS

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "bigconv")


def test_dirty_sysctl_raise_is_refused():
    assert P.check_dirty(lambda p: 20 if p.endswith("dirty_ratio") else 10)[1] == "ok"
    assert P.check_dirty(lambda p: 90 if p.endswith("dirty_ratio") else 80)[1] == "FAIL"
    assert P.check_dirty(lambda p: 15 if p.endswith("dirty_ratio") else 5)[1] == "ok"       # lower is fine


def test_spool_budget_qlc_and_disk():
    b = P.spool_budget(free_gb=400, spool_gb=30, floor_gb=60, capture_min_per_layer=1.0)
    assert b["max_pending"] == 3 and b["qlc_warning"] and b["write_min_per_layer"] == pytest.approx(30 * 1000 / 300 / 60, abs=0.1)
    assert P.spool_budget(400, 30, 60, capture_min_per_layer=10)["qlc_warning"] is None
    assert P.check_spool(100, 30)[1] == "FAIL"             # room for 1 pending spool: capture would serialise
    assert P.check_spool(400, 30, capture_min_per_layer=60)[1] == "ok"


def test_calib_default_is_500k():
    assert DEFAULTS["rows"] * DEFAULTS["row_len"] == 512000
    assert P.check_calib(250, 2048)[1] == "ok" and P.check_calib(1000, 2048)[1] == "warn"


def test_fstrim_hint(tmp_path):
    st = tmp_path / "stamp"
    st.write_text("")
    assert P.check_fstrim(str(st))[1] == "ok"
    assert P.check_fstrim(str(st), now=os.path.getmtime(st) + 30 * 86400)[1] == "warn"
    assert P.check_fstrim(str(tmp_path / "none"))[1] == "warn"


def test_gpurun_exemption_check(tmp_path):
    g = tmp_path / "gpurun"
    g.write_text('case "$who" in workerq*) idle=0;; esac\n')
    assert P.check_gpurun_exempt(str(g), "bigconv-q0")[1] == "FAIL"
    g.write_text('case "$who" in workerq*|bigconv-q*) idle=0;; esac\n')
    assert P.check_gpurun_exempt(str(g), "bigconv-q0")[1] == "ok"


@pytest.mark.parametrize("arch", ["glm5", "minimax_m3"])
def test_archcheck_clean_on_the_bundled_release(arch):
    cfg = json.load(open(f"{TOOLS}/configs/{arch}.config.json"))
    idx = json.load(gzip.open(f"{TOOLS}/fixtures/{arch}.index.json.gz", "rt"))
    r = A.check(arch, cfg, idx, cfg, idx)
    assert r["problems"] == [] and r["new_patterns"] == []


def test_archcheck_flags_a_new_tensor_family_and_config_drift():
    arch = "glm5"
    cfg = json.load(open(f"{TOOLS}/configs/{arch}.config.json"))
    idx = json.load(gzip.open(f"{TOOLS}/fixtures/{arch}.index.json.gz", "rt"))
    new_idx = {"metadata": idx["metadata"], "weight_map": dict(idx["weight_map"])}
    new_idx["weight_map"]["model.language_model.layers.5.self_attn.new_gate_proj.weight"] = "model-00001.safetensors"
    new_cfg = json.loads(json.dumps(cfg))
    new_cfg["text_config"]["num_hidden_layers"] += 3
    new_cfg["text_config"]["brand_new_key"] = 1
    r = A.check(arch, new_cfg, new_idx, cfg, idx)
    assert any("new_gate_proj" in p for p in r["new_patterns"])
    assert r["problems"] and "text_config.brand_new_key" in r["config"]["added"]
    assert any(k.endswith("num_hidden_layers") for k, *_ in r["config"]["changed"])


def test_archcheck_flags_missing_experts():
    cfg = json.load(open(f"{TOOLS}/configs/minimax_m3.config.json"))
    idx = json.load(gzip.open(f"{TOOLS}/fixtures/minimax_m3.index.json.gz", "rt"))
    wm = {k: v for k, v in idx["weight_map"].items() if ".experts.17.w2." not in k}
    r = A.check("minimax_m3", cfg, {"metadata": idx["metadata"], "weight_map": wm}, cfg, idx)
    assert any("routed-linear count" in p for p in r["problems"])


# ---- HTTP range fetch survives http.client.IncompleteRead
class Flaky(http.server.BaseHTTPRequestHandler):
    hits = 0
    body = bytes(range(256)) * 4

    def do_GET(self):
        Flaky.hits += 1
        a, b = [int(x) for x in self.headers["Range"].split("=")[1].split("-")]
        data = self.body[a:b + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.body)}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data[:len(data) // 2] if Flaky.hits <= 2 else data)      # first two answers end early
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *a):
        pass


@pytest.mark.parametrize("which", ["ds41", "tensor_map"])
def test_range_get_retries_incomplete_read(monkeypatch, which):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    srv = http.server.HTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Flaky.hits = 0
    url = f"http://127.0.0.1:{srv.server_port}/f"
    try:
        if which == "ds41":
            import ds41.common as C
            got = C.range_get(url, 10, 209)
        else:
            from bigconv.tensor_map import _get
            got = _get(url, 10, 209)[0]
    finally:
        srv.shutdown()
    assert got == Flaky.body[10:210] and Flaky.hits == 3


def test_scratch_must_be_tmpfs_and_big_enough(tmp_path):
    m = tmp_path / "mounts"
    m.write_text(f"/dev/nvme0n1p2 / ext4 rw 0 0\ntmpfs {tmp_path}/ram tmpfs rw 0 0\n")
    (tmp_path / "ram").mkdir()
    (tmp_path / "ssd").mkdir()
    assert P.check_scratch(str(tmp_path / "ssd"), 10, str(m))[1] == "FAIL"
    assert P.check_scratch(str(tmp_path / "ram"), 10, str(m))[1] in ("ok", "FAIL")      # ok when the test fs has the room
    assert P.check_scratch(str(tmp_path / "ram"), 1e9, str(m))[1] == "FAIL"
