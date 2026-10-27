import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("bench", Path(__file__).parent.parent / "tools" / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _node(root, n, version):
    d = root / str(n)
    d.mkdir()
    (d / "properties").write_text(f"cpu_cores_count 16\nsimd_count {0 if version == 0 else 80}\n"
                                  f"gfx_target_version {version}\nmax_waves_per_simd 16\n")


def test_gfx_name_uses_hex_digits_for_minor_and_stepping():
    assert bench.gfx_name(110501) == "gfx1151"
    assert bench.gfx_name(110000) == "gfx1100"
    assert bench.gfx_name(100300) == "gfx1030"
    assert bench.gfx_name(90010) == "gfx90a"
    assert bench.gfx_name(90402) == "gfx942"


def test_kfd_skips_the_cpu_node(tmp_path):
    _node(tmp_path, 0, 0)
    _node(tmp_path, 1, 110501)
    assert bench.kfd_gpu_target(tmp_path) == "gfx1151"


def test_kfd_takes_the_first_gpu_by_node_number(tmp_path):
    _node(tmp_path, 0, 0)
    _node(tmp_path, 2, 110000)
    _node(tmp_path, 10, 90402)
    assert bench.kfd_gpu_target(tmp_path) == "gfx1100"


def test_kfd_missing_or_cpu_only_gives_empty(tmp_path):
    assert bench.kfd_gpu_target(tmp_path / "absent") == ""
    _node(tmp_path, 0, 0)
    assert bench.kfd_gpu_target(tmp_path) == ""


def test_kfd_ignores_odd_names_and_undecodable_nodes(tmp_path):
    _node(tmp_path, 0, 0)
    (tmp_path / "²").mkdir()                                  # isdigit() but not int()-able
    bad = tmp_path / "1"
    bad.mkdir()
    (bad / "properties").write_bytes(b"\xff\xfe gfx_target_version 110000\n")  # not UTF-8
    _node(tmp_path, 2, 110501)
    assert bench.kfd_gpu_target(tmp_path) == "gfx1151"


def test_gpu_name_prefers_rocminfo(monkeypatch):
    monkeypatch.setattr(bench, "sh", lambda cmd: "gfx1100")
    monkeypatch.setattr(bench, "kfd_gpu_target", lambda nodes=None: "gfx1151")
    assert bench.gpu_name() == "gfx1100"


def test_gpu_name_falls_back_to_kfd_without_rocminfo(monkeypatch):
    monkeypatch.setattr(bench, "sh", lambda cmd: "")
    monkeypatch.setattr(bench, "kfd_gpu_target", lambda nodes=None: "gfx1151")
    assert bench.gpu_name() == "gfx1151"
    monkeypatch.setattr(bench, "kfd_gpu_target", lambda nodes=None: "")
    assert bench.gpu_name() == "unknown"


ROCMINFO = """Agent 1
  Name:                    AMD Ryzen 9 7950X
  Marketing Name:          AMD Ryzen 9 7950X 16-Core Processor
Agent 2
  Name:                    gfx1100
  Uuid:                    GPU-0
  Marketing Name:          AMD Radeon RX 7900 XTX
"""


def _card(root, n, vendor, product=None):
    d = root / f"card{n}" / "device"
    d.mkdir(parents=True)
    (d / "vendor").write_text(vendor + "\n")
    if product:
        (d / "product_name").write_text(product + "\n")


def test_rocminfo_model_skips_cpu_agents():
    assert bench.rocminfo_gpu_model(ROCMINFO) == "AMD Radeon RX 7900 XTX"
    assert bench.rocminfo_gpu_model("") == ""


def test_drm_model_reads_amd_cards_only(tmp_path):
    _card(tmp_path, 0, "0x10de", "NVIDIA thing")
    _card(tmp_path, 1, "0x1002")                                  # AMD without a product name
    _card(tmp_path, 2, "0x1002", "AMD Instinct MI300X")
    assert bench.drm_gpu_model(tmp_path) == "AMD Instinct MI300X"
    assert bench.drm_gpu_model(tmp_path / "absent") == ""


def test_gpu_model_order_and_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "sh", lambda cmd: ROCMINFO if cmd.startswith("rocminfo") else "lspci name")
    assert bench.gpu_model() == "AMD Radeon RX 7900 XTX"
    monkeypatch.setattr(bench, "sh", lambda cmd: "lspci name" if "lspci" in cmd else "")
    monkeypatch.setattr(bench, "drm_gpu_model", lambda root=None: "")
    assert bench.gpu_model() == "lspci name"
    monkeypatch.setattr(bench, "sh", lambda cmd: "")
    assert bench.gpu_model() == "unknown"


def test_gpu_name_falls_back_to_torch(monkeypatch):
    monkeypatch.setattr(bench, "sh", lambda cmd: "gfx1201:sramecc+:xnack-" if "gcnArchName" in cmd else "")
    monkeypatch.setattr(bench, "kfd_gpu_target", lambda nodes=None: "")
    assert bench.gpu_name() == "gfx1201"


def test_detection_survives_a_path_without_the_tools(tmp_path, monkeypatch):
    """Real sh(), PATH with no rocminfo, hipconfig or lspci and fake sysfs: every probe answers unknown, none raises."""
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(bench, "KFD_NODES", tmp_path / "kfd")
    monkeypatch.setattr(bench, "DRM_CARDS", tmp_path / "drm")
    monkeypatch.setattr(bench, "ROCM_INFO_FILES", ())
    monkeypatch.setattr(bench, "sdk_trees", lambda: [])
    monkeypatch.setattr(bench, "AMDGPU_VERSION_FILE", str(tmp_path / "none"))
    monkeypatch.setattr(bench, "torch_gpu", lambda attr: "")
    monkeypatch.setattr(bench, "sys", type("S", (), {"executable": str(tmp_path / "py")}))
    assert bench.gpu_model() == "unknown"
    assert bench.gpu_name() == "unknown"
    assert bench.rocm_version() == "unknown"
