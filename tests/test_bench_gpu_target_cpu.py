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
