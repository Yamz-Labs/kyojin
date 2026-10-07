"""CPU tests for the ROCm version detection of tools/bench.py, one per source and in order: /opt/rocm, then
rocm-sdk and hipconfig (on PATH, then next to the running interpreter), then the .info/version of a
rocm-sdk devel tree, then torch, then "unknown". The tree and the tools next to the interpreter are what a
container with only the rocm-sdk-devel wheel has, where /opt/rocm and PATH have neither."""
import importlib.util
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location("bench", Path(__file__).parent.parent / "tools" / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _tree(root, version):
    """A rocm-sdk devel tree root with a .info/version file in it."""
    (root / ".info").mkdir(parents=True, exist_ok=True)
    (root / ".info" / "version").write_text(f"{version}\n")
    return root


def _tool(root, name):
    """A fake ROCm command line tool (bench.py runs it only if the file is there)."""
    root.mkdir(parents=True, exist_ok=True)
    p = root / name
    p.write_text("#!/bin/sh\n")
    return p


@pytest.fixture
def bare(tmp_path, monkeypatch):
    """bench.py with no source at all: no /opt/rocm, nothing on PATH, no devel tree, an interpreter in an
    empty directory (so the venv running the tests cannot leak a real rocm-sdk or hipconfig in) and a sh()
    that answers nothing. rocm_version() then reports "unknown"."""
    monkeypatch.setattr(bench, "ROCM_INFO_FILES", ())
    monkeypatch.setattr(bench, "sdk_trees", lambda: [])
    monkeypatch.setattr(bench.shutil, "which", lambda tool: None)
    monkeypatch.setattr(bench, "sys", SimpleNamespace(executable=str(tmp_path / "nowhere" / "python")))
    monkeypatch.setattr(bench, "sh", lambda cmd: "")
    return tmp_path


def test_opt_rocm_wins_over_every_other_source(bare, tmp_path, monkeypatch):
    f = tmp_path / "version"
    f.write_text("7.1.1\n")
    monkeypatch.setattr(bench, "ROCM_INFO_FILES", (str(f),))
    monkeypatch.setattr(bench, "sdk_trees", lambda: [str(_tree(tmp_path / "sdk", "7.13.0"))])
    monkeypatch.setattr(bench, "sh", lambda cmd: "HIP 7.13.60980")
    assert bench.rocm_version() == "7.1.1"


def test_an_empty_opt_rocm_file_is_not_the_answer(bare, tmp_path, monkeypatch):
    f = tmp_path / "version"
    f.write_text("\n")
    monkeypatch.setattr(bench, "ROCM_INFO_FILES", (str(f),))
    monkeypatch.setattr(bench, "sdk_trees", lambda: [str(_tree(tmp_path / "sdk", "7.13.0"))])
    assert bench.rocm_version() == "7.13.0 (rocm-sdk)"


def test_rocm_sdk_on_path_wins_over_the_tree_and_torch(bare, tmp_path, monkeypatch):
    usr_bin = tmp_path / "usr" / "bin"
    monkeypatch.setattr(bench.shutil, "which", lambda tool: str(_tool(usr_bin, tool)) if tool == "rocm-sdk" else None)
    monkeypatch.setattr(bench, "sdk_trees", lambda: [str(_tree(tmp_path / "sdk", "7.13.0"))])
    monkeypatch.setattr(bench, "sh", lambda cmd: "7.13.0a20260411\nsecond line\n")
    assert bench.rocm_version() == "7.13.0a20260411"                 # first line only


def test_a_tool_that_prints_nothing_falls_through_to_the_next(bare, tmp_path, monkeypatch):
    usr_bin = tmp_path / "usr" / "bin"
    _tool(usr_bin, "rocm-sdk")
    _tool(usr_bin, "hipconfig")
    monkeypatch.setattr(bench.shutil, "which", lambda tool: str(usr_bin / tool))
    monkeypatch.setattr(bench, "sdk_trees", lambda: [str(_tree(tmp_path / "sdk", "7.13.0"))])
    monkeypatch.setattr(bench, "sh", lambda cmd: "" if "rocm-sdk" in cmd else "7.13.60980-c76140fa27")
    assert bench.rocm_version() == "7.13.60980-c76140fa27"


def test_hipconfig_next_to_the_interpreter_when_path_has_none(bare, tmp_path, monkeypatch):
    # The venv that runs bench.py installs rocm-sdk and hipconfig in its own bin; a shell that did not
    # activate it leaves them out of PATH. bench.py asks its own interpreter's directory next.
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "hipconfig").write_text("#!/bin/sh\necho 7.13.60980-c76140fa27\n")
    monkeypatch.setattr(bench, "sys", SimpleNamespace(executable=str(bin_dir / "python")))
    monkeypatch.setattr(bench, "sdk_trees", lambda: [str(_tree(tmp_path / "sdk", "7.13.0"))])
    asked = []

    def sh(cmd):
        asked.append(cmd)
        return "7.13.60980-c76140fa27" if cmd.endswith("--version") else ""

    monkeypatch.setattr(bench, "sh", sh)
    assert bench.rocm_version() == "7.13.60980-c76140fa27"
    assert asked == [f"{shlex.quote(str(bin_dir / 'hipconfig'))} --version"]   # the path it found, quoted for the shell


def test_devel_tree_wins_over_torch(bare, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "sdk_trees", lambda: [str(_tree(tmp_path / "sdk", "7.13.0"))])
    monkeypatch.setattr(bench, "sh", lambda cmd: "7.13.60980" if "torch" in cmd else "")
    assert bench.rocm_version() == "7.13.0 (rocm-sdk)"


def test_a_tree_without_a_version_file_does_not_answer(bare, tmp_path, monkeypatch):
    (tmp_path / "sdk").mkdir()
    monkeypatch.setattr(bench, "sdk_trees", lambda: [str(tmp_path / "sdk")])
    monkeypatch.setattr(bench, "sh", lambda cmd: "7.13.60980" if "torch" in cmd else "")
    assert bench.rocm_version() == "HIP 7.13.60980 (PyTorch)"


def test_torch_is_the_last_source_before_unknown(bare, monkeypatch):
    monkeypatch.setattr(bench, "sh", lambda cmd: "7.13.60980" if "torch" in cmd else "")
    assert bench.rocm_version() == "HIP 7.13.60980 (PyTorch)"


def test_unknown_when_nothing_answers(bare):
    assert bench.rocm_version() == "unknown"


def test_sdk_trees_order_and_sources(tmp_path, monkeypatch):
    # $EXL3_ROCM_SDK, then the wheel next to the running interpreter, then $EXL3_VENV.
    monkeypatch.setenv("EXL3_ROCM_SDK", str(tmp_path / "exported"))
    monkeypatch.setenv("EXL3_VENV", str(tmp_path / "venv"))
    monkeypatch.setattr(bench, "site_packages", lambda: [str(tmp_path / "site")])
    _tree(tmp_path / "exported", "7.13.0")
    _tree(tmp_path / "site" / bench.SDK_TREE, "7.13.0")
    _tree(tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / bench.SDK_TREE, "7.13.0")
    assert bench.sdk_trees() == [str(tmp_path / "exported"),
                                 str(tmp_path / "site" / bench.SDK_TREE),
                                 str(tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / bench.SDK_TREE)]


def test_sdk_trees_without_any_switch(tmp_path, monkeypatch):
    monkeypatch.delenv("EXL3_ROCM_SDK", raising=False)
    monkeypatch.delenv("EXL3_VENV", raising=False)
    monkeypatch.setattr(bench, "site_packages", lambda: [])
    assert bench.sdk_trees() == []


def test_read_version_of_a_missing_or_empty_file(tmp_path):
    assert bench.read_version(tmp_path / "absent") == ""
    (tmp_path / "empty").write_text("  \n")
    assert bench.read_version(tmp_path / "empty") == ""
    (tmp_path / "version").write_text(" 7.13.0 \n")
    assert bench.read_version(tmp_path / "version") == "7.13.0"


def test_tool_version_returns_nothing_without_the_tool(bare):
    assert bench.tool_version("rocm-sdk version") == ""
