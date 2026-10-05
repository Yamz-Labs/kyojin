import importlib.util, io, os, contextlib
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "hip_lib", Path(__file__).parent.parent / "exllamav3" / "util" / "hip_lib.py")
hip_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hip_lib)


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"")


def _find(d):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        return hip_lib.find_hip_runtime(str(d)), err.getvalue()


def test_plain_name_wins_and_is_silent(tmp_path):
    _touch(tmp_path, "libamdhip64.so", "libamdhip64.so.7")
    path, log = _find(tmp_path)
    assert path == str(tmp_path / "libamdhip64.so") and log == ""


def test_versioned_fallback_picks_highest_and_logs_once(tmp_path):
    _touch(tmp_path, "libamdhip64.so.6", "libamdhip64.so.7", "libamdhip64.so.7.1.2", "libamdhip64.so.10",
           "libamdhip64.so.bak", "libamdhip64.so.7.txt")
    path, log = _find(tmp_path)
    assert path == str(tmp_path / "libamdhip64.so.10")
    assert log.count("\n") == 1 and "libamdhip64.so.10" in log
    assert _find(tmp_path)[1] == ""   # already reported


def test_dotted_versions_sort_numerically(tmp_path):
    _touch(tmp_path, "libamdhip64.so.7", "libamdhip64.so.7.1.2", "libamdhip64.so.7.1.10")
    assert _find(tmp_path)[0] == str(tmp_path / "libamdhip64.so.7.1.10")


def test_none_when_absent_and_load_raises(tmp_path):
    assert _find(tmp_path)[0] is None
    try:
        hip_lib.load_hip_runtime(str(tmp_path))
    except OSError:
        pass
    else:
        raise AssertionError("expected OSError")
