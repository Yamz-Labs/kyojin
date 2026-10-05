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


def _maps(*paths):
    return "\n".join(f"7f00{i}000-7f00{i}fff r-xp 00000000 08:01 {i}  {p}" for i, p in enumerate(paths)) + "\n"


def _default_dir_find(maps):
    import sys, types
    fake = types.ModuleType("torch")
    fake.__file__ = "/nonexistent/torch/__init__.py"
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return hip_lib.find_hip_runtime(read_maps = lambda: maps), err.getvalue()
    finally:
        if saved is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved


def test_default_dir_falls_back_to_the_mapped_library_and_logs_once():
    hip_lib._logged.clear()
    maps = _maps("/usr/lib/libc.so.6", "/venv/_rocm_sdk_core/lib/libamdhip64.so.7")
    path, log = _default_dir_find(maps)
    assert path == "/venv/_rocm_sdk_core/lib/libamdhip64.so.7"
    assert log.count("\n") == 1 and "already loaded" in log
    assert _default_dir_find(maps)[1] == ""


def test_default_dir_none_when_nothing_is_mapped_or_maps_unreadable():
    hip_lib._logged.clear()
    assert _default_dir_find(_maps("/usr/lib/libc.so.6"))[0] is None

    def boom():
        raise OSError("no /proc")
    assert hip_lib._loaded_hip_runtime(boom) is None


def test_explicit_dir_never_uses_the_mapped_library(tmp_path):
    hip_lib._logged.clear()
    assert hip_lib.find_hip_runtime(str(tmp_path), read_maps = lambda: _maps("/x/libamdhip64.so.7")) is None
