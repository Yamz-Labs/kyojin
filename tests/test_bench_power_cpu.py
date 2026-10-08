"""CPU tests for the power line of tools/bench.py: the limit the user passes, the platform profile, mains or
battery, and "unknown" when the machine exposes none of them."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("bench", Path(__file__).parent.parent / "tools" / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _supply(root, name, kind, online=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "type").write_text(f"{kind}\n")
    if online is not None:
        (d / "online").write_text(f"{online}\n")


def test_nothing_readable_is_unknown(tmp_path):
    assert bench.power_mode(profile=tmp_path / "none", supplies=tmp_path / "none") == "unknown"


def test_limit_profile_and_mains(tmp_path):
    (tmp_path / "profile").write_text("performance\n")
    _supply(tmp_path / "ps", "ADP1", "Mains", 1)
    _supply(tmp_path / "ps", "BAT0", "Battery")
    assert bench.power_mode(" 71 W ", tmp_path / "profile", tmp_path / "ps") == "71 W, profile performance, on mains"


def test_on_battery(tmp_path):
    _supply(tmp_path / "ps", "ADP1", "Mains", 0)
    assert bench.power_mode(profile=tmp_path / "none", supplies=tmp_path / "ps") == "on battery"


def test_blank_limit_is_dropped(tmp_path):
    assert bench.power_mode("  ", tmp_path / "none", tmp_path / "none") == "unknown"
