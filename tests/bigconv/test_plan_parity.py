"""plan.json byte parity: bigconv (DS4.1 adapter) vs the untouched tools/ds41/plan.py on the real DS4.1 tensor map.
BIGCONV_DS41_MAP=<tensor_map.json of the real DS4.1 index (96085 tensors)>; skipped without it."""
import os, subprocess, sys, filecmp
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP = os.environ.get("BIGCONV_DS41_MAP", "")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
pytestmark = pytest.mark.skipif(not (MAP and os.path.exists(MAP)), reason="set BIGCONV_DS41_MAP")

VARIANTS = {
    "default": [],
    "t26": ["--target", "2.6"],
    "freq": ["--freq", f"{D}/freq.json"],
    "layerbits": ["--layer-bits", "0-9:3,10-29:2,30-39:3"],
    "noeng": ["--no-engram"],
    "mtpq": ["--mtp-queue"],
    "alt": ["--alt-attn", "3,5", "--alt-shared", "3,5"],
    "rolebits": ["--role-bits", "q_b=3,wo_a=3,wo_b=3,q_a=5,kv=5"],
    "ebf": ["--expert-bits-file", f"{D}/eb.json"],
    "alloc": ["--alloc", f"{D}/alloc.json"],
    "allocmtpq": ["--alloc", f"{D}/alloc.json", "--mtp-queue"],
    "freq_mtpexp": ["--freq", f"{D}/freq.json", "--mtp-exp-bits", "3"],
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_plan_bytes_identical(name, tmp_path):
    args = VARIANTS[name]
    ref, new = str(tmp_path / "ref.json"), str(tmp_path / "new.json")
    r = subprocess.run([sys.executable, f"{ROOT}/tools/ds41/plan.py", "--map", MAP, "--out", ref, *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    n = subprocess.run([f"{ROOT}/tools/bigconv/bigconv", "plan", "--arch", "ds41", "--map", MAP,
                        "--out", new, *args], capture_output=True, text=True)
    assert n.returncode == 0, n.stderr[-500:]
    assert filecmp.cmp(ref, new, shallow=False), name
    assert r.stdout == n.stdout, "summary text differs"
