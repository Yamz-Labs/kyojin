"""CPU tests for the ablit tools (arch table GLM / MiMo-V2 / DS4.1, hook proofs, mock directions, gate verdict). No model, no GPU."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

T = Path(__file__).resolve().parents[1] / "tools" / "ablit"
sys.path.insert(0, str(T))
import arch as A  # noqa: E402


def run(*a):
    r = subprocess.run([sys.executable, *map(str, a)], capture_output=True, text=True, cwd=T.parents[1])
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    return r.stdout


def test_arch_table():
    g, m = A.get("glm"), A.get("mimo_v2")
    assert (g["n_layers"], g["readout"], g["pca_layer"]) == (45, "hc_mean", 29)
    assert (m["n_layers"], m["readout"], m["hidden"]) == (48, "plain", 4096)
    assert m["pca_layer"] == round(29 * 47 / 44) and abs(m["proj_f"] - 29.344821947743544 * 47 / 44) < 1e-9
    assert m["chat"].endswith("<|im_start|>assistant\n")


def test_detect(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mimo_v2", "architectures": ["MiMoV2ForCausalLM"]}))
    assert A.detect(tmp_path) == "mimo_v2"
    assert A.detect(tmp_path / "missing") == "glm"


def test_hook_proofs():
    assert "ALL CPU PROOFS PASS" in run(T / "hook_ablate.py", "--prove")


@pytest.mark.parametrize("arch", ["glm", "mimo_v2"])
def test_mock_directions(arch, tmp_path):
    assert "DRY-RUN-TEST PASS" in run(T / "collect_dirs.py", "--mock", "--arch", arch, "--out", tmp_path, "--dry-run-test")


def test_mimo_spec_export(tmp_path):
    run(T / "collect_dirs.py", "--mock", "--arch", "mimo_v2", "--mock-layers", 48, "--mock-hidden", 4096,
        "--mock-n", 16, "--out", tmp_path / "d")
    kernel = T.parents[1] / "tools" / "ablit" / "kernel_glm_ablit5.json"
    run(T / "export_spec.py", "--dirs", tmp_path / "d", "--arch", "mimo_v2", "--kernel", kernel,
        "--scale-kernel", "--out", tmp_path / "s")
    s = json.loads((tmp_path / "s" / "edit_spec.json").read_text())
    assert s["n_layers"] == 48 and len(s["attn_w"]) == 48 and len(s["mlp_w"]) == 48
    assert "MTP" in s["apply"]
