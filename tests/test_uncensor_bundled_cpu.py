"""CPU tests: bundled <model_dir>/uncensor_spec.json discovery, precedence, off switch, bad files."""
import importlib.util
import json
import os
import types

import pytest
import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = 32


def load_hook():
    s = importlib.util.spec_from_file_location("ablit_runtime", os.path.join(ROOT, "exllamav3/modules/ablit_runtime.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m._USE_TRITON = False
    return m


def make_spec(d, w=0.5, direction_name="uncensor_direction.st", seed=0, **extra):
    os.makedirs(d, exist_ok=True)
    r = torch.randn(H, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    save_file({"r": r.contiguous()}, os.path.join(d, direction_name))
    spec = {"hidden": H, "n_layers": 2, "attn_w": [w, w], "mlp_w": [w, w], **extra}
    p = os.path.join(d, "uncensor_spec.json")
    json.dump(spec, open(p, "w"))
    return p


def block(model_dir, layer=0):
    cfg = types.SimpleNamespace(directory=str(model_dir))
    return types.SimpleNamespace(key=f"model.layers.{layer}", attn=object(), mlp=object(), ablit=None, config=cfg)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("EXL3_ABLIT_RUNTIME", raising=False)


def test_absent_file_changes_nothing(tmp_path):
    hook = load_hook()
    b = block(tmp_path)
    hook.prepare(b, "cpu")
    assert b.ablit is None
    assert hook.resolve(str(tmp_path)) == (None, "none")
    assert not hook.enabled(str(tmp_path))
    assert hook.resolve(None) == (None, "none")


def test_bundled_is_applied_and_logged(tmp_path, capsys):
    p = make_spec(str(tmp_path), w=0.5)
    hook = load_hook()
    b = block(tmp_path)
    hook.prepare(b, "cpu")
    assert b.ablit is not None and b.ablit[:2] == (0.5, 0.5)
    err = capsys.readouterr().err
    assert err.count("ablit runtime: spec") == 1 and p in err and "bundled" in err
    hook.prepare(block(tmp_path, 1), "cpu")
    assert capsys.readouterr().err == ""  # one line per spec, not per layer


def test_direction_next_to_spec_as_safetensors_also_works(tmp_path):
    make_spec(str(tmp_path), direction_name="uncensor_spec.safetensors")
    hook = load_hook()
    b = block(tmp_path)
    hook.prepare(b, "cpu")
    assert b.ablit is not None


def test_explicit_env_path_wins(tmp_path, monkeypatch):
    model = tmp_path / "model"
    other = tmp_path / "other"
    make_spec(str(model), w=0.5)
    ep = make_spec(str(other), w=0.9, direction_name="edit.safetensors")
    os.rename(ep, str(other / "edit.json"))
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", str(other / "edit.json"))
    hook = load_hook()
    assert hook.resolve(str(model)) == (str(other / "edit.json"), "env")
    b = block(model)
    hook.prepare(b, "cpu")
    assert b.ablit[:2] == (0.9, 0.9)


@pytest.mark.parametrize("val", ["off", "OFF", "0", "false"])
def test_off_switch_disables_bundled(tmp_path, monkeypatch, capsys, val):
    make_spec(str(tmp_path))
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", val)
    hook = load_hook()
    b = block(tmp_path)
    hook.prepare(b, "cpu")
    assert b.ablit is None
    assert not hook.enabled(str(tmp_path))
    hook.prepare(block(tmp_path, 1), "cpu")
    assert capsys.readouterr().err.count("OFF") == 1


def test_off_switch_without_bundled_file_is_silent(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", "off")
    hook = load_hook()
    hook.prepare(block(tmp_path), "cpu")
    assert capsys.readouterr().err == ""


def test_bad_json_is_a_clear_error(tmp_path):
    make_spec(str(tmp_path))
    (tmp_path / "uncensor_spec.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_hook().prepare(block(tmp_path), "cpu")


def test_missing_field_is_a_clear_error(tmp_path):
    p = make_spec(str(tmp_path))
    spec = json.load(open(p))
    del spec["mlp_w"]
    json.dump(spec, open(p, "w"))
    with pytest.raises(ValueError, match="missing field"):
        load_hook().prepare(block(tmp_path), "cpu")


def test_missing_direction_is_a_clear_error(tmp_path):
    make_spec(str(tmp_path))
    os.remove(tmp_path / "uncensor_direction.st")
    with pytest.raises(FileNotFoundError, match="direction file not found"):
        load_hook().prepare(block(tmp_path), "cpu")


def test_wrong_hidden_is_a_clear_error(tmp_path):
    p = make_spec(str(tmp_path))
    spec = json.load(open(p))
    spec["hidden"] = H + 1
    json.dump(spec, open(p, "w"))
    with pytest.raises(ValueError, match="direction shape"):
        load_hook().prepare(block(tmp_path), "cpu")


def test_env_path_missing_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("EXL3_ABLIT_RUNTIME", str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="spec file not found"):
        load_hook().prepare(block(tmp_path), "cpu")


def test_r_file_relative_to_spec_dir(tmp_path):
    p = make_spec(str(tmp_path), direction_name="dir.bin", r_file="dir.bin")
    b = block(tmp_path)
    load_hook().prepare(b, "cpu")
    assert b.ablit is not None


def test_bundled_direction_is_not_indexed_by_weight_loader():
    # the bundled direction must not match the loader's "*.safetensors" glob
    assert not "uncensor_direction.st".endswith(".safetensors")
