"""zero-GPU check of the 1.9.1 shipped flags: each one is switched on in serve.SHIPPED_ENV, is read from the environment by its module with the 1.9.0 behaviour ("0") as the code default,
and apply_env() never overrides a value the user set (HY4_<FLAG>=0 is the rollback to 1.9.0)"""
import ast
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)

# flag -> (module that reads it, shipped value in 1.9.1)
FLAGS = {
    "HY4_PF_HC2": ("exllamav3/loader/hc_fused.py", "1"),
    "HY4_PF_ROUTER16": ("exllamav3/loader/hy4_forward.py", "1"),
    "HY4_PF_KV_HM": ("exllamav3/loader/prefill_unabs.py", "1"),
    "HY4_PF_QO_HM": ("exllamav3/loader/prefill_unabs.py", "1"),
    "HY4_PF_Q3K_V2": ("exllamav3/loader/kq_moe_hip.py", "1"),
    "HY4_HEAD_Q8": ("exllamav3/loader/hy4_forward.py", "1"),
    "HY4_GRAPH_FORK": ("exllamav3/loader/hy4_forward.py", "1"),
    "HY4_RG_TUNE": ("exllamav3/loader/router_gemv.py", "1"),
    "HY4_BG_V2": ("exllamav3/loader/mla_bgemv.py", "1"),
}


def _shipped_env():
    src = open(os.path.join(TREE, "tools", "hy4", "serve.py")).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SHIPPED_ENV" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("SHIPPED_ENV not found")


def test_every_flag_is_shipped_on():
    env = _shipped_env()
    for flag, (_, val) in FLAGS.items():
        assert env.get(flag) == val, flag


def test_every_flag_has_an_env_off_switch():
    for flag, (path, _) in FLAGS.items():
        src = open(os.path.join(TREE, path)).read()
        assert re.search(r'environ\.get\("%s", "0"\)' % flag, src), f"{flag}: {path} must read it from the environment with the 1.9.0 behaviour (0) as code default"


@pytest.fixture
def env(monkeypatch):
    """apply_env() sets many HY4_* variables: it runs on a private copy of the environment, nothing leaks into the other tests"""
    monkeypatch.setattr(os, "environ", dict(os.environ))


def _apply_env():
    """apply_env() and the constants it reads, cut out of serve.py (importing the module would pull the server stack in)"""
    src = open(os.path.join(TREE, "tools", "hy4", "serve.py")).read()
    keep = {"SHIPPED_ENV", "SLOTS_OFF"}
    nodes = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in keep for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "apply_env":
            nodes.append(node)
    assert len(nodes) == 3, "SHIPPED_ENV, SLOTS_OFF and apply_env must all be found in tools/hy4/serve.py"
    ns = {"os": os, "TREE": TREE}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "serve.py", "exec"), ns)
    return ns


def test_apply_env_keeps_user_values(env):
    ns = _apply_env()
    for flag in FLAGS:
        os.environ.pop(flag, None)
    os.environ["HY4_PF_HC2"] = "0"             # the rollback to 1.9.0: an explicit value always wins, with one slot or several
    for slots in (1, 4):
        os.environ["HY4_PF_HC2"] = "0"
        ns["apply_env"](4096, slots = slots)
        assert os.environ["HY4_PF_HC2"] == "0"
        assert os.environ["HY4_PF_ROUTER16"] == "1"    # an unset flag takes its shipped value
        os.environ.pop("HY4_PF_ROUTER16")


def test_apply_env_turns_slot_unsafe_levers_off_with_slots(env):
    ns = _apply_env()
    for flag in ns["SLOTS_OFF"]:
        assert ns["SHIPPED_ENV"].get(flag) == "1", flag
        os.environ.pop(flag, None)
        ns["apply_env"](4096, slots = 1)
        assert os.environ[flag] == "1"
        os.environ.pop(flag)
        ns["apply_env"](4096, slots = 2)
        assert os.environ[flag] == "0"
        os.environ[flag] = "1"                  # explicit value still wins (MuEngine then refuses it)
        ns["apply_env"](4096, slots = 2)
        assert os.environ[flag] == "1"
