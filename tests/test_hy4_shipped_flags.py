"""zero-GPU check of the 1.9.1 shipped flags: each one is switched on in serve.SHIPPED_ENV, is read from the environment by its module with the 1.9.0 behaviour ("0") as the code default,
and apply_env() never overrides a value the user set (HY4_<FLAG>=0 is the rollback to 1.9.0)"""
import ast
import os
import re

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


def test_apply_env_keeps_user_values():
    src = open(os.path.join(TREE, "tools", "hy4", "serve.py")).read()
    body = src[src.index("def apply_env"):]
    body = body[:body.index("\ndef ", 10)]
    assert "os.environ.setdefault(k, v)" in body
