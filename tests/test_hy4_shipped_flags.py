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


# ---- 1.9.2: opt-in fp8 KV (long context) and the head-blocked attend that reads it


def _read(path):
    return open(os.path.join(TREE, path)).read()


def test_kv_fp8_is_off_by_default_and_not_shipped():
    """-c 65536 keeps the fp32 cache: HY4_KV_FP8 is not in SHIPPED_ENV and its code default is 0 (the long-context option is the user's env)"""
    assert "HY4_KV_FP8" not in _shipped_env()
    assert re.search(r'environ\.get\("HY4_KV_FP8", "0"\) == "1"', _read("exllamav3/loader/step_graph.py"))


def test_head_blocked_attend_defaults_and_switches():
    """HY4_LC_HB defaults on and HY4_LC_HB=0 switches it off; it can only run on the fp8 cache (qf), so it is on by default only when HY4_KV_FP8 is on"""
    src = _read("exllamav3/loader/lc_sparse.py")
    assert re.search(r'HB_ON = os\.environ\.get\("HY4_LC_HB", "1"\) == "1"', src)
    assert re.search(r'HB_SPLIT = int\(os\.environ\.get\("HY4_LC_HB_SPLIT", "-1"\)\)', src)
    assert "HY4_LC_HB" not in _shipped_env() and "HY4_LC_HB_SPLIT" not in _shipped_env()
    assert len(re.findall(r"if qf and HB_ON", src)) == 2, "both head-blocked launches must be gated by the fp8 cache flag qf"
