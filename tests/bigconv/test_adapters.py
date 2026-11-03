"""GLM-5.x-Flash and MiniMax-M3 adapters against their real safetensors header maps (committed fixtures)."""
import gzip, json, os, sys, collections
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "bigconv", "fixtures")
CFG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "bigconv", "configs")
from bigconv import adapters  # noqa
from bigconv.arch import get_adapter
from bigconv.plan import build_plan

EXPECT = {  # arch -> (quant linears, routed experts linears, mtp routed)
    "glm5": (37648, 36288, 864),
    "minimax_m3": (22423, 21888, 0),
}


def load(arch):
    tm = json.load(gzip.open(f"{FIX}/{arch}.tensor_map.json.gz", "rt"))
    return get_adapter(arch, json.load(open(f"{CFG}/{arch}.config.json"))), tm["tensors"]


@pytest.mark.parametrize("arch", sorted(EXPECT))
def test_every_quant_linear_has_the_real_shape(arch):
    a, T = load(arch)
    names = set(T)
    kinds = collections.Counter()
    for n, t in T.items():
        c = a.classify(n, T)
        if c.action == "quant":
            kinds[c.kind] += 1
            assert list(a.shape_of(n, names)) == list(t["shape"]), n
    assert sum(kinds.values()) == EXPECT[arch][0]
    assert kinds["routed"] == EXPECT[arch][1]
    assert kinds["mtp_routed"] == EXPECT[arch][2]


@pytest.mark.parametrize("arch", sorted(EXPECT))
def test_plan_builds_and_hits_target(arch):
    a, T = load(arch)
    p = build_plan({"tensors": T}, 2.3, a, None, {"attn": 5, "shared": 5, "mtp": 4, "head": 6, "hot": 3, "cold": 2, "mtp_exp": None},
                   None, False, False, {"attn": [], "shared": []}, {}, None, None)
    assert abs(p["summary"]["routed_avg_bpw"] - 2.3) < 0.01
    q = [e for e in p["entries"] if e["action"] == "quant"]
    assert q and all(e["bits"] in (2, 3, 4, 5, 6) for e in q)


def test_todo_ops_are_listed_where_the_forward_differs():
    for arch in EXPECT:
        a, _ = load(arch)
        assert a.TODO_OPS and all(isinstance(x, str) for x in a.TODO_OPS)
        with pytest.raises(NotImplementedError):
            a.capture_runner()


def test_sheet_numbers_glm_and_minimax_params():
    from bigconv.sheet import build_sheet
    for arch, tot, act in (("glm5", 320.8, 17.6), ("minimax_m3", 426.2, 26.0)):
        a, T = load(arch)
        idx = json.load(gzip.open(f"{FIX}/{arch}.index.json.gz", "rt"))
        s = build_sheet(a, idx, {"tensors": T}, 2_000_000, 2.3)
        assert abs(s["params"]["total_b"] - tot) < 0.2 and abs(s["params"]["active_b"] - act) < 0.2
        assert abs(s["params"]["config_b_approx"] - s["params"]["total_b"]) < 0.5      # two ways agree
        assert s["capture_ram_memmap"]["total_gib"] < 16 < s["capture_ram_current"]["total_gib"]
