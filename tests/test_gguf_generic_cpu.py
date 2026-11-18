"""CPU tests for the generic GGUF loader hooks (EXL3_GGUF_GENERIC, default off). No GPU, no extension at import:
the modules under test are loaded by path under a stub `exllamav3` package.
  flag off: gguf_config.hf_config == origin/main's (tests/fixtures/gguf_config_main.py, `git show origin/main:exllamav3/loader/gguf_config.py`)
            for llama / qwen3 / qwen35moe / qwen35 / qwen4exp, unknown archs raise main's ValueError (same message)
  flag on : unknown archs dispatch to hf_config_generic; olmoe / gpt-oss resolve to the generic architecture
  .bias expert slicing of a fake stacked F32 tensor; norm_topk on / off numeric check."""
import ast, importlib.util, os, sys, types, struct
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAG = "EXL3_GGUF_GENERIC"


def _stub_pkgs():
    for name, sub in (("exllamav3", ""), ("exllamav3.loader", "loader")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(ROOT, "exllamav3", sub)]
            sys.modules[name] = m


def _load(modname, path, pkg="exllamav3.loader"):
    _stub_pkgs()
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def cfg_new():
    return _load("exllamav3.loader.gguf_config", os.path.join(ROOT, "exllamav3/loader/gguf_config.py"))

def cfg_main():
    return _load("exllamav3.loader.gguf_config_main", os.path.join(ROOT, "tests/fixtures/gguf_config_main.py"))

def coll():
    return _load("exllamav3.loader.gguf_collection", os.path.join(ROOT, "exllamav3/loader/gguf_collection.py"))


class Meta(dict):
    """every missing key reads as 4 (dict.get still returns its default): enough for the mapping functions"""
    def __missing__(self, k):
        return 4


class FakeG:
    def __init__(self, arch, extra=None):
        self.meta = Meta({"general.architecture": arch, arch + ".attention.layer_norm_rms_epsilon": 1e-6,
                          arch + ".rope.freq_base": 1e6})
        self.meta.update({"qwen4exp.ple.head_offsets": [0, 1], "qwen4exp.rope.dimension_sections": [2, 2, 2, 2], "qwen4exp.ple.layers": [1, 3]} if arch == "qwen4exp" else {})
        self.meta.update(extra or {})
        self.infos = {"token_embd.weight": (0, (64, 1000), 0), "output.weight": (0, (64, 1000), 0)}
        self.name_map = {}


def run(mod, g):
    try:
        return ("ok", mod.hf_config(g))
    except Exception as e:                                   # noqa: BLE001 - compare class + message
        return ("err", type(e).__name__, str(e))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    return monkeypatch


@pytest.mark.parametrize("arch", ["llama", "qwen3", "qwen35moe", "qwen35", "qwen4exp"])
def test_flag_off_same_as_main(env, arch):
    new, main = cfg_new(), cfg_main()
    assert set(new.GGUF_ARCHS) == set(main.GGUF_ARCHS)
    r_new, r_main = run(new, FakeG(arch)), run(main, FakeG(arch))
    print(arch, r_main[0], str(r_main[1:])[:100])
    assert r_new == r_main
    env.setenv(FLAG, "0")
    assert run(new, FakeG(arch)) == r_main
    # nextn layers: the hard-coded qwen35 mappings only subtract them with the flag on
    if arch in ("qwen35moe", "qwen35") and r_main[0] == "ok":
        g = FakeG(arch, {arch + ".nextn_predict_layers": 1})
        assert run(new, g) == run(main, g)
        env.setenv(FLAG, "1")
        a, b = run(new, g), run(main, g)
        assert a[0] == "ok" and a[1]["text_config"]["num_hidden_layers"] == b[1]["text_config"]["num_hidden_layers"] - 1


@pytest.mark.parametrize("arch", ["gpt-oss", "foo", "olmoe"])
def test_flag_off_unknown_raises_main_error(env, arch):
    new, main = cfg_new(), cfg_main()
    r_new, r_main = run(new, FakeG(arch)), run(main, FakeG(arch))
    assert r_main[0] == "err" and r_main[1] == "ValueError" and "has no standard-class mapping" in r_main[2]
    assert r_new == r_main
    env.setenv(FLAG, "0")
    assert run(new, FakeG(arch)) == r_main


@pytest.mark.parametrize("arch", ["gpt-oss", "foo", "olmoe"])
def test_flag_on_dispatches_to_generic(env, arch):
    new = cfg_new()
    env.setenv(FLAG, "1")
    stub = types.ModuleType("exllamav3.architecture.gguf_generic")
    stub.hf_config_generic = lambda g: {"generic": g.meta["general.architecture"]}
    pkg = types.ModuleType("exllamav3.architecture"); pkg.__path__ = []
    env.setitem(sys.modules, "exllamav3.architecture", pkg)
    env.setitem(sys.modules, "exllamav3.architecture.gguf_generic", stub)
    assert new.hf_config(FakeG(arch)) == {"generic": arch}
    # a hard-coded arch never reaches the generic path, flag on or off
    assert run(new, FakeG("llama")) == run(cfg_main(), FakeG("llama"))


def test_flag_on_architecture_resolves_generic():
    """real package in a subprocess (the other tests install stub packages in this one); skipped when the extension is not built"""
    import subprocess
    code = ("import os\nimport exllamav3.architecture.architectures as A\n"
            "os.environ.pop('EXL3_GGUF_GENERIC', None)\n"
            "assert 'GGUFGenericForCausalLM' not in A.get_architectures()\n"
            "os.environ['EXL3_GGUF_GENERIC'] = '0'\nassert 'GGUFGenericForCausalLM' not in A.get_architectures()\n"
            "os.environ['EXL3_GGUF_GENERIC'] = '1'\n"
            "a = A.get_architectures()['GGUFGenericForCausalLM']\n"
            "import exllamav3.architecture.gguf_generic as G\n"
            "assert a['model_class'] is G.GGUFGenericModel and a['config_class'].arch_string == 'GGUFGenericForCausalLM'\n"
            "from exllamav3.loader import gguf_config as C\n"
            "assert C.hf_config.__module__\n"
            "print('RESOLVED')")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", PYTHONPATH=ROOT)
    env.pop(FLAG, None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, timeout=200)
    if r.returncode != 0 and ("exllamav3_ext" in r.stderr or "ModuleNotFoundError" in r.stderr):
        pytest.skip("extension not built: " + r.stderr.strip().splitlines()[-1][:100])
    assert r.returncode == 0 and "RESOLVED" in r.stdout, r.stderr[-800:]


def test_generic_resolves_gpt_oss_olmoe_config():
    """flag on, real hf_config_generic (needs the built package, subprocess): gpt-oss and olmoe headers -> GGUFGenericForCausalLM config"""
    import subprocess
    code = r"""
import os, types
os.environ['EXL3_GGUF_GENERIC'] = '1'
from exllamav3.loader import gguf_config as C
class M(dict): pass
def fake(arch, moe_dims, extra):
    nl, ne, nq, E = 2, 64, 4, 4
    m = M({'general.architecture': arch, arch+'.embedding_length': ne, arch+'.attention.head_count': nq, arch+'.attention.head_count_kv': 2,
           arch+'.block_count': nl, arch+'.attention.key_length': 16, arch+'.attention.layer_norm_rms_epsilon': 1e-5,
           arch+'.context_length': 128, arch+'.expert_count': E, arch+'.expert_used_count': 2, arch+'.expert_feed_forward_length': 32})
    m.update(extra)
    inf = {'token_embd.weight': (0, (ne, 100), 0), 'output.weight': (0, (ne, 100), 0), 'output_norm.weight': (0, (ne,), 0)}
    for i in range(nl):
        b = f'blk.{i}.'
        inf.update({b+'attn_norm.weight': (0,(ne,),0), b+'ffn_norm.weight': (0,(ne,),0), b+'attn_q.weight': (0,(ne, nq*16),0),
                    b+'attn_k.weight': (0,(ne, 32),0), b+'attn_v.weight': (0,(ne, 32),0), b+'attn_output.weight': (0,(nq*16, ne),0),
                    b+'ffn_gate_inp.weight': (0,(ne, E),0), b+'ffn_gate_exps.weight': (8,(ne, 32, E),0),
                    b+'ffn_up_exps.weight': (8,(ne, 32, E),0), b+'ffn_down_exps.weight': (8,(32, ne, E),0)})
    return types.SimpleNamespace(meta=m, infos=inf, name_map={})
for arch in ('olmoe', 'gpt-oss'):
    cfg = C.hf_config(fake(arch, None, {}))
    assert cfg['architectures'] == ['GGUFGenericForCausalLM'], cfg
    print(arch, 'norm_topk_prob', cfg.get('norm_topk_prob'), 'gaps', cfg.get('gaps'))
print('RESOLVED')
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, timeout=200)
    if r.returncode != 0 and ("exllamav3_ext" in r.stderr or "ModuleNotFoundError" in r.stderr):
        pytest.skip("extension not built")
    assert r.returncode == 0 and "RESOLVED" in r.stdout, r.stderr[-800:]
    print(r.stdout)


def _fake_collection(tmp_path, E=4, R=5):
    c = coll()
    stacked = (np.arange(E * R, dtype=np.float32) * 0.5 + 1)     # ggml dims (R, E): row e = values [e*R:(e+1)*R]
    f = tmp_path / "bias.bin"; f.write_bytes(b"\0" * 8 + stacked.tobytes())
    g = object.__new__(c.GGUFCollection)
    g.__dict__.update(infos={"blk.0.ffn_up_exps.bias": (0, (R, E), 8), "blk.0.ffn_up_exps.weight": (2, (32, 3, E), 0),
                             "blk.0.ffn_gate_inp.bias": (0, (E,), 8)},
                      name_map={"eng.e2": "blk.0.ffn_up_exps.e2.bias", "eng.e9": "blk.0.ffn_up_exps.e9.bias",
                                "eng.w": "blk.0.ffn_up_exps.e1.weight"},
                      new_tensors=None, aux=None, path="fake")
    g._pread = lambda off, n: np.frombuffer(f.read_bytes()[off:off + n], dtype=np.uint8)
    return g, stacked


def test_bias_slice(tmp_path):
    g, st = _fake_collection(tmp_path)
    assert g._exps_slice("blk.0.ffn_up_exps.e3.bias") == ("blk.0.ffn_up_exps.bias", 3)
    assert g._exps_slice("blk.0.ffn_up_exps.e3.weight") == ("blk.0.ffn_up_exps.weight", 3)
    assert g._exps_slice("blk.0.ffn_up_exps.e3") == ("blk.0.ffn_up_exps.weight", 3)
    assert g._exps_slice("blk.0.ffn_up_exps.bias") is None and g._exps_slice("blk.0.ffn_down_exps.e3.bias") is None
    assert g._exps_slice("blk.0.ffn_up_exps.eX.bias") is None
    assert g.has_tensor("eng.e2") is True
    for e in range(4):
        g.name_map["k"] = f"blk.0.ffn_up_exps.e{e}.bias"
        t = g.get_tensor("k")
        assert t.dtype == torch.float32 and t.shape == (5,) and np.array_equal(t.numpy(), st[e * 5:(e + 1) * 5])
    with pytest.raises(AssertionError):
        g.get_tensor("eng.e9")                                   # expert 9 of 4
    # quant weight slices still use the old protocol (name resolution only; the payload read is not faked here)
    assert g._exps_slice("blk.0.ffn_up_exps.e1.weight") == ("blk.0.ffn_up_exps.weight", 1)


def test_f32_weight_slice_still_refused(tmp_path):
    g, _ = _fake_collection(tmp_path)
    g.infos["blk.1.ffn_up_exps.weight"] = (0, (5, 4), 8)
    g.name_map["k"] = "blk.1.ffn_up_exps.e1.weight"
    with pytest.raises(ValueError, match="per-expert slice of a F32 stacked tensor is not supported: blk.1.ffn_up_exps.weight \\(expert 1\\)"):
        g.get_tensor("k")


def _routing_ns():
    """_router_logits / _routing_std_nonorm / routing_std source taken from the file (the module imports the extension)."""
    path = os.path.join(ROOT, "exllamav3/modules/block_sparse_mlp_routing.py")
    src = open(path).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_router_logits", "_routing_std_nonorm")]
    ns = {"torch": torch, "os": os, "_pad_rows": lambda y: y}
    exec(compile(ast.Module(keep, []), path, "exec"), ns)
    return ns, src


def test_norm_topk():
    ns, src = _routing_ns()
    assert "if not cfg.norm_topk:\n        return _routing_std_nonorm(cfg, y, params)" in src
    assert "norm_topk: bool = True" in src                       # RoutingCFG default = today's renormalised routing
    torch.manual_seed(0)
    H, E, K, B = 16, 8, 3, 6
    cfg = types.SimpleNamespace(gate_tensor=torch.randn(H, E).half(), num_experts=E, num_experts_per_tok=K, per_expert_scale=None)
    y = torch.randn(B, H).half()
    sel, w = ns["_routing_std_nonorm"](cfg, y, {})
    logits = (y @ cfg.gate_tensor).float()
    probs = torch.softmax(logits, -1)
    ref_sel = torch.sort(logits, dim=-1, descending=True, stable=True)[1][:, :K]
    assert torch.equal(sel, ref_sel)
    assert torch.allclose(w.float(), torch.gather(probs, 1, ref_sel).half().float())
    assert (w.float().sum(-1) < 1.001).all() and (w.float().sum(-1) < 0.99).any()   # off: NOT renormalised
    renorm = torch.softmax(torch.gather(logits, 1, ref_sel), -1)  # on (main's routing_std): top-k softmax, sums to 1
    assert torch.allclose(renorm.sum(-1), torch.ones(B))
    assert not torch.allclose(renorm, w.float(), atol=1e-3)
    s = torch.rand(E).half()                                      # per-expert scale applies in both
    cfg.per_expert_scale = s
    _, w2 = ns["_routing_std_nonorm"](cfg, y, {})
    assert torch.allclose(w2.float(), (torch.gather(probs, 1, ref_sel) * s.float()[ref_sel]).half().float())
    _, wa = ns["_routing_std_nonorm"](cfg, y, {"activate_all_experts": True})
    assert wa.shape == (B, E)
