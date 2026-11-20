"""Hy4-flash hyper-connection sites + head vs the llama.cpp oracle dumps (CPU only, no GPU, no extension).
The modules under test are imported by path under stub packages (the extension needs libcudart); Module / RMSNorm are stubs,
the rms is done inline in HyperConnectionPP.gates, so only the maths is exercised. Skips when the GGUF / oracle dumps are absent.

  pre weights   hc_pre-L_o0 (attn site) / hc_pre-L_o1 (ffn site), from the input stream
  post mixing   s1 = s0 + post * attn_out (identity comb); the ffn-site pre weights hc_pre-L_o1 are computed on s1, so they check it
  ffn site mix  l_out - s1 = post_ffn * ffn_out for all 4 streams with ONE ffn_out (least squares over the streams, residual)
Bar: max abs / max |target| <= 1e-4 (fp32 vs fp32, ggml vs torch reduction order; the existing GGUF tests vs llama.cpp are bit-exact
dequant checks, which do not apply to float maths)."""
import json, os, sys, types
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.expanduser("/path/to/oracle_out")
GGUF1 = "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf"
BAR = 1e-4
pytestmark = pytest.mark.skipif(not (os.path.isdir(ORACLE) and os.path.exists(GGUF1)), reason="oracle dumps / Hy4 GGUF absent")


def _mod(n, **kw):
    m = types.ModuleType(n)
    for k, v in kw.items():
        setattr(m, k, v)
    sys.modules[n] = m
    return m


def _stubs():
    import importlib.util
    class _Ext:
        def __getattr__(self, n):
            raise AttributeError(n)
    class RMSNorm:
        def __init__(self, *a, **k): pass
    class Module:
        def __init__(self, config=None, key="", qmap=None, **k): self.config, self.key = config, key
        def register_submodule(self, m): pass
        def load(self, *a, **k): pass
        def unload(self): pass
    for n, s in (("exllamav3", ""), ("exllamav3.modules", "modules"), ("exllamav3.loader", "loader")):
        _mod(n).__path__ = [os.path.join(ROOT, "exllamav3", s)]
    _mod("exllamav3.ext", exllamav3_ext=_Ext())
    _mod("exllamav3.model").__path__ = []
    _mod("exllamav3.model.config", Config=object)
    _mod("exllamav3.util").__path__ = []
    _mod("exllamav3.util.tensor", g_tensor_cache=None)
    _mod("exllamav3.modules.quant").__path__ = []
    _mod("exllamav3.modules.quant.exl3", PAD_MIN_ROWS=1, PAD_ELEMS=1)
    _mod("exllamav3.modules.rmsnorm", RMSNorm=RMSNorm)
    _mod("exllamav3.modules.module", Module=Module)
    for name, rel in (("exllamav3.loader.gguf_collection", "loader/gguf_collection.py"),
                      ("exllamav3.loader.gguf_registry", "loader/gguf_registry.py")):
        sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "exllamav3", rel))
        m = importlib.util.module_from_spec(sp); m.__package__ = "exllamav3.loader"; sys.modules[name] = m; sp.loader.exec_module(m)
    sys.modules["exllamav3.loader.gguf_registry"]._load_blocks()


_stubs()
from exllamav3.modules.hyperconnections import HyperConnectionPP, HyperHead   # noqa: E402
from exllamav3.loader.gguf_collection import GGUFCollection                   # noqa: E402
from exllamav3.loader.gguf_registry import build_plan                         # noqa: E402
from exllamav3.loader.gguf_blocks.hyperconnections import build_site, _head_build, _site_names, _head_names  # noqa: E402

_G = {}


def gg():
    if not _G:
        g = GGUFCollection(GGUF1)
        _G["g"], _G["plan"] = g, build_plan(g)
    return _G["g"], _G["plan"]


class Cfg:
    hidden_size, rms_norm_eps = 3072, float(np.float32(9.999999747378752e-06))


def site(layer, tag):
    g, plan = gg()
    cfg = Cfg(); cfg.gguf_plan = plan
    m = build_site(cfg, plan, layer, f"model.layers.{layer}", tag)
    for t in ("fn", "base", "scale"):
        w = g.get_tensor(f"blk.{layer}.hc_{tag}_{t}.weight").float()
        setattr(m, t, w.reshape(-1, m.hidden_size * m.hc_mult) if t == "fn" else w.reshape(-1))
    return m


def emb_rows(ids):
    """token_embd.weight is Q8_0: 34-byte blocks (fp16 d + 32 int8), 96 blocks per 3072-wide row; read by offset, row by row"""
    g, _ = gg()
    ty, dims, off = g.infos["token_embd.weight"]
    assert ty == 8 and dims[0] == 3072
    rb = dims[0] // 32 * 34
    out = np.empty((len(ids), dims[0]), np.float32)
    for k, t in enumerate(ids):
        raw = g._pread(off + int(t) * rb, rb).reshape(-1, 34)
        d = raw[:, :2].copy().view(np.float16).astype(np.float32)          # (96, 1)
        q = raw[:, 2:].copy().view(np.int8).astype(np.float32)             # (96, 32)
        out[k] = (q * d).reshape(-1)
    return torch.from_numpy(out)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max()), float((a - b).abs().max() / b.abs().max())


def load_call(prompt, call, layers):
    d = os.path.join(ORACLE, prompt)
    idx = json.load(open(os.path.join(d, "index.json")))
    ent = {(t["name"], t["layer"], t["occ"]): t for t in idx["tensors"] if t["call"] == call}
    def get(name, layer, occ=0):
        t = ent[(name, layer, occ)]
        return torch.from_numpy(np.fromfile(os.path.join(d, t["file"]), np.float32).reshape(tuple(reversed(t["shape_ggml"]))).copy())
    pos = ent[("hc_pre", 0, 0)]["positions"]
    return idx, get, pos


def check(prompt, call=0, layers=(0, 1, 2)):
    """-> list of (tensor, target, prompt, max abs, rel, PASS/FAIL)"""
    idx, get, pos = load_call(prompt, call, layers)
    ids = [idx["prompt_ids"][p] for p in pos]
    rows, nt = [], len(pos)
    def rec(name, target, a, b):
        ma, r = rel(a, b)
        rows.append((name, target, f"{prompt}#{call}", ma, r, "PASS" if r <= BAR else "FAIL"))
    for L in layers:
        if L == 0:
            s0 = emb_rows(ids).unsqueeze(1).repeat(1, 4, 1)
        elif L == 1:
            s0 = get("engram_out", 1)
        elif L == 2:
            s0 = get("l_out", 1)
        s0 = s0.unsqueeze(0)
        a, f = site(L, "attn"), site(L, "ffn")
        pre, post = a.gates(s0)
        rec(f"L{L} attn site pre", f"hc_pre-{L}_o0", pre[0], get("hc_pre", L, 0))
        post_g, comb, coll = a.mix(s0, {})
        y = get("attn_out", L)
        s1 = a.apply_(s0.clone(), y.unsqueeze(0), post_g, comb, {})
        pre2, post2 = f.gates(s1)
        rec(f"L{L} post mix -> ffn site pre", f"hc_pre-{L}_o1", pre2[0], get("hc_pre", L, 1))
        d = get("l_out", L) - s1[0]                                   # (nt, 4, D) = post2_i * ffn_out
        p2 = post2[0].unsqueeze(-1)                                   # (nt, 4, 1)
        fo = (p2 * d).sum(1, keepdim=True) / (p2 * p2).sum(1, keepdim=True)
        rec(f"L{L} ffn post mix (l_out = s1 + post*ffn_out, 1 ffn_out for 4 streams)", f"l_out-{L}", s1[0] + p2 * fo, get("l_out", L))
    return rows


def test_registry_selects_identity_and_head():
    g, plan = gg()
    c = plan.configs[("hyper_connection", 0)]
    assert c["mode"] == "identity" and c["sinkhorn_iterations"] is None and c["count"] == 4 and c["magnitude"] == 2.0
    assert plan.glob["head_mod"] == ["hyper_head"] and plan.layers[0]["stream"] == "hyper_connection"
    assert plan.layers[42]["stream"] != "hyper_connection"          # the MTP block has no HC tensors
    assert _site_names({}, 3, "L", "G")["L.hc_ffn_fn"] == "G.hc_ffn_fn.weight"
    assert g.infos["output_hc_fn.weight"][1] == (12288, 4) and g.infos["blk.7.hc_attn_fn.weight"][1] == (12288, 8)


@pytest.mark.parametrize("prompt,call", [("p1_short_en", 0), ("p4_long_n700", 0), ("p4_long_n700", 1), ("p5_long_n4000", 0), ("p5_long_n4000", 7)])
def test_hc_sites_vs_oracle(prompt, call):
    for row in check(prompt, call):
        assert row[5] == "PASS", row


def test_head_matches_formula():
    """the oracle dumps no stream into the head (l_out of layer 41 is not dumped), so only the formula is checked,
    against numpy on the real output_hc_* tensors"""
    g, plan = gg()
    cfg = Cfg(); cfg.gguf_plan = plan
    h = _head_build(cfg, "model")[0]
    h.fn, h.base, h.scale = (g.get_tensor(f"output_hc_{t}.weight").float().reshape(-1, 12288) if t == "fn"
                             else g.get_tensor(f"output_hc_{t}.weight").float().reshape(-1) for t in ("fn", "base", "scale"))
    h.norm = types.SimpleNamespace(forward=lambda x, params: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + h.rms_eps))
    x = torch.randn(1, 3, 4, 3072)
    out = h.forward(x, {})
    xn = x.double().numpy(); fl = xn.reshape(1, 3, -1)
    fl = fl / np.sqrt((fl ** 2).mean(-1, keepdims=True) + h.rms_eps)
    m = fl @ h.fn.double().numpy().T
    pre = 1 / (1 + np.exp(-(m * h.scale.double().numpy() + h.base.double().numpy()))) + h.hc_eps
    ref = (pre[..., None] * xn).sum(2)
    assert np.abs(out.double().numpy() - ref).max() / np.abs(ref).max() < 1e-5
    assert _head_names({}, None, None, None)["model.hc_head_fn"] == "output_hc_fn.weight"


if __name__ == "__main__":
    for p, c in [("p1_short_en", 0), ("p4_long_n700", 0), ("p5_long_n4000", 0), ("p5_long_n4000", 7)]:
        for r in check(p, c):
            print(f"{r[0]} | {r[1]} | {r[2]} | maxabs {r[3]:.3e} | rel {r[4]:.3e} | bar {BAR} | {r[5]}")
