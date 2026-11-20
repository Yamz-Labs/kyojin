"""Hy4-flash hyper-connection stream block (TransformerBlock with attn_hc / mlp_hc + post norms) vs the llama.cpp oracle (CPU only).
The REAL TransformerBlock.forward and the REAL registry build (loader/gguf_blocks/hyperconnections._stream_build) run; the extension is
not loadable here, so Module / RMSNorm (same maths as modules/rmsnorm.forward_torch) and the sublayers are stubs:
  attention : returns attn_out-L / post_attention_norm.weight (scaled so the eps is exact, `unnorm`), so the real post-norm gives back the oracle's attn_out-L
              (llama.cpp dumps attn_out AFTER the post norm, M:5943)
  ffn       : ffn_moe_out-L is only the 1536-wide latent (no up-projection / shared expert), so the ffn output is derived from the
              oracle: (l_out-L - s1) / post_gate on stream 0 (s1 = the stream after the attn site, from the real block), divided by
              post_ffw_norm.weight; the 4 streams must agree (identity comb, no cross-stream mix) -> `spread` (bar 1e-2: the division by a small post gate amplifies the oracle's rounding). The ffn side is
              therefore partly circular: it checks the post-norm + post-mix wiring, not the MoE
Input stream: engram_out-1 (layer 1: the engram is applied before the layer, not built) / l_out-1 (layer 2). Target: l_out-L.
Bar: max abs / max|target| <= 1e-3. Skips when the GGUF / oracle dumps are absent."""
import os, sys, types, importlib.util
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import test_gguf_hy4_hc as T                      # installs the stub packages, loads the registry
from test_gguf_hy4_hc import ROOT, ORACLE, gg, load_call, Cfg, rel

BAR = 1e-3
pytestmark = pytest.mark.skipif(not (os.path.isdir(ORACLE) and os.path.exists(T.GGUF1)), reason="oracle dumps / Hy4 GGUF absent")


class Module:
    def __init__(self, config=None, key="", qmap=None, **k): self.config, self.key = config, key
    def register_submodule(self, m): pass


class WRMS(Module):
    def __init__(self, config=None, key="", rms_norm_eps=1e-6, out_dtype=torch.half, **k):
        super().__init__(config, key); self.rms_norm_eps, self.out_dtype, self.weight = rms_norm_eps, out_dtype, None
    def forward(self, x, params, out_dtype=None, residual=None, residual_in=None):
        assert residual is None and residual_in is None
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps) * self.weight.float()
        return h.to(out_dtype or self.out_dtype)


class Sub:
    num_slices = 1
    def __init__(self): self.out = None
    def forward(self, y, params): return self.out.clone()


def _block_cls():
    pk = sys.modules["exllamav3.modules"]
    for n in ("LayerNorm", "Attention", "GatedDeltaNet", "GatedMLP", "MLP", "BlockSparseMLP", "Linear"):
        setattr(pk, n, type(n, (), {}))
    pk.Module, pk.RMSNorm = Module, WRMS
    sys.modules["exllamav3.util.tensor"].to2 = lambda x, *a: x
    sys.modules["exllamav3.util"].profile_opt = types.SimpleNamespace()
    for n, kw in (("ablit_runtime", {}), ("moe_fused", {"enabled_by_env": lambda: False}),
                  ("block_graph", {"maybe_graph_forward": lambda *a, **k: None})):
        m = types.ModuleType("exllamav3.modules." + n); m.__dict__.update(kw)
        sys.modules[m.__name__] = m; setattr(pk, n, m)
    sp = importlib.util.spec_from_file_location("exllamav3.modules.transformer", os.path.join(ROOT, "exllamav3/modules/transformer.py"))
    m = importlib.util.module_from_spec(sp); m.__package__ = "exllamav3.modules"; sys.modules[sp.name] = m; sp.loader.exec_module(m)
    pk.TransformerBlock = m.TransformerBlock
    return m.TransformerBlock


TB = _block_cls()
from exllamav3.loader.gguf_blocks.hyperconnections import _stream_build, _stream_names   # noqa: E402


def unnorm(o, w, eps):
    """y with rmsnorm(y) * w == o exactly: u = o / w has mean(u^2) = q < 1 (the eps shrinks it), y = c u with c^2 = eps / (1 - q)"""
    u = o / w
    q = u.double().pow(2).mean(-1, keepdim=True)
    assert float(q.max()) < 1.0
    return (u.double() * torch.sqrt(eps / (1 - q))).float()


def block(L):
    g, plan = gg()
    cfg = Cfg(); cfg.gguf_plan = plan
    attn, mlp = Sub(), Sub()
    b = _stream_build(cfg, plan, L, f"model.layers.{L}", attn, mlp)
    for site, tag in ((b.attn_hc, "attn"), (b.mlp_hc, "ffn")):
        for t in ("fn", "base", "scale"):
            w = g.get_tensor(f"blk.{L}.hc_{tag}_{t}.weight").float()
            setattr(site, t, w.reshape(-1, site.hidden_size * site.hc_mult) if t == "fn" else w.reshape(-1))
    for n, gn in ((b.attn_norm, "attn_norm"), (b.mlp_norm, "ffn_norm"), (b.attn_post_norm, "post_attention_norm"), (b.mlp_post_norm, "post_ffw_norm")):
        n.weight = g.get_tensor(f"blk.{L}.{gn}.weight").float().reshape(-1)
    return b, attn, mlp


def test_names_and_order():
    g, plan = gg()
    nm = _stream_names(plan.configs[("hyper_connection", 1)], 1, "L", "G")
    assert nm["L.attn_post_norm.weight"] == "G.post_attention_norm.weight" and nm["L.mlp_post_norm.weight"] == "G.post_ffw_norm.weight"
    assert nm["L.input_layernorm.weight"] == "G.attn_norm.weight" and nm["L.post_attention_layernorm.weight"] == "G.ffn_norm.weight"
    assert nm["L.hc_ffn_scale"] == "G.hc_ffn_scale.weight"
    b, _, _ = block(1)
    assert b.attn_post_norm is not None and b.mlp_post_norm is not None and b.attn_hc is not None
    # one post norm without the other, or a scalar, is still refused
    with pytest.raises(AssertionError):
        TB(config=None, key="k", layer_idx=0, attn_hc=b.attn_hc, mlp_hc=b.mlp_hc, attn_post_norm=b.attn_post_norm)
    with pytest.raises(AssertionError):
        TB(config=None, key="k", layer_idx=0, attn_hc=b.attn_hc, mlp_hc=b.mlp_hc, key_layer_scalar="s")


def check(prompt, call=0, layers=(1, 2)):
    idx, get, pos = load_call(prompt, call, layers)
    rows = []
    rows_extra = []
    for L in layers:
        b, attn, mlp = block(L)
        s0 = get("engram_out", 1) if L == 1 else get("l_out", L - 1)
        attn.out = unnorm(get("attn_out", L), b.attn_post_norm.weight, b.attn_post_norm.rms_norm_eps).unsqueeze(0)
        tgt = get("l_out", L)
        mlp.out = torch.zeros(1, tgt.shape[0], b.mlp_post_norm.weight.numel())
        s1 = b.forward(s0.unsqueeze(0).clone(), {})            # attn site only (post norm of 0 is 0)
        post = b.mlp_hc.gates(s1)[1][0]                        # (T, 4)
        fo = (tgt - s1[0]) / post.unsqueeze(-1)                # (T, 4, D) ffn_out per stream
        spread = float((fo - fo[:, :1]).abs().max() / fo[:, 0].abs().max())
        mlp.out = unnorm(fo[:, 0], b.mlp_post_norm.weight, b.mlp_post_norm.rms_norm_eps).unsqueeze(0)
        out = b.forward(s0.unsqueeze(0).clone(), {})
        ma, r = rel(out[0], tgt)
        rows_extra.append((L, spread))
        rows.append((f"l_out-{L} (stream in {'engram_out-1' if L == 1 else 'l_out-1'})", f"l_out-{L}", f"{prompt}#{call}", ma, r, "PASS" if r <= BAR and spread <= 1e-2 else "FAIL", spread))
    return rows


@pytest.mark.parametrize("prompt,call", [("p1_short_en", 0), ("p4_long_n700", 0), ("p5_long_n4000", 0)])
def test_layer_vs_oracle(prompt, call):
    for row in check(prompt, call):
        assert row[5] == "PASS", row


if __name__ == "__main__":
    for p, c in [("p1_short_en", 0), ("p4_long_n700", 0), ("p5_long_n4000", 0)]:
        for r in check(p, c):
            print(f"{r[0]} | {r[1]} | {r[2]} | maxabs {r[3]:.3e} | rel {r[4]:.3e} | bar {BAR} | {r[5]} | ffn stream spread {r[6]:.2e}")
