"""Hy4-flash latent MoE block (registry block `latent_moe`, loader/gguf_blocks/mlp.py) vs the llama.cpp oracle (CPU only, no extension).
Maths (llama.cpp build_latent_moe, M:5949; build_moe_ffn with gating func 2): x = ffn_norm(collapsed ffn input) (3072);
  router: p = sigmoid(x @ ffn_gate_inp^T) (512, F32), pick top-16 of p + exp_probs_b, weights = p[pick] / sum * 3.998;
  routed: z = ffn_routed_down x (3072 -> 1536, Q8_0); per expert h = silu(min(gate z, 10)) * clip(up z, +-10) (Q4_K), e = down h (1536);
          ffn_moe_out (the oracle's 1536-wide dump) = sum_k w_k e_k;  y = ffn_routed_up ffn_moe_out (1536 -> 3072, Q8_0)
  shared: ffn_down_shexp(silu(ffn_gate_shexp x) * ffn_up_shexp x) at full width, NOT clamped; ffn = y + shared; then post_ffw_norm (the stream block).
The module path (BlockSparseMLP + ForeignLinear) needs the extension, so this checks the registry (names, config, constructor kwargs through stubs)
and the maths with the GGUF's own tensors: only the routed experts the router picks for the tokens under test are read (by offset, one expert at a time).
The ffn input is rebuilt as the HC child did: s1 = attn site of the oracle's input stream + attn_out-L, collapsed by the ffn site, rms * ffn_norm.
Layer 5 cannot be rebuilt (no l_out-4 / engram dump of its input stream).
Float matmuls on CPU: llama.cpp quantises the activation to Q8_0 (Q8_0 weights) / Q8_K (Q4_K weights); both variants are reported (q8 = emulated).
Bars: cosine (min over tokens) >= 0.9995 on the oracle's own input; rel = max abs / max|target| reported, not gated at 1e-3."""
import os, sys, json
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import test_gguf_hy4_hc as T                      # stubs + registry
from test_gguf_hy4_hc import ORACLE, gg, Cfg, rel, site
from exllamav3.loader.engram_rows import load_dense, q8_0_roundtrip

COS_BAR = 0.9995
pytestmark = pytest.mark.skipif(not (os.path.isdir(ORACLE) and os.path.exists(T.GGUF1)), reason="oracle dumps / Hy4 GGUF absent")
NT_MAX = 8            # prefill calls: first tokens only (a few experts' rows, never the whole stack)
EPS = float(np.float32(9.999999747378752e-06))
_NDIM = {"attn_out": 2, "l_out": 3, "engram_out": 3, "ffn_moe_out": 2}


def q8k_roundtrip(x):
    """ggml quantize_row_q8_K_ref + dequant: per 256 values iscale = -127 / (signed value of max |x|), q = nearest_int (even), d = 1/iscale"""
    b = x.reshape(*x.shape[:-1], -1, 256)
    idx = b.abs().argmax(-1, keepdim=True)
    mx = torch.gather(b, -1, idx)
    iscale = torch.where(mx != 0, -127.0 / mx, torch.zeros_like(mx))
    q = torch.round(b * iscale).clamp(-127, 127)
    d = torch.where(iscale != 0, 1.0 / iscale, torch.zeros_like(iscale))
    return (q * d).reshape(x.shape)


def dq_q4k(raw, N, K):
    """Q4_K (144 B / 256: fp16 d, fp16 dmin, 12 B packed 6-bit scales / mins, 128 B nibbles) -> fp32 [N, K] (ggml dequantize_row_q4_K)"""
    nb = K // 256
    r = torch.from_numpy(np.ascontiguousarray(raw)).reshape(N * nb, 144)
    d = r[:, 0:2].contiguous().view(torch.float16).float().reshape(-1, 1)
    dm = r[:, 2:4].contiguous().view(torch.float16).float().reshape(-1, 1)
    s = r[:, 4:16].int()
    sc = torch.cat([s[:, 0:4] & 63, (s[:, 8:12] & 15) | ((s[:, 0:4] >> 6) << 4)], 1).float()
    mn = torch.cat([s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], 1).float()
    qs = r[:, 16:].int().reshape(-1, 4, 32)
    out = torch.empty(N * nb, 8, 32)
    for j in range(4):
        out[:, 2 * j] = (qs[:, j] & 15).float() * (d * sc[:, 2 * j:2 * j + 1]) - dm * mn[:, 2 * j:2 * j + 1]
        out[:, 2 * j + 1] = (qs[:, j] >> 4).float() * (d * sc[:, 2 * j + 1:2 * j + 2]) - dm * mn[:, 2 * j + 1:2 * j + 2]
    return out.reshape(N, K)


def rmsn(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def ffn_ref(g, L, x, q8, clamp_pre=True, rounding=None):
    """x (nt, 3072) normed ffn input -> dict(moe latent (nt,1536), ffn (nt,3072) pre post-norm, sel (nt,16), w)"""
    qa8 = (lambda t: q8_0_roundtrip(t)) if q8 else (lambda t: t)
    qak = (lambda t: q8k_roundtrip(t)) if q8 else (lambda t: t)
    b = f"blk.{L}."
    gate = g.get_tensor(b + "ffn_gate_inp.weight").float()                 # (512, 3072)
    bias = g.get_tensor(b + "exp_probs_b.bias").float()
    p = torch.sigmoid(x @ gate.T)
    sel = torch.topk(p + bias, 16, dim=-1).indices
    w = torch.gather(p, 1, sel)
    w = w / w.sum(-1, keepdim=True).clamp(min=6.103515625e-5) * 3.997999906539917
    z = qa8(x) @ load_dense(g, b + "ffn_routed_down.weight").T            # (nt, 1536)
    moe = torch.zeros(x.shape[0], 1536)
    for e in sorted(set(sel.flatten().tolist())):
        rows, slot = (sel == e).nonzero(as_tuple=True)
        zi = qak(z[rows])
        W = {n: dq_q4k(g.get_tensor(f"{b}ffn_{n}_exps.e{e}.weight"), 1536, 1536) for n in ("gate", "up", "down")}
        ga, up = zi @ W["gate"].T, zi @ W["up"].T
        ga = torch.minimum(ga, torch.tensor(10.0)) if clamp_pre else ga
        h = torch.nn.functional.silu(ga) * up.clamp(-10, 10)
        o = qak(h) @ W["down"].T
        moe.index_add_(0, rows, o * w[rows, slot].unsqueeze(-1))
    y = qa8(moe) @ load_dense(g, b + "ffn_routed_up.weight").T
    sg, su = qa8(x) @ load_dense(g, b + "ffn_gate_shexp.weight").T, qa8(x) @ load_dense(g, b + "ffn_up_shexp.weight").T
    sh = qa8(torch.nn.functional.silu(sg) * su) @ load_dense(g, b + "ffn_down_shexp.weight").T
    return {"moe": moe, "ffn": y + sh, "sel": sel, "w": w}


def cos(a, b):
    a, b = a.double().reshape(a.shape[0], -1), b.double().reshape(b.shape[0], -1)
    return float(((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))).min())


def call_data(prompt, call):
    d = os.path.join(ORACLE, prompt)
    idx = json.load(open(os.path.join(d, "index.json")))
    ent = {(t["name"], t["layer"], t["occ"]): t for t in idx["tensors"] if t["call"] == call}
    def get(name, layer, occ=0):
        t = ent[(name, layer, occ)]
        a = torch.from_numpy(np.fromfile(os.path.join(d, t["file"]), np.float32).reshape(tuple(reversed(t["shape_ggml"]))).copy())
        return a.unsqueeze(0) if a.dim() < _NDIM.get(name, 99) else a
    return get


def check(prompt, call, layers=(1, 2)):
    g, plan = gg()
    get = call_data(prompt, call)
    rows = []
    for L in layers:
        s0 = (get("engram_out", 1) if L == 1 else get("l_out", L - 1))[:NT_MAX].unsqueeze(0)
        a, f = site(L, "attn"), site(L, "ffn")
        post_g, comb, _ = a.mix(s0, {})
        s1 = a.apply_(s0.clone(), get("attn_out", L)[:NT_MAX].unsqueeze(0), post_g, comb, {})
        coll = f.mix(s1, {})[2][0]
        x = rmsn(coll, g.get_tensor(f"blk.{L}.ffn_norm.weight").float())
        post = f.gates(s1)[1][0]                                                  # (nt, 4)
        pn = g.get_tensor(f"blk.{L}.post_ffw_norm.weight").float()
        fo = ((get("l_out", L)[:NT_MAX] - s1[0]) / post.unsqueeze(-1))[:, 0]      # oracle ffn output after post_ffw_norm, stream 0
        tm = get("ffn_moe_out", L)[:NT_MAX]
        for q8 in (True, False):
            r = ffn_ref(g, L, x, q8)
            ma, rm = rel(r["moe"], tm)
            mb, rf = rel(rmsn(r["ffn"], pn), fo)
            rows.append((f"L{L} {'q8' if q8 else 'f32'}", f"{prompt}#{call} nt={x.shape[0]}", rm, cos(r["moe"], tm), rf, cos(rmsn(r["ffn"], pn), fo),
                         "PASS" if cos(r["moe"], tm) >= COS_BAR and cos(rmsn(r["ffn"], pn), fo) >= COS_BAR else "FAIL"))
    return rows


CASES = [("p1_short_en", 0), ("p1_short_en", 1), ("p1_short_en", 2), ("p4_long_n700", 0), ("p4_long_n700", 2), ("p5_long_n4000", 0), ("p5_long_n4000", 8)]


def test_registry_selects_latent_moe():
    g, plan = gg()
    assert plan.layers[0]["ffn"] == "dense_ffn" and all(plan.layers[i]["ffn"] == "latent_moe" for i in range(1, 42))
    c = plan.configs[("latent_moe", 1)]
    assert (c["latent"], c["num_experts"], c["top_k"], c["gating_func"], c["weights_norm"], c["e_score_bias"], c["clamp"]) == (1536, 512, 16, 2, True, True, 10.0)
    assert abs(c["weights_scale"] - 3.998) < 1e-6 and c["expert_dims"] == (1536, 1536, 512) and c["shared_dims"] == (3072, 1536)
    assert not [x for x in plan.gaps if "latent" in x.lower() or "ffn_" in x]


def test_names_resolve():
    from exllamav3.loader.gguf_blocks.mlp import _latent_names
    g, plan = gg()
    nm = _latent_names(plan.configs[("latent_moe", 1)], 1, "L", "blk.1")
    assert len(nm) == 4 + 3 + 3 * 512
    for k in ("L.mlp.gate.weight", "L.mlp.gate.e_score_correction_bias", "L.mlp.fc1_latent_proj.weight", "L.mlp.fc2_latent_proj.weight",
              "L.mlp.shared_experts.up_proj.weight", "L.mlp.experts.37.down_proj.weight"):
        assert g.has_tensor(nm[k]), k


def test_build_kwargs():
    """the registry build, with module stubs: latent projections, ds3 router with one group, scale 3.998, clamp 10 on the routed experts only"""
    import types
    from exllamav3.loader.gguf_blocks import mlp as M
    seen = {}
    class Cap:
        def __init__(self, **k): self.k = k; seen.setdefault("n", []).append(k)
        def register_submodule(self, m): pass
        gates = ups = downs = []
    ga = types.ModuleType("exllamav3.architecture.gguf_generic"); ga._fl = lambda cfg, key, i, o, **k: (key, i, o)
    pk = sys.modules["exllamav3.modules"]; pk.GatedMLP = pk.BlockSparseMLP = Cap
    sys.modules["exllamav3.architecture"] = types.ModuleType("exllamav3.architecture"); sys.modules["exllamav3.architecture"].__path__ = []
    sys.modules["exllamav3.architecture.gguf_generic"] = ga
    sys.modules["exllamav3.architecture"].gguf_generic = ga
    g, plan = gg()
    cfg = Cfg(); cfg.gguf_plan = plan
    mod = M._latent_build(cfg, plan, 1, "L")
    k = mod.k
    assert k["latent_size"] == 1536 and k["router_type"] == "ds3" and k["n_group"] == k["topk_group"] == 1 and k["norm_topk"] is True
    assert abs(k["routed_scaling_factor"] - 3.998) < 1e-6 and k["num_experts_per_tok"] == 16
    import math
    assert "act_limit" not in k and k["act_limit_up"] == 10.0 and abs(k["act_limit_gate"] - 10.0 / (1.0 + math.exp(-10.0))) < 1e-12   # exact split clamp: silu(c), c
    assert k["latent_in"] == ("L.mlp.fc1_latent_proj", 3072, 1536) and k["latent_out"] == ("L.mlp.fc2_latent_proj", 1536, 3072)
    assert k["gates"][0] == ("L.mlp.experts.0.gate_proj", 1536, 1536)
    sh = seen["n"][0]
    assert sh["hidden_size"] == 3072 and sh["intermediate_size"] == 1536 and "act_limit" not in sh        # shared expert: full width, unclamped


def test_q4k_decoder_matches_numpy_reference():
    from exllamav3.foreign_quant.k_quants import dequant_q4_K_bulk
    g, _ = gg()
    raw = g.get_tensor("blk.1.ffn_gate_exps.e3.weight")
    a = dq_q4k(raw, 1536, 1536)[:4]
    b = torch.from_numpy(dequant_q4_K_bulk(raw.numpy().reshape(1536, -1)[:4].reshape(-1, 144)).astype(np.float32)).reshape(4, 1536)
    assert (a - b).abs().max() <= 2e-3 * a.abs().max()      # the tree decoder returns fp16


@pytest.mark.parametrize("prompt,call", CASES)
def test_ffn_vs_oracle(prompt, call):
    for row in check(prompt, call):
        assert row[-1] == "PASS", row


if __name__ == "__main__":
    for p, c in CASES:
        for r in check(p, c):
            print(f"{r[0]} | {r[1]} | ffn_moe_out rel {r[2]:.2e} cos {r[3]:.6f} | ffn_out(l_out-derived) rel {r[4]:.2e} cos {r[5]:.6f} | {r[6]}", flush=True)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- split clamp (Hy4 swiglu_clamp_exp): silu(min(g, c)) * clip(u, -c, c) == min(silu(g), silu(c)) * clip(u, -c, c) ------------------------
# These run in a clean interpreter: the module tests above replace exllamav3.modules with stubs in this process.
_SPLIT_SCRIPT = r'''
import json, math, torch, torch.nn.functional as F
from exllamav3.ext_fallbacks import silu_mul, silu_mul_split
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
out = {}
torch.manual_seed(0)
def data(dt, c):
    g = (torch.rand(2048, 64) * 40 - 20).to(dt); u = (torch.rand(2048, 64) * 40 - 20).to(dt)
    g[:16] = (torch.rand(16, 64) * 8 + 10).to(dt)                      # rows with g > 10
    return g, u
mx, nbig, mxdef, mxdef_dtype = {}, 0, 0.0, []
for dt in (torch.float32, torch.float16):
    for c in (10.0, 2.0, 6.0, 7.5, 15.0, 0.75):
        g, u = data(dt, c); nbig = int((g > 10).any(1).sum())
        z = torch.empty(g.shape, dtype = torch.half)
        silu_mul_split(g, u, z, c / (1.0 + math.exp(-c)), c)
        ref = F.silu(torch.minimum(g, torch.tensor(c, dtype = dt))) * u.clamp(-c, c)
        if dt == torch.float32: ref = ref.clamp(-65504.0, 65504.0)
        mx[f"{str(dt)[6:]}_{c}"] = float((z.float() - ref.to(torch.half).float()).abs().max())
        # (b) default path against the previous formula, bit for bit
        for lim in (0.0, c):
            z0 = torch.empty(g.shape, dtype = torch.half); silu_mul(g, u, z0, lim)
            xa = F.silu(g)
            if lim != 0.0: xa = xa.clamp(max = lim); uu = u.clamp(min = -lim, max = lim)
            else: uu = u
            r0 = xa * uu
            if dt == torch.float32: r0 = r0.clamp(-65504.0, 65504.0)
            mxdef = max(mxdef, float((z0.float() - r0.to(torch.half).float()).abs().max()))
out["mx"], out["nbig"], out["mxdef"] = mx, nbig, mxdef
# (c) decline, no GPU: stubbed device check
m = BlockSparseMLP.__new__(BlockSparseMLP)
m._split_lim, m.split_declined = True, []
m.is_quantized, m.cpu_offload, m.cpu_split_first = True, False, None
out["decline_true"] = m._split_decline("hip_grouped")
try:
    m.forward(torch.zeros(1, 4), {})
    out["raised"] = None
except NotImplementedError as e:
    out["raised"] = str(e)
out["declined"] = m.split_declined
n = BlockSparseMLP.__new__(BlockSparseMLP); n._split_lim = False; n.split_declined = []
out["decline_default"] = n._split_decline("hip_grouped")
print("RESULT" + json.dumps(out))
'''


def _split_run():
    import subprocess
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", _SPLIT_SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    if p.returncode != 0 and "libcudart" in p.stderr:
        pytest.skip("exllamav3 extension not importable here")
    assert p.returncode == 0, p.stderr[-800:]
    return json.loads(p.stdout.split("RESULT", 1)[1])


@pytest.fixture(scope="module")
def split_res():
    return _split_run()


def test_split_clamp_exact_vs_reference(split_res):
    """(a) torch split path == silu(min(g, c)) * clip(u, -c, c), fp32 and fp16 inputs, g and u in [-20, 20], rows with g > 10 present"""
    assert split_res["nbig"] >= 16
    assert max(split_res["mx"].values()) == 0.0, split_res["mx"]


def test_split_clamp_default_path_bit_identical(split_res):
    """(b) no split limits: ext_fallbacks.silu_mul (single act_limit) is bit-identical to the previous formula"""
    assert split_res["mxdef"] == 0.0


def test_split_clamp_declines_gpu_routes(split_res):
    """(c) split limits record the decline on a GPU route and raise the clear error where no torch fallback exists; default never declines"""
    assert split_res["decline_true"] is True and split_res["decline_default"] is False
    assert split_res["raised"] == "split clamp limits need the two-limit kernel (Hy4 engine path)"
    assert "exl3_or_cpu_host" in split_res["declined"]


def test_split_clamp_every_kernel_route_guarded():
    """every kernel `elif (` of the routed dispatch chain and every foreign grouped / prefill route checks the split flag"""
    src = open(os.path.join(ROOT, "exllamav3/modules/block_sparse_mlp.py")).read()
    a, b = src.index("# gfx11.5 batch-1 fused decode (exl3_dec_moe)"), src.index("# Torch/C++/fused path")
    assert src[a:b].count("elif (\n            not self._split_lim and\n") == 7
    assert src.count("not split_no_foreign") == 4
