"""GLM-5.3-Flash fast paths, step 2: mpw_gemm (grouped WMMA MoE prefill) and the MLA decode wiring.

Synthetic weights, no model. Two items, one section each:

(a) `exl3_moe_prefill_wmma` (quant/exl3_moe_prefill.cu, the mpw_gemm kernel the grouped prefill
    route actually uses) gains act_limit (clamped SwiGLU) and the mcg codebook, and its shape
    gates are checked against GLM's expert shapes. Reference = the non-fused per-expert route:
    ext.reconstruct + ext.hgemm with the same suh/svh Hadamard stages, ext.silu_mul for the
    activation, fp32 weighted combine over top-k.

(b) `mla_attn.py` batch-1 decode: the MLA projections (q_a/q_b, kv_a, o, the DSA indexer's
    wq_b) run the exl3_dec GEMV kernels. Reference = the same reconstruct + hgemm path.

Same construction as tests/test_hip_dec_glm_gates.py: random trellis *states* packed with
ext.pack_trellis (the ROCm build has no quantizer), every state a valid index
for both codebooks.

Run: PYTHONPATH=<repo> python tests/run_glm_dec2.py
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.environ.get("EXL3_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

DEV = "cuda:0"
ACT_LIMIT = 10.0        # GLM-5.3-Flash swiglu_limit

# GLM-5.3-Flash expert shape, from the published config (zai-org/GLM-5.3-Flash
# config.json -> text_config): hidden_size 4096, moe_intermediate_size 2048,
# n_routed_experts 288, num_experts_per_tok 8, n_shared_experts 1, swiglu_limit 10.0.
# The Qwen3.8-Flash-Next shape the route was written for runs alongside as the control.
GLM_H, GLM_I, GLM_TOPK, GLM_EXPERTS = 4096, 2048, 8, 288
QWEN_H, QWEN_I, QWEN_TOPK = 2560, 768, 10
EXPERTS = 16            # expert count for the fused-vs-reference cases (288 is covered separately)
ROWS = 32

_MPW_READY = "act_limit" in (getattr(ext.exl3_moe_prefill_wmma, "__doc__", "") or "")
_GEMV_READY = "mcg" in (getattr(ext.exl3_dec_gemv, "__doc__", "") or "")


def _require(*names):
    if not (torch.version.hip and torch.cuda.is_available()):
        pytest.skip("ROCm build / device not available")
    for n in names:
        assert hasattr(ext, n), f"missing ext.{n}"
    if not ext.exl3_gemv_supported(torch.cuda.current_device()):
        pytest.skip("exl3 kernels not supported on this device")


def _ptrs(tensors):
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.long, device=DEV)


def _trellis(k, n, K, seed, mcg):
    """Random packed trellis + suh/svh sign vectors for one projection."""
    kb = 16 * K
    gen = torch.Generator(device=DEV).manual_seed(seed)
    states = torch.randint(0, 1 << K, (k // 16, n // 16, 256), generator=gen,
                           device=DEV, dtype=torch.int16)
    packed = torch.zeros((k // 16, n // 16, kb), dtype=torch.int16, device=DEV)
    ext.pack_trellis(packed, states, K)
    g = torch.Generator(device=DEV).manual_seed(seed * 7 + 1)
    suh = (torch.randint(0, 2, (k,), generator=g, device=DEV) * 2 - 1).half()
    svh = (torch.randint(0, 2, (n,), generator=g, device=DEV) * 2 - 1).half()
    return packed, suh, svh


def _ref_linear(x, trellis, suh, svh, K, mcg):
    """reconstruct + Hadamard + hgemm: the reference a fused GEMV/GEMM has to match."""
    k, n, rows = suh.numel(), svh.numel(), x.shape[0]
    xh = torch.empty((rows, k), dtype=torch.half, device=DEV)
    y = torch.empty((rows, n), dtype=torch.half, device=DEV)
    yh = torch.empty((rows, n), dtype=torch.half, device=DEV)
    w = torch.empty((k, n), dtype=torch.half, device=DEV)
    ext.had_r_128(x.contiguous(), xh, suh, None, 1.0)
    ext.reconstruct(w, trellis, K, mcg, not mcg)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, yh, None, svh, 1.0)
    return yh


def _rel(a, b):
    a, b = a.float(), b.float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


# ------------------------------------------------------------------------------------------------
# (a) mpw_gemm: act_limit + mcg + GLM expert shapes

def _mpw_projection(k, n, K, seed, mcg, experts):
    proj = []
    for e in range(experts):
        proj.append(_trellis(k, n, K, seed + 97 * e, mcg))
    return proj


# The reference's down projection runs on the activation scaled by 2^-DOWN_SHIFT and is rescaled
# in fp32. ext.hgemm / had_r_128 store fp16, and one unclamped expert's down output reaches
# 1.7e5 at GLM's shape with random weights, past fp16's 65504 -- while the fused
# kernel accumulates down in fp32 (down_out) and does not overflow. Hadamard, matmul and the
# power-of-two scale are all linear, so this changes only the reference's rounding of the
# smallest activations (subnormal below 2^-14 * 2^6), far under the 5e-3 tolerance.
DOWN_SHIFT = 6


def _mpw_ref(x, proj, K, sel, w, act_limit, mcg, alias=False):
    """Per-row, per-expert reference: reconstruct+hadamard+matmul, silu_mul, fp32 combine."""
    hidden = x.shape[1]
    interm = proj["down"][0][1].numel()          # down's suh length == intermediate
    topk = sel.shape[1]
    out = torch.zeros((x.shape[0], hidden), dtype=torch.float, device=DEV)
    a = torch.empty((1, interm), dtype=torch.half, device=DEV)
    for r in range(x.shape[0]):
        xr = x[r:r + 1].half()
        for k in range(topk):
            e = 0 if alias else int(sel[r, k])
            gt, gs, gv = proj["gate"][e]
            ut, us, uv = proj["up"][e]
            dt, ds, dv = proj["down"][e]
            g = _ref_linear(xr, gt, gs, gv, K, mcg)
            u = _ref_linear(xr, ut, us, uv, K, mcg)
            ext.silu_mul(g, u, a, act_limit)
            d = _ref_linear(a * (2.0 ** -DOWN_SHIFT), dt, ds, dv, K, mcg)
            out[r] += float(w[r, k]) * (2.0 ** DOWN_SHIFT) * d[0].float()
    return out


def _mpw_run(x, proj, K, sel, w, act_limit, mcg, hidden, interm, experts, alias=False):
    rows, topk = sel.shape
    A = rows * topk
    gen = torch.Generator(device=DEV).manual_seed(4)
    order = sel.reshape(-1).argsort(stable=True)
    count = torch.zeros(experts + 1, dtype=torch.long, device=DEV)
    count.scatter_add_(0, sel.reshape(-1),
                       torch.ones(A, dtype=torch.long, device=DEV))
    gu_had = torch.empty((2 * A, hidden), dtype=torch.half, device=DEV)
    gu_out = torch.empty((2 * A, interm), dtype=torch.half, device=DEV)
    down_out = torch.empty((A, hidden), dtype=torch.float, device=DEV)
    output = torch.empty((rows, hidden), dtype=torch.float, device=DEV)
    slots = (A + 127) // 128 + experts
    offsets = torch.empty(experts + 1, dtype=torch.long, device=DEV)
    inverse = torch.empty(A, dtype=torch.long, device=DEV)
    tiles = torch.empty(slots, dtype=torch.int32, device=DEV)
    tile_count = torch.empty(1, dtype=torch.int32, device=DEV)

    def _names(p):
        if alias:
            p = p[:1]
            t0, s0, v0 = p[0]
            return (_ptrs([t0] * experts), _ptrs([s0] * experts), _ptrs([v0] * experts))
        return [_ptrs([t for t, _, _ in p]), _ptrs([s for _, s, _ in p]), _ptrs([v for _, _, v in p])]

    gt, gs, gv = _names(proj["gate"])
    ut, us, uv = _names(proj["up"])
    dt, ds, dv = _names(proj["down"])
    ext.exl3_moe_prefill_wmma(
        x.contiguous(), output, sel.contiguous(), w.contiguous(), order.contiguous(), count,
        gt, gs, gv, ut, us, uv, dt, ds, dv,
        float(K), float(K), gu_had, gu_out, down_out,
        offsets, inverse, tiles, tile_count, act_limit, mcg,
    )
    torch.cuda.synchronize()
    return output


@pytest.mark.parametrize("shape", ["glm", "qwen"])
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("mcg", [False, True])
@pytest.mark.parametrize("act_limit", [0.0, ACT_LIMIT])
def test_mpw_prefill_matches_reference(shape, K, mcg, act_limit):
    """Fused grouped prefill vs the per-expert route, GLM and Qwen expert shapes."""
    _require("exl3_moe_prefill_wmma", "reconstruct", "hgemm", "had_r_128", "silu_mul", "pack_trellis")
    if not _MPW_READY:
        pytest.skip("this build's exl3_moe_prefill_wmma has no act_limit/mcg support")
    if shape == "glm":
        hidden, interm, experts, topk = GLM_H, GLM_I, EXPERTS, GLM_TOPK
    else:
        hidden, interm, experts, topk = QWEN_H, QWEN_I, EXPERTS, QWEN_TOPK
    rows = ROWS
    seed = 1000 + K * 10 + (3 if mcg else 0)

    proj = {
        "gate": _mpw_projection(hidden, interm, K, seed + 1, mcg, experts),
        "up":   _mpw_projection(hidden, interm, K, seed + 2, mcg, experts),
        "down": _mpw_projection(interm, hidden, K, seed + 3, mcg, experts),
    }
    g = torch.Generator(device=DEV).manual_seed(seed)
    # Unclamped runs at reduced scale: both routes hold silu(g)*u in fp16 (the kernel's gu_out,
    # the reference's silu_mul output), and at full scale random mcg weights push it past 65504.
    # The clamped runs keep full scale so the clamp is
    # exercised. The down projection's own overflow is handled in _mpw_ref (DOWN_SHIFT).
    scale = 1.0 if act_limit else 0.35
    x = (torch.randn(rows, hidden, generator=g, device=DEV) * scale).half()
    sel = torch.stack([torch.randperm(experts, generator=g, device=DEV)[:topk]
                       for _ in range(rows)]).to(torch.long)
    w = torch.rand(rows, topk, generator=g, device=DEV)
    w = (w / w.sum(dim=-1, keepdim=True))

    ref = _mpw_ref(x, proj, K, sel, w, act_limit, mcg)
    got = _mpw_run(x, proj, K, sel, w, act_limit, mcg, hidden, interm, experts)
    err = _rel(got, ref)
    assert err < 5e-3, f"{shape} K={K} mcg={mcg} L={act_limit}: rel err {err:.2e}"

    # Determinism: the two calls must be bit-identical
    got2 = _mpw_run(x, proj, K, sel, w, act_limit, mcg, hidden, interm, experts)
    assert torch.equal(got, got2)


@pytest.mark.parametrize("K", [3])
@pytest.mark.parametrize("mcg", [False, True])
def test_mpw_gates_are_discriminating(K, mcg):
    """A kernel ignoring act_limit or the codebook lands at O(1) relative, not 1e-3."""
    _require("exl3_moe_prefill_wmma", "reconstruct", "hgemm", "had_r_128", "silu_mul", "pack_trellis")
    if not _MPW_READY:
        pytest.skip("this build's exl3_moe_prefill_wmma has no act_limit/mcg support")
    hidden, interm, experts, topk = GLM_H, GLM_I, EXPERTS, GLM_TOPK
    rows = 16
    proj = {
        "gate": _mpw_projection(hidden, interm, K, 77, mcg, experts),
        "up":   _mpw_projection(hidden, interm, K, 78, mcg, experts),
        "down": _mpw_projection(interm, hidden, K, 79, mcg, experts),
    }
    g = torch.Generator(device=DEV).manual_seed(555)
    # Scale 0.5: at 1.0 the unclamped mcg run's silu(g)*u reaches 5.7e4 and overflows the fp16
    # activation buffer (inf -> NaN in the down GEMM; the per-expert route overflows the same way),
    # so "open" is NaN and the comparison is meaningless. At 0.5 both runs are finite and the clamp
    # still moves the output by ~0.95 relative.
    x = (torch.randn(rows, hidden, generator=g, device=DEV) * 0.5).half()
    sel = torch.stack([torch.randperm(experts, generator=g, device=DEV)[:topk]
                       for _ in range(rows)]).to(torch.long)
    w = torch.rand(rows, topk, generator=g, device=DEV)
    w = w / w.sum(dim=-1, keepdim=True)

    closed = _mpw_run(x, proj, K, sel, w, ACT_LIMIT, mcg, hidden, interm, experts)
    open_ = _mpw_run(x, proj, K, sel, w, 0.0, mcg, hidden, interm, experts)
    assert torch.isfinite(closed).all() and torch.isfinite(open_).all(), "fp16 activation overflow"
    clamp_effect = float((closed - open_).abs().max() / open_.abs().max().clamp_min(1e-6))
    assert clamp_effect > 0.05, f"clamp does nothing: {clamp_effect:.3e}"

    other = _mpw_run(x, proj, K, sel, w, ACT_LIMIT, not mcg, hidden, interm, experts)
    cb_effect = float((other - closed).abs().max() / closed.abs().max().clamp_min(1e-6))
    assert cb_effect > 0.05, f"codebook flag does nothing: {cb_effect:.3e}"


@pytest.mark.parametrize("act_limit", [0.0, ACT_LIMIT])
def test_mpw_prefill_glm_expert_count(act_limit):
    """n_routed_experts = 288: the metadata path (per-expert offsets, tile list, tile_slots
    sizing) must take GLM's expert count. Every expert's trellis table entry aliases one expert's
    buffers -- the kernel only touches the experts its tiles reference, so this exercises the
    count and the index arithmetic, not 288 weight sets (2.7 GB of trellis)."""
    _require("exl3_moe_prefill_wmma", "reconstruct", "hgemm", "had_r_128", "silu_mul", "pack_trellis")
    if not _MPW_READY:
        pytest.skip("this build's exl3_moe_prefill_wmma has no act_limit/mcg support")
    hidden, interm, experts, topk = GLM_H, GLM_I, GLM_EXPERTS, GLM_TOPK
    K, mcg, rows = 3, True, ROWS
    proj = {
        "gate": _mpw_projection(hidden, interm, K, 611, mcg, 1),
        "up":   _mpw_projection(hidden, interm, K, 612, mcg, 1),
        "down": _mpw_projection(interm, hidden, K, 613, mcg, 1),
    }
    g = torch.Generator(device=DEV).manual_seed(614)
    scale = 1.0 if act_limit else 0.35
    x = (torch.randn(rows, hidden, generator=g, device=DEV) * scale).half()
    sel = torch.stack([torch.randperm(experts, generator=g, device=DEV)[:topk]
                       for _ in range(rows)]).to(torch.long)
    w = torch.rand(rows, topk, generator=g, device=DEV)
    w = w / w.sum(dim=-1, keepdim=True)

    ref = _mpw_ref(x, proj, K, sel, w, act_limit, mcg, alias=True)
    got = _mpw_run(x, proj, K, sel, w, act_limit, mcg, hidden, interm, experts, alias=True)
    err = _rel(got, ref)
    assert err < 5e-3, f"E=288 K={K} mcg={mcg} L={act_limit}: rel err {err:.2e}"


# ------------------------------------------------------------------------------------------------
# (b) MLA batch-1 decode: the exl3_dec GEMV wiring in mla_attn.py

class _Inner:
    """Stand-in for a LinearEXL3: only the attributes the wiring reads."""

    def __init__(self, k, n, K, seed, mcg):
        self.trellis, self.suh, self.svh = _trellis(k, n, K, seed, mcg)
        self.in_features, self.out_features = k, n
        self.K = float(K)
        self.mcg, self.mul1 = mcg, not mcg
        self.bias = None
        self.default_out_dtype = torch.half


class _Lin:
    """Stand-in for the wrapper Linear (Module.forward's contract, minus the wrappers)."""

    def __init__(self, k, n, K, seed, mcg, out_dtype=None, bias=False, pad_out=False):
        self.inner = _Inner(k, n, K, seed, mcg)
        # Linear passes its out_dtype to LinearEXL3 (linear.py: LinearEXL3(..., self.out_dtype)),
        # which stores it as default_out_dtype -- the dtype the fused launch allocates.
        self.inner.default_out_dtype = out_dtype or torch.half
        self.quant_type = "exl3"
        self.in_features, self.out_features = k, n
        self.out_features_unpadded = n - 128 if pad_out else n
        self.out_dtype = out_dtype
        self.pre_scale = self.post_scale = 1.0
        self.softcap = 0.0
        self.lora_a_tensors = {}
        self.bias = bias
        self.calls = 0

    def forward(self, x, params, out_dtype=None):
        """The generic path: reconstruct + hadamard + hgemm, then the wrapper's padded-out trim."""
        self.calls += 1
        y = _ref_linear(x, self.inner.trellis, self.inner.suh, self.inner.svh,
                        int(self.inner.K), self.inner.mcg)
        if self.out_features != self.out_features_unpadded:
            y = y[..., :self.out_features_unpadded].contiguous()
        return y


def _mla():
    from exllamav3.modules.mla_attn import MLAttention, dec_proj_multi, dec_proj_ok, dec_proj_shapes
    attn = types.SimpleNamespace(dec_gemv=True)
    attn._dec_proj = types.MethodType(MLAttention._dec_proj, attn)
    attn._dec_proj_pair = types.MethodType(MLAttention._dec_proj_pair, attn)
    return attn, dec_proj_ok, dec_proj_multi, dec_proj_shapes


# GLM-5.3-Flash MLA projection widths, straight from the published config: q_lora_rank 1536,
# kv_lora_rank 512 (qk_rope_head_dim 0, so kv_a is the latent alone), num_attention_heads 64,
# qk_nope_head_dim 256, v_head_dim 256, index_n_heads 32, index_head_dim 128, hidden_size 4096.
# Every in_features % 512 == 0 and out_features % 128 == 0, which is what the exl3_dec GEMV
# requires -- the indexer's weights_proj (4096 -> 32) is the one that does not, and it is
# unquantized in the config (pad_to = 1) anyway.
HIDDEN = 4096
Q_LORA = 1536
KV_A = 512
Q_B = 64 * 256          # heads * qk_nope_head_dim
O_IN = 64 * 256         # heads * v_head_dim
IDX_WQ_B = 32 * 128     # index_n_heads * index_head_dim
WIDTHS = [("q_a", HIDDEN, Q_LORA), ("kv_a", HIDDEN, KV_A), ("q_b", Q_LORA, Q_B),
          ("o", O_IN, HIDDEN), ("wq_b", Q_LORA, IDX_WQ_B)]


@pytest.mark.parametrize("name,k,n", WIDTHS)
@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("mcg", [False, True])
def test_mla_dec_proj_matches_reference(name, k, n, K, mcg):
    """The wired projection vs reconstruct+hgemm, both codebooks, deterministic."""
    _require("exl3_dec_gemv", "reconstruct", "hgemm", "had_r_128", "pack_trellis")
    if not _GEMV_READY:
        pytest.skip("this build's exl3_dec_gemv has no mcg codebook support")
    from exllamav3.modules.mla_attn import dec_proj_ok, dec_proj_single

    attn, _, _, _ = _mla()
    lin = _Lin(k, n, K, 900 + K + (5 if mcg else 0), mcg)
    assert dec_proj_ok(lin), f"{name}: wiring refuses an eligible projection"

    g = torch.Generator(device=DEV).manual_seed(31)
    x = (torch.randn(1, k, generator=g, device=DEV) * 0.5).half()

    got = attn._dec_proj(lin, x, {})
    assert lin.calls == 0, f"{name}: fell back to Linear.forward"
    ref = _ref_linear(x, lin.inner.trellis, lin.inner.suh, lin.inner.svh, K, mcg)
    err = _rel(got, ref)
    assert err < 5e-3, f"{name} K={K} mcg={mcg}: rel err {err:.2e}"
    out2 = attn._dec_proj(lin, x, {})
    assert torch.equal(got, out2)


@pytest.mark.parametrize("mcg", [False, True])
def test_mla_dec_proj_pair_matches_singles(mcg, monkeypatch):
    """q_a + kv_a in one exl3_dec_gemv_multi launch == the two separate launches.

    Bit for bit only at a fixed k-split: dec_gemv_impl picks ktw (hence kbs, the number of blocks
    summing partials along k) from the launch's total strip count, so the pair (4 strips) and q_a
    alone (3 strips) split k differently by design and round the fp32 partial sums differently.
    With EXL3_DEC_KTW forcing one ktw the two must be identical; at the automatic pick they must
    agree to fp16 level."""
    _require("exl3_dec_gemv", "exl3_dec_gemv_multi", "reconstruct", "hgemm", "pack_trellis")
    if not _GEMV_READY:
        pytest.skip("this build's exl3_dec_gemv has no mcg codebook support")
    from exllamav3.modules.mla_attn import dec_proj_single

    attn, _, _, _ = _mla()
    q_a = _Lin(HIDDEN, Q_LORA, 3, 41, mcg)
    kv_a = _Lin(HIDDEN, KV_A, 3, 42, mcg)
    attn.q_a_proj, attn.kv_a_proj_with_mqa = q_a, kv_a
    g = torch.Generator(device=DEV).manual_seed(43)
    x = (torch.randn(1, HIDDEN, generator=g, device=DEV) * 0.5).half()

    fused = attn._dec_proj_pair(x, {})
    assert set(fused) == {id(q_a), id(kv_a)}, "pair launch did not fire"
    for lin in (q_a, kv_a):
        assert _rel(fused[id(lin)], dec_proj_single(lin, x)) < 1e-3
    assert all(torch.equal(fused[k], v) for k, v in attn._dec_proj_pair(x, {}).items())
    for ktw in ("4", "8"):
        monkeypatch.setenv("EXL3_DEC_KTW", ktw)
        forced = attn._dec_proj_pair(x, {})
        assert torch.equal(forced[id(q_a)], dec_proj_single(q_a, x)), f"ktw={ktw}"
        assert torch.equal(forced[id(kv_a)], dec_proj_single(kv_a, x)), f"ktw={ktw}"
    monkeypatch.delenv("EXL3_DEC_KTW")
    # And through the pair-aware single projection: served from the shared launch, no recompute
    y = attn._dec_proj(q_a, x, {}, fused)
    assert torch.equal(y, fused[id(q_a)]) and q_a.calls == 0


def test_mla_dec_proj_eligibility():
    """The wiring must refuse everything the fused launch cannot reproduce exactly."""
    _require("exl3_dec_gemv", "pack_trellis")
    from exllamav3.modules.mla_attn import dec_proj_ok

    assert dec_proj_ok(_Lin(HIDDEN, Q_LORA, 3, 7, False))
    assert dec_proj_ok(_Lin(HIDDEN, Q_LORA, 2, 7, True))
    # Non-exl3 storage (unquantized projection: GLM's published packs keep attention in source dtype)
    lin = _Lin(HIDDEN, Q_LORA, 3, 7, False)
    lin.quant_type = "bfloat16"
    assert not dec_proj_ok(lin)
    # Padded output columns: the kernel writes the padded width, the wrapper trims
    assert not dec_proj_ok(_Lin(HIDDEN, Q_LORA, 3, 7, False, pad_out=True))
    # Bias: the fused kernel has none
    lin = _Lin(HIDDEN, Q_LORA, 3, 7, False)
    lin.inner.bias = torch.zeros(Q_LORA, dtype=torch.half, device=DEV)
    assert not dec_proj_ok(lin)
    # Wrapper scales / softcap
    lin = _Lin(HIDDEN, Q_LORA, 3, 7, False)
    lin.post_scale = 0.5
    assert not dec_proj_ok(lin)
    # in_features not a multiple of 512 (the indexer's wk / weight projections are this case)
    assert not dec_proj_ok(_Lin(384, 32, 3, 7, False))
    # Output dtype the kernel cannot write into
    assert not dec_proj_ok(_Lin(HIDDEN, Q_LORA, 3, 7, False, out_dtype=torch.bfloat16))


class _CpuLin:
    """A stub Linear on CPU tensors: enough for the eligibility and selection logic, which never
    touches the kernel. Used so the wiring's branching is testable without the GPU lock."""

    def __init__(self, key, k, n, K=3, mcg=False, quant_type="exl3", pad_out=False,
                 out_dtype=None, bias=None, post_scale=1.0, in_features=None):
        self.key = key
        self.inner = types.SimpleNamespace(
            trellis=torch.empty((k // 16, n // 16, 16 * int(K)), dtype=torch.int16),
            suh=object(), svh=object(), in_features=k, out_features=n, K=float(K),
            mcg=mcg, mul1=not mcg, bias=bias, default_out_dtype=out_dtype or torch.half)
        self.quant_type = quant_type
        self.in_features = k if in_features is None else in_features
        self.out_features = n
        self.out_features_unpadded = n - 128 if pad_out else n
        self.out_dtype = out_dtype
        self.pre_scale = 1.0
        self.post_scale = post_scale
        self.softcap = 0.0
        self.lora_a_tensors = {}
        self.calls = 0

    def forward(self, x, params, out_dtype=None):
        self.calls += 1
        return ("generic", self.key)


def test_mla_dec_proj_selection(monkeypatch):
    """Which projection takes the fused route, which falls back, and which share one launch.

    Pure branching logic on CPU stubs -- the numeric kernel tests are the ones that need the GPU.
    """
    if not torch.version.hip:
        pytest.skip("ROCm build only")
    from exllamav3.modules import mla_attn

    assert hasattr(ext, "exl3_dec_gemv") and hasattr(ext, "exl3_dec_gemv_multi")

    seen = {"single": [], "multi": []}
    monkeypatch.setattr(mla_attn, "dec_proj_single",
                        lambda lin, x: seen["single"].append(lin.key) or ("fused", lin.key))

    def fake_multi(lins, x):
        seen["multi"].append([l.key for l in lins])
        return [("fused", l.key) for l in lins]
    monkeypatch.setattr(mla_attn, "dec_proj_multi", fake_multi)

    attn = types.SimpleNamespace(dec_gemv=True)
    attn._dec_proj = types.MethodType(mla_attn.MLAttention._dec_proj, attn)
    attn._dec_proj_pair = types.MethodType(mla_attn.MLAttention._dec_proj_pair, attn)
    x = torch.zeros((1, HIDDEN), dtype=torch.half)

    # Eligible, one row: the fused launch, Linear.forward untouched
    q_b = _CpuLin("q_b", Q_LORA, Q_B)
    assert attn._dec_proj(q_b, torch.zeros((1, Q_LORA), dtype=torch.half), {}) == ("fused", "q_b")
    assert q_b.calls == 0 and seen["single"] == ["q_b"]

    # Ineligible (unquantized indexer projection): the generic path
    wk = _CpuLin("wk", HIDDEN, 128, quant_type="bfloat16")
    assert attn._dec_proj(wk, x, {}) == ("generic", "wk")
    assert wk.calls == 1 and seen["single"] == ["q_b"]

    # More than one row: not the batch-1 route
    o = _CpuLin("o", O_IN, HIDDEN)
    attn._dec_proj(o, torch.zeros((4, O_IN), dtype=torch.half), {})
    assert o.calls == 1

    # `capture`/`ovr` in params: the fused route stands down
    q_a = _CpuLin("q_a", HIDDEN, Q_LORA)
    attn._dec_proj(q_a, x, {"capture": {}})
    assert q_a.calls == 1

    # The pair: q_a and kv_a in ONE multi launch when they share width, K and codebook
    attn.q_a_proj = _CpuLin("q_a2", HIDDEN, Q_LORA)
    attn.kv_a_proj_with_mqa = _CpuLin("kv_a", HIDDEN, KV_A)
    fused = attn._dec_proj_pair(x, {})
    assert seen["multi"] == [["q_a2", "kv_a"]]
    assert set(fused) == {id(attn.q_a_proj), id(attn.kv_a_proj_with_mqa)}
    # ... and served from it, not recomputed
    assert attn._dec_proj(attn.q_a_proj, x, {}, fused) == ("fused", "q_a2")
    assert attn.q_a_proj.calls == 0

    # A K mismatch between the two: no shared launch, two singles
    attn.kv_a_proj_with_mqa = _CpuLin("kv_a_k2", HIDDEN, KV_A, K=2)
    assert attn._dec_proj_pair(x, {}) == {}
    assert seen["multi"] == [["q_a2", "kv_a"]]

    # A codebook mismatch between the two: same refusal
    attn.kv_a_proj_with_mqa = _CpuLin("kv_a_mcg", HIDDEN, KV_A, mcg=True)
    assert attn._dec_proj_pair(x, {}) == {}

    # An ineligible pair member: the whole pair stays on the generic path
    attn.kv_a_proj_with_mqa = _CpuLin("kv_a_pad", HIDDEN, KV_A, pad_out=True)
    assert attn._dec_proj_pair(x, {}) == {}

    # No q_a_proj (a direct q projection model): nothing to pair
    attn.q_a_proj = None
    assert attn._dec_proj_pair(x, {}) == {}

    # EXL3_MLA_DEC=0 kills the whole wiring
    monkeypatch.setattr(mla_attn, "EXL3_MLA_DEC", False)
    assert not mla_attn.dec_proj_ok(_CpuLin("off", HIDDEN, Q_LORA))


def test_mla_dec_proj_falls_back():
    """An ineligible projection goes through Linear.forward, unchanged."""
    _require("exl3_dec_gemv", "reconstruct", "hgemm", "had_r_128", "pack_trellis")
    attn, _, _, _ = _mla()
    lin = _Lin(HIDDEN, Q_LORA, 3, 11, False, pad_out=True)
    g = torch.Generator(device=DEV).manual_seed(12)
    x = (torch.randn(1, HIDDEN, generator=g, device=DEV) * 0.5).half()
    y = attn._dec_proj(lin, x, {})
    assert lin.calls == 1 and y.shape == (1, lin.out_features_unpadded)
